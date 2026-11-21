//! Completion preparation stage: resolve prompt(s), tokenize, create stop decoder.
//!
//! This is the `/v1/completions` Stage 1 equivalent. It intentionally builds on top of
//! the native completion pipeline typing introduced in PR #840. It keeps
//! `CompletionRequest` native in the request context instead of laundering it
//! through `GenerateRequest`.

use async_trait::async_trait;
use axum::response::Response;
use futures::future::try_join_all;
use openai_protocol::common::StringOrArray;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{CompletionItem, PreparationOutput, RequestContext},
        utils,
    },
};

pub(crate) struct CompletionPreparationStage;

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.completion_request_arc();

        let tokenizer =
            utils::resolve_tokenizer(ctx, "CompletionPreparationStage::execute").map_err(|e| *e)?;

        let prompts: Vec<String> = match &request.prompt {
            StringOrArray::String(text) => vec![text.clone()],
            // Empty arrays are rejected at the boundary (`validate_completion_prompt`).
            StringOrArray::Array(texts) => texts.clone(),
        };

        let encodings = try_join_all(prompts.iter().map(|prompt| {
            utils::encode_blocking(
                tokenizer.clone(),
                prompt.clone(),
                request.add_special_tokens,
            )
        }))
        .await
        .map_err(|e| {
            error!(
                function = "CompletionPreparationStage::execute",
                error = %e,
                "Tokenization failed"
            );
            error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
        })?;

        let items: Vec<CompletionItem> = prompts
            .into_iter()
            .zip(encodings)
            .map(|(text, encoding)| CompletionItem {
                text,
                token_ids: encoding.token_ids().to_vec(),
            })
            .collect();

        let joined_routing_text = (items.len() > 1).then(|| {
            items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        });

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        ctx.state.preparation = Some(PreparationOutput::Completion {
            items,
            joined_routing_text,
        });

        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::{traits::Tokenizer, HuggingFaceTokenizer, TokenizerRegistry};
    use openai_protocol::completion::CompletionRequest;
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::CompletionPreparationStage;
    use crate::{
        routers::grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext, SharedComponents},
            regular::symphony::template::Templates,
            utils::ParserResolver,
        },
        worker::WorkerRegistry,
    };

    /// A `tokenizer.json` whose post-processor adds `<bos>` (id 0) in front of
    /// a single sequence, like Gemma 3's, over a word-level vocabulary.
    const BOS_TOKENIZER_JSON: &str = r#"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{"id": 0, "content": "<bos>", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [0], "tokens": ["<bos>"]}}},
        "decoder": null,
        "model": {"type": "WordLevel",
            "vocab": {"<bos>": 0, "<unk>": 1, "hi": 2, "there": 3}, "unk_token": "<unk>"}
    }"#;

    /// The token ids the stage prepares for `request`, tokenized by a real
    /// `HuggingFaceTokenizer` loaded from [`BOS_TOKENIZER_JSON`].
    async fn prepared_ids(request: serde_json::Value) -> Vec<Vec<u32>> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, BOS_TOKENIZER_JSON).unwrap();
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        tokenizer_registry
            .load("tokenizer-id", "m", "test", || async move { Ok(tokenizer) })
            .await
            .unwrap();
        let components = Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        });
        let request: CompletionRequest = serde_json::from_value(request).unwrap();
        let mut ctx =
            RequestContext::for_completion(Arc::new(request), None, "m".into(), components);
        if CompletionPreparationStage.execute(&mut ctx).await.is_err() {
            panic!("preparation failed");
        }
        let Some(PreparationOutput::Completion { items, .. }) = ctx.state.preparation else {
            panic!("no completion preparation");
        };
        items.into_iter().map(|item| item.token_ids).collect()
    }

    /// A completion prompt gets the tokenizer's special tokens unless the
    /// request turns them off, the same as vLLM's `/v1/completions`; a literal
    /// `<bos>` in the text is not deduplicated against the added one.
    #[tokio::test]
    async fn completion_prompt_gets_special_tokens_unless_the_request_declines() {
        let ids = |prompt: &str, add: Option<bool>| {
            let mut request = serde_json::json!({"model": "m", "prompt": prompt});
            if let Some(add) = add {
                request["add_special_tokens"] = add.into();
            }
            prepared_ids(request)
        };
        assert_eq!(ids("hi there", None).await, [[0, 2, 3]]);
        assert_eq!(ids("hi there", Some(true)).await, [[0, 2, 3]]);
        assert_eq!(ids("hi there", Some(false)).await, [[2, 3]]);
        assert_eq!(ids("<bos> hi there", None).await, [[0, 0, 2, 3]]);
        assert_eq!(ids("<bos> hi there", Some(false)).await, [[0, 2, 3]]);
    }
}
