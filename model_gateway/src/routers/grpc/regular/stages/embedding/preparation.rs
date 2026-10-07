//! Preparation stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use futures::future::try_join_all;
use openai_protocol::common::GenerationRequest;
use serde_json::Value;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{EmbeddingItem, PreparationOutput, RequestContext, RequestType},
        utils,
    },
};

pub(crate) struct EmbeddingPreparationStage;

impl EmbeddingPreparationStage {
    pub fn new() -> Self {
        Self
    }
}

/// Extract OpenAI embedding `input` into one or more text strings.
///
/// Supports a string or an array of strings. Non-string array elements are
/// skipped (same filter as `extract_text_for_routing`). Token-id inputs are
/// not yet handled here.
fn extract_embedding_texts(input: &Value) -> Result<Vec<String>, Response> {
    match input {
        Value::String(s) => {
            if s.is_empty() {
                return Err(error::bad_request(
                    "empty_input",
                    "Input text cannot be empty",
                ));
            }
            Ok(vec![s.clone()])
        }
        Value::Array(arr) => {
            let texts: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            if texts.is_empty() {
                return Err(error::bad_request(
                    "empty_input",
                    "Input text cannot be empty",
                ));
            }
            if texts.iter().any(|t| t.is_empty()) {
                return Err(error::bad_request(
                    "empty_input",
                    "Input text cannot be empty",
                ));
            }
            Ok(texts)
        }
        _ => Err(error::bad_request(
            "unsupported_input",
            "Embedding input must be a string or an array of strings",
        )),
    }
}

#[async_trait]
impl PipelineStage for EmbeddingPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let texts = match &ctx.input.request_type {
            RequestType::Embedding(req) => extract_embedding_texts(&req.input)?,
            // Classify keeps the legacy single-string routing extraction.
            RequestType::Classify(req) => {
                let text = req.extract_text_for_routing();
                if text.is_empty() {
                    return Err(error::bad_request(
                        "empty_input",
                        "Input text cannot be empty",
                    ));
                }
                vec![text]
            }
            _ => {
                error!(
                    function = "EmbeddingPreparationStage::execute",
                    "Invalid request type: expected Embedding or Classify"
                );
                return Err(error::internal_error(
                    "invalid_request_type",
                    "Expected Embedding or Classify request",
                ));
            }
        };

        let tokenizer =
            utils::resolve_tokenizer(ctx, "EmbeddingPreparationStage::execute").map_err(|e| *e)?;

        // Tokenize with special tokens (BOS/EOS) for embeddings — matches
        // transformers' add_bos_token/add_eos_token behavior.
        let encodings = try_join_all(
            texts
                .iter()
                .map(|text| utils::encode_blocking(tokenizer.clone(), text.clone(), true)),
        )
        .await
        .map_err(|e| {
            error!(
                function = "EmbeddingPreparationStage::execute",
                error = %e,
                "Tokenization failed"
            );
            error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
        })?;

        let items: Vec<EmbeddingItem> = texts
            .into_iter()
            .zip(encodings)
            .map(|(text, encoding)| EmbeddingItem {
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

        ctx.state.preparation = Some(PreparationOutput::Embedding {
            items,
            joined_routing_text,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "EmbeddingPreparation"
    }
}
