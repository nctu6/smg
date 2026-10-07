//! Response processing stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::embedding::{EmbeddingObject, EmbeddingResponse};
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::{DispatchContext, ExecutionResult, FinalResponse},
        proto_wrapper::ProtoEmbedComplete,
        spec::ResponseSpec,
    },
};

/// Response processing stage for embedding requests
pub(crate) struct EmbeddingResponseProcessingStage;

impl EmbeddingResponseProcessingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ProcessStage for EmbeddingResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        if !matches!(spec, ResponseSpec::Embedding) {
            error!(
                function = "EmbeddingResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        }

        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::process",
                "Execution result missing"
            );
            error::internal_error("execution_result_missing", "Execution result missing")
        })?;

        let embedding_response = match execution_result {
            ExecutionResult::Embedding { response } => self
                .convert_responses(ctx, vec![response])
                .map_err(|boxed_err| *boxed_err)?,
            ExecutionResult::Batch { results } => {
                let mut protos = Vec::with_capacity(results.len());
                for (i, result) in results.into_iter().enumerate() {
                    match result {
                        ExecutionResult::Embedding { response } => protos.push(response),
                        other => {
                            error!(
                                function = "EmbeddingResponseProcessingStage::process",
                                index = i,
                                "Invalid batched execution result: expected Embedding"
                            );
                            let _ = other;
                            return Err(error::internal_error(
                                "invalid_execution_result",
                                "Expected Embedding result in batch",
                            ));
                        }
                    }
                }
                self.convert_responses(ctx, protos)
                    .map_err(|boxed_err| *boxed_err)?
            }
            _ => {
                error!(
                    function = "EmbeddingResponseProcessingStage::process",
                    "Invalid execution result: expected Embedding or Batch"
                );
                return Err(error::internal_error(
                    "invalid_execution_result",
                    "Expected Embedding result",
                ));
            }
        };

        ctx.response.final_response = Some(FinalResponse::Embedding(embedding_response));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingResponseProcessing"
    }
}

impl EmbeddingResponseProcessingStage {
    #[expect(
        clippy::unused_self,
        reason = "method on stage struct for consistency with other stage convert_response methods"
    )]
    fn convert_responses(
        &self,
        ctx: &DispatchContext,
        protos: Vec<ProtoEmbedComplete>,
    ) -> Result<EmbeddingResponse, Box<Response>> {
        let dispatch = ctx.dispatch.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::convert_responses",
                "Dispatch metadata missing in context"
            );
            error::internal_error("dispatch_missing", "Dispatch metadata missing")
        })?;

        let model = dispatch.model.clone();
        let mut prompt_tokens: u32 = 0;
        let mut data = Vec::with_capacity(protos.len());
        for (index, proto) in protos.into_iter().enumerate() {
            prompt_tokens = prompt_tokens.saturating_add(proto.prompt_tokens());
            data.push(EmbeddingObject {
                object: "embedding".to_string(),
                embedding: proto.embedding().to_vec(),
                index: index as u32,
            });
        }

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model,
            usage: openai_protocol::common::UsageInfo {
                prompt_tokens,
                total_tokens: prompt_tokens,
                completion_tokens: 0,
                prompt_tokens_details: None,
                reasoning_tokens: None,
            },
        })
    }
}
