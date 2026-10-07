//! Request building stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{helpers, BuildStage},
        context::{
            AttemptStamp, BuildOutput, ExecutionPlan, PreparationOutput, RequestContext,
            RequestType,
        },
        proto_wrapper::ProtoEmbedRequest,
        spec::ResponseSpec,
    },
};

/// Request building stage for embedding requests
pub(crate) struct EmbeddingRequestBuildingStage;

impl EmbeddingRequestBuildingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingRequestBuildingStage {
    fn default() -> Self {
        Self::new()
    }
}

fn build_one_embed(
    client: &BackendClient,
    request_id: String,
    original_text: Option<String>,
    token_ids: Vec<u32>,
) -> Result<ProtoEmbedRequest, Response> {
    match client {
        BackendClient::Grpc(GrpcClient::Sglang(c)) => {
            let req = c.build_embed_request(request_id, original_text, token_ids);
            Ok(ProtoEmbedRequest::Sglang(Box::new(req)))
        }
        BackendClient::Grpc(GrpcClient::Vllm(c)) => {
            let req = c.build_embed_request(request_id, original_text, token_ids);
            Ok(ProtoEmbedRequest::Vllm(Box::new(req)))
        }
        BackendClient::Grpc(GrpcClient::Trtllm(_)) => {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "TensorRT-LLM embedding not yet supported"
            );
            Err(error::not_implemented(
                "unsupported_backend",
                "TensorRT-LLM embedding is not yet supported via gRPC",
            ))
        }
        BackendClient::Grpc(GrpcClient::Mlx(_)) => {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "MLX embedding not supported"
            );
            Err(error::not_implemented(
                "unsupported_backend",
                "MLX embedding is not supported via gRPC",
            ))
        }
        BackendClient::Grpc(GrpcClient::TokenSpeed(c)) => {
            let req = c.build_embed_request(request_id, original_text, token_ids);
            Ok(ProtoEmbedRequest::TokenSpeed(Box::new(req)))
        }
        BackendClient::Zmq(_) => Err(error::not_implemented(
            "unsupported_backend",
            "ZMQ backend does not support embeddings yet",
        )),
    }
}

#[async_trait]
impl BuildStage for EmbeddingRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        let prep_output = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "Preparation output missing"
            );
            error::internal_error("preparation_missing", "Preparation output missing")
        })?;

        let PreparationOutput::Embedding { items, .. } = prep_output else {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "Expected Embedding preparation output"
            );
            return Err(error::internal_error(
                "preparation_wrong_type",
                "Expected Embedding preparation output",
            ));
        };

        let client = ctx
            .state
            .clients
            .as_ref()
            .and_then(|c| c.single())
            .ok_or_else(|| {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    "Client not selected"
                );
                error::internal_error("client_missing", "Client not selected")
            })?;

        // Embeddings/classify are single-worker only (never disaggregated).
        let (prefix, spec) = match &ctx.input.request_type {
            RequestType::Classify(_) => ("classify-", ResponseSpec::Classify),
            RequestType::Embedding(_) => ("embed-", ResponseSpec::Embedding),
            request_type => {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    request_type = %request_type,
                    "{request_type} should not reach this stage"
                );
                return Err(error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ));
            }
        };

        if items.len() == 1 {
            let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
                &ctx.input.request_type,
                ctx.input.tenant_request_meta.as_ref(),
                prefix,
                false,
            );
            let item = &items[0];
            let proto_req = build_one_embed(
                client,
                request_id,
                Some(item.text.clone()),
                item.token_ids.clone(),
            )?;
            return Ok(BuildOutput {
                plan: ExecutionPlan::embed(proto_req),
                spec,
                stamp: AttemptStamp {
                    id: id_stamp,
                    sampling_mask: None,
                    sampling_baseline: None,
                    inject_pd_metadata: false,
                },
            });
        }

        // N inputs → N Embed RPCs (mirrors completion batch fan-out).
        let (shared_request_id, batch_stamp) = helpers::resolve_batch_id_stamp(
            &ctx.input.request_type,
            ctx.input.tenant_request_meta.as_ref(),
            prefix,
            false,
        );
        let mut requests = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let sub_id = helpers::batch_sub_id(&shared_request_id, i, batch_stamp.unique_subs);
            requests.push(build_one_embed(
                client,
                sub_id,
                Some(item.text.clone()),
                item.token_ids.clone(),
            )?);
        }

        Ok(BuildOutput {
            plan: ExecutionPlan::EmbedBatch {
                shared_request_id,
                requests,
            },
            spec,
            stamp: AttemptStamp {
                id: helpers::IdStamp::Batch(batch_stamp),
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
            },
        })
    }

    fn name(&self) -> &'static str {
        "EmbeddingRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        "EmbeddingRequestBuildingStage".to_string()
    }
}
