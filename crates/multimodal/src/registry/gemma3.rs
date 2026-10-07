use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
};

pub(super) struct Gemma3VisionSpec;

impl Gemma3VisionSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_index"])
            .or_else(|| metadata.config_u32(&["image_token_id"]))
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_index".to_string(),
            })
    }

    fn boi_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["boi_token_index"])
            .or_else(|| metadata.config_u32(&["boi_token_id"]))
            .map(|v| v as TokenId)
    }

    fn eoi_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["eoi_token_index"])
            .or_else(|| metadata.config_u32(&["eoi_token_id"]))
            .map(|v| v as TokenId)
    }

    fn soft_sequence(
        soft_id: TokenId,
        boi: Option<TokenId>,
        eoi: Option<TokenId>,
        count: usize,
    ) -> (Vec<TokenId>, usize, usize) {
        let mut tokens: Vec<TokenId> = Vec::with_capacity(count + 2);
        if let Some(b) = boi {
            tokens.push(b);
        }
        let soft_offset = tokens.len();
        tokens.extend(std::iter::repeat_n(soft_id, count));
        if let Some(e) = eoi {
            tokens.push(e);
        }
        (tokens, soft_offset, count)
    }

    fn encode_pas_narrative(
        metadata: &ModelMetadata,
        soft_id: TokenId,
        boi: Option<TokenId>,
        eoi: Option<TokenId>,
        soft_per_view: usize,
        num_crops: usize,
    ) -> RegistryResult<(Vec<TokenId>, Vec<PlaceholderRange>)> {
        // HF: "Here is the original image {full} and here are some crops to help you see better {full}..."
        // full_image_sequence wraps soft tokens with BOI/EOI (and optional newlines).
        let prefix = metadata
            .tokenizer
            .encode_text("Here is the original image ")
            .unwrap_or_default();
        let mid = metadata
            .tokenizer
            .encode_text(" and here are some crops to help you see better ")
            .unwrap_or_default();
        let mut tokens: Vec<TokenId> = prefix.into_iter().map(|t| t as TokenId).collect();
        let mut ranges = Vec::with_capacity(1 + num_crops);

        let (orig, off, len) = Self::soft_sequence(soft_id, boi, eoi, soft_per_view);
        ranges.push(PlaceholderRange {
            offset: tokens.len() + off,
            length: len,
        });
        tokens.extend(orig);

        tokens.extend(mid.into_iter().map(|t| t as TokenId));

        for i in 0..num_crops {
            if i > 0 {
                // Space between crop sequences (HF joins with " ").
                if let Some(space_ids) = metadata.tokenizer.encode_text(" ") {
                    tokens.extend(space_ids.into_iter().map(|t| t as TokenId));
                }
            }
            let (crop, off, len) = Self::soft_sequence(soft_id, boi, eoi, soft_per_view);
            ranges.push(PlaceholderRange {
                offset: tokens.len() + off,
                length: len,
            });
            tokens.extend(crop);
        }
        Ok((tokens, ranges))
    }

    fn num_crops_list(preprocessed: &PreprocessedEncoderInputs) -> Vec<usize> {
        match preprocessed.model_specific.get("pan_and_scan_num_crops") {
            Some(ModelSpecificValue::IntVec(v)) => v
                .iter()
                .map(|&c| usize::try_from(c).unwrap_or(0))
                .collect(),
            Some(ModelSpecificValue::IntTensor { data, .. }) => data
                .iter()
                .map(|&c| usize::try_from(c).unwrap_or(0))
                .collect(),
            _ => vec![0; preprocessed.feature_token_counts.len()],
        }
    }
}

impl ModelProcessorSpec for Gemma3VisionSpec {
    fn name(&self) -> &'static str {
        "gemma3"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        let mt = metadata.config_model_type().unwrap_or("");
        // Prefer gemma3 over gemma4 / gemma3n.
        if mt == "gemma4" || id.contains("gemma-4") || id.contains("gemma4") {
            return false;
        }
        mt == "gemma3"
            || id.contains("gemma-3")
            || id.contains("gemma3")
            || metadata
                .config
                .get("architectures")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| {
                    arr.iter().any(|a| {
                        a.as_str()
                            .is_some_and(|s| s.contains("Gemma3ForConditionalGeneration"))
                    })
                })
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<start_of_image>".to_owned())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        // Chat templates emit BOI; soft tokens use image_token_index.
        Self::boi_token_id(metadata)
            .or_else(|| Self::image_token_id(metadata).ok())
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "boi_token_index".to_string(),
            })
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 16)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        // Opt-in pan-and-scan via preprocessor_config / mm kwargs; default off.
        Ok(json!({}))
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Flat so pan-and-scan can attach multiple patches per logical image.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            ("num_patches".to_string(), FieldLayout::Batched),
            (
                "pan_and_scan_num_crops".to_string(),
                FieldLayout::Batched,
            ),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "num_patches".to_string(),
            "pan_and_scan_num_crops".to_string(),
        ]
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let soft_id = Self::image_token_id(metadata)?;
        let boi = Self::boi_token_id(metadata);
        let eoi = Self::eoi_token_id(metadata);
        let placeholder = self.placeholder_token(metadata)?;
        let crops = Self::num_crops_list(preprocessed);
        if crops.len() != preprocessed.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "pan_and_scan_num_crops".to_string(),
            });
        }

        preprocessed
            .feature_token_counts
            .iter()
            .zip(crops)
            .map(|(&total_soft, num_crops)| {
                let views = 1 + num_crops;
                let soft_per_view = if views == 0 {
                    total_soft
                } else {
                    total_soft / views
                };
                if num_crops == 0 {
                    let (tokens, soft_offset, count) =
                        Self::soft_sequence(soft_id, boi, eoi, soft_per_view.max(1));
                    Ok(
                        PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
                            .with_feature_span(soft_offset, count),
                    )
                } else {
                    let (tokens, ranges) = Self::encode_pas_narrative(
                        metadata,
                        soft_id,
                        boi,
                        eoi,
                        soft_per_view.max(1),
                        num_crops,
                    )?;
                    Ok(
                        PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
                            .with_feature_ranges(ranges),
                    )
                }
            })
            .collect()
    }
}
