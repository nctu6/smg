use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, Gemma4AudioProcessor},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

pub(super) struct Gemma4VisionSpec;

impl Gemma4VisionSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .or_else(|| metadata.config_u32(&["image_token_index"]))
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn video_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["video_token_id"])
            .map(|v| v as TokenId)
            .or_else(|| Self::image_token_id(metadata).ok())
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "video_token_id".to_string(),
            })
    }

    fn audio_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["audio_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "audio_token_id".to_string(),
            })
    }

    fn boi_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["boi_token_id"])
            .or_else(|| metadata.config_u32(&["boi_token_index"]))
            .map(|v| v as TokenId)
    }

    fn eoi_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["eoi_token_id"])
            .or_else(|| metadata.config_u32(&["eoi_token_index"]))
            .map(|v| v as TokenId)
    }

    fn soft_tokens(metadata: &ModelMetadata) -> usize {
        metadata
            .config_u32(&["vision_soft_tokens_per_image"])
            .or_else(|| {
                metadata
                    .config
                    .get("vision_config")
                    .and_then(|v| v.get("default_output_length"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
            })
            .unwrap_or(280) as usize
    }

    fn has_audio_config(metadata: &ModelMetadata) -> bool {
        metadata
            .config
            .get("audio_config")
            .is_some_and(|v| !v.is_null())
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

    fn num_patches_list(preprocessed: &PreprocessedEncoderInputs) -> Vec<usize> {
        match preprocessed.model_specific.get("num_patches") {
            Some(ModelSpecificValue::IntVec(v)) => v
                .iter()
                .map(|&c| usize::try_from(c).unwrap_or(1).max(1))
                .collect(),
            Some(ModelSpecificValue::IntTensor { data, .. }) => data
                .iter()
                .map(|&c| usize::try_from(c).unwrap_or(1).max(1))
                .collect(),
            _ => vec![1; preprocessed.feature_token_counts.len()],
        }
    }

    fn encode_pas_narrative(
        metadata: &ModelMetadata,
        soft_id: TokenId,
        boi: Option<TokenId>,
        eoi: Option<TokenId>,
        soft_per_view: usize,
        num_crops: usize,
    ) -> RegistryResult<(Vec<TokenId>, Vec<PlaceholderRange>)> {
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

    fn format_timestamp(seconds: f32) -> String {
        let total = seconds.max(0.0) as u32;
        format!("{:02}:{:02}", total / 60, total % 60)
    }
}

impl ModelProcessorSpec for Gemma4VisionSpec {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        let mt = metadata.config_model_type().unwrap_or("");
        mt == "gemma4"
            || id.contains("gemma-4")
            || id.contains("gemma4")
            || metadata
                .config
                .get("architectures")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| {
                    arr.iter().any(|a| {
                        a.as_str()
                            .is_some_and(|s| s.contains("Gemma4ForConditionalGeneration"))
                    })
                })
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_owned())
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Ok("<|video|>".to_owned()),
            Modality::Audio => Ok("<|audio|>".to_owned()),
            other => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality: other,
            }),
        }
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::boi_token_id(metadata)
            .or_else(|| Self::image_token_id(metadata).ok())
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "boi_token_id".to_string(),
            })
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => self.placeholder_token_id(metadata),
            Modality::Video => Self::boi_token_id(metadata)
                .or_else(|| Self::video_token_id(metadata).ok())
                .ok_or_else(|| ModelRegistryError::MissingConfigField {
                    field: "boi_token_id".to_string(),
                }),
            Modality::Audio => Self::audio_token_id(metadata),
            other => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality: other,
            }),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::from([(Modality::Image, 16), (Modality::Video, 1)]);
        if Self::has_audio_config(metadata) {
            limits.insert(Modality::Audio, 8);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        if model_config.get("audio_config").is_some_and(|v| !v.is_null()) {
            Some(Box::new(Gemma4AudioProcessor::from_model_config(model_config)))
        } else {
            None
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            (
                "image_position_ids".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            ("num_patches".to_string(), FieldLayout::Batched),
            (
                "pan_and_scan_num_crops".to_string(),
                FieldLayout::Batched,
            ),
        ])
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> crate::types::EncoderFieldLayouts {
        use crate::types::EncoderFieldLayouts;
        match modality {
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("input_features_mask".to_string(), FieldLayout::Batched),
                    ("audio_feature_lengths".to_string(), FieldLayout::Batched),
                ]),
            ),
            _ => EncoderFieldLayouts::from_legacy_fields(self.field_layouts()),
        }
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
        let default_soft = Self::soft_tokens(metadata);
        let crops = Self::num_crops_list(preprocessed);
        let patches = Self::num_patches_list(preprocessed);
        if crops.len() != preprocessed.feature_token_counts.len()
            || patches.len() != preprocessed.feature_token_counts.len()
        {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "num_patches / pan_and_scan_num_crops".to_string(),
            });
        }

        preprocessed
            .feature_token_counts
            .iter()
            .zip(crops)
            .zip(patches)
            .map(|((&total_soft, num_crops), n_patches)| {
                let views = n_patches.max(1);
                let soft_per_view = if total_soft == 0 {
                    default_soft
                } else {
                    (total_soft / views).max(1)
                };
                if num_crops == 0 {
                    let (tokens, soft_offset, count) =
                        Self::soft_sequence(soft_id, boi, eoi, soft_per_view);
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
                        soft_per_view,
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

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            Modality::Video => {
                let soft_id = Self::video_token_id(metadata)?;
                let boi = Self::boi_token_id(metadata);
                let eoi = Self::eoi_token_id(metadata);
                let placeholder = self.placeholder_token_for(metadata, Modality::Video)?;
                let patches = Self::num_patches_list(preprocessed);
                let timestamps = match preprocessed.model_specific.get("video_timestamps") {
                    Some(ModelSpecificValue::FloatVec(v)) => v.clone(),
                    _ => vec![0.0; patches.first().copied().unwrap_or(1)],
                };
                if preprocessed.feature_token_counts.len() != 1 || patches.len() != 1 {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "gemma4 video expects one clip".to_string(),
                    });
                }
                let n_frames = patches[0];
                let total_soft = preprocessed.feature_token_counts[0];
                let soft_per = (total_soft / n_frames).max(1);
                let mut tokens = Vec::new();
                let mut ranges = Vec::with_capacity(n_frames);
                for frame_idx in 0..n_frames {
                    let ts = timestamps.get(frame_idx).copied().unwrap_or(0.0);
                    let label = Self::format_timestamp(ts);
                    if let Some(ids) = metadata.tokenizer.encode_text(&format!("{label} ")) {
                        tokens.extend(ids.into_iter().map(|t| t as TokenId));
                    }
                    let (seq, off, len) = Self::soft_sequence(soft_id, boi, eoi, soft_per);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len() + off,
                        length: len,
                    });
                    tokens.extend(seq);
                    if frame_idx + 1 < n_frames {
                        if let Some(space_ids) = metadata.tokenizer.encode_text(" ") {
                            tokens.extend(space_ids.into_iter().map(|t| t as TokenId));
                        }
                    }
                }
                Ok(vec![
                    PromptReplacement::sequence(Modality::Video, &placeholder, tokens)
                        .with_feature_ranges(ranges),
                ])
            }
            Modality::Audio => {
                if !Self::has_audio_config(metadata) {
                    return Err(ModelRegistryError::UnsupportedModality {
                        spec: self.name(),
                        modality,
                    });
                }
                let soft_id = Self::audio_token_id(metadata)?;
                let placeholder = self.placeholder_token_for(metadata, Modality::Audio)?;
                let boa = metadata.config_u32(&["boa_token_id"]).map(|v| v as TokenId);
                let eoa = metadata.config_u32(&["eoa_token_id"]).map(|v| v as TokenId);
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&count| {
                        let n = count.max(1);
                        let (tokens, off, len) = Self::soft_sequence(soft_id, boa, eoa, n);
                        PromptReplacement::sequence(Modality::Audio, &placeholder, tokens)
                            .with_feature_span(off, len)
                    })
                    .collect())
            }
            other => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality: other,
            }),
        }
    }
}
