//! Gemma 4 image / video preprocessor (Gemma4ImageProcessor parity).
//!
//! Aspect-ratio-preserving resize into a patch budget of
//! ``max_soft_tokens * pooling_kernel^2``, patchify to ``patch_size`` squares,
//! pad to ``max_patches``. Emits ``pixel_values`` as ``[N, max_patches, P]``
//! (P = patch_size^2 * 3) plus ``image_position_ids`` and per-image soft-token
//! counts.
//!
//! Optional pan-and-scan (gemma3-style crops, gated by ``do_pan_and_scan``)
//! expands one logical image into original + crops, each patchified at the
//! image soft-token budget. Video frames use ``max_soft_tokens=70`` and carry
//! per-frame timestamps for prompt expansion.

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array2, Array3};

use crate::{
    encoder_inputs::ModelSpecificValue,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        processors::gemma3::Gemma3Processor,
        transforms::{pil_to_filter, resize, TransformError},
    },
};

const DEFAULT_PATCH: u32 = 16;
const DEFAULT_SOFT: usize = 280;
const DEFAULT_POOL: u32 = 3;
const VIDEO_MAX_SOFT_TOKENS: usize = 70;
const VIDEO_MAX_FRAMES: usize = 32;
const SUPPORTED_SOFT: [usize; 5] = [70, 140, 280, 560, 1120];

#[derive(Debug, Clone)]
pub struct Gemma4Processor {
    pub patch_size: u32,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: u32,
}

impl Default for Gemma4Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4Processor {
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH,
            max_soft_tokens: DEFAULT_SOFT,
            pooling_kernel_size: DEFAULT_POOL,
        }
    }

    fn max_patches_for(&self, soft: usize) -> usize {
        soft * (self.pooling_kernel_size as usize).pow(2)
    }

    fn max_patches(&self) -> usize {
        self.max_patches_for(self.max_soft_tokens)
    }

    fn soft_budget(config: &PreProcessorConfig, default: usize) -> usize {
        let raw = config
            .extra
            .get("max_soft_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default);
        if SUPPORTED_SOFT.contains(&raw) {
            raw
        } else {
            default
        }
    }

    fn pas_enabled(config: &PreProcessorConfig) -> bool {
        config
            .extra
            .get("do_pan_and_scan")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn pas_params(config: &PreProcessorConfig) -> (u32, u32, f64) {
        let min_crop = config
            .extra
            .get("pan_and_scan_min_crop_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(256);
        let max_crops = config
            .extra
            .get("pan_and_scan_max_num_crops")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(4);
        let min_ratio = config
            .extra
            .get("pan_and_scan_min_ratio_to_activate")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.2);
        (min_crop, max_crops, min_ratio)
    }

    /// Resize so the patch grid fits in ``max_patches`` and is divisible by
    /// ``pooling_kernel_size`` (HF ``aspect_ratio_preserving_resize``).
    fn resize_for_budget(
        &self,
        image: &DynamicImage,
        filter: FilterType,
        soft_budget: usize,
    ) -> DynamicImage {
        let (w, h) = image.dimensions();
        if w == 0 || h == 0 {
            return image.clone();
        }
        let patch = self.patch_size as f64;
        let pool = self.pooling_kernel_size as f64;
        let max_patches = self.max_patches_for(soft_budget) as f64;
        let aspect = w as f64 / h as f64;
        let soft = soft_budget as f64;
        let mut soft_w = (soft * aspect).sqrt().floor().max(1.0);
        let soft_h = (soft / soft_w).floor().max(1.0);
        if soft_w * soft_h > soft {
            soft_w = (soft / soft_h).floor().max(1.0);
        }
        let target_w = ((soft_w * pool) * patch).round().max(patch) as u32;
        let target_h = ((soft_h * pool) * patch).round().max(patch) as u32;
        let step = self.patch_size * self.pooling_kernel_size;
        let tw = ((target_w + step - 1) / step) * step;
        let th = ((target_h + step - 1) / step) * step;
        let tw = tw
            .min((max_patches.sqrt() * patch).ceil() as u32 * 4)
            .max(step);
        let th = th
            .min((max_patches.sqrt() * patch).ceil() as u32 * 4)
            .max(step);
        resize(image, tw, th, filter)
    }

    fn patchify(&self, image: &DynamicImage) -> (Array2<f32>, Array2<i64>, usize) {
        let rgb = image.to_rgb8();
        let (w, h) = rgb.dimensions();
        let ps = self.patch_size;
        let ph = h / ps;
        let pw = w / ps;
        let patch_dim = (ps * ps * 3) as usize;
        let n = (ph * pw) as usize;
        let mut patches = Array2::<f32>::zeros((n, patch_dim));
        let mut positions = Array2::<i64>::zeros((n, 2));
        let mut idx = 0usize;
        for y in 0..ph {
            for x in 0..pw {
                let mut p = 0usize;
                for py in 0..ps {
                    for px in 0..ps {
                        let pixel = rgb.get_pixel(x * ps + px, y * ps + py);
                        patches[[idx, p]] = pixel[0] as f32 / 255.0;
                        patches[[idx, p + 1]] = pixel[1] as f32 / 255.0;
                        patches[[idx, p + 2]] = pixel[2] as f32 / 255.0;
                        p += 3;
                    }
                }
                positions[[idx, 0]] = x as i64;
                positions[[idx, 1]] = y as i64;
                idx += 1;
            }
        }
        let soft = n / (self.pooling_kernel_size as usize).pow(2);
        (patches, positions, soft)
    }

    fn patchify_into(
        &self,
        image: &DynamicImage,
        filter: FilterType,
        soft_budget: usize,
        do_resize: bool,
        batch: &mut Array3<f32>,
        pos_batch: &mut Array3<i64>,
        row: usize,
    ) -> usize {
        let resized = if do_resize {
            self.resize_for_budget(image, filter, soft_budget)
        } else {
            image.clone()
        };
        let (patches, positions, soft) = self.patchify(&resized);
        let max_patches = batch.shape()[1];
        let patch_dim = batch.shape()[2];
        let n = patches.nrows().min(max_patches);
        for r in 0..n {
            for c in 0..patch_dim {
                batch[[row, r, c]] = patches[[r, c]];
            }
            pos_batch[[row, r, 0]] = positions[[r, 0]];
            pos_batch[[row, r, 1]] = positions[[r, 1]];
        }
        soft.max(1).min(soft_budget)
    }

    fn build_batch(
        &self,
        views: &[DynamicImage],
        config: &PreProcessorConfig,
        soft_budget: usize,
        item_sizes: Vec<(u32, u32)>,
        num_patches: Vec<i64>,
        num_crops: Vec<i64>,
        timestamps: Option<Vec<f32>>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if views.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let filter = pil_to_filter(config.resampling);
        let max_patches = self.max_patches_for(soft_budget);
        let patch_dim = (self.patch_size * self.patch_size * 3) as usize;
        let mut batch = Array3::<f32>::zeros((views.len(), max_patches, patch_dim));
        let mut pos_batch = Array3::<i64>::zeros((views.len(), max_patches, 2));
        pos_batch.fill(-1);
        let mut feature_token_counts = Vec::with_capacity(num_patches.len());
        let do_resize = config.do_resize.unwrap_or(true);

        let mut view_idx = 0usize;
        for &patches in &num_patches {
            let p = patches.max(1) as usize;
            let mut soft_sum = 0usize;
            for _ in 0..p {
                let soft = self.patchify_into(
                    &views[view_idx],
                    filter,
                    soft_budget,
                    do_resize,
                    &mut batch,
                    &mut pos_batch,
                    view_idx,
                );
                soft_sum += soft;
                view_idx += 1;
            }
            feature_token_counts.push(soft_sum.max(1));
        }

        let mut out = PreprocessedEncoderInputs::new(batch, feature_token_counts, item_sizes);
        out.model_specific.insert(
            "image_position_ids".to_string(),
            ModelSpecificValue::IntTensor {
                data: pos_batch.iter().copied().collect(),
                shape: vec![views.len(), max_patches, 2],
            },
        );
        out.model_specific.insert(
            "num_patches".to_string(),
            ModelSpecificValue::IntVec(num_patches),
        );
        out.model_specific.insert(
            "pan_and_scan_num_crops".to_string(),
            ModelSpecificValue::IntVec(num_crops),
        );
        if let Some(ts) = timestamps {
            out.model_specific.insert(
                "video_timestamps".to_string(),
                ModelSpecificValue::FloatVec(ts),
            );
            out.model_specific.insert(
                "video_max_soft_tokens".to_string(),
                ModelSpecificValue::Int(VIDEO_MAX_SOFT_TOKENS as i64),
            );
        }
        Ok(out)
    }
}

impl VisionPreProcessor for Gemma4Processor {
    fn default_mean(&self) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    fn default_std(&self) -> [f64; 3] {
        [1.0, 1.0, 1.0]
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let soft_budget = Self::soft_budget(config, self.max_soft_tokens);
        let do_pas = Self::pas_enabled(config);
        let (min_crop, max_crops, min_ratio) = Self::pas_params(config);

        let item_sizes: Vec<(u32, u32)> = images.iter().map(|i| i.dimensions()).collect();
        let mut views: Vec<DynamicImage> = Vec::new();
        let mut num_patches = Vec::with_capacity(images.len());
        let mut num_crops = Vec::with_capacity(images.len());

        for image in images {
            let crops = if do_pas {
                Gemma3Processor::pan_and_scan_crops(image, min_crop, max_crops, min_ratio)
            } else {
                Vec::new()
            };
            let n_crops = crops.len();
            views.push(image.clone());
            views.extend(crops);
            num_patches.push((1 + n_crops) as i64);
            num_crops.push(n_crops as i64);
        }

        self.build_batch(
            &views,
            config,
            soft_budget,
            item_sizes,
            num_patches,
            num_crops,
            None,
        )
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let fps = config
            .extra
            .get("fps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            .max(1e-3) as f32;
        let sampled: Vec<&DynamicImage> = if frames.len() > VIDEO_MAX_FRAMES {
            // Uniform sample down to VIDEO_MAX_FRAMES.
            (0..VIDEO_MAX_FRAMES)
                .map(|i| {
                    let idx = i * (frames.len() - 1) / (VIDEO_MAX_FRAMES - 1);
                    &frames[idx]
                })
                .collect()
        } else {
            frames.iter().collect()
        };
        let n = sampled.len();
        let timestamps: Vec<f32> = (0..n).map(|i| i as f32 / fps).collect();
        let views: Vec<DynamicImage> = sampled.into_iter().cloned().collect();
        let item_sizes = vec![views[0].dimensions()];
        self.build_batch(
            &views,
            config,
            VIDEO_MAX_SOFT_TOKENS,
            item_sizes,
            vec![n as i64],
            vec![0],
            Some(timestamps),
        )
    }

    fn calculate_num_tokens(
        &self,
        _width: u32,
        _height: u32,
        config: &PreProcessorConfig,
    ) -> usize {
        Self::soft_budget(config, self.max_soft_tokens)
    }

    fn model_name(&self) -> &'static str {
        "gemma4"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn video_preprocess_emits_timestamps() {
        let proc = Gemma4Processor::new();
        let frames = vec![
            DynamicImage::new_rgb8(320, 240),
            DynamicImage::new_rgb8(320, 240),
        ];
        let mut cfg = PreProcessorConfig::default();
        cfg.extra.insert("fps".into(), json!(2.0));
        let out = proc.preprocess_video(&frames, &cfg).unwrap();
        assert_eq!(out.feature_token_counts.len(), 1);
        assert!(out.feature_token_counts[0] >= 2);
        match out.model_specific.get("num_patches") {
            Some(ModelSpecificValue::IntVec(v)) => assert_eq!(v, &vec![2]),
            other => panic!("bad num_patches {other:?}"),
        }
        match out.model_specific.get("video_timestamps") {
            Some(ModelSpecificValue::FloatVec(v)) => assert_eq!(v.len(), 2),
            other => panic!("bad timestamps {other:?}"),
        }
    }

    #[test]
    fn pas_expands_wide_image() {
        let proc = Gemma4Processor::new();
        let img = DynamicImage::new_rgb8(1800, 900);
        let mut cfg = PreProcessorConfig::default();
        cfg.extra.insert("do_pan_and_scan".into(), json!(true));
        let out = proc.preprocess(&[img], &cfg).unwrap();
        assert_eq!(out.feature_token_counts.len(), 1);
        match out.model_specific.get("num_patches") {
            Some(ModelSpecificValue::IntVec(v)) => assert_eq!(v[0], 3),
            other => panic!("bad num_patches {other:?}"),
        }
    }
}
