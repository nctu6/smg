//! Gemma 3 image preprocessor (SigLIP / Gemma3ImageProcessor parity).
//!
//! Pipeline: optional pan-and-scan crops for exaggerated aspect ratios, then
//! resize each view to square ``image_size`` (default 896), rescale to [0,1],
//! normalize with mean/std 0.5, emit ``mm_tokens_per_image`` (default 256)
//! soft tokens per view. Pan-and-scan is off unless ``do_pan_and_scan`` is set
//! (HF processor_kwargs default is false; checkpoint fields are often null).

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::{
    encoder_inputs::ModelSpecificValue,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{normalize, pil_to_filter, resize, stack_batch, to_tensor, TransformError},
    },
};

/// SigLIP / Gemma 3 default mean.
pub const GEMMA3_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
/// SigLIP / Gemma 3 default std.
pub const GEMMA3_STD: [f64; 3] = [0.5, 0.5, 0.5];

const DEFAULT_PAS_MIN_CROP: u32 = 256;
const DEFAULT_PAS_MAX_CROPS: u32 = 4;
const DEFAULT_PAS_MIN_RATIO: f64 = 1.2;

#[derive(Debug, Clone)]
pub struct Gemma3Processor {
    pub image_size: u32,
    pub mm_tokens_per_image: usize,
}

impl Default for Gemma3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma3Processor {
    pub fn new() -> Self {
        Self {
            image_size: 896,
            mm_tokens_per_image: 256,
        }
    }

    fn mean_std(config: &PreProcessorConfig) -> ([f64; 3], [f64; 3]) {
        let mean = match &config.image_mean {
            Some(v) if v.len() >= 3 => [v[0], v[1], v[2]],
            _ => GEMMA3_MEAN,
        };
        let std = match &config.image_std {
            Some(v) if v.len() >= 3 => [v[0], v[1], v[2]],
            _ => GEMMA3_STD,
        };
        (mean, std)
    }

    fn mm_tokens(config: &PreProcessorConfig, fallback: usize) -> usize {
        config
            .extra
            .get("image_seq_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(fallback)
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
            .unwrap_or(DEFAULT_PAS_MIN_CROP);
        let max_crops = config
            .extra
            .get("pan_and_scan_max_num_crops")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_PAS_MAX_CROPS);
        let min_ratio = config
            .extra
            .get("pan_and_scan_min_ratio_to_activate")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_PAS_MIN_RATIO);
        (min_crop, max_crops, min_ratio)
    }

    /// HF ``Gemma3ImageProcessor.pan_and_scan`` crop grid for one image.
    /// Returns crop images only (not the original thumbnail).
    pub fn pan_and_scan_crops(
        image: &DynamicImage,
        min_crop_size: u32,
        max_num_crops: u32,
        min_ratio_to_activate: f64,
    ) -> Vec<DynamicImage> {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 || min_crop_size == 0 || max_num_crops < 2 {
            return Vec::new();
        }

        let (num_crops_w, num_crops_h) = if width >= height {
            if (width as f64) / (height as f64) < min_ratio_to_activate {
                return Vec::new();
            }
            let mut num_crops_w = ((width as f64) / (height as f64) + 0.5).floor() as u32;
            num_crops_w = num_crops_w.min(width / min_crop_size);
            num_crops_w = num_crops_w.max(2).min(max_num_crops);
            (num_crops_w, 1u32)
        } else {
            if (height as f64) / (width as f64) < min_ratio_to_activate {
                return Vec::new();
            }
            let mut num_crops_h = ((height as f64) / (width as f64) + 0.5).floor() as u32;
            num_crops_h = num_crops_h.min(height / min_crop_size);
            num_crops_h = num_crops_h.max(2).min(max_num_crops);
            (1u32, num_crops_h)
        };

        let crop_w = (width + num_crops_w - 1) / num_crops_w;
        let crop_h = (height + num_crops_h - 1) / num_crops_h;
        if crop_w.min(crop_h) < min_crop_size {
            return Vec::new();
        }

        let mut crops = Vec::with_capacity((num_crops_w * num_crops_h) as usize);
        for y in 0..num_crops_h {
            for x in 0..num_crops_w {
                let left = x * crop_w;
                let top = y * crop_h;
                let w = crop_w.min(width.saturating_sub(left)).max(1);
                let h = crop_h.min(height.saturating_sub(top)).max(1);
                crops.push(image.crop_imm(left, top, w, h));
            }
        }
        crops
    }

    fn process_one(&self, image: &DynamicImage, config: &PreProcessorConfig) -> Array3<f32> {
        let size = config
            .get_target_size()
            .map(|(h, _)| h)
            .unwrap_or(self.image_size);
        let filter = pil_to_filter(config.resampling);
        let processed = if config.do_resize.unwrap_or(true) {
            resize(image, size, size, filter)
        } else {
            image.clone()
        };
        let mut tensor = to_tensor(&processed);
        if config.do_normalize.unwrap_or(true) {
            let (mean, std) = Self::mean_std(config);
            normalize(&mut tensor, &mean, &std);
        }
        tensor
    }
}

impl VisionPreProcessor for Gemma3Processor {
    fn default_mean(&self) -> [f64; 3] {
        GEMMA3_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        GEMMA3_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let mm_tokens = Self::mm_tokens(config, self.mm_tokens_per_image);
        let do_pas = Self::pas_enabled(config);
        let (min_crop, max_crops, min_ratio) = Self::pas_params(config);

        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
        let mut tensors: Vec<Array3<f32>> = Vec::new();
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut num_patches = Vec::with_capacity(images.len());
        let mut num_crops_vec = Vec::with_capacity(images.len());

        for image in images {
            let crops = if do_pas {
                Self::pan_and_scan_crops(image, min_crop, max_crops, min_ratio)
            } else {
                Vec::new()
            };
            let n_crops = crops.len();
            // Original thumbnail first, then crops (HF / vLLM order).
            tensors.push(self.process_one(image, config));
            for crop in &crops {
                tensors.push(self.process_one(crop, config));
            }
            let patches = 1 + n_crops;
            num_patches.push(patches as i64);
            num_crops_vec.push(n_crops as i64);
            feature_token_counts.push(patches * mm_tokens);
        }

        let encoder_input = stack_batch(&tensors)?;
        let mut out =
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes);
        out.model_specific.insert(
            "num_patches".to_string(),
            ModelSpecificValue::IntVec(num_patches),
        );
        out.model_specific.insert(
            "pan_and_scan_num_crops".to_string(),
            ModelSpecificValue::IntVec(num_crops_vec),
        );
        Ok(out)
    }

    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<usize, TransformError> {
        let mm_tokens = Self::mm_tokens(config, self.mm_tokens_per_image);
        if !Self::pas_enabled(config) {
            return Ok(mm_tokens);
        }
        let (min_crop, max_crops, min_ratio) = Self::pas_params(config);
        // Synthetic image for crop-count only.
        let dummy = DynamicImage::new_rgb8(width.max(1), height.max(1));
        let n_crops = Self::pan_and_scan_crops(&dummy, min_crop, max_crops, min_ratio).len();
        Ok((1 + n_crops) * mm_tokens)
    }

    fn model_name(&self) -> &'static str {
        "gemma3"
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        let size = config
            .get_target_size()
            .map(|(h, _)| h)
            .unwrap_or(self.image_size);
        Some((size, size))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn pas_config() -> PreProcessorConfig {
        let mut cfg = PreProcessorConfig::default();
        cfg.do_resize = Some(true);
        cfg.do_normalize = Some(true);
        cfg.extra.insert("do_pan_and_scan".into(), json!(true));
        cfg.extra
            .insert("pan_and_scan_min_crop_size".into(), json!(256));
        cfg.extra
            .insert("pan_and_scan_max_num_crops".into(), json!(4));
        cfg.extra
            .insert("pan_and_scan_min_ratio_to_activate".into(), json!(1.2));
        cfg
    }

    #[test]
    fn square_image_skips_pan_and_scan() {
        let img = DynamicImage::new_rgb8(896, 896);
        let crops = Gemma3Processor::pan_and_scan_crops(&img, 256, 4, 1.2);
        assert!(crops.is_empty());
    }

    #[test]
    fn wide_image_yields_two_crops() {
        let img = DynamicImage::new_rgb8(1800, 900);
        let crops = Gemma3Processor::pan_and_scan_crops(&img, 256, 4, 1.2);
        assert_eq!(crops.len(), 2);
    }

    #[test]
    fn preprocess_flat_expands_crops() {
        let proc = Gemma3Processor::new();
        let img = DynamicImage::new_rgb8(1800, 900);
        let out = proc.preprocess(&[img], &pas_config()).unwrap();
        assert_eq!(out.feature_token_counts, vec![3 * 256]);
        assert_eq!(out.encoder_input.shape(), &[3, 3, 896, 896]);
        match out.model_specific.get("num_patches") {
            Some(ModelSpecificValue::IntVec(v)) => assert_eq!(v, &vec![3]),
            other => panic!("unexpected num_patches: {other:?}"),
        }
        match out.model_specific.get("pan_and_scan_num_crops") {
            Some(ModelSpecificValue::IntVec(v)) => assert_eq!(v, &vec![2]),
            other => panic!("unexpected crops: {other:?}"),
        }
    }

    #[test]
    fn preprocess_default_no_pas() {
        let proc = Gemma3Processor::new();
        let img = DynamicImage::new_rgb8(1800, 900);
        let out = proc
            .preprocess(&[img], &PreProcessorConfig::default())
            .unwrap();
        assert_eq!(out.feature_token_counts, vec![256]);
        assert_eq!(out.encoder_input.shape(), &[1, 3, 896, 896]);
    }
}
