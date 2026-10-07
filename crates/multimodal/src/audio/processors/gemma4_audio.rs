//! Gemma 4 audio preprocessing (log-mel frontend).
//!
//! Produces padded mel spectrograms and masks for the HF ``Gemma4Audio`` tower.
//! Soft-token counts replicate the tower's framing + two Conv2d stride-2
//! subsampling layers, capped at ``audio_seq_length`` (default 750).

use std::sync::Arc;

use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde_json::Value;

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

const DEFAULT_SR: usize = 16_000;
const DEFAULT_MELS: usize = 128;
const DEFAULT_FRAME_MS: f64 = 20.0;
const DEFAULT_HOP_MS: f64 = 10.0;
const DEFAULT_AUDIO_SEQ: usize = 750;

#[derive(Debug, Clone)]
pub struct Gemma4AudioParams {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub audio_seq_length: usize,
}

impl Default for Gemma4AudioParams {
    fn default() -> Self {
        let sample_rate = DEFAULT_SR;
        Self {
            sample_rate,
            n_mels: DEFAULT_MELS,
            frame_length: ((sample_rate as f64) * DEFAULT_FRAME_MS / 1000.0).round() as usize,
            hop_length: ((sample_rate as f64) * DEFAULT_HOP_MS / 1000.0).round() as usize,
            audio_seq_length: DEFAULT_AUDIO_SEQ,
        }
    }
}

impl Gemma4AudioParams {
    pub fn from_model_config(config: &Value) -> Self {
        let mut params = Self::default();
        if let Some(v) = config
            .get("audio_config")
            .and_then(|a| a.get("output_proj_dims"))
        {
            let _ = v; // dims used by tower, not frontend
        }
        if let Some(v) = config.get("audio_seq_length").and_then(|v| v.as_u64()) {
            params.audio_seq_length = v as usize;
        }
        params
    }

    /// Soft-token count after mel framing + two stride-2 convs.
    pub fn soft_tokens_for_samples(&self, num_samples: usize) -> usize {
        let frame_size_for_unfold = self.frame_length + 1;
        let pad_left = self.frame_length / 2;
        let padded = num_samples + pad_left;
        if padded < frame_size_for_unfold {
            return 0;
        }
        let mut t = (padded - frame_size_for_unfold) / self.hop_length + 1;
        for _ in 0..2 {
            // Conv2d kernel=3, stride=2, semicausal pad top=1 bottom=1 → (t+2-3)//2+1
            t = (t + 2 - 3) / 2 + 1;
        }
        t.min(self.audio_seq_length)
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4AudioProcessor {
    params: Gemma4AudioParams,
}

impl Gemma4AudioProcessor {
    pub fn new(params: Gemma4AudioParams) -> Self {
        Self { params }
    }

    pub fn from_model_config(config: &Value) -> Self {
        Self::new(Gemma4AudioParams::from_model_config(config))
    }

    fn ensure_mono_sr(&self, decoded: &DecodedAudio) -> Result<Vec<f32>, TransformError> {
        let samples = if decoded.sample_rate != self.params.sample_rate {
            bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                self.params.sample_rate,
            )?
        } else {
            decoded.samples.clone()
        };
        Ok(samples)
    }

    fn log_mel(&self, samples: &[f32]) -> Result<Array2<f32>, TransformError> {
        let n_fft = self.params.frame_length.next_power_of_two();
        let fft_bins = n_fft / 2 + 1;
        let window = hann_window(n_fft);
        let mel_filters = mel_basis(self.params.sample_rate, n_fft, self.params.n_mels);

        let frame_size_for_unfold = self.params.frame_length + 1;
        let pad_left = self.params.frame_length / 2;
        let mut padded = vec![0.0f32; pad_left];
        padded.extend_from_slice(samples);
        if padded.len() < frame_size_for_unfold {
            padded.resize(frame_size_for_unfold, 0.0);
        }

        let hop = self.params.hop_length;
        let n_frames = (padded.len() - frame_size_for_unfold) / hop + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut spectro = Array2::<f32>::zeros((self.params.n_mels, n_frames.max(1)));

        for f in 0..n_frames {
            let start = f * hop;
            let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
            for i in 0..self.params.frame_length.min(n_fft) {
                let w = window.get(i).copied().unwrap_or(1.0);
                buf[i] = Complex32::new(padded[start + i] * w, 0.0);
            }
            fft.process(&mut buf);
            for m in 0..self.params.n_mels {
                let filter = &mel_filters[m * fft_bins..(m + 1) * fft_bins];
                let mut acc = 0.0f32;
                for bin in 0..fft_bins {
                    let c = buf[bin];
                    let power = c.re.mul_add(c.re, c.im * c.im);
                    acc = filter[bin].mul_add(power, acc);
                }
                spectro[[m, f]] = acc.max(1e-3).ln();
            }
        }
        Ok(spectro)
    }
}

impl AudioPreProcessor for Gemma4AudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let mut mels = Vec::with_capacity(clips.len());
        let mut lengths = Vec::with_capacity(clips.len());
        let mut soft_counts = Vec::with_capacity(clips.len());
        let mut max_frames = 0usize;

        for clip in clips {
            let samples = self.ensure_mono_sr(&clip.decoded)?;
            let soft = self.params.soft_tokens_for_samples(samples.len()).max(1);
            let mel = self.log_mel(&samples)?;
            max_frames = max_frames.max(mel.ncols());
            lengths.push(mel.ncols());
            soft_counts.push(soft);
            mels.push(mel);
        }

        let mut batch = Array3::<f32>::zeros((clips.len(), self.params.n_mels, max_frames));
        let mut mask = Array2::<i64>::zeros((clips.len(), max_frames));
        for (i, mel) in mels.iter().enumerate() {
            let frames = mel.ncols();
            for m in 0..self.params.n_mels {
                for t in 0..frames {
                    batch[[i, m, t]] = mel[[m, t]];
                }
            }
            for t in 0..frames {
                mask[[i, t]] = 1;
            }
        }

        let item_sizes = clips
            .iter()
            .map(|c| (c.decoded.samples.len() as u32, c.decoded.sample_rate as u32))
            .collect();
        let mut out = PreprocessedEncoderInputs::new(batch, soft_counts, item_sizes);
        out.model_specific.insert(
            "input_features_mask".to_string(),
            ModelSpecificValue::IntTensor {
                data: mask.iter().copied().collect(),
                shape: vec![clips.len(), max_frames],
            },
        );
        out.model_specific.insert(
            "audio_feature_lengths".to_string(),
            ModelSpecificValue::IntVec(lengths.into_iter().map(|v| v as i64).collect()),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_token_arithmetic_positive() {
        let p = Gemma4AudioParams::default();
        // 1 second of audio at 16 kHz.
        let soft = p.soft_tokens_for_samples(16_000);
        assert!(soft > 0);
        assert!(soft <= p.audio_seq_length);
    }
}
