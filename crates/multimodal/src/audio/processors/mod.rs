//! Model-specific audio preprocessing implementations.

mod gemma4_audio;
mod inkling;
mod qwen3_audio;

pub use gemma4_audio::{Gemma4AudioParams, Gemma4AudioProcessor};
pub use inkling::{InklingAudioEncoderParams, InklingAudioProcessor};
pub use qwen3_audio::{Qwen3AudioParams, Qwen3AudioProcessor};
