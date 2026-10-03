//! Granite 4 causal LM. Linear weights stay in compressed-tensors FP8 and are
//! dequantized for one projection at a time so an 8 GB GPU is not asked to hold
//! a dense 3B model.

mod fp8;
mod model;

pub use model::{is_granite_model, load_granite, summarize_text, GraniteConfig};
