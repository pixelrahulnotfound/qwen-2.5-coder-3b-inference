//! Sampler: greedy first, then temp/top-k/top-p to match llama.cpp.

use anyhow::Result;
use candle_core::Tensor;

/// Argmax (greedy) sampling, temp = 0. Matches `llama-cli --temp 0`.
pub fn greedy_next_token(logits: &Tensor) -> Result<u32> {
    let token = logits.argmax(1)?.get(0)?.to_scalar::<u32>()?;
    Ok(token)
}
