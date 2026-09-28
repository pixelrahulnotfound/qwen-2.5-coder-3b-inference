//! Model structure placeholders.
//!
//! Next steps:
//! M1: RMSNorm + RoPE helpers + single forward (no cache) in bf16/f32.
//! M2: KV-cache wired in (see cache.rs).
//! M3: QMatMul path for Q4_K_M GGUF.

use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        anyhow::bail!("TODO(M1): implement RMSNorm")
    }

    pub fn device(&self) -> &Device {
        self.weight.device()
    }
}

// Placeholder for one decoder layer (attention + MLP).
pub struct Qwen2Layer {
    _hidden_size: usize,
}

impl Qwen2Layer {
    pub fn forward(&self, _x: &Tensor, _pos: usize) -> Result<Tensor> {
        anyhow::bail!("TODO(M1): implement attention (GQA 16Q/2KV) + RoPE + SwiGLU MLP")
    }
}
