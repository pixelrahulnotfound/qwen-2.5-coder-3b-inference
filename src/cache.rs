//! KV-cache placeholder.
//!
//! M2: store K/V per layer, append on each decode step, reuse buffers (no alloc per token).

use candle_core::Tensor;

pub struct KvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}
