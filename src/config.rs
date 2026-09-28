//! Qwen2 architecture config for Qwen2.5-Coder-3B.
//!
//! Values from `config.json` of Qwen/Qwen2.5-3B:
//! hidden 2048, layers 36, heads 16 Q / 2 KV, intermediate 11008,
//! vocab 151936, RMS eps 1e-6, RoPE theta 1_000_000, tied embeddings.

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self::qwen25_coder_3b()
    }
}

impl Qwen2Config {
    pub fn qwen25_coder_3b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 11008,
            num_hidden_layers: 36,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            tie_word_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
