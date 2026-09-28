mod cache;
mod config;
mod loader;
mod model;
mod sampler;

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "qwen-infer", about = "Qwen2.5-Coder-3B inference (CPU + GGUF + candle-core)")]
struct Args {
    /// Path to Qwen2.5-Coder-3B GGUF file (e.g. Q4_K_M for bench parity with llama.cpp)
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// Prompt to tokenize / run (tokenizer wiring is M1)
    #[arg(long, default_value = "def fibonacci(n):")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    println!("device: {device:?}");

    let cfg = config::Qwen2Config::qwen25_coder_3b();
    println!(
        "config: hidden={} layers={} heads={}Q/{}KV inter={} vocab={} head_dim={} rope_theta={} tie_emb={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.head_dim(),
        cfg.rope_theta,
        cfg.tie_word_embeddings
    );
    println!("prompt: {}", args.prompt);

    if let Some(path) = args.gguf.as_deref() {
        loader::inspect_gguf(path)?;
    } else {
        println!("tip: pass --gguf <Qwen2.5-Coder-3B-*.gguf> to inspect weights");
        println!("tip: use the same Q4_K_M file as llama.cpp for fair bench later");
    }

    // M1 next: load tokenizer.json, GGUF tensors, single forward pass.
    Ok(())
}
