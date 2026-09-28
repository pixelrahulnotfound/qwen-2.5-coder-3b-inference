mod cache;
mod config;
mod loader;
mod model;
mod sampler;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use cache::KvCache;
use config::Qwen2Config;
use sampler::Sampler;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "def fibonacci(n):")]
    prompt: String,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temp: f64,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,
    #[arg(long, default_value_t = false)]
    raw: bool,
}

fn chatml(prompt: &str) -> String {
    format!("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn find_tokenizer() -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let base = PathBuf::from(format!("{home}/.cache/huggingface/hub"));
    let cands = [
        "models--Qwen--Qwen2.5-Coder-3B-Instruct",
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "models--Qwen--Qwen2.5-0.5B-Instruct",
    ];
    for c in cands {
        let snapshots = base.join(c).join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for e in entries.flatten() {
                let p = e.path().join("tokenizer.json");
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn load_tokenizer(path_opt: Option<PathBuf>) -> Result<tokenizers::Tokenizer> {
    if let Some(p) = path_opt {
        return tokenizers::Tokenizer::from_file(p).map_err(anyhow::Error::msg);
    }
    if let Some(p) = find_tokenizer() {
        println!("using tokenizer {}", p.display());
        return tokenizers::Tokenizer::from_file(p).map_err(anyhow::Error::msg);
    }
    anyhow::bail!("no tokenizer.json found, pass --tokenizer or run: huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct tokenizer.json")
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = Device::Cpu;

    let cfg = Qwen2Config::qwen25_coder_3b();
    println!("loading {}", args.gguf.display());
    let g = loader::open(&args.gguf)?;
    println!("tensors: {}", g.names().len());
    loader::require(&g, "blk.0.")?;

    let tok = load_tokenizer(args.tokenizer)?;
    let text = if args.raw { args.prompt.clone() } else { chatml(&args.prompt) };
    let enc = tok.encode(text, false).map_err(anyhow::Error::msg)?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    println!("prompt tokens: {}", ids.len());

    let im_end = tok.token_to_id("<|im_end|>").unwrap_or(151645);
    let eot = tok.token_to_id("<|endoftext|>").unwrap_or(151643);

    let w = model::Weights::load(&g, &cfg, &dev)?;
    println!("weights loaded");
    let mut cache = KvCache::new(cfg.num_hidden_layers);
    let sampler = Sampler::new(args.temp, args.top_k, args.top_p);

    let t0 = Instant::now();
    let logits = w.forward(&ids, 0, &mut cache)?;
    let prefill_dt = t0.elapsed();
    let last = logits
        .narrow(1, ids.len() - 1, 1)
        .map_err(anyhow::Error::msg)?
        .squeeze(0)
        .map_err(anyhow::Error::msg)?
        .squeeze(0)
        .map_err(anyhow::Error::msg)?
        .unsqueeze(0)
        .map_err(anyhow::Error::msg)?;
    let mut next = sampler.sample(&last).with_context(|| "prefill sample")?;

    let mut out_ids = vec![next];
    let mut pos = ids.len();

    let t1 = Instant::now();
    for _ in 1..args.max_tokens {
        if next == im_end || next == eot {
            break;
        }
        let l = w.forward(&[next], pos, &mut cache)?;
        let l = l.squeeze(0).map_err(anyhow::Error::msg)?.squeeze(0).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        next = sampler.sample(&l)?;
        out_ids.push(next);
        pos += 1;
        if next == im_end || next == eot {
            break;
        }
    }
    let decode_dt = t1.elapsed();

    let txt = tok.decode(&out_ids, true).map_err(anyhow::Error::msg)?;
    println!("{txt}");

    let pf_s = prefill_dt.as_secs_f64().max(1e-6);
    let dc_s = decode_dt.as_secs_f64().max(1e-6);
    println!();
    println!("prefill {} tok in {:.2}s = {:.1} tok/s", ids.len(), pf_s, ids.len() as f64 / pf_s);
    println!("decode {} tok in {:.2}s = {:.1} tok/s", out_ids.len(), dc_s, out_ids.len() as f64 / dc_s);

    Ok(())
}
