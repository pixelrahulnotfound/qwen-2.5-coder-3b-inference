# qwen-infer

Small CPU inference engine for Qwen2.5-Coder-3B, built to learn Rust and to compare against llama.cpp.

## Why

llama.cpp is the reference. This repo reimplements the same Qwen2 decoder with `candle-core` tensors on CPU + GGUF weights, with KV-cache and greedy / temp sampling, so prefill tok/s and decode tok/s can be compared head to head.

## Model

- Qwen2.5-Coder-3B-Instruct, qwen2 arch
- hidden 2048, 36 layers, 16 Q heads / 2 KV heads, head_dim 128
- SwiGLU 11008, vocab 151936, RMSNorm eps 1e-6, RoPE theta 1M, tied embeddings
- Weights: GGUF (`Q4_K_M` for bench parity, or F16 for correctness checks)

## Run

```bash
cargo run --release -- --gguf /path/to/qwen2.5-coder-3b-instruct-q4_k_m.gguf --prompt "def fibonacci(n):" --max-tokens 256
```

With sampling:

```bash
cargo run --release -- --gguf model.gguf --prompt "write quicksort in python" --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 256
```

The tokenizer is fetched from `Qwen/Qwen2.5-Coder-3B-Instruct` via hf-hub on first run, or pass `--tokenizer tokenizer.json`.

## Bench vs llama.cpp

Use the same file and sampler on both sides:

```bash
llama-bench -m qwen2.5-coder-3b-instruct-q4_k_m.gguf -p 128 -n 256
cargo run --release -- --gguf qwen2.5-coder-3b-instruct-q4_k_m.gguf --prompt "<128 token code prompt>" --max-tokens 256 --temp 0
```

Compare prefill tok/s (prompt processing) and decode tok/s separately, plus peak RSS. Same threads, same machine, 5 runs, drop the cold one.

Measured here (6 threads, Q4_K_M, 24 prompt + 32 gen, i9-13900H):

- llama.cpp: pp 45.9 tok/s, tg 11.7 tok/s
- qwen-infer: prefill 26.9 tok/s, decode 11.6 tok/s

Decode ties llama.cpp single-stream after grouped GQA (no KV repeat allocs) + native LTO build. Prefill is 59% — remaining gap is quantized GEMM blocking + repacked weights, which is hand-SIMD territory.

## Layout

- `src/config.rs` model dims
- `src/loader.rs` GGUF loading
- `src/model.rs` RMSNorm, RoPE, attention, MLP, full decoder
- `src/cache.rs` KV cache
- `src/sampler.rs` greedy / temp / top-k / top-p
- `src/main.rs` CLI + generate loop + stats
