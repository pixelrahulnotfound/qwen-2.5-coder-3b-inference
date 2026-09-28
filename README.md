

## Model

Qwen2.5-Coder-3B-Instruct, the qwen2 arch one:

hidden 2048, 36 layers, 16 Q heads and only 2 KV heads, head_dim 128. SwiGLU with 11008 intermediate, vocab 151936, RMSNorm 1e-6, RoPE theta 1M. Embeddings are tied.

I bench with the Q4_K_M GGUF since that's what llama.cpp uses. F16 is better if you just want to check correctness.

## Run it

```bash
cargo run --release -- --gguf /path/to/qwen2.5-coder-3b-instruct-q4_k_m.gguf --prompt "def fibonacci(n):" --max-tokens 256
```

With sampling:

```bash
cargo run --release -- --gguf model.gguf --prompt "write quicksort in python" --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 256
```

For the tokenizer pass `--tokenizer tokenizer.json` or let it find the cached one from the Qwen models in `~/.cache/huggingface`. Any Qwen2.5 tokenizer works, they all share the same vocab. If it can't find one it'll tell you to download it with huggingface-cli.

Add `--raw` if you don't want the ChatML wrapper.

## Against llama.cpp

Same file, same sampler, same machine, that's the only fair way:

```bash
llama-bench -m qwen2.5-coder-3b-instruct-q4_k_m.gguf -p 24 -n 32
RAYON_NUM_THREADS=6 ./target/release/qwen-infer --gguf qwen2.5-coder-3b-instruct-q4_k_m.gguf --prompt "def fibonacci(n):" --max-tokens 32 --temp 0
```

On my i9-13900H with 6 threads I get:

- llama.cpp: pp 45.9, tg 11.7
- this: prefill 26.9, decode 11.6

So decode basically ties now. Prefill is about half. The big wins were building with target-cpu=native + fat LTO, doing grouped attention instead of repeating the KV heads 8x, and hoisting the causal mask out of the layer loop. The rest of the prefill gap is just better quantized GEMM kernels and repacked weights, which is where llama.cpp has years on me.

Run it a few times and drop the first, laptop thermals make the numbers jump around.

## Code

Nothing fancy:

- `config.rs` just the dims for the 3B model
- `loader.rs` opens the GGUF with candle's gguf reader
- `model.rs` RMSNorm, RoPE, attention, MLP, the whole forward
- `cache.rs` KV-cache per layer
- `sampler.rs` greedy plus temp / top-k / top-p
- `main.rs` CLI, ChatML prompt, generate loop, timing
