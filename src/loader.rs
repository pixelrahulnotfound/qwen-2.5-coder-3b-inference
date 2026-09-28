
use anyhow::{Context, Result};
use std::path::Path;

pub fn inspect_gguf(path: &Path) -> Result<()> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("cannot stat gguf file: {}", path.display()))?;
    println!("gguf: {} ({} bytes)", path.display(), meta.len());
    println!("TODO(M1): parse with candle_core::quantized::gguf_file::Content::read");
    println!("TODO(M1): list tensors: embed_tokens, layers.*.q/k/v/o_proj, mlp.gate/up/down_proj, norm weights");
    Ok(())
}
