use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::path::Path;

pub struct Gguf {
    pub content: Content,
    pub path: String,
}

pub fn open(path: &Path) -> Result<Gguf> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let content = Content::read(&mut f).map_err(anyhow::Error::msg)?;
    Ok(Gguf {
        content,
        path: path.display().to_string(),
    })
}

impl Gguf {
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.content.tensor_infos.keys().cloned().collect();
        v.sort();
        v
    }

    fn file(&self) -> Result<File> {
        File::open(&self.path).with_context(|| format!("reopen {}", self.path))
    }

    pub fn qmat(&self, name: &str, dev: &Device) -> Result<QMatMul> {
        let mut f = self.file()?;
        let qt = self
            .content
            .tensor(&mut f, name, dev)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("missing tensor {name}"))?;
        QMatMul::from_qtensor(qt).map_err(anyhow::Error::msg)
    }

    pub fn tensor(&self, name: &str, dev: &Device) -> Result<Tensor> {
        let mut f = self.file()?;
        let qt = self
            .content
            .tensor(&mut f, name, dev)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("missing tensor {name}"))?;
        qt.dequantize(dev).map_err(anyhow::Error::msg)
    }

    pub fn bias(&self, name: &str, dev: &Device) -> Result<Option<Tensor>> {
        if !self.content.tensor_infos.contains_key(name) {
            return Ok(None);
        }
        Ok(Some(self.tensor(name, dev)?))
    }
}

pub fn require(g: &Gguf, prefix: &str) -> Result<()> {
    let n = g.names().iter().filter(|x| x.starts_with(prefix)).count();
    if n == 0 {
        bail!("no tensors with prefix {prefix}");
    }
    Ok(())
}
