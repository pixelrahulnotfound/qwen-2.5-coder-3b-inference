use candle_core::Tensor;

pub struct LayerCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn update(&mut self, k: Tensor, v: Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(old_k), Some(old_v)) => {
                let k = Tensor::cat(&[old_k, &k], 2).map_err(anyhow::Error::msg)?;
                let v = Tensor::cat(&[old_v, &v], 2).map_err(anyhow::Error::msg)?;
                (k, v)
            }
            _ => (k, v),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

pub struct KvCache {
    pub layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerCache::new()).collect(),
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }
}
