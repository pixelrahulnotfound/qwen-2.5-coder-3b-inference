use anyhow::Result;
use candle_core::Tensor;

pub struct Sampler {
    pub temp: f64,
    pub top_k: usize,
    pub top_p: f64,
}

impl Sampler {
    pub fn new(temp: f64, top_k: usize, top_p: f64) -> Self {
        Self { temp, top_k, top_p }
    }

    pub fn greedy(logits: &Tensor) -> Result<u32> {
        let t = logits
            .argmax(1)
            .map_err(anyhow::Error::msg)?
            .get(0)
            .map_err(anyhow::Error::msg)?
            .to_scalar::<u32>()
            .map_err(anyhow::Error::msg)?;
        Ok(t)
    }

    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        if self.temp <= 0.0 {
            return Self::greedy(logits);
        }
        let l = logits.get(0).map_err(anyhow::Error::msg)?;
        let v: Vec<f32> = l.to_vec1().map_err(anyhow::Error::msg)?;
        let mut idx: Vec<usize> = (0..v.len()).collect();

        if self.top_k > 0 && self.top_k < v.len() {
            idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            idx.truncate(self.top_k);
        }

        let mut vals: Vec<f32> = idx.iter().map(|&i| v[i] / self.temp as f32).collect();
        let m = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for x in &mut vals {
            *x = (*x - m).exp();
        }

        if self.top_p < 1.0 {
            let mut order: Vec<usize> = (0..vals.len()).collect();
            order.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap());
            let total: f32 = vals.iter().sum();
            let mut cum = 0.0;
            let mut keep = vec![false; vals.len()];
            for &o in &order {
                cum += vals[o] / total;
                keep[o] = true;
                if cum >= self.top_p as f32 {
                    break;
                }
            }
            let mut nv = Vec::new();
            let mut ni = Vec::new();
            for i in 0..vals.len() {
                if keep[i] {
                    nv.push(vals[i]);
                    ni.push(idx[i]);
                }
            }
            vals = nv;
            idx = ni;
        }

        let total: f32 = vals.iter().sum();
        let r: f32 = rand::random::<f32>() * total;
        let mut acc = 0.0;
        for i in 0..vals.len() {
            acc += vals[i];
            if r <= acc {
                return Ok(idx[i] as u32);
            }
        }
        Ok(*idx.last().unwrap() as u32)
    }
}
