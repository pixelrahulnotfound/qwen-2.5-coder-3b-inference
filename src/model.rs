use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_core::{Module, D, DType, Device, Tensor};

use crate::cache::KvCache;
use crate::config::Qwen2Config;
use crate::loader;

pub struct RmsNorm {
    pub w: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.dim(D::Minus1)?;
        let s2 = x.sqr().map_err(anyhow::Error::msg)?.sum_keepdim(D::Minus1).map_err(anyhow::Error::msg)?;
        let v = (s2 / h as f64).map_err(anyhow::Error::msg)?;
        let std = (v + self.eps).map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
        let n = x.broadcast_div(&std).map_err(anyhow::Error::msg)?;
        n.broadcast_mul(&self.w).map_err(anyhow::Error::msg)
    }
}

pub struct Rotary {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rotary {
    pub fn new(cfg: &Qwen2Config, dev: &Device) -> Result<Self> {
        let d = cfg.head_dim();
        let half = d / 2;
        let mut inv = Vec::with_capacity(half);
        for i in 0..half {
            inv.push(1.0 / cfg.rope_theta.powf(2.0 * i as f64 / d as f64));
        }
        let max = cfg.max_position_embeddings.min(32768);
        let mut cos_data = Vec::with_capacity(max * d);
        let mut sin_data = Vec::with_capacity(max * d);
        for p in 0..max {
            let mut c = Vec::with_capacity(half);
            let mut s = Vec::with_capacity(half);
            for &f in &inv {
                let a = p as f64 * f;
                c.push(a.cos() as f32);
                s.push(a.sin() as f32);
            }
            cos_data.extend_from_slice(&c);
            cos_data.extend_from_slice(&c);
            sin_data.extend_from_slice(&s);
            sin_data.extend_from_slice(&s);
        }
        let cos = Tensor::from_vec(cos_data, (max, d), dev).map_err(anyhow::Error::msg)?;
        let sin = Tensor::from_vec(sin_data, (max, d), dev).map_err(anyhow::Error::msg)?;
        Ok(Self { cos, sin })
    }
}

fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1).map_err(anyhow::Error::msg)?;
    let h = d / 2;
    let x1 = x.narrow(D::Minus1, 0, h).map_err(anyhow::Error::msg)?;
    let x2 = x.narrow(D::Minus1, h, h).map_err(anyhow::Error::msg)?;
    let neg_x2 = x2.neg().map_err(anyhow::Error::msg)?;
    let rot = Tensor::cat(&[&neg_x2, &x1], D::Minus1).map_err(anyhow::Error::msg)?;
    let a = x.broadcast_mul(cos).map_err(anyhow::Error::msg)?;
    let b = rot.broadcast_mul(sin).map_err(anyhow::Error::msg)?;
    a.broadcast_add(&b).map_err(anyhow::Error::msg)
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1).map_err(anyhow::Error::msg)?;
    let e = x
        .broadcast_sub(&m)
        .map_err(anyhow::Error::msg)?
        .exp()
        .map_err(anyhow::Error::msg)?;
    let s = e.sum_keepdim(D::Minus1).map_err(anyhow::Error::msg)?;
    e.broadcast_div(&s).map_err(anyhow::Error::msg)
}

fn attn_grouped(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, off: usize, t: usize, dev: &Device) -> Result<Tensor> {
    let n_kv = k.dim(1).map_err(anyhow::Error::msg)?;
    let n_h = q.dim(1).map_err(anyhow::Error::msg)?;
    let gsize = n_h / n_kv;
    let kt = k.dim(2).map_err(anyhow::Error::msg)?;
    let mut outs = Vec::new();
    for g in 0..n_kv {
        let qg = q.narrow(1, g * gsize, gsize).map_err(anyhow::Error::msg)?;
        let kg = k.narrow(1, g, 1).map_err(anyhow::Error::msg)?;
        let vg = v.narrow(1, g, 1).map_err(anyhow::Error::msg)?;
        let kt_ = kg.t().map_err(anyhow::Error::msg)?;
        let mut s = qg.broadcast_matmul(&kt_).map_err(anyhow::Error::msg)?;
        s = (s * scale).map_err(anyhow::Error::msg)?;
        if t > 1 {
            let mut m = vec![0f32; t * kt];
            for i in 0..t {
                for j in 0..kt {
                    if j > off + i {
                        m[i * kt + j] = -1e9;
                    }
                }
            }
            let mask = Tensor::from_vec(m, (t, kt), dev).map_err(anyhow::Error::msg)?.broadcast_as((1, gsize, t, kt)).map_err(anyhow::Error::msg)?;
            s = s.broadcast_add(&mask).map_err(anyhow::Error::msg)?;
        }
        let p = softmax_last(&s)?;
        let o = p.broadcast_matmul(&vg).map_err(anyhow::Error::msg)?;
        outs.push(o);
    }
    Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1).map_err(anyhow::Error::msg)
}

pub struct LayerW {
    pub q: QMatMul,
    pub k: QMatMul,
    pub v: QMatMul,
    pub o: QMatMul,
    pub gate: QMatMul,
    pub up: QMatMul,
    pub down: QMatMul,
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

pub struct Weights {
    pub embed: QMatMul,
    pub layers: Vec<LayerW>,
    pub out_norm: RmsNorm,
    pub out: QMatMul,
    pub cfg: Qwen2Config,
    pub rot: Rotary,
    pub dev: Device,
}

impl Weights {
    pub fn load(g: &loader::Gguf, cfg: &Qwen2Config, dev: &Device) -> Result<Self> {
        let embed = g.qmat("token_embd.weight", dev)?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("blk.{i}");
            let q = g.qmat(&format!("{p}.attn_q.weight"), dev)?;
            let k = g.qmat(&format!("{p}.attn_k.weight"), dev)?;
            let v = g.qmat(&format!("{p}.attn_v.weight"), dev)?;
            let o = g.qmat(&format!("{p}.attn_output.weight"), dev)?;
            let gate = g.qmat(&format!("{p}.ffn_gate.weight"), dev)?;
            let up = g.qmat(&format!("{p}.ffn_up.weight"), dev)?;
            let down = g.qmat(&format!("{p}.ffn_down.weight"), dev)?;
            let q_bias = g.bias(&format!("{p}.attn_q.bias"), dev)?;
            let k_bias = g.bias(&format!("{p}.attn_k.bias"), dev)?;
            let v_bias = g.bias(&format!("{p}.attn_v.bias"), dev)?;
            let an = g.tensor(&format!("{p}.attn_norm.weight"), dev)?;
            let ffn = g.tensor(&format!("{p}.ffn_norm.weight"), dev)?;
            layers.push(LayerW {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                q_bias,
                k_bias,
                v_bias,
                attn_norm: RmsNorm { w: an, eps: cfg.rms_norm_eps },
                ffn_norm: RmsNorm { w: ffn, eps: cfg.rms_norm_eps },
            });
        }
        let on = g.tensor("output_norm.weight", dev)?;
        let out = if g.content.tensor_infos.contains_key("output.weight") {
            g.qmat("output.weight", dev)?
        } else {
            embed.clone()
        };
        let rot = Rotary::new(cfg, dev)?;
        Ok(Self {
            embed,
            layers,
            out_norm: RmsNorm { w: on, eps: cfg.rms_norm_eps },
            out,
            cfg: cfg.clone(),
            rot,
            dev: dev.clone(),
        })
    }

    pub fn forward(&self, toks: &[u32], off: usize, cache: &mut KvCache) -> Result<Tensor> {
        let cfg = &self.cfg;
        let dev = &self.dev;
        let t = toks.len();
        let ids = Tensor::from_vec(toks.to_vec(), (1, t), dev).map_err(anyhow::Error::msg)?;
        let mut h = self.embed.embedding(&ids).map_err(anyhow::Error::msg)?.to_dtype(DType::F32).map_err(anyhow::Error::msg)?;

        let hd = cfg.head_dim();
        let scale = 1.0 / (hd as f64).sqrt();
        let n_h = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;

        let cos_all = self.rot.cos.narrow(0, off, t).map_err(anyhow::Error::msg)?;
        let sin_all = self.rot.sin.narrow(0, off, t).map_err(anyhow::Error::msg)?;
        let cos_q = cos_all.reshape((1, 1, t, hd)).map_err(anyhow::Error::msg)?;
        let sin_q = sin_all.reshape((1, 1, t, hd)).map_err(anyhow::Error::msg)?;

        for (li, lw) in self.layers.iter().enumerate() {
            let r = lw.attn_norm.forward(&h)?;
            let mut q = lw.q.forward(&r).map_err(anyhow::Error::msg)?;
            let mut k = lw.k.forward(&r).map_err(anyhow::Error::msg)?;
            let mut v = lw.v.forward(&r).map_err(anyhow::Error::msg)?;
            if let Some(b) = &lw.q_bias {
                q = q.broadcast_add(b).map_err(anyhow::Error::msg)?;
            }
            if let Some(b) = &lw.k_bias {
                k = k.broadcast_add(b).map_err(anyhow::Error::msg)?;
            }
            if let Some(b) = &lw.v_bias {
                v = v.broadcast_add(b).map_err(anyhow::Error::msg)?;
            }

            let q = q.reshape((1, t, n_h, hd)).map_err(anyhow::Error::msg)?.transpose(1, 2).map_err(anyhow::Error::msg)?;
            let k = k.reshape((1, t, n_kv, hd)).map_err(anyhow::Error::msg)?.transpose(1, 2).map_err(anyhow::Error::msg)?;
            let v = v.reshape((1, t, n_kv, hd)).map_err(anyhow::Error::msg)?.transpose(1, 2).map_err(anyhow::Error::msg)?;

            let q = rope_apply(&q, &cos_q, &sin_q)?;
            let k = rope_apply(&k, &cos_q, &sin_q)?;

            let (k, v) = cache.layers[li].update(k, v)?;

            let o = attn_grouped(&q, &k, &v, scale, off, t, dev)?;
            let o = o.transpose(1, 2).map_err(anyhow::Error::msg)?.reshape((1, t, cfg.hidden_size)).map_err(anyhow::Error::msg)?;
            let o = lw.o.forward(&o).map_err(anyhow::Error::msg)?;
            h = (&h + &o).map_err(anyhow::Error::msg)?;

            let r2 = lw.ffn_norm.forward(&h)?;
            let g = lw.gate.forward(&r2).map_err(anyhow::Error::msg)?.silu().map_err(anyhow::Error::msg)?;
            let u = lw.up.forward(&r2).map_err(anyhow::Error::msg)?;
            let f = (&g * &u).map_err(anyhow::Error::msg)?;
            let f = lw.down.forward(&f).map_err(anyhow::Error::msg)?;
            h = (&h + &f).map_err(anyhow::Error::msg)?;
        }

        let h = self.out_norm.forward(&h)?;
        self.out.forward(&h).map_err(anyhow::Error::msg)
    }
}
