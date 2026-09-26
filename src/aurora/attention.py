"""Attention primitives (handouts/stage04_perceiver.md, 2.1-2.2): multi-head attention, LayerNorm,
MLP and the res-post-norm Perceiver block. Parameters are plain pytrees."""
import jax
import jax.numpy as jnp
import numpy as np


def _w(key, n_in, n_out, dtype):
    return jax.random.normal(key, (n_in, n_out), dtype) / jnp.sqrt(n_in)


def init_ln(D, dtype):
    return {"g": jnp.ones(D, dtype), "b": jnp.zeros(D, dtype)}


def layernorm(p, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * p["g"] + p["b"]


def init_mha(key, D, dtype):
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {"Wq": _w(kq, D, D, dtype), "Wk": _w(kk, D, D, dtype), "Wv": _w(kv, D, D, dtype),
            "Wo": _w(ko, D, D, dtype), "bo": jnp.zeros(D, dtype)}


def _check_mask(mask):
    try:
        m = np.asarray(mask)
    except Exception:  # traced under jit: cannot be checked here
        return
    if not m.any():
        raise ValueError("attention mask has no unmasked key")


def mha(p, q_in, kv_in, H, mask=None):
    """q_in (Q, D), kv_in (K, D), mask (K,) bool (True = attend) -> (Q, D)."""
    Q, D = q_in.shape
    K = kv_in.shape[0]
    dh = D // H
    q = (q_in @ p["Wq"]).reshape(Q, H, dh).transpose(1, 0, 2)
    k = (kv_in @ p["Wk"]).reshape(K, H, dh).transpose(1, 0, 2)
    v = (kv_in @ p["Wv"]).reshape(K, H, dh).transpose(1, 0, 2)
    logits = q @ k.transpose(0, 2, 1) / jnp.sqrt(jnp.asarray(dh, q.dtype))
    if mask is not None:
        _check_mask(mask)
        logits = jnp.where(mask[None, None, :], logits, -jnp.inf)
    w = jax.nn.softmax(logits, axis=-1)
    o = (w @ v).transpose(1, 0, 2).reshape(Q, D)
    return o @ p["Wo"] + p["bo"]


def init_mlp(key, D, dtype, mult=4):
    k1, k2 = jax.random.split(key)
    return {"W1": _w(k1, D, mult * D, dtype), "b1": jnp.zeros(mult * D, dtype),
            "W2": _w(k2, mult * D, D, dtype), "b2": jnp.zeros(D, dtype)}


def mlp(p, x):
    return jax.nn.gelu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def init_block(key, D, dtype):
    ka, km = jax.random.split(key)
    return {"attn": init_mha(ka, D, dtype), "ln1": init_ln(D, dtype), "mlp": init_mlp(km, D, dtype),
            "ln2": init_ln(D, dtype)}


def block(p, q_in, kv_in, H, mask=None):
    """Res-post-norm cross-attention block: h = LN(q + attn); out = LN(h + MLP(h))."""
    h = layernorm(p["ln1"], q_in + mha(p["attn"], q_in, kv_in, H, mask))
    return layernorm(p["ln2"], h + mlp(p["mlp"], h))
