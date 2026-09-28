"""3-D windowed / shifted-window attention block (handouts/stage05_swin.md, section 2).

Matches Aurora's Swin3DTransformerBlock (res-post-norm with adaptive LayerNorm), with the
shifted-window mask generalised by a `periodic` flag per axis (C, H, W). This project uses
periodic = (False, True, True): levels are not periodic, x and y are.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

PERIODIC = (False, True, True)


def adjust(window, shift, res):
    """Aurora's maybe_adjust_windows: an axis with res <= window gets window = res, shift = 0."""
    w, s = list(window), list(shift)
    for i, r in enumerate(res):
        if r <= window[i]:
            w[i], s[i] = r, 0
    return tuple(w), tuple(s)


def window_partition(x, ws):
    """(B, C, H, W, D) -> (B*nW, Wc, Wh, Ww, D), windows ordered (B, C1, H1, W1)."""
    B, C, H, W, D = x.shape
    x = x.reshape(B, C // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], D)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(-1, ws[0], ws[1], ws[2], D)


def window_reverse(win, ws, B, C, H, W):
    """Inverse of window_partition."""
    D = win.shape[-1]
    x = win.reshape(B, C // ws[0], H // ws[1], W // ws[2], ws[0], ws[1], ws[2], D)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(B, C, H, W, D)


def _axis_groups(n, w, s, periodic):
    if periodic or s == 0:
        return np.zeros(n, np.int64)
    return np.concatenate([np.zeros(n - w, np.int64), np.ones(w - s, np.int64), np.full(s, 2, np.int64)])


def shift_mask(res, ws, ss, periodic=PERIODIC):
    """(nW, N, N) bool: True where two tokens of a window may attend (same group). None if unneeded."""
    if all(s == 0 or p for s, p in zip(ss, periodic)):
        return None
    g = [_axis_groups(r, w, s, p) for r, w, s, p in zip(res, ws, ss, periodic)]
    ids = (g[0][:, None, None] * 9 + g[1][None, :, None] * 3 + g[2][None, None, :]).astype(np.float64)
    # numpy throughout, so the mask is a trace-time constant: under jit, a jnp array here would be a
    # tracer, and converting it back to numpy fails (reached when a non-periodic axis is shifted)
    win = window_partition(ids[None, ..., None], ws).reshape(-1, ws[0] * ws[1] * ws[2])
    return jnp.asarray(win[:, :, None] == win[:, None, :])


def layernorm_noaffine(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)


def adaln(p, y, c, scale_bias=0.0):
    """Adaptive LayerNorm: LN(y) * (scale_bias + scale) + shift, [shift, scale] = Linear(SiLU(c))."""
    mod = jax.nn.silu(c) @ p["W"] + p["b"]
    shift, scale = jnp.split(mod, 2, axis=-1)
    return layernorm_noaffine(y) * (scale_bias + scale[:, None]) + shift[:, None]


def window_attention(p, xw, heads, mask=None):
    """xw (B*nW, N, D); mask (nW, N, N) bool or None."""
    Bw, N, D = xw.shape
    dh = D // heads
    qkv = (xw @ p["qkv"]["W"] + p["qkv"]["b"]).reshape(Bw, N, 3, heads, dh).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    logits = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(jnp.asarray(dh, xw.dtype))
    if mask is not None:
        m = jnp.tile(mask, (Bw // mask.shape[0], 1, 1))[:, None]
        logits = jnp.where(m, logits, -jnp.inf)
    o = jax.nn.softmax(logits, axis=-1) @ v
    o = o.transpose(0, 2, 1, 3).reshape(Bw, N, D)
    return o @ p["proj"]["W"] + p["proj"]["b"]


def mlp(p, x):
    return jax.nn.gelu(x @ p["W1"] + p["b1"], approximate=False) @ p["W2"] + p["b2"]


def block(p, x, c, res, heads, window, shift, periodic=PERIODIC, scale_bias=0.0):
    """x (B, C*H*W, D), c (B, D_t) -> (B, C*H*W, D)."""
    C, H, W = res
    B, L, D = x.shape
    if L != C * H * W:
        raise ValueError(f"token count {L} != {C}*{H}*{W}")
    ws, ss = adjust(window, shift, res)
    if any(r % w for r, w in zip(res, ws)):
        raise ValueError(f"resolution {res} not divisible by window {ws}")
    xv = x.reshape(B, C, H, W, D)
    shifted = any(ss)
    if shifted:
        xv = jnp.roll(xv, shift=(-ss[0], -ss[1], -ss[2]), axis=(1, 2, 3))
    mask = shift_mask(res, ws, ss, periodic) if shifted else None
    N = ws[0] * ws[1] * ws[2]
    aw = window_attention(p, window_partition(xv, ws).reshape(-1, N, D), heads, mask)
    xv = window_reverse(aw.reshape(-1, *ws, D), ws, B, C, H, W)
    if shifted:
        xv = jnp.roll(xv, shift=ss, axis=(1, 2, 3))
    x = x + adaln(p["norm1"], xv.reshape(B, L, D), c, scale_bias)
    return x + adaln(p["norm2"], mlp(p["mlp"], x), c, scale_bias)


def _tn(key, shape, dtype, std=0.02):
    return std * jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)


def init_block(key, D, D_t, mlp_ratio=4, dtype=None):
    """Aurora's initialisation: truncated normal (std 0.02) linears, zero AdaLN modulation."""
    dtype = dtype or jnp.asarray(0.0).dtype
    k = jax.random.split(key, 4)
    Hd = int(mlp_ratio * D)
    zeros_mod = lambda: {"W": jnp.zeros((D_t, 2 * D), dtype), "b": jnp.zeros(2 * D, dtype)}
    return {"qkv": {"W": _tn(k[0], (D, 3 * D), dtype), "b": jnp.zeros(3 * D, dtype)},
            "proj": {"W": _tn(k[1], (D, D), dtype), "b": jnp.zeros(D, dtype)},
            "norm1": zeros_mod(), "norm2": zeros_mod(),
            "mlp": {"W1": _tn(k[2], (D, Hd), dtype), "b1": jnp.zeros(Hd, dtype),
                    "W2": _tn(k[3], (Hd, D), dtype), "b2": jnp.zeros(D, dtype)}}


def from_torch_state(sd):
    """Map an Aurora Swin3DTransformerBlock state_dict (numpy arrays) to this pytree."""
    lin = lambda name: {"W": jnp.asarray(sd[f"{name}.weight"].T), "b": jnp.asarray(sd[f"{name}.bias"])}
    fc1, fc2 = lin("mlp.fc1"), lin("mlp.fc2")
    return {"qkv": lin("attn.qkv"), "proj": lin("attn.proj"),
            "norm1": lin("norm1.ln_modulation.1"), "norm2": lin("norm2.ln_modulation.1"),
            "mlp": {"W1": fc1["W"], "b1": fc1["b"], "W2": fc2["W"], "b2": fc2["b"]}}
