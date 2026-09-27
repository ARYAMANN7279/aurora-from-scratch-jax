"""3-D Swin U-Net backbone (handouts/stage06_backbone.md, section 1), matching Aurora's
Swin3DTransformerBackbone in deterministic mode. Parameters are plain pytrees."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import swin
from .attention import init_ln, layernorm


class Config(NamedTuple):
    D: int
    enc_depths: tuple
    enc_heads: tuple
    dec_depths: tuple
    dec_heads: tuple
    window: tuple
    periodic: tuple = swin.PERIODIC
    shift: bool = True  # False: every block uses unshifted windows (E4 ablation)


def _tn(key, shape, dtype):
    return swin._tn(key, shape, dtype)


def init_merge(key, w, dtype):
    return {"norm": init_ln(4 * w, dtype), "W": _tn(key, (4 * w, 2 * w), dtype)}


def merge(p, x, res):
    """Patch merging: 2x2 in (H, W) -> LayerNorm(4w) -> Linear(4w -> 2w, no bias)."""
    C, H, W = res
    B, L, D = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2, D).transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, C * (H // 2) * (W // 2), 4 * D)
    return layernorm(p["norm"], x) @ p["W"]


def init_split(key, w, dtype):
    k1, k2 = jax.random.split(key)
    out = w // 2
    return {"W1": _tn(k1, (w, 4 * out), dtype), "norm": init_ln(out, dtype), "W2": _tn(k2, (out, out), dtype)}


def split(p, x, res):
    """Patch splitting: Linear(w -> 4*(w/2), no bias) -> 2x2 tokens -> LayerNorm -> Linear(no bias)."""
    C, H, W = res
    B, L, _ = x.shape
    out = p["W2"].shape[0]
    x = (x @ p["W1"]).reshape(B, C, H, W, 2, 2, out).transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, C * 2 * H * 2 * W, out)
    return layernorm(p["norm"], x) @ p["W2"]


def init(key, cfg, dtype=None):
    dtype = dtype or jnp.asarray(0.0).dtype
    S = len(cfg.enc_depths)
    if sum(cfg.enc_depths) != sum(cfg.dec_depths) or len(cfg.dec_depths) != S:
        raise ValueError("encoder and decoder must have the same number of stages and blocks")
    kt, ke, kd = jax.random.split(key, 3)
    k1, k2 = jax.random.split(kt)
    D = cfg.D
    p = {"time": {"W1": _tn(k1, (D, D), dtype), "b1": jnp.zeros(D, dtype),
                  "W2": _tn(k2, (D, D), dtype), "b2": jnp.zeros(D, dtype)},
         "enc": [], "dec": []}
    for i, (dep, ki) in enumerate(zip(cfg.enc_depths, jax.random.split(ke, S))):
        w = D * 2**i
        kb = jax.random.split(ki, dep + 1)
        layer = {"blocks": [swin.init_block(k, w, D, dtype=dtype) for k in kb[:dep]]}
        if i < S - 1:
            layer["merge"] = init_merge(kb[dep], w, dtype)
        p["enc"].append(layer)
    for i, (dep, ki) in enumerate(zip(cfg.dec_depths, jax.random.split(kd, S))):
        w = D * 2 ** (S - 1 - i)
        kb = jax.random.split(ki, dep + 1)
        layer = {"blocks": [swin.init_block(k, w, D, dtype=dtype) for k in kb[:dep]]}
        if i < S - 1:
            layer["split"] = init_split(kb[dep], w, dtype)
        p["dec"].append(layer)
    return p


def stage_resolutions(res, S):
    C, H, W = res
    out = [res]
    for _ in range(S - 1):
        if H % 2 or W % 2:
            raise ValueError(f"grid {res} cannot be halved {S - 1} times without padding")
        H, W = H // 2, W // 2
        out.append((C, H, W))
    return out


def _run_blocks(blocks, x, c, res, heads, cfg):
    half = tuple(w // 2 for w in cfg.window)
    for j, b in enumerate(blocks):
        shift = half if (cfg.shift and j % 2 == 1) else (0, 0, 0)
        x = swin.block(b, x, c, res, heads, cfg.window, shift, cfg.periodic)
    return x


def apply(p, x, t_emb, res, cfg):
    """x (B, C*H*W, D), t_emb (B, D) lead-time embedding -> (B, C*H*W, 2D)."""
    S = len(cfg.enc_depths)
    all_res = stage_resolutions(res, S)
    c = jax.nn.silu(t_emb @ p["time"]["W1"] + p["time"]["b1"]) @ p["time"]["W2"] + p["time"]["b2"]
    skips = []
    for i, layer in enumerate(p["enc"]):
        x = _run_blocks(layer["blocks"], x, c, all_res[i], cfg.enc_heads[i], cfg)
        skips.append(x)
        if "merge" in layer:
            x = merge(layer["merge"], x, all_res[i])
    for i, layer in enumerate(p["dec"]):
        idx = S - i - 1
        x = _run_blocks(layer["blocks"], x, c, all_res[idx], cfg.dec_heads[i], cfg)
        if "split" in layer:
            x = split(layer["split"], x, all_res[idx])
        if 0 < i < S - 1:
            x = x + skips[idx - 1]
        elif i == S - 1:
            x = jnp.concatenate([x, skips[0]], axis=-1)
    return x


def from_torch_state(sd, cfg):
    """Map an Aurora Swin3DTransformerBackbone state_dict (numpy arrays) to this pytree."""
    T = lambda k: jnp.asarray(sd[k].T)
    A = lambda k: jnp.asarray(sd[k])
    blk = lambda pre: swin.from_torch_state({k[len(pre) + 1:]: v for k, v in sd.items() if k.startswith(pre + ".")})
    S = len(cfg.enc_depths)
    p = {"time": {"W1": T("time_mlp.0.weight"), "b1": A("time_mlp.0.bias"),
                  "W2": T("time_mlp.2.weight"), "b2": A("time_mlp.2.bias")}, "enc": [], "dec": []}
    for i in range(S):
        pre = f"encoder_layers.{i}"
        layer = {"blocks": [blk(f"{pre}.blocks.{j}") for j in range(cfg.enc_depths[i])]}
        if i < S - 1:
            layer["merge"] = {"norm": {"g": A(f"{pre}.downsample.norm.weight"), "b": A(f"{pre}.downsample.norm.bias")},
                              "W": T(f"{pre}.downsample.reduction.weight")}
        p["enc"].append(layer)
    for i in range(S):
        pre = f"decoder_layers.{i}"
        layer = {"blocks": [blk(f"{pre}.blocks.{j}") for j in range(cfg.dec_depths[i])]}
        if i < S - 1:
            layer["split"] = {"W1": T(f"{pre}.upsample.lin1.weight"),
                              "norm": {"g": A(f"{pre}.upsample.norm.weight"), "b": A(f"{pre}.upsample.norm.bias")},
                              "W2": T(f"{pre}.upsample.lin2.weight")}
        p["dec"].append(layer)
    return p
