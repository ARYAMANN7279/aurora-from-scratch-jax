"""LoRA on the backbone attention (handouts/stage10_lora.md, section 1).

The LoRA factors are a separate pytree mirroring the backbone's Swin blocks. `merge` returns the
full parameter pytree with W' = W + (alpha/r) A B on every block's qkv and proj. Training
differentiates only the LoRA pytree, so the base parameters can never change.
"""
import jax
import jax.numpy as jnp


def _pair(key, d_in, d_out, r, dtype):
    return {"A": jax.random.normal(key, (d_in, r), dtype) / jnp.sqrt(d_in), "B": jnp.zeros((r, d_out), dtype)}


def init(key, base, r=4):
    """base: full model params (with base["bb"]). Returns the LoRA pytree (B = 0)."""
    out = {"enc": [], "dec": []}
    for side in ("enc", "dec"):
        for layer in base["bb"][side]:
            blocks = []
            for b in layer["blocks"]:
                key, k1, k2 = jax.random.split(key, 3)
                D = b["proj"]["W"].shape[0]
                dt = b["proj"]["W"].dtype
                blocks.append({"qkv": _pair(k1, D, 3 * D, r, dt), "proj": _pair(k2, D, D, r, dt)})
            out[side].append(blocks)
    return out


def merge(base, lora, r=4, alpha=None):
    scale = (alpha if alpha is not None else r) / r
    p = dict(base)
    bb = {**base["bb"], "enc": [], "dec": []}
    for side in ("enc", "dec"):
        for layer, lblocks in zip(base["bb"][side], lora[side]):
            blocks = []
            for b, lb in zip(layer["blocks"], lblocks):
                nb = dict(b)
                for name in ("qkv", "proj"):
                    nb[name] = {"W": b[name]["W"] + scale * lb[name]["A"] @ lb[name]["B"], "b": b[name]["b"]}
                blocks.append(nb)
            bb[side].append({**layer, "blocks": blocks})
    p["bb"] = bb
    return p


def n_params(lora):
    return sum(x.size for x in jax.tree_util.tree_leaves(lora))
