"""E9, exploratory (not a pre-registered prediction): how selective is the trained attention?

For the kept weights of E8's B, R and Q and E9's RQ (seeds 0-2), runs the model eagerly on 8 evaluation inputs
(trajectory 2, frames 99-106 and 100-107) with the window attention instrumented. For each of the 12 Swin blocks it
records, averaged over windows, heads and queries:
  entropy   the attention entropy over the keys a query may see, divided by the log of their number
            (1 = uniform over the window, 0 = all weight on one key);
  max_attn  the largest attention weight of a query;
  logit_sd  the standard deviation of a query's logits over those keys (the bias included).
The spec's overlap hypothesis (section 1) says both changes let attention become selective; if so, B's trained
attention should stay closer to uniform than R's, Q's and RQ's.
usage: attention.py <E8 params dir> <E9 params dir>   (writes results/attention.json)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aurora import harness, model, swin  # noqa: E402
from kdata import traj_dir  # noqa: E402

REC = []
original = swin.window_attention


def recording(p, xw, heads, mask=None, bias=None):
    """swin.window_attention, recording its attention statistics on the way."""
    Bw, N, D = xw.shape
    dh = D // heads
    qkv = (xw @ p["qkv"]["W"] + p["qkv"]["b"]).reshape(Bw, N, 3, heads, dh).transpose(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    if "qn" in p:
        q = swin.layernorm_noaffine(q) * p["qn"]["g"] + p["qn"]["b"]
        k = swin.layernorm_noaffine(k) * p["kn"]["g"] + p["kn"]["b"]
    lg = np.asarray(q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(jnp.asarray(dh, xw.dtype)))
    if bias is not None:
        lg = lg + np.asarray(bias)[None]
    ok = np.ones((Bw, 1, N, N), bool) if mask is None else np.tile(np.asarray(mask), (Bw // mask.shape[0], 1, 1))[:, None]
    m = np.where(ok, lg, -np.inf)
    a = np.exp(m - m.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    H = -np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0)), 0.0).sum(-1)
    REC.append(dict(entropy=float(np.mean(H / np.log(np.maximum(ok.sum(-1), 2)))), max_attn=float(np.mean(a.max(-1))),
                    logit_sd=float(np.nanmean(np.nanstd(np.where(ok, lg, np.nan), axis=-1)))))
    return original(p, xw, heads, mask, bias)


swin.window_attention = recording
ds = os.path.join(ROOT, "data", "stage2", "e1_v1")
v = np.load(os.path.join(traj_dir(ds, "re300_kf4", 2), "frames_64.npy"))
runs = {f"e8_{a}_s{s}": sys.argv[1] for a in ("B", "R", "Q") for s in (0, 1, 2)}
runs.update({f"e9_RQ_s{s}": sys.argv[2] for s in (0, 1, 2)})
cfgs = {}
for d in (sys.argv[1], sys.argv[2]):
    for line in open(os.path.join(d, "rows.jsonl")):
        r = json.loads(line)
        cfgs[r["name"]] = r["config"]
out = {}
for name, d in runs.items():
    mc = harness.model_config(cfgs[name]["model"])
    z = np.load(os.path.join(d, f"{name}.npz"))
    p = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32))[1](jnp.asarray(z["flat"]))
    x = ((v - z["mean"][:, None, None]) / z["std"][:, None, None]).astype(np.float32)
    per_input = []
    for t in range(100, 108):
        REC.clear()
        model.apply(p, jnp.asarray(x[t - 1]), jnp.asarray(x[t]), model.VARS, 0.2, mc)
        per_input.append([dict(r) for r in REC])
    blocks = [{k: float(np.mean([pi[b][k] for pi in per_input])) for k in ("entropy", "max_attn", "logit_sd")}
              for b in range(len(per_input[0]))]
    out[name] = dict(blocks=blocks, mean={k: float(np.mean([b[k] for b in blocks])) for k in ("entropy", "max_attn", "logit_sd")})
    print(name, {k: round(x, 3) for k, x in out[name]["mean"].items()}, flush=True)
with open(os.path.join(HERE, "results", "attention.json"), "w") as f:
    json.dump(out, f, indent=1)
