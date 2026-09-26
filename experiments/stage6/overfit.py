"""V3/V4 (handouts/stage06_backbone.md, section 3): overfit one trajectory, one-step prediction.

usage: overfit.py [--steps 4000]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import importlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from aurora import backbone as bb  # noqa: E402
from aurora import encodings as enc  # noqa: E402
from aurora import n_params  # noqa: E402
from kdata import traj_dir  # noqa: E402

emb = importlib.import_module("aurora.embed")
VARS = ("u", "v", "p", "omega")
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
args = ap.parse_args()
D, P, B, DT = 64, 4, 8, 0.2
cfg = bb.Config(D, (2, 2, 2), (2, 4, 8), (2, 2, 2), (8, 4, 2), (1, 4, 4))

f = np.load(os.path.join(traj_dir(os.path.join(ROOT, "data", "stage2", "acc_v1"), "re100_kf4", 0), "frames_64.npy"))
f = (f - f.mean(axis=(0, 2, 3), keepdims=True)) / f.std(axis=(0, 2, 3), keepdims=True)
X = jnp.asarray(f, jnp.float32)
N = X.shape[-1]
idx = np.arange(1, X.shape[0] - 1)  # triples (t-1, t, t+1)
res = (1, N // P, N // P)

k = jax.random.split(jax.random.PRNGKey(0), 4)
params = {"emb": emb.init(k[0], D, P, VARS, dtype=jnp.float32),
          "bb": bb.init(k[1], cfg, dtype=jnp.float32),
          "frame": 0.02 * jax.random.normal(k[2], (2, D), jnp.float32),
          "out": {"W": 0.02 * jax.random.normal(k[3], (2 * D, D), jnp.float32), "b": jnp.zeros(D, jnp.float32)}}
t_emb = enc.lead_time(jnp.float32(DT), D).astype(jnp.float32)


def predict(prm, x_prev, x_cur):
    tok = (emb.embed(prm["emb"], x_prev, VARS, P, bands=False) + prm["frame"][0]
           + emb.embed(prm["emb"], x_cur, VARS, P, bands=False) + prm["frame"][1])  # (1, T, D)
    y = bb.apply(prm["bb"], tok, t_emb[None], res, cfg)  # (1, T, 2D)
    y = y @ prm["out"]["W"] + prm["out"]["b"]
    return x_cur + emb.unembed(prm["emb"], y, VARS, P, N)


batched = jax.vmap(predict, in_axes=(None, 0, 0))
sched = optax.cosine_decay_schedule(1e-3, args.steps, alpha=1e-2)
opt = optax.adam(sched)
state = opt.init(params)


@jax.jit
def step(prm, st, i):
    loss, g = jax.value_and_grad(lambda p: jnp.mean((batched(p, X[i - 1], X[i]) - X[i + 1]) ** 2))(prm)
    upd, st = opt.update(g, st, prm)
    return optax.apply_updates(prm, upd), st, loss


@jax.jit
def sq(prm, i):
    return jnp.sum((batched(prm, X[i - 1], X[i]) - X[i + 1]) ** 2)


def rel_err(prm):
    den = float(jnp.sum(X[idx + 1] ** 2))
    return float(np.sqrt(sum(float(sq(prm, idx[j:j + 16])) for j in range(0, len(idx), 16)) / den))


persistence = float(jnp.sqrt(jnp.sum((X[idx] - X[idx + 1]) ** 2) / jnp.sum(X[idx + 1] ** 2)))
print(f"params {n_params(params)}, persistence rel err {persistence:.4e}", flush=True)
rng = np.random.default_rng(0)
t0 = time.time()
curve = []
for s in range(args.steps):
    params, state, loss = step(params, state, jnp.asarray(rng.choice(idx, B)))
    if (s + 1) % 250 == 0:
        curve.append(dict(step=s + 1, loss=float(loss), train_rel_err=rel_err(params), wall_s=time.time() - t0))
        print(curve[-1], flush=True)
final = rel_err(params)
out = dict(D=D, P=P, batch=B, steps=args.steps, config=cfg._asdict(), n_params=int(n_params(params)),
           persistence_rel_err=persistence, train_rel_err=final, ratio_to_persistence=final / persistence,
           curve=curve, wall_s=time.time() - t0,
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "overfit.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(f"trained {final:.4e} vs persistence {persistence:.4e} (ratio {final / persistence:.3f})")
