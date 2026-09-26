"""R2/R3 (handouts/stage03_embed.md, section 4): learned embed -> unembed autoencoders, float32.

usage: train_ae.py --case R2      (D=128, bands on, encodings on, 3000 steps)
       train_ae.py --case R3      (D=32, one level, encodings off, 5000 steps, vs PCA optimum)
"""
import argparse
import json
import os
import sys
import time
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from aurora import VARS, embed, init, n_params, patchify, unembed  # noqa: E402
from kdata import traj_dir  # noqa: E402

CASES = {"R2": dict(D=128, bands=True, encodings=True, steps=3000),
         "R3": dict(D=32, bands=False, encodings=False, steps=5000)}
ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=CASES, required=True)
ap.add_argument("--dataset", default="acc_v1")
ap.add_argument("--tag", default="re100_kf4")
ap.add_argument("--steps", type=int, help="override the pre-registered step count (diagnostic runs only)")
args = ap.parse_args()
c = dict(CASES[args.case])
if args.steps:
    c["steps"] = args.steps
name = args.case + (f"_steps{args.steps}" if args.steps else "")
P, B, LR = 4, 16, 1e-3

root = os.path.join(ROOT, "data", "stage2", args.dataset)
load = lambda s: np.load(os.path.join(traj_dir(root, args.tag, s), "frames_64.npy"))
tr, ev = load(0), load(1)
mean = tr.mean(axis=(0, 2, 3), keepdims=True)
std = tr.std(axis=(0, 2, 3), keepdims=True)
tr = jnp.asarray((tr - mean) / std, jnp.float32)
ev = jnp.asarray((ev - mean) / std, jnp.float32)
N = tr.shape[-1]

params = init(jax.random.PRNGKey(0), c["D"], P, VARS, dtype=jnp.float32)
fwd = partial(embed, variables=VARS, P=P, bands=c["bands"], encodings=c["encodings"])
rec = lambda prm, x: unembed(prm, fwd(prm, x), VARS, P, N)
batched = jax.vmap(rec, in_axes=(None, 0))
opt = optax.adam(LR)
state = opt.init(params)


@jax.jit
def step(prm, st, x):
    loss, g = jax.value_and_grad(lambda p: jnp.mean((batched(p, x) - x) ** 2))(prm)
    upd, st = opt.update(g, st, prm)
    return optax.apply_updates(prm, upd), st, loss


@jax.jit
def rel_err(prm, x):
    return jnp.sqrt(jnp.sum((batched(prm, x) - x) ** 2) / jnp.sum(x**2))


rng = np.random.default_rng(0)
t0 = time.time()
curve = []
for i in range(c["steps"]):
    params, state, loss = step(params, state, tr[rng.integers(0, tr.shape[0], B)])
    if (i + 1) % max(250, c["steps"] // 40) == 0:
        e = float(np.mean([rel_err(params, ev[j:j + 50]) for j in range(0, ev.shape[0] - 50, 50)]))
        curve.append(dict(step=i + 1, loss=float(loss), eval_rel_err=e))
        print(curve[-1], flush=True)
err_eval = float(jnp.sqrt(sum(float(jnp.sum((batched(params, ev[j:j + 50]) - ev[j:j + 50]) ** 2))
                              for j in range(0, ev.shape[0], 50)) / float(jnp.sum(ev**2))))
res = dict(case=args.case, **c, P=P, batch=B, lr=LR, n_params=int(n_params(params)),
           eval_rel_err=err_eval, curve=curve, wall_s=time.time() - t0)

if args.case == "R3":
    # Eckart-Young optimum for an affine rank-D map on the concatenated (V*P*P) patch vectors:
    # fit mean and top-D principal directions on the training patches, score on the eval patches.
    to_vec = lambda x: np.asarray(jnp.concatenate([patchify(x[:, i], P) for i in range(4)], -1)).reshape(-1, 4 * P * P)
    Xtr, Xev = to_vec(tr), to_vec(ev)
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    Vd = Vt[: c["D"]].T
    Xp = (Xev - mu) @ Vd @ Vd.T + mu
    pca = float(np.sqrt(((Xp - Xev) ** 2).sum() / (Xev**2).sum()))
    res.update(pca_rel_err=pca, ratio=err_eval / pca)
    print(f"PCA optimum {pca:.4e}, trained {err_eval:.4e}, ratio {err_eval / pca:.3f}")

res["command"] = " ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"{args.case}: eval relative error {err_eval:.4e}, params {res['n_params']}")
