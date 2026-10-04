"""Stage 10 F2, exploratory (not a pre-registered prediction): what kind of error does the base make in each regime?

The F1 base model's one-step prediction x^ from (x_-1, x_0), on each regime's evaluation seeds 6-7 (every frame, as in
finetune.py), split into the size and the direction of its predicted change d = x^ - x_0 against the true change
d* = x_1 - x_0 (normalised variables, all four channels, sums over every frame):
  amp    ||d|| / ||d*||                     (1 = the field changes by the right amount)
  corr   <d, d*> / (||d|| ||d*||)           (1 = it changes in the right direction)
  ratio  ||x^ - x_1|| / ||d*||, error over persistence; exactly ratio^2 = 1 + amp^2 - 2 amp corr
  ratio_scaled = sqrt(1 - corr^2), the ratio after the best single factor on d (amp set to corr)
A wrong amount of change is a systematic error that one number fixes; a wrong direction is not.
usage: diagnose_f2.py   (writes results/f2_diagnosis.json)
"""
import json
import os
import sys
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aurora import model  # noqa: E402
from kdata import traj_dir  # noqa: E402

MC = model.ModelConfig()
z = np.load(os.path.join(ROOT, "data", "stage7", "onestep_mse_params.npz"))
_, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32))
base = unravel(jnp.asarray(z["flat"]))
mu, sd = z["mean"][:, None, None], z["std"][:, None, None]
fwd = jax.jit(jax.vmap(partial(model.apply, variables=model.VARS, dt=0.2, mc=MC), in_axes=(None, 0, 0)))
REGIMES = [("s10f2_v1", "re30_kf4"), ("acc_v1", "re100_kf4"), ("acc_v1", "re300_kf4"), ("acc_v1", "re1000_kf4"),
           ("s10f2_v1", "re3000_kf4")]
out = {}
for ds, tag in REGIMES:
    S = dict(A=np.zeros(4), P=np.zeros(4), D=np.zeros(4), C=np.zeros(4))
    for s in (6, 7):
        x = ((np.load(os.path.join(traj_dir(os.path.join(ROOT, "data", "stage2", ds), tag, s), "frames_64.npy")) - mu)
             / sd).astype(np.float32)
        for j in range(1, x.shape[0] - 1, 16):
            k = np.arange(j, min(j + 16, x.shape[0] - 1))
            xm, x0, xp = x[k - 1], x[k], x[k + 1]
            d, dt_ = np.asarray(fwd(base, jnp.asarray(xm), jnp.asarray(x0)), np.float64) - x0, (xp - x0).astype(np.float64)
            for key, v in (("A", (d - dt_) ** 2), ("P", dt_**2), ("D", d**2), ("C", d * dt_)):
                S[key] += v.sum(axis=(0, 2, 3))
    row = {}
    for name, sel in [("all", slice(None))] + [(c, i) for i, c in enumerate(("u", "v", "p", "omega"))]:
        A, P, D, C = (float(np.sum(S[key][sel])) for key in ("A", "P", "D", "C"))
        corr = C / np.sqrt(D * P)
        row[name] = dict(ratio=float(np.sqrt(A / P)), amp=float(np.sqrt(D / P)), corr=float(corr),
                         ratio_scaled=float(np.sqrt(1 - corr**2)), best_factor=float(C / D))
    out[tag] = dict(dataset=ds, **row)
    print(tag, {k: round(v, 4) for k, v in row["all"].items()}, flush=True)
with open(os.path.join(HERE, "results", "f2_diagnosis.json"), "w") as f:
    json.dump(out, f, indent=1)
