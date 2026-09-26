"""N1-N4 (handouts/stage07_model.md, section 3): train the full model on one-step prediction.

usage: train_onestep.py [--steps 3000] [--loss mse|mae] [--pushforward] [--time-only]
--pushforward trains with the pushforward trick (handouts/stage09_pushforward.md): loss on the
one-step prediction plus the second step from stop_gradient(first prediction).
Also writes the trained parameters to data/stage7/<name>_params.npz (used by Stages 8-9).
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
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from aurora import model, n_params  # noqa: E402
from kdata import traj_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--time-only", action="store_true")
ap.add_argument("--loss", choices=["mse", "mae"], default="mse")
ap.add_argument("--pushforward", action="store_true")
args = ap.parse_args()
B, DT, TAGS = 8, 0.2, ("re100_kf4", "re300_kf4")
MC = model.ModelConfig()
VARS = model.VARS
acc = os.path.join(ROOT, "data", "stage2", "acc_v1")
load = lambda tag, s: np.load(os.path.join(traj_dir(acc, tag, s), "frames_64.npy"))
train = {t: np.stack([load(t, s) for s in range(6)]) for t in TAGS}          # (6, F, 4, 64, 64)
evals = {t: np.stack([load(t, s) for s in (6, 7)]) for t in TAGS}
allf = np.concatenate([train[t].reshape(-1, 4, 64, 64) for t in TAGS])
mu, sd = allf.mean(axis=(0, 2, 3)), allf.std(axis=(0, 2, 3))
norm = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
train = {t: norm(v) for t, v in train.items()}
evals = {t: norm(v) for t, v in evals.items()}
F = train[TAGS[0]].shape[1]

params = model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32)
fwd = jax.vmap(partial(model.apply, variables=VARS, dt=DT, mc=MC), in_axes=(None, 0, 0))
sched = optax.cosine_decay_schedule(1e-3, args.steps, alpha=1e-2)
opt = optax.adam(sched)
state = opt.init(params)
lossf = (lambda d: jnp.mean(d**2)) if args.loss == "mse" else (lambda d: jnp.mean(jnp.abs(d)))


def objective(p, xm, x0, xp, xp2):
    y1 = fwd(p, xm, x0)
    if not args.pushforward:
        return lossf(y1 - xp)
    y2 = fwd(p, x0, jax.lax.stop_gradient(y1))
    return lossf(y1 - xp) + lossf(y2 - xp2)


@jax.jit
def step(prm, st, xm, x0, xp, xp2):
    loss, g = jax.value_and_grad(objective)(prm, xm, x0, xp, xp2)
    upd, st = opt.update(g, st, prm)
    return optax.apply_updates(prm, upd), st, loss


@jax.jit
def sq(prm, xm, x0, xp):
    return jnp.sum((fwd(prm, xm, x0) - xp) ** 2), jnp.sum((x0 - xp) ** 2), jnp.sum(xp**2)


def evaluate(prm):
    out = {}
    for t, v in evals.items():
        a = b = c = 0.0
        for s in range(v.shape[0]):
            for j in range(1, F - 1, 16):
                k = np.arange(j, min(j + 16, F - 1))
                e, pe, n = sq(prm, v[s, k - 1], v[s, k], v[s, k + 1])
                a, b, c = a + float(e), b + float(pe), c + float(n)
        out[t] = dict(rel_err=float(np.sqrt(a / c)), persistence=float(np.sqrt(b / c)), ratio=float(np.sqrt(a / b)))
    return out


rng = np.random.default_rng(0)
t0 = time.time()
curve, times = [], []
for i in range(args.steps):
    t = TAGS[rng.integers(len(TAGS))]
    s = rng.integers(0, 6, B)
    k = rng.integers(1, F - 2 if args.pushforward else F - 1, B)  # same sampling as Stage 7 when not pushforward
    v = train[t]
    ts = time.perf_counter()
    params, state, loss = step(params, state, v[s, k - 1], v[s, k], v[s, k + 1], v[s, np.minimum(k + 2, F - 1)])
    loss.block_until_ready()
    times.append(time.perf_counter() - ts)
    if args.time_only and i == 4:
        print("median s/step:", float(np.median(times[1:])), "params", n_params(params))
        sys.exit()
    if (i + 1) % 250 == 0:
        curve.append(dict(step=i + 1, loss=float(loss), eval=evaluate(params), wall_s=time.time() - t0))
        print(curve[-1], flush=True)
ev = evaluate(params)
name = f"onestep_{args.loss}" + (f"_pf{args.steps}" if args.pushforward else "")
res = dict(config=str(MC), loss=args.loss, pushforward=args.pushforward, steps=args.steps, batch=B, n_params=int(n_params(params)),
           eval=ev, curve=curve, step_s=float(np.median(times[1:])), wall_s=time.time() - t0,
           norm_mean=mu.tolist(), norm_std=sd.tolist(),
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
    json.dump(res, f, indent=1)
os.makedirs(os.path.join(ROOT, "data", "stage7"), exist_ok=True)
flat, _ = jax.flatten_util.ravel_pytree(params)
np.savez(os.path.join(ROOT, "data", "stage7", f"{name}_params.npz"), flat=np.asarray(flat), mean=mu, std=sd)
print(ev)
