"""A1-A4 (handouts/stage10_lora.md, section 3): adapt the Stage 7 model to the unseen Re = 1000.

usage: finetune.py --arm zeroshot|lora|full|scratch [--steps 1000] [--rank r] [--out name]
       [--dataset acc_v1 --target re1000_kf4 --seed 0]   (F2: other unseen regimes, other seeds; the defaults are
                                                          Stage 10's and give its runs unchanged)
       [--train-seeds 0 1] [--eval-every 250]             (F3: more fine-tuning trajectories, longer runs)
--arm scratch trains every parameter from a random initialisation (key --seed) at lr 1e-3, the Stage 7 recipe,
with the base model's normalisation, so its error is measured exactly as the other arms'.
(results: finetune_<arm>[_lr..][_r..].json, or <name>.json)
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

from aurora import lora, model, n_params  # noqa: E402
from kdata import traj_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["zeroshot", "lora", "full", "scratch"], required=True)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--lr", type=float, help="override the pre-registered learning rate (exploratory runs only)")
ap.add_argument("--rank", type=int, help="LoRA rank (default: the pre-registered 4; follow-up F1 sweeps it)")
ap.add_argument("--out", help="results file name (default: finetune_<arm>[_lr..][_r..]); follow-ups must not overwrite Stage 10 files")
ap.add_argument("--dataset", default="acc_v1", help="dataset under data/stage2 holding the target regime")
ap.add_argument("--target", default="re1000_kf4", help="the unseen regime: fine-tune on --train-seeds, evaluate on seeds 6-7")
ap.add_argument("--seed", type=int, default=0, help="key of the LoRA or scratch initialisation and seed of the batch order")
ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1], help="the target's trajectories to fine-tune on")
ap.add_argument("--eval-every", type=int, default=250, help="steps between evaluations on the curve")
args = ap.parse_args()
MC, B, DT, R, TAG = model.ModelConfig(), 8, 0.2, 4, args.target
VARS = model.VARS
z = np.load(os.path.join(ROOT, "data", "stage7", "onestep_mse_params.npz"))
_, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32))
base = unravel(jnp.asarray(z["flat"]))
mu, sd = z["mean"][:, None, None], z["std"][:, None, None]
acc = os.path.join(ROOT, "data", "stage2", args.dataset)
load = lambda s: jnp.asarray(((np.load(os.path.join(traj_dir(acc, TAG, s), "frames_64.npy")) - mu) / sd).astype(np.float32))
train = jnp.stack([load(s) for s in args.train_seeds])
evals = jnp.stack([load(s) for s in (6, 7)])
F = train.shape[1]
fwd = jax.vmap(partial(model.apply, variables=VARS, dt=DT, mc=MC), in_axes=(None, 0, 0))

if args.arm == "lora":
    R = args.rank or R
    trainable = lora.init(jax.random.PRNGKey(args.seed), base, r=R)
    full = lambda t: lora.merge(base, t, r=R)
    lr = 1e-3
elif args.arm == "scratch":
    trainable = model.init(jax.random.PRNGKey(args.seed), MC, dtype=jnp.float32)
    full = lambda t: t
    lr = 1e-3
else:
    trainable = base
    full = lambda t: t
    lr = 1e-4
if args.lr:
    lr = args.lr
opt = optax.adam(optax.cosine_decay_schedule(lr, max(args.steps, 1), alpha=1e-2))
state = opt.init(trainable)


@jax.jit
def step(t, st, xm, x0, xp):
    loss, g = jax.value_and_grad(lambda q: jnp.mean((fwd(full(q), xm, x0) - xp) ** 2))(t)
    upd, st = opt.update(g, st, t)
    return optax.apply_updates(t, upd), st, loss


@jax.jit
def sq(t, xm, x0, xp):
    return jnp.sum((fwd(full(t), xm, x0) - xp) ** 2), jnp.sum((x0 - xp) ** 2), jnp.sum(xp**2)


def evaluate(t):
    a = b = c = 0.0
    for s in range(evals.shape[0]):
        for j in range(1, F - 1, 16):
            k = np.arange(j, min(j + 16, F - 1))
            e, pe, n = sq(t, evals[s, k - 1], evals[s, k], evals[s, k + 1])
            a, b, c = a + float(e), b + float(pe), c + float(n)
    return dict(rel_err=float(np.sqrt(a / c)), persistence=float(np.sqrt(b / c)), ratio=float(np.sqrt(a / b)))


rng = np.random.default_rng(args.seed)
t0 = time.time()
curve = []
steps = 0 if args.arm == "zeroshot" else args.steps
for i in range(steps):
    s, k = rng.integers(0, len(args.train_seeds), B), rng.integers(1, F - 1, B)
    trainable, state, loss = step(trainable, state, train[s, k - 1], train[s, k], train[s, k + 1])
    if (i + 1) % args.eval_every == 0:
        curve.append(dict(step=i + 1, loss=float(loss), eval=evaluate(trainable)))
        print(curve[-1], flush=True)
ev = evaluate(trainable)
n_tr = lora.n_params(trainable) if args.arm == "lora" else (0 if args.arm == "zeroshot" else n_params(trainable))
res = dict(arm=args.arm, dataset=args.dataset, target=TAG, seed=args.seed, train_seeds=args.train_seeds, steps=steps, lr=lr,
           r=R if args.arm == "lora" else None, n_trainable=int(n_tr),
           n_total=int(n_params(base)), trainable_frac=n_tr / n_params(base), eval=ev, curve=curve,
           wall_s=time.time() - t0,
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
suffix = (f"_lr{args.lr:g}" if args.lr else "") + (f"_r{args.rank}" if args.rank else "")
name = args.out or f"finetune_{args.arm}{suffix}"
with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(args.arm, ev)
