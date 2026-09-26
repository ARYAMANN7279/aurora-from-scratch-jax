"""S1-S6 (handouts/stage04_perceiver.md, section 4): embed -> aggregate (C -> L) -> decode -> heads.

usage: train_perceiver.py --run main                  (L=3, all resolutions: S1-S4)
       train_perceiver.py --run nocompress --res 64   (L=C for one resolution: S5 control)
       train_perceiver.py --run nolevelenc --res 64   (level encodings ablated: S6 control)
       add --steps K --time-only to time K steps without writing results
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

import importlib  # noqa: E402

from aurora import n_levels, n_params, perceiver  # noqa: E402
from kdata import traj_dir  # noqa: E402

emb = importlib.import_module("aurora.embed")
VARS = ("u", "v", "p", "omega")
ap = argparse.ArgumentParser()
ap.add_argument("--run", choices=["main", "nocompress", "nolevelenc"], required=True)
ap.add_argument("--res", type=int, nargs="+", default=[16, 64, 128, 512])
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--time-only", action="store_true")
ap.add_argument("--tag", default="re300_kf4", help="regime for the 16/64/128 data (spec: re300_kf4)")
args = ap.parse_args()
D, H, L, P, B, LR = 128, 4, 3, 4, 4, 3e-4

# ---- data
acc = os.path.join(ROOT, "data", "stage2", "acc_v1")


def truncate(x, n):
    """Spectral truncation of (..., N, N) float32 frames to n x n."""
    xh = np.fft.rfft2(x, norm="forward")
    N, K = x.shape[-1], n // 2
    out = np.zeros(x.shape[:-2] + (n, n // 2 + 1), complex)
    out[..., :K, :K] = xh[..., :K, :K]
    out[..., n - (K - 1):, :K] = xh[..., N - (K - 1):, :K]
    return np.fft.irfft2(out, s=(n, n), norm="forward").astype(np.float32)


def load(n):
    if n == 512:
        f = np.load(os.path.join(ROOT, "data", "stage4", "f512", "re1000_kf4", "seed_0000", "frames_512.npy"))
        return f[:32], f[32:]
    src = 128 if n == 128 else 64
    get = lambda s: np.load(os.path.join(traj_dir(acc, args.tag, s), f"frames_{src}.npy"))
    tr = np.concatenate([get(s) for s in range(6)])
    ev = get(6)
    if n < src:
        tr, ev = truncate(tr, n), truncate(ev, n)
    return tr, ev


data = {}
for n in args.res:
    tr, ev = load(n)
    mu, sd = tr.mean(axis=(0, 2, 3), keepdims=True), tr.std(axis=(0, 2, 3), keepdims=True)
    data[n] = (jnp.asarray((tr - mu) / sd), jnp.asarray((ev - mu) / sd))
    print(f"N={n}: C={n_levels(n)}, train {tr.shape[0]}, eval {ev.shape[0]}", flush=True)

# ---- model
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
L_of = (lambda n: n_levels(n)) if args.run == "nocompress" else (lambda n: L)
if args.run == "nocompress" and len(args.res) != 1:
    raise SystemExit("nocompress runs one resolution at a time")
params = {"emb": emb.init(k1, D, P, VARS, dtype=jnp.float32),
          "per": perceiver.init(k2, D, L_of(args.res[0]), dtype=jnp.float32)}


def forward(prm, x):
    N = x.shape[-1]
    C = n_levels(N)
    tok = emb.embed(prm["emb"], x, VARS, P)
    if args.run == "nolevelenc":  # every level gets level 0's encoding
        _, _, lev = emb.encoding_terms(prm["emb"], N, P, C)
        tok = tok - lev[:, None] + lev[:1, None]
    z = perceiver.aggregate(prm["per"], tok, H)
    return emb.unembed(prm["emb"], perceiver.decode(prm["per"], z, C, H), VARS, P, N)


batched = jax.vmap(forward, in_axes=(None, 0))
opt = optax.adam(LR)
state = opt.init(params)


@jax.jit
def step(prm, st, x):
    loss, g = jax.value_and_grad(lambda p: jnp.mean((batched(p, x) - x) ** 2))(prm)
    upd, st = opt.update(g, st, prm)
    return optax.apply_updates(prm, upd), st, loss


@jax.jit
def sq_err(prm, x):
    return jnp.sum((batched(prm, x) - x) ** 2), jnp.sum(x**2)


def eval_err(prm, n):
    ev = data[n][1]
    num = den = 0.0
    for j in range(0, ev.shape[0], B):
        a, b = sq_err(prm, ev[j:j + B])
        num, den = num + float(a), den + float(b)
    return float(np.sqrt(num / den))


rng = np.random.default_rng(0)
t0 = time.time()
curve, times = [], {n: [] for n in args.res}
for i in range(args.steps):
    n = args.res[i % len(args.res)]
    tr = data[n][0]
    ts = time.perf_counter()
    params, state, loss = step(params, state, tr[rng.integers(0, tr.shape[0], B)])
    loss.block_until_ready()
    times[n].append(time.perf_counter() - ts)
    if not args.time_only and (i + 1) % 500 == 0:
        row = dict(step=i + 1, **{f"eval_{m}": eval_err(params, m) for m in args.res})
        curve.append(row)
        print(row, flush=True)
step_s = {n: float(np.median(t[1:])) if len(t) > 1 else None for n, t in times.items()}
print("median s/step:", step_s, flush=True)
if args.time_only:
    sys.exit()

res = dict(run=args.run, res=args.res, D=D, H=H, L=L if args.run != "nocompress" else n_levels(args.res[0]),
           P=P, batch=B, lr=LR, steps=args.steps, n_params=int(n_params(params)),
           eval_rel_err={str(n): eval_err(params, n) for n in args.res}, curve=curve, step_s=step_s,
           wall_s=time.time() - t0,
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
name = args.run + ("" if args.run == "main" else f"_{args.res[0]}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(res["eval_rel_err"])
