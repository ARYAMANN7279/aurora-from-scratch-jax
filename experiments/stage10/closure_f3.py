"""Stage 10 F3 (handouts/stage10_lora.md, F3): the floor. How much of one frame's change in the 64² band depends on the
scales finer than the band, which a model with 64² inputs never sees?

For each regime, the exact solver runs on the generation grid from the post-spin-up state of evaluation seeds 6 and 7
(state_t0.npy). Every 5 frames (1 time unit), 20 samples per seed, it takes the state w and advances one frame (0.2):
  the true state                 S(w)
  the band of the state          S(P w),  P keeping only |kx|, |ky| < 32 (what resample(., 64) keeps)
Both are saved as 64² frames F (u, v, p, omega), normalised with the base model's mean and std as in finetune.py, and
  ratio = ||F(S(P w)) - F(S(w))|| / ||F(w) - F(S(w))||    (sums over all samples; the denominator is persistence)
is the error over persistence of a predictor that knows the equations exactly but sees only the band of the current
state. A learned model also sees the previous frame, so this is a reference, not a strict bound.
usage: closure_f3.py   (writes results/f3_closure.json)
       [--only re100_kf4 ...] [--samples 20] [--out name]   (smoke tests)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kdata import traj_dir  # noqa: E402
from kdata.generate import make_frame_fn  # noqa: E402
from kolmogorov import Params, choose_m, make_grid, make_step, resample  # noqa: E402
from kolmogorov.solver import SAVE_DT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", help="regime tags to measure (default: all four)")
ap.add_argument("--samples", type=int, default=20, help="samples per seed")
ap.add_argument("--out", default="f3_closure")
args = ap.parse_args()
REGIMES = [("acc_v1", "re100_kf4"), ("acc_v1", "re300_kf4"), ("acc_v1", "re1000_kf4"), ("s10f2_v1", "re3000_kf4")]
REGIMES = [(ds, t) for ds, t in REGIMES if not args.only or t in args.only]
SAMPLES, GAP = args.samples, 5  # samples per seed, frames between samples
sd = np.load(os.path.join(ROOT, "data", "stage7", "onestep_mse_params.npz"))["std"][:, None, None].astype(np.float64)
out = {}
for ds, tag in REGIMES:
    A, P, per_seed, per_sample = np.zeros(4), np.zeros(4), {}, []
    for seed in (6, 7):
        d = traj_dir(os.path.join(ROOT, "data", "stage2", ds), tag, seed)
        meta = json.load(open(os.path.join(d, "meta.json")))
        prm = Params(nu=1.0 / meta["Re"], kf=meta["kf"])  # as the generator builds them
        N = meta["N_gen"]
        g, m = make_grid(N), choose_m(N, prm)
        dt = SAVE_DT / m
        assert abs(dt - meta["dt"]) < 1e-15, (dt, meta["dt"])
        step = make_step(g, prm, dt)
        adv = jax.jit(lambda w: jax.lax.scan(lambda x, _: (step(x), None), w, None, length=round(0.2 / SAVE_DT) * m)[0])
        frame = jax.jit(lambda w: make_frame_fn(g, prm, dt, (64,))(w)[0][0])
        band = jax.jit(lambda w: resample(resample(w, 64), N))
        w = jnp.asarray(np.load(os.path.join(d, "state_t0.npy")))
        a_s, p_s = np.zeros(4), np.zeros(4)
        for i in range(SAMPLES):
            w1 = adv(w)
            f0, f1, fc = (np.asarray(frame(x), np.float64) / sd for x in (w, w1, adv(band(w))))
            a, p = ((fc - f1) ** 2).sum(axis=(1, 2)), ((f0 - f1) ** 2).sum(axis=(1, 2))
            a_s, p_s = a_s + a, p_s + p
            per_sample.append(float(np.sqrt(a.sum() / p.sum())))
            for _ in range(GAP - 1):
                w1 = adv(w1)
            w = w1
        A, P = A + a_s, P + p_s
        per_seed[seed] = float(np.sqrt(a_s.sum() / p_s.sum()))
        print(tag, seed, round(per_seed[seed], 4), flush=True)
    out[tag] = dict(dataset=ds, N_gen=N, ratio=float(np.sqrt(A.sum() / P.sum())),
                    by_channel={c: float(np.sqrt(A[i] / P[i])) for i, c in enumerate(("u", "v", "p", "omega"))},
                    by_seed=per_seed, per_sample_quartiles=[float(q) for q in np.percentile(per_sample, [25, 50, 75])],
                    samples=len(per_sample))
    print(tag, {k: v for k, v in out[tag].items() if k in ("ratio", "by_channel")}, flush=True)
with open(os.path.join(HERE, "results", f"{args.out}.json"), "w") as f:
    json.dump(out, f, indent=1)
