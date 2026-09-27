"""E1 generation grid per regime (handouts/e1_volume_diversity.md, section 2).

For each regime: spin up seed 0 for T_spin on the spin-up grid, truncate the state to the smallest
candidate grid (common IC), then advance N and 2N by 1 time unit with the 2N grid's dt and compare
in the 64² output band: F = ||B(w_N) - B(w_2N)|| / ||B(w_2N)||. N_gen is the smallest candidate
with F <= 1e-3 (the Stage 2 rule). Larger candidates are only computed when needed. Resumable:
finished regimes are kept in the output JSON.

usage: measure_ngen.py --out experiments/e1/results/ngen.json [--regimes 300:4 700:6] [--spin-n 256]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from e1_design import all_regimes  # noqa: E402
from kolmogorov import Params, choose_m, make_advance, make_grid, random_ic, resample  # noqa: E402
from kolmogorov.diagnostics import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "results", "ngen.json"))
ap.add_argument("--regimes", nargs="*", help="Re:kf (default: every E1 regime, target included)")
ap.add_argument("--candidates", type=int, nargs="+", default=[128, 256, 512])
ap.add_argument("--band", type=int, default=64)
ap.add_argument("--tol", type=float, default=1e-3)
ap.add_argument("--spin-n", type=int, default=256)
ap.add_argument("--t-spin", type=float, default=50.0)
args = ap.parse_args()

regs = [tuple(int(x) for x in r.split(":")) for r in args.regimes] if args.regimes else all_regimes()
res = json.load(open(args.out)) if os.path.exists(args.out) else {"rows": {}}
res.update(candidates=args.candidates, band=args.band, tol=args.tol, spin_n=args.spin_n, t_spin=args.t_spin)
gB = make_grid(args.band)


def advance(ic, n, prm, dt, T):
    g = make_grid(n)
    return make_advance(g, prm, dt, round(T / dt))(resample(ic, n))


for Re, kf in regs:
    key = f"{Re}:{kf}"
    if key in res["rows"]:
        continue
    t0 = time.time()
    prm = Params(nu=1.0 / Re, kf=kf)
    gs = make_grid(args.spin_n)
    dts = 0.1 / choose_m(args.spin_n, prm)
    w = make_advance(gs, prm, dts, round(args.t_spin / dts))(random_ic(gs, seed=0))
    ic = make_grid(args.candidates[0]).mask * resample(w, args.candidates[0])
    pairs, n_gen = [], None
    for n in args.candidates:
        dt = 0.1 / choose_m(2 * n, prm)
        a, b = advance(ic, n, prm, dt, 1.0), advance(ic, 2 * n, prm, dt, 1.0)
        ref = resample(b, args.band)
        F = float(norm(gB, resample(a, args.band) - ref) / norm(gB, ref))
        pairs.append(dict(N=n, vs=2 * n, dt=dt, F=F))
        print(f"Re {Re} kf {kf}: N {n} vs {2 * n}: F{args.band} = {F:.2e}", flush=True)
        if F <= args.tol:
            n_gen = n
            break
    res["rows"][key] = dict(Re=Re, kf=kf, N_gen=n_gen, pairs=pairs, wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"Re {Re} kf {kf}: N_gen = {n_gen} ({time.time() - t0:.0f}s)", flush=True)
