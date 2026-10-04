"""Stage 10 F2: is each new regime's generation grid fine enough? Stage 2's rule (M1, handouts/stage02_dataset.md,
section 2.3) on the trajectory's own post-spin-up state: from that state, truncated to the smallest grid, every grid
in --grids runs 2 time units with the finest grid's time step, and

    F_64(N) = ||B(w_N) - B(w_2N)|| / ||B(w_2N)||   at t = 1 and t = 2,

with B the truncation to |kx|, |ky| < 32 (the 64² band). N is resolved iff F_64(N) <= 1e-3 at t = 1.
usage: resolution_f2.py --tag re3000_kf4 --grids 512 1024 2048   (state from seed 0 of data/stage2/s10f2_v1;
                                                                   writes results/f2_resolution_<tag>.json)
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
from kolmogorov import Params, choose_m, make_advance, make_grid, resample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--grids", type=int, nargs="+", required=True)
args = ap.parse_args()
d = traj_dir(os.path.join(ROOT, "data", "stage2", "s10f2_v1"), args.tag, 0)
meta = json.load(open(os.path.join(d, "meta.json")))
Re, kf = meta["Re"], meta["kf"]
prm = Params(nu=meta["nu"], alpha=meta["alpha"], A=meta["A"], kf=kf)  # the generator's own parameters
grids = sorted(args.grids)
dt = 0.1 / choose_m(grids[-1], prm)
n1 = round(1.0 / dt)
src = jnp.asarray(np.load(os.path.join(d, "state_t0.npy")))
g0 = make_grid(grids[0])
ic = g0.mask * resample(src, grids[0])
band = {}
for N in grids:
    g = make_grid(N)
    adv = make_advance(g, prm, dt, n1)
    w1 = adv(resample(ic, N))
    w2 = adv(w1)
    band[N] = [np.asarray(resample(w, 64)) for w in (w1, w2)]
    print(f"N={N} done", flush=True)
out = dict(tag=args.tag, Re=Re, kf=kf, dt=dt, rule="F_64(N) <= 1e-3 at t = 1", F_64={})
for a, b in zip(grids, grids[1:]):
    if b == 2 * a:
        f = [float(np.linalg.norm(band[a][i] - band[b][i]) / np.linalg.norm(band[b][i])) for i in (0, 1)]
        out["F_64"][f"{a} vs {b}"] = dict(t1=f[0], t2=f[1], resolved=f[0] <= 1e-3)
        print(f"{a} vs {b}: F_64 t=1 {f[0]:.2e}, t=2 {f[1]:.2e}", flush=True)
with open(os.path.join(HERE, "results", f"f2_resolution_{args.tag}.json"), "w") as f:
    json.dump(out, f, indent=1)
