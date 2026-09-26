"""Resolution map (handout 5.3.2).

Common IC = the source run's t=50 state truncated to the smallest grid's retained set; the same
dt (finest grid's) on every grid; T=2. Grid N is resolved iff ||up(w_N) - w_2N|| / ||w_2N|| <= 1e-3
at t=1, compared on the 2N grid.

Stage 2 (M1) adds --bands: F_out(N) = ||B(w_N) - B(w_2N)|| / ||B(w_2N)|| with B the truncation to
|kx|, |ky| < N_out/2, at t=1 and t=2 (handouts/stage02_dataset.md, section 2.3).

usage: run_resolution.py --src L1000 --grids 128 256 512 [--bands 64 128 --out path.json]
"""
import argparse
import os

from common import DATA, command_line, setup_precision, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--grids", type=int, nargs="+", required=True)
ap.add_argument("--bands", type=int, nargs="*", default=[])
ap.add_argument("--out", help="write the JSON here instead of results/resolution_Re<Re>.json")
args = ap.parse_args()
setup_precision()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kolmogorov import Params, choose_m, make_advance, make_grid, resample  # noqa: E402
from kolmogorov.diagnostics import norm, shell_spectrum  # noqa: E402

d = os.path.join(DATA, args.src)
meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
prm = Params(nu=meta["nu"], alpha=meta["alpha"], A=meta["A"], kf=meta["kf"])
grids = sorted(args.grids)
dt = 0.1 / choose_m(grids[-1], prm)
n1 = round(1.0 / dt)

src = jnp.asarray(np.load(os.path.join(d, "state_t50.npy")))
g_min = make_grid(grids[0])
ic_min = g_min.mask * resample(src, grids[0])

states = {}
for n in grids:
    g = make_grid(n)
    w = resample(ic_min, n)
    adv = make_advance(g, prm, dt, n1)
    w1 = adv(w)
    w2 = adv(w1)
    states[n] = (g, w1, w2)
    print("N", n, "done", flush=True)

rows = []
for a, b in zip(grids[:-1], grids[1:]):
    gb = states[b][0]
    rel = [float(norm(gb, resample(states[a][i], b) - states[b][i]) / norm(gb, states[b][i])) for i in (1, 2)]
    # supplementary (not the pre-registered criterion): compare only the modes grid a retains,
    # and the norm share of w_2N outside them (a lower bound on rel_t1)
    ga = states[a][0]
    down = [ga.mask * resample(states[b][i], a) for i in (1, 2)]
    rel_c = [float(norm(ga, states[a][i] - down[i - 1]) / norm(ga, down[i - 1])) for i in (1, 2)]
    tail = float(jnp.sqrt(1 - (norm(ga, down[0]) / norm(gb, states[b][1])) ** 2))
    row = dict(N=a, vs=b, rel_t1=rel[0], rel_t2=rel[1], resolved=rel[0] <= 1e-3,
               rel_common_t1=rel_c[0], rel_common_t2=rel_c[1], tail_share_t1=tail)
    for nb in args.bands:
        gB = make_grid(nb)
        F = []
        for i in (1, 2):
            ref = resample(states[b][i], nb)
            F.append(float(norm(gB, resample(states[a][i], nb) - ref) / norm(gB, ref)))
        row[f"F{nb}_t1"], row[f"F{nb}_t2"] = F
    rows.append(row)
spectra = {str(n): np.asarray(shell_spectrum(states[n][0], states[n][1])).tolist() for n in grids}
res = dict(src=args.src, Re=meta["Re"], grids=grids, dt=dt, ic_truncation=grids[0], rows=rows,
           spectra_t1=spectra, command=command_line())
if args.out:
    import json

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)
else:
    write_json(f"resolution_Re{int(meta['Re'])}.json", res)
for r in rows:
    print(r)
