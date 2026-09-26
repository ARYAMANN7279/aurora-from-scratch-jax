"""Temporal order (handout 5.3.1): from a run's t=50 state, T=1 at dt/2^j, j=0..3, ref j=5.

usage: run_order.py --name L1000
"""
import argparse
import math
import os

from common import DATA, command_line, setup_precision, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--T", type=float, default=1.0)
args = ap.parse_args()
setup_precision()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kolmogorov import Params, make_advance, make_grid  # noqa: E402
from kolmogorov.diagnostics import norm  # noqa: E402

d = os.path.join(DATA, args.name)
meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
g = make_grid(meta["N"])
prm = Params(nu=meta["nu"], alpha=meta["alpha"], A=meta["A"], kf=meta["kf"])
w0 = jnp.asarray(np.load(os.path.join(d, "state_t50.npy")))
n0 = round(args.T / meta["dt"])

final = {}
for j in (0, 1, 2, 3, 5):
    final[j] = make_advance(g, prm, meta["dt"] / 2**j, n0 * 2**j)(w0)
    print("j", j, "done", flush=True)
ref = final[5]
errs = {j: float(norm(g, final[j] - ref) / norm(g, ref)) for j in (0, 1, 2, 3)}
orders, roundoff = [], []
for j in (0, 1, 2):
    if errs[j + 1] < 1e-12:
        roundoff.append(j)
        continue
    orders.append(math.log2(errs[j] / errs[j + 1]))
res = dict(name=args.name, dt=meta["dt"], T=args.T, errors=errs, orders=orders,
           roundoff_limited_pairs=roundoff, command=command_line())
write_json(f"order_{args.name}.json", res)
print(res)
