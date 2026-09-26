"""Lyapunov exponent (handout 5.3.3): 4 twins perturbed by 1e-8, from a run's t=50 state.

usage: run_lyapunov.py --name L1000 [--tmax 100]
"""
import argparse
import os

from common import DATA, command_line, setup_precision, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--tmax", type=float, default=100.0)
ap.add_argument("--eps", type=float, default=1e-8)
args = ap.parse_args()
setup_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kolmogorov import Params, make_grid, make_step, rfft2  # noqa: E402
from kolmogorov.diagnostics import norm  # noqa: E402

d = os.path.join(DATA, args.name)
meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
g = make_grid(meta["N"])
prm = Params(nu=meta["nu"], alpha=meta["alpha"], A=meta["A"], kf=meta["kf"])
m, dt = meta["m"], meta["dt"]
w0 = jnp.asarray(np.load(os.path.join(d, "state_t50.npy")))

twins = [w0]
for seed in (1, 2, 3, 4):
    p = g.mask * rfft2(jnp.asarray(np.random.default_rng(seed).standard_normal((g.N, g.N))))
    twins.append(w0 + p * (args.eps * norm(g, w0) / norm(g, p)))
W = jnp.stack(twins)

step = jax.vmap(make_step(g, prm, dt))
dist = jax.vmap(lambda a, b: norm(g, a - b) / norm(g, b), in_axes=(0, None))


@jax.jit
def chunk(W):
    def save(W, _):
        W = jax.lax.scan(lambda w, _: (step(w), None), W, None, length=m)[0]
        return W, dist(W[1:], W[0])
    return jax.lax.scan(save, W, None, length=100)


ds = [np.array([float(dist(W[1:], W[0])[i]) for i in range(4)])[None]]
t = 0.0
while t < args.tmax - 1e-9:
    W, dd = chunk(W)
    ds.append(np.asarray(dd))
    t += 10.0
    print(f"t={t:.0f} d={np.asarray(dd)[-1]}", flush=True)
    if np.all(np.asarray(dd)[-1] > 1e-2):
        break
D = np.concatenate(ds)
ts = 0.1 * np.arange(D.shape[0])

fits = []
for i in range(4):
    di = D[:, i]
    start = np.nonzero(di >= 1e-6)[0]
    if not start.size:
        fits.append(None)
        continue
    s = start[0]
    e = s
    while e + 1 < di.size and 1e-6 <= di[e + 1] <= 1e-2:
        e += 1
    sel = slice(s, e + 1)
    fits.append(dict(lam=float(np.polyfit(ts[sel], np.log(di[sel]), 1)[0]),
                     t_range=[float(ts[s]), float(ts[e])], npts=int(e - s + 1)))
lams = [f["lam"] for f in fits if f]
res = dict(name=args.name, Re=meta["Re"], N=meta["N"], eps=args.eps, fits=fits,
           lam_mean=float(np.mean(lams)) if lams else None,
           lam_spread=[float(min(lams)), float(max(lams))] if lams else None,
           t=ts[::5].tolist(), d=D[::5].tolist(), command=command_line())
write_json(f"lyapunov_{args.name}.json", res)
print({k: res[k] for k in ("lam_mean", "lam_spread", "fits")})
