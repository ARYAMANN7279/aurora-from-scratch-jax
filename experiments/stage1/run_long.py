"""Long run (handout 5.2): chunked, resumable, writes raw outputs under data/stage1/<name>/.

usage: run_long.py --name L1000 --re 1000 --n 256 [--t 250] [--kf 4] [--f32]
"""
import argparse
import glob
import os
import time

from common import DATA, command_line, setup_precision

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--re", type=float, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--t", type=float, default=250.0)
ap.add_argument("--kf", type=int, default=4)
ap.add_argument("--chunk", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--f32", action="store_true")
ap.add_argument("--ic", help="initial state .npy (default: random_ic with --seed)")
args = ap.parse_args()
setup_precision(args.f32)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kolmogorov import Params, choose_m, make_chunk, make_grid, random_ic  # noqa: E402

out = os.path.join(DATA, args.name)
os.makedirs(out, exist_ok=True)
prm = Params(nu=1.0 / args.re, kf=args.kf)
g = make_grid(args.n)
m = choose_m(args.n, prm)
dt = 0.1 / m
n_saves = round(args.chunk / 0.1)
n_chunks = round(args.t / args.chunk)
meta = dict(name=args.name, Re=args.re, N=args.n, nu=prm.nu, alpha=prm.alpha, A=prm.A, kf=prm.kf,
            T=args.t, ic=args.ic, m=m, dt=dt, seed=args.seed, f32=args.f32, dtype=str(g.k2.dtype),
            command=command_line())
np.save(os.path.join(out, "meta.npy"), meta, allow_pickle=True)
print(meta, flush=True)

done = sorted(glob.glob(os.path.join(out, "chunk_*.npz")))
if done:
    wh = jnp.asarray(np.load(os.path.join(out, "state_last.npy")))
    start = len(done)
    print(f"resuming after chunk {start}", flush=True)
else:
    ic = np.load(args.ic) if args.ic else np.asarray(random_ic(g, seed=args.seed))
    wh = jnp.asarray(ic, dtype=jnp.complex64 if args.f32 else jnp.complex128)
    np.save(os.path.join(out, "state_t0.npy"), np.asarray(wh))
    start = 0

chunk = make_chunk(g, prm, m, n_saves)
for c in range(start, n_chunks):
    t0 = time.time()
    wh, o = chunk(wh)
    wh.block_until_ready()
    wall = time.time() - t0
    o = {k: np.asarray(v) for k, v in o.items()}
    t_end = (c + 1) * args.chunk
    np.savez(os.path.join(out, f"chunk_{c:04d}.npz"), wall=wall, t=t_end - args.chunk + 0.1 * (1 + np.arange(n_saves)), **o)
    np.save(os.path.join(out, "state_last.npy"), np.asarray(wh))
    if abs(t_end - 50.0) < 1e-9:
        np.save(os.path.join(out, "state_t50.npy"), np.asarray(wh))
    print(f"t={t_end:6.1f} wall={wall:7.1f}s E={o['E'][-1]:.4f} Z={o['Z'][-1]:.3f} "
          f"maxCFL={o['cfl'].max():.3f} steps/s={m * n_saves / wall:.0f}", flush=True)
    if not np.all(np.isfinite(o["E"])):
        raise SystemExit("non-finite energy")
print("done", flush=True)
