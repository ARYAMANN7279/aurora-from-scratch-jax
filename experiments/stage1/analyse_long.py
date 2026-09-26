"""Analyse a long run (handout 5.2) -> results/<name>.json.

usage: analyse_long.py --name L1000 [--t0 50] [--t1 250]
"""
import argparse
import glob
import os

import numpy as np

from common import DATA, command_line, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--t0", type=float, default=50.0)
ap.add_argument("--t1", type=float, default=250.0)
ap.add_argument("--nbatch", type=int, default=10)
ap.add_argument("--max-lag", type=float, default=30.0)
args = ap.parse_args()

d = os.path.join(DATA, args.name)
meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
files = sorted(glob.glob(os.path.join(d, "chunk_*.npz")))
parts = [dict(np.load(f)) for f in files]
cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "wall"}
wall = float(sum(float(p["wall"]) for p in parts))
t = cat["t"]
T_done = float(t[-1])
assert T_done >= args.t1 - 1e-9, f"run only reached t={T_done}"

nu, alpha, A, kf, N = meta["nu"], meta["alpha"], meta["A"], meta["kf"], meta["N"]
lam = nu * kf**2 + alpha
E_lam = A**2 / (4 * lam**2)
K = (N - 1) // 3
f64 = lambda k: cat[k].astype(np.float64)
E, Z, Pi, P, Q, IE, IZ = (f64(k) for k in ("E", "Z", "Pi", "P", "Q", "IE", "IZ"))

i0 = int(np.argmin(np.abs(t - args.t0)))          # save at t0
i1 = int(np.argmin(np.abs(t - args.t1)))          # save at t1
win = slice(i0 + 1, i1 + 1)                        # saves in (t0, t1]

# budgets: IE/IZ are per-step trapezoid integrals over each preceding 0.1 interval
R_E = E[i1] - E[i0] - IE[win].sum()
R_Z = Z[i1] - Z[i0] - IZ[win].sum()
intP = np.trapezoid(P[i0:i1 + 1], t[i0:i1 + 1])    # normalisers only
intQ = np.trapezoid(Q[i0:i1 + 1], t[i0:i1 + 1])


def mean_se(x):
    b = np.array_split(x, args.nbatch)
    bm = np.array([bb.mean() for bb in b])
    return float(x.mean()), float(bm.std(ddof=1) / np.sqrt(args.nbatch))


spec = cat["spec"][win].astype(np.float64).mean(axis=0)
ks = np.arange(spec.size)


def slope(klo, khi):
    sel = (ks >= klo) & (ks <= khi)
    return float(np.polyfit(np.log(ks[sel]), np.log(spec[sel]), 1)[0])


def batch_slopes(klo, khi):
    s = cat["spec"][win].astype(np.float64)
    out = []
    for b in np.array_split(np.arange(s.shape[0]), args.nbatch):
        sb = s[b].mean(axis=0)
        sel = (ks >= klo) & (ks <= khi)
        out.append(np.polyfit(np.log(ks[sel]), np.log(sb[sel]), 1)[0])
    return float(np.std(out, ddof=1) / np.sqrt(args.nbatch))


k_top = int(np.ceil(0.8 * K))
Em, Ese = mean_se(E[win])
Zm, Zse = mean_se(Z[win])
Pim = float(Pi[win].mean())
Pm = float(P[win].mean())
Qm = float(Q[win].mean())
k_d = (2 * nu * Pim) ** (1 / 6) / nu**0.5

# spin-up: first time E enters the [5, 95] percentile band of E over [150, t1]
late = (t > 150.0) & (t <= args.t1 + 1e-9)
lo, hi = np.percentile(E[late], [5, 95])
inside = np.nonzero((E >= lo) & (E <= hi))[0]
spin_up = float(t[inside[0]]) if inside.size else None
blocks = [float(E[(t > a) & (t <= a + 10 + 1e-9)].mean()) for a in np.arange(0, args.t1, 10)]

# decorrelation time from the 64^2 snapshots
X = cat["snap"][win].astype(np.float64)
X = X.reshape(X.shape[0], -1)
X = X - X.mean(axis=0)
var = float(np.mean(X**2))
max_lag = int(round(args.max_lag / 0.1))
C = np.array([np.mean(X[: X.shape[0] - l] * X[l:]) / var for l in range(max_lag + 1)])
below = np.nonzero(C < np.exp(-1))[0]
if below.size:
    l = below[0]
    tau_e = float(0.1 * (l - 1 + (C[l - 1] - np.exp(-1)) / (C[l - 1] - C[l])))
else:
    tau_e = None

res = dict(
    name=args.name, Re=meta["Re"], N=N, nu=nu, alpha=alpha, kf=kf, dtype=meta["dtype"],
    m=meta["m"], dt=meta["dt"], T=T_done, window=[args.t0, args.t1],
    wall_s=wall, steps_per_s=meta["m"] * len(t) / wall,
    max_cfl=float(cat["cfl"].max()), cfl_void=bool(cat["cfl"].max() > 0.7),
    E_lam=E_lam, E_mean=Em, E_se=Ese, Z_mean=Zm, Z_se=Zse, Pi_mean=Pim, P_mean=Pm, Q_mean=Qm,
    R_E=float(R_E), R_Z=float(R_Z), int_P=float(intP), int_Q=float(intQ),
    R_E_rel=float(abs(R_E) / intP), R_Z_rel=float(abs(R_Z) / intQ),
    visc_frac=float(2 * nu * Zm / Pm), k_d=float(k_d),
    K=K, E_K_over_max=float(spec[K] / spec.max()),
    slope_8_16=slope(8, 16), slope_8_16_se=batch_slopes(8, 16),
    slope_8_32=slope(8, 32) if K >= 32 else None,
    slope_top20=slope(k_top, K), top20_range=[k_top, K],
    spin_up=spin_up, spin_band=[float(lo), float(hi)], E_block_means=blocks,
    tau_e=tau_e, C_lag=C[:: 5].tolist(),
    spectrum=spec.tolist(),
    command=command_line(), run_command=meta["command"],
)
write_json(f"{args.name}.json", res)
for k in ("E_mean", "E_se", "Z_mean", "R_E_rel", "R_Z_rel", "max_cfl", "slope_8_16", "slope_top20",
          "E_K_over_max", "k_d", "visc_frac", "spin_up", "tau_e", "wall_s"):
    print(f"{k:14s} {res[k]}")
