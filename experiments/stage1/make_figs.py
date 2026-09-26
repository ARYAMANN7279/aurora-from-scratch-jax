"""Stage 1 figures -> figs/*.png.

usage: make_figs.py
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import DATA, FIGS, read_json  # noqa: E402

# reference categorical palette, first three slots (validated all-pairs, light surface)
COL = {100: "#2a78d6", 300: "#eb6834", 1000: "#1baf7a"}
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LONG = {100: "L100", 300: "L300", 1000: "L1000"}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "font.size": 9, "legend.frameon": False,
})
os.makedirs(FIGS, exist_ok=True)


def load(name):
    files = sorted(glob.glob(os.path.join(DATA, name, "chunk_*.npz")))
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "wall"}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. time series: E, Z, P, CFL per run (small multiples, one series per panel)
fig, ax = plt.subplots(3, 4, figsize=(13, 7), sharex=True)
for i, (re, name) in enumerate(LONG.items()):
    d = load(name)
    for j, (key, lab) in enumerate([("E", "E"), ("Z", "Z"), ("P", "P"), ("cfl", "CFL")]):
        a = ax[i, j]
        a.plot(d["t"], d[key], color=COL[re], lw=0.8)
        a.axvspan(0, 50, color=GRID, alpha=0.6, lw=0)
        if key == "cfl":
            a.axhline(0.7, color=MUTED, ls="--", lw=1)
            a.text(d["t"][-1], 0.7, " void above 0.7", va="bottom", ha="right", color=MUTED, fontsize=8)
        if key == "E":
            r = read_json(f"{name}.json")
            a.axhline(r["E_mean"], color=INK, lw=0.8, ls=":")
        a.set_ylabel(f"Re {re}\n{lab}" if j == 0 else lab)
        if i == 2:
            a.set_xlabel("t")
fig.suptitle("Long runs: time series (grey = discarded spin-up; dotted = <E> over [50, 250])", x=0.01, ha="left")
save(fig, "timeseries.png")

# 2. time-mean spectra
fig, a = plt.subplots(figsize=(6.5, 4.5))
for re, name in LONG.items():
    r = read_json(f"{name}.json")
    s = np.array(r["spectrum"])
    k = np.arange(s.size)
    sel = (k >= 1) & (k <= r["K"])
    a.loglog(k[sel], s[sel], color=COL[re], label=f"Re {re} (N {r['N']}), slope[8,16] {r['slope_8_16']:.2f}")
kk = np.array([6, 40.0])
ref = read_json("L1000.json")
anchor = np.array(ref["spectrum"])[8]
a.loglog(kk, anchor * (kk / 8) ** -3, color=MUTED, ls="--", lw=1, label="k^-3 reference")
a.axvspan(8, 16, color=GRID, alpha=0.6, lw=0)
a.text(11.3, a.get_ylim()[0] * 10, "fit range\nk ∈ [8, 16]", ha="center", color=MUTED, fontsize=8)
a.set_xlabel("k")
a.set_ylabel("E(k), time mean over [50, 250]")
a.legend(fontsize=8, loc="lower left")
save(fig, "spectra.png")

# 3. resolution map spectra at t=1
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, re in zip(ax, LONG):
    q = read_json(f"resolution_Re{re}.json")
    rows = {row["N"]: row for row in q["rows"]}
    for c, n in zip(SLOTS, q["grids"]):
        s = np.array(q["spectra_t1"][str(n)])
        k = np.arange(s.size)
        sel = (k >= 1) & (k <= (n - 1) // 3)
        lab = f"N {n}" + (f"  rel diff {rows[n]['rel_t1']:.1e}" if n in rows else "  (reference)")
        a.loglog(k[sel], s[sel], color=c, label=lab)
    a.set_title(f"Re {re}: spectra at t = 1, common IC", loc="left", fontsize=9)
    a.set_xlabel("k")
    a.legend(fontsize=8, loc="lower left")
ax[0].set_ylabel("E(k)")
save(fig, "resolution.png")

# 4. Lyapunov separation
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, (re, name) in zip(ax, LONG.items()):
    q = read_json(f"lyapunov_{name}.json")
    t, D = np.array(q["t"]), np.array(q["d"])
    a.axhspan(1e-6, 1e-2, color=GRID, alpha=0.6, lw=0)
    for i in range(D.shape[1]):
        a.semilogy(t, D[:, i], color=SLOTS[i], label=f"twin {i + 1}: λ {q['fits'][i]['lam']:.3f}")
    a.set_title(f"Re {re}: mean λ = {q['lam_mean']:.3f} (grey = fit band)", loc="left", fontsize=9)
    a.set_xlabel("t − 50")
    a.legend(fontsize=8, loc="lower right")
ax[0].set_ylabel("‖δω‖ / ‖ω‖")
save(fig, "lyapunov.png")

# 5. vorticity snapshots (64^2, t = 250)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, (re, name) in zip(ax, LONG.items()):
    w = load(name)["snap"][-1]
    v = np.abs(w).max()
    im = a.imshow(w, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, 2 * np.pi, 0, 2 * np.pi])
    a.set_title(f"Re {re}: ω at t = 250 (64² snapshot)", loc="left", fontsize=9)
    a.set_xlabel("x")
    a.grid(False)
    fig.colorbar(im, ax=a, shrink=0.8)
ax[0].set_ylabel("y")
save(fig, "snapshots.png")
