"""Stage 2 figures (handouts/stage02_dataset.md, section 6) -> figs/*.png.

usage: make_figs.py --dataset acc_v1   (run after analyse.py)
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="acc_v1")
args = ap.parse_args()
res = json.load(open(os.path.join(HERE, "results", f"{args.dataset}_analysis.json")))
root = os.path.join(ROOT, "data", "stage2", args.dataset)
COL = {100: "#2a78d6", 300: "#eb6834", 1000: "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
st = sorted(res["stability"].items(), key=lambda kv: kv[1]["Re"])

# 1. time-mean spectra of all 8 seeds per regime (one panel per regime)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, (tag, s) in zip(ax, st):
    for row in s["rows"]:
        spec = np.load(os.path.join(root, tag, f"seed_{row['seed']:04d}", "diag.npz"))["spec"].mean(axis=0)
        k = np.arange(spec.size)
        sel = (k >= 1) & (k <= 31)
        a.loglog(k[sel], spec[sel], color=COL[int(s["Re"])], lw=0.8, alpha=0.6)
    m = np.array(s["mean_spectrum"])
    k = np.arange(m.size)
    sel = (k >= 1) & (k <= 31)
    a.loglog(k[sel], m[sel], color=INK, lw=1.4, ls="--", label="regime mean")
    a.set_title(f"Re {int(s['Re'])} (N_gen {s['N_gen']}): 8 seeds, max shell deviation {s['max_spec_dev']:.2f}",
                loc="left", fontsize=9)
    a.set_xlabel("k")
    a.legend(loc="lower left", fontsize=8)
ax[0].set_ylabel("E(k), time mean over the record")
fig.savefig(os.path.join(HERE, "figs", "spectra_by_seed.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# 2. per-seed <E> with batch-means SE
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for a, (tag, s) in zip(ax, st):
    seeds = [r["seed"] for r in s["rows"]]
    a.errorbar(seeds, [r["E_mean"] for r in s["rows"]], yerr=[r["E_se"] for r in s["rows"]], fmt="o",
               color=COL[int(s["Re"])], ms=5, capsize=3)
    a.axhline(s["E_weighted_mean"], color=MUTED, ls="--", lw=1)
    a.set_title(f"Re {int(s['Re'])}: ⟨E⟩ per seed ± SE, reduced χ² {s['chi2_red']:.2f}", loc="left", fontsize=9)
    a.set_xlabel("seed")
ax[0].set_ylabel("⟨E⟩ over the record")
fig.savefig(os.path.join(HERE, "figs", "energy_by_seed.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote figs/spectra_by_seed.png, figs/energy_by_seed.png")
