"""E10 figure: error by field, error by scale, and the training/evaluation gap (results/analysis.json).

usage: python experiments/e10/make_figs.py   (writes figs/e10_diagnostics.png)
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "results", "analysis.json")))
M = A["models"]
GREY, BLUE, INK2 = "#9aa5b1", "#2a78d6", "#5d6d7e"
groups = {"from scratch": sorted(n for n in M if "scratch" in n), "pretrained + fine-tuned (n = 32)": sorted(n for n in M if "_n32_" in n)}
cols = [GREY, BLUE]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
f = ("u", "v", "p", "omega")
x = np.arange(4)
for i, (g, names) in enumerate(groups.items()):
    s = np.array([[M[n]["share_eval"][k] for k in f] for n in names])
    ax[0].bar(x + (i - 0.5) * 0.38, s.mean(0), 0.38, yerr=s.std(0), color=cols[i], label=g)
ax[0].axhline(0.25, color=INK2, lw=0.8, ls=":")
ax[0].set(xticks=x, xticklabels=["u", "v", "p", "ω"], ylabel="share of the evaluation error", ylim=(0, 0.8))
ax[0].set_title("(a) ω holds most of the error", loc="left", fontweight="bold", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8.5)

k = np.arange(1, 33)
for i, g in enumerate(groups):
    ax[1].semilogy(k, np.asarray(A["shell_ratio"]["scratch" if i == 0 else "ft n32"])[1:33], "-o", ms=3, color=cols[i], label=g)
ax[1].axhline(1, color=INK2, lw=0.8, ls=":")
ax[1].axvline(4, color=INK2, lw=0.8, ls="--")
ax[1].text(4.6, 0.15, "forcing\nk_f = 4", color=INK2, fontsize=8)
ax[1].text(0.5, 1.3, "error = signal", color=INK2, fontsize=8)
ax[1].set(xlabel="wavenumber |k| (larger = smaller eddies)", ylabel="error power / true power")
ax[1].set_title("(b) the error is a small-scale error", loc="left", fontweight="bold", fontsize=10.5)

gap = [[M[n]["gap"] for n in names] for names in groups.values()]
b = ax[2].bar([0, 1], [np.mean(g) for g in gap], 0.55, color=cols, yerr=[np.std(g) for g in gap])
ax[2].bar_label(b, fmt="%.1f×", fontweight="bold", padding=3)
ax[2].set(xticks=[0, 1], xticklabels=["from scratch", "pretrained +\nfine-tuned (n = 32)"], ylabel="evaluation MSE / training MSE",
          ylim=(0, 40))
ax[2].set_title("(c) scratch memorises its two trajectories", loc="left", fontweight="bold", fontsize=10.5)
for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "e10_diagnostics.png"), dpi=160, facecolor="white")
