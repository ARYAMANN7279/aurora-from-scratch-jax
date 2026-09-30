"""E1 figure from experiments/e1/results/analysis.json: advantage (one-step, drift) and zero-shot error per cell.

usage: python experiments/e1/figs/plot_e1.py   (writes e1_cells.png beside it)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))["cells"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TS = ((32, "T = 32", "#2a78d6"), (64, "T = 64", "#eb6834"), (128, "T = 128", "#1baf7a"))
NS = (2, 8, 32)
PANELS = (("adv_rel_err", 100, "Pretraining advantage, one-step", "1 − e_ft / e_scratch, %"),
          ("adv_rollout_drift", 100, "Pretraining advantage, 20-step drift", "1 − d_ft / d_scratch, %"),
          ("zero_shot", 1, "Zero-shot one-step error on (300, 4)", "relative error"))

fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.9), facecolor=SURFACE)
x = np.arange(len(NS))
w = 0.26
for ax, (key, scale, title, ylab) in zip(axs, PANELS):
    ax.set_facecolor(SURFACE)
    for j, (T, lab, col) in enumerate(TS):
        mu = [scale * C[f"n{n}_T{T}"][f"{key}_mean"] for n in NS]
        se = [scale * C[f"n{n}_T{T}"][f"{key}_se"] for n in NS]
        xs = x + (j - 1) * (w + 0.02)
        ax.bar(xs, mu, width=w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, mu, yerr=se, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n in NS], color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylab, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=3)
fig.suptitle("E1: regimes n × total pretraining trajectories T, target (Re, k_f) = (300, 4); mean ± SE, 3 seeds",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(HERE, "e1_cells.png"), dpi=150, facecolor=SURFACE)
print("wrote e1_cells.png")
