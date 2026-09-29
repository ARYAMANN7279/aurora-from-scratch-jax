"""E4 figures from experiments/e4/results/analysis.json: change against base, mean ± SE over 3 seeds.

usage: python experiments/e4/figs/plot_e4.py   (writes e4_onestep.png and e4_drift.png beside it)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))["change_vs_base"]
VARIANTS = ["noshift", "depth6", "depth24", "L1", "L6", "pf_steps", "pf_compute", "mae"]
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def axes(title, xlabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(-3, 3, color=BAND, zorder=0)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(f"← better      {xlabel}      worse →", color=INK2, fontsize=9)
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(VARIANTS, color=INK)
    ax.invert_yaxis()
    return fig, ax


def finish(fig, ax, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# 1. one-step error, averaged over Re 100 and Re 300
fig, ax = axes("E4: one-step error vs base (shaded: ±3%)", "change in one-step error vs base, % (mean ± SE, 3 seeds)")
m = np.array([100 * A[v]["onestep"][0] for v in VARIANTS])
se = np.array([100 * A[v]["onestep"][1] for v in VARIANTS])
y = np.arange(len(VARIANTS))
ax.barh(y, m, height=0.5, color=BLUE, zorder=2)
ax.errorbar(m, y, xerr=se, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
finish(fig, ax, os.path.join(HERE, "e4_onestep.png"))

# 2. 50-step rollout drift, per regime
fig, ax = axes("E4: 50-step rollout drift vs base (shaded: ±3%)", "change in rollout drift vs base, % (mean ± SE, 3 seeds)")
h = 0.36
for k, (t, lab, col) in enumerate((("re100_kf4", "Re 100", BLUE), ("re300_kf4", "Re 300", ORANGE))):
    yk = y + (k - 0.5) * (h + 0.04)
    mk = np.array([100 * A[v][f"{t}/rollout_drift"][0] for v in VARIANTS])
    sk = np.array([100 * A[v][f"{t}/rollout_drift"][1] for v in VARIANTS])
    ax.barh(yk, mk, height=h, color=col, label=lab, zorder=2)
    ax.errorbar(mk, yk, xerr=sk, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.text(6, yk[0], lab, ha="left", va="center", color=INK2, fontsize=8)  # direct label, top row
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
finish(fig, ax, os.path.join(HERE, "e4_drift.png"))
print("wrote e4_onestep.png, e4_drift.png")
