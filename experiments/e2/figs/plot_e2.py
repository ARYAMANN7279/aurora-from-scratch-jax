"""E2 figure from experiments/e2/results/analysis.json: one-step relative error by set and method.

usage: python experiments/e2/figs/plot_e2.py   (writes e2_rel_err.png beside it)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))["rel_err"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
METHODS = (("perceiver", "Perceiver", "#2a78d6"), ("pad", "pad-and-mask", "#eb6834"),
           ("per_source", "per-source encoders", "#1baf7a"))
SETS = (("a", "(a) mixed\nresolution"), ("b", "(b) mixed\nvariables"), ("c", "(c) both"))

fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=SURFACE)
for ax, (src, title) in zip(axs, (("A", "Source A (Re 100)"), ("B", "Source B (Re 300)"))):
    ax.set_facecolor(SURFACE)
    x = np.arange(len(SETS))
    w = 0.26
    for j, (m, lab, col) in enumerate(METHODS):
        mu = [R[f"{s}/{m}/{src}"][0] for s, _ in SETS]
        se = [R[f"{s}/{m}/{src}"][1] for s, _ in SETS]
        xs = x + (j - 1) * (w + 0.02)
        ax.bar(xs, mu, width=w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, mu, yerr=se, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    href = R[f"h/perceiver/{src}"][0]
    ax.axhline(href, color=INK2, lw=1, ls="--", zorder=1)
    ax.annotate("(h) homogeneous", (len(SETS) - 0.45, href), xytext=(0, 3), textcoords="offset points",
                va="bottom", ha="left", color=INK2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in SETS], color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlim(-0.5, len(SETS) + 0.6)
axs[0].set_ylabel("one-step relative error (mean ± SE, 3 seeds)", color=INK2, fontsize=9)
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=3)
fig.suptitle("E2: one-step error in each source's own format (lower is better; dashed: Perceiver on (h))",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(HERE, "e2_rel_err.png"), dpi=150, facecolor=SURFACE)
print("wrote e2_rel_err.png")
