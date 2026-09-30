"""E3 figure from experiments/e3/results/analysis.json: zero-shot ratio and advantage against distance beyond the band.

usage: python experiments/e3/figs/plot_e3.py   (writes e3_distance.png beside it)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))["regimes"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
tags = sorted(R, key=lambda t: R[t]["d"])
d = [R[t]["d"] for t in tags]
re = [t.split("_")[0][2:] for t in tags]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel("distance beyond the band, d = ln(Re / 250)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.9), facecolor=SURFACE)
style(a1, "Zero-shot error / persistence", "ratio (mean ± SE, 3 seeds)")
a1.errorbar(d, [R[t]["zero_shot"][0] for t in tags], yerr=[R[t]["zero_shot"][1] for t in tags],
            color=BLUE, lw=2, marker="o", ms=6, capsize=3, zorder=3)
for di, ri, zi in zip(d, re, [R[t]["zero_shot"][0] for t in tags]):
    a1.annotate(f"Re {ri}", (di, zi), xytext=(8, -5), textcoords="offset points", ha="left", va="top",
                color=INK2, fontsize=8)
a1.set_ylim(0, None)
style(a2, "Pretraining advantage 1 − e_ft / e_scratch", "advantage, % (mean ± SE, 3 seeds)")
for key, lab, col in (("adv", "one-step", BLUE), ("adv_drift", "20-step drift", ORANGE)):
    mu = [100 * R[t][key][0] for t in tags]
    se = [100 * R[t][key][1] for t in tags]
    a2.errorbar(d, mu, yerr=se, color=col, lw=2, marker="o", ms=6, capsize=3, label=lab, zorder=3)
    a2.annotate(lab, (d[-1], mu[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK2, fontsize=8)
a2.axhline(0, color=INK2, lw=1, zorder=1)
a2.set_xlim(None, d[-1] + 0.25)
a2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
fig.suptitle("E3: pretrained on Re 125–250 × k_f 3–6, tested at k_f = 4 beyond the band",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e3_distance.png"), dpi=150, facecolor=SURFACE)
print("wrote e3_distance.png")
