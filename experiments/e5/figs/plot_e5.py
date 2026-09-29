"""E5 figure from experiments/e5/results/analysis.json: gain over zero-shot and the pretraining advantage vs k.

usage: python experiments/e5/figs/plot_e5.py   (writes e5_gain_adv.png beside it)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))["sizes"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
K = sorted(int(k) for k in S)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xticks(K)
    ax.set_xlabel("fine-tuning trajectories k (Re 1000)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), facecolor=SURFACE)
style(a1, "Gain over zero-shot: 1 − e / e_zero-shot", "gain, % (mean ± SE, 3 seeds)")
for key, lab, col in (("gain_full", "full fine-tuning", BLUE), ("gain_lora", "LoRA r = 4", ORANGE)):
    mu = [100 * S[str(k)][key][0] for k in K]
    se = [100 * S[str(k)][key][1] for k in K]
    a1.errorbar(K, mu, yerr=se, color=col, lw=2, marker="o", ms=6, capsize=3, label=lab, zorder=3)
    a1.text(K[-1] + 0.15, mu[-1], lab, color=INK2, fontsize=8, va="center")
a1.set_xlim(0.5, K[-1] + 1.6)
a1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
style(a2, "Pretraining advantage: 1 − e_full / e_scratch", "advantage, % (mean ± SE, 3 seeds)")
mu = [100 * S[str(k)]["adv"][0] for k in K]
se = [100 * S[str(k)]["adv"][1] for k in K]
a2.errorbar(K, mu, yerr=se, color=BLUE, lw=2, marker="o", ms=6, capsize=3, zorder=3)
a2.set_xlim(0.5, K[-1] + 0.5)
fig.suptitle("E5: fine-tuning on Re 1000 from the Re 100/300 pretrained model", color=INK, fontsize=11,
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e5_gain_adv.png"), dpi=150, facecolor=SURFACE)
print("wrote e5_gain_adv.png")
