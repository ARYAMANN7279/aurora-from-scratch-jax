"""Figures for the combined UGP report and deck (Part I: RealPDE airfoil wake; Part II: Aurora in JAX).

usage: ../UGP-II/.venv/bin/python make_figs.py   (writes figs/*.png; reads the UGP-II and UGP folders beside this one)
"""
import json
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
U2 = os.path.join(HERE, "..", "UGP-II")  # the UGP-II repository, next to this folder
U1 = os.path.join(U2, "..", "UGP")  # the UGP-I project folder, for three crops
OUT = os.path.join(HERE, "figs")
INK, MUT, BLUE, OR, GREY = "#2C3E50", "#5D6D7E", "#558ED5", "#CA6F1E", "#AAB4BF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT})


def crop(path, box, size=None):
    im = Image.open(path).convert("RGB").crop(box)
    return np.asarray(im if size is None else im.resize(size, Image.LANCZOS))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def clean(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# flow images: the measured wake (Part I) and Kolmogorov vorticity at three Re (Part II)
wake = crop(os.path.join(U1, "figs", "fig6_data.png"), (1562, 185, 2245, 500))
SNAP = os.path.join(U2, "experiments", "stage1", "figs", "snapshots.png")
snaps = [crop(SNAP, (x, 38, x + 280, 318), (400, 400)) for x in (63, 490, 917)]

# overview: the two parts side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [2.2, 1]})
ax[0].imshow(wake)
ax[0].set_title("Part I: a real airfoil wake, measured in a water tunnel", fontsize=11.5, loc="left")
ax[1].imshow(snaps[1])
ax[1].set_title("Part II: a flow we simulate and control", fontsize=11.5, loc="left")
for a in ax:
    a.set(xticks=[], yticks=[])
fig.tight_layout()
save(fig, "overview.png")

# Part I: our entry against the organisers' baseline, per scoring channel
ch = ["accuracy\n(rel_l2)", "fluctuations\n(tke)", "probes\n(mvpe)", "speed\n(time)", "uncertainty\n(sps)", "overall"]
kit = [94.17, 74.03, 92.84, 92.66, 14.08, 72.65]
ours = [93.977, 78.623, 93.244, 90.771, 39.558, 80.081]
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(ch))
ax.bar(x - 0.2, kit, 0.4, color=GREY, label="organisers' baseline")
b = ax.bar(x + 0.2, ours, 0.4, color=[BLUE] * 5 + [OR], label="our entry")
ax.bar_label(b, fmt="%.1f", fontweight="bold", fontsize=10)
ax.set(xticks=x, xticklabels=ch, ylim=(0, 118), ylabel="score (out of 100)")
ax.legend(frameon=False, loc="upper center", ncol=2)
clean(ax)
save(fig, "ugp1_channels.png")

# Part I: every submission that set a new best, on the day it was scored (dates from the project record)
import datetime as dt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
subs = [((8, 23), 78.4566), ((8, 30), 79.2464), ((9, 1), 79.3418), ((9, 2), 79.3695), ((9, 4), 79.3789), ((9, 5), 79.4626), ((9, 9), 79.4844),
        ((9, 13), 79.4870), ((9, 14), 79.4939), ((9, 15), 79.5907), ((9, 16), 79.6026), ((9, 17), 79.6567), ((9, 18), 79.7535),
        ((9, 21), 79.8574), ((9, 23), 80.0613), ((9, 24), 80.0813)]
days = [dt.date(2026, m, d) for (m, d), _ in subs]
prog = [v for _, v in subs]
fig, ax = plt.subplots(figsize=(11.2, 4.5))
ax.axhline(80, color="#1E8B4C", lw=2, ls="--")
ax.text(days[0], 80.04, "target: 80", color="#1E8B4C", ha="left", va="bottom", fontsize=14, fontweight="bold")
ax.plot(days, prog, "-o", color="#2E86C1", lw=3.5, ms=9)
ax.text(days[0] + dt.timedelta(days=1.1), prog[0], "78.45: first fine-tuned model", fontsize=13, color=INK, ha="left", va="center")
for i, t, dx, dy, ha in ((1, "uncertainty bands added\n(the biggest jump, +0.8)", 1.5, -0.55, "left"), (9, "turbulence head added", 0.8, -0.5, "left"),
                         (14, "band widths calibrated", -5.0, 0.3, "center")):
    ax.annotate(t, (days[i], prog[i]), (days[i] + dt.timedelta(days=dx), prog[i] + dy), fontsize=14, fontweight="bold", color=INK, ha=ha,
                arrowprops=dict(arrowstyle="->", color=MUT, lw=1.6))
ax.text(days[-1] + dt.timedelta(days=0.6), prog[-1] + 0.05, "80.08", ha="left", va="center", fontsize=16, fontweight="bold", color="#2E86C1")
ax.text(0.01, 0.97, "16 submissions set a new best in one month", transform=ax.transAxes, fontsize=15, fontweight="bold", color="#1F4E79", va="top")
ax.set(ylim=(78.2, 80.5), xlim=(dt.date(2026, 8, 21), dt.date(2026, 9, 29)), yticks=[78.5, 79.0, 79.5, 80.0], ylabel="leaderboard score")
ax.set_xticks([dt.date(2026, 8, 23), dt.date(2026, 8, 30), dt.date(2026, 9, 8), dt.date(2026, 9, 15), dt.date(2026, 9, 22)])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
ax.tick_params(labelsize=13)
ax.yaxis.label.set_size(14)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#E2E8F0")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "ugp1_climb.png"), dpi=200, facecolor="white")
plt.close(fig)

# Part II E10: error by swirl size (simple)
D = json.load(open(os.path.join(U2, "experiments", "e10", "results", "analysis.json")))
fig, ax = plt.subplots(figsize=(5.2, 3.7))
kk = np.arange(1, 33)
for g, c, l in (("scratch", GREY, "trained from scratch"), ("ft n32", BLUE, "pretrained")):
    ax.plot(kk, 100 * np.asarray(D["shell_ratio"][g])[1:33], lw=3, color=c, label=l)
ax.axhline(100, color=MUT, lw=1, ls=":")
ax.annotate("error as big as\nthe swirls themselves", xy=(31.5, 100), xytext=(31.5, 55), color=MUT, fontsize=10, ha="right", arrowprops=dict(arrowstyle="->", color=MUT))
ax.set(xlabel="swirl size  (big  →  tiny)", ylabel="error, % of the swirls' strength", ylim=(0, 180), xticks=[], xlim=(1, 32))
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
clean(ax)
save(fig, "p2_tiny_swirls.png")

# Part II E11: 20-step forecast error, three arms
E = json.load(open(os.path.join(U2, "experiments", "e11", "results", "analysis.json")))["runs"]
fig, ax = plt.subplots(figsize=(5.2, 3.7))
for a, c, l in (("B", GREY, "normal model"), ("W", BLUE, "spin only, physics built in")):
    e = np.mean([E[f"e11_{a}_s{s}"]["err_by_step"] for s in (0, 1, 2)], 0)
    ax.plot(np.arange(1, 51), e, lw=3, color=c, label=l)
ax.axvline(20, color=MUT, lw=1, ls=":")
ax.set(xlabel="forecast steps ahead", ylabel="forecast error", xlim=(1, 50))
ax.legend(frameon=False, loc="lower right")
clean(ax)
save(fig, "p2_physics.png")
print(sorted(os.listdir(OUT)))
