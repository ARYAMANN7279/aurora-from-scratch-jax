"""Figures for the combined UGP report and deck (Part I: RealPDE airfoil wake; Part II: Aurora in JAX).

usage: ../.venv/bin/python make_figs.py   (writes figs/*.png; reads this repository and the UGP-I folder beside it)
"""
import json
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
U2 = os.path.join(HERE, "..")  # this repository (UGP-II)
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
Image.fromarray(wake).save(os.path.join(OUT, "wake.png"))
SNAP = os.path.join(U2, "experiments", "stage1", "figs", "snapshots.png")
snaps = [crop(SNAP, (x, 38, x + 280, 318), (400, 400)) for x in (63, 490, 917)]
for i, s in enumerate(snaps):
    Image.fromarray(s).save(os.path.join(OUT, f"kolmo{i}.png"))
CONT = os.path.join(U2, "experiments", "e1", "figs", "e1_contours_n32.png")
for name, (y0, y1) in zip(("truth", "pretrained", "zeroshot", "scratch"), ((80, 321), (680, 922), (380, 622), (980, 1223))):
    Image.fromarray(crop(CONT, (116, y0 + 26, 116 + y1 - y0 - 30, y1 - 4), (400, 400))).save(os.path.join(OUT, f"c_{name}.png"))

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

# Part II: the three headline effects, simple bars (E1, E1 zero-shot, E5)
A = json.load(open(os.path.join(U2, "experiments", "e1", "results", "analysis.json")))
sc = {s: v["rel_err"] for s, v in A["scratch"].items()}
ft = [sc[str(s)] * (1 - a) for c in A["cells"].values() for s, a in zip(c["seeds"], c["adv_rel_err"])]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
b = ax.bar(["from scratch", "pretrained"], [np.mean(list(sc.values())), np.mean(ft)], color=[GREY, BLUE], width=0.55)
ax.bar_label(b, fmt="%.3f", fontweight="bold")
ax.set(ylabel="next-step error (lower is better)", ylim=(0, 0.18))
clean(ax)
save(fig, "p2_halves.png")
zs = {n: np.mean([c["zero_shot_mean"] for c in A["cells"].values() if c["n"] == n]) for n in (2, 8, 32)}
fig, ax = plt.subplots(figsize=(4.6, 3.6))
b = ax.bar(["2 settings", "8 settings", "32 settings\n(incl. close ones)"], list(zs.values()), color=[GREY, GREY, OR], width=0.55)
ax.bar_label(b, fmt="%.3f", fontweight="bold")
ax.set(ylabel="error before any fine-tuning", ylim=(0, 0.23))
clean(ax)
save(fig, "p2_closeness.png")
E5 = json.load(open(os.path.join(U2, "experiments", "e5", "results", "analysis.json")))["sizes"]
k = sorted(E5, key=int)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot([int(i) for i in k], [100 * E5[i]["adv"][0] for i in k], "-o", color=BLUE, lw=3, ms=8)
for i in k:
    ax.annotate(f"{100 * E5[i]['adv'][0]:.0f}%", (int(i), 100 * E5[i]["adv"][0]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontweight="bold")
ax.set(xticks=[int(i) for i in k], xlabel="target simulations available", ylabel="gain from pretraining (%)", ylim=(0, 52))
clean(ax)
save(fig, "p2_scarce.png")

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
