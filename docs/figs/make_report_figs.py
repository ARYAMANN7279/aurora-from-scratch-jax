"""Illustrative figures for FINAL_REPORT.md's main text: concept, testbed, task, model and results at a glance.

usage: python docs/figs/make_report_figs.py   (writes fig_*.png beside it)
Flow images are cropped from experiments/stage1/figs/snapshots.png and experiments/e1/figs/e1_contours_n32.png;
the result panels read experiments/e1/results/analysis.json and experiments/e5/results/analysis.json.
"""
import json
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
INK, INK2, GRID, CARD = "#1f2a36", "#5d6d7e", "#e6e5e1", "#eef2f7"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b8c2cc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})


def crop(path, box, size):
    return np.asarray(Image.open(os.path.join(ROOT, path)).convert("RGB").crop(box).resize((size, size), Image.LANCZOS))


SNAP = "experiments/stage1/figs/snapshots.png"  # Re 100 / 300 / 1000 vorticity at t = 250, 64² snapshots
snaps = [crop(SNAP, (x, 38, x + 280, 318), 400) for x in (63, 490, 917)]
CONT = "experiments/e1/figs/e1_contours_n32.png"  # rows: truth, zero-shot, fine-tuned, scratch; column 1 = step 1
truth, ft, scratch = (crop(CONT, (116, y0 + 26, 116 + y1 - y0 - 30, y1 - 4), 400)
                      for y0, y1 in ((80, 321), (380, 622), (980, 1223)))


def box(ax, x, y, w, h, text, fill=CARD, color=INK, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fill, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# 1. pretraining vs scratch
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
ax.text(0, 2.75, "Pretrained", color=BLUE, fontweight="bold", fontsize=11)
box(ax, 0, 1.55, 3.0, 0.95, "Pretrain\nmany related flow regimes", size=9.5)
box(ax, 3.6, 1.55, 2.6, 0.95, "Fine-tune\nshort, on the target", size=9.5)
box(ax, 6.8, 1.55, 3.0, 0.95, "Predict\nunseen target trajectories", fill="#e4eefb", size=9.5)
arrow(ax, 3.02, 2.02, 3.58, 2.02)
arrow(ax, 6.22, 2.02, 6.78, 2.02)
ax.text(0, 1.1, "From scratch", color=INK2, fontweight="bold", fontsize=11)
box(ax, 3.6, 0.0, 2.6, 0.95, "Train only on the\ntarget, same total steps", size=9.5)
box(ax, 6.8, 0.0, 3.0, 0.95, "Predict\nthe same unseen trajectories", fill="#e4eefb", size=9.5)
arrow(ax, 6.22, 0.47, 6.78, 0.47)
save(fig, "fig_concept.png")

# 2. the testbed: forcing bands and three Reynolds numbers
fig = plt.figure(figsize=(9.6, 2.9))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.12)
ax = fig.add_subplot(gs[0])
kf = 4
for j in range(2 * kf):
    y0 = j * np.pi / kf
    ax.axhspan(y0, y0 + np.pi / kf, color="#f6cdb0" if j % 2 == 0 else "#c4d8f2", lw=0)
    for x in (0.9, 2.5, 4.1, 5.7):
        dx = 0.9 if j % 2 == 0 else -0.9
        ax.annotate("", xy=(x + dx / 2, y0 + np.pi / (2 * kf)), xytext=(x - dx / 2, y0 + np.pi / (2 * kf)),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.set(xlim=(0, 2 * np.pi), ylim=(0, 2 * np.pi), xticks=[], yticks=[], aspect="equal")
ax.set_title("forcing: bands pushed\nleft and right (k_f = 4)", fontsize=10)
for i, (img, lab) in enumerate(zip(snaps, ("Re 100: calmer", "Re 300: the target", "Re 1000: wilder"))):
    a = fig.add_subplot(gs[i + 1])
    a.imshow(img)
    a.set(xticks=[], yticks=[])
    a.set_title(f"vorticity, {lab}", fontsize=10, color=ORANGE if i == 1 else INK)
fig.text(0.62, 0.06, "red and blue: fluid spinning in opposite directions; white: little spin", ha="center",
         color=INK2, fontsize=9)
save(fig, "fig_testbed.png")

# 3. the task
fig, ax = plt.subplots(figsize=(9.0, 2.7))
ax.set_xlim(0, 10)
ax.set_ylim(-0.55, 2.4)
ax.axis("off")
for img, x, lab in ((snaps[1], 0.0, "X(t − Δt)"), (truth, 1.9, "X(t)")):
    ax.imshow(img, extent=(x, x + 1.7, 0.3, 2.0))
    ax.text(x + 0.85, 0.08, lab, ha="center", color=INK2)
arrow(ax, 3.7, 1.15, 4.3, 1.15)
box(ax, 4.35, 0.6, 1.9, 1.1, "model\n5.1 M\nparameters", fill=INK, color="white", bold=True, size=9)
arrow(ax, 6.3, 1.15, 6.9, 1.15)
ax.imshow(ft, extent=(7.0, 8.7, 0.3, 2.0))
ax.text(7.85, 0.08, "X̂(t + Δt)", ha="center", color=INK2)
ax.text(5.3, -0.35, "inputs: two snapshots Δt = 0.2 apart, 4 fields each (u, v, p, ω) on 64 × 64; "
        "output: the next snapshot.\nA rollout feeds each prediction back in as the newest input. (Illustrative snapshots.)",
        ha="center", color=INK2, fontsize=9)
save(fig, "fig_task.png")

# 4. the model
fig, ax = plt.subplots(figsize=(12, 2.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.6)
ax.axis("off")
steps = [("two snapshots", "u, v, p, ω on 64²"), ("spectral levels", "5 bands per field"),
         ("patch embed", "4 × 4 patches\n→ 256 tokens"), ("Perceiver", "5 levels →\n3 latent levels"),
         ("Swin U-Net", "windowed attention,\n3 scales down and up"), ("decode", "next snapshot")]
w, gap = 1.45, 0.26
for i, (h, d) in enumerate(steps):
    x = i * (w + gap)
    box(ax, x, 0.2, w, 1.2, "", fill="#e4eefb" if i in (0, 5) else CARD)
    ax.text(x + w / 2, 1.08, h, ha="center", va="center", fontweight="bold", fontsize=9.5)
    ax.text(x + w / 2, 0.6, d, ha="center", va="center", color=INK2, fontsize=8.2)
    if i < len(steps) - 1:
        arrow(ax, x + w + 0.02, 0.8, x + w + gap - 0.02, 0.8)
save(fig, "fig_model.png")

# 5. results at a glance
e1 = json.load(open(os.path.join(ROOT, "experiments/e1/results/analysis.json")))
sc = {s: v["rel_err"] for s, v in e1["scratch"].items()}
ft_err = [sc[str(s)] * (1 - a) for c in e1["cells"].values() for s, a in zip(c["seeds"], c["adv_rel_err"])]
zs = {n: np.mean([c["zero_shot_mean"] for c in e1["cells"].values() if c["n"] == n]) for n in (2, 8, 32)}
e5 = json.load(open(os.path.join(ROOT, "experiments/e5/results/analysis.json")))["sizes"]
k = sorted(e5, key=int)


def style(a, title, ylabel):
    a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
    a.set_ylabel(ylabel, color=INK2)
    a.grid(axis="y", color=GRID)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
v = [np.mean(list(sc.values())), np.mean(ft_err)]
b = ax[0].bar(["from scratch", "pretrained +\nfine-tuned"], v, color=[GREY, BLUE], width=0.55)
ax[0].bar_label(b, fmt="%.3f", fontweight="bold")
ax[0].set_ylim(0, 0.175)
style(ax[0], "(a) pretraining halves the error", "one-step error (E1)")
b = ax[1].bar(["2 regimes", "8 regimes", "32 regimes\n(neighbours at k_f = 4)"], list(zs.values()),
              color=[GREY, GREY, ORANGE], width=0.55)
ax[1].bar_label(b, fmt="%.3f", fontweight="bold")
ax[1].set_ylim(0, 0.22)
style(ax[1], "(b) proximity sets the zero-shot start", "zero-shot error (E1)")
ax[2].plot([int(x) for x in k], [100 * e5[x]["adv"][0] for x in k], "-o", color=BLUE, lw=2.2, ms=7)
for x in k:
    ax[2].annotate(f"{100 * e5[x]['adv'][0]:.0f}%", (int(x), 100 * e5[x]["adv"][0]), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontweight="bold")
ax[2].set(xticks=[int(x) for x in k], ylim=(0, 52), xlabel="target trajectories available")
style(ax[2], "(c) largest when target data is scarce", "pretraining advantage (E5), %")
save(fig, "fig_glance.png")
