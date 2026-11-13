"""Charts for the combined deck, drawn at each slide's exact size so the text is the size it looks.

usage: ../../.venv/bin/python make_deck_figs.py   (reads this repository's result files, writes ../figs/d_*.png)
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, "..", "figs")
R2 = os.path.join(HERE, "..", "..")  # the repository root
INK, BLUE, OR, MUT, GREY, LIGHT = "#2C3E50", "#2E75B6", "#CA6F1E", "#5D6D7E", "#AAB4BF", "#E2E8F0"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "text.color": INK, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": GREY})
J = lambda p: json.load(open(os.path.join(R2, p)))


def fig(w, h):
    f, a = plt.subplots(figsize=(w, h), layout="constrained")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    return f, a


def save(f, name):
    f.savefig(os.path.join(OUT, name), dpi=220, facecolor="white")
    plt.close(f)


# 5. scoring weights
f, a = fig(5.6, 3.4)
names = ["Accuracy", "Uncertainty", "Fluctuations", "Speed", "Probes"]
w = np.array([46.7, 24.7, 10.0, 9.7, 9.4])
b = a.barh(names, w, color=[BLUE, OR, BLUE, BLUE, BLUE], height=0.62)
for r, v in zip(b, w):
    a.text(v + 1, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center", fontsize=16, fontweight="bold")
a.invert_yaxis()
a.set(xlim=(0, 58), xticks=[])
a.tick_params(axis="y", labelsize=16, length=0)
a.spines["bottom"].set_visible(False)
a.spines["left"].set_visible(False)
save(f, "d_weights.png")

# 8. channels
f, a = fig(6.3, 3.55)
ch = ["Accuracy", "Fluct.", "Probes", "Speed", "Uncert.", "Overall"]
kit = [94.2, 74.0, 92.8, 92.7, 14.1, 72.7]
ours = [94.0, 78.6, 93.2, 90.8, 39.6, 80.1]
x = np.arange(6)
a.bar(x - 0.2, kit, 0.4, color=GREY, label="Organisers' baseline")
b = a.bar(x + 0.2, ours, 0.4, color=[BLUE] * 4 + [OR, OR], label="Our entry")
for xi, v in zip(x, kit):
    a.text(xi - 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, color=MUT)
for xi, v in zip(x, ours):
    a.text(xi + 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
a.set(xticks=x, xticklabels=ch, ylim=(0, 118), yticks=[])
a.tick_params(axis="x", labelsize=12.5, length=0)
a.spines["left"].set_visible(False)
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=BLUE)], labels=["Organisers' baseline", "Our entry"], frameon=False, ncol=2, loc="upper left", fontsize=13)
save(f, "d_channels.png")

# 9. climb
v = [78.450, 79.2464, 79.3418, 79.3695, 79.3789, 79.4626, 79.4844, 79.4870, 79.4939, 79.5907, 79.6026, 79.6567, 79.7535, 79.8574, 80.0613, 80.0813]
f, a = fig(9, 3.2)
a.axhline(80, color=OR, lw=2, ls="--")
a.text(-0.4, 80.07, "target 80", color=OR, ha="left", fontsize=13, fontweight="bold")
a.plot(range(16), v, "-o", color=BLUE, lw=3, ms=7)
for i, t, dy in ((1, "+ uncertainty bands", -0.42), (9, "+ turbulence fix", -0.42), (13, "+ band calibration", -0.5)):
    a.annotate(t, (i, v[i]), (i + (0.6 if i == 1 else 0.2), v[i] + dy), fontsize=14, fontweight="bold", color=INK, ha="left" if i == 1 else "center",
               arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))
a.text(0, v[0] - 0.14, "78.45", ha="center", va="top", fontsize=14, fontweight="bold")
a.text(15, v[-1] + 0.12, "80.08", ha="center", fontsize=15, fontweight="bold", color=BLUE)
a.set(ylim=(78.0, 80.5), xlim=(-0.6, 15.8), xticks=[], yticks=[78.5, 79.0, 79.5, 80.0], xlabel="each submission that beat the best so far")
a.tick_params(axis="y", labelsize=13)
a.grid(axis="y", color=LIGHT)
a.set_axisbelow(True)
save(f, "d_climb.png")

# 16. halves
E1 = J("experiments/e1/results/analysis.json")
sc = [x["rel_err"] for x in E1["scratch"].values()]
ft = [E1["scratch"][str(s)]["rel_err"] * (1 - ad) for c in E1["cells"].values() for s, ad in zip(c["seeds"], c["adv_rel_err"])]
H = [np.mean(sc), np.mean(ft)]
f, a = fig(4.8, 3.5)
b = a.bar(["From scratch", "Pretrained"], H, color=[GREY, BLUE], width=0.55)
for r, val in zip(b, H):
    a.text(r.get_x() + r.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", fontsize=20, fontweight="bold")
a.text(0.5, 0.115, "−46%", color=OR, fontsize=26, fontweight="bold", ha="center", va="center")
a.set(ylim=(0, 0.19), yticks=[], ylabel="next-step error (lower is better)")
a.tick_params(axis="x", labelsize=16, length=0)
a.spines["left"].set_visible(False)
save(f, "d_halves.png")

# 17. closeness
ZS = [np.mean([c["zero_shot_mean"] for c in E1["cells"].values() if c["n"] == n]) for n in (2, 8, 32)]
f, a = fig(5.2, 3.5)
b = a.bar(["2 settings", "8 settings", "32 settings"], ZS, color=[GREY, GREY, OR], width=0.55)
for r, val in zip(b, ZS):
    a.text(r.get_x() + r.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", fontsize=18, fontweight="bold")
a.text(2, ZS[2] / 2, "includes\nnear-target\nsettings", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
a.set(ylim=(0, 0.23), yticks=[], ylabel="error before any fine-tuning")
a.tick_params(axis="x", labelsize=15, length=0)
a.spines["left"].set_visible(False)
save(f, "d_closeness.png")

# 18. scarce
E5 = J("experiments/e5/results/analysis.json")["sizes"]
ks = sorted(E5, key=int)
y = [100 * E5[k]["adv"][0] for k in ks]
f, a = fig(5.4, 3.5)
a.plot([int(k) for k in ks], y, "-o", color=BLUE, lw=4, ms=11)
for k, val in zip(ks, y):
    a.annotate(f"{val:.0f}%", (int(k), val), (0, 12), textcoords="offset points", ha="center", fontsize=20, fontweight="bold")
a.set(xlim=(0.6, 6.4), xticks=[int(k) for k in ks], ylim=(0, 55), yticks=[], xlabel="target simulations available", ylabel="gain from pretraining")
a.spines["left"].set_visible(False)
a.tick_params(axis="x", labelsize=16)
save(f, "d_scarce.png")

# 21. tiny swirls
S = J("experiments/e10/results/analysis.json")["shell_ratio"]
k = np.arange(1, 25)
f, a = fig(5.4, 3.6)
a.plot(k, 100 * np.array(S["scratch"][1:25]), color=GREY, lw=3.5, label="Trained from scratch")
a.plot(k, 100 * np.array(S["ft n32"][1:25]), color=BLUE, lw=3.5, label="Pretrained")
a.axhline(100, color=OR, lw=2, ls="--")
a.text(1, 106, "error as big as the swirls", color=OR, fontsize=13, fontweight="bold")
a.set(ylim=(0, 200), xlim=(1, 24), xticks=[], yticks=[0, 50, 100, 150, 200], xlabel="swirl size   (big  →  tiny)", ylabel="error, % of the swirls' strength")
a.set_yticklabels(["0", "50", "100", "150", "200"])
a.tick_params(axis="y", labelsize=13)
a.legend(frameon=False, loc="upper left", fontsize=13)
a.grid(axis="y", color=LIGHT)
a.set_axisbelow(True)
save(f, "d_swirls.png")

# 22. rollout (average over the first 20 steps is what the 13% measures)
E11 = J("experiments/e11/results/analysis.json")["runs"]
roll = {arm: np.mean([E11[f"e11_{arm}_s{s}"]["err_by_step"] for s in (0, 1, 2)], 0) for arm in ("B", "W")}
d20 = {arm: np.mean([E11[f"e11_{arm}_s{s}"]["drift20"] for s in (0, 1, 2)]) for arm in ("B", "W")}
f, a = fig(5.4, 3.6)
a.axvspan(1, 20, color="#EAF1FB", zorder=0)
st = np.arange(1, 51)
a.plot(st, roll["B"], color=GREY, lw=3.5, label="Normal model")
a.plot(st, roll["W"], color=BLUE, lw=3.5, label="Spin only, physics built in")
a.text(11, 1.52, f"first 20 steps:\naverage {d20['B']:.2f} → {d20['W']:.2f}", ha="center", va="top", fontsize=12, fontweight="bold", color=BLUE)
a.set(xlim=(1, 50), ylim=(0, 1.6), xticks=[1, 10, 20, 30, 40, 50], yticks=[0, 0.5, 1.0, 1.5], xlabel="forecast steps ahead", ylabel="forecast error")
a.tick_params(labelsize=13)
a.legend(frameon=False, loc="lower right", fontsize=13)
save(f, "d_rollout.png")
print(sorted(x for x in os.listdir(OUT) if x.startswith("d_")), [round(v, 3) for v in H], round(d20["B"], 3), round(d20["W"], 3))

# ---------------------------------------------------------------- v3 additions
import glob  # noqa: E402

# Part I standing: a number line
f, a = fig(9, 2.6)
pts = [(80.081, "our entry\n80.08", OR, 1), (80.582, "top-50 cutoff\n80.58", MUT, -1), (81.727, "perfect uncertainty\non our forecast: 81.73", BLUE, 1),
       (81.901, "top-10 cutoff\n81.90", MUT, -1), (82.218, "development\nleader 82.22", INK, 1)]
TXT_X = {82.218: 82.36, 81.727: 81.62}
a.plot([79.8, 82.5], [0, 0], color=GREY, lw=3, solid_capstyle="round")
for x0, lab, col, up in pts:
    a.scatter([x0], [0], s=260 if col == OR else 140, color=col, zorder=3, edgecolor="white", linewidth=2)
    a.annotate(lab, (x0, 0), (TXT_X.get(x0, x0), 0.55 * up), ha="center", va="bottom" if up > 0 else "top", fontsize=13, fontweight="bold", color=col,
               arrowprops=dict(arrowstyle="-", color=col, lw=1.2))
a.text(79.75, -1.32, "organisers' baseline: 72.65 (off the scale)", fontsize=12, color=MUT)
a.set(xlim=(79.7, 82.6), ylim=(-1.45, 1.25), yticks=[], xticks=[80, 80.5, 81, 81.5, 82, 82.5])
for sp in ("left", "bottom"):
    a.spines[sp].set_visible(False)
a.tick_params(axis="x", labelsize=12, length=0)
save(f, "d_numberline.png")

# Part I: baseline -> ours, per channel (dumbbell)
rows = [("Accuracy", 94.2, 94.0), ("Fluctuations", 74.0, 78.6), ("Probes", 92.8, 93.2), ("Speed", 92.7, 90.8), ("Uncertainty", 14.1, 39.6), ("Overall", 72.7, 80.1)]
f, a = fig(6.3, 3.6)
for i, (nm, b0, b1) in enumerate(rows):
    hi = nm in ("Uncertainty", "Overall")
    c = OR if hi else BLUE
    a.plot([b0, b1], [i, i], color=c if hi else GREY, lw=4 if hi else 2.5, zorder=1)
    a.scatter([b0], [i], s=110, color=GREY, zorder=2)
    a.scatter([b1], [i], s=150, color=c, zorder=3)
    a.text(max(b0, b1) + 2.5, i, f"{b1:.1f}" + (f"  (+{b1 - b0:.1f})" if b1 - b0 > 1 else ""), va="center", fontsize=13, fontweight="bold" if hi else "normal", color=c if hi else INK)
a.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], xlim=(0, 118), xticks=[0, 25, 50, 75, 100], ylim=(len(rows) - 0.4, -0.7))
a.tick_params(axis="y", labelsize=14, length=0)
a.tick_params(axis="x", labelsize=11)
a.spines["left"].set_visible(False)
a.scatter([], [], s=90, color=GREY, label="organisers' baseline")
a.scatter([], [], s=110, color=BLUE, label="our entry")
a.legend(frameon=False, loc="lower left", fontsize=12, ncol=2, bbox_to_anchor=(0, 1.0))
a.grid(axis="x", color=LIGHT)
a.set_axisbelow(True)
save(f, "d_dumbbell.png")

# Part II: which settings each pretraining set covers (E1 design)
SET2 = [(200, 3), (450, 5)]
SET8 = [(re, kf) for re in (160, 200, 450, 560) for kf in (3, 5)]
SET32 = [(re, kf) for re in (125, 160, 200, 250, 350, 450, 560, 700) for kf in (3, 4, 5, 6)]
f, a = fig(5.6, 3.6)
for st, sz, col, lab in ((SET32, 90, "#C9DCF0", "32-setting set"), (SET8, 160, "#6FA3D9", "8-setting set"), (SET2, 260, INK, "2-setting set")):
    a.scatter([np.log(r) for r, _ in st], [k for _, k in st], s=sz, color=col, label=lab, zorder=3 if st is SET2 else 2, edgecolor="white", linewidth=1)
a.scatter([np.log(300)], [4], s=520, marker="*", color=OR, zorder=4, label="target (never seen)", edgecolor="white", linewidth=1)
a.annotate("closest settings\nonly in the 32-set", (np.log(350), 4), (np.log(530), 4.3), fontsize=11.5, color=BLUE, fontweight="bold", ha="center", va="bottom",
           arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
res = [125, 200, 300, 450, 700]
a.set(xticks=[np.log(r) for r in res], xticklabels=[str(r) for r in res], yticks=[3, 4, 5, 6], ylim=(2.5, 6.6), xlabel="Reynolds number (how wild)", ylabel="stirring bands $k_f$")
a.tick_params(labelsize=12)
a.legend(frameon=False, fontsize=11, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.17), columnspacing=1, handletextpad=0.3)
save(f, "d_map.png")

# Part II: learning curves on the target (F1 reruns, 3 seeds)
C = {}
for fn in glob.glob(os.path.join(R2, "experiments", "e1", "results", "curves_rows_*.jsonl")):
    for line in open(fn):
        r = json.loads(line)
        C[r["name"]] = r["curve"]
def curve(prefix):
    cs = [C[f"{prefix}_s{s}"] for s in (0, 1, 2)]
    key = [k for k in cs[0][0] if k not in ("step", "train_loss")][0]
    steps = np.array([max(p["step"], 30) for p in cs[0]])
    return steps, np.mean([[p[key] for p in c] for c in cs], 0)
f, a = fig(5.6, 3.6)
for pre, col, lab in (("e1c_scratch", GREY, "from scratch"), ("e1c_ft_n2_T32", "#6FA3D9", "pretrained, 2 settings"), ("e1c_ft_n32_T32", OR, "pretrained, 32 settings")):
    st, e = curve(pre)
    a.plot(st, e, color=col, lw=3.5, label=lab)
    if "ft" in pre:
        a.plot([st[-1], 22000], [e[-1], e[-1]], color=col, lw=2, ls=":")
st, e = curve("e1c_scratch")
a.annotate(f"{e[-1]:.3f} after 22,000 steps", (22000, e[-1]), (2500, 0.205), fontsize=12, color=MUT, fontweight="bold", arrowprops=dict(arrowstyle="->", color=MUT))
a.annotate("both pretrained models:\n0.08 within 100 steps", (100, 0.083), (330, 0.022), fontsize=12, color=INK, fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK))
a.set(xscale="log", xlim=(25, 30000), ylim=(0, 0.33), xlabel="training steps on the target", ylabel="next-step error")
a.set_xticks([30, 100, 1000, 10000])
a.set_xticklabels(["start", "100", "1,000", "10,000"])
a.tick_params(labelsize=12)
a.legend(frameon=False, fontsize=11.5, loc="upper right")
a.grid(axis="y", color=LIGHT)
a.set_axisbelow(True)
save(f, "d_curves.png")

# Part II E10: error-to-signal heat strip, by swirl size, for each group of models
D = J("experiments/e10/results/diag.json")
groups = [("from scratch", "e1_scratch_"), ("pretrained, 2 settings", "e1_ft_n2_"), ("pretrained, 8 settings", "e1_ft_n8_"), ("pretrained, 32 settings", "e1_ft_n32_")]
M = np.array([np.mean([np.array(v["trained"]["shell_err"][1:33]) / np.array(v["trained"]["shell_truth"][1:33]) for n, v in D.items() if n.startswith(p)], 0) for _, p in groups])
f, a = fig(9, 2.9)
from matplotlib.colors import LogNorm  # noqa: E402
im = a.imshow(M, aspect="auto", cmap="magma_r", norm=LogNorm(1e-4, 5), extent=(0.5, 32.5, len(groups) - 0.5, -0.5))
for i in range(len(groups)):
    kc = 1 + int(np.argmax(M[i] >= 1)) if (M[i] >= 1).any() else None
    if kc:
        a.plot([kc - 0.5, kc - 0.5], [i - 0.45, i + 0.45], color="white", lw=3)
a.set_title("white bar: where the error becomes as large as the signal", loc="right", fontsize=11, color=MUT)
a.set(yticks=range(len(groups)), yticklabels=[g for g, _ in groups], xticks=[1, 4, 8, 16, 32], xticklabels=["1", "4 (forcing)", "8", "16", "32"], xlabel="swirl size: wavenumber  (big  →  tiny)")
a.tick_params(labelsize=12, length=0)
for sp in a.spines.values():
    sp.set_visible(False)
cb = f.colorbar(im, ax=a, pad=0.01, aspect=12)
cb.set_label("error / signal", fontsize=12, color=MUT)
cb.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
cb.set_ticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
cb.ax.tick_params(labelsize=10, colors=MUT)
save(f, "d_heat.png")

# Part II E10: where the error sits (donuts) and memorisation
A10 = J("experiments/e10/results/analysis.json")["models"]
sh = {g: np.mean([[A10[n]["share_eval"][v] for v in ("u", "v", "p", "omega")] for n in A10 if n.startswith(p)], 0) for g, p in (("scratch", "e1_scratch_"), ("pre", "e1_ft_n32_"))}
gap = {g: np.mean([A10[n]["gap"] for n in A10 if n.startswith(p)]) for g, p in (("scratch", "e1_scratch_"), ("pre", "e1_ft_n32_"))}
f, ax = plt.subplots(1, 3, figsize=(9, 3.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.05]})
cols = ["#C9DCF0", "#6FA3D9", GREY, OR]
for a, g, lab in ((ax[0], "scratch", "from scratch"), (ax[1], "pre", "pretrained")):
    a.pie(sh[g], colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    a.text(0, 0.08, f"{100 * sh[g][3]:.0f}%", ha="center", va="center", fontsize=24, fontweight="bold", color=OR)
    a.text(0, -0.22, "in spin ω", ha="center", va="center", fontsize=12, color=MUT)
    a.set_title(lab, fontsize=14, fontweight="bold")
ax[0].legend(["u", "v", "p", "ω"], frameon=False, ncol=4, fontsize=11, loc="upper center", bbox_to_anchor=(1.05, 0.02), handlelength=1)
a = ax[2]
b = a.bar(["from\nscratch", "pre-\ntrained"], [gap["scratch"], gap["pre"]], color=[GREY, BLUE], width=0.55)
for r, v in zip(b, (gap["scratch"], gap["pre"])):
    a.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.1f}×", ha="center", fontsize=18, fontweight="bold")
a.set(ylim=(0, 39), yticks=[])
a.set_title("test error ÷ training error", fontsize=13, fontweight="bold")
a.tick_params(axis="x", labelsize=12, length=0)
for sp in ("top", "right", "left"):
    a.spines[sp].set_visible(False)
save(f, "d_donut.png")

# Part II: what improves the model (one change at a time)
f, a = fig(6.0, 3.4)
items = [("Pushforward\n(long-forecast drift)", 31, 50, OR), ("Two attention upgrades\n(next-step error)", 8.9, 8.9, BLUE), ("Deeper model\n(next-step error)", 6.7, 6.7, BLUE), ("Faster training loop\n(time per step)", 51.7, 51.7, BLUE)]
for i, (nm, lo, hi, c) in enumerate(items):
    if hi > lo:
        a.plot([lo, hi], [i, i], color=c, lw=14, solid_capstyle="round", alpha=0.85)
        a.text(hi + 2, i, f"{lo:.0f}–{hi:.0f}%", va="center", fontsize=15, fontweight="bold", color=c)
    else:
        a.plot([0, hi], [i, i], color=c, lw=3)
        a.scatter([hi], [i], s=160, color=c, zorder=3)
        a.text(hi + 2, i, f"{hi:.0f}%" if hi < 50 else "2.07× faster", va="center", fontsize=15, fontweight="bold", color=c)
a.set(yticks=range(len(items)), yticklabels=[x[0] for x in items], xlim=(0, 88), xticks=[0, 25, 50], ylim=(len(items) - 0.5, -0.6), xlabel="improvement (%)")
a.tick_params(axis="y", labelsize=12.5, length=0)
a.tick_params(axis="x", labelsize=11)
a.spines["left"].set_visible(False)
a.grid(axis="x", color=LIGHT)
a.set_axisbelow(True)
save(f, "d_components.png")

# Part II E6: accuracy on real recordings (dot plot)
f, a = fig(6.0, 2.9)
pts = [("Organisers' FNO\n(sees 20 frames)", 95.47, GREY), ("Our Aurora, real data only\n(sees 2 frames)", 94.71, BLUE), ("Our Aurora, simulation-pretrained\n(sees 2 frames)", 94.81, "#6FA3D9")]
for i, (nm, v, c) in enumerate(pts):
    a.plot([90, v], [i, i], color=LIGHT, lw=3, zorder=1)
    a.scatter([v], [i], s=260, color=c, zorder=3, edgecolor="white", linewidth=2)
    a.text(v + 0.25, i, f"{v:.1f}", va="center", fontsize=16, fontweight="bold", color=INK)
a.set(yticks=range(3), yticklabels=[p[0] for p in pts], xlim=(90, 97), xticks=[90, 92, 94, 96], ylim=(2.6, -0.6), xlabel="accuracy score (axis starts at 90)")
a.tick_params(axis="y", labelsize=12, length=0)
a.tick_params(axis="x", labelsize=11)
a.spines["left"].set_visible(False)
save(f, "d_realdata.png")
print("v3 figures written")
