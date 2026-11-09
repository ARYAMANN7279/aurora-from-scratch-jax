"""E11 figure: per-field one-step error, rollout error by step, and the physics check (results/analysis.json).

usage: python experiments/e11/make_figs.py   (writes figs/e11_omega.png)
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
U = json.load(open(os.path.join(HERE, "results", "analysis.json")))["runs"]
ARMS = {"B": ("B: four outputs", "#9aa5b1"), "W": ("W: ω only, loss on all four", "#2a78d6"),
        "WO": ("WO: ω only, loss on ω", "#eb6834")}
S = (0, 1, 2)
run = lambda a, s: U[f"e11_{a}_s{s}"]

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), layout="constrained")
x = np.arange(5)
for i, (a, (lab, c)) in enumerate(ARMS.items()):
    v = np.array([[run(a, s)["recomputed"]["field_rel_err"][f] for f in ("u", "v", "p", "omega")] + [run(a, s)["rel_err"]] for s in S])
    ax[0].bar(x + (i - 1) * 0.27, v.mean(0), 0.27, yerr=v.std(0), color=c, label=lab)
ax[0].set(xticks=x, xticklabels=["u", "v", "p", "ω", "all four"], ylabel="one-step relative error")
ax[0].set_title("(a) error by field", loc="left", fontweight="bold", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8.5)

k = np.arange(1, 51)
for a, (lab, c) in ARMS.items():
    e = np.mean([run(a, s)["err_by_step"] for s in S], 0)
    ax[1].plot(k, e, color=c, lw=2, label=lab)
ax[1].axvline(20, color="#5d6d7e", lw=0.8, ls=":")
ax[1].set(xlabel="rollout step", ylabel="relative error (84 rollouts)")
ax[1].set_title("(b) error along a rollout", loc="left", fontweight="bold", fontsize=10.5)

for i, (a, (lab, c)) in enumerate(ARMS.items()):
    for j, key in enumerate(("div_ratio", "curl_mismatch")):
        v = np.mean([run(a, s)["recomputed"][key] for s in S])
        ax[2].bar(j + (i - 1) * 0.27, v, 0.27, color=c)
ax[2].set_yscale("log")
ax[2].set(xticks=[0, 1], xticklabels=["divergence ratio", "curl mismatch"], ylabel="(true fields: < 1e-4)", ylim=(1e-8, 1))
ax[2].set_title("(c) the flow's rules", loc="left", fontweight="bold", fontsize=10.5)
for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "e11_omega.png"), dpi=160, facecolor="white")
