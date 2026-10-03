"""E8 figure from results/analysis.json and results/amplitude.json: the rollout error e(k) of each arm, the advantage
over B at each step, and the amplitude and correlation of B's and BP's rollouts (exploratory, amplitude.py).
usage: plot_e8.py   (writes figs/e8_rollouts.png)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
a = json.load(open(os.path.join(RES, "analysis.json")))
amp = json.load(open(os.path.join(RES, "amplitude.json")))
k = np.arange(1, 51)
style = {"B": ("k", "-"), "R": ("tab:blue", "-"), "Q": ("tab:green", "-"), "BP": ("tab:red", "--"),
         "RP": ("tab:purple", "--"), "BS": ("0.6", ":")}
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for arm, (c, ls) in style.items():
    ax[0].plot(k, a["curves"][arm], color=c, ls=ls, label=arm)
ax[0].axhline(np.sqrt(2), color="0.8", lw=0.8)
ax[0].set(xlabel="rollout step k", ylabel="relative error e(k)", title="error along the rollout (84 rollouts, 3 seeds)")
ax[0].legend(fontsize=8)
B = np.array(a["curves"]["B"])
for arm in ("R", "Q", "BP", "RP"):
    c, ls = style[arm]
    ax[1].plot(k, 100 * (1 - np.array(a["curves"][arm]) / B), color=c, ls=ls, label=arm)
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set(xlabel="rollout step k", ylabel="advantage over B (%)", title="1 - e_arm(k) / e_B(k)")
ax[1].legend(fontsize=8)
for arm in ("B", "BP"):
    c, ls = style[arm]
    runs = [d for n, d in amp.items() if n.split("_")[1] == arm]
    ax[2].plot(k, np.mean([d["amp"] for d in runs], 0), color=c, ls=ls, label=f"{arm} amplitude")
    ax[2].plot(k, np.mean([d["corr"] for d in runs], 0), color=c, ls=ls, alpha=0.5, label=f"{arm} correlation")
ax[2].set(xlabel="rollout step k", ylabel="ratio / correlation", title="amplitude and correlation (exploratory)")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e8_rollouts.png"), dpi=130)
print("wrote figs/e8_rollouts.png")
