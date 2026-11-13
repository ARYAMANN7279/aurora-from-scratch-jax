"""Physics illustrations for the deck, from real data and E11's trained models (normal model vs spin-only).

usage: ../../.venv/bin/python make_physics_figs.py   (writes ../figs/d_fields.png, d_div.png, d_rollsnap.png)
Uses the repository's test trajectory re300_kf4 seed 2 and the E11 weights in experiments/e11/results/params/ (seed 0).
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
R2 = os.path.abspath(os.path.join(HERE, "..", ".."))  # the repository root
OUT = os.path.join(HERE, "..", "figs")
sys.path.insert(0, os.path.join(R2, "src"))
jax.config.update("jax_default_matmul_precision", "highest")
from aurora import harness, model  # noqa: E402
from aurora.embed import VARS  # noqa: E402
from aurora.rollout import rollout  # noqa: E402

INK, BLUE, OR, MUT = "#2C3E50", "#2E75B6", "#CA6F1E", "#5D6D7E"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 13, "text.color": INK})
F = np.load(os.path.join(R2, "data", "stage2", "e1_v1", "re300_kf4", "seed_0002", "frames_64.npy")).astype(np.float64)
N = F.shape[-1]
k = np.fft.fftfreq(N, 1.0 / N)


def ddx(f, axis):
    sh = [1] * f.ndim
    sh[axis] = N
    return np.real(np.fft.ifft(1j * k.reshape(sh) * np.fft.fft(f, axis=axis), axis=axis))


def div(x):  # x: (4, N, N) physical
    return ddx(x[0], -1) + ddx(x[1], -2)


def load(name):
    rows = {json.loads(l)["name"]: json.loads(l) for l in open(os.path.join(R2, "experiments", "e11", "results", "rows.jsonl"))}
    cfg = rows[name]["config"]
    z = np.load(os.path.join(R2, "experiments", "e11", "results", "params", name + ".npz"))
    mc = harness.model_config(cfg["model"])
    if cfg["model"].get("derive_from_omega"):
        mc = mc._replace(derive=(tuple(map(float, z["mean"])), tuple(map(float, z["std"]))))
    _, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32))
    return unravel(jnp.asarray(z["flat"])), mc, z["mean"], z["std"]


def clean(a):
    a.set(xticks=[], yticks=[])
    for s in a.spines.values():
        s.set_visible(False)


# 1. one snapshot: the spin fixes the other three fields
t = 100
fig = plt.figure(figsize=(9.0, 3.3), layout="constrained")
gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 0.35, 1, 1, 1])
a = fig.add_subplot(gs[:, 0])
lim = np.abs(F[t, 3]).max()
a.imshow(F[t, 3], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
a.set_title("spin ω  (predicted)", fontsize=14, fontweight="bold", color=BLUE)
clean(a)
ax = fig.add_subplot(gs[:, 1])
ax.axis("off")
ax.annotate("", xy=(1, 0.5), xytext=(0, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=OR, lw=3, mutation_scale=25))
ax.text(0.5, 0.62, "exact", ha="center", fontsize=12, color=OR, fontweight="bold", transform=ax.transAxes)
for j, (i, lab) in enumerate(((0, "velocity u"), (1, "velocity v"), (2, "pressure p"))):
    a = fig.add_subplot(gs[:, 2 + j])
    lim = np.abs(F[t, i]).max()
    a.imshow(F[t, i], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    a.set_title(lab, fontsize=13, color=INK)
    clean(a)
fig.savefig(os.path.join(OUT, "d_fields.png"), dpi=220, facecolor="white")
plt.close(fig)

# 2. where the flow's rules break: divergence of one next-step prediction
pB, mcB, muB, sdB = load("e11_B_s0")
pW, mcW, muW, sdW = load("e11_W_s0")


def predict(p, mc, mu, sd, i):
    nz = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
    y = np.asarray(model.apply(p, nz(F[i - 1]), nz(F[i]), VARS, 0.2, mc), np.float64)
    return y * sd[:, None, None] + mu[:, None, None]


yB, yW = predict(pB, mcB, muB, sdB, t), predict(pW, mcW, muW, sdW, t)
grad = lambda x: np.sqrt(np.mean(ddx(x[0], -1) ** 2 + ddx(x[0], -2) ** 2 + ddx(x[1], -1) ** 2 + ddx(x[1], -2) ** 2))
dT, dB, dW = div(F[t + 1]), div(yB), div(yW)
lim = np.abs(dB).max()
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.4), layout="constrained")
for a, d, x, lab, col in zip(axs, (dT, dB, dW), (F[t + 1], yB, yW), ("true flow", "normal model", "spin only, physics built in"), (INK, MUT, BLUE)):
    im = a.imshow(d, cmap="PuOr_r", vmin=-lim, vmax=lim, origin="lower")
    r = np.sqrt(np.mean(d**2)) / grad(x)
    a.set_title(f"{lab}\nrule broken by {100 * r:.2g}%" if r > 1e-3 else f"{lab}\nrule holds (< 0.001%)", fontsize=13, color=col, fontweight="bold")
    clean(a)
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
cb.set_label("fluid appearing (+) or vanishing (−)", fontsize=11, color=MUT)
cb.ax.tick_params(labelsize=9, colors=MUT)
fig.savefig(os.path.join(OUT, "d_div.png"), dpi=220, facecolor="white")
plt.close(fig)
print("divergence ratios", [float(np.sqrt(np.mean(d ** 2)) / grad(x)) for d, x in ((dT, F[t + 1]), (dB, yB), (dW, yW))])

# 3. forecasts from frame 50: truth vs the two models, spin at steps 1, 10, 20
s0, steps = 50, (1, 10, 20)


def roll(p, mc, mu, sd):
    nz = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
    y = np.asarray(rollout(p, nz(F[s0 - 1]), nz(F[s0]), n=max(steps), dt=0.2, mc=mc), np.float64)
    return y * sd[None, :, None, None] + mu[None, :, None, None]


RB, RW = roll(pB, mcB, muB, sdB), roll(pW, mcW, muW, sdW)
lim = np.abs(F[s0 + 1:s0 + 21, 3]).max()
fig, axs = plt.subplots(2, 3, figsize=(5.6, 3.8), layout="constrained")
for r, (lab, src, col) in enumerate((("truth", None, INK), ("forecast", RB, OR))):
    for c, st in enumerate(steps):
        a = axs[r, c]
        w = F[s0 + st, 3] if src is None else src[st - 1, 3]
        a.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        clean(a)
        if src is not None:
            e = np.linalg.norm(src[st - 1] - F[s0 + st]) / np.linalg.norm(F[s0 + st])
            a.text(0.03, 0.04, f"error {e:.2f}", transform=a.transAxes, fontsize=11, fontweight="bold", color="white",
                   bbox=dict(facecolor=col, edgecolor="none", pad=2.5, alpha=0.9))
        if r == 0:
            a.set_title(f"step {st}", fontsize=13, fontweight="bold")
        if c == 0:
            a.set_ylabel(lab, fontsize=13, fontweight="bold", color=col)
fig.savefig(os.path.join(OUT, "d_chaos.png"), dpi=220, facecolor="white")
plt.close(fig)
print("rollout errors B", [round(float(np.linalg.norm(RB[s - 1] - F[s0 + s]) / np.linalg.norm(F[s0 + s])), 3) for s in steps],
      "W", [round(float(np.linalg.norm(RW[s - 1] - F[s0 + s]) / np.linalg.norm(F[s0 + s])), 3) for s in steps])
