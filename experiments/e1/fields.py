"""Exploratory E1 diagnostics and fixed-example contours for the report feedback.

Uses original E1 checkpoints, never fits to evaluation data. Run on CPU or through glane3.sh.
Outputs results/fields.json, figs/e1_rollout_diagnosis.png, e1_contours_n{2,32}.png,
and e1_errors.png. Raw arrays remain under data/feedback_scratch/.
"""
import json
import os
from pathlib import Path
import sys
from functools import partial

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from aurora import harness, model  # noqa: E402
from aurora.rollout import rollout  # noqa: E402
from e1_design import run_configs  # noqa: E402
from kdata import traj_dir  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    pre, ft, sc = run_configs(harness.config_hash)
    chosen = [c for c in ft if any(f"_n{n}_T32_" in c["name"] for n in (2, 8, 32))] + sc
    chosen += [c for c in pre if c["train"]["seed"] == 0 and
               any(f"_n{n}_T32_" in c["name"] for n in (2, 32))]
    originals = {r["config_hash"]: r for r in map(json.loads, (HERE / "results/all.jsonl").read_text().splitlines())}
    raw = np.stack([np.load(Path(traj_dir(str(ROOT / "data/stage2/e1_v1"), "re300_kf4", s)) /
                           "frames_64.npy", mmap_mode="r")[49:71] for s in (2, 3, 4, 5)])
    mc = harness.model_config(chosen[0]["model"])
    template = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32)
    unravel = jax.flatten_util.ravel_pytree(template)[1]
    roll = jax.jit(jax.vmap(partial(rollout, n=20, dt=0.2, mc=mc), in_axes=(None, 0, 0)))
    records, examples = {}, {}
    for cfg in chosen:
        h, name = harness.config_hash(cfg), cfg["name"]
        z = np.load(ROOT / "data/runs" / h / "params.npz")
        p = unravel(jnp.asarray(z["flat"]))
        mu, sd = z["mean"][None, None, :, None, None], z["std"][None, None, :, None, None]
        v = ((raw - mu) / sd).astype(np.float32)
        prediction = np.asarray(roll(p, jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1])))
        truth = v[:, 2:]
        # Float64 reductions keep the diagnostic identity accurate without changing predictions.
        p64, t64 = prediction.astype(np.float64), truth.astype(np.float64)
        axes = (2, 3, 4)
        tt, pp, dot = (t64**2).sum(axes), (p64**2).sum(axes), (p64 * t64).sum(axes)
        err = np.sqrt(((p64 - t64)**2).sum(axes) / tt)
        amp, cosine = np.sqrt(pp / tt), dot / np.sqrt(pp * tt)
        key = "test/re300_kf4" if "_pre_" in name else "val/re300_kf4"
        records[name] = dict(config_hash=h, checkpoint="original E1", n_rollouts=4,
                             model_seed=cfg["train"]["seed"],
                             err=err.mean(0).tolist(), amp=amp.mean(0).tolist(), cosine=cosine.mean(0).tolist(),
                             drift=float(err.mean()), original_drift=originals[h]["metrics"][key]["rollout_drift"],
                             identity_max_residual=float(np.max(np.abs(err**2 - (1 + amp**2 - 2*amp*cosine)))))
        if cfg["train"]["seed"] == 0:
            examples[name.rsplit("_s", 1)[0]] = (prediction * sd + mu)[0, :, 3]
        print(name, "drift CPU/original", round(float(err.mean()), 5),
              round(records[name]["original_drift"], 5), flush=True)
    out = dict(description="Exploratory: original E1 weights, trajectory seeds 2–5, start frame 50, 20 steps",
               coordinates="error, amplitude and uncentred cosine in each model's training-normalised variables",
               contour_example=dict(model_seed=0, trajectory_seed=2, frame=50, start_time=60.0,
                                    variable="physical vorticity", horizons=[1, 5, 20]), runs=records)
    (HERE / "results/fields.json").write_text(json.dumps(out, indent=1) + "\n")
    examples["truth"] = raw[0, 2:, 3]
    scratch = ROOT / "data/feedback_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch / "e1_fields.npz", **examples)
    plot_diagnosis(records)
    plot_contours(examples)


def plot_diagnosis(records):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9), layout="constrained")
    styles = [("e1_ft_n2_T32", "fine-tune, n = 2", "#2674ce"),
              ("e1_ft_n8_T32", "fine-tune, n = 8", "#e97832"),
              ("e1_ft_n32_T32", "fine-tune, n = 32", "#9266b1"),
              ("e1_scratch", "scratch", "#249a75")]
    x = np.arange(1, 21)
    for prefix, label, col in styles:
        runs = [r for name, r in records.items() if name.rsplit("_s", 1)[0] == prefix]
        for ax, key in zip(axs, ("err", "amp", "cosine")):
            a = np.array([r[key] for r in runs])
            avg, se = a.mean(0), a.std(0, ddof=1)/np.sqrt(3)
            ax.plot(x, avg, label=label, color=col)
            ax.fill_between(x, avg-se, avg+se, alpha=.12, color=col)
    for ax, title in zip(axs, ("Relative error", "Amplitude / truth", "Uncentred alignment")):
        ax.set(title=title, xlabel="forecast step")
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=.2)
    axs[1].axhline(1, color="gray", ls="--", lw=.8)
    axs[2].set_ylim(-.05, 1.05)
    axs[0].legend(fontsize=11, frameon=False)
    fig.suptitle("E1 diagnosis: T = 32, original frame-50 rollouts; mean ± SE across 3 model seeds", fontsize=12)
    fig.savefig(HERE / "figs/e1_rollout_diagnosis.png", dpi=160)
    plt.close(fig)


def plot_contours(ex):
    horizons = (1, 5, 20)
    # The truth sets the scale so its structure remains readable when a rollout grows.
    # Both figures use the same limits; colourbar triangles explicitly mark clipped values.
    limit = float(np.max(np.abs(ex["truth"][[0, 4, 19]])))
    levels = np.linspace(-limit, limit, 25)
    coord = np.linspace(0, 2*np.pi, 64, endpoint=False)
    for n in (2, 32):
        rows = [("truth", "Truth"), (f"e1_pre_n{n}_T32", "Zero-shot"),
                (f"e1_ft_n{n}_T32", "Fine-tuned"), ("e1_scratch", "Scratch")]
        fig, axs = plt.subplots(4, 3, figsize=(8, 7.6), layout="constrained")
        for i, (key, label) in enumerate(rows):
            for j, k in enumerate(horizons):
                a = axs[i, j]
                field = ex[key][k-1]
                m = a.contourf(coord, coord, field, levels=levels, cmap="RdBu_r", extend="both")
                if i == 0:
                    a.set_title(f"Step {k}, t = {60 + .2*k:.1f}", fontsize=9)
                else:
                    e = np.linalg.norm(field-ex["truth"][k-1])/np.linalg.norm(ex["truth"][k-1])
                    a.text(.02, .98, f"ω error {e:.2f}", transform=a.transAxes, va="top", fontsize=7,
                           bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
                a.set_aspect("equal")
                a.set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
                a.set_yticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
                a.tick_params(labelsize=7)
                if j == 0:
                    a.set_ylabel(label + "\ny", fontsize=9)
                if i == 3:
                    a.set_xlabel("x", fontsize=8)
        fig.colorbar(m, ax=axs, shrink=.65, label="Physical ω; truth range, triangles mark clipped predictions")
        fig.suptitle(f"E1: n = {n}, T = 32; model seed 0, held-out trajectory 2, start t = 60", fontsize=10)
        fig.savefig(HERE / f"figs/e1_contours_n{n}.png", dpi=170)
        plt.close(fig)
    rows = [("e1_pre_n2_T32", "Zero-shot, n = 2"), ("e1_ft_n2_T32", "Fine-tuned, n = 2"),
            ("e1_pre_n32_T32", "Zero-shot, n = 32"), ("e1_ft_n32_T32", "Fine-tuned, n = 32"),
            ("e1_scratch", "Scratch")]
    emax = max(float(np.max(np.abs(ex[k][19]-ex["truth"][19]))) for k, _ in rows)
    fig, axs = plt.subplots(1, 5, figsize=(11.5, 3), layout="constrained")
    for a, (key, title) in zip(axs, rows):
        m = a.contourf(coord, coord, np.abs(ex[key][19]-ex["truth"][19]),
                       levels=np.linspace(0, emax, 21), cmap="magma")
        a.set(title=title, aspect="equal", xticks=[], yticks=[])
        a.title.set_fontsize(8)
    fig.colorbar(m, ax=axs, shrink=.7, label="|ω prediction − truth|")
    fig.suptitle("E1: absolute vorticity error at step 20, same fixed example and shared error scale", fontsize=10)
    fig.savefig(HERE / "figs/e1_errors.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
