"""Z1-Z4 (handouts/stage08_rollout.md, section 3): 50-step rollouts of the trained Stage 7 models.

usage: rollout_eval.py          (needs data/stage7/onestep_{mse,mae}_params.npz and their JSONs)
"""
import json
import os
import sys
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aurora import model  # noqa: E402
from aurora.rollout import rollout  # noqa: E402
from kdata import traj_dir  # noqa: E402

MC, DT, NSTEP, TAGS, SEEDS, T0 = model.ModelConfig(), 0.2, 50, ("re100_kf4", "re300_kf4"), (6, 7), (50, 250)
acc = os.path.join(ROOT, "data", "stage2", "acc_v1")
S7 = os.path.join(ROOT, "experiments", "stage7", "results")
template = model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32)
_, unravel = jax.flatten_util.ravel_pytree(template)
roll = jax.jit(partial(rollout, n=NSTEP, dt=DT, mc=MC))

out = {"models": {}}
for name in ("mse", "mae"):
    z = np.load(os.path.join(ROOT, "data", "stage7", f"onestep_{name}_params.npz"))
    p = unravel(jnp.asarray(z["flat"]))
    mu, sd = z["mean"][:, None, None], z["std"][:, None, None]
    one = json.load(open(os.path.join(S7, f"onestep_{name}.json")))
    res = {"onestep": one["eval"], "regimes": {}}
    for tag in TAGS:
        raw = {s: np.load(os.path.join(traj_dir(acc, tag, s), "frames_64.npy")) for s in SEEDS}
        clim_E = np.mean([0.5 * (r[:, 0] ** 2 + r[:, 1] ** 2).mean() for r in raw.values()])
        clim_Z = np.mean([0.5 * (r[:, 3] ** 2).mean() for r in raw.values()])
        clim_mean = np.mean(np.concatenate(list(raw.values())), axis=0)
        runs = []
        for s, r in raw.items():
            xn = (r - mu) / sd
            for t0 in T0:
                pred = np.asarray(roll(p, jnp.asarray(xn[t0 - 1], jnp.float32), jnp.asarray(xn[t0], jnp.float32)))
                pred_phys = pred * sd + mu
                truth = xn[t0 + 1:t0 + 1 + NSTEP]
                rel = lambda a: np.sqrt(((a - truth) ** 2).sum(axis=(1, 2, 3)) / (truth**2).sum(axis=(1, 2, 3)))
                E = 0.5 * (pred_phys[:, 0] ** 2 + pred_phys[:, 1] ** 2).mean(axis=(1, 2))
                Z = 0.5 * (pred_phys[:, 3] ** 2).mean(axis=(1, 2))
                runs.append(dict(seed=s, t0=t0, finite=bool(np.isfinite(pred).all()),
                                 rel_err=rel(pred).tolist(),
                                 persistence=rel(np.repeat(xn[t0][None], NSTEP, 0)).tolist(),
                                 climatology=rel(np.repeat(((clim_mean - mu) / sd)[None], NSTEP, 0)).tolist(),
                                 E_ratio=(E / clim_E).tolist(), Z_ratio=(Z / clim_Z).tolist()))
        res["regimes"][tag] = dict(runs=runs, clim_E=float(clim_E), clim_Z=float(clim_Z))
    out["models"][name] = res

# ---- predictions
rows = []
add = lambda i, q, pr, m, h: rows.append(dict(id=i, quantity=q, prediction=pr, measured=m, verdict="HIT" if h else "MISS"))
z1 = {}
for name, res in out["models"].items():
    ok, eminmax, zminmax = True, [9e9, -9e9], [9e9, -9e9]
    for reg in res["regimes"].values():
        for r in reg["runs"]:
            ok &= r["finite"] and all(0.5 <= e <= 2 for e in r["E_ratio"]) and all(0.3 <= z <= 3 for z in r["Z_ratio"])
            eminmax = [min(eminmax[0], min(r["E_ratio"])), max(eminmax[1], max(r["E_ratio"]))]
            zminmax = [min(zminmax[0], min(r["Z_ratio"])), max(zminmax[1], max(r["Z_ratio"]))]
    z1[name] = (ok, eminmax, zminmax)
add("Z1", "finite; E ratio in [0.5,2], Z ratio in [0.3,3] at all 50 steps", "HIT for both models",
    "; ".join(f"{n}: E [{v[1][0]:.2f}, {v[1][1]:.2f}], Z [{v[2][0]:.2f}, {v[2][1]:.2f}]" for n, v in z1.items()),
    all(v[0] for v in z1.values()))
ratio = {t: out["models"]["mae"]["onestep"][t]["rel_err"] / out["models"]["mse"]["onestep"][t]["rel_err"] for t in TAGS}
add("Z2", "one-step error, MAE / MSE model", "in [0.7, 1.3]", ", ".join(f"{t}: {v:.2f}" for t, v in ratio.items()),
    all(0.7 <= v <= 1.3 for v in ratio.values()))
m = lambda name, tag, key, k: float(np.mean([r[key][k] for r in out["models"][name]["regimes"][tag]["runs"]]))
z3 = {(n, t): (m(n, t, "rel_err", 4), m(n, t, "persistence", 4)) for n in out["models"] for t in TAGS}
add("Z3", "step-5 error: model < persistence", "each regime, each model",
    "; ".join(f"{n}/{t}: {a:.3f} vs {b:.3f}" for (n, t), (a, b) in z3.items()), all(a < b for a, b in z3.values()))
z4 = {(n, t): m(n, t, "rel_err", NSTEP - 1) for n in out["models"] for t in TAGS}
add("Z4", "step-50 error", "≤ 1.6", "; ".join(f"{n}/{t}: {v:.3f}" for (n, t), v in z4.items()), all(v <= 1.6 for v in z4.values()))
out["predictions"] = rows
out["command"] = " ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "rollout.json"), "w"), indent=1)
print("| # | Quantity | Prediction | Measured | Verdict |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['id']} | {r['quantity']} | {r['prediction']} | {r['measured']} | **{r['verdict']}** |")
