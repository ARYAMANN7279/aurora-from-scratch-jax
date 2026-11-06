"""Score E11's predictions P1-P6 (handouts/e11_omega_only.md) from the Kaggle job's output.

usage: python experiments/e11/score.py   (reads results/rows.jsonl and results/params/<name>.npz, the
       trained weights, which stay out of git; writes results/analysis.json)
Per-field one-step errors and the physics measures are recomputed from the weights on the evaluation
pairs, with each arm's own output path (derived fields for W and WO).
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "e10"))
jax.config.update("jax_default_matmul_precision", "highest")

from aurora import harness, model  # noqa: E402
from aurora.embed import VARS  # noqa: E402
import diag  # noqa: E402

R = os.path.join(HERE, "results")
rows = {r["name"]: r for r in map(json.loads, open(os.path.join(R, "rows.jsonl")))}
e8 = {r["name"]: r for r in map(json.loads, open(os.path.join(ROOT, "experiments", "e8", "results", "rows.jsonl")))}
M = lambda r: r["metrics"]["val/re300_kf4"]
SEEDS = (0, 1, 2)


def fields(name):
    """Per-field one-step relative error and E10's physics measures on the evaluation pairs."""
    r = rows[name]
    z = np.load(os.path.join(R, "params", name + ".npz"))
    mu, sd = z["mean"], z["std"]
    mc = harness.model_config(r["config"]["model"])
    if r["config"]["model"].get("derive_from_omega"):
        mc = mc._replace(derive=(tuple(float(x) for x in mu), tuple(float(x) for x in sd)))
    _, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32))
    p = unravel(jnp.asarray(z["flat"]))
    norm = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
    xm, x0, xp = diag.pairs(diag.EVAL, norm)
    f = jax.jit(jax.vmap(lambda a, b: model.apply(p, a, b, VARS, diag.DT, mc)))
    y = np.concatenate([np.asarray(f(xm[i:i + 16], x0[i:i + 16])) for i in range(0, xm.shape[0], 16)])
    t = np.asarray(xp, np.float64)
    e2, t2 = ((y - t) ** 2).sum(axis=(0, 2, 3)), (t**2).sum(axis=(0, 2, 3))
    D, C = diag.physics(y * sd[None, :, None, None] + mu[None, :, None, None])
    return dict(rel_err=float(np.sqrt(e2.sum() / t2.sum())), field_rel_err=dict(zip(VARS, np.sqrt(e2 / t2).tolist())),
                div_ratio=D, curl_mismatch=C)


out = {"runs": {}}
for n, r in sorted(rows.items()):
    m = M(r)
    out["runs"][n] = dict(rel_err=m["rel_err"], drift20=m["rollout_drift_many"], drift50=m["rollout_drift_many_long"],
                          err_by_step=m["rollout_err_by_step"], train_s=r["train_s"], **{"recomputed": fields(n)})
    print(n, round(m["rel_err"], 5), round(out["runs"][n]["recomputed"]["rel_err"], 5), flush=True)
U = out["runs"]
g = lambda arm, s, k: U[f"e11_{arm}_s{s}"][k]
f = lambda arm, s, v: U[f"e11_{arm}_s{s}"]["recomputed"]["field_rel_err"][v]
adv = lambda Y, X, k: [1 - g(Y, s, k) / g(X, s, k) for s in SEEDS]
ms = lambda x: f"{100 * np.mean(x):.1f}% ± {100 * np.std(x, ddof=1) / np.sqrt(len(x)):.1f}%"
out["advantages"] = {f"{Y} over {X}, {k}": ms(adv(Y, X, k)) for Y, X in (("W", "B"), ("WO", "B"), ("WO", "W"))
                     for k in ("rel_err", "drift20", "drift50")}
out["e8_B"] = [M(e8[f"e8_B_s{s}"])["rel_err"] for s in SEEDS]
v = {
    "P1": all(a >= 0.05 for a in adv("W", "B", "rel_err")),
    "P2": all(a > 0 for a in adv("W", "B", "drift20")),
    "P3": all(f("WO", s, "omega") < f("W", s, "omega") and f("W", s, "u") < f("WO", s, "u") for s in SEEDS),
    "P4": np.mean(adv("WO", "W", "rel_err")) <= 0.02,
    "P5": all(abs(g("B", s, "rel_err") / out["e8_B"][s] - 1) < 0.01 for s in SEEDS),
    "P6": abs(np.mean([g("W", s, "train_s") for s in SEEDS]) / np.mean([g("B", s, "train_s") for s in SEEDS]) - 1) <= 0.10,
}
out["verdicts"] = {k: "HIT" if x else "MISS" for k, x in v.items()}
json.dump(out, open(os.path.join(R, "analysis.json"), "w"), indent=1)
print(json.dumps(out["verdicts"]))
print(json.dumps(out["advantages"], indent=1))
for arm in ("B", "W", "WO"):
    print(arm, {k: round(np.mean([f(arm, s, k) for s in SEEDS]), 4) for k in VARS},
          "div", round(np.mean([U[f"e11_{arm}_s{s}"]["recomputed"]["div_ratio"] for s in SEEDS]), 6),
          "curl", round(np.mean([U[f"e11_{arm}_s{s}"]["recomputed"]["curl_mismatch"] for s in SEEDS]), 6),
          "train_s", round(np.mean([g(arm, s, "train_s") for s in SEEDS])))
print("B vs E8 B:", [round(g("B", s, "rel_err") / out["e8_B"][s] - 1, 5) for s in SEEDS])
