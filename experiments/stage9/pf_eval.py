"""F1-F5 (handouts/stage09_pushforward.md, section 3): paired rollout comparison.

usage: pf_eval.py   (needs data/stage7/onestep_mse{,_pf1500,_pf3000}_params.npz and their JSONs)
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

MC, DT, NSTEP, TAGS, SEEDS, T0, WIN = model.ModelConfig(), 0.2, 50, ("re100_kf4", "re300_kf4"), (6, 7), (50, 250), 20
MODELS = {"baseline": "onestep_mse", "pf_compute": "onestep_mse_pf1500", "pf_steps": "onestep_mse_pf3000"}
acc = os.path.join(ROOT, "data", "stage2", "acc_v1")
S7 = os.path.join(ROOT, "experiments", "stage7", "results")
_, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32))
roll = jax.jit(partial(rollout, n=NSTEP, dt=DT, mc=MC))

res = {}
for key, name in MODELS.items():
    z = np.load(os.path.join(ROOT, "data", "stage7", f"{name}_params.npz"))
    p = unravel(jnp.asarray(z["flat"]))
    mu, sd = z["mean"][:, None, None], z["std"][:, None, None]
    one = json.load(open(os.path.join(S7, f"{name}.json")))["eval"]
    per = {}
    for tag in TAGS:
        clim_E = clim_Z = 0.0
        raw = {s: np.load(os.path.join(traj_dir(acc, tag, s), "frames_64.npy")) for s in SEEDS}
        clim_E = np.mean([0.5 * (r[:, 0] ** 2 + r[:, 1] ** 2).mean() for r in raw.values()])
        clim_Z = np.mean([0.5 * (r[:, 3] ** 2).mean() for r in raw.values()])
        runs = []
        for s, r in raw.items():
            xn = (r - mu) / sd
            for t0 in T0:
                pred = np.asarray(roll(p, jnp.asarray(xn[t0 - 1], jnp.float32), jnp.asarray(xn[t0], jnp.float32)))
                truth = xn[t0 + 1:t0 + 1 + NSTEP]
                rel = np.sqrt(((pred - truth) ** 2).sum(axis=(1, 2, 3)) / (truth**2).sum(axis=(1, 2, 3)))
                ph = pred * sd + mu
                E = 0.5 * (ph[:, 0] ** 2 + ph[:, 1] ** 2).mean(axis=(1, 2)) / clim_E
                Z = 0.5 * (ph[:, 3] ** 2).mean(axis=(1, 2)) / clim_Z
                stable = bool(np.isfinite(pred).all() and ((E >= 0.5) & (E <= 2)).all() and ((Z >= 0.3) & (Z <= 3)).all())
                runs.append(dict(seed=s, t0=t0, rel_err=rel.tolist(), drift=float(rel[:WIN].mean()), stable=stable))
        per[tag] = runs
    res[key] = dict(file=name, onestep=one, regimes=per)

rows = []
add = lambda i, q, pr, m, h: rows.append(dict(id=i, quantity=q, prediction=pr, measured=m, verdict="HIT" if h else "MISS"))


def paired(a, b, tag):
    d = np.array([x["drift"] - y["drift"] for x, y in zip(res[a]["regimes"][tag], res[b]["regimes"][tag])])
    base = np.mean([x["drift"] for x in res[a]["regimes"][tag]])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d.mean() / base


f1 = {t: paired("baseline", "pf_compute", t) for t in TAGS}
add("F1", "baseline − pushforward (matched compute), drift over steps 1-20", "> 2 SE, each regime",
    "; ".join(f"{t}: {m:.4f} ± {se:.4f}" for t, (m, se, _) in f1.items()), all(m > 2 * se for m, se, _ in f1.values()))
add("F2", "relative drift reduction (matched compute)", "5-30%",
    "; ".join(f"{t}: {100 * r:.1f}%" for t, (_, _, r) in f1.items()), all(0.05 <= r <= 0.30 for _, _, r in f1.values()))
f3 = {t: res["pf_compute"]["onestep"][t]["rel_err"] / res["baseline"]["onestep"][t]["rel_err"] for t in TAGS}
add("F3", "one-step error, pushforward / baseline", "in [0.9, 1.5]", "; ".join(f"{t}: {v:.2f}" for t, v in f3.items()),
    all(0.9 <= v <= 1.5 for v in f3.values()))
dm = lambda k, t: float(np.mean([x["drift"] for x in res[k]["regimes"][t]]))
add("F4", "drift: matched-steps < matched-compute pushforward", "each regime",
    "; ".join(f"{t}: {dm('pf_steps', t):.4f} vs {dm('pf_compute', t):.4f}" for t in TAGS),
    all(dm("pf_steps", t) < dm("pf_compute", t) for t in TAGS))
st = {k: all(r["stable"] for t in TAGS for r in res[k]["regimes"][t]) for k in ("pf_compute", "pf_steps")}
add("F5", "stability (Stage 8 Z1 criterion), pushforward models", "HIT", str(st), all(st.values()))
out = dict(models=res, drift_window=WIN, predictions=rows,
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "pushforward.json"), "w"), indent=1)
print("| # | Quantity | Prediction | Measured | Verdict |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['id']} | {r['quantity']} | {r['prediction']} | {r['measured']} | **{r['verdict']}** |")
