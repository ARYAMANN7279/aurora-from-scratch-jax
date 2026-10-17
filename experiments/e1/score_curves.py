"""Score E1 follow-up F1 (handouts/e1_volume_diversity.md, section F1) and draw the training-curve figure.

usage: score_curves.py --results experiments/e1/results/curves_rows.jsonl
Each rerun is matched to its E1 original by config hash (the rerun's config without `curve_every`, and for a
fine-tune with the original pretrain's hash as `init_from`). Writes results/f1_analysis.json and figs/e1_curves.png.
F1-P1: every rerun's final one-step error is within 2% of its original's.
F1-P2: every scratch rerun ends within 2% of the minimum of its own validation curve.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from aurora.harness import config_hash  # noqa: E402
from e1_design import run_configs  # noqa: E402
from make_curves import curves  # noqa: E402

KEY = {"pre": "test/re300_kf4", "ft": "val/re300_kf4", "scratch": "val/re300_kf4"}

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", required=True)
ap.add_argument("--originals", default=os.path.join(HERE, "results", "all.jsonl"))
args = ap.parse_args()
rows = {}
for path in args.results:
    rows.update({r["config_hash"]: r for r in map(json.loads, filter(str.strip, open(path)))})
orig = {r["config_hash"]: r for r in map(json.loads, filter(str.strip, open(args.originals)))}
pre, ft, sc = run_configs(config_hash)
by_name = {c["name"]: c for c in pre + ft + sc}

out, curves_by = {}, {}
for c in curves():
    arm = c["name"].split("_")[1]
    o_name = c["name"].replace("e1c_", "e1_")
    o_cfg = by_name[o_name]
    stripped = copy.deepcopy(c)
    stripped["train"].pop("curve_every")
    stripped["train"]["init_from"] = o_cfg["train"]["init_from"]
    assert config_hash(stripped) == config_hash(o_cfg), c["name"]
    r, o = rows[config_hash(c)], orig[config_hash(o_cfg)]
    e_new, e_old = r["metrics"][KEY[arm]]["rel_err"], o["metrics"][KEY[arm]]["rel_err"]
    curve = [x[KEY[arm]] for x in r["curve"]]
    out[c["name"]] = dict(rel_err=e_new, original=e_old, rel_change=e_new / e_old - 1, curve_min=min(curve),
                          end_over_min=e_new / min(curve) - 1, drift=r["metrics"][KEY[arm]].get("rollout_drift"),
                          original_drift=o["metrics"][KEY[arm]].get("rollout_drift"))
    curves_by[c["name"]] = r["curve"]
v = {"F1-P1": all(abs(x["rel_change"]) <= 0.02 for x in out.values()),
     "F1-P2": all(x["end_over_min"] <= 0.02 for k, x in out.items() if "_scratch_" in k)}
res = dict(runs=out, verdicts={k: "HIT" if x else "MISS" for k, x in v.items()})
with open(os.path.join(HERE, "results", "f1_analysis.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res["verdicts"]), {k: round(x["rel_change"], 4) for k, x in out.items()})

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"n2_T32": "#2a6fdb", "n32_T32": "#e8743b", "scratch": "#19a979"}
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for name, cv in curves_by.items():
    arm = name.split("_")[1]
    cell = "scratch" if arm == "scratch" else name.split("_", 2)[2].rsplit("_s", 1)[0]
    st = np.array([x["step"] for x in cv])
    kw = dict(color=COL[cell], lw=1.2, alpha=0.85, label=None if not name.endswith("_s0") else
              {"n2_T32": "n = 2, T = 32", "n32_T32": "n = 32, T = 32", "scratch": "scratch"}[cell])
    if arm == "pre":
        ax[0].plot(st, [x[KEY[arm]] for x in cv], **kw)
    elif arm == "ft":
        ax[1].plot(st, [x[KEY[arm]] for x in cv], **kw)
    else:
        ax[1].plot(st, [x[KEY[arm]] for x in cv], ls="--", **kw)
        ax[2].plot(st[1:], [x["train_loss"] for x in cv[1:]], ls="--", **kw)
    if arm != "scratch":
        ax[2].plot(st[1:] + (20_000 if arm == "ft" else 0), [x["train_loss"] for x in cv[1:]], **kw)
ax[0].set(title="Pretraining: zero-shot error on the target", xlabel="pretraining step", ylabel="one-step relative error")
ax[1].set(title="Target: fine-tuning (solid) vs scratch (dashed)", xlabel="step on the target", yscale="log", xscale="symlog",
          ylabel="one-step relative error")
ax[2].set(title="Training loss (normalised MSE)", xlabel="step (fine-tuning after step 20,000)", yscale="log")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
fig.suptitle("E1 follow-up F1: training curves, target (Re, k_f) = (300, 4), evaluation trajectories 2-5, seeds 0-2",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "e1_curves.png"), dpi=150)
