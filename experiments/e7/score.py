"""Score E7 (handouts/e7_attention_rpb.md, section 6) from results/rows.jsonl.

err = one-step relative error on the evaluation trajectories (metrics["val/re300_kf4"]["rel_err"]);
drift = its 20-step rollout drift. Paired by seed; mean and SE over seeds (ddof 1). A prediction is scored
only when both arms have all 3 seeds. For a seed with several rows, the last counts.
E7-P1  mean adv >= 0.03 and mean adv > 2 SE, with adv = 1 - err(R) / err(B)
E7-P2  mean drift advantage > 0, with 1 - drift(R) / drift(B)
E7-P3  |mean wall(R) / mean wall(B) - 1| < 0.10
E1's own scratch rows (lab card, the same config as B) are reported alongside as a cross-device check.
usage: score.py   (writes results/analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
KEY = "val/re300_kf4"

rows = {}
p = os.path.join(RES, "rows.jsonl")
for line in (open(p) if os.path.exists(p) else []):
    r = json.loads(line)
    arm, seed = r["name"].split("_")[1], int(r["name"].split("_s")[-1])
    rows.setdefault(arm, {})[seed] = r


def get(arm, field):
    pick = {"err": lambda r: r["metrics"][KEY]["rel_err"], "drift": lambda r: r["metrics"][KEY]["rollout_drift"],
            "ratio": lambda r: r["metrics"][KEY]["ratio"], "wall": lambda r: r["wall_s"]}[field]
    return {s: pick(r) for s, r in sorted(rows.get(arm, {}).items())}


def mean_se(x):
    x = np.asarray(list(x), float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)


full = all(set(SEEDS) <= set(rows.get(a, {})) for a in ("B", "R"))
summary = {a: {f: dict(per_seed=get(a, f), mean_se=mean_se(get(a, f).values()))
               for f in ("err", "drift", "ratio", "wall")} for a in ("B", "R") if rows.get(a)}
P, detail = {"E7-P1": None, "E7-P2": None, "E7-P3": None}, {}
if full:
    B, R = get("B", "err"), get("R", "err")
    m, se = mean_se(1 - R[s] / B[s] for s in SEEDS)
    detail["E7-P1"] = dict(adv=m, se=se)
    P["E7-P1"] = m >= 0.03 and m > 2 * se
    Bd, Rd = get("B", "drift"), get("R", "drift")
    m, se = mean_se(1 - Rd[s] / Bd[s] for s in SEEDS)
    detail["E7-P2"] = dict(drift_adv=m, se=se)
    P["E7-P2"] = m > 0
    ratio = float(np.mean(list(get("R", "wall").values())) / np.mean(list(get("B", "wall").values())))
    detail["E7-P3"] = dict(wall_R_over_B=ratio)
    P["E7-P3"] = abs(ratio - 1) < 0.10
e1 = {}
e1p = os.path.join(HERE, "..", "e1", "results", "all.jsonl")
for line in (open(e1p) if os.path.exists(e1p) else []):
    r = json.loads(line)
    if (r.get("name") or "").startswith("e1_scratch_s"):
        e1[r["name"]] = dict(err=r["metrics"][KEY]["rel_err"], drift=r["metrics"][KEY]["rollout_drift"])
out = dict(arms=summary, predictions=P, detail=detail, e1_scratch_lab_card=e1)
os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, detail=detail), indent=1))
