"""Score E6 follow-up F1 (handouts/e6_realpde_sim2real.md, section 10) from results/rows_f1.jsonl, against
the E6 rows in results/rows.jsonl (S, and PD and PA with the 2000-step fine-tune).

err = the rollout rel_l2 over the 900 re_lohi windows. Paired by seed; mean and SE over seeds (ddof 1).
A prediction is scored only when every arm it needs has all 3 seeds.
E6F-P1  |mean adv(PDm)| < 0.05, with adv = 1 - err(PDm) / err(S)
E6F-P2  mean(err(PDm) - err(PAm)) <= 2 SE   (PAm not below PDm by more than 2 SE)
E6F-P3  mean(1 - err(PDm) / err(PD)) >= 0.10
usage: score_f1.py   (writes results/analysis_f1.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)

rows = {}
for f in ("rows.jsonl", "rows_f1.jsonl"):
    p = os.path.join(RES, f)
    for line in (open(p) if os.path.exists(p) else []):
        r = json.loads(line)
        arm = {("PD", "finetune_matched"): "PDm", ("PA", "finetune_matched"): "PAm"}.get((r["arm"], r["stage"]))
        arm = arm or {("PD", "finetune"): "PD", ("PA", "finetune"): "PA", ("S", None): "S"}.get((r["arm"], r["stage"]))
        if arm:
            rows.setdefault(arm, {})[r["seed"]] = r


def err(arm, part="rollout", key="rel_l2"):
    return {s: r["eval"][part][key] for s, r in sorted(rows.get(arm, {}).items())}


def mean_se(x):
    x = np.asarray(list(x), float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)


def full(*arms):
    return all(set(SEEDS) <= set(rows.get(a, {})) for a in arms)


def paired(a, b, f):
    ea, eb = err(a), err(b)
    return [f(ea[s], eb[s]) for s in SEEDS]


summary = {a: dict(rel_l2_per_seed=err(a), rel_l2=mean_se(err(a).values()),
                   subscore_rel_l2=mean_se(err(a, "scores", "rel_l2").values()),
                   tke_subscore=mean_se(err(a, "scores", "tke").values()),
                   one_step_vs_persistence=mean_se(err(a, "one_step", "ratio").values()))
           for a in ("S", "PD", "PA", "PDm", "PAm") if rows.get(a)}
P, detail = {"E6F-P1": None, "E6F-P2": None, "E6F-P3": None}, {}
if full("S", "PDm"):
    m, se = mean_se(paired("PDm", "S", lambda x, s: 1 - x / s))
    detail["E6F-P1"] = dict(adv_PDm=m, se=se)
    P["E6F-P1"] = abs(m) < 0.05
if full("PDm", "PAm"):
    m, se = mean_se(paired("PDm", "PAm", lambda d, a: d - a))
    detail["E6F-P2"] = dict(err_PDm_minus_PAm=m, se=se)
    P["E6F-P2"] = m <= 2 * se
if full("PDm", "PD"):
    m, se = mean_se(paired("PDm", "PD", lambda x, y: 1 - x / y))
    detail["E6F-P3"] = dict(gain_PDm_over_PD=m, se=se)
    P["E6F-P3"] = m >= 0.10
if full("S", "PAm"):
    detail["adv_PAm"] = dict(zip(("mean", "se"), mean_se(paired("PAm", "S", lambda x, s: 1 - x / s))))
out = dict(arms=summary, predictions=P, detail=detail)
with open(os.path.join(RES, "analysis_f1.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, detail=detail), indent=1))
