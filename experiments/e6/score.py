"""Score E6 (handouts/e6_realpde_sim2real.md, sections 5-6 and 8) from results/rows.jsonl.

err = the rollout rel_l2 over the 900 re_lohi windows (each row's eval.rollout.rel_l2). ZD and ZA are
the PD and PA pretrain rows (amendment A1). For an arm and seed with several rows, the last counts.
A prediction is scored only when every arm it needs has all 3 seeds; otherwise it stays unscored.
adv(P) = 1 - err(P) / err(S), paired by seed; mean and SE over seeds (SE = sd / sqrt(n), ddof 1).
E6-P1  adv(PD) >= 0.10 and adv(PD) > 2 SE
E6-P2  err(PD) - err(PA), paired by seed: mean > 2 SE
E6-P3  mean err(ZA) < mean err(S)
E6-P4  1 - err(PD) / err(ZD), paired by seed: mean >= 0.10
E6-P5  every arm's mean rel_l2 subscore is below sim_real_fno's, re-measured in results/fno_baseline.json
usage: score.py   (writes results/analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
ARMS = ("S", "PD", "PA", "ZD", "ZA")

rows = {}
p = os.path.join(RES, "rows.jsonl")
for line in (open(p) if os.path.exists(p) else []):
    r = json.loads(line)
    arm = {("PD", "pretrain"): "ZD", ("PA", "pretrain"): "ZA"}.get((r["arm"], r["stage"]), r["arm"])
    rows.setdefault(arm, {})[r["seed"]] = r


def per_seed(arm, part="rollout", key="rel_l2"):
    return {s: r["eval"][part][key] for s, r in sorted(rows.get(arm, {}).items())}


def mean_se(x):
    x = np.asarray(list(x), float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)


def full(*arms):
    return all(set(SEEDS) <= set(rows.get(a, {})) for a in arms)


def paired(a, b, f):
    ea, eb = per_seed(a), per_seed(b)
    return [f(ea[s], eb[s]) for s in SEEDS]


fno_p = os.path.join(RES, "fno_baseline.json")
fno = json.load(open(fno_p)) if os.path.exists(fno_p) else None
summary = {}
for arm in ARMS:
    if not rows.get(arm):
        continue
    e = per_seed(arm)
    summary[arm] = dict(seeds=sorted(e), rel_l2_per_seed=e, rel_l2=mean_se(e.values()),
                        subscores={k: mean_se(per_seed(arm, "scores", k).values()) for k in ("rel_l2", "tke", "mvpe")},
                        tke=mean_se(per_seed(arm, "rollout", "tke").values()),
                        mvpe=mean_se(per_seed(arm, "rollout", "mvpe").values()),
                        one_step_rel_err=mean_se(per_seed(arm, "one_step", "rel_err").values()),
                        one_step_vs_persistence=mean_se(per_seed(arm, "one_step", "ratio").values()))

P, detail = {}, {}
if full("S", "PD"):
    m, se = mean_se(paired("PD", "S", lambda pd, s: 1 - pd / s))
    detail["E6-P1"] = dict(adv_PD=m, se=se)
    P["E6-P1"] = m >= 0.10 and m > 2 * se
if full("PD", "PA"):
    m, se = mean_se(paired("PD", "PA", lambda pd, pa: pd - pa))
    detail["E6-P2"] = dict(err_PD_minus_PA=m, se=se)
    P["E6-P2"] = m > 2 * se
if full("ZA", "S"):
    za, s = summary["ZA"]["rel_l2"][0], summary["S"]["rel_l2"][0]
    detail["E6-P3"] = dict(err_ZA=za, err_S=s)
    P["E6-P3"] = za < s
if full("PD", "ZD"):
    m, se = mean_se(paired("PD", "ZD", lambda pd, zd: 1 - pd / zd))
    detail["E6-P4"] = dict(gain_PD_over_ZD=m, se=se)
    P["E6-P4"] = m >= 0.10
if full(*ARMS) and fno:
    bar = fno["sim_real_fno"]["scores"]["rel_l2"]
    detail["E6-P5"] = dict(fno_subscore=bar, arm_subscores={a: summary[a]["subscores"]["rel_l2"][0] for a in ARMS})
    P["E6-P5"] = all(summary[a]["subscores"]["rel_l2"][0] < bar for a in ARMS)
for k in ("E6-P1", "E6-P2", "E6-P3", "E6-P4", "E6-P5"):
    P.setdefault(k, None)

out = dict(arms=summary, predictions=P, detail=detail,
           fno={k: fno[k] for k in ("sim_real_fno", "sim_fno")} if fno else None)
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, detail=detail), indent=1))
