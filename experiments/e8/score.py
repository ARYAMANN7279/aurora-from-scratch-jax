"""Score E8 (handouts/e8_followups.md, section 6) from results/rows.jsonl.

On the evaluation trajectories (metrics["val/re300_kf4"]): err = one-step relative error; d20 and d50 = the 20- and
50-step drift from 84 rollouts (rollout_drift_many, rollout_drift_many_long); e(k) = the error at rollout step k
(rollout_err_by_step); train_s = the training loop's time. adv_X(Y) = 1 - m(Y) / m(X), paired by seed; mean and SE
over seeds (ddof 1). A prediction is scored only when every arm it uses has all 3 seeds; for a seed with several
rows, the last counts.
E8-P1   adv_B(R) on err >= 0.03 and > 2 SE            E8-P6   adv_BP(RP) on err >= 0.03 and > 2 SE
E8-P2   |adv_B(R) on d20| < 0.03                       E8-P7   |adv_B(Q) on err| < 0.03
E8-P3   adv_B(R) on e(1) >= 0.03, and on e(20) < half that on e(1) (means over seeds)
E8-P4   adv_B(BP) on d50 >= 0.10 and > 2 SE            E8-P8   |mean train_s(Q) / mean train_s(B) - 1| < 0.10
E8-P5   |adv_B(BP) on err| < 0.03                      E8-P10  mean train_s(B) / mean train_s(BS) >= 1.10
E8-P9   in every seed, |err(BS) / err(B) - 1| < 0.01 and |d20(BS) / d20(B) - 1| < 0.03
Also: E7's rows for B and R against E8's (the same training), and each arm's mean error curve e(k).
usage: score.py   (writes results/analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
KEY = "val/re300_kf4"
ARMS = ("B", "R", "BP", "RP", "Q", "BS")

rows = {}
p = os.path.join(RES, "rows.jsonl")
for line in (open(p) if os.path.exists(p) else []):
    r = json.loads(line)
    arm, seed = r["name"].split("_")[1], int(r["name"].split("_s")[-1])
    rows.setdefault(arm, {})[seed] = r

PICK = {"err": lambda r: r["metrics"][KEY]["rel_err"], "ratio": lambda r: r["metrics"][KEY]["ratio"],
        "drift1": lambda r: r["metrics"][KEY]["rollout_drift"],
        "d20": lambda r: r["metrics"][KEY]["rollout_drift_many"], "d50": lambda r: r["metrics"][KEY]["rollout_drift_many_long"],
        "train_s": lambda r: r["train_s"], "wall_s": lambda r: r["wall_s"]}


def get(arm, field):
    return {s: PICK[field](r) for s, r in sorted(rows.get(arm, {}).items())}


def curve(arm):
    return {s: np.asarray(r["metrics"][KEY]["rollout_err_by_step"]) for s, r in sorted(rows.get(arm, {}).items())}


def mean_se(x):
    x = np.asarray(list(x), float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)


def adv(base, arm, field):
    X, Y = get(base, field), get(arm, field)
    return mean_se(1 - Y[s] / X[s] for s in SEEDS)


def full(*arms):
    return all(set(SEEDS) <= set(rows.get(a, {})) for a in arms)


summary = {a: {f: dict(per_seed=get(a, f), mean_se=mean_se(get(a, f).values())) for f in PICK} for a in ARMS if rows.get(a)}
curves = {a: [float(x) for x in np.mean(list(curve(a).values()), axis=0)] for a in ARMS if rows.get(a)}
P, detail = {f"E8-P{i}": None for i in range(1, 11)}, {}
if full("B", "R"):
    m, se = adv("B", "R", "err")
    detail["E8-P1"] = dict(adv=m, se=se)
    P["E8-P1"] = m >= 0.03 and m > 2 * se
    m, se = adv("B", "R", "d20")
    detail["E8-P2"] = dict(adv_d20=m, se=se, adv_d50=adv("B", "R", "d50"))
    P["E8-P2"] = abs(m) < 0.03
    cB, cR = curve("B"), curve("R")
    by_step = [mean_se(1 - cR[s][k] / cB[s][k] for s in SEEDS) for k in range(len(cB[0]))]
    a1, a20 = by_step[0][0], by_step[19][0]
    detail["E8-P3"] = dict(adv_step1=by_step[0], adv_step20=by_step[19], adv_by_step=by_step)
    P["E8-P3"] = a1 >= 0.03 and a20 < a1 / 2
if full("B", "BP"):
    m, se = adv("B", "BP", "d50")
    detail["E8-P4"] = dict(adv_d50=m, se=se, adv_d20=adv("B", "BP", "d20"))
    P["E8-P4"] = m >= 0.10 and m > 2 * se
    m, se = adv("B", "BP", "err")
    detail["E8-P5"] = dict(adv=m, se=se)
    P["E8-P5"] = abs(m) < 0.03
if full("BP", "RP"):
    m, se = adv("BP", "RP", "err")
    detail["E8-P6"] = dict(adv=m, se=se, adv_d20=adv("BP", "RP", "d20"), adv_d50=adv("BP", "RP", "d50"))
    P["E8-P6"] = m >= 0.03 and m > 2 * se
if full("B", "Q"):
    m, se = adv("B", "Q", "err")
    detail["E8-P7"] = dict(adv=m, se=se, adv_d20=adv("B", "Q", "d20"), adv_d50=adv("B", "Q", "d50"))
    P["E8-P7"] = abs(m) < 0.03
    ratio = float(np.mean(list(get("Q", "train_s").values())) / np.mean(list(get("B", "train_s").values())))
    detail["E8-P8"] = dict(train_s_Q_over_B=ratio)
    P["E8-P8"] = abs(ratio - 1) < 0.10
if full("B", "BS"):
    eB, eS, dB, dS = get("B", "err"), get("BS", "err"), get("B", "d20"), get("BS", "d20")
    de = {s: eS[s] / eB[s] - 1 for s in SEEDS}
    dd = {s: dS[s] / dB[s] - 1 for s in SEEDS}
    detail["E8-P9"] = dict(err_change=de, d20_change=dd)
    P["E8-P9"] = all(abs(de[s]) < 0.01 and abs(dd[s]) < 0.03 for s in SEEDS)
    speed = float(np.mean(list(get("B", "train_s").values())) / np.mean(list(get("BS", "train_s").values())))
    detail["E8-P10"] = dict(train_s_B_over_BS=speed)
    P["E8-P10"] = speed >= 1.10
e7 = {}
e7p = os.path.join(HERE, "..", "e7", "results", "rows.jsonl")
for line in (open(e7p) if os.path.exists(e7p) else []):
    r = json.loads(line)
    arm, seed = r["name"].split("_")[1], int(r["name"].split("_s")[-1])
    if arm in rows and seed in rows[arm]:
        mine = rows[arm][seed]["metrics"][KEY]
        e7[r["name"]] = dict(err_e7=r["metrics"][KEY]["rel_err"], err_e8=mine["rel_err"],
                             drift1_e7=r["metrics"][KEY]["rollout_drift"], drift1_e8=mine["rollout_drift"])
out = dict(arms=summary, predictions=P, detail=detail, curves=curves, e7_vs_e8=e7)
os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, detail={k: {kk: vv for kk, vv in v.items() if kk != "adv_by_step"}
                                             for k, v in detail.items()}), indent=1))
