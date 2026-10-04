"""Score Stage 10 F4 (handouts/stage10_lora.md, F4) from results/f4_<target>_<arm>_s<seed>.json.

Bases: b3k, the F1 base (Stage 7's recipe, 3000 steps), and b30k, the same recipe for 30,000 steps. Arms, full
fine-tuning on the target's seeds 0-5 for 20,000 steps: b3k_lo / b3k_hi (lr 1e-4 / 1e-3) and b30k_lo / b30k_hi; plus
b30k_zeroshot and b30k_lora (Stage 10's setting: seeds 0-1, 1000 steps). Reference: F3's scratch arm. Means over
seeds 0-2 (SE with ddof 1), both targets (re3000_kf4, re1000_kf4):
F4-P1  e(b3k_hi) < e(b3k_lo)                  F4-P4  e(b30k_lo) < e(scratch)
F4-P2  e(b3k_hi) <= e(scratch)                F4-P5  zero-shot e(b30k) < zero-shot e(b3k)
F4-P3  e(b30k_lo) < e(b3k_lo)                 F4-P6  LoRA on b30k gains < 20% over b30k's zero-shot
A prediction holds only if it holds at both targets.
usage: score_f4.py   (writes results/f4_analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TARGETS, SEEDS = ("re3000_kf4", "re1000_kf4"), (0, 1, 2)
ARMS = ("b3k_lo", "b3k_hi", "b30k_lo", "b30k_hi", "b30k_lora")


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


out = dict(targets={}, predictions={f"F4-P{i}": None for i in range(1, 7)})
T = out["targets"]
for t in TARGETS:
    short = t.split("_")[0]
    zs3, zs30 = load(f"f2_{short}_zeroshot"), load(f"f4_{short}_b30k_zeroshot")
    scratch = [load(f"f3_{short}_scratch_s{s}") for s in SEEDS]
    if not (zs3 and zs30 and all(scratch)):
        continue
    row = dict(zero_shot_b3k=zs3["eval"]["rel_err"], zero_shot_b30k=zs30["eval"]["rel_err"],
               ratio_zs_b3k=zs3["eval"]["ratio"], ratio_zs_b30k=zs30["eval"]["ratio"])
    e = [r["eval"]["rel_err"] for r in scratch]
    row["scratch"] = dict(err=e, err_mean_se=mean_se(e))
    for arm in ARMS:
        runs = [load(f"f4_{short}_{arm}_s{s}") for s in SEEDS]
        if all(runs):
            e = [r["eval"]["rel_err"] for r in runs]
            z = row["zero_shot_b30k"] if arm.startswith("b30k") else row["zero_shot_b3k"]
            row[arm] = dict(err=e, err_mean_se=mean_se(e), gain_over_own_zero_shot=mean_se([1 - x / z for x in e]),
                            vs_scratch=1 - float(np.mean(e)) / row["scratch"]["err_mean_se"][0],
                            ratio_mean=float(np.mean([r["eval"]["ratio"] for r in runs])),
                            best_along_curve=[min(c["eval"]["rel_err"] for c in r["curve"]) if r["curve"] else None for r in runs],
                            lr=runs[0]["lr"], steps=runs[0]["steps"], wall_s=[r["wall_s"] for r in runs])
    T[t] = row
P = out["predictions"]
if all(t in T and all(a in T[t] for a in ARMS) for t in TARGETS):
    m = lambda t, a: T[t][a]["err_mean_se"][0]
    both = lambda f: bool(all(f(t) for t in TARGETS))
    P["F4-P1"] = both(lambda t: m(t, "b3k_hi") < m(t, "b3k_lo"))
    P["F4-P2"] = both(lambda t: m(t, "b3k_hi") <= m(t, "scratch"))
    P["F4-P3"] = both(lambda t: m(t, "b30k_lo") < m(t, "b3k_lo"))
    P["F4-P4"] = both(lambda t: m(t, "b30k_lo") < m(t, "scratch"))
    P["F4-P5"] = both(lambda t: T[t]["zero_shot_b30k"] < T[t]["zero_shot_b3k"])
    P["F4-P6"] = both(lambda t: T[t]["b30k_lora"]["gain_over_own_zero_shot"][0] < 0.20)
with open(os.path.join(RES, "f4_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, errors={t: {a: round(r[a]["err_mean_se"][0], 4) for a in ("scratch",) + ARMS if a in r}
                                             for t, r in T.items()},
                      zero_shot={t: (round(r["zero_shot_b3k"], 4), round(r["zero_shot_b30k"], 4)) for t, r in T.items()}), indent=1))
