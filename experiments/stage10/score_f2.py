"""Score Stage 10 F2 (handouts/stage10_lora.md, F2) from results/f2_<target>_<arm>[_s<seed>].json.

For each target (re3000_kf4, re30_kf4, re1000_kf4): the zero-shot error and its ratio to persistence, and per seed
gain(arm) = 1 - e(arm) / e(zero-shot) for LoRA r = 4 and full fine-tuning, mean and SE over seeds 0-2 (ddof 1).
F2-P1  ratio_zs(3000) > ratio_zs(1000)            F2-P4  ratio_zs(30) >= 1
F2-P2  gain_full(3000) > gain_full(1000)          F2-P5  gain_lora(30) >= 0.20
F2-P3  gain_lora(3000) < 0.20                     F2-P6  gain_lora(30) >= 0.5 * gain_full(30)
Acceptance (A2 at the farther viscosity): mean e(LoRA) <= 0.8 * e(zero-shot) at Re 3000.
usage: score_f2.py   (writes results/f2_analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TARGETS, SEEDS = ("re3000_kf4", "re30_kf4", "re1000_kf4"), (0, 1, 2)


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


out = dict(targets={}, predictions={f"F2-P{i}": None for i in range(1, 7)}, acceptance_at_re3000=None)
T = out["targets"]
for t in TARGETS:
    short = t.split("_")[0]
    zs = load(f"f2_{short}_zeroshot")
    if not zs:
        continue
    row = dict(zero_shot=zs["eval"]["rel_err"], persistence=zs["eval"]["persistence"], ratio_zs=zs["eval"]["ratio"])
    for arm in ("lora", "full"):
        runs = [load(f"f2_{short}_{arm}_s{s}") for s in SEEDS]
        if all(runs):
            e = [r["eval"]["rel_err"] for r in runs]
            g = [1 - x / row["zero_shot"] for x in e]
            row[arm] = dict(err=e, ratio=[r["eval"]["ratio"] for r in runs], gain=g, gain_mean_se=mean_se(g),
                            err_mean=float(np.mean(e)), trainable_frac=runs[0]["trainable_frac"])
    T[t] = row
f1 = {a: load(n) for a, n in (("zeroshot", "f1_zeroshot"), ("lora", "f1_lora_r4"), ("full", "f1_full"))}
if all(f1.values()):
    out["f1_cpu_re1000"] = dict(zero_shot=f1["zeroshot"]["eval"]["rel_err"],
                                gain_lora=1 - f1["lora"]["eval"]["rel_err"] / f1["zeroshot"]["eval"]["rel_err"],
                                gain_full=1 - f1["full"]["eval"]["rel_err"] / f1["zeroshot"]["eval"]["rel_err"])
g = lambda t, arm: T[t][arm]["gain_mean_se"][0]
ok = lambda *ts: all(t in T and "lora" in T[t] and "full" in T[t] for t in ts)
if ok("re3000_kf4", "re1000_kf4"):
    out["predictions"]["F2-P1"] = T["re3000_kf4"]["ratio_zs"] > T["re1000_kf4"]["ratio_zs"]
    out["predictions"]["F2-P2"] = g("re3000_kf4", "full") > g("re1000_kf4", "full")
    out["predictions"]["F2-P3"] = g("re3000_kf4", "lora") < 0.20
    out["acceptance_at_re3000"] = dict(lora_over_zero_shot=T["re3000_kf4"]["lora"]["err_mean"] / T["re3000_kf4"]["zero_shot"],
                                       trainable_frac=T["re3000_kf4"]["lora"]["trainable_frac"],
                                       met=T["re3000_kf4"]["lora"]["err_mean"] <= 0.8 * T["re3000_kf4"]["zero_shot"])
if ok("re30_kf4"):
    out["predictions"]["F2-P4"] = T["re30_kf4"]["ratio_zs"] >= 1
    out["predictions"]["F2-P5"] = g("re30_kf4", "lora") >= 0.20
    out["predictions"]["F2-P6"] = g("re30_kf4", "lora") >= 0.5 * g("re30_kf4", "full")
with open(os.path.join(RES, "f2_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=out["predictions"], acceptance_at_re3000=out["acceptance_at_re3000"],
                      targets={t: {k: (v if not isinstance(v, dict) else v.get("gain_mean_se")) for k, v in r.items()}
                               for t, r in T.items()}), indent=1))
