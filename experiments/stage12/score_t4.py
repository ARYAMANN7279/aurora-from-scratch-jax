"""Score the Stage 12 supplement on Kaggle's T4 (handouts/stage12_speed.md, section 9) from results/t4/.

Each arm's time is its median ms per step; J0 is the J0 that opens the job. A missing arm leaves its
predictions unscored.
S12-K1  J1 at batch 64 >= 2.5 x J0                S12-K8   T0 >= 1.3 x J0
S12-K2  J0 / J3 < 1.25                            S12-K9   |T1 / J0 - 1| <= 0.25
S12-K3  J4 (one step of 3 models) >= 2.5 x J0     S12-K10  T1 / T2 < 1.1
S12-K4  |J2 / J0 - 1| < 0.10                      S12-K11  |T3 / T1 - 1| < 0.10
S12-K5  |J5 / J0 - 1| < 0.10                      S12-K12  |forward: torch.compile / JAX - 1| <= 0.25
S12-K6  J0 / J6 < 1.1 and |bf16 change| < 1%      S12-K13  |J0 at the end / J0 at the start - 1| < 0.05
S12-K7  1.2 <= J0 / J7 < 1.9
usage: score_t4.py [results_dir]   (default results/t4; writes <results_dir>/analysis.json)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "results", "t4")


def t(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p))["median_ms"] if os.path.exists(p) else None


def verdict(ok, *need):
    return None if any(x is None for x in need) else bool(ok())


J = {k: t(f"{k}_b16") for k in ("J0", "J2", "J3", "J4", "J5", "J6", "J6t", "J7", "fwd")}
J["J1_b32"], J["J1_b64"] = t("J1_b32"), t("J1_b64")
J["J0_start"], J["J0_end"] = t("J0_b16_start"), t("J0_b16_end")
T = {k: t(f"{k}_b16") for k in ("T0", "T1", "T2", "T3", "fwd_eager", "fwd_compile", "A0_eager", "A0_compile", "B0_eager")}
acc_p = os.path.join(RES, "accuracy_bf16.json")
acc = json.load(open(acc_p)) if os.path.exists(acc_p) else None
j0 = J["J0"]
P = {
    "S12-K1": verdict(lambda: J["J1_b64"] >= 2.5 * j0, j0, J["J1_b64"]),
    "S12-K2": verdict(lambda: j0 / J["J3"] < 1.25, j0, J["J3"]),
    "S12-K3": verdict(lambda: J["J4"] >= 2.5 * j0, j0, J["J4"]),
    "S12-K4": verdict(lambda: abs(J["J2"] / j0 - 1) < 0.10, j0, J["J2"]),
    "S12-K5": verdict(lambda: abs(J["J5"] / j0 - 1) < 0.10, j0, J["J5"]),
    "S12-K6": verdict(lambda: j0 / J["J6"] < 1.1 and abs(acc["rel_change"]) < 0.01, j0, J["J6"], acc),
    "S12-K7": verdict(lambda: 1.2 <= j0 / J["J7"] < 1.9, j0, J["J7"]),
    "S12-K8": verdict(lambda: T["T0"] >= 1.3 * j0, j0, T["T0"]),
    "S12-K9": verdict(lambda: abs(T["T1"] / j0 - 1) <= 0.25, j0, T["T1"]),
    "S12-K10": verdict(lambda: T["T1"] / T["T2"] < 1.1, T["T1"], T["T2"]),
    "S12-K11": verdict(lambda: abs(T["T3"] / T["T1"] - 1) < 0.10, T["T1"], T["T3"]),
    "S12-K12": verdict(lambda: abs(T["fwd_compile"] / J["fwd"] - 1) <= 0.25, J["fwd"], T["fwd_compile"]),
    "S12-K13": verdict(lambda: abs(J["J0_end"] / J["J0_start"] - 1) < 0.05, J["J0_start"], J["J0_end"]),
}
speedups = {k: j0 / J[k] for k in ("J2", "J3", "J5", "J6", "J6t", "J7") if j0 and J[k]}
if j0 and J["J4"]:
    speedups["J4 (per model)"] = j0 / (J["J4"] / 3)
out = dict(ms_per_step=dict(jax=J, torch=T), bf16_accuracy=acc, predictions=P, speedup_vs_J0=speedups)
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
