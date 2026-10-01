"""Score Stage 12 (handouts/stage12_speed.md, sections 4-5) from the result JSONs in results/.

Each arm's time is its median ms per step over the repeats. A missing arm leaves its predictions
unscored.
S12-P1  J1 at batch 64 < 2 x J0 (batch 16)          S12-P6  J0 / J6 >= 1.3 and |bf16 change| < 1%
S12-P2  J3 <= 0.75 x J0                             S12-P7  T0 >= 1.5 x J0
S12-P3  J4 (one step of 3 models) < 1.67 x J0       S12-P8  |T1 / J0 - 1| <= 0.25
S12-P4  |J2 / J0 - 1| < 0.05                        S12-P9  T2 <= 0.8 x T1
S12-P5  |J5 / J0 - 1| < 0.10                        S12-P10 |forward: torch.compile / JAX - 1| <= 0.25
Acceptance: some JAX arm trains the E1 model >= 1.5x faster than J0. J4 counts per model (3 models
per step). J3 counts as the same numerics (the same step, with batches drawn by a different random
stream). J6 counts only if the S12-P6 accuracy check passes.
usage: score.py   (writes results/analysis.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def t(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p))["median_ms"] if os.path.exists(p) else None


def verdict(ok, *need):
    return None if any(x is None for x in need) else bool(ok())


J = {k: t(f"{k}_b16") for k in ("J0", "J2", "J3", "J4", "J5", "J6", "J6t", "fwd")}
J["J1_b32"], J["J1_b64"] = t("J1_b32"), t("J1_b64")
T = {k: t(f"{k}_b16") for k in ("T0", "T1", "T2", "T3", "fwd_eager", "fwd_compile", "A0_eager", "A0_compile", "B0_eager")}
acc_p = os.path.join(RES, "accuracy_bf16.json")
acc = json.load(open(acc_p)) if os.path.exists(acc_p) else None
j0 = J["J0"]
P = {
    "S12-P1": verdict(lambda: J["J1_b64"] < 2 * j0, j0, J["J1_b64"]),
    "S12-P2": verdict(lambda: J["J3"] <= 0.75 * j0, j0, J["J3"]),
    "S12-P3": verdict(lambda: J["J4"] < 1.67 * j0, j0, J["J4"]),
    "S12-P4": verdict(lambda: abs(J["J2"] / j0 - 1) < 0.05, j0, J["J2"]),
    "S12-P5": verdict(lambda: abs(J["J5"] / j0 - 1) < 0.10, j0, J["J5"]),
    "S12-P6": verdict(lambda: j0 / J["J6"] >= 1.3 and abs(acc["rel_change"]) < 0.01, j0, J["J6"], acc),
    "S12-P7": verdict(lambda: T["T0"] >= 1.5 * j0, j0, T["T0"]),
    "S12-P8": verdict(lambda: abs(T["T1"] / j0 - 1) <= 0.25, j0, T["T1"]),
    "S12-P9": verdict(lambda: T["T2"] <= 0.8 * T["T1"], T["T1"], T["T2"]),
    "S12-P10": verdict(lambda: abs(T["fwd_compile"] / J["fwd"] - 1) <= 0.25, J["fwd"], T["fwd_compile"]),
}
speedups = {}
if j0:
    for k in ("J2", "J3", "J5"):
        if J[k]:
            speedups[k] = j0 / J[k]
    if J["J4"]:
        speedups["J4 (per model)"] = j0 / (J["J4"] / 3)
    if J["J6"] and acc and abs(acc["rel_change"]) < 0.01:
        speedups["J6"] = j0 / J["J6"]
best = max(speedups.items(), key=lambda kv: kv[1]) if speedups else None
out = dict(ms_per_step=dict(jax=J, torch=T), bf16_accuracy=acc, predictions=P, speedup_vs_J0=speedups,
           acceptance=dict(bar="a JAX arm >= 1.5x faster than J0", best=best, met=None if best is None else best[1] >= 1.5))
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
