"""Score pre-registered predictions P1-P16 from results/*.json -> results/predictions.json.

usage: summarise.py
"""
import math
import os

import numpy as np

from common import RESULTS, command_line, read_json, write_json

LONG = {100: "L100", 300: "L300", 1000: "L1000"}
have = lambda f: os.path.exists(os.path.join(RESULTS, f))
runs = {re: read_json(f"{n}.json") for re, n in LONG.items()}
rows = []


def add(pid, what, pred, measured, hit):
    rows.append(dict(id=pid, quantity=what, prediction=pred, measured=measured,
                     verdict=("HIT" if hit else "MISS") if hit is not None else "NOT RUN"))


fmt = lambda x: f"{x:.3g}"
per = lambda key: ", ".join(f"Re{re}: {fmt(r[key])}" for re, r in runs.items())

add("P1", "|R_E| / ∫P over [50,250]", "< 1e-4 each", per("R_E_rel"),
    all(r["R_E_rel"] < 1e-4 for r in runs.values()))
add("P2", "|R_Z| / ∫Q over [50,250]", "< 1e-3 each", per("R_Z_rel"),
    all(r["R_Z_rel"] < 1e-3 for r in runs.values()))

orders = {n: read_json(f"order_{n}.json")["orders"] for n in LONG.values()}
add("P3", "temporal orders", "all in [3.5, 4.5]",
    "; ".join(f"{n}: " + ", ".join(f"{o:.3f}" for o in v) for n, v in orders.items()),
    all(3.5 <= o <= 4.5 for v in orders.values() for o in v) and all(orders.values()))
add("P4", "E(K)/max E(k)", "< 1e-6 each", per("E_K_over_max"),
    all(r["E_K_over_max"] < 1e-6 for r in runs.values()))
add("P5", "top-20% shell slope", "< -3 each", per("slope_top20"),
    all(r["slope_top20"] < -3 for r in runs.values()))
s = runs[1000]["slope_8_16"]
add("P6", "Re1000 slope, k in [8,16]", "in [-6, -2.8] (guess -4.3)",
    f"{s:.3f} ± {runs[1000]['slope_8_16_se']:.3f} (SE)", -6 <= s <= -2.8)
if have("L5000.json"):
    r5 = read_json("L5000.json")
    add("P7", "Re5000 slope, k in [8,32]", "in [-4.5, -3]", fmt(r5["slope_8_32"]),
        -4.5 <= r5["slope_8_32"] <= -3)
else:
    add("P7", "Re5000 slope, k in [8,32]", "in [-4.5, -3]", "optional run not done", None)

res_ = np.array(list(runs))
kd = np.array([runs[re]["k_d"] for re in runs])
expo = float(np.polyfit(np.log(res_), np.log(kd), 1)[0])
add("P8", "k_d(Re1000); exponent of k_d vs Re", "in [15, 50]; exponent in [0.4, 0.6]",
    f"k_d: {per('k_d')}; exponent {expo:.3f}", 15 <= kd[-1] <= 50 and 0.4 <= expo <= 0.6)
add("P9", "<E> < E_lam", "each Re",
    ", ".join(f"Re{re}: {fmt(r['E_mean'])} vs {fmt(r['E_lam'])}" for re, r in runs.items()),
    all(r["E_mean"] < r["E_lam"] for r in runs.values()))
v = runs[1000]["visc_frac"]
add("P10", "2ν<Z>/<P> at Re1000", "in [0.02, 0.4]", fmt(v), 0.02 <= v <= 0.4)

expect = {100: (64, 32), 300: (128, 64), 1000: (256, 128)}
res_txt, ok = [], True
for re, (good, bad) in expect.items():
    rr = {row["N"]: row for row in read_json(f"resolution_Re{re}.json")["rows"]}
    res_txt.append(f"Re{re}: N{bad} {rr[bad]['rel_t1']:.2e}, N{good} {rr[good]['rel_t1']:.2e}")
    ok &= rr[good]["resolved"] and not rr[bad]["resolved"]
add("P11", "resolution map (rel diff at t=1, bar 1e-3)",
    "resolved at Re100@64, Re300@128, Re1000@256; not at half", "; ".join(res_txt), ok)
add("P12", "spin-up time", "≤ 50 each", per("spin_up"),
    all(r["spin_up"] is not None and r["spin_up"] <= 50 for r in runs.values()))
te = {re: r["tau_e"] for re, r in runs.items()}
add("P13", "τ_e", "τ_e(1000) in [0.5, 5]; τ_e(100) > τ_e(1000)", per("tau_e"),
    None not in te.values() and 0.5 <= te[1000] <= 5 and te[100] > te[1000])
lam = {re: read_json(f"lyapunov_{n}.json")["lam_mean"] for re, n in LONG.items()}
add("P14", "Lyapunov λ", "λ(1000) in [0.2, 3]; λ100 < λ300 < λ1000; λ100 > 0",
    ", ".join(f"Re{re}: {fmt(v)}" for re, v in lam.items()),
    0.2 <= lam[1000] <= 3 and 0 < lam[100] < lam[300] < lam[1000])

r64, r32 = runs[1000], read_json("L1000_f32.json")
dE = abs(r32["E_mean"] - r64["E_mean"]) / r64["E_se"]
dZ = abs(r32["Z_mean"] - r64["Z_mean"]) / r64["Z_se"]
dS = abs(r32["slope_8_16"] - r64["slope_8_16"]) / r64["slope_8_16_se"]
tp64 = read_json("throughput.json")["results"]["256"]["steps_per_s"]
tp32 = read_json("throughput_f32.json")["results"]["256"]["steps_per_s"]
speed = tp32 / tp64
add("P15", "f32 vs f64 (L1000)", "<E>, <Z>, slope within 2 SE; |R_E|/∫P < 1e-3; f32 1.3–2.5× faster",
    f"|Δ<E>| {dE:.2f} SE, |Δ<Z>| {dZ:.2f} SE, |Δslope| {dS:.2f} SE; R_E {r32['R_E_rel']:.2e}; "
    f"speed-up {speed:.2f}× (idle throughput, N=256)",
    dE <= 2 and dZ <= 2 and dS <= 2 and r32["R_E_rel"] < 1e-3 and 1.3 <= speed <= 2.5)

bands = {"64": (1e3, 6e3), "128": (250, 1600), "256": (60, 400), "512": (15, 90)}
tp = read_json("throughput.json")["results"]
add("P16", "f64 steps/s", "64: 1e3–6e3 · 128: 250–1600 · 256: 60–400 · 512: 15–90",
    ", ".join(f"{n}: {tp[n]['steps_per_s']:.0f}" for n in bands),
    all(lo <= tp[n]["steps_per_s"] <= hi for n, (lo, hi) in bands.items()))

write_json("predictions.json", dict(rows=rows, command=command_line()))
print("| # | Quantity | Prediction | Measured | Verdict |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['id']} | {r['quantity']} | {r['prediction']} | {r['measured']} | **{r['verdict']}** |")
