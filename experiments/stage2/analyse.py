"""M3-M5 analysis and prediction scoring for a Stage 2 dataset (handouts/stage02_dataset.md, 4-5).

usage: analyse.py --dataset acc_v1
Reads data/stage2/<dataset>/, experiments/stage2/results/m1_Re*.json, repro_<dataset>.json,
loader_<dataset>.json and the Stage 1 long-run JSONs. Writes results/<dataset>_analysis.json and
results/predictions.json, and prints the prediction table.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "results")
S1 = os.path.join(ROOT, "experiments", "stage1", "results")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
args = ap.parse_args()
root = os.path.join(ROOT, "data", "stage2", args.dataset)
with open(os.path.join(root, "manifest.json")) as f:
    man = json.load(f)
rj = lambda p: json.load(open(p))
S1RUN = {100: "L100", 300: "L300", 1000: "L1000"}

# ---- M1: generation grid rule
m1 = {}
for re in (100, 300, 1000):
    q = rj(os.path.join(RES, f"m1_Re{re}.json"))
    m1[re] = {}
    for band in (64, 128):
        ok = [r["N"] for r in q["rows"] if r[f"F{band}_t1"] <= 1e-3]
        m1[re][band] = dict(N_gen=min(ok) if ok else None,
                            F_t1={str(r["N"]): r[f"F{band}_t1"] for r in q["rows"]})

# ---- M4: spectral stability per regime
regimes = {}
for t in man["trajectories"]:
    regimes.setdefault(t["tag"], []).append(t)
stab = {}
KMAX = 31
for tag, trajs in sorted(regimes.items(), key=lambda kv: kv[1][0]["Re"]):
    Re = trajs[0]["Re"]
    rows = []
    specs = []
    halves = []
    for t in sorted(trajs, key=lambda t: t["seed"]):
        d = np.load(os.path.join(root, tag, f"seed_{t['seed']:04d}", "diag.npz"))
        E, spec = d["E"], d["spec"]
        bm = np.array([b.mean() for b in np.array_split(E, 5)])
        k = np.arange(spec.shape[1])
        ms = spec.mean(axis=0)
        sel = (k >= 8) & (k <= 16)
        slope = float(np.polyfit(np.log(k[sel]), np.log(ms[sel]), 1)[0])
        specs.append(ms)
        h = len(E) // 2
        halves.append((E[:h].mean(), E[h:].mean()))
        rows.append(dict(seed=t["seed"], E_mean=float(E.mean()), E_se=float(bm.std(ddof=1) / np.sqrt(5)),
                         slope=slope, max_cfl=t["max_cfl"], wall_s=t["wall_s"], N_gen=t["N_gen"]))
    specs = np.array(specs)
    mean_spec = specs.mean(axis=0)
    k = np.arange(mean_spec.size)
    band = (k >= 1) & (k <= KMAX)
    for r, s in zip(rows, specs):
        r["max_spec_dev"] = float(np.abs(s[band] / mean_spec[band] - 1).max())
    Em = np.array([r["E_mean"] for r in rows])
    Ese = np.array([r["E_se"] for r in rows])
    w = 1 / Ese**2
    Ebar = float((w * Em).sum() / w.sum())
    chi2 = float(((Em - Ebar) ** 2 * w).sum() / (len(Em) - 1))
    slopes = np.array([r["slope"] for r in rows])
    h1, h2 = np.array(halves).T
    diff = h2.mean() - h1.mean()
    se_diff = np.hypot(h1.std(ddof=1), h2.std(ddof=1)) / np.sqrt(len(h1))
    s1 = rj(os.path.join(S1, f"{S1RUN[int(Re)]}.json")) if int(Re) in S1RUN and trajs[0]["kf"] == 4 else None
    s1_dev = None
    if s1:
        ref = np.array(s1["spectrum"])[1:KMAX + 1]
        s1_dev = float(np.abs(mean_spec[1:KMAX + 1] / ref - 1).max())
    stab[tag] = dict(Re=Re, kf=trajs[0]["kf"], N_gen=trajs[0]["N_gen"], rows=rows, E_weighted_mean=Ebar,
                     chi2_red=chi2, slope_mean=float(slopes.mean()),
                     max_slope_dev=float(np.abs(slopes - slopes.mean()).max()),
                     max_spec_dev=float(max(r["max_spec_dev"] for r in rows)),
                     half_diff=float(diff), half_diff_se=float(se_diff),
                     half_ratio=float(abs(diff) / se_diff), stage1_spec_dev=s1_dev,
                     mean_spectrum=mean_spec.tolist())

repro = rj(os.path.join(RES, f"repro_{args.dataset}.json"))
loader = rj(os.path.join(RES, f"loader_{args.dataset}.json"))

out = dict(dataset=args.dataset, m1=m1, stability=stab, command=" ".join([".venv/bin/python"] + [os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
with open(os.path.join(RES, f"{args.dataset}_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---- predictions
P = []


def add(pid, q, pred, meas, hit):
    P.append(dict(id=pid, quantity=q, prediction=pred, measured=meas, verdict="HIT" if hit else "MISS"))


want1 = {100: 128, 300: 256, 1000: 256}
want2 = {100: 256, 300: 256, 1000: 512}
add("Q1", "N_gen, 64 band", "Re100 128, Re300 256, Re1000 256",
    ", ".join(f"Re{re}: {m1[re][64]['N_gen']}" for re in m1), all(m1[re][64]["N_gen"] == want1[re] for re in m1))
add("Q2", "N_gen, 128 band", "Re100 256, Re300 256, Re1000 512",
    ", ".join(f"Re{re}: {m1[re][128]['N_gen']}" for re in m1), all(m1[re][128]["N_gen"] == want2[re] for re in m1))
add("Q3", "cross-process sha256", "all equal", f"{repro['n_equal']}/{repro['n']} equal", repro["n_equal"] == repro["n"])
st = list(stab.values())
tagfmt = lambda f: ", ".join(f"Re{s['Re']}: {f(s)}" for s in st)
add("Q4", "reduced χ² of per-trajectory ⟨E⟩", "in [0.2, 3] each", tagfmt(lambda s: f"{s['chi2_red']:.2f}"),
    all(0.2 <= s["chi2_red"] <= 3 for s in st))
add("Q5", "max |slope − regime mean|, k∈[8,16]", "≤ 0.15 every trajectory", tagfmt(lambda s: f"{s['max_slope_dev']:.3f}"),
    all(s["max_slope_dev"] <= 0.15 for s in st))
add("Q6", "max_k |E_i(k)/Ē(k) − 1|, 1≤k≤31", "< 0.3 every trajectory", tagfmt(lambda s: f"{s['max_spec_dev']:.3f}"),
    all(s["max_spec_dev"] < 0.3 for s in st))
add("Q7", "regime-mean spectrum vs Stage 1, 1≤k≤31", "every shell within 15%",
    tagfmt(lambda s: f"{s['stage1_spec_dev']:.3f}"), all(s["stage1_spec_dev"] is not None and s["stage1_spec_dev"] <= 0.15 for s in st))
add("Q8", "first vs second half ⟨E⟩", "within 3 combined SE", tagfmt(lambda s: f"{s['half_ratio']:.2f} SE"),
    all(s["half_ratio"] <= 3 for s in st))
lw = loader["results"]
add("Q9", "loader windows/s, batch 32, n_in 2", "≥ 2000 at 64; ≥ 500 at 128",
    f"64: {lw['64']['windows_per_s']:.0f}, 128: {lw['128']['windows_per_s']:.0f}",
    lw["64"]["windows_per_s"] >= 2000 and lw["128"]["windows_per_s"] >= 500)
with open(os.path.join(RES, "predictions.json"), "w") as f:
    json.dump(dict(rows=P, command=out["command"]), f, indent=1)
print("| # | Quantity | Prediction | Measured | Verdict |\n|---|---|---|---|---|")
for r in P:
    print(f"| {r['id']} | {r['quantity']} | {r['prediction']} | {r['measured']} | **{r['verdict']}** |")
