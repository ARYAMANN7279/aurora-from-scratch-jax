"""Stage 2 F2 (handouts/stage02_dataset.md, F2): where do two generations of the same trajectory part?

For each tag:seed in --picks, between dataset roots --a and --b:
  files       whether each saved file is identical (the sha256 in each meta.json);
  state_t0    whether the post-spin-up generation-grid states are identical, and their relative difference;
  dE, dw      per frame, the relative difference of the energy E (diag.npz, float64) and of the vorticity in the
              128² frames (float32);
  first frame at which E differs;
  growth      if the difference of E grows exponentially, its rate (a fit of ln dE over the frames from the first
              difference until dE first reaches 1e-4, those above 1e-13, at least 5 of them, positive slope only) and
              the time it would have been a last-bit difference (dE = 1e-16), extrapolated back;
  start_t0    when the states already differ after the spin-up: the time at which a last-bit difference (1e-16) would
              grow to their difference, at Stage 1's Lyapunov exponent for the regime (0.394 / 0.472 / 0.565 for
              Re 100 / 300 / 1000) and, when available, at the fitted rate.
Times are simulation times; the spin-up runs from 0 to 50 and the recording from 50 to 150.
usage: divergence_f2.py --a data/stage2/<root> --b data/stage2/<root> --picks re1000_kf4:7 ... --out <name>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kdata import traj_dir  # noqa: E402

LYAPUNOV = {100: 0.394, 300: 0.472, 1000: 0.565}  # Stage 1, P14
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--picks", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = dict(a=args.a, b=args.b, pairs={})
for p in args.picks:
    tag, seed = p.split(":")
    da, db = (traj_dir(os.path.join(ROOT, r), tag, int(seed)) for r in (args.a, args.b))
    ma, mb = (json.load(open(os.path.join(d, "meta.json"))) for d in (da, db))
    rec = dict(files_equal={k: ma["sha256"][k] == mb["sha256"].get(k) for k in ma["sha256"]})
    sa, sb = (np.load(os.path.join(d, "state_t0.npy")) for d in (da, db))
    rec["state_t0_equal"] = bool(np.array_equal(sa, sb))
    rec["state_t0_rel_diff"] = float(np.linalg.norm(sa - sb) / np.linalg.norm(sa))
    ga, gb = (np.load(os.path.join(d, "diag.npz")) for d in (da, db))
    t, dE = ga["t"], np.abs(ga["E"] - gb["E"]) / np.abs(ga["E"])
    fa, fb = (np.load(os.path.join(d, "frames_128.npy"), mmap_mode="r") for d in (da, db))
    dw = np.array([np.linalg.norm(fa[i, 3].astype(np.float64) - fb[i, 3]) / np.linalg.norm(fa[i, 3].astype(np.float64))
                   for i in range(len(t))])
    nz = np.flatnonzero(dE > 0)
    rec["first_E_diff_frame"] = int(nz[0]) if len(nz) else None
    rec["first_E_diff_t"] = float(t[nz[0]]) if len(nz) else None
    rec["dE_every_25_frames"] = [float(x) for x in dE[::25]]
    rec["dw_every_25_frames"] = [float(x) for x in dw[::25]]
    rate = None
    if len(nz):
        j = nz[0]
        while j < len(t) and dE[j] < 1e-4:
            j += 1
        seg = np.arange(nz[0], j)
        seg = seg[dE[seg] > 1e-13]
        if len(seg) >= 5:
            slope, icpt = np.polyfit(t[seg], np.log(dE[seg]), 1)
            if slope > 0:
                rate = float(slope)
                rec["growth"] = dict(rate=rate, fit_frames=len(seg), start_t=float((np.log(1e-16) - icpt) / slope))
    if not rec["state_t0_equal"]:
        lam = LYAPUNOV.get(ma["Re"])
        d0 = rec["state_t0_rel_diff"]
        rec["start_t0"] = {f"lambda_{l:.3f}": float(ma["t0"] - np.log(d0 / 1e-16) / l) for l in (lam, rate) if l}
    out["pairs"][p] = rec
    print(p, {k: v for k, v in rec.items() if k not in ("dE_every_25_frames", "dw_every_25_frames")}, flush=True)
with open(os.path.join(HERE, "results", f"{args.out}.json"), "w") as f:
    json.dump(out, f, indent=1)
