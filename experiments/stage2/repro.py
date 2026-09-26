"""M3: regenerate trajectories in a fresh process and compare sha256 with the manifest.

usage: repro.py --dataset acc_v1 --pick re100_kf4:3 re300_kf4:5 re1000_kf4:7
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from kdata import generate_trajectory, traj_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--pick", nargs="+", required=True)
args = ap.parse_args()
root = os.path.join(ROOT, "data", "stage2", args.dataset)
man = json.load(open(os.path.join(root, "manifest.json")))
by = {(t["tag"], t["seed"]): t for t in man["trajectories"]}
cfg = man["config"]
rows = []
with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "data", "stage2")) as tmp:
    for p in args.pick:
        tag, seed = p.split(":")
        t = by[(tag, int(seed))]
        meta, _ = generate_trajectory(tmp, t["Re"], t["kf"], t["seed"], t["N_gen"], cfg["T_spin"], cfg["T_rec"],
                                      cfg["save_dt"], tuple(cfg["N_out"]), verbose=False)
        eq = {k: meta["sha256"][k] == t["sha256"][k] for k in t["sha256"]}
        rows.append(dict(tag=tag, seed=int(seed), equal=eq))
        print(tag, seed, eq, flush=True)
n_equal = sum(all(r["equal"].values()) for r in rows)
out = dict(dataset=args.dataset, rows=rows, n=len(rows), n_equal=n_equal,
           command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
json.dump(out, open(os.path.join(HERE, "results", f"repro_{args.dataset}.json"), "w"), indent=1)
print(f"{n_equal}/{len(rows)} identical")
