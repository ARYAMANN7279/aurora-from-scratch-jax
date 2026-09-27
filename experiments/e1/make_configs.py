"""Write the E1 dataset config and run configs (handouts/e1_volume_diversity.md).

usage: make_configs.py [--ngen experiments/e1/results/ngen*.json]  (rows of several files are merged)
Writes experiments/e1/configs/{e1_v1.json, pretrain.json, finetune.json, scratch.json}.
Every regime's N_gen comes from measure_ngen.py; a regime without a measured N_gen is an error.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from aurora.harness import config_hash  # noqa: E402
from e1_design import all_regimes, dataset_config, run_configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ngen", nargs="+", default=[os.path.join(HERE, "results", "ngen.json")])
ap.add_argument("--out", default=os.path.join(HERE, "configs"))
args = ap.parse_args()

rows = {}
for path in args.ngen:
    rows.update(json.load(open(path))["rows"])
ngen = {}
for r in all_regimes():
    row = rows.get(f"{r[0]}:{r[1]}")
    if row is None or row["N_gen"] is None:
        sys.exit(f"no measured N_gen for Re {r[0]} kf {r[1]}: run measure_ngen.py first")
    ngen[r] = row["N_gen"]
os.makedirs(args.out, exist_ok=True)
pre, ft, sc = run_configs(config_hash)
files = {"e1_v1.json": dataset_config(ngen), "pretrain.json": pre, "finetune.json": ft, "scratch.json": sc}
for name, obj in files.items():
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(obj, f, indent=1)
n_traj = sum(len(r["seeds"]) for r in files["e1_v1.json"]["regimes"])
print(f"wrote {args.out}: {n_traj} trajectories; {len(pre)} pretrain, {len(ft)} finetune, {len(sc)} scratch runs")
