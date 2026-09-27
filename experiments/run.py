"""Run one config, or every config of a sweep, through the ablation harness (Stage 11).

usage: run.py experiments/configs/<name>.json [more.json ...] [--skip-done] [--shard i --n-shards k]
A file holds a config, a sweep {"base": ..., "grid": ...}, or a list of configs.
Each run appends one row to experiments/results.jsonl; parameters go to data/runs/<config_hash>/.
--skip-done skips configs whose hash already has a row (resume after an interruption).
--shard i --n-shards k runs every k-th config starting at i (one lane per GPU).
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aurora import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--skip-done", action="store_true")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--results", default=os.path.join(ROOT, "experiments", "results.jsonl"))
args = ap.parse_args()

todo = []
for path in args.configs:
    cfg = json.load(open(path))
    todo += cfg if isinstance(cfg, list) else harness.expand(cfg) if "grid" in cfg else [cfg]
todo = todo[args.shard::args.n_shards]
done = set()
if args.skip_done and os.path.exists(args.results):
    done = {json.loads(line)["config_hash"] for line in open(args.results) if line.strip()}
for c in todo:
    if harness.config_hash(c) in done:
        print(f"skip {c.get('name')}: already in {os.path.relpath(args.results, ROOT)}", flush=True)
        continue
    harness.run(c, results_path=args.results, verbose=True)
