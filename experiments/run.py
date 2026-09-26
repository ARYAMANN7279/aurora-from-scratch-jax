"""Run one config, or every config of a sweep, through the ablation harness (Stage 11).

usage: run.py experiments/configs/<name>.json          (a config, or {"base": ..., "grid": ...})
Each run appends one row to experiments/results.jsonl; parameters go to data/runs/<config_hash>/.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aurora import harness  # noqa: E402

cfg = json.load(open(sys.argv[1]))
for c in (harness.expand(cfg) if "grid" in cfg else [cfg]):
    harness.run(c, verbose=True)
