"""M5: loader throughput, batch 32, n_in=2, n_out=1, shuffled, best of 3 epochs.

usage: loader_bench.py --dataset acc_v1
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kdata import WindowDataset, read_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
root = os.path.join(ROOT, "data", "stage2", args.dataset)
members = [(t["tag"], t["seed"]) for t in read_manifest(root)["trajectories"]]
res = {}
for n in (64, 128):
    ds = WindowDataset(root, n, members)
    times = []
    for ep in range(3):
        t0 = time.perf_counter()
        nb = sum(1 for _ in ds.batches(args.batch, seed=ep))
        times.append(time.perf_counter() - t0)
    res[str(n)] = dict(windows=len(ds), windows_per_s=nb * args.batch / min(times), epoch_s=times)
    print(n, res[str(n)], flush=True)
json.dump(dict(dataset=args.dataset, batch=args.batch, results=res, cpu_count=os.cpu_count(),
               command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])),
          open(os.path.join(HERE, "results", f"loader_{args.dataset}.json"), "w"), indent=1)
