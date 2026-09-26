"""Throughput (handout 5.3.5): f64 steps/s, Re=1000 parameters, 1000 scanned steps, best of 3.

usage: run_throughput.py [--n 64 128 256 512] [--steps 1000]
"""
import argparse
import os
import time

from common import command_line, setup_precision, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[64, 128, 256, 512])
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--f32", action="store_true")
args = ap.parse_args()
setup_precision(args.f32)

from kolmogorov import Params, choose_m, make_advance, make_grid, random_ic  # noqa: E402

prm = Params(nu=1e-3)
res = {}
for n in args.n:
    g = make_grid(n)
    adv = make_advance(g, prm, 0.1 / choose_m(n, prm), args.steps)
    wh = random_ic(g)
    adv(wh).block_until_ready()  # compile + warm-up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        adv(wh).block_until_ready()
        times.append(time.perf_counter() - t0)
    res[str(n)] = dict(steps_per_s=args.steps / min(times), times=times)
    print(n, res[str(n)], flush=True)
write_json("throughput_f32.json" if args.f32 else "throughput.json",
           dict(results=res, cpu_count=os.cpu_count(), command=command_line()))
