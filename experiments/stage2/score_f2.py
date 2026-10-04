"""Score Stage 2 F2 (handouts/stage02_dataset.md, F2) from results/f2_divergence_{m,s}.json.

F2-Q1  the multi-threaded rerun of re1000_kf4:7 (F1's settings) differs from F1's first pass (some file differs)
F2-Q2  the two single-threaded generations are identical for all three M3 picks (every file and the post-spin-up state)
F2-Q3  the multi-threaded pair parts during the spin-up: their post-spin-up states already differ
usage: score_f2.py   (writes results/f2_analysis.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
m = json.load(open(os.path.join(RES, "f2_divergence_m.json")))["pairs"]["re1000_kf4:7"]
s = json.load(open(os.path.join(RES, "f2_divergence_s.json")))["pairs"]
same = lambda r: all(r["files_equal"].values()) and r["state_t0_equal"]
out = dict(predictions={"F2-Q1": not all(m["files_equal"].values()),
                        "F2-Q2": len(s) == 3 and all(same(r) for r in s.values()),
                        "F2-Q3": not m["state_t0_equal"]},
           single_thread={p: same(r) for p, r in s.items()},
           multi_thread=dict(files_equal=m["files_equal"], state_t0_equal=m["state_t0_equal"],
                             state_t0_rel_diff=m["state_t0_rel_diff"], first_E_diff_t=m["first_E_diff_t"],
                             growth=m.get("growth"), start_t0=m.get("start_t0")))
with open(os.path.join(RES, "f2_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
