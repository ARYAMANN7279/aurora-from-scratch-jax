"""E6 FNO baselines (handouts/e6_realpde_sim2real.md, section 4): the competition kit's FNOs on the
900 re_lohi windows, through UGP-I's canonical harness (UGP/r12_eval.py), imported rather than
rewritten.

* sim_real_fno.pth: UGP-I's selftest exactly, which must reproduce its recorded subscores (rel_l2
  95.4738, tke 75.8957, mvpe 96.0945) to 1e-3, or the run stops. Fine-tuned by the organisers on
  real data that may include the held-out conditions, so it is an upper reference, not an honest one.
* sim_fno.pth: simulations only, never fine-tuned on real data; normalised with the same real-data
  statistics (each source with its own statistics, as for our zero-shot arms).
Also checks that run_e6.val_starts selects exactly UGP-I's 900 windows.

usage (PyTorch environment): fno_baseline.py [--device cuda|cpu]   -> results/fno_baseline.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UGP = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, UGP)
import r12_eval as R  # noqa: E402  (adds the kit to sys.path)
from load_baseline import load_baseline  # noqa: E402

sys.path.insert(0, HERE)
from splits import val_starts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = torch.device(a.device)

ours = json.load(open(os.path.join(HERE, "..", "..", "data", "e6", "real_meta.json")))
sub, _ = R.val_starts(900)
same = bool(np.array_equal(val_starts(ours), sub))
if not same:
    raise SystemExit("run_e6.val_starts does not select UGP-I's windows")

out = dict(windows=len(sub), same_windows_as_ugp1=same, device=str(dev), torch=torch.__version__)
mi, si, mt, st_ = [t.to(dev) for t in (R.MI, R.SI, R.MT, R.ST)]
for name in ("sim_real_fno", "sim_fno"):
    t0 = time.time()
    model, _ = load_baseline(f"{UGP}/data/comp_real/{name}.pth", device=str(dev))
    model = model.to(dev).eval()
    P, T = [], []
    with torch.no_grad():
        for i in range(0, len(sub), 24):
            x, y = R.window(sub[i:i + 24])
            P.append((model((torch.from_numpy(x).to(dev) - mi) / si) * st_ + mt).float().cpu().numpy())
            T.append(y)
    P, T = np.concatenate(P), np.concatenate(T)
    S = R.S
    err = dict(rel_l2=float(S.rel_l2_per_sample(P, T, 2).mean()), tke=float(S.tke_rel_l2_per_sample(P, T, 2).mean()),
               mvpe=float(S.mvpe_rel_l2_per_sample(P, T).mean()))
    out[name] = dict(rollout=err, scores={k: S.score_error(v) for k, v in err.items()}, wall_s=time.time() - t0)
    if name == "sim_real_fno":
        bad = [k for k in R.BASE if abs(out[name]["scores"][k] - R.BASE[k]) > 1e-3]
        out[name]["selftest"] = "PASS" if not bad else f"FAIL on {bad}"
        if bad:
            print(json.dumps(out, indent=1))
            raise SystemExit("UGP-I's selftest does not reproduce: the harness is broken")
    print(name, out[name], flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "fno_baseline.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
