"""Stage 12 same-computation gate (handouts/stage12_speed.md, section 2).

The PyTorch port, loaded with the trained E1 weights, must reproduce the JAX forward pass to
<= 1e-5 relative and every parameter gradient to <= 1e-4 relative, in float32. Run in the PyTorch
environment after `bench_jax.py --export` has written the reference.

usage: check_torch.py [--ref data/stage12/ref_e1_cpu.npz] [--device cpu|cuda] [--sdpa] [--f64]
Writes results/torch_equivalence_<device>[_sdpa][_f64].json. With --f64 (reference from
`bench_jax.py --export --f64`) both sides run in float64, where rounding cannot hide a porting error.
A parameter the model never uses (emb/enc/time) has no PyTorch gradient and a zero JAX gradient.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from torch_model import AuroraTorch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default=os.path.join(ROOT, "data", "stage12", "ref_e1_cpu.npz"))
ap.add_argument("--device", default="cpu")
ap.add_argument("--sdpa", action="store_true")
ap.add_argument("--f64", action="store_true")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

z = np.load(args.ref)
flat = {k[len("param:"):]: z[k] for k in z.files if k.startswith("param:")}
dt = torch.float64 if args.f64 else torch.float32
m = AuroraTorch(flat, sdpa=args.sdpa, dtype=np.float64 if args.f64 else np.float32).to(args.device, dt)
xm, x0, xp = (torch.tensor(z[k]).to(args.device, dt) for k in ("xm", "x0", "xp"))
y = m(xm, x0)
loss = ((y - xp) ** 2).mean()
loss.backward()

yj = z["y"]
fwd_rel = float(np.abs(y.detach().cpu().numpy() - yj).max() / np.abs(yj).max())
grads, missing = {}, []
for k in flat:
    g = m.p[k.replace("/", "__")].grad
    gj = z["grad:" + k]
    nj = float(np.linalg.norm(gj))
    if g is None:
        if nj > 0:
            missing.append(k)
        continue
    if nj > 0:
        grads[k] = float(np.linalg.norm(g.cpu().numpy() - gj) / nj)
worst = max(grads, key=grads.get)
top5 = sorted(grads.items(), key=lambda kv: -kv[1])[:5]
res = dict(ref=os.path.relpath(args.ref, ROOT), device=args.device, sdpa=args.sdpa, torch=torch.__version__,
           forward_max_rel=fwd_rel, loss_jax=float(z["loss"]), loss_torch=float(loss.detach()), dtype=str(dt),
           grad_max_rel=grads[worst], grad_worst_leaf=worst, grad_top5=top5, n_leaves=len(flat), missing_grads=missing,
           gate=dict(forward=fwd_rel <= 1e-5, grads=grads[worst] <= 1e-4 and not missing))
res["gate"]["pass"] = res["gate"]["forward"] and res["gate"]["grads"]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
name = f"torch_equivalence_{args.device}" + ("_sdpa" if args.sdpa else "") + ("_f64" if args.f64 else "")
with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
