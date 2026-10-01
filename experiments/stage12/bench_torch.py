"""Stage 12 PyTorch arms (handouts/stage12_speed.md, section 3), timed as the JAX arms are.

usage: bench_torch.py --arm T0|T1|T2|T3|fwd_eager|fwd_compile|A0_eager|A0_compile|B0_eager [--batch 16]
  T0-T3: the training step of the port (eager; torch.compile; mode reduce-overhead; compile + SDPA)
  fwd_*: the forward pass alone (S12-P10)
  A0_*:  Aurora's own Swin3DTransformerBackbone at our configuration, forward + backward (reference
         point only); B0_eager is our ported backbone on the same input, for comparison
Weights: the trained E1 model from data/stage12/ref_e1_cpu.npz (bench_jax.py --export).
Results: results/<arm>_b<batch>.json.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from torch_model import AuroraTorch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["T0", "T1", "T2", "T3", "fwd_eager", "fwd_compile",
                                                 "A0_eager", "A0_compile", "B0_eager"])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--ref", default=os.path.join(ROOT, "data", "stage12", "ref_e1_cpu.npz"))
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
F_, S, TAGS, V, N, B = 501, 16, 2, 4, 64, args.batch


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.on = [], True

    def run(self):
        card = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        while self.on:
            try:
                q = subprocess.run(["nvidia-smi", "-i", card, "--query-gpu=temperature.gpu,utilization.gpu,clocks.sm",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                self.rows.append([float(x) for x in q.strip().split(",")])
            except Exception:
                pass
            time.sleep(1)


def timed(call):
    """call() runs one step and returns a tensor. Same structure as bench_jax.timed."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    call()
    torch.cuda.synchronize()
    first = time.perf_counter() - t
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    smp = Sampler()
    smp.start()
    reps = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        for _ in range(args.steps):
            call()
        torch.cuda.synchronize()
        reps.append(1e3 * (time.perf_counter() - t) / args.steps)
    smp.on = False
    return first, reps, smp.rows


def main():
    z = np.load(args.ref)
    flat = {k[len("param:"):]: z[k] for k in z.files if k.startswith("param:")}
    g = torch.Generator(device=dev).manual_seed(0)
    by_tag = [torch.randn(S, F_, V, N, N, device=dev, generator=g) for _ in range(TAGS)]
    rng = np.random.default_rng(0)
    extra = {}

    def sample():  # the J0 / harness.py pattern: host sampling, four gathers (the fourth unused)
        v = by_tag[rng.integers(TAGS)]
        s = torch.as_tensor(rng.integers(0, S, B), device=dev)
        k = torch.as_tensor(rng.integers(1, F_ - 1, B), device=dev)
        return v[s, k - 1], v[s, k], v[s, k + 1], v[s, torch.clamp(k + 2, max=F_ - 1)]

    if args.arm in ("T0", "T1", "T2", "T3"):
        model = AuroraTorch(flat, sdpa=args.arm == "T3").to(dev)
        run = {"T0": model, "T1": torch.compile(model), "T2": torch.compile(model, mode="reduce-overhead"),
               "T3": torch.compile(model)}[args.arm]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(  # optax.cosine_decay_schedule(1e-3, 20000, alpha=1e-2)
            opt, lambda i: 0.99 * 0.5 * (1 + math.cos(math.pi * min(i, 20000) / 20000)) + 0.01)

        def call():
            xm, x0, xp, _ = sample()
            opt.zero_grad(set_to_none=True)
            loss = ((run(xm, x0) - xp) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
            return loss
    elif args.arm.startswith("fwd"):
        model = AuroraTorch(flat).to(dev).eval()
        run = model if args.arm == "fwd_eager" else torch.compile(model)
        xm, x0 = by_tag[0][:B, 0].contiguous(), by_tag[0][:B, 1].contiguous()

        def call():
            with torch.no_grad():
                return run(xm, x0)
    else:  # backbone alone: Aurora's own (A0) or our port (B0), forward + backward of a scalar loss
        x = torch.randn(B, 3 * 16 * 16, 64, device=dev, generator=g, requires_grad=False)
        if args.arm.startswith("A0"):
            from aurora.model.swin3d import Swin3DTransformerBackbone
            bb = Swin3DTransformerBackbone(embed_dim=64, encoder_depths=(2, 2, 2), encoder_num_heads=(2, 4, 8),
                                           decoder_depths=(2, 2, 2), decoder_num_heads=(8, 4, 2),
                                           window_size=(3, 4, 4)).to(dev)
            lead = torch.full((B,), 6.0, device=dev)
            fn = (lambda: bb(x, lead, rollout_step=0, patch_res=(3, 16, 16)))
            if args.arm == "A0_compile":
                cbb = torch.compile(bb)
                fn = (lambda: cbb(x, lead, rollout_step=0, patch_res=(3, 16, 16)))
            params = list(bb.parameters())
            extra["note"] = "Aurora's backbone masks shifted windows along H; ours is periodic (not the same computation)"
        else:
            model = AuroraTorch(flat).to(dev)
            fn = (lambda: model.backbone(x))
            params = list(model.parameters())

        def call():
            for p in params:
                p.grad = None
            y = fn()
            y.square().mean().backward()
            return y

    first, reps, rows = timed(call)
    rows = np.array(rows) if rows else np.zeros((1, 3))
    res = dict(arm=args.arm, batch=B, precision="float32, TF32 off", ms_per_step=reps, median_ms=float(np.median(reps)),
               first_call_s=first, steps=args.steps, warmup=args.warmup, repeats=len(reps), n_samples=len(rows),
               card=os.environ.get("CUDA_VISIBLE_DEVICES", ""), device=torch.cuda.get_device_name(0),
               torch=torch.__version__, gpu_temp_max=float(rows[:, 0].max()), gpu_util_mean=float(rows[:, 1].mean()),
               sm_clock_mean=float(rows[:, 2].mean()), **extra)
    res["git_commit"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    res["command"] = " ".join(["torchenv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{args.arm}_b{B}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "ms_per_step"}))


main()
