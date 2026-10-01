"""Stage 12 JAX arms (handouts/stage12_speed.md, section 3) on the E1 pretrain configuration.

usage: bench_jax.py --arm J0|J1|J2|J3|J4|J5|J6|J6t|fwd [--batch 16]   (timing, on one idle card)
       bench_jax.py --accuracy                                         (the forward-pass check of S12-P6)
       bench_jax.py --export                                           (reference for check_torch.py)
Results: results/<arm>_b<batch>.json, results/accuracy_bf16.json; the export goes to data/stage12/.

J6 sets matmul precision bfloat16 (each matmul's inputs rounded to bfloat16, float32 accumulation);
elementwise ops and FFTs stay float32. J6t (TF32) is a diagnostic without a prediction.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="J0", choices=["J0", "J1", "J2", "J3", "J4", "J5", "J6", "J6t", "fwd"])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--accuracy", action="store_true")
ap.add_argument("--export", action="store_true", help="trained E1 weights, a real batch, output and gradients")
ap.add_argument("--f64", action="store_true", help="export in float64 (exactness check of the PyTorch port)")
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--scan", type=int, default=50, help="J3: steps per jitted call")
ap.add_argument("--smoke", action="store_true", help="dry run with tiny counts; writes no result")
args = ap.parse_args()
PREC = {"J6": "bfloat16", "J6t": "tensorfloat32"}.get(args.arm, "highest")
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = PREC  # before jax is imported
if args.f64:
    os.environ["JAX_ENABLE_X64"] = "1"

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from aurora import model, swin  # noqa: E402
from aurora.harness import model_config  # noqa: E402
from kdata import traj_dir  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "experiments", "e1", "configs", "pretrain.json")))[0]
MC = model_config(CFG["model"])
F, S, TAGS, V, N = 501, 16, 2, len(model.VARS), 64
fwd = jax.vmap(partial(model.apply, variables=model.VARS, dt=0.2, mc=MC), in_axes=(None, 0, 0))
opt = optax.adam(optax.cosine_decay_schedule(1e-3, 20000, alpha=1e-2))


def objective(q, xm, x0, xp, xp2):  # xp2 is unused without pushforward, exactly as in harness.py
    return jnp.mean((fwd(q, xm, x0) - xp) ** 2)


def raw_step(q, st, xm, x0, xp, xp2):
    loss, g = jax.value_and_grad(objective)(q, xm, x0, xp, xp2)
    upd, st = opt.update(g, st, q)
    return optax.apply_updates(q, upd), st, loss


def window_attention_dpa(p, xw, heads, mask=None):
    """J5: swin.window_attention through jax.nn.dot_product_attention (same maths, BTNH layout)."""
    Bw, Nw, D = xw.shape
    qkv = (xw @ p["qkv"]["W"] + p["qkv"]["b"]).reshape(Bw, Nw, 3, heads, D // heads)
    m = None if mask is None else jnp.tile(mask, (Bw // mask.shape[0], 1, 1))[:, None]
    o = jax.nn.dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=m, implementation="xla")
    return o.reshape(Bw, Nw, D) @ p["proj"]["W"] + p["proj"]["b"]


def card():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


class Sampler(threading.Thread):
    """Samples the card's temperature, utilisation and SM clock once a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.on = [], True

    def run(self):
        while self.on:
            try:
                q = subprocess.run(["nvidia-smi", "-i", card(), "--query-gpu=temperature.gpu,utilization.gpu,clocks.sm",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                self.rows.append([float(x) for x in q.strip().split(",")])
            except Exception:
                pass
            time.sleep(1)


def timed(run_n, per_call=1):
    """run_n(n) makes n calls of per_call steps each. Returns the first call's time (compile + run),
    ms per step for each repeat, and the card samples taken while the repeats ran."""
    t = time.perf_counter()
    jax.block_until_ready(run_n(1))
    first = time.perf_counter() - t
    jax.block_until_ready(run_n(max(args.warmup // per_call, 1)))
    calls = max(args.steps // per_call, 1)
    smp = Sampler()
    smp.start()
    reps = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        jax.block_until_ready(run_n(calls))
        reps.append(1e3 * (time.perf_counter() - t) / (calls * per_call))
    smp.on = False
    return first, reps, smp.rows


def trained_e1():
    """The trained e1_pre_n32_T128_s0 model (experiments/e1/results/rows_2.jsonl) and its test data."""
    run = "2f09979ae05dea9b"
    z = np.load(os.path.join(ROOT, "data", "runs", run, "params.npz"))
    _, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32))
    mu, sd = z["mean"][:, None, None], z["std"][:, None, None]
    ds = os.path.join(ROOT, "data", "stage2", "e1_v1")
    vs = [jnp.asarray(((np.load(os.path.join(traj_dir(ds, "re300_kf4", s), "frames_64.npy")) - mu) / sd)
                      .astype(np.float32)) for s in (2, 3, 4, 5)]
    return run, unravel(jnp.asarray(z["flat"])), vs


def path_key(path):
    return "/".join(str(k.key) if hasattr(k, "key") else str(k.idx) for k in path)


def export():
    """Weights, a fixed real batch, the forward output, the MSE loss and its gradients, all float32."""
    run, p, vs = trained_e1()
    k = np.array([10, 60, 110, 160])
    xm, x0, xp = vs[0][k - 1], vs[0][k], vs[0][k + 1]
    if args.f64:
        p, xm, x0, xp = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), (p, xm, x0, xp))
    loss_f = lambda prm: jnp.mean((fwd(prm, xm, x0) - xp) ** 2)
    y = jax.jit(fwd)(p, xm, x0)
    loss, g = jax.jit(jax.value_and_grad(loss_f))(p)
    out = {"xm": xm, "x0": x0, "xp": xp, "y": y, "loss": loss}
    for path, leaf in jax.tree_util.tree_flatten_with_path(p)[0]:
        out["param:" + path_key(path)] = leaf
    for path, leaf in jax.tree_util.tree_flatten_with_path(g)[0]:
        out["grad:" + path_key(path)] = leaf
    os.makedirs(os.path.join(ROOT, "data", "stage12"), exist_ok=True)
    dest = os.path.join(ROOT, "data", "stage12", f"ref_e1_{jax.default_backend()}{'_f64' if args.f64 else ''}.npz")
    np.savez(dest, **{k: np.asarray(v) for k, v in out.items()})
    return dict(run=run, file=os.path.relpath(dest, ROOT), backend=jax.default_backend(), dtype=str(y.dtype), loss=float(loss),
                n_params=len([k for k in out if k.startswith("param:")]))


def accuracy():
    """S12-P6: one-step test error of a trained E1 pretrain model, float32 against bfloat16 matmuls."""
    run, p, vs = trained_e1()
    out = {}
    for prec in ("highest", "bfloat16"):
        with jax.default_matmul_precision(prec):
            sq = jax.jit(lambda prm, xm, x0, xp: (jnp.sum((fwd(prm, xm, x0) - xp) ** 2), jnp.sum(xp**2)))
            a = c = 0.0
            for v in vs:
                for j in range(1, v.shape[0] - 1, 16):
                    k = np.arange(j, min(j + 16, v.shape[0] - 1))
                    e, n = sq(p, v[k - 1], v[k], v[k + 1])
                    a, c = a + float(e), c + float(n)
        out[prec] = float(np.sqrt(a / c))
    rel = out["bfloat16"] / out["highest"] - 1
    res = dict(run=run, committed_rel_err=0.08398959236568616, rel_err=out, rel_change=rel,
               S12_P6_accuracy=dict(bar="|change| < 1%", hit=abs(rel) < 0.01), device=jax.devices()[0].device_kind)
    print(json.dumps(res, indent=1))
    return res


def main():
    if args.export:
        print(json.dumps(export()))
        return
    if args.accuracy:
        res = accuracy()
        name = "accuracy_bf16"
    else:
        B = args.batch
        keys = jax.random.split(jax.random.PRNGKey(0), TAGS + 1)
        by_tag = [jax.random.normal(k, (S, F, V, N, N), jnp.float32) for k in keys[:TAGS]]
        q = model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32)
        st = opt.init(q)
        rng = np.random.default_rng(0)
        extra = {}

        def sample():
            v = by_tag[rng.integers(TAGS)]
            s, k = rng.integers(0, S, B), rng.integers(1, F - 1, B)
            return v[s, k - 1], v[s, k], v[s, k + 1], v[s, np.minimum(k + 2, F - 1)]

        if args.arm == "J5":
            ref = jax.jit(fwd)(q, by_tag[0][:B, 0], by_tag[0][:B, 1])
            swin.window_attention = window_attention_dpa
            new = jax.jit(fwd)(q, by_tag[0][:B, 0], by_tag[0][:B, 1])
            extra["max_rel_diff_vs_J0_forward"] = float(jnp.max(jnp.abs(new - ref)) / jnp.max(jnp.abs(ref)))

        if args.arm in ("J0", "J1", "J2", "J5", "J6", "J6t"):
            step = jax.jit(raw_step, donate_argnums=(0, 1) if args.arm == "J2" else ())
            state = [q, st]

            def run_n(n):
                for _ in range(n):
                    state[0], state[1], loss = step(state[0], state[1], *sample())
                return state[0]
            per_call = 1
        elif args.arm == "J3":
            data, K = jnp.stack(by_tag), args.scan

            def chunk(q, st, key, data):  # data is an argument: a closed-over 2 GB array would become a constant
                def body(carry, key):
                    q, st = carry
                    k1, k2, k3 = jax.random.split(key, 3)
                    v = data[jax.random.randint(k1, (), 0, TAGS)]
                    s, k = jax.random.randint(k2, (B,), 0, S), jax.random.randint(k3, (B,), 1, F - 1)
                    q, st, loss = raw_step(q, st, v[s, k - 1], v[s, k], v[s, k + 1], v[s, jnp.minimum(k + 2, F - 1)])
                    return (q, st), loss
                (q, st), losses = jax.lax.scan(body, (q, st), jax.random.split(key, K))
                return q, st, losses

            chunk = jax.jit(chunk)
            state = [q, st, jax.random.PRNGKey(1)]

            def run_n(n):
                for _ in range(n):
                    state[2], sub = jax.random.split(state[2])
                    state[0], state[1], _ = chunk(state[0], state[1], sub, data)
                return state[0]
            per_call = K
            extra["steps_per_call"] = K
        elif args.arm == "J4":
            qs = jax.vmap(lambda k: model.init(k, MC, dtype=jnp.float32))(jax.random.split(jax.random.PRNGKey(0), 3))
            sts = jax.vmap(opt.init)(qs)
            data = jnp.stack(by_tag)
            step3 = jax.jit(jax.vmap(raw_step))
            state = [qs, sts]

            def run_n(n):
                for _ in range(n):
                    ti = rng.integers(0, TAGS, (3, 1))
                    s, k = rng.integers(0, S, (3, B)), rng.integers(1, F - 1, (3, B))
                    xs = [data[ti, s, k + d] for d in (-1, 0, 1)] + [data[ti, s, np.minimum(k + 2, F - 1)]]
                    state[0], state[1], _ = step3(state[0], state[1], *xs)
                return state[0]
            per_call = 1
            extra["models_per_step"] = 3
        else:  # fwd: forward pass alone on a fixed batch
            f = jax.jit(fwd)
            xm, x0 = by_tag[0][:B, 0], by_tag[0][:B, 1]

            def run_n(n):
                for _ in range(n):
                    out = f(q, xm, x0)
                return out
            per_call = 1

        first, reps, smp_rows = timed(run_n, per_call)
        rows = np.array(smp_rows) if smp_rows else np.zeros((1, 3))
        res = dict(arm=args.arm, batch=B, matmul_precision=PREC, ms_per_step=reps, median_ms=float(np.median(reps)),
                   first_call_s=first, steps=args.steps, warmup=args.warmup, repeats=len(reps), n_samples=len(smp_rows),
                   card=card(), device=jax.devices()[0].device_kind, jax=jax.__version__,
                   gpu_temp_max=float(rows[:, 0].max()), gpu_util_mean=float(rows[:, 1].mean()),
                   sm_clock_mean=float(rows[:, 2].mean()), **extra)
        name = f"{args.arm}_b{B}"
    res["git_commit"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    res["command"] = " ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])
    if args.smoke:
        print("smoke ok", name, res.get("median_ms"))
        return
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "ms_per_step"}))


main()
