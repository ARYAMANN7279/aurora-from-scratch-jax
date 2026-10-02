"""E6 runs (handouts/e6_realpde_sim2real.md, sections 2-5): train and evaluate on RealPDE foil.

usage:
  run_e6.py --arm S --seed s                       real data from scratch: 22,000 steps, lr 1e-3
  run_e6.py --arm PD|PA --stage pretrain --seed s  simulations (disjoint | all): 20,000 steps, lr 1e-3
  run_e6.py --arm PD|PA --stage finetune --seed s  from that pretrain, real data: 2,000 steps, lr 1e-4
  run_e6.py --arm PD|PA --stage finetune_matched --seed s
                                                   follow-up F1: from that pretrain, S's exact recipe
                                                   (22,000 steps, lr 1e-3, cosine; S's batches for
                                                   the seed). Pass --results results/rows_f1.jsonl.
  run_e6.py --arm ZD|ZA --seed s                   zero-shot: the PD | PA pretrain on real data. Every
                                                   pretrain row already carries this evaluation.
Options: --steps N overrides the budget; --smoke runs 200 steps and scores 100 windows, writing
no row and saving under data/e6/runs/smoke_<name>; --results rows file.
Each run appends one row to the rows file and saves its parameters to data/e6/runs/<name>/.

Training samples a trajectory uniformly, then a frame, as harness.py does, but 100 steps per jitted
call (lax.scan, sampling on the device). Simulated pretraining drops p from the inputs in a random
half of those 100-step chunks (amendment A1), so that the model also runs on (u, v) alone; its loss
always covers u, v and p.

On a GPU, training sleeps between chunks to hold the card near 82 C (Throttle): the lab machine's
cards heat past the lane's 87 C pause within seconds under a sustained job. Wall time only; the
numbers are unchanged.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from aurora import backbone as bb, model  # noqa: E402

sys.path.insert(0, HERE)
from splits import HOLD, val_starts  # noqa: E402

KIT = os.environ.get("E6_KIT",  # the competition kit's scoring.py: lab path unless E6_KIT is set (Kaggle)
                     "/SML_DISK_24TB/rajeshr/Aryamann/UGP/starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py")
DATA = os.path.join(ROOT, "data", "e6")
MC = model.ModelConfig(64, 2, 3, 4, bb.Config(64, (2, 2, 2), (2, 4, 8), (2, 2, 2), (8, 4, 2), (3, 4, 4),
                                              periodic=(False, False, False)))
CHUNK = 100
BUDGET = {"S": (22000, 1e-3), "pretrain": (20000, 1e-3), "finetune": (2000, 1e-4)}


def load_source(kind):
    meta = json.load(open(os.path.join(DATA, f"{kind}_meta.json")))
    X = np.load(os.path.join(DATA, f"{kind}_frames.npy"), mmap_mode="r")
    re_ = meta["re"] if kind == "sim" else [int(n.split("_")[0]) for n in meta["names"]]
    return X, meta, np.array(re_)


def stats(X, meta, traj):
    """Per-variable mean and std over the frames of the given trajectories (float64 accumulation)."""
    s = np.zeros(X.shape[-1]), np.zeros(X.shape[-1]), 0
    for i in traj:
        a = np.asarray(X[meta["off"][i]:meta["off"][i] + meta["lens"][i]], np.float64).reshape(-1, X.shape[-1])
        s = s[0] + a.sum(0), s[1] + (a ** 2).sum(0), s[2] + a.shape[0]
    mu = s[0] / s[2]
    return mu, np.sqrt(s[1] / s[2] - mu ** 2)


def device_frames(X, meta, traj, mu, sd):
    """The chosen trajectories, normalised, channels first, concatenated; with their offsets and lengths."""
    parts, offs, lens, o = [], [], [], 0
    for i in traj:
        a = np.asarray(X[meta["off"][i]:meta["off"][i] + meta["lens"][i]], np.float32)
        parts.append(((a - mu) / sd).astype(np.float32).transpose(0, 3, 1, 2))
        offs.append(o), lens.append(a.shape[0])
        o += a.shape[0]
    return jnp.asarray(np.concatenate(parts)), jnp.asarray(offs), jnp.asarray(lens)


def make_chunk(variables, out_vars, dt, lr_sched):
    fwd = jax.vmap(partial(model.apply, variables=variables, dt=dt, mc=MC), in_axes=(None, 0, 0))
    vin = [("u", "v", "p").index(v) for v in variables]
    vout = [model.VARS.index(v) for v in out_vars]
    vtgt = [("u", "v", "p").index(v) for v in out_vars]
    opt = optax.adam(lr_sched)

    def loss_f(q, xm, x0, xp):
        return jnp.mean((fwd(q, xm[:, vin], x0[:, vin])[:, vout] - xp[:, vtgt]) ** 2)

    @jax.jit
    def chunk(q, st, key, Xd, offs, lens):
        def body(carry, key):
            q, st = carry
            k1, k2 = jax.random.split(key)
            t = jax.random.randint(k1, (16,), 0, offs.shape[0])
            k = 1 + jnp.floor(jax.random.uniform(k2, (16,)) * (lens[t] - 2)).astype(jnp.int32)
            g = offs[t] + k
            loss, grad = jax.value_and_grad(loss_f)(q, Xd[g - 1], Xd[g], Xd[g + 1])
            upd, st = opt.update(grad, st, q)
            return (optax.apply_updates(q, upd), st), loss
        (q, st), losses = jax.lax.scan(body, (q, st), jax.random.split(key, CHUNK))
        return q, st, losses
    return chunk, opt


class Throttle:
    """After each chunk, sleep long enough to hold the card near TARGET C. Off on the CPU."""
    TARGET = 82

    def __init__(self):
        self.card = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.on = jax.devices()[0].platform != "cpu" and self.card.isdigit()
        self.gap, self.slept = (0.5 if self.on else 0.0), 0.0  # start cautious

    def temp(self):
        q = ["nvidia-smi", "-i", self.card, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        try:
            return int(subprocess.run(q, capture_output=True, text=True).stdout.strip())
        except ValueError:  # no reading: keep the current gap
            return self.TARGET

    def __call__(self, ready):
        if not self.on:
            return
        jax.block_until_ready(ready)  # so the sleep is real idle time on the card
        t = self.temp()
        self.gap = min(max(self.gap + 0.05 * (t - self.TARGET), 0.0), 10.0)
        nap = self.gap + (5.0 if t >= 85 else 0.0)  # an extra 5 s when near the lane's 87 C pause
        time.sleep(nap)
        self.slept += nap


def train(q, Xd, offs, lens, steps, lr, dt, seed, drop_p):
    sched = optax.cosine_decay_schedule(lr, max(steps, 1), alpha=1e-2)
    full = ("u", "v", "p") if drop_p is not None else ("u", "v")
    c_full, opt = make_chunk(full, full, dt, sched)
    c_uv = make_chunk(("u", "v"), full, dt, sched)[0] if drop_p is not None else None
    st = opt.init(q)
    key = jax.random.PRNGKey(1000 + seed)
    rng = np.random.default_rng(seed)
    curve, thr = [], Throttle()
    for i in range(steps // CHUNK):
        key, sub = jax.random.split(key)
        fn = c_uv if (drop_p is not None and rng.random() < drop_p) else c_full
        q, st, losses = fn(q, st, sub, Xd, offs, lens)
        thr(losses)
        if (i + 1) % 20 == 0:
            curve.append(dict(step=(i + 1) * CHUNK, loss=float(losses[-1])))
            print(curve[-1], flush=True)
    return q, curve, thr.slept


def evaluate(q, X, meta, mu, sd, dt, n_windows=None):
    """The 900 re_lohi windows: 20-step rollout from the last 2 of 20 input frames, scored with the
    competition kit's own functions; plus the one-step error and persistence."""
    spec = importlib.util.spec_from_file_location("kit_scoring", KIT)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    starts = val_starts(meta)[:n_windows]
    fwd = partial(model.apply, variables=("u", "v"), dt=dt, mc=MC)

    @jax.jit
    def roll(prev, cur):
        def body(carry, _):
            a, b = carry
            c = jax.vmap(fwd, in_axes=(None, 0, 0))(q, a, b)[:, :2]
            return (b, c), c
        return jax.lax.scan(body, (prev, cur), None, length=20)[1]          # (20, n, 2, H, W)

    norm = lambda a: ((a - mu) / sd).astype(np.float32).transpose(0, 3, 1, 2)
    P, T, e1 = [], [], np.zeros(3)
    for j in range(0, len(starts), 100):
        s = starts[j:j + 100]
        w = np.stack([np.asarray(X[t:t + 40], np.float32) for t in s])      # (n, 40, H, W, 2)
        pred = np.asarray(roll(jnp.asarray(np.stack([norm(x[18:19])[0] for x in w])),
                               jnp.asarray(np.stack([norm(x[19:20])[0] for x in w]))))
        pred = pred.transpose(1, 0, 3, 4, 2) * sd.astype(np.float32) + mu.astype(np.float32)  # (n, 20, H, W, 2)
        P.append(pred), T.append(w[:, 20:])
        e1 += [((pred[:, 0] - w[:, 20]) ** 2).sum(), ((w[:, 19] - w[:, 20]) ** 2).sum(), (w[:, 20] ** 2).sum()]
    P, T = np.concatenate(P), np.concatenate(T)
    z = np.zeros(P.shape[:-1] + (1,), np.float32)
    P3, T3 = np.concatenate([P, z], -1), np.concatenate([T, z], -1)        # as r12_eval.window: p channel zero
    err = dict(rel_l2=float(S.rel_l2_per_sample(P3, T3, 2).mean()), tke=float(S.tke_rel_l2_per_sample(P3, T3, 2).mean()),
               mvpe=float(S.mvpe_rel_l2_per_sample(P3, T3).mean()))
    out = dict(windows=len(starts), rollout=err, scores={k: S.score_error(v) for k, v in err.items()},
               one_step=dict(rel_err=float(np.sqrt(e1[0] / e1[2])), persistence=float(np.sqrt(e1[1] / e1[2])),
                             ratio=float(np.sqrt(e1[0] / e1[1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["S", "PD", "PA", "ZD", "ZA"])
    ap.add_argument("--stage", choices=["pretrain", "finetune", "finetune_matched"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "rows.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps = a.steps or 200
    t0 = time.time()
    Xr, mr, rer = load_source("real")
    train_real = [i for i in range(len(mr["names"])) if rer[i] not in HOLD]
    mu_r, sd_r = stats(Xr, mr, train_real)
    pretrain_of = {"PD": "PD", "PA": "PA", "ZD": "PD", "ZA": "PA"}  # whose pretrained weights an arm starts from
    if a.arm in ("PD", "PA") and a.stage is None:
        raise SystemExit("--stage pretrain|finetune is required for PD and PA")
    name = ("smoke_" if a.smoke else "") + f"e6_{a.arm}" + (f"_{a.stage}" if a.stage else "") + f"_s{a.seed}"
    _, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), MC, dtype=jnp.float32))
    load = lambda n: unravel(jnp.asarray(np.load(os.path.join(DATA, "runs", n, "params.npz"))["flat"]))
    curve, slept, row = [], 0.0, dict(name=name, arm=a.arm, stage=a.stage, seed=a.seed)
    if a.arm in ("PD", "PA") and a.stage == "pretrain":
        Xs, ms, res = load_source("sim")
        traj = [i for i in range(len(ms["names"])) if a.arm == "PA" or res[i] not in HOLD]
        mu_s, sd_s = stats(Xs, ms, traj)
        steps, lr = a.steps or BUDGET["pretrain"][0], BUDGET["pretrain"][1]
        q = model.init(jax.random.PRNGKey(a.seed), MC, dtype=jnp.float32)
        Xd, offs, lens = device_frames(Xs, ms, traj, mu_s, sd_s)
        dt = float(np.mean(ms["dt"]))
        q, curve, slept = train(q, Xd, offs, lens, steps, lr, dt, a.seed, drop_p=0.5)
        row.update(train_trajectories=len(traj), source="sim", norm_mean=mu_s.tolist(), norm_std=sd_s.tolist())
        del Xd
    else:
        steps, lr = (a.steps or BUDGET["S"][0], BUDGET["S"][1]) if a.arm == "S" or a.stage == "finetune_matched" else (
            (a.steps or BUDGET["finetune"][0], BUDGET["finetune"][1]) if a.stage == "finetune" else (0, 0.0))
        q = (model.init(jax.random.PRNGKey(a.seed), MC, dtype=jnp.float32) if a.arm == "S"
             else load(("smoke_" if a.smoke else "") + f"e6_{pretrain_of[a.arm]}_pretrain_s{a.seed}"))
        if steps:
            Xd, offs, lens = device_frames(Xr, mr, train_real, mu_r, sd_r)
            q, curve, slept = train(q, Xd, offs, lens, steps, lr, 0.02, a.seed, drop_p=None)
            del Xd
        row.update(train_trajectories=len(train_real) if steps else 0, source="real")
    row.update(steps=steps, lr=lr, curve=curve, eval=evaluate(q, Xr, mr, mu_r, sd_r, 0.02, 100 if a.smoke else None),
               real_norm_mean=mu_r.tolist(),
               real_norm_std=sd_r.tolist(), wall_s=time.time() - t0, throttle_sleep_s=slept,
               device=jax.devices()[0].device_kind,
               git_commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
               command=" ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]))
    os.makedirs(os.path.join(DATA, "runs", name), exist_ok=True)
    flat, _ = jax.flatten_util.ravel_pytree(q)
    np.savez(os.path.join(DATA, "runs", name, "params.npz"), flat=np.asarray(flat))
    if not a.smoke:
        os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
        with open(a.results, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("name", "eval", "wall_s")}))


main()
