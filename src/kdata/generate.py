"""Trajectory generator (handouts/stage02_dataset.md, section 2).

A trajectory is (regime, seed): Stage 1 random IC on the generation grid, T_spin discarded, then
frames (u, v, p, omega) every save_dt, spectrally truncated to each output resolution, float32.
"""
import hashlib
import json
import math
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from kolmogorov import Params, choose_m, irfft2, make_advance, make_grid, make_step, random_ic, resample
from kolmogorov.diagnostics import make_diagnostics, shell_spectrum
from kolmogorov.ops import pressure_hat, vel_hat
from kolmogorov.solver import SAVE_DT, cfl

CHANNELS = ("u", "v", "p", "omega")


def regime_tag(Re, kf):
    return f"re{int(Re) if float(Re).is_integer() else Re}_kf{int(kf)}"


def traj_dir(root, tag, seed):
    return os.path.join(root, tag, f"seed_{seed:04d}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def make_frame_fn(g, prm, dt, out_res):
    diag, _ = make_diagnostics(g, prm)

    def frame(wh):
        uh, vh = vel_hat(g, wh)
        ph = pressure_hat(g, wh)
        outs = tuple(jnp.stack([irfft2(resample(x, n), n) for x in (uh, vh, ph, wh)]).astype(jnp.float32)
                     for n in out_res)
        E, Z, _, P, _ = diag(wh)
        d = dict(E=E, Z=Z, P=P, cfl=cfl(g, wh, dt), spec=shell_spectrum(g, wh))
        return outs, d

    return frame


def generate_trajectory(root, Re, kf, seed, N_gen, T_spin=50.0, T_rec=100.0, save_dt=0.2,
                        out_res=(64, 128), chunk_frames=50, verbose=True, _stop_after_chunks=None):
    """Generate one trajectory into root/<tag>/seed_<seed>/. Returns (meta, final generation-grid state)."""
    t_start = time.time()
    prm = Params(nu=1.0 / Re, kf=kf)
    g = make_grid(N_gen)
    m = choose_m(N_gen, prm)
    dt = SAVE_DT / m
    per_frame = round(save_dt / SAVE_DT) * m
    if abs(per_frame * dt - save_dt) > 1e-12 or any(n > N_gen for n in out_res):
        raise ValueError("save_dt must be a multiple of 0.1 and out_res <= N_gen")
    n_spin = round(T_spin / SAVE_DT) * m
    n_frames = round(T_rec / save_dt) + 1
    tag = regime_tag(Re, kf)
    d = traj_dir(root, tag, seed)
    os.makedirs(d, exist_ok=True)

    step = make_step(g, prm, dt)
    frame = jax.jit(make_frame_fn(g, prm, dt, out_res))

    def body(wh, _):
        wh = jax.lax.scan(lambda w, _: (step(w), None), wh, None, length=per_frame)[0]
        return wh, frame(wh)

    chunk = jax.jit(lambda wh, n: jax.lax.scan(body, wh, None, length=n), static_argnums=1)

    # Restart safety: the generation-grid state is checkpointed after every spin-up chunk and every
    # recording chunk (ckpt.npz, written atomically). Chunking does not change a single bit of the
    # result (gate D9), so an interrupted trajectory resumes to the same files as an uninterrupted one.
    ckpt_path = os.path.join(d, "ckpt.npz")
    paths = {n: os.path.join(d, f"frames_{n}.npy") for n in out_res}
    diag_keys = ("E", "Z", "P", "cfl", "spec")
    ck = dict(np.load(ckpt_path)) if os.path.exists(ckpt_path) else None
    if ck is not None and not all(os.path.exists(paths[n] + ".part") for n in out_res) and int(ck["i"]) > 0:
        ck = None  # frames missing: start over

    def save_ckpt(wh, spin_done, i, diags):
        tmp = ckpt_path + ".tmp.npz"
        extra = {k: np.concatenate(v) for k, v in diags.items() if v} if i else {}
        np.savez(tmp, wh=np.asarray(wh), spin_done=spin_done, i=i, **extra)
        os.replace(tmp, ckpt_path)

    spin_chunk = round(10.0 / SAVE_DT) * m  # 10 time units
    if ck is None:
        wh, spin_done = random_ic(g, seed=seed), 0
    else:
        wh, spin_done = jnp.asarray(ck["wh"]), int(ck["spin_done"])
    while spin_done < n_spin:
        n = min(spin_chunk, n_spin - spin_done)
        wh = make_advance(g, prm, dt, n)(wh)
        spin_done += n
        save_ckpt(wh, spin_done, 0, {})
        if verbose:
            print(f"  {tag} seed {seed}: spin-up {spin_done}/{n_spin} steps", flush=True)

    resume = ck is not None and int(ck["i"]) > 0
    mm = {n: np.lib.format.open_memmap(paths[n] + ".part", mode="r+" if resume else "w+", dtype=np.float32,
                                        shape=(n_frames, 4, n, n)) for n in out_res}
    diags = {k: ([ck[k]] if resume else []) for k in diag_keys}

    def store(i0, outs, dg):
        for n, o in zip(out_res, outs):
            o = np.asarray(o)
            mm[n][i0:i0 + o.shape[0]] = o
        for k in diags:
            diags[k].append(np.atleast_1d(np.asarray(dg[k])) if k != "spec" else np.asarray(dg[k]).reshape(-1, g.n_shells))

    if resume:
        i = int(ck["i"])
    else:
        outs, dg = frame(wh)
        store(0, [o[None] for o in outs], dg)
        i = 1
    while i < n_frames:
        n = min(chunk_frames, n_frames - i)
        wh, (outs, dg) = chunk(wh, n)
        store(i, outs, dg)
        i += n
        for mmap in mm.values():
            mmap.flush()
        save_ckpt(wh, spin_done, i, diags)
        if verbose:
            print(f"  {tag} seed {seed}: frame {i}/{n_frames}, E={float(dg['E'][-1]):.4f}", flush=True)
        if _stop_after_chunks is not None:  # test hook: simulate a crash after this many chunks
            _stop_after_chunks -= 1
            if _stop_after_chunks == 0:
                raise InterruptedError("simulated interruption")

    dg = {k: np.concatenate(v) for k, v in diags.items()}
    max_cfl = float(dg["cfl"].max())
    if max_cfl > 0.7:
        raise RuntimeError(f"{tag} seed {seed}: CFL {max_cfl:.3f} > 0.7, trajectory void")
    for n in out_res:
        mm[n].flush()
        del mm[n]
        os.replace(paths[n] + ".part", paths[n])
    t = T_spin + save_dt * np.arange(n_frames)
    diag_path = os.path.join(d, "diag.npz")
    np.savez(diag_path, t=t, **dg)

    files = {os.path.basename(p): sha256(p) for p in list(paths.values()) + [diag_path]}
    meta = dict(tag=tag, Re=Re, kf=kf, seed=seed, nu=prm.nu, alpha=prm.alpha, A=prm.A, N_gen=N_gen,
                m=m, dt=dt, T_spin=T_spin, T_rec=T_rec, save_dt=save_dt, t0=T_spin, n_frames=n_frames,
                out_res=list(out_res), channels=list(CHANNELS), max_cfl=max_cfl,
                dtype=str(g.k2.dtype), sha256=files, wall_s=time.time() - t_start,
                git_commit=git_commit())
    with open(os.path.join(d, "meta.json"), "w") as f:  # written last: marks the trajectory complete
        json.dump(meta, f, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return meta, wh


def load_config(path):
    with open(path) as f:
        return json.load(f)


def tasks(cfg):
    return [(r, s) for r in cfg["regimes"] for s in cfg["seeds"]]


def generate_dataset(cfg, root, shard=0, n_shards=1, verbose=True):
    os.makedirs(root, exist_ok=True)
    cfg_path = os.path.join(root, "dataset.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)
    done = []
    for i, (r, s) in enumerate(tasks(cfg)):
        if i % n_shards != shard:
            continue
        d = traj_dir(root, regime_tag(r["Re"], r["kf"]), s)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, _ = generate_trajectory(root, r["Re"], r["kf"], s, r["N_gen"], cfg["T_spin"], cfg["T_rec"],
                                      cfg["save_dt"], tuple(cfg["N_out"]), verbose=verbose)
        done.append(meta)
        if verbose:
            print(f"done {meta['tag']} seed {s} in {meta['wall_s']:.0f}s", flush=True)
    return done


def finalize(root):
    """Collect every trajectory's meta into manifest.json after verifying all sha256."""
    cfg = load_config(os.path.join(root, "dataset.json"))
    entries = []
    for r, s in tasks(cfg):
        d = traj_dir(root, regime_tag(r["Re"], r["kf"]), s)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        for name, h in meta["sha256"].items():
            if sha256(os.path.join(d, name)) != h:
                raise RuntimeError(f"checksum mismatch: {d}/{name}")
        entries.append(meta)
    man = dict(config=cfg, trajectories=entries)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man
