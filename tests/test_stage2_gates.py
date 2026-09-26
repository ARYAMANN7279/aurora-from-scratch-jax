"""Stage 2 gates D1-D8 (handouts/stage02_dataset.md, section 3), on a tiny dataset:
2 regimes x 2 seeds, N_gen = 64, N_out in {16, 32}, T_spin = 1, T_rec = 2, save_dt = 0.2.
"""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from kdata import WindowDataset, finalize, generate_dataset, generate_trajectory, make_splits, sha256, traj_dir
from kolmogorov import Params, choose_m, make_advance, make_grid, random_ic, resample, rfft2, irfft2
from kolmogorov.diagnostics import norm
from kolmogorov.ops import pressure_hat, vel_hat

CFG = dict(name="tiny", regimes=[dict(Re=100, kf=4, N_gen=64), dict(Re=300, kf=4, N_gen=64)],
           seeds=[0, 1], T_spin=1.0, T_rec=2.0, save_dt=0.2, N_out=[16, 32])
N_FRAMES = 11


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = str(tmp_path_factory.mktemp("tiny"))
    generate_dataset(CFG, r, verbose=False)
    finalize(r)
    return r


def frames(root, tag, seed, n):
    return np.load(os.path.join(traj_dir(root, tag, seed), f"frames_{n}.npy"))


def test_D1_regeneration_is_bitwise_identical(root, tmp_path):
    generate_trajectory(str(tmp_path), 300, 4, 1, 64, 1.0, 2.0, 0.2, (16, 32), verbose=False)
    for n in (16, 32):
        a = frames(root, "re300_kf4", 1, n)
        b = frames(str(tmp_path), "re300_kf4", 1, n)
        assert a.shape == (N_FRAMES, 4, n, n) and a.dtype == np.float32
        assert np.array_equal(a, b)


def test_D2_generator_matches_stage1_solver(tmp_path):
    _, wh = generate_trajectory(str(tmp_path), 100, 4, 0, 64, 1.0, 2.0, 0.2, (16, 32), verbose=False)
    prm, g = Params(nu=0.01), make_grid(64)
    m = choose_m(64, prm)
    n_total = 10 * m + (N_FRAMES - 1) * 2 * m
    ref = make_advance(g, prm, 0.1 / m, n_total)(random_ic(g, seed=0))
    err = float(norm(g, wh - ref) / norm(g, ref))
    report("D2 generator vs make_advance", err, 1e-12)
    assert err <= 1e-12


def _derived_uv(w32):
    n = w32.shape[-1]
    g = make_grid(n)
    wh = rfft2(jnp.asarray(w32, dtype=jnp.float64))
    uh, vh = vel_hat(g, wh)
    return g, wh, irfft2(uh, n), irfft2(vh, n)


def test_D3_D4_stored_velocity_consistent_and_solenoidal(root):
    e3 = e4 = 0.0
    for tag in ("re100_kf4", "re300_kf4"):
        for n in (16, 32):
            f = frames(root, tag, 0, n).astype(np.float64)
            for k in (0, N_FRAMES - 1):
                u, v, w = f[k, 0], f[k, 1], f[k, 3]
                g, _, ud, vd = _derived_uv(w)
                e3 = max(e3, float(max(jnp.max(jnp.abs(ud - u)), jnp.max(jnp.abs(vd - v))) / np.abs(u).max()))
                div = irfft2(1j * g.kx * rfft2(jnp.asarray(u)) + 1j * g.ky * rfft2(jnp.asarray(v)), n)
                e4 = max(e4, float(jnp.max(jnp.abs(div))) / float(np.sqrt(np.mean(w**2))))
    report("D3 stored u,v vs derived from stored omega (rel max)", e3, 1e-5)
    report("D4 divergence of stored u,v / rms(omega)", e4, 1e-5)
    assert e3 <= 1e-5 and e4 <= 1e-5


def test_D5_pressure_matches_independent_solve(root):
    prm, g = Params(nu=1 / 300), make_grid(64)
    m = choose_m(64, prm)
    wh = make_advance(g, prm, 0.1 / m, 10 * m)(random_ic(g, seed=0))  # state at the first frame
    err = 0.0
    for n in (16, 32):
        p_ref = np.asarray(irfft2(resample(pressure_hat(g, wh), n), n))
        p = frames(root, "re300_kf4", 0, n)[0, 2]
        err = max(err, float(np.abs(p - p_ref).max() / np.abs(p_ref).max()))
    report("D5 stored p vs independent solve (rel max)", err, 1e-5)
    assert err <= 1e-5


def test_D6_windows(root):
    members = [("re100_kf4", 0), ("re100_kf4", 1), ("re300_kf4", 0)]
    for stride, n_in, n_out in ((1, 2, 1), (2, 2, 1), (3, 1, 2)):
        ds = WindowDataset(root, 32, members, stride=stride, n_in=n_in, n_out=n_out)
        span = (n_in + n_out - 1) * stride
        assert len(ds) == len(members) * (N_FRAMES - span)
        full = [frames(root, t, s, 32) for t, s in members]
        for j in range(len(ds)):
            i, t = ds.index[j]
            x, y = ds.window(j)
            want = full[i][[t + k * stride for k in range(n_in + n_out)]]
            assert np.array_equal(np.concatenate([x, y]), want)
            assert t + span < N_FRAMES  # never crosses into another trajectory


def test_D7_splits_order_and_stats(root):
    with pytest.raises(ValueError):
        make_splits(root, train=["re100_kf4", "re300_kf4"], test=["re300_kf4"])
    with pytest.raises(ValueError):
        make_splits(root, train=[{"tag": "re100_kf4", "seeds": [0, 1]}], val=[{"tag": "re100_kf4", "seeds": [1]}])
    sp = make_splits(root, train=["re100_kf4"], test=["re300_kf4"])
    assert sp["train"] == [("re100_kf4", 0), ("re100_kf4", 1)] and sp["test"] == [("re300_kf4", 0), ("re300_kf4", 1)]
    ds = WindowDataset(root, 16, sp["train"])
    order = lambda s: np.concatenate([x[:, 0, 3, 0, 0] for x, _ in ds.batches(4, seed=s)])
    assert np.array_equal(order(0), order(0)) and not np.array_equal(order(0), order(1))
    mean, std = ds.channel_stats()
    allf = np.concatenate([frames(root, t, s, 16) for t, s in sp["train"]]).astype(np.float64)
    ref_m, ref_s = allf.mean(axis=(0, 2, 3)), allf.std(axis=(0, 2, 3))
    err = max(np.abs(mean - ref_m).max() / ref_s.max(), np.abs(std / ref_s - 1).max())
    report("D7 normalisation stats vs training frames", err, 1e-6)
    assert err <= 1e-6


def test_D8_manifest_checksums(root):
    with open(os.path.join(root, "manifest.json")) as f:
        man = json.load(f)
    assert len(man["trajectories"]) == 4
    for t in man["trajectories"]:
        d = traj_dir(root, t["tag"], t["seed"])
        for name, h in t["sha256"].items():
            assert sha256(os.path.join(d, name)) == h


def test_D9_restart_safety(tmp_path):
    """Added after the container restart of 26 Sep (not in the original spec): a chunked spin-up is
    bitwise identical to a single scan, and an interrupted-then-resumed trajectory gives the same
    files as an uninterrupted one."""
    prm, g = Params(nu=0.01), make_grid(64)
    m = choose_m(64, prm)
    ic = random_ic(g, seed=3)
    one = make_advance(g, prm, 0.1 / m, 7 * m)(ic)
    two = make_advance(g, prm, 0.1 / m, 3 * m)(make_advance(g, prm, 0.1 / m, 4 * m)(ic))
    assert jnp.array_equal(one, two)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    ma, _ = generate_trajectory(a, 100, 4, 2, 64, 12.0, 2.0, 0.2, (16, 32), chunk_frames=4, verbose=False)
    with pytest.raises(InterruptedError):
        generate_trajectory(b, 100, 4, 2, 64, 12.0, 2.0, 0.2, (16, 32), chunk_frames=4, verbose=False,
                            _stop_after_chunks=1)
    assert os.path.exists(os.path.join(traj_dir(b, "re100_kf4", 2), "ckpt.npz"))
    mb, _ = generate_trajectory(b, 100, 4, 2, 64, 12.0, 2.0, 0.2, (16, 32), chunk_frames=4, verbose=False)
    assert ma["sha256"] == mb["sha256"]
    assert not os.path.exists(os.path.join(traj_dir(b, "re100_kf4", 2), "ckpt.npz"))
