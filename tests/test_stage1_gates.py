"""Stage 1 gates G1-G11 (handouts/stage01_solver.md, section 5.1). float64 throughout.

Each gate prints its measured value; run with `pytest -s` to see them.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from kolmogorov.diagnostics import inner, make_diagnostics, norm, shell_spectrum
from kolmogorov.grid import irfft2, make_grid, resample, rfft2
from kolmogorov.ops import Params, advection, nonlinear, nonlinear_unmasked, pressure_hat, psi_hat, vel_hat
from kolmogorov.solver import make_advance, make_step


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def mesh(g):
    Y, X = np.meshgrid(np.asarray(g.y), np.asarray(g.x), indexing="ij")
    return X, Y


def random_field(g, seed=0, band_limited=True):
    """Random real field, rms 1. band_limited: projected with M. Otherwise Nyquist zeroed only."""
    rng = np.random.default_rng(seed)
    fh = rfft2(jnp.asarray(rng.standard_normal((g.N, g.N))))
    if band_limited:
        fh = g.mask * fh
    else:
        nyq = (jnp.abs(g.kx) == g.N // 2) | (jnp.abs(g.ky) == g.N // 2) | (g.k2 == 0)
        fh = jnp.where(nyq, 0, fh)
    return fh / norm(g, fh)


def exact_product_hat(g, ah, bh):
    """Alias-free product: zero-pad both to 2N, multiply, transform, truncate to N."""
    n2 = 2 * g.N
    p = irfft2(resample(ah, n2), n2) * irfft2(resample(bh, n2), n2)
    return resample(rfft2(p), g.N)


def test_G1_fft_round_trip():
    g = make_grid(64)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((64, 64)))
    err = float(jnp.max(jnp.abs(irfft2(rfft2(x), 64) - x)))
    report("G1 FFT round trip", err, 1e-14)
    assert err <= 1e-14


def test_G2_divergence_and_curl():
    g = make_grid(64)
    wh = random_field(g)
    uh, vh = vel_hat(g, wh)
    div = float(jnp.max(jnp.abs(irfft2(1j * g.kx * uh + 1j * g.ky * vh, g.N))))
    curl = float(jnp.max(jnp.abs(irfft2(1j * g.kx * vh - 1j * g.ky * uh - wh, g.N))))
    report("G2 div u", div, 1e-12)
    report("G2 curl u - omega", curl, 1e-12)
    assert div <= 1e-12 and curl <= 1e-12


def test_G3_parseval_and_shell_sum():
    g = make_grid(64)
    wh = random_field(g)
    w = irfft2(wh, g.N)
    phys = float(jnp.mean(w**2))
    spec = float(inner(g, wh, wh))
    e_parseval = abs(phys - spec) / phys
    diag, _ = make_diagnostics(g, Params(nu=0.01))
    E = float(diag(wh)[0])
    u, v = irfft2(vel_hat(g, wh)[0], g.N), irfft2(vel_hat(g, wh)[1], g.N)
    E_phys = 0.5 * float(jnp.mean(u**2 + v**2))
    e_energy = abs(E - E_phys) / E_phys
    e_shell = abs(float(jnp.sum(shell_spectrum(g, wh))) - E) / E
    report("G3 Parseval <w^2>", e_parseval, 1e-12)
    report("G3 Parseval E", e_energy, 1e-12)
    report("G3 shell sum", e_shell, 1e-12)
    assert max(e_parseval, e_energy, e_shell) <= 1e-12


def test_G4_nonlinear_vs_sympy():
    x, y = sp.symbols("x y", real=True)
    psi = sp.sin(x) * sp.cos(2 * y) + sp.Rational(7, 10) * sp.cos(3 * x) * sp.sin(y + sp.Rational(2, 5)) \
        + sp.Rational(3, 10) * sp.sin(2 * x + y)
    u, v = sp.diff(psi, y), -sp.diff(psi, x)
    om = -(sp.diff(psi, x, 2) + sp.diff(psi, y, 2))
    J = u * sp.diff(om, x) + v * sp.diff(om, y)
    g = make_grid(32)
    X, Y = mesh(g)
    om_grid = sp.lambdify((x, y), om, "numpy")(X, Y)
    J_grid = sp.lambdify((x, y), J, "numpy")(X, Y)
    wh = rfft2(jnp.asarray(om_grid))
    assert float(jnp.max(jnp.abs(wh - g.mask * wh))) < 1e-14  # every mode retained
    err = float(jnp.max(jnp.abs(irfft2(nonlinear(g, wh), g.N) - J_grid)))
    report("G4 N vs sympy", err, 1e-12)
    assert err <= 1e-12


def test_G5_dealiased_product_is_exact():
    g = make_grid(64)
    wh = random_field(g, seed=0)
    uh, vh = vel_hat(g, wh)
    exact = g.mask * (exact_product_hat(g, uh, 1j * g.kx * wh) + exact_product_hat(g, vh, 1j * g.ky * wh))
    err = float(jnp.max(jnp.abs(irfft2(nonlinear(g, wh) - exact, g.N))))
    report("G5 masked product vs 2N-grid product", err, 1e-12)
    assert err <= 1e-12


def test_G5c_control_aliasing_shows():
    g = make_grid(64)
    ah = random_field(g, seed=1, band_limited=False)
    bh = random_field(g, seed=2, band_limited=False)
    on_grid = rfft2(irfft2(ah, g.N) * irfft2(bh, g.N))
    err = float(jnp.max(jnp.abs(irfft2(on_grid - exact_product_hat(g, ah, bh), g.N))))
    report("G5c control (non-band-limited, unmasked)", err, 1e-3, ">")
    assert err > 1e-3


def _conservation(g, wh, nh):
    ph = psi_hat(g, wh)
    rms = lambda fh: float(norm(g, fh))
    cE = abs(float(inner(g, ph, nh))) / (rms(ph) * rms(nh))
    cZ = abs(float(inner(g, wh, nh))) / (rms(wh) * rms(nh))
    return cE, cZ


def test_G6_nonlinear_conserves_E_and_Z():
    g = make_grid(64)
    wh = random_field(g)
    cE, cZ = _conservation(g, wh, nonlinear(g, wh))
    report("G6 <psi N>", cE, 1e-12)
    report("G6 <omega N>", cZ, 1e-12)
    assert cE <= 1e-12 and cZ <= 1e-12


def test_G6c_control_unmasked_breaks_conservation():
    g = make_grid(64)
    wh = random_field(g, band_limited=False)
    cE, cZ = _conservation(g, wh, nonlinear_unmasked(g, wh))
    report("G6c control <psi N>", cE, 1e-8, ">")
    report("G6c control <omega N>", cZ, 1e-8, ">")
    assert cE > 1e-8 and cZ > 1e-8


def test_G7_taylor_green_pressure():
    g = make_grid(32)
    X, Y = mesh(g)
    wh = rfft2(jnp.asarray(2 * np.sin(X) * np.sin(Y)))  # psi = sin x sin y
    u, v = irfft2(vel_hat(g, wh)[0], g.N), irfft2(vel_hat(g, wh)[1], g.N)
    assert float(jnp.max(jnp.abs(u - np.sin(X) * np.cos(Y)))) < 1e-14
    assert float(jnp.max(jnp.abs(v + np.cos(X) * np.sin(Y)))) < 1e-14
    p = irfft2(pressure_hat(g, wh), g.N)
    err = float(jnp.max(jnp.abs(p - 0.25 * (np.cos(2 * X) + np.cos(2 * Y)))))
    report("G7 Taylor-Green pressure", err, 1e-12)
    assert err <= 1e-12


def test_G8_single_shell_free_decay():
    g = make_grid(32)
    X, Y = mesh(g)
    rng = np.random.default_rng(0)
    modes = [(3, 4), (4, 3), (5, 0), (0, 5), (3, -4), (4, -3)]
    w0 = sum(rng.standard_normal() * np.cos(kx * X + ky * Y + rng.uniform(0, 2 * np.pi)) for kx, ky in modes)
    w0 = w0 / np.sqrt(np.mean(w0**2))
    wh0 = rfft2(jnp.asarray(w0))
    prm = Params(nu=0.05, A=0.0)
    T, dt = 5.0, 0.01
    wh = make_advance(g, prm, dt, round(T / dt))(wh0)
    exact = wh0 * math.exp(-(25 * prm.nu + prm.alpha) * T)
    err = float(norm(g, wh - exact) / norm(g, exact))
    report("G8 single-shell decay", err, 1e-11)
    assert err <= 1e-11


def test_G9_laminar_spin_up_and_order():
    g = make_grid(32)
    prm = Params(nu=0.05)
    T = 10.0
    lam = prm.lam
    X, Y = mesh(g)
    wL = rfft2(jnp.asarray(-(prm.A * prm.kf / lam) * np.cos(prm.kf * Y)))
    exact = wL * (1 - math.exp(-lam * T))
    errs = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        wh = make_advance(g, prm, dt, round(T / dt))(jnp.zeros_like(wL))
        errs.append(float(norm(g, wh - exact) / norm(g, wL)))
    orders = [math.log2(errs[j] / errs[j + 1]) for j in range(3)]
    report("G9 laminar error at dt=0.1", errs[0], 1e-6)
    print("G9 errors", ["%.3e" % e for e in errs], "orders", ["%.4f" % o for o in orders],
          "(pass if each in [3.8, 4.2])")
    assert errs[0] <= 1e-6
    assert all(3.8 <= o <= 4.2 for o in orders)


def test_G10_resolution_change():
    g = make_grid(32)
    wh = random_field(g)
    err_rt = float(jnp.max(jnp.abs(irfft2(resample(resample(wh, 64), 32) - wh, 32))))
    g2 = make_grid(64)
    f = lambda X, Y: np.sin(X) * np.cos(2 * Y) + 0.7 * np.cos(3 * X) * np.sin(Y + 0.4) + 0.3 * np.sin(2 * X + Y)
    fh = rfft2(jnp.asarray(f(*mesh(g))))
    err_up = float(jnp.max(jnp.abs(irfft2(resample(fh, 64), 64) - f(*mesh(g2)))))
    report("G10 down(up(w)) round trip", err_rt, 1e-13)
    report("G10 up(analytic) vs analytic on 2N grid", err_up, 1e-13)
    assert err_rt <= 1e-13 and err_up <= 1e-13


def test_G11_masked_modes_stay_zero():
    g = make_grid(32)
    wh = make_advance(g, Params(nu=0.01), 0.01, 100)(random_field(g))
    outside = jnp.where(g.mask == 0, jnp.abs(wh), 0.0)
    val = float(jnp.max(outside))
    report("G11 max |coeff| outside mask", val, 0.0, "==")
    assert val == 0.0
