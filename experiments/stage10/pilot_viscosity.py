"""Stage 10 F2 pilot (physics only, no model): at which Reynolds numbers below the training band is Kolmogorov flow
still chaotic? For each Re, the k_f = 4 flow on a 128² grid is spun up for 50 time units from seed 0's random initial
condition. Two copies, one perturbed by 1e-6 of the vorticity's RMS, are then run for 100 time units. Recorded:
  growth    log10 of the twins' relative vorticity difference after 100 time units over its start
  t_1e-2    the time at which that difference first exceeds 1e-2 (None if never)
  E_cv      the energy's coefficient of variation over the 100 time units (0 for a steady flow)
  step_0.2  the relative change of the 64² vorticity over one frame interval (0.2), averaged
usage: pilot_viscosity.py [Re ...]   (writes results/f2_pilot.json)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kolmogorov import Params, choose_m, irfft2, make_advance, make_grid, random_ic, resample, rfft2  # noqa: E402
from kolmogorov.diagnostics import make_diagnostics  # noqa: E402
from kolmogorov.solver import SAVE_DT  # noqa: E402

N, KF = 128, 4
out = {}
for Re in [float(x) for x in sys.argv[1:]] or [10.0, 15.0, 20.0, 30.0, 100.0]:
    prm, g = Params(nu=1.0 / Re, kf=KF), make_grid(N)
    m = choose_m(N, prm)
    dt = SAVE_DT / m
    diag = jax.jit(make_diagnostics(g, prm)[0])
    adv = make_advance(g, prm, dt, round(0.2 / SAVE_DT) * m)  # 0.2 time units
    wh = make_advance(g, prm, dt, round(50.0 / SAVE_DT) * m)(random_ic(g, seed=0))
    w = irfft2(wh, N)
    rng = np.random.default_rng(1)
    wb = rfft2(w + 1e-6 * jnp.sqrt(jnp.mean(w**2)) * jnp.asarray(rng.standard_normal(w.shape)))
    sep, E, step = [], [], []
    a, b = wh, wb
    w64 = irfft2(resample(a, 64), 64)
    for i in range(500):  # 100 time units in steps of 0.2
        a, b = adv(a), adv(b)
        wa, wbb = irfft2(a, N), irfft2(b, N)
        sep.append(float(jnp.sqrt(jnp.sum((wa - wbb) ** 2) / jnp.sum(wa**2))))
        E.append(float(diag(a)[0]))
        n64 = irfft2(resample(a, 64), 64)
        step.append(float(jnp.sqrt(jnp.sum((n64 - w64) ** 2) / jnp.sum(w64**2))))
        w64 = n64
    sep, E = np.array(sep), np.array(E)
    hit = np.where(sep > 1e-2)[0]
    out[str(Re)] = dict(m=int(m), dt=dt, growth=float(np.log10(sep[-1] / sep[0])), sep_final=float(sep[-1]),
                        t_1e_2=float(0.2 * (hit[0] + 1)) if len(hit) else None, E_mean=float(E.mean()),
                        E_cv=float(E.std() / E.mean()), step_0_2=float(np.mean(step)))
    print(Re, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[str(Re)].items()}, flush=True)
with open(os.path.join(HERE, "results", "f2_pilot.json"), "w") as f:
    json.dump(out, f, indent=1)
