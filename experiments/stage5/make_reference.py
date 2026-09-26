"""Reference outputs from Aurora's Swin3D block (handouts/stage05_swin.md, section 3).

Run in the separate reference environment, never in the project venv:
    python3.11 -m venv refenv && refenv/bin/pip install microsoft-aurora==2.0.1
    refenv/bin/python experiments/stage5/make_reference.py
Writes tests/fixtures/swin3d_ref.npz.
"""
import copy
import os

import numpy as np
import torch
from aurora.model.swin3d import (Swin3DTransformerBlock, compute_3d_shifted_window_mask,
                                 window_partition_3d)
from aurora.model.util import maybe_adjust_windows

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "tests", "fixtures", "swin3d_ref.npz")
torch.manual_seed(0)
CASES = [  # name, B, (C, H, W), D, heads, window, shift
    ("a_noshift", 2, (2, 8, 8), 32, 4, (2, 4, 4), (0, 0, 0)),
    ("a_shift", 2, (2, 8, 8), 32, 4, (2, 4, 4), (1, 2, 2)),
    ("b_noshift", 1, (4, 8, 16), 48, 3, (2, 4, 8), (0, 0, 0)),
    ("b_shift", 1, (4, 8, 16), 48, 3, (2, 4, 8), (1, 2, 4)),
]
out = {}
for name, B, res, D, heads, ws, ss in CASES:
    blk = Swin3DTransformerBlock(dim=D, num_heads=heads, time_dim=D, window_size=ws, shift_size=ss).double()
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(0.1 * torch.randn_like(p))
    C, H, W = res
    x = torch.randn(B, C * H * W, D, dtype=torch.float64)
    c = torch.randn(B, D, dtype=torch.float64)
    blk.eval()
    # save the float64 weights before any cast: the float32 output below uses a float32 copy
    sd = {k: v.detach().clone().numpy() for k, v in blk.state_dict().items()}
    with torch.no_grad():
        y64 = blk(x, c, res, rollout_step=0)
        y32 = copy.deepcopy(blk).float()(x.float(), c.float(), res, rollout_step=0)
    for k, v in sd.items():
        out[f"{name}/w/{k}"] = v
    out[f"{name}/x"] = x.numpy()
    out[f"{name}/c"] = c.numpy()
    out[f"{name}/y64"] = y64.numpy()
    out[f"{name}/y32"] = y32.double().numpy()
    out[f"{name}/meta"] = np.array([B, C, H, W, D, heads, *ws, *ss])
    out[f"{name}/windows"] = window_partition_3d(x.view(B, C, H, W, D), ws).numpy()
    if any(ss):
        m, _ = compute_3d_shifted_window_mask(C, H, W, ws, ss, torch.device("cpu"), torch.float64, warped=True)
        out[f"{name}/mask"] = m.numpy()
adj = []
for ws, ss, res in [((2, 4, 4), (1, 2, 2), (2, 8, 8)), ((2, 4, 4), (1, 2, 2), (1, 8, 8)),
                    ((3, 4, 4), (1, 2, 2), (3, 4, 8)), ((2, 8, 8), (1, 4, 4), (4, 4, 16)),
                    ((1, 4, 4), (0, 2, 2), (3, 16, 16)), ((2, 6, 12), (1, 3, 6), (4, 6, 6))]:
    w2, s2 = maybe_adjust_windows(ws, ss, res)
    adj.append([*ws, *ss, *res, *w2, *s2])
out["adjust"] = np.array(adj)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **out)
print("wrote", OUT, "torch", torch.__version__, "keys", len(out))
