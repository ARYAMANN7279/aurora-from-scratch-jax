"""Reference outputs from Aurora's Swin3DTransformerBackbone (handouts/stage06_backbone.md, section 2).

Run in the separate reference environment (see experiments/stage5/reference_env.txt):
    refenv/bin/python experiments/stage6/make_reference.py
Writes tests/fixtures/backbone_ref.npz.
"""
import copy
import os

import numpy as np
import torch
from aurora.model.fourier import lead_time_expansion
from aurora.model.swin3d import Swin3DTransformerBackbone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "tests", "fixtures", "backbone_ref.npz")
torch.manual_seed(0)
CFGS = {  # D, enc depths, enc heads, dec depths, dec heads, window, patch_res, B, lead times (h)
    "c1": (16, (2, 2, 2), (1, 2, 4), (2, 2, 2), (4, 2, 1), (2, 4, 4), (2, 16, 16), 2, (6.0, 12.0)),
    "c2": (24, (1, 2), (2, 4), (2, 1), (4, 2), (1, 4, 8), (2, 8, 16), 1, (24.0,)),
}
out = {}
for name, (D, ed, eh, dd, dh, ws, res, B, lt) in CFGS.items():
    bb = Swin3DTransformerBackbone(embed_dim=D, encoder_depths=ed, encoder_num_heads=eh,
                                   decoder_depths=dd, decoder_num_heads=dh, window_size=ws).double()
    with torch.no_grad():
        for p in bb.parameters():
            p.copy_(0.1 * torch.randn_like(p))
    bb.eval()
    sd = {k: v.detach().clone().numpy() for k, v in bb.state_dict().items()}
    C, H, W = res
    x = torch.randn(B, C * H * W, D, dtype=torch.float64)
    lead = torch.tensor(lt, dtype=torch.float64)
    with torch.no_grad():
        y64 = bb(x, lead, rollout_step=0, patch_res=res)
        y32 = copy.deepcopy(bb).float()(x.float(), lead.float(), rollout_step=0, patch_res=res)
        t_emb = lead_time_expansion(lead, D).double()
        # patch merging / splitting of the first stage, on their own
        xm = bb.encoder_layers[0].downsample(x, res)
        xs_in = torch.randn(B, C * (H // 2) * (W // 2), 2 * D, dtype=torch.float64)
        xs = bb.decoder_layers[len(dd) - 2].upsample(xs_in, (C, H // 2, W // 2))
    for k, v in sd.items():
        out[f"{name}/w/{k}"] = v
    out.update({f"{name}/x": x.numpy(), f"{name}/t_emb": t_emb.numpy(), f"{name}/y64": y64.numpy(),
                f"{name}/y32": y32.double().numpy(), f"{name}/merge_out": xm.numpy(),
                f"{name}/split_in": xs_in.numpy(), f"{name}/split_out": xs.numpy(),
                f"{name}/meta": np.array([D, B, *ws, *res, len(ed)]),
                f"{name}/depths": np.array([*ed, *dd]), f"{name}/heads": np.array([*eh, *dh])})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **out)
print("wrote", OUT, "keys", len(out))
print("\n".join(k for k in out if k.startswith("c1/w/") and ".blocks." not in k))
