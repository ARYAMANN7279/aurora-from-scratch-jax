"""PyTorch port of src/aurora for Stage 12 (handouts/stage12_speed.md): the same model, operation by
operation, so that its speed can be compared with the JAX code.

Parameters come from the JAX pytree, flattened to path keys such as "bb/enc/0/blocks/1/qkv/W"
(`bench_jax.py --export`); weights keep the JAX layout (x @ W + b). Batched over the first axis,
which the JAX model gets from jax.vmap. `sdpa=True` routes every attention through
torch.nn.functional.scaled_dot_product_attention (arm T3).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARS = ("u", "v", "p", "omega")
AREA_RANGE, LEVEL_RANGE, TIME_RANGE = (0.01, 100.0), (0.5, 64.0), (0.05, 100.0)


def n_levels(N):
    return int(round(math.log2(N))) - 1


def band_masks(N):
    """As src/aurora/levels.py: (C, N, N//2+1) square-shell octave bands, Nyquist removed."""
    ky = np.abs(np.fft.fftfreq(N, 1.0 / N))
    kx = np.fft.rfftfreq(N, 1.0 / N)
    n = np.maximum(ky[:, None], kx[None, :]).astype(np.int64)
    idx = np.where(n < 2, 0, np.floor(np.log2(np.maximum(n, 1))).astype(np.int64))
    idx = np.where(n <= N // 2 - 1, idx, -1)
    return np.stack([idx == j for j in range(n_levels(N))]).astype(np.float32)


def fourier(s, D, lmin, lmax):
    lam = np.geomspace(lmin, lmax, D // 2).astype(np.float32)
    ang = (2 * math.pi * np.asarray(s, np.float32)[..., None] / lam).astype(np.float32)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


def position(N, P, D):
    c = (2 * math.pi * (np.arange(N // P) + 0.5) * P / N).astype(np.float32)
    yc, xc = np.meshgrid(c, c, indexing="ij")
    cen = np.stack([xc.ravel(), yc.ravel()], axis=-1)
    m = np.arange(1, D // 4 + 1, dtype=np.float32)
    x, y = cen[:, :1] * m, cen[:, 1:] * m
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=-1).astype(np.float32)


def layernorm(x, g=None, b=None, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    y = (x - mu) / torch.sqrt(var + eps)
    return y if g is None else y * g + b


def patchify(x, P):
    *b, H, W = x.shape
    x = x.reshape(*b, H // P, P, W // P, P).transpose(-3, -2)
    return x.reshape(*b, (H // P) * (W // P), P * P)


def unpatchify(t, P, N):
    *b, _, _ = t.shape
    return t.reshape(*b, N // P, N // P, P, P).transpose(-3, -2).reshape(*b, N, N)


def adjust(window, shift, res):
    w, s = list(window), list(shift)
    for i, r in enumerate(res):
        if r <= window[i]:
            w[i], s[i] = r, 0
    return tuple(w), tuple(s)


def window_partition(x, ws):
    B, C, H, W, D = x.shape
    x = x.reshape(B, C // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], D)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws[0], ws[1], ws[2], D)


def window_reverse(win, ws, B, C, H, W):
    D = win.shape[-1]
    x = win.reshape(B, C // ws[0], H // ws[1], W // ws[2], ws[0], ws[1], ws[2], D)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, C, H, W, D)


def shift_mask(res, ws, ss, periodic=(False, True, True)):
    """As src/aurora/swin.py; None when no window needs a mask (every case in the E1 configuration)."""
    if all(s == 0 or p for s, p in zip(ss, periodic)):
        return None
    g = []
    for n, w, s, p in zip(res, ws, ss, periodic):
        g.append(np.zeros(n, np.int64) if (p or s == 0) else
                 np.concatenate([np.zeros(n - w, np.int64), np.ones(w - s, np.int64), np.full(s, 2, np.int64)]))
    ids = (g[0][:, None, None] * 9 + g[1][None, :, None] * 3 + g[2][None, None, :]).astype(np.float64)
    win = window_partition(torch.tensor(ids)[None, ..., None], ws).reshape(-1, ws[0] * ws[1] * ws[2])
    return win[:, :, None] == win[:, None, :]


class AuroraTorch(nn.Module):
    def __init__(self, flat, D=64, P=4, L=3, heads=4, N=64, enc_depths=(2, 2, 2), enc_heads=(2, 4, 8),
                 dec_depths=(2, 2, 2), dec_heads=(8, 4, 2), window=(3, 4, 4), dt=0.2, sdpa=False):
        super().__init__()
        self.p = nn.ParameterDict({k.replace("/", "__"): nn.Parameter(torch.tensor(np.asarray(v, np.float32)))
                                   for k, v in flat.items()})
        self.D, self.P, self.L, self.H, self.N, self.sdpa = D, P, L, heads, N, sdpa
        self.enc_depths, self.enc_heads, self.dec_depths, self.dec_heads = enc_depths, enc_heads, dec_depths, dec_heads
        self.window = window
        self.C = n_levels(N)
        T = (N // P) ** 2
        self.register_buffer("bands", torch.tensor(band_masks(N)))
        self.register_buffer("pos_in", torch.tensor(position(N, P, D)))
        self.register_buffer("area_in", torch.tensor(fourier(np.log((2 * math.pi * P / N) ** 2), D, *AREA_RANGE)))
        self.register_buffer("level_in", torch.tensor(fourier(np.arange(self.C, dtype=np.float32), D, *LEVEL_RANGE)))
        self.register_buffer("time_in", torch.tensor(fourier(np.float32(dt), D, *TIME_RANGE)))
        res = [(L, N // P, N // P)]
        for _ in range(len(enc_depths) - 1):
            res.append((L, res[-1][1] // 2, res[-1][2] // 2))
        self.res = res
        self.T = T

    def w(self, path):
        return self.p[path.replace("/", "__")]

    def lin(self, path, x, bias=True):
        y = x @ self.w(path + "/W")
        return y + self.w(path + "/b") if bias else y

    # ---- attention
    def attend(self, q, k, v):
        """q (..., H, Q, dh), k and v (..., H, K, dh) -> (..., H, Q, dh)."""
        if self.sdpa:
            return F.scaled_dot_product_attention(q, k, v)
        logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        return torch.softmax(logits, dim=-1) @ v

    def mha(self, pre, q_in, kv_in, H):
        *b, Q, D = q_in.shape
        K, dh = kv_in.shape[-2], D // H
        q = (q_in @ self.w(pre + "/Wq")).reshape(*b, Q, H, dh).transpose(-3, -2)
        k = (kv_in @ self.w(pre + "/Wk")).reshape(*b, K, H, dh).transpose(-3, -2)
        v = (kv_in @ self.w(pre + "/Wv")).reshape(*b, K, H, dh).transpose(-3, -2)
        o = self.attend(q, k, v).transpose(-3, -2).reshape(*b, Q, D)
        return o @ self.w(pre + "/Wo") + self.w(pre + "/bo")

    def mlp(self, pre, x):
        h = F.gelu(x @ self.w(pre + "/W1") + self.w(pre + "/b1"), approximate="none")
        return h @ self.w(pre + "/W2") + self.w(pre + "/b2")

    def perceiver_block(self, pre, q_in, kv_in):
        h = layernorm(q_in + self.mha(pre + "/attn", q_in, kv_in, self.H), self.w(pre + "/ln1/g"), self.w(pre + "/ln1/b"))
        return layernorm(h + self.mlp(pre + "/mlp", h), self.w(pre + "/ln2/g"), self.w(pre + "/ln2/b"))

    # ---- encoder
    def embed(self, x):
        """x (B, V, N, N) -> tokens (B, C, T, D)."""
        xh = torch.fft.rfft2(x, norm="forward")
        xl = torch.fft.irfft2(xh[:, :, None] * self.bands, s=(self.N, self.N), norm="forward")  # (B, V, C, N, N)
        tok = 0.0
        for i, v in enumerate(VARS):  # (tok + X @ W) + b, in the order of src/aurora/embed.py
            tok = tok + patchify(xl[:, i], self.P) @ self.w(f"emb/embed/{v}/W") + self.w(f"emb/embed/{v}/b")
        pos = self.lin("emb/enc/pos", self.pos_in)
        ar = self.lin("emb/enc/area", self.area_in)
        lev = self.lin("emb/enc/level", self.level_in)
        return tok + pos[None, None] + ar + lev[None, :, None]

    def aggregate(self, tok):
        B, C, T, D = tok.shape
        kv = tok.permute(0, 2, 1, 3).reshape(B * T, C, D)
        z = self.w("per/latents").expand(B * T, self.L, D)
        z = self.perceiver_block("per/agg/0", z, kv)
        return z.reshape(B, T, self.L, D).permute(0, 2, 1, 3)

    def decode(self, lat):
        B, L, T, D = lat.shape
        q0 = self.lin("per/dec_q", self.level_in)
        kv = lat.permute(0, 2, 1, 3).reshape(B * T, L, D)
        x = self.perceiver_block("per/dec/0", q0.expand(B * T, self.C, D), kv)
        return x.reshape(B, T, self.C, D).permute(0, 2, 1, 3)

    # ---- backbone
    def adaln(self, pre, y, c):
        shift, scale = (F.silu(c) @ self.w(pre + "/W") + self.w(pre + "/b")).chunk(2, dim=-1)
        return layernorm(y) * scale[:, None] + shift[:, None]

    def swin_block(self, pre, x, c, res, heads, shift):
        C, H, W = res
        B, Ltok, D = x.shape
        ws, ss = adjust(self.window, shift, res)
        xv = x.reshape(B, C, H, W, D)
        shifted = any(ss)
        if shifted:
            xv = torch.roll(xv, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
        mask = shift_mask(res, ws, ss) if shifted else None
        Nw = ws[0] * ws[1] * ws[2]
        xw = window_partition(xv, ws).reshape(-1, Nw, D)
        Bw, dh = xw.shape[0], D // heads
        qkv = (xw @ self.w(pre + "/qkv/W") + self.w(pre + "/qkv/b")).reshape(Bw, Nw, 3, heads, dh).permute(2, 0, 3, 1, 4)
        if mask is None:
            o = self.attend(qkv[0], qkv[1], qkv[2])
        else:
            m = mask.to(x.device).repeat(Bw // mask.shape[0], 1, 1)[:, None]
            logits = (qkv[0] @ qkv[1].transpose(-2, -1) / math.sqrt(dh)).masked_fill(~m, float("-inf"))
            o = torch.softmax(logits, dim=-1) @ qkv[2]
        aw = o.transpose(1, 2).reshape(Bw, Nw, D) @ self.w(pre + "/proj/W") + self.w(pre + "/proj/b")
        xv = window_reverse(aw.reshape(-1, *ws, D), ws, B, C, H, W)
        if shifted:
            xv = torch.roll(xv, shifts=ss, dims=(1, 2, 3))
        x = x + self.adaln(pre + "/norm1", xv.reshape(B, Ltok, D), c)
        return x + self.adaln(pre + "/norm2", self.mlp(pre + "/mlp", x), c)

    def run_blocks(self, pre, depth, x, c, res, heads):
        half = tuple(w // 2 for w in self.window)
        for j in range(depth):
            x = self.swin_block(f"{pre}/blocks/{j}", x, c, res, heads, half if j % 2 == 1 else (0, 0, 0))
        return x

    def merge(self, pre, x, res):
        C, H, W = res
        B, _, D = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2, D).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C * (H // 2) * (W // 2), 4 * D)
        return layernorm(x, self.w(pre + "/norm/g"), self.w(pre + "/norm/b")) @ self.w(pre + "/W")

    def split(self, pre, x, res):
        C, H, W = res
        B = x.shape[0]
        out = self.w(pre + "/W2").shape[0]
        x = (x @ self.w(pre + "/W1")).reshape(B, C, H, W, 2, 2, out).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C * 4 * H * W, out)
        return layernorm(x, self.w(pre + "/norm/g"), self.w(pre + "/norm/b")) @ self.w(pre + "/W2")

    def backbone(self, x):
        S = len(self.enc_depths)
        c = F.silu(self.time_in @ self.w("bb/time/W1") + self.w("bb/time/b1")) @ self.w("bb/time/W2") + self.w("bb/time/b2")
        c = c.expand(x.shape[0], -1)
        skips = []
        for i in range(S):
            x = self.run_blocks(f"bb/enc/{i}", self.enc_depths[i], x, c, self.res[i], self.enc_heads[i])
            skips.append(x)
            if i < S - 1:
                x = self.merge(f"bb/enc/{i}/merge", x, self.res[i])
        for i in range(S):
            idx = S - i - 1
            x = self.run_blocks(f"bb/dec/{i}", self.dec_depths[i], x, c, self.res[idx], self.dec_heads[i])
            if i < S - 1:
                x = self.split(f"bb/dec/{i}/split", x, self.res[idx])
            if 0 < i < S - 1:
                x = x + skips[idx - 1]
            elif i == S - 1:
                x = torch.cat([x, skips[0]], dim=-1)
        return x

    # ---- full model
    def forward(self, x_prev, x_cur):
        """x_prev, x_cur (B, 4, N, N) in VARS order -> (B, 4, N, N)."""
        B = x_cur.shape[0]
        tok = self.embed(x_prev) + self.w("frame")[0] + self.embed(x_cur) + self.w("frame")[1]
        z = self.aggregate(tok)                                                     # (B, L, T, D)
        y = self.backbone(z.reshape(B, self.L * self.T, self.D))                    # (B, L*T, 2D)
        y = y.reshape(B, self.L, self.T, 2 * self.D) @ self.w("mid/W") + self.w("mid/b")
        tokens = self.decode(y)                                                     # (B, C, T, D)
        delta = torch.stack([unpatchify(self.lin(f"emb/head/{v}", tokens), self.P, self.N) for v in VARS], dim=1).sum(2)
        return x_cur + delta
