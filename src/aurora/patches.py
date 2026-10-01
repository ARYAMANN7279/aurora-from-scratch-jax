"""Patchify / unpatchify (handouts/stage03_embed.md, section 2.2)."""


def patchify(x, P):
    """(..., N, N) -> (..., T, P*P), patches row-major over [py, px], pixels row-major inside."""
    *b, H, W = x.shape
    x = x.reshape(*b, H // P, P, W // P, P)
    x = x.swapaxes(-3, -2)
    return x.reshape(*b, (H // P) * (W // P), P * P)


def unpatchify(t, P, N, W=None):
    """(..., T, P*P) -> (..., N, W); exact inverse of patchify. W defaults to N (square)."""
    W = N if W is None else W
    *b, _, _ = t.shape
    x = t.reshape(*b, N // P, W // P, P, P)
    x = x.swapaxes(-3, -2)
    return x.reshape(*b, N, W)
