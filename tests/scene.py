"""Synthetic test scene.

Deliberately contains one of each thing the pipeline must handle correctly:
a smooth gradient sky (flat area -- grain must stay out), a dense highlight
(grain must drop off, halation must bloom), deep shadow with hard vertical
edges, a grid of high-contrast micro-edges (the edge-destruction target), and
a full-range tonal ramp for measuring the luminance response.
"""

from __future__ import annotations

import numpy as np


def scene(h: int = 900, w: int = 1400) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    img = np.zeros((h, w, 3), np.float32)

    # smooth gradient sky -- flat-area rule
    sky = 0.92 - 0.35 * v
    img[..., 0] = sky * 0.86
    img[..., 1] = sky * 0.93
    img[..., 2] = sky * 1.00

    # midtone ground, where grain should peak
    g = v > 0.62
    img[g] = np.stack([0.34 + 0.10 * u, 0.30 + 0.08 * u, 0.24 + 0.06 * u], -1)[g]

    # dense highlight -- grain drops off, halation blooms
    d = np.sqrt((xx - w * 0.76) ** 2 + (yy - h * 0.20) ** 2)
    img[d < 70] = 0.99

    # deep shadow block with hard vertical edges
    b = (xx > w * 0.10) & (xx < w * 0.34) & (yy > h * 0.30)
    img[b] = np.stack([0.07 + 0.02 * u, 0.07 + 0.02 * u, 0.09 + 0.02 * u], -1)[b]

    # high-contrast micro-edge grid
    for r in range(6):
        for c in range(4):
            y0 = int(h * 0.34 + r * h * 0.09)
            x0 = int(w * 0.13 + c * w * 0.055)
            img[y0:y0 + int(h * 0.045), x0:x0 + int(w * 0.032)] = 0.58

    # Smooth skin-tone patch. Midtone, so the luminance response wants grain
    # here -- only the texture measure can tell it should stay clean.
    (sy, sx, ph, pw) = patch(h, w, "smooth")
    ramp = np.linspace(0, 1, pw, dtype=np.float32)[None, :, None]
    skin = np.array([0.72, 0.56, 0.47], np.float32)[None, None, :]
    img[sy:sy + ph, sx:sx + pw] = skin * (0.94 + 0.12 * ramp)

    # Fine-textured patch at the same mean luminance. Grain belongs here.
    (ty, tx, _, _) = patch(h, w, "textured")
    rng = np.random.default_rng(7)
    t = rng.random((ph, pw, 1)).astype(np.float32)
    t = 0.25 * (t[:-2, :-2] + t[2:, :-2] + t[:-2, 2:] + t[2:, 2:])  # decorrelate a little
    t = np.pad(t, ((1, 1), (1, 1), (0, 0)), mode="edge")
    img[ty:ty + ph, tx:tx + pw] = np.clip(skin * 1.0 + (t - 0.5) * 0.22, 0, 1)

    # full tonal ramp for the luminance-response measurement
    img[int(h * 0.90):] = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]

    return np.ascontiguousarray(np.clip(img, 0, 1))


def patch(h: int, w: int, which: str) -> tuple[int, int, int, int]:
    """(y, x, h, w) of the named test patch, so checks can find them."""
    ph, pw = int(h * 0.16), int(w * 0.14)
    y = int(h * 0.68)
    x = int(w * 0.42) if which == "smooth" else int(w * 0.62)
    return y, x, ph, pw
