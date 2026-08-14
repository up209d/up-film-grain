"""Deliberately slow reference implementations of three noise routines and
the exact gaussian blur.

A faster rewrite of a noise generator is only correct if it changes
nothing, and "the render still looks like grain" cannot tell you that --
so `verify.py` asserts bit-equality against these rather than measuring a
property. `grain_ref` carries a second job: it searches a wider 5x5
neighbourhood than the engine's 3x3, so it is also the proof behind
`_GRAIN_RINGS` written out as a measurement.

`blur_ref` is here for a related but distinct reason: `_blur` is *not*
bit-exact any more above `_BLUR_EXACT_MAX_SIGMA`, so the check needs
something that still is, on both sides of that threshold.

Deleting these turns those checks into tautologies. See CLAUDE.md.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from server.engine import (
    _GRAIN_CLUSTER_REF, _GRAIN_COS, _GRAIN_FILL, _GRAIN_SHARE, _GRAIN_SIN,
    _GRAIN_SLOTS, _grain_cluster, _grain_gain, _lattice_np, _smoothstep,
)


def lattice_ref(iy0, ix0, hl, wl, seed, nfields):
    yy = np.arange(iy0, iy0 + hl, dtype=np.int64).view(np.uint64)[:, None]
    xx = np.arange(ix0, ix0 + wl, dtype=np.int64).view(np.uint64)[None, :]
    out = np.empty((nfields, hl, wl), dtype=np.float32)
    for f in range(nfields):
        s = np.uint64(((seed + f * 7919) * 0x165667B19E3779F9) % (1 << 64))
        n = (xx * np.uint64(0x9E3779B97F4A7C15)
             + yy * np.uint64(0xC2B2AE3D27D4EB4F))
        n = n + s
        n = n ^ (n >> np.uint64(29))
        n = n * np.uint64(0xBF58476D1CE4E5B9)
        n = n ^ (n >> np.uint64(32))
        n = n * np.uint64(0x94D049BB133111EB)
        n = n ^ (n >> np.uint64(31))
        out[f] = (n >> np.uint64(40)).astype(np.float32) / float(1 << 24)
    return out


def span_ref(n, origin, cell, pad_lo, pad_hi, dev):
    t = (torch.arange(n, device=dev, dtype=torch.float32)
         + float(origin)) / cell
    i0 = int(math.floor(float(t[0]))) - pad_lo
    return i0, int(math.floor(float(t[-1]))) + pad_hi - i0 + 1


def grain_ref(h, w, y0, x0, lo, hi, seed, device, nfields=1,
              cluster=_GRAIN_CLUSTER_REF):
    rings = 2
    cell = hi
    ca, sa = _GRAIN_COS, _GRAIN_SIN
    Y = (torch.arange(h, device=device, dtype=torch.float32)
         + float(y0))[:, None]
    X = (torch.arange(w, device=device, dtype=torch.float32)
         + float(x0))[None, :]
    Yr = (Y * ca + X * sa) / cell
    Xr = (X * ca - Y * sa) / cell
    ys = (float(y0), float(y0) + h - 1)
    xs = (float(x0), float(x0) + w - 1)
    vs = [(yy * ca + xx * sa) / cell for yy in ys for xx in xs]
    us = [(xx * ca - yy * sa) / cell for yy in ys for xx in xs]
    pad = rings + 1
    iy0 = int(math.floor(min(vs))) - pad
    hl = int(math.floor(max(vs))) + pad + 1 - iy0
    ix0 = int(math.floor(min(us))) - pad
    wl = int(math.floor(max(us))) + pad + 1 - ix0
    per = 3 + nfields
    lat = torch.from_numpy(
        _lattice_np(iy0, ix0, hl, wl, seed, _GRAIN_SLOTS * per)).to(device)
    ciy = torch.arange(iy0, iy0 + hl, device=device,
                       dtype=torch.float32)[:, None]
    cix = torch.arange(ix0, ix0 + wl, device=device,
                       dtype=torch.float32)[None, :]
    # Deliberately *not* mirroring the engine's skip-at-zero shortcut: the
    # reference is here to say what the field is, and computing the field at
    # depth 0 and multiplying is the honest statement of that. If the two ever
    # disagree the shortcut is wrong, which is exactly what this should catch.
    camp = _grain_cluster(iy0, ix0, hl, wl, seed + 991, device, cluster)
    piy = (torch.floor(Yr).long() - iy0).clamp(0, hl - 1)
    pix = (torch.floor(Xr).long() - ix0).clamp(0, wl - 1)
    peak = torch.zeros(h, w, device=device)
    num = torch.zeros(nfields, h, w, device=device)
    den = torch.zeros(h, w, device=device)
    for s in range(_GRAIN_SLOTS):
        b = s * per
        u = lat[b + 2]
        rad = torch.where(
            u < _GRAIN_FILL,
            (lo + (hi - lo) * (u / _GRAIN_FILL)) / cell,
            torch.zeros_like(u),
        )
        su = lat[b + 3: b + 3 + nfields]
        bri = torch.where(su < 0.5, -torch.ones_like(su),
                          torch.ones_like(su)) * camp
        py, px = ciy + lat[b], cix + lat[b + 1]
        for dy in range(-rings, rings + 1):
            for dx in range(-rings, rings + 1):
                ny = (piy + dy).clamp(0, hl - 1)
                nx = (pix + dx).clamp(0, wl - 1)
                dyp, dxp = Yr - py[ny, nx], Xr - px[ny, nx]
                sh = 1.0 - _smoothstep(
                    0.0, 1.0,
                    (torch.sqrt(dyp * dyp + dxp * dxp) + 1e-7)
                    / rad[ny, nx].clamp_min(1e-12))
                wgt = sh ** _GRAIN_SHARE
                num = num + wgt * bri[:, ny, nx]
                den = den + wgt
                peak = torch.maximum(peak, sh)
    val = num / den.clamp_min(1e-12)
    return (0.5 + (0.5 * _grain_gain(lo, hi, cluster))
            * peak.unsqueeze(0) * val).unsqueeze(0)


def blur_ref(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """The separable gaussian, always exact -- `_blur` before it grew a
    decimated path for large sigma.

    Kept verbatim rather than derived, because it is the thing the threshold is
    measured against. Below `_BLUR_EXACT_MAX_SIGMA` the engine must equal this
    bit for bit; above it, this is what the tolerance is a tolerance *of*.
    """
    if sigma < 0.05:
        return x
    r = max(1, int(math.ceil(sigma * 3.0)))
    r = min(r, min(x.shape[-1], x.shape[-2]) - 1)
    if r < 1:
        return x
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return x
