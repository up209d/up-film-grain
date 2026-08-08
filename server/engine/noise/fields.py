from __future__ import annotations

import math

import torch

from ..constants.core import _MIN_CELL, _SMOOTH_GAIN_K_FIT, _SMOOTH_MAX
from .grain import _SCATTER_STENCILS
from .lattice import _value_noise
from ..primitives import _blur

def _scatter_offsets(
    sel: torch.Tensor, mag_n: torch.Tensor, reach: float, spread: float,
    pattern: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel travel in whole working pixels, on the chosen stencil.

    ``sel`` and ``mag_n`` must be uniform on 0..1, which is why they come from
    ``_cell_noise`` and not from ``_value_noise``: a quantised direction reads
    the distribution directly, and value noise does not have one worth reading.

    Rounded to whole pixels here and nowhere else. That is what keeps the
    gather a copy rather than an interpolation, so it has to happen after every
    shaping term rather than to the reach up front.
    """
    _, first, count, locus, inner, alt = _SCATTER_STENCILS[pattern]

    if count == 0:
        th = sel * (2.0 * math.pi)
        spoke: torch.Tensor | float = 1.0
    else:
        bin_ = torch.floor(sel * count).clamp(0.0, count - 1.0)
        th = math.radians(first) + bin_ * (2.0 * math.pi / count)
        # Alternate directions run short. With eight directions starting on
        # the axes that puts the long spokes N/E/S/W and the short ones on the
        # diagonals, which is the shape a cross filter actually flares into.
        spoke = 1.0 if alt == 1.0 else torch.where(
            (bin_ % 2.0) < 0.5, 1.0, alt
        )

    c, s = torch.cos(th), torch.sin(th)
    # Angular shaping. A diamond's vertices sit on the axes, so travel is the
    # full reach there and 1/sqrt(2) of it on the diagonals -- the locus is
    # |dx| + |dy| = reach.
    shape: torch.Tensor | float = (
        1.0 / (c.abs() + s.abs()).clamp_min(1e-4) if locus == "diamond" else 1.0
    )
    # Reach Spread fills the shape inward from its edge; `inner` holds a hole
    # open in the middle of it whatever Spread says. At inner = 0 this is
    # exactly reach * (1 - spread * u).
    radial = inner + (1.0 - inner) * (1.0 - spread * mag_n)
    r = reach * radial * shape * spoke
    return torch.round(r * c), torch.round(r * s)


def _fbm(
    h: int, w: int, y0: float, x0: float, cell: float, seed: int, nfields: int,
    octaves: int, roughness: float, device: torch.device,
) -> torch.Tensor:
    """Stacked value-noise octaves, returned in 0..1.

    The cascade runs **coarser**, not finer: ``cell`` is the finest structure
    and each octave doubles it. Conventional fBm subdivides downward, and that
    is what this used to do -- but the base cell here is already at the pixel
    grid, so there was nowhere finer to go. Every octave was immediately
    clamped to ``_MIN_CELL`` and differed from the previous one only by seed,
    which is why the Octaves and Roughness sliders measured 0.02% and 0.18%
    mean change on a real proxy: visually nothing.

    Running coarse fixes that at both ends. Larger cells are always
    representable, so the controls do something at every zoom and working
    scale and can never alias. And it is the right model: emulsion clumps into
    clusters, and clusters into mottling, so stacking *larger* structure over
    the base clump is what film actually looks like -- particularly pushed
    film, where the clumping is the look.

    ``octaves = 1`` is exactly the base cell alone, i.e. the old behaviour with
    the cascade switched off.
    """
    total = None
    wsum = 0.0
    wsq = 0.0
    # Floor the base, not each octave: below _MIN_CELL the lattice is denser
    # than the pixel grid and would be pure aliasing. Everything above the
    # base is coarser, so nothing after this can breach it.
    base = max(cell, _MIN_CELL)
    for o in range(int(octaves)):
        c = base * (2.0**o)
        wgt = roughness**o if o else 1.0
        n = _value_noise(h, w, y0, x0, c, seed + o * 1301, nfields, device)
        total = n * wgt if total is None else total + n * wgt
        wsum += wgt
        wsq += wgt * wgt

    # Dividing by wsum alone holds the *mean* at 0.5 but lets the variance
    # collapse: the octaves are decorrelated, so summing them and dividing by
    # the weight sum leaves variance scaled by sum(w^2)/sum(w)^2 -- 0.56 at
    # three octaves. Every octave added structure and quietly turned the grain
    # down by the same stroke, which is most of why the slider read as doing
    # nothing. Rescaling the deviation by sum(w)/sqrt(sum(w^2)) preserves
    # variance instead, so octaves change *structure* at constant strength and
    # Intensity remains the only control over amplitude.
    field = total / wsum
    gain = wsum / math.sqrt(wsq)
    return 0.5 + (field - 0.5) * gain if gain != 1.0 else field


def _smooth_noise(
    n: torch.Tensor, cell: float, amount: float, ratio: float = 1.0,
) -> torch.Tensor:
    """Blur a 0..1 noise field and put back the amplitude the blur costs.

    ``ratio`` is the field's Min/Max grain-size ratio, which the gain depends
    on -- see `_SMOOTH_GAIN_K_FIT`. It defaults to 1.0, a single grain size.

    It was written to remove a defect: the layer was value noise, a quilt of
    axis-aligned cells, and a filter on the field was the only cure. There is no
    quilt left to remove -- `_grain_points` has no lattice-aligned structure to
    begin with -- so this is now a shape control, rounding grains off and
    softening where they meet.

    **Variance is restored, and that is the point.** `_fbm` preserves variance
    so Octaves changes structure at constant strength; this follows the same
    rule for the same reason. A smoothing control that quietly turned the layer
    down would leave Global Intensity fighting it, and "smoother" would be
    indistinguishable from "less".

    The gain is a closed form in ``sigma/cell`` and the size ratio rather than a
    measurement of the field in hand. Normalising against ``n.std()`` would be a
    statistic of the region -- invariant 1 -- and would restore a different
    amount in every tile of an export while every preview looked fine.
    """
    if amount <= 0.001:
        return n
    sigma = amount * _SMOOTH_MAX * cell
    if sigma < 0.05:
        return n
    a, b, c = _SMOOTH_GAIN_K_FIT
    r = min(max(ratio, 0.0), 1.0)
    gain = math.sqrt(1.0 + ((a * r + b) * r + c) * (sigma / cell) ** 2)
    return 0.5 + _blur(n - 0.5, sigma) * gain
