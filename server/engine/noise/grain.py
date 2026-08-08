from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .lattice import _lattice_np
from ..primitives import _smoothstep

# direction of their own once they are off the grid.
_GRAIN_ROT = math.atan(2.0 / (1.0 + 5.0**0.5))
_GRAIN_COS = math.cos(_GRAIN_ROT)
_GRAIN_SIN = math.sin(_GRAIN_ROT)

# Candidate points per lattice cell, and the fraction of those slots that
# actually hold a point.
#
# One point per cell -- what this used to be -- is a *stratified* process: the
# count in any region is fixed by its area, so density cannot vary and the
# layer reads as an even mesh however well the individual points are jittered.
# That evenness is the "repetitive when zooming out" complaint, and no amount
# of jitter fixes it, because jitter moves points without ever changing how
# many there are.
#
# Three slots at 0.62 gives a count per cell of Binomial(3, 0.62) -- mean 1.86,
# and genuinely 0, 1, 2 or 3 -- so clumps crowd in some places and leave gaps
# in others. Swept 2 to 5 slots at matched mean density and the rendered fields
# are hard to tell apart, so this is set at the cheapest count that still gives
# real variation: 3 slots over a 3x3 search is 27 candidate evaluations, which
# is what the old 5x5 single-slot search already cost.
_GRAIN_SLOTS = 3
_GRAIN_FILL = 0.62

# Rings of neighbouring cells the search checks, each way -- 1 means 3x3.
#
# **Exact, not a heuristic, and the proof is what buys back full-cell jitter.**
# Work in cell units. A pixel in cell (0,0) is somewhere in [0,1)^2. A point in
# an *excluded* cell -- one at least two cells away on either axis -- has that
# coordinate in [2, 3), so it is strictly more than 1 cell from the pixel. No
# point's radius can exceed one cell, because the lattice is pitched at ``hi``
# and radii are drawn from ``[lo, hi]``; and the falloff is exactly zero at and
# beyond the radius. So an excluded point contributes exactly nothing, whatever
# its jitter -- which is why the jitter may now cover the whole cell rather than
# the middle half the old 5x5 search needed.
#
# Verified rather than merely argued: rendering the same field at 1 ring and at
# 2 rings agrees to 2.7e-07 (float noise) across narrow and wide size ranges.
_GRAIN_RINGS = 1

# Exponent on a point's falloff when its brightness is mixed into a pixel.
#
# The old field read out the *winning* point's brightness -- a hard argmax --
# and that leaves a visible discontinuity wherever two overlapping discs of
# different brightness change places, which is a hard cusp cutting across an
# otherwise round grain. Weighting every candidate by ``falloff ** SHARE``
# instead makes the readout continuous while staying close to winner-take-all:
# at 3 a disc dominates its own middle completely and only trades with a
# neighbour where the two falloffs are genuinely comparable.
#
# **This is not the "sum the candidates" construction the old docstring warned
# about**, and the difference matters. Summing would let a distant point add
# light where there is none, which reads as fog; here the sum is *normalised*
# (it is a weighted mean of brightness, not a total) and the field's amplitude
# still comes from `peak`, the single largest falloff. So a gap is still exactly
# a gap: with every falloff at zero the amplitude is zero regardless of what any
# brightness nearby happens to be.
#
# It also removed the need for a tie-break margin. The old argmax could flip
# winners on a last-bit difference in the distance arithmetic between two tile
# layouts -- a discrete jump to another point's brightness, patched with a fixed
# margin. A weighted mean has no winner to flip: measured tile independence went
# from 2.6e-04 to 1.2e-06.
_GRAIN_SHARE = 3

# The cluster field: how deeply a grain's brightness is modulated by a smooth
# multi-octave field, its base pitch in *cells*, and that field's own octave
# count and roughness.
#
# **This is the answer to "no pattern when zooming out", and nothing else in
# the construction can supply it.** Points, however well randomised, give a
# process whose density is flat at large scales -- step back far enough and any
# such field averages to a featureless screen, which is exactly what reads as a
# repeating mesh. Real emulsion does not do that: crystals clump, clumps mottle,
# and the mottling has no single size. So each grain's brightness is scaled by
# ``1 + CLUSTER * (m * 2 - 1)`` with ``m`` a three-octave field, giving the
# layer real contrast variation at 6, 12 and 24 cells at once.
#
# Single-octave clustering was built first and is visibly wrong: it gives every
# clump-of-clumps the same diameter, so zoomed out the frame reads as regular
# blobs -- a different repeating pattern rather than none. Three octaves is
# where the eye stops finding a characteristic size.
#
# Depth 0.6 keeps ``1 + 0.6*(...)`` strictly positive, so the modulation can
# thin a region out but never invert a grain's sign. Swept 0.4 / 0.6 / 0.8:
# 0.4 is barely visible at a distance, 0.8 starts reading as patchiness in the
# image rather than as grain.
#
# Pitched in *cells* rather than pixels on purpose, so the mottling scales with
# the clump the way `reference_mp` scales everything else: a preset dialled in
# at one grain size keeps the same relationship between clump and cluster when
# the size slider moves.
_GRAIN_CLUSTER = 0.6
_GRAIN_CLUSTER_CELLS = 6.0
_GRAIN_CLUSTER_OCTAVES = 3
_GRAIN_CLUSTER_ROUGHNESS = 0.7

# Amplitude normalisation. The field is scaled so its standard deviation is
# `_GRAIN_TARGET_STD` at every Min/Max setting, using a closed form in the
# Min/Max *ratio* -- `_GRAIN_STD_FIT` is a cubic in that ratio, highest power
# first, giving the field's own std before the gain.
#
# **A closed form rather than a measurement, for invariant 1.** Dividing by
# ``field.std()`` would be a statistic of the region and would normalise every
# tile of an export differently while every preview looked fine. It does not
# have to be measured: the point pattern in cell units is scale-free, so the
# std depends on the ratio alone -- verified across an 8x range of absolute
# size (pitch 4, 8 and 16 working px agree to 1.4%), and the cubic fits the
# ratio sweep to 0.12%.
#
# Normalising at all is new, and it fixes a real inconsistency. The two old
# constructions disagreed about loudness by 43% -- the value-noise field
# measured 0.684 rendered sigma against the point field's 0.477 -- so
# `global_intensity` meant two different things depending on whether Max
# happened to exceed Min. The target is the *point* field's old level, which
# keeps the default preset's global layer where it was; the four presets that
# used the value-noise path get about 31% quieter. See CLAUDE.md.
#
# The cubic was **scaled, not re-fitted**, when the per-grain opacity draw was
# removed on 2026-08-05 (see `bri` in `_grain_points`). Full-density grains make
# the raw field louder by the ratio of the brightness term's own sigma --
# Rademacher +-1 against uniform[-1, 1), so sqrt(3) less what the weighted-mean
# readout averages away -- and that factor is *independent of the size ratio*,
# because the brightness draw shares no hash channel with the geometry. Measured
# across the whole sweep it is 1.711x with a spread of 0.2% (1.708-1.712), so
# every coefficient carries the same multiplier and the ratio dependence -- the
# only thing this cubic exists to describe -- is provably untouched. Scaling
# rather than re-fitting is what makes that a statement about the shape rather
# than a new set of numbers that happen to fit; it also holds rendered loudness
# *exactly* where it was, since the gain divides out the factor it multiplied in.
_GRAIN_TARGET_STD = 0.29
_GRAIN_STD_FIT = (-0.04938, -0.02158, 0.16008, 0.48341)


def _grain_gain(lo: float, hi: float) -> float:
    """Amplitude normaliser for `_grain_points`, a closed form in ``lo/hi``.

    See `_GRAIN_TARGET_STD` for why this is a fitted constant rather than a
    measurement of the field in hand.
    """
    r = min(max(lo / hi, 0.0), 1.0)
    a, b, c, d = _GRAIN_STD_FIT
    return _GRAIN_TARGET_STD / (((a * r + b) * r + c) * r + d)


def _grain_lattice_noise(
    iy0: int, ix0: int, hl: int, wl: int, pitch: float, seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Quintic value noise over *lattice cell indices*, pitch in cells.

    The cluster field below is a property of a grain, not of a pixel -- every
    point in a cell shares one value -- so it is evaluated on the lattice and
    never on the pixel grid. That is why the most expensive-sounding part of
    this rewrite costs almost nothing: the lattice is smaller than the frame by
    the square of the clump size, and at the coarse pitches the cluster field
    uses it is smaller again.

    Addressed by absolute cell index, so two tiles asking about the same cell
    agree -- the same discipline `_value_noise` follows one level down.
    """
    def span(i0: int, n: int) -> tuple[int, int]:
        j0 = int(math.floor(i0 / pitch)) - 1
        j1 = int(math.floor((i0 + n - 1) / pitch)) + 2
        return j0, j1 - j0 + 1

    j0, jn = span(iy0, hl)
    k0, kn = span(ix0, wl)
    lat = torch.from_numpy(_lattice_np(j0, k0, jn, kn, seed, 1)).to(device)

    v = (torch.arange(iy0, iy0 + hl, device=device, dtype=torch.float32)
         / pitch)[:, None]
    u = (torch.arange(ix0, ix0 + wl, device=device, dtype=torch.float32)
         / pitch)[None, :]

    def remap(t: torch.Tensor) -> torch.Tensor:
        fl = torch.floor(t)
        f = t - fl
        return fl + f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

    gy = ((remap(v) - j0) / max(jn - 1, 1) * 2.0 - 1.0).expand(hl, wl)
    gx = ((remap(u) - k0) / max(kn - 1, 1) * 2.0 - 1.0).expand(hl, wl)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return F.grid_sample(
        lat.unsqueeze(0), grid, mode="bilinear", align_corners=True,
        padding_mode="border",
    )[0, 0]


def _grain_cluster(
    iy0: int, ix0: int, hl: int, wl: int, seed: int, device: torch.device,
) -> torch.Tensor:
    """Per-cell brightness multiplier: the multi-octave clumping field.

    Variance-preserving across octaves for `_fbm`'s reason -- otherwise adding
    structure would quietly turn the modulation down, and the cluster depth
    would stop meaning one thing.
    """
    total: torch.Tensor | None = None
    wsum = 0.0
    wsq = 0.0
    for o in range(_GRAIN_CLUSTER_OCTAVES):
        wgt = _GRAIN_CLUSTER_ROUGHNESS**o if o else 1.0
        n = _grain_lattice_noise(
            iy0, ix0, hl, wl, _GRAIN_CLUSTER_CELLS * (2.0**o),
            seed + o * 1301, device,
        )
        total = n * wgt if total is None else total + n * wgt
        wsum += wgt
        wsq += wgt * wgt
    m = 0.5 + (total / wsum - 0.5) * (wsum / math.sqrt(wsq))
    return 1.0 + _GRAIN_CLUSTER * (m * 2.0 - 1.0)


def _grain_points(
    h: int, w: int, y0: float, x0: float, lo: float, hi: float, seed: int,
    device: torch.device, nfields: int = 1,
) -> torch.Tensor:
    """The Global Grain layer's noise: discrete grains of independently drawn
    size, scattered on a rotated lattice and clustered at every scale.

    This is the *only* construction the global layer uses now, at every Min/Max
    setting including Min == Max. It replaced two: a value-noise fBm below Max
    and a cellular field above it. Both were reported as showing a grid, and the
    measurements agreed -- see the comment block above `_GRAIN_ROT` and the
    section in CLAUDE.md.

    **Why the fBm had to go rather than be repaired.** Value noise interpolates
    between lattice points with a curve whose derivative vanishes *at* those
    points, so every cell reads as a blob with flat corners and the blobs tile a
    visible quilt: measured gridiness 1.47 at a 12px clump, with the field's own
    autocorrelation peaking at exactly the lattice pitch (0.24 at lag 5 for a
    5px cell). That is intrinsic to the interpolant, not to the lattice's
    orientation -- rendering it through a rotated lattice was tried and simply
    produces a rotated quilt. The only repair is a different kind of field.

    Grains rather than noise is also the better model for what this layer *is*.
    It stands in for print stock and scanner grain, and a grain is a particle
    with a position, a size and a density -- which is what a point field says
    and what an interpolated lattice cannot.

    The construction, one point-slot at a time:

    * The lattice is pitched at ``hi``, the largest radius any grain can have,
      which is what makes the 3x3 neighbour search exact (see `_GRAIN_RINGS`).
      It is **rotated** against the pixel grid (see `_GRAIN_ROT`).
    * Each cell carries `_GRAIN_SLOTS` slots. A slot draws one uniform value
      that decides both whether it holds a grain at all (`_GRAIN_FILL`) and, if
      it does, that grain's radius in ``[lo, hi]`` -- one hash channel doing two
      jobs, which keeps the per-cell hash cost down where the lattice is dense.
      Conditional on being present the radius is still uniform on ``[lo, hi]``,
      so the size distribution is exactly what the two sliders promise.
    * Position jitters over the **whole** cell. Brightness is a *sign* drawn per
      output field -- every grain that exists is at full density, never a random
      fraction of it -- scaled by that cell's cluster multiplier.
    * A pixel's amplitude is the single largest falloff over every candidate
      (so a gap stays a gap), and its brightness is those candidates' brightness
      averaged under ``falloff ** _GRAIN_SHARE`` (so overlapping grains trade
      smoothly instead of cutting a cusp across each other).

    **Centred on 0.5**, the convention `_fbm` returned and `_smooth_noise`
    requires -- that function re-centres explicitly, so a field meaning
    something else at 0.5 would be blurred about the wrong point. A gap has
    every falloff at zero and therefore lands exactly on 0.5 whatever the
    brightness of anything nearby, which is what "nothing is here" has to mean;
    brightness is ``+-1`` with equal odds so a grain is lighter or darker, as
    real grain is. Note what "lighter or darker" does *not* mean here: it is a
    coin flip on direction, not a draw on strength. Density varies across the
    frame -- it has to, or the layer is a screen -- but it varies through how
    many grains land where, how they overlap and the cluster field, never by
    handing an individual grain a fractional opacity.

    Returns ``[1, nfields, h, w]``. Geometry -- which cells hold grains, where,
    and how big -- is shared across fields and only brightness is drawn per
    field, which is what lets the chroma variant give one grain its own
    intensity per channel without moving its edge from channel to channel.

    Needs **no tile overlap at all**: every quantity is a function of absolute
    global coordinates, and the lattice window is derived per call from the
    window it was asked for, so a pixel always sees its true neighbours however
    the frame was split. Measured at 1.2e-06 between a whole-frame render and
    arbitrary sub-windows with zero padding.
    """
    cell = hi
    ca, sa = _GRAIN_COS, _GRAIN_SIN

    Y = (torch.arange(h, device=device, dtype=torch.float32) + float(y0))[:, None]
    X = (torch.arange(w, device=device, dtype=torch.float32) + float(x0))[None, :]
    # Rotated, and in cell units. Rotation is an isometry, so distances -- and
    # therefore radii and falloffs -- mean exactly what they did; only the cell
    # grid's orientation against the pixel grid changes.
    Yr = (Y * ca + X * sa) / cell
    Xr = (X * ca - Y * sa) / cell

    # The rotated window's bounds. Both coordinates are affine in (y, x), so
    # their extrema over the tile are attained at its four corners -- no device
    # reduction and no scalar read-back, the same reasoning as `_lat_span`. The
    # pad is `_GRAIN_RINGS + 1`: one cell for the ring the search reaches into,
    # and one spare so the float64 bound here can never fall on the wrong side
    # of an integer from the float32 ramp above. Over-covering is free -- the
    # sampling grid is an affine map of the absolute lattice index, so unused
    # rows simply go unread.
    ys = (float(y0), float(y0) + h - 1)
    xs = (float(x0), float(x0) + w - 1)
    vs = [(yy * ca + xx * sa) / cell for yy in ys for xx in xs]
    us = [(xx * ca - yy * sa) / cell for yy in ys for xx in xs]
    pad = _GRAIN_RINGS + 1
    iy0 = int(math.floor(min(vs))) - pad
    hl = int(math.floor(max(vs))) + pad + 1 - iy0
    ix0 = int(math.floor(min(us))) - pad
    wl = int(math.floor(max(us))) + pad + 1 - ix0

    # Per slot: jitter y, jitter x, the combined presence/radius draw, then one
    # brightness per output field. One CPU hash call covers every slot.
    per = 3 + nfields
    lat = torch.from_numpy(
        _lattice_np(iy0, ix0, hl, wl, seed, _GRAIN_SLOTS * per)
    ).to(device)

    cell_iy = torch.arange(iy0, iy0 + hl, device=device, dtype=torch.float32)[:, None]
    cell_ix = torch.arange(ix0, ix0 + wl, device=device, dtype=torch.float32)[None, :]
    camp = _grain_cluster(iy0, ix0, hl, wl, seed + 991, device)

    piy = (torch.floor(Yr).long() - iy0).clamp(0, hl - 1)
    pix = (torch.floor(Xr).long() - ix0).clamp(0, wl - 1)

    peak = torch.zeros(h, w, device=device)
    num = torch.zeros(nfields, h, w, device=device)
    den = torch.zeros(h, w, device=device)
    rad_lo, rad_span = lo / cell, (hi - lo) / cell
    for s in range(_GRAIN_SLOTS):
        b = s * per
        # One draw, two jobs: below `_GRAIN_FILL` the slot holds a grain and the
        # draw is stretched back over the full [lo, hi] range for its radius;
        # above it the slot is empty, which a zero radius says exactly (the
        # falloff is zero everywhere, so it can never win and never contributes).
        u = lat[b + 2]
        rad = torch.where(
            u < _GRAIN_FILL, rad_lo + rad_span * (u / _GRAIN_FILL),
            torch.zeros_like(u),
        )
        # **Sign only -- a grain that exists is at full density.** This draw used
        # to be `u * 2 - 1`, uniform on [-1, 1), which gave every grain its own
        # random *opacity* as well as its own direction: half of all grains came
        # out at under half strength and a grain near u = 0.5 contributed
        # essentially nothing while still occupying its cell slot. That is the
        # wrong model twice over. A developed silver halide crystal is opaque and
        # an undeveloped one is clear -- there is no half-developed crystal -- so
        # density variation in real emulsion comes from how many grains land in a
        # region and how they overlap, which `_GRAIN_FILL`, the radius draw and
        # `_GRAIN_CLUSTER` already supply. And it read as veiling: a population of
        # weak grains is a low-amplitude haze spread over the whole frame, which
        # the amplitude normaliser then has to *amplify* to hit its target sigma,
        # so the few full-strength grains got pushed into the clamp to pay for the
        # faint ones.
        #
        # Signed rather than all-positive because the field must stay mean-zero
        # about 0.5 (see the docstring) and because real grain is lighter *or*
        # darker with equal odds. `torch.where` against 0.5 rather than
        # `torch.sign(u - 0.5)`: lattice values are exact multiples of 2**-24, so
        # 0.5 is attainable, and `sign` would hand that slot a zero-density grain
        # -- reintroducing the thing this removes, rarely enough to never be seen
        # and often enough to exist.
        su = lat[b + 3: b + 3 + nfields]
        bri = torch.where(su < 0.5, -torch.ones_like(su),
                          torch.ones_like(su)) * camp
        py = cell_iy + lat[b]
        px = cell_ix + lat[b + 1]
        for dy in range(-_GRAIN_RINGS, _GRAIN_RINGS + 1):
            for dx in range(-_GRAIN_RINGS, _GRAIN_RINGS + 1):
                ny = (piy + dy).clamp(0, hl - 1)
                nx = (pix + dx).clamp(0, wl - 1)
                dyp = Yr - py[ny, nx]
                dxp = Xr - px[ny, nx]
                # The epsilon on the *distance* is what makes an empty slot
                # unreachable rather than merely improbable. An empty slot has
                # radius exactly 0, and a bare `radius.clamp_min(tiny)` would
                # give a pixel landing exactly on that slot's phantom position
                # a ratio of 0/tiny = 0 -- a full-strength grain out of a slot
                # that holds none. Biasing the numerator instead forces the
                # ratio to `1e-7 / 1e-12` there, comfortably past 1, whatever
                # the distance happens to be. It costs a real grain 1e-7 of a
                # cell, which is under 1e-7 of a pixel.
                shape = 1.0 - _smoothstep(
                    0.0, 1.0,
                    (torch.sqrt(dyp * dyp + dxp * dxp) + 1e-7)
                    / rad[ny, nx].clamp_min(1e-12),
                )
                wgt = shape
                for _ in range(_GRAIN_SHARE - 1):
                    wgt = wgt * shape
                num = num + wgt * bri[:, ny, nx]
                den = den + wgt
                peak = torch.maximum(peak, shape)

    # In a true gap every `shape` is exactly zero -- `_smoothstep` clamps, so
    # the falloff is 0 at and beyond the radius rather than merely small -- so
    # `num` and `den` are both exactly zero and this is 0, not an amplified
    # ratio of two small numbers. `peak` is zero there too, so it would not
    # matter either way.
    val = num / den.clamp_min(1e-12)
    return (
        0.5 + (0.5 * _grain_gain(lo, hi)) * peak.unsqueeze(0) * val
    ).unsqueeze(0)


# Scatter stencils, indexed by the ``scatter_pattern`` parameter. A stencil is
# the *set of places a displaced pixel may land*, and it takes three
# independent things to describe one -- which is why this is a table and not
# just a direction count:
#
#   name    matches the parameter's ``choices`` tuple in params.py, entry for
#           entry. The two are one list in two places.
#   first   angle of the first direction, in degrees.
#   count   how many directions, evenly spaced from ``first``. 0 is the
#           continuous case: any angle at all.
#   locus   how travel varies with angle -- "circle" is the same distance
#           every way; "diamond" is |dx|+|dy| = reach, so the shape reaches
#           furthest along the axes and pulls in on the diagonals.
#   inner   the hole, as a fraction of the reach. 0 fills the shape solid;
#           a donut keeps every pixel out past this however Reach Spread is
#           set, so nothing lands near where it started.
#   alt     length multiplier on every other direction, which is what makes a
#           star a star: long spokes on the axes, short ones between them. 1
#           is uniform.
#
# **Every stencil must keep peak travel at or under the reach**, because that
# is the figure `pad_for` reserves overlap for. "circle" and "diamond" both do
# (diamond is shorter off-axis, never longer); an L-infinity "square" locus
# would reach 1.41x on the diagonals and would have to be paid for there.
#
# The value stored in a preset file is the *index*, so renumbering these
# silently changes the look of every preset that used one. Append, do not
# insert. (Diamond, Donut and Star were inserted mid-list on 2026-08-01, while
# the feature was still unreleased and no preset had ever stored a pattern.)
_SCATTER_STENCILS: tuple[tuple[str, float, int, str, float, float], ...] = (
    # name          first  count  locus      inner  alt
    ("Any",           0.0,     0, "circle",   0.0,  1.00),
    ("Cross",         0.0,     4, "circle",   0.0,  1.00),
    ("Diagonal",     45.0,     4, "circle",   0.0,  1.00),
    ("Box",           0.0,     8, "circle",   0.0,  1.00),
    ("Diamond",       0.0,     0, "diamond",  0.0,  1.00),
    ("Donut",         0.0,     0, "circle",  0.65,  1.00),
    ("Star",          0.0,     8, "circle",   0.0,  0.40),
    ("Horizontal",    0.0,     2, "circle",   0.0,  1.00),
    ("Vertical",     90.0,     2, "circle",   0.0,  1.00),
)

# How far out a donut's hole pushes the *shortest* journey, and how far a
# star's short spokes fall behind its long ones. Both live in the table above;
# these names exist so the numbers there read as something.
_SCATTER_NAMES: tuple[str, ...] = tuple(s[0] for s in _SCATTER_STENCILS)
