"""Approach A -- edge-destruction procedural grain pipeline.

Design notes that matter for correctness:

* **Scale invariance.** Every spatial quantity (clump size, high-pass radius,
  micro-blur) is specified in *full-resolution* pixels and multiplied by the
  working ``scale`` at render time. The noise lattice is indexed by working
  coordinates divided by the scaled cell size, which equals the full-res
  coordinate divided by the full-res cell size. A supersampled pass and a
  plain one therefore show the same grain structure, not the same pixel noise.

* **Tile independence.** Nothing in the pipeline depends on a statistic of the
  region being rendered -- no per-tile normalisation, no global mean. Edge
  strength is normalised against the fixed ``EDGE_REF`` constant and the noise
  lattice is addressed by absolute global coordinates. Two adjacent tiles
  sampling the same global position get bit-identical values, so tiles composite
  without seams given enough overlap to cover the blur kernels.

* **Not every softening stage is a filter.** ``scatter`` displaces a share of
  the pixels onto their neighbours and averages nothing at all, so it takes the
  image's exactness without taking its micro-contrast. It samples nearest-
  neighbour on whole-pixel offsets precisely so each output pixel stays a copy
  of a real one; measured against a blur of the same reach it keeps 100% of
  fine-texture sigma where the blur keeps 14%. Anything that turns it into an
  average -- bilinear resampling, cross-fading the moved pixel with the
  original -- destroys the only reason it exists.

* **Grain is structural.** Alongside the weighted additive term, the grain field
  multiplies the image's own micro-detail (``edge_erosion``). That term is zero
  in flat areas and grows on edges, so grain erodes existing edge structure
  rather than being stamped over it.

* **One deliberate exception.** The final stage, ``global_*``, is a flat grain
  overlay applied after everything else and weighted by no mask at all. It is
  not emulsion behaviour and is not meant to be -- it stands in for grain that
  arrives with the print stock or the scan, and it is the only way to put grain
  into the smooth regions the masks above exist to protect. It ships at zero.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import params as P

# Luma coefficients (Rec. 709).
_LUMA = (0.2126, 0.7152, 0.0722)

# Fixed reference for normalising high-pass edge magnitude into 0..1. Must be a
# constant rather than a per-image statistic, or tiles would normalise
# differently and seam.
EDGE_REF = 0.06

# Normalising divisor applied to the raw noise field before the clump curve.
# Measured field std is ~0.27, so 0.55 puts roughly 2 sigma at full scale and
# clips only ~3.6% of samples -- tight enough for the clump curve to bite,
# loose enough to leave the distribution's tails intact. Constant, not a
# per-image statistic, so tiles stay seamless.
_GNORM = 0.55

# Converts the 0..100 intensity slider into image-referred amplitude. Chosen so
# the default intensity of 32 lands near 3.5% luminance sigma in the midtones,
# which is about right for a 400-speed stock viewed at 100%.
#
# Was 0.5. Recalibrated to 0.38 when _fbm started preserving variance across
# octaves: the old normaliser let the field's variance collapse as octaves were
# added, so the default 3-octave field was running at 43% strength and 0.5 was
# compensating for it. Measured back to 99.7% of the previous look on the
# textured patch, with grain and erosion separated (they share the residual).
_AMP_SCALE = 0.38

# Grain finer than this many working pixels cannot be represented, so the
# lattice is clamped. Below Nyquist it would simply alias.
_MIN_CELL = 0.8

# Local mean-absolute-deviation thresholds, in luma units, separating "smooth"
# from "textured" over a medium radius. Skin and clear sky sit near or below
# _TEX_LO; fabric, foliage and hair sit above _TEX_HI. Fixed constants, not
# per-image statistics, so tiles stay independent.
_TEX_LO = 0.002
_TEX_HI = 0.015

# Luma-step thresholds separating a real transition from fine texture, for the
# edge-softening mask. Calibrated by measurement: fine texture measures a step
# an order of magnitude under a hard border, so the gap between these is what
# lets softening take the snap off a border while leaving fabric and hair
# intact. Fixed constants, not per-image statistics, so tiles stay independent.
_STEP_LO = 0.030
_STEP_HI = 0.110

# Hue the blue compensation centres on, in degrees, measured *in linear light*
# because that is where the stage runs. Skies land at 222 (pale) to 236
# (zenith) there, so 230 sits in the middle of them; cyan water is 194 and
# purple shadow 249, comfortably outside a narrow Blue Range. Note these are
# not the sRGB numbers -- the transfer curve is per-channel and monotonic, so
# it preserves the hue *sector* but moves the angle inside it by 6-10 degrees.
_BLUE_HUE = 230.0

# Half-width of the hue window, in degrees. Fixed rather than exposed: the
# discriminator that actually matters is *brightness*, not hue width -- the
# wash only reaches what is near the light, so a deep blue is untouched
# whatever its hue. This was a slider and it was the wrong control.
_BLUE_RANGE = 70.0

# Saturation below which a pixel counts as grey and the compensation leaves it
# alone. Without it the mask would strengthen colour in something that has
# none, which is the failure `vibrance` is written to avoid as well.
_BLUE_SAT_FLOOR = 0.12

# Peak edge displacement in full-resolution pixels at edge_jitter = 1.
#
# Was an inline 0.6, which made the control useless: the noise field averages
# well under its own peak, so the typical displacement measured 0.227px, and
# that is *before* the edge mask scales it down again. A quarter-pixel wobble
# survives neither a proxy render nor the browser downscale on top of it, and
# the slider read as doing nothing. At 3.0 the low fifth of the slider still
# covers the old sub-pixel range and the top of it actually ripples an edge.
_JITTER_MAX = 3.0

# Tap offsets and weights for the tangential sanding filter, in units of the
# sanding radius, which is the gaussian sigma. Offsets run to +/-2 sigma rather
# than +/-1: a contour's roughness sits at longer wavelengths than it looks
# like it should -- measured on a jittered border, only 8% of the contour's
# energy is below 8px, and 92% is above -- so a filter that reaches only one
# sigma barely touches it. Weights are gaussian, normalised.
_SAND_TAPS = (
    (-2.0, 0.054), (-1.0, 0.242), (0.0, 0.399), (1.0, 0.242), (2.0, 0.054),
)

# Maximum sanding passes. Short passes that re-aim follow a curving edge where
# one wide pass cuts across it; three is where the returns flatten. pad_for
# assumes this count exactly, so the two must not drift apart.
_SAND_PASSES = 3

# Direction-estimate blur, as a fraction of the sanding radius. Must scale with
# the radius rather than being fixed: see the seam note in render(). pad_for
# depends on this value.
_SAND_DIR_K = 0.6

# Gradient magnitude below which the sanding tangent is treated as undefined
# and the effect faded out. Well under a real edge's gradient, so it only
# catches genuinely flat ground -- where there is nothing to sand anyway.
_SAND_MIN_GRAD = 0.012

# Thresholds that turn a value-noise field into sparse marks for the film
# texture, one band per mark type. These have to be read off the field's actual
# distribution rather than guessed: value noise is heavily centre-weighted, and
# a threshold of 0.88 -- which sounds extreme -- selects 10% of the frame. The
# measured quantiles are 1% above 0.943, 0.1% above 0.988, 0.01% above 0.998.
#
# Target coverage at full strength, chosen to look like film rather than like
# weather: dust ~0.3% of the frame, scratches ~0.15%, hair ~0.03%. Anything
# near a percent stops reading as damage and starts reading as texture.
_DUST_THRESH = (0.9800, 0.9955)
_SCRATCH_THRESH = (0.9836, 0.9970)
# A hair is drawn as the level set |n - 0.5| < eps, so its width in pixels is
# roughly 2 * eps * cell -- eps is scale-invariant on its own, but it has to be
# solved for a real width, not picked. At 0.0016 with a 55px cell the filament
# came out 0.35px wide, i.e. sub-pixel before supersampling even halved it, and
# rendered as literally nothing. 0.014 gives about 1.5px.
_HAIR_EPS = 0.006

# Floor on a mark's per-mark brightness multiplier. Marks vary in density from
# this to full; taking it to zero would just delete marks rather than vary them,
# which thins the population instead of making it look weathered.
_TEX_LUM_FLOOR = 0.25

# Dust softening radius as a multiple of the speck's own cell. Was 1.6, which
# at a 2px speck is a 3px blur -- not enough to read as out of focus at all.
_DUST_SOFT_REACH = 1.1

# How far Dust Softness widens the speck's threshold band, which is what makes
# the speck itself gradual rather than a disc. This does the real work; the
# blur only takes the last of the edge off.
_DUST_SOFT_BAND = 12.0

# Inverse CDF of the value-noise field: (fraction of pixels above, threshold).
# Measured over 3.2M samples. Needed because the film-texture marks are counted
# rather than dialled by amount -- to put N marks on a frame you have to know
# what threshold selects N cells' worth of field, and value noise is far too
# centre-weighted to guess at (a threshold of 0.88 selects 4% of the frame, not
# 12%). Interpolated in log(fraction), which is close to linear here.
_NOISE_ICDF = (
    (0.30, 0.6342), (0.20, 0.7147), (0.12, 0.7900), (0.07, 0.8444),
    (0.04, 0.8829), (0.02, 0.9164), (0.01, 0.9422), (0.005, 0.9616),
    (0.002, 0.9792), (0.001, 0.9878), (5e-4, 0.9933), (2e-4, 0.9970),
    (1e-4, 0.9985), (5e-5, 0.9993), (2e-5, 0.9996),
    (1e-5, 0.999822), (5e-6, 0.999913), (2e-6, 0.999960),
    (1e-6, 0.999977), (5e-7, 0.999986),
)

# A threshold picked as N/cells delivers many times more than N marks: the
# field's peaks are broad and clustered, so one excursion above the threshold
# becomes several detectable blobs. Purely a calibration constant -- the
# geometric argument predicts about 1.3 and measurement says otherwise, so
# measurement wins. Tuned against delivered counts on a 1.5MP frame.
# One per mark type, because the calibration is not shared: a compact speck, a
# 70:1 scratch and a level-set filament each turn a given coverage fraction
# into a different number of countable marks. Measured against delivered counts
# on a 1.5MP frame and accurate to roughly a factor of 1.5 across the range --
# these are counts you steer by, not guarantees.
_BLOB_CELLS_DUST = 14.0
_BLOB_CELLS_SCRATCH = 26.0
_BLOB_CELLS_HAIR = 0.5

# -- light leaks ---------------------------------------------------------- #
# A leak is a *shaft* of light past an obstruction, so it is drawn as a small
# number of discrete oriented beams anchored on the perimeter, not as a wash
# gated along the whole border. See `_leak_sites` for why a list of them does
# not break tile independence the way a list of dust specks would.

# Step used to place leaks around the perimeter, as a fraction of it. Golden
# ratio, i.e. a low-discrepancy sequence rather than a stratification, so leak
# k lands in the same place whatever the count is -- raising the count must add
# a leak, not reshuffle the ones already on the frame.
_LEAK_PHI = 0.6180339887498949

# How hard leaks are pulled toward the ends of their border. The film gate's
# corners and the cassette mouth are where light actually gets past, and an
# even spread along the perimeter is the single most "generated"-looking thing
# a leak field can do. Applied inside one border segment, so it biases a leak
# toward a corner without ever moving it onto a different edge.
#
# Must stay under 1 / 2pi = 0.159, or `t - bias * sin(2 pi t)` stops being
# monotonic and starts *folding*: at 0.24 its slope goes to -0.51 near the
# ends, which maps a quarter of the way along a border to one hundredth of the
# way along it. Every leak then piles into a corner, which is not a bias, it is
# a collapse -- and it looks exactly like the four-corner symmetry this stage
# was rewritten to get away from.
_LEAK_CORNER_BIAS = 0.10

# Peak of the domain warp that breaks a leak's outline up, as a fraction of its
# reach. The shape has a definite edge by construction -- that is the point,
# real leaks have one -- and this is what stops that edge being a drawn curve.
_LEAK_WARP = 0.15

# Divisor on the reach cap. The cap exists so a leak cannot fog the centre, and
# the warp above can carry the falloff `_LEAK_WARP * reach` further in than the
# reach alone, so the cap has to be paid for twice over. 1.25 against a warp of
# 0.15 leaves real margin rather than landing exactly on zero -- a falloff
# exponent below 1 turns a float epsilon into a visible lift.
_LEAK_REACH_SAFETY = 1.25

# Exposure one unit of leak deposits, before `leak_strength`. Calibrated so the
# default strength lands a hot leak's core just into saturation.
_LEAK_GAIN = 2.0


def _threshold_for(fraction: float) -> float:
    """Field threshold that leaves roughly ``fraction`` of pixels above it."""
    f = min(max(fraction, 5e-7), 0.30)
    pts = _NOISE_ICDF
    for (f0, q0), (f1, q1) in zip(pts, pts[1:]):
        if f1 <= f <= f0:
            # log-linear between the bracketing samples
            w = (math.log(f) - math.log(f0)) / (math.log(f1) - math.log(f0))
            return q0 + w * (q1 - q0)
    return pts[0][1] if f >= pts[0][0] else pts[-1][1]


def _count_threshold(
    count: float, cell_area: float, frame_area: float, blob_cells: float,
) -> float:
    """Threshold that puts roughly ``count`` marks on a frame.

    A mark occupies about one lattice cell, so the frame holds
    ``frame_area / cell_area`` of them and the fraction wanted is the ratio.
    Approximate by construction -- blobs merge and clip at the frame edge --
    but it makes the control a *count* rather than an opaque 0-1, and a
    requested 20 lands within a few of 20 rather than an order of magnitude
    away.
    """
    cells = max(frame_area / max(cell_area, 1e-6), 1.0)
    return _threshold_for(max(count, 0.0) / (cells * blob_cells))


def _leak_sites(count: float, seed: int, var: float) -> list[dict]:
    """Per-leak parameters, in units that do not depend on the frame's size.

    This *is* a list of objects, which the rest of the film-texture section
    refuses to use -- and it is still tile-independent, because the list is a
    function of the count, the seed and nothing else. Every tile builds the
    identical list; so does the proxy, so does the export. What breaks tile
    independence is deriving a list from the *region being rendered* (N specks
    per tile, or positions drawn against the tile's own area), and nothing
    here reads either.

    Objects rather than thresholded noise because a leak is not a mark, it is a
    beam: it has a source, a direction and a length, and a field that only
    knows "how far am I from the nearest border" can express none of those.

    ``var`` is `leak_variation`: every draw except the reach is a blend from
    the middle of its range toward the drawn value, so 0 makes every leak
    identical in everything but where it sits and how far it comes in.
    """
    n = int(min(max(round(count), 0), 64))

    def mix(u: float, lo: float, hi: float) -> float:
        return 0.5 * (lo + hi) + var * (u - 0.5) * (hi - lo)

    sites = []
    for k in range(n):
        # Seeded per leak, not once per frame, for the same reason the
        # positions come off a low-discrepancy sequence: leak 3 must not
        # change when leak 9 is added.
        rng = np.random.default_rng(
            np.uint64((int(seed) & 0xFFFF) * 1000003 + k * 7919 + 17)
        )
        u = rng.random(10)
        sites.append({
            # Where on the perimeter, 0..1. The jitter is small on purpose --
            # the golden step already spreads them, and a large jitter just
            # lets two leaks land on top of each other.
            "pos": (0.37 + _LEAK_PHI * k + 0.10 * (u[0] - 0.5)) % 1.0,
            # Reach is the one draw `var` does not touch: the two size sliders
            # state its spread outright, and the help text promises variation
            # changes everything *except* size.
            "reach_t": u[1],
            # Half-length along the border, as a fraction of that border.
            # A fraction of the *border* rather than a multiple of the reach:
            # a failed seal runs along a seam, so a leak is long sideways and
            # shallow inward, and sizing it off its own depth makes blobs.
            "width": mix(u[2], 0.03, 0.30),
            # Lateral drift per unit depth. This is what makes a leak a streak
            # leaning across the frame instead of a symmetric wedge, and it is
            # kept as a shear rather than as a rotation so that "reach" stays
            # exactly the perpendicular depth the slider claims.
            "shear": mix(u[3], -1.7, 1.7),
            # How much the beam fans out as it travels in.
            "flare": mix(u[4], -0.55, 0.75),
            # Asymmetry of the two long edges. A leak is light spilling past an
            # obstruction, so one side is the obstruction's shadow and is much
            # harder than the other; a shape soft on both sides reads as haze.
            "hard": mix(u[5], 0.25, 1.0),
            "hard_side": 1.0 if u[6] >= 0.5 else -1.0,
            "strength": mix(u[7], 0.45, 1.35),
            # Halo: pushes this leak's half-strength distance around.
            "halo": u[8],
            # Hue jitter, added to `leak_hue`.
            "hue": mix(u[9], -0.20, 0.20),
        })
    return sites


def _leak_anchor(pos: float, fh: float, fw: float) -> tuple[int, float]:
    """Map a perimeter position to (border, along-border coordinate in px).

    Borders are 0 top, 1 bottom, 2 left, 3 right; the coordinate is x on the
    horizontal borders and y on the vertical ones.
    """
    a = (pos % 1.0) * 2.0 * (fh + fw)
    for border, length in ((0, fw), (3, fh), (1, fw), (2, fh)):
        if a < length or length <= 0.0:
            t = (a / length) if length > 0.0 else 0.0
            # Pull toward both ends of the segment, i.e. toward the corners.
            t = min(max(t - _LEAK_CORNER_BIAS * math.sin(2.0 * math.pi * t),
                        0.0), 1.0)
            return border, t * length
        a -= length
    return 0, 0.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    return {"cuda": "CUDA", "mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(dev.type, dev.type)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur with reflect padding."""
    if sigma < 0.05:
        return x
    r = max(1, int(math.ceil(sigma * 3.0)))
    # reflect padding requires the pad to be smaller than the dimension
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


def _luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = _LUMA
    return x[:, 0:1] * r + x[:, 1:2] * g + x[:, 2:3] * b


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    if e1 - e0 < 1e-5:
        return (x >= e1).to(x.dtype)
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    """Quintic easing -- second derivative is continuous at both ends.

    Used for the luminance band. Cubic smoothstep has a discontinuity in
    curvature where it meets the flat region, and on a wide tonal ramp that
    shows up as a faint edge where grain "switches on". Quintic does not.
    """
    if e1 - e0 < 1e-5:
        return (x >= e1).to(x.dtype)
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _spread(v: torch.Tensor) -> torch.Tensor:
    """Stretch a value-noise field so it actually uses 0..1.

    Quintic value noise is strongly centre-weighted: measured, p10-p90 spans
    only 0.41-0.71 with a standard deviation of 0.11. Used raw as a variation
    field it makes everything land near the middle however wide the range it
    is mapped onto -- which is why the light leaks stayed uniform even with an
    9x spread of reach available to them. The endpoints straddle the field's
    median (0.578), so this stretches without biasing the result up or down.
    """
    return _smoothstep(0.38, 0.78, v)


def _hsv_to_rgb(h_deg: float, sat: float, val: float = 1.0) -> tuple[float, float, float]:
    """HSV to RGB for a single colour, in plain Python.

    The halation tint is one constant per render, not a field, so there is no
    reason to build a tensor for it.
    """
    h = (h_deg % 360.0) / 60.0
    c = val * max(0.0, min(1.0, sat))
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = val - c
    r, g, b = (
        (c, x, 0.0), (x, c, 0.0), (0.0, c, x),
        (0.0, x, c), (x, 0.0, c), (c, 0.0, x),
    )[int(h) % 6]
    return (r + m, g + m, b + m)


def _hue_sat(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hue in degrees and HSV saturation, per pixel.

    Saturation is chroma over value, the HSV definition, matching what
    ``vibrance`` already uses: it reads a deep blue as fully saturated however
    dark it is, where distance from the luma axis would call the same blue
    unsaturated.
    """
    mx = x.amax(dim=1, keepdim=True)
    mn = x.amin(dim=1, keepdim=True)
    c = mx - mn
    sat = c / mx.clamp_min(1e-6)
    cc = c.clamp_min(1e-6)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    h = torch.where(
        mx == r, ((g - b) / cc) % 6.0,
        torch.where(mx == g, (b - r) / cc + 2.0, (r - g) / cc + 4.0),
    ) * 60.0
    # Hue is undefined on grey, and the ratio above is 0/0 there.
    return torch.where(c < 1e-6, torch.zeros_like(h), h), sat


def _rotate_hue(x: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """Rotate colours about the grey axis by a per-pixel angle.

    Rodrigues about (1,1,1)/sqrt(3), which is exactly a hue rotation in RGB:
    it leaves grey untouched by construction (grey lies *on* the axis) and
    preserves the channel sum, so it changes colour without changing how
    bright the pixel is. Cheaper and better behaved than a round trip through
    HSV, which has to divide by a chroma that goes to zero.
    """
    th = deg * (math.pi / 180.0)
    c, s = torch.cos(th), torch.sin(th)
    k = 1.0 / math.sqrt(3.0)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    # k . v, and the axis-aligned part that the rotation leaves alone.
    axis = (r + g + b) * k * k * (1.0 - c)
    return torch.cat([
        r * c + (b - g) * k * s + axis,
        g * c + (r - b) * k * s + axis,
        b * c + (g - r) * k * s + axis,
    ], dim=1)


def _warp(
    x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, mode: str = "bilinear",
) -> torch.Tensor:
    """Resample ``x`` displaced by ``(dx, dy)``, both given in working pixels.

    Shared by edge jitter, edge sanding and scatter -- they differ only in the
    spatial frequency of the field they hand in, not in how it is applied.

    ``mode="nearest"`` makes the result an exact *copy* of a source pixel
    rather than a blend of four, which is the whole point of the scatter
    stage: bilinear resampling at a fractional offset is a 2x2 average, and an
    average is precisely the thing that stage exists not to do. Callers using
    it hand in whole-pixel displacements, so the choice of nearest neighbour
    is unambiguous rather than resting on which side of a half-pixel the
    floating-point arithmetic lands.
    """
    h, w = x.shape[-2:]
    ys = torch.linspace(-1.0, 1.0, h, device=x.device)
    xs = torch.linspace(-1.0, 1.0, w, device=x.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    # Pixel displacements into grid_sample's normalised -1..1 coordinates.
    gx = X.unsqueeze(0).unsqueeze(0) + dx * (2.0 / max(w - 1, 1))
    gy = Y.unsqueeze(0).unsqueeze(0) + dy * (2.0 / max(h - 1, 1))
    grid = torch.stack([gx[:, 0], gy[:, 0]], dim=-1)
    return F.grid_sample(
        x, grid, mode=mode, align_corners=True, padding_mode="border"
    )


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _soft_knee(x: torch.Tensor, amount: float, span: float) -> torch.Tensor:
    """Roll values off asymptotically as they approach 1.0.

    Deliberately *not* normalised to land on 1.0. A shoulder is a region of
    falling slope; if it starts at slope 1 and the slope only decreases, the
    curve mathematically cannot reach 1.0 at the top. Forcing it to would make
    the "shoulder" a highlight *boost*, which is the opposite of film. Letting
    it asymptote below white is what gives film its creamy highlights -- and is
    why a film scan's brightest tone is rarely paper white.
    """
    if amount <= 0.001:
        return x
    knee = 1.0 - span * amount
    denom = max(1.0 - knee, 1e-4)
    t = ((x - knee) / denom).clamp_min(0.0)
    return torch.where(x > knee, knee + denom * torch.tanh(t), x)


# Middle grey (0.18 linear) sits near here once sRGB-encoded; the straight-line
# section of the characteristic curve pivots about it.
_MID_GREY = 0.46


def _characteristic_curve(
    x: torch.Tensor, contrast: float, toe: float, shoulder: float,
) -> torch.Tensor:
    """Film's density-vs-log-exposure response.

    The classical three-part model, in the order film exhibits it: a toe where
    too little light was recorded to develop proportionally, a straight-line
    section whose slope is the gamma, and a shoulder where the halide is
    approaching saturation.
    """
    if contrast > 0.001:
        x = _MID_GREY + (x - _MID_GREY) * (1.0 + 1.1 * contrast)
    x = _soft_knee(x, shoulder, 0.55)
    if toe > 0.001:
        x = 1.0 - _soft_knee(1.0 - x, toe, 0.40)
    return x


def _lattice_np(iy0: int, ix0: int, hl: int, wl: int, seed: int, nfields: int) -> np.ndarray:
    """Deterministic hash noise on an integer lattice window.

    Computed in uint64 on the CPU: integer hashing is exact here and portable,
    whereas 64-bit integer ops are poorly supported on MPS. The lattice is far
    smaller than the pixel grid, so this is cheap.
    """
    # Lattice indices go negative near the origin; reinterpret the bits rather
    # than casting, so negative coordinates wrap into the hash domain.
    yy = np.arange(iy0, iy0 + hl, dtype=np.int64).view(np.uint64)[:, None]
    xx = np.arange(ix0, ix0 + wl, dtype=np.int64).view(np.uint64)[None, :]
    out = np.empty((nfields, hl, wl), dtype=np.float32)
    for f in range(nfields):
        # Fold the seed in Python ints so the wrap is explicit and numpy does
        # not warn about the (intentional) scalar overflow.
        s = np.uint64(((seed + f * 7919) * 0x165667B19E3779F9) % (1 << 64))
        n = xx * np.uint64(0x9E3779B97F4A7C15) + yy * np.uint64(0xC2B2AE3D27D4EB4F)
        n = n + s
        n = n ^ (n >> np.uint64(29))
        n = n * np.uint64(0xBF58476D1CE4E5B9)
        n = n ^ (n >> np.uint64(32))
        n = n * np.uint64(0x94D049BB133111EB)
        n = n ^ (n >> np.uint64(31))
        out[f] = (n >> np.uint64(40)).astype(np.float32) / float(1 << 24)
    return out


def _value_noise(
    h: int, w: int, y0: float, x0: float, cell: float,
    seed: int, nfields: int, device: torch.device, cell_y: float | None = None,
) -> torch.Tensor:
    """Quintic-interpolated value noise addressed by global coordinates.

    Trick: ``grid_sample`` in bilinear mode interpolates by the fractional part
    of the sampling coordinate. Feeding it ``floor(t) + quintic(frac(t))``
    yields exact quintic value noise while keeping the sampler bilinear, which
    is the mode with the broadest backend support.

    ``cell_y`` defaults to ``cell``, giving the isotropic field the grain uses.
    Setting it far larger stretches the field along y, which is how the film
    texture draws scratches: a scratch is just noise whose cells are hundreds
    of pixels tall and a couple wide.
    """
    cy = cell if cell_y is None else cell_y
    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cy
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell

    iy0 = int(math.floor(float(ys[0]))) - 1
    ix0 = int(math.floor(float(xs[0]))) - 1
    hl = int(math.floor(float(ys[-1]))) + 2 - iy0 + 1
    wl = int(math.floor(float(xs[-1]))) + 2 - ix0 + 1

    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, nfields))
    lat = lat.to(device).unsqueeze(0)

    def remap(t: torch.Tensor) -> torch.Tensor:
        fl = torch.floor(t)
        f = t - fl
        return fl + f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

    vi = remap(ys) - iy0
    ui = remap(xs) - ix0
    gy = vi / max(hl - 1, 1) * 2.0 - 1.0
    gx = ui / max(wl - 1, 1) * 2.0 - 1.0
    Y, X = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)

    return F.grid_sample(
        lat, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )


def _cell_noise(
    h: int, w: int, y0: float, x0: float, cell: float, seed: int, nfields: int,
    device: torch.device,
) -> torch.Tensor:
    """One constant hash value per lattice cell -- blocky, and *uniform*.

    The counterpart to ``_value_noise``, and it exists for one reason:
    interpolation destroys the distribution. Quintic value noise is heavily
    centre-weighted -- p10-p90 spans only 0.41-0.71 -- which is fine for a
    field you are going to threshold or spread, and useless for one you are
    going to *quantise*. ``floor(n * 4)`` over an interpolated field returns 1
    or 2 almost every time, so a four-way stencil would fire two of its four
    directions and the scatter would come out with a diagonal bias nobody
    asked for. Reading the lattice without interpolating gives back the hash's
    own uniform distribution, so every direction is equally likely.

    Blockiness is the other half of the point. Every pixel inside one cell
    reads the same value, so a whole cell of image is displaced as a unit --
    that is what ``scatter_cell`` means, and it is why detail can survive the
    trip instead of dissolving. Addressed by global coordinates like every
    other field here, so two tiles asking about the same pixel agree.
    """
    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cell
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell
    iy0 = int(math.floor(float(ys[0])))
    ix0 = int(math.floor(float(xs[0])))
    hl = int(math.floor(float(ys[-1]))) - iy0 + 1
    wl = int(math.floor(float(xs[-1]))) - ix0 + 1

    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, nfields)).to(device)
    iy = (torch.floor(ys).long() - iy0).clamp(0, hl - 1)
    ix = (torch.floor(xs).long() - ix0).clamp(0, wl - 1)
    return lat[:, iy][:, :, ix].unsqueeze(0)


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


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

class GrainEngine:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or pick_device()

    # ------------------------------------------------------------------ #
    def _grain_field(
        self, h: int, w: int, y0: float, x0: float, lum: torch.Tensor,
        p: dict, scale: float,
    ) -> torch.Tensor:
        """Signed, roughly unit-scale grain field, shape [1,3,h,w]."""
        dev = self.device
        cell = max(_MIN_CELL, p["grain_size"] * scale)
        seed = int(p["seed"])
        octaves = int(round(p["octaves"]))
        rough = p["roughness"]

        n = _fbm(h, w, y0, x0, cell, seed, 3, octaves, rough, dev)

        # Shadows carry larger, less densely packed crystals.
        ss = p["shadow_size"]
        if ss > 0.02:
            big = _fbm(
                h, w, y0, x0, cell * (1.0 + 1.2 * ss), seed + 5077, 3,
                max(1, octaves - 1), rough, dev,
            )
            sw = ss * (1.0 - _smoothstep(0.0, 0.6, lum))
            n = n * (1.0 - sw) + big * sw

        s = n * 2.0 - 1.0

        # Monochrome component is the mean of the three dye layers, rescaled to
        # preserve variance; chroma_grain blends toward independent layers.
        mono = s.mean(dim=1, keepdim=True) * math.sqrt(3.0)
        g = mono + p["chroma_grain"] * (s - mono)

        # Clump curve: push the distribution toward discrete clumps.
        t = (g / _GNORM).clamp(-1.0, 1.0)
        gamma = 1.0 - 0.75 * p["clump"]
        if abs(gamma - 1.0) > 1e-3:
            t = torch.sign(t) * t.abs().clamp_min(1e-6) ** gamma
        return t

    # ------------------------------------------------------------------ #
    @staticmethod
    def _blue_guard(
        lin: torch.Tensor, amount: float, level: float, falloff: float,
        shift: float,
    ) -> torch.Tensor:
        """Strengthen blue in linear light, before the halation wash lands.

        Three things have to agree before a pixel is compensated, and the
        third is the one that took a correction from the user to get right:

        * **It is blue** -- a hue window around ``_BLUE_HUE``.
        * **It has colour to strengthen** -- weighted by existing saturation,
          the same principle as ``vibrance`` and for the same reason: it must
          strengthen blue that is *there* and never invent it in something
          grey, or every neutral shadow in the frame picks up a cast.
        * **It is light enough to have been damaged.** The wash only reaches
          what is near the light. Measured up a sky gradient away from the
          sun, saturation loss is 23% at the bright end and flat *zero* below
          about half brightness -- so compensating a deep blue is pure
          overshoot, and at amount 2.0 it drove an untouched sky from 0.872
          saturation to 1.000, i.e. a channel clamped to black. That is not a
          setting to avoid, it is a missing term in the mask.

        Knee and falloff are separate controls. Deriving the width from the
        knee would make moving one change the other, and a sky is precisely
        the broad smooth gradient that shows up a hard switch-on -- which is
        also why the ramp is quintic, like the luminance band's.

        Both operations are weighted by the mask *inside* themselves rather
        than blended toward a fully-processed copy. A hue rotation of ``m *
        shift`` degrees is the identity at ``m = 0``; cross-fading toward a
        fully rotated colour instead would mix two different hues and lose a
        little saturation in the middle of the ramp, which is exactly the
        artifact this stage exists to fix.
        """
        h, sat = _hue_sat(lin)
        d = (h - _BLUE_HUE).abs()
        d = torch.minimum(d, 360.0 - d)  # the wheel wraps
        m = (1.0 - _smoothstep(0.0, _BLUE_RANGE, d)) * _smoothstep(
            0.0, _BLUE_SAT_FLOOR, sat
        )
        # Brightness gate, read display-referred so the slider means the same
        # thing as every other luminance control in the app. Linear luma would
        # crush an ordinary sky down to 0.05 and make the top nine tenths of
        # the slider useless.
        #
        # Encode first, *then* take the luma. Taking the luma of the linear
        # image and encoding that single number is cheaper and wrong: the
        # transfer curve is non-linear, so it does not commute with a weighted
        # sum. Measured, it reads a deep sky 23% brighter than it is, which
        # would put this slider on a different scale from the Luminance
        # Response knees it is meant to match.
        lum_d = _luma(_linear_to_srgb(lin))
        m = m * _smootherstep(max(0.0, level - falloff), level, lum_d)

        if abs(shift) > 0.5:
            lin = _rotate_hue(lin, m * shift)
        if amount > 0.001:
            lum_b = _luma(lin)
            lin = lum_b + (lin - lum_b) * (1.0 + amount * m)
        # The rotation can put a channel marginally below zero on a very
        # saturated colour; halation adds to this and sRGB encoding assumes
        # non-negative.
        return lin.clamp_min(0.0)

    # ------------------------------------------------------------------ #
    def _scatter(
        self, x: torch.Tensor, h: int, w: int, y0: float, x0: float,
        p: dict, scale: float,
    ) -> torch.Tensor:
        """Displace a share of the pixels onto their neighbours, without averaging.

        A blur and this stage model the same physics from opposite ends. Light
        diffusing through the emulsion is a stochastic process: a photon either
        goes straight or is deflected onto a neighbouring grain. Average over
        infinitely many photons and you get a convolution -- ``micro_blur``,
        which is smooth because it is an expectation. Resolve the deflections
        individually and you get this: detail lands somewhere it was not,
        every value survives intact, and the result is *disordered* rather
        than smoothed. That is the whole reason the stage exists. A digital
        frame softened with a blur reads as out of focus because the blur
        removes the micro-contrast along with the edge; scatter removes
        neither, and takes the exactness instead.

        Three properties follow from never averaging, and all three are why
        this is not just another kernel:

        * **No value is invented.** Every output pixel is a bit-exact copy of
          some input pixel, so the frame's histogram, its grit and its noise
          come through untouched. Sampling is nearest-neighbour on whole-pixel
          offsets specifically to keep that true -- bilinear at a fractional
          offset would quietly turn each sample into a 2x2 average.
        * **Amount is coverage, not opacity.** ``scatter`` moves the threshold
          on a uniform field, so it sets *how many* pixels travel. Cross-fading
          a displaced pixel with the one it left would be an average by
          another name, and at 0.5 it would read as exactly the blur this
          replaces.
        * **It masks itself.** Displacing a pixel whose neighbours already
          match it changes nothing, so smooth sky, skin and studio backdrops
          come out untouched with no mask anywhere in the code. The stage acts
          only where there is detail to disorder, which is the inverse of
          ``micro_blur``'s failure mode -- that one takes texture down first
          and edges second.

        There is deliberately no frequency split here, and I built one before
        working out why it was pointless -- see the note in CLAUDE.md. The
        stage is already frequency-selective by construction: a displacement
        can only change a pixel by as much as the picture varies over the
        distance travelled, so structure coarser than the reach survives for
        free and ``scatter_radius`` is the frequency control.
        """
        amt = p["scatter"]
        reach = max(0.5, p["scatter_radius"] * scale)
        # Cells finer than a working pixel cannot be resolved; below that the
        # nearest-neighbour read just aliases between them.
        cell = max(1.0, p["scatter_cell"] * scale)
        pattern = int(round(p["scatter_pattern"])) % len(_SCATTER_STENCILS)

        n = _cell_noise(h, w, y0, x0, cell, int(p["seed"]) + 3301, 3, self.device)
        sel, mag_n, gate = n[:, 0:1], n[:, 1:2], n[:, 2:3]

        # Direction and distance, on the stencil, in whole pixels -- whole so
        # the gather stays a copy rather than an interpolation. Reach Spread:
        # 0 puts every displaced pixel on the shape's edge (detail hollows
        # out), 1 fills it inward.
        dx, dy = _scatter_offsets(
            sel, mag_n, reach, p["scatter_spread"], pattern
        )
        # Coverage: a uniform field thresholded at the amount, so `amt` is
        # literally the fraction of the frame that moves. Applied after the
        # rounding so a pixel that is not travelling gets a displacement of
        # exactly zero and reads itself back.
        move = (gate < amt).to(x.dtype)
        dx, dy = dx * move, dy * move

        return _warp(x, dx, dy, mode="nearest")

    # ------------------------------------------------------------------ #
    def render(
        self, img: torch.Tensor, p: dict, scale: float = 1.0,
        y0: float = 0.0, x0: float = 0.0,
        full_hw: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Render one tile. ``img`` is [1,3,h,w] float in 0..1 on ``self.device``.

        ``scale`` is working-res / full-res; ``y0``/``x0`` are the tile's offset
        in working-resolution coordinates.
        """
        h, w = img.shape[-2:]
        hp_r = max(0.3, p["highpass_radius"] * scale)
        mb = p["micro_blur"] * scale

        # 0. Pre-sharpen, on the untouched input.
        #
        #    Placed before everything so it sharpens the *photograph* and
        #    nothing else -- there is no grain yet to amplify. It is not
        #    cosmetic to put it here rather than at the end: every mask
        #    downstream is measured from this image, so sharpening now makes
        #    edges read as harder to the edge mask and pulls grain onto them.
        ps = p["pre_sharpen"]
        if ps > 0.01:
            img = (
                img
                + (img - _blur(img, max(0.3, p["pre_sharpen_radius"] * scale))) * ps
            ).clamp(0.0, 1.0)

        # ---- EXPOSURE STAGE (linear light) --------------------------------
        # Diffusion and halation are things that happen to *light*, before the
        # emulsion records anything, so they are done in linear light. Doing
        # them in gamma-encoded space is the usual reason simulated halation
        # looks like a painted-on glow rather than light.
        lin = _srgb_to_linear(img)

        # 1. Light diffusing sideways through the gel layers, as an average.
        lin = _blur(lin, mb)

        # 1b. The same diffusion resolved as discrete deflections instead of
        #     as an average -- see _scatter for why that is a different
        #     operation and not a slower blur. Here beside micro-blur because
        #     it is the same physical event, and in linear light for the same
        #     reason: it happens to the light, before the emulsion records
        #     anything.
        #
        #     Note the masks below are measured from the *untouched* tile
        #     input, so scattering the frame does not talk the edge mask or
        #     the smooth-area guard into turning grain down -- the same
        #     independence micro-blur has, and for the same reason.
        if p["scatter"] > 0.001:
            lin = self._scatter(lin, h, w, y0, x0, p, scale)

        # 2. Halation: light reaching the film base reflects and re-exposes the
        #    emulsion from behind, blooming warm around bright highlights.
        hal = p["halation"]
        if hal > 0.01:
            thr = min(p["halation_threshold"], 0.98)
            thr_lin = ((thr + 0.055) / 1.055) ** 2.4
            lum0 = _luma(lin)
            hi = ((lum0 - thr_lin) / max(1.0 - thr_lin, 0.02)).clamp(0.0, 1.0)
            glow = _blur(hi, max(1.0, p["halation_radius"] * scale))

            # 2a. Blue compensation, applied to the image the wash is about to
            #     land on rather than to the result.
            #
            #     Halation adds warm light, and *adding light desaturates
            #     whatever it lands on* -- that is not a side effect to be
            #     tuned out, it is what addition does. A red-tinted bloom
            #     lifts a blue sky's red channel by the full glow and its blue
            #     channel by a tenth of it, so the sky loses colour and drifts
            #     toward grey and then toward purple.
            #
            #     Correcting afterwards was the obvious alternative and is
            #     worse for two measured reasons. It has no brake: the wash
            #     eats a fixed share of anything added *before* it, so
            #     compensating here self-limits -- everything from amount 1.0
            #     to 3.0 lands 3% past the untouched sky's own saturation --
            #     where the identical correction applied *after* is 9% past by
            #     0.5 and by 1.0 has driven a channel to black and pinned the
            #     sky at fully saturated. And it cannot tell blue
            #     that was unfairly washed from blue the bloom is *supposed*
            #     to be sitting on, so re-saturating there fights the glow you
            #     paid for -- it would need the glow field carried out of this
            #     block to know the difference. Here the question never
            #     arises: this changes what was recorded, and halation then
            #     does its job to it. That is also the physical order --
            #     a punchier blue layer or a polariser, not retouching.
            #
            #     Deliberately after `glow` is computed, so compensation
            #     cannot move the bloom: the two controls stay independent and
            #     `verify.py` pins it. Purely per-pixel, so `pad_for` is
            #     unaffected.
            blue = p["halation_blue"]
            bshift = p["halation_blue_shift"]
            if blue > 0.001 or abs(bshift) > 0.5:
                lin = self._blue_guard(
                    lin, blue, p["halation_blue_level"],
                    p["halation_blue_falloff"], bshift,
                )
            # Tint from a full hue wheel rather than the old red-to-amber
            # ramp, which spanned about 25 degrees and could not desaturate.
            # Real halation is red -- that is what the antihalation layer and
            # the red-sensitive layer conspire to produce, and 0-40 degrees is
            # the physically honest region -- but this is a look tool, so the
            # rest of the wheel is reachable.
            tint = torch.tensor(
                _hsv_to_rgb(p["halation_hue"], p["halation_sat"]),
                device=lin.device, dtype=lin.dtype,
            ).view(1, 3, 1, 1)
            lin = lin + glow * tint * (hal * 0.9)

        # ---- DEVELOPMENT STAGE (density / display space) ------------------
        base = _linear_to_srgb(lin)

        # 3. Brightness, then the characteristic curve: toe, straight line,
        #    shoulder.
        #
        #    Brightness is a multiply in *linear* light, which is what makes it
        #    behave like exposure rather than like a levels slider: doubling
        #    the light doubles it everywhere, and the sRGB encoding on the way
        #    back rolls the top off by itself. Multiplying the display-referred
        #    signal instead would stretch the highlights straight into a flat
        #    clip.
        #
        #    Before the curve, not after, so the shoulder catches the
        #    highlights brightness raises instead of being applied to the
        #    unbrightened image and then overrun.
        br = p["brightness"]
        if abs(br) > 0.001:
            base = _linear_to_srgb(_srgb_to_linear(base) * (2.0 ** br))
        base = _characteristic_curve(base, p["contrast"], p["toe"], p["shoulder"])

        # 4. Dye layers desaturate as they approach saturation, rather than
        #    clipping to a hue-shifted edge the way a sensor does.
        hd = p["highlight_desat"]
        if hd > 0.01:
            lum_h = _luma(base)
            wgt = _smoothstep(0.62, 1.0, lum_h) * hd
            base = base + wgt * (lum_h - base)

        # 4b. Vibrance: a saturation push weighted *against* how saturated a
        #     pixel already is, so muted colour comes up while colour that is
        #     already strong is left alone. That weighting is the whole
        #     difference from a flat saturation control, which drags everything
        #     up together and takes already-saturated regions straight out of
        #     gamut -- skin and skies being the usual casualties.
        #
        #     Saturation is measured as chroma over value, the HSV definition,
        #     which reads a deep red as fully saturated regardless of how dark
        #     it is. Distance from the luma axis would call the same red
        #     unsaturated and then boost it further.
        vib = p["vibrance"]
        if abs(vib) > 0.001:
            mx = base.amax(dim=1, keepdim=True)
            mn = base.amin(dim=1, keepdim=True)
            sat = (mx - mn) / mx.clamp_min(1e-4)
            lum_v = _luma(base)
            # Clamped at zero so a strong negative setting lands on neutral
            # grey rather than inverting the colour through it.
            gain = (1.0 + vib * (1.0 - sat)).clamp_min(0.0)
            base = lum_v + (base - lum_v) * gain

        # 5. Cross-channel bias: warm highlights, cool shadows. Most of what
        #    reads as "a film palette" lives here, not in the grain.
        wh, cs = p["warm_highlights"], p["cool_shadows"]
        if wh > 0.01 or cs > 0.01:
            lum_s = _luma(base)
            warm = torch.tensor([0.055, 0.012, -0.040], device=base.device,
                                dtype=base.dtype).view(1, 3, 1, 1)
            cool = torch.tensor([-0.030, 0.002, 0.050], device=base.device,
                                dtype=base.dtype).view(1, 3, 1, 1)
            base = base + _smoothstep(0.45, 1.0, lum_s) * warm * wh
            base = base + (1.0 - _smoothstep(0.0, 0.5, lum_s)) * cool * cs

        # 6. Base fog: the film base has a minimum density, so there is no true
        #    black. Lifts the floor without touching the white point.
        fog = p["base_fog"]
        if fog > 0.001:
            base = fog + (1.0 - fog) * base

        base = base.clamp(0.0, 1.0)

        # 7. Edge isolation (needed before jitter so we only warp real edges).
        #
        #    Measured from the *untouched tile input*, not from `base`. Every
        #    softening stage above -- micro-blur especially -- flattens exactly
        #    the micro-edges this mask keys on, so reading `base` meant that
        #    softening the picture also quietly turned the grain down: dial in
        #    some diffusion and you lost noise you never asked to lose. Keying
        #    off the original structure decouples the two, so softness and
        #    grain amount are independent controls. Tone curves ship neutral,
        #    so this is also very close to what `base` used to give.
        lum = _luma(base)
        lum_ref = _luma(img)
        hp = lum_ref - _blur(lum_ref, hp_r)
        edge = (hp.abs() / EDGE_REF).clamp(0.0, 1.0)
        edge = _blur(edge, hp_r * 0.8)

        # 7b. Edge softening. A global blur is the wrong tool for "make it
        #     softer": it takes the whole frame down, texture and all, and
        #     reads as out of focus rather than as film.
        #
        #     Note this cannot key on `edge` above. That mask asks "is there a
        #     micro-edge here", and fine texture is *made of* micro-edges -- so
        #     weighting by it softened fabric and hair almost as much as it
        #     softened a hard border. The discriminator has to be edge
        #     *amplitude*: a real transition steps a long way in luminance,
        #     where texture wobbles by a little. Measured over the softening
        #     radius, a hard border reads several times _STEP_HI while fine
        #     texture sits under _STEP_LO, so the threshold cleanly separates
        #     them where a high-pass alone cannot.
        es = p["edge_soften"]
        if es > 0.01:
            sr = max(0.3, p["edge_soften_radius"] * scale)
            step = (lum_ref - _blur(lum_ref, sr)).abs()
            hard = _smoothstep(_STEP_LO, _STEP_HI, step)
            hard = _blur(hard, sr * 0.6)
            base = base + (_blur(base, sr) - base) * (hard * es)
            lum = _luma(base)

        # A smooth envelope traces an edge too precisely and reads as a digital
        # outline. Emulsion erodes an edge unevenly, so break the envelope up
        # with its own noise field (mean preserved at ~1.0).
        edge_clean = edge
        ragged = _fbm(
            h, w, y0, x0, max(_MIN_CELL, p["grain_size"] * scale * 2.0),
            int(p["seed"]) + 4241, 1, 2, 0.6, self.device,
        )
        edge = edge * (0.55 + 0.9 * ragged)

        # 8. Sub-pixel edge jitter -- destroys hyper-sharp digital borders
        #    without wobbling flat areas. The noise cell is several times the
        #    clump size, so the displacement field is smooth along the edge and
        #    a border *wanders*: long, slow deviations.
        jit = p["edge_jitter"]
        if jit > 0.01:
            d = _fbm(h, w, y0, x0, max(_MIN_CELL, p["grain_size"] * scale * 3.0),
                     int(p["seed"]) + 911, 2, 1, 1.0, self.device) * 2.0 - 1.0
            dx, dy = d[:, 0:1], d[:, 1:2]

            # Directional bias. The raw field is isotropic -- measured, every
            # 45-degree sector takes 12-13% of displacements at the same mean
            # magnitude -- so simply *rotating* it would be a no-op: a rotated
            # isotropic field is the same field. What makes an angle mean
            # something is squeezing the displacement onto one axis first.
            #
            # Work in the rotated frame: u runs along the chosen axis, v across
            # it. Scaling v down concentrates the travel along u, so at
            # anisotropy 1 edges only ever move parallel to the angle. At 0
            # this is exactly the isotropic behaviour, whatever the angle says.
            aniso = p["jitter_aniso"]
            if aniso > 0.01:
                th = math.radians(p["jitter_angle"])
                ca, sa = math.cos(th), math.sin(th)
                u = dx * ca + dy * sa
                v = (dy * ca - dx * sa) * (1.0 - aniso)
                dx, dy = u * ca - v * sa, u * sa + v * ca

            amp = _JITTER_MAX * jit * max(scale, 0.25) * edge
            base = _warp(base, dx * amp, dy * amp)
            lum = _luma(base)

        # 8b. Edge sanding -- takes the jaggedness back off, the way sandpaper
        #     does. Jitter roughens a border; left alone that reads as stair-
        #     stepped and harsh. This polishes it.
        #
        #     The operation is a blur *along* the edge, not across it. Smooth
        #     across a border and you have destroyed the border; smooth along
        #     it and the fine burrs average out while the transition stays as
        #     sharp as it was. So each pixel is averaged with its neighbours in
        #     the direction perpendicular to the local gradient -- the isophote
        #     tangent, i.e. the direction the edge actually runs.
        #
        #     The radius is what "grit" means here: a small radius reaches only
        #     the pixel-scale jaggies (a fine polish, shape untouched), a large
        #     one flattens broader undulations too.
        snd = p["edge_sand"]
        if snd > 0.01:
            total = max(0.5, p["edge_sand_grit"] * scale)
            # Applied as several short passes rather than one long one, with
            # the direction recomputed each time. The taps run in a straight
            # line, but the edge being sanded is precisely one that wanders --
            # so a single wide pass runs off the contour and cuts across it,
            # costing sharpness the filter exists to preserve. Short passes
            # re-aim, following the curve.
            #
            # The gain is real but modest: matched at 32% of the jaggedness
            # removed, iterating keeps 81% of the wander and 73% of the edge
            # sharpness against 79% and 71% for a single wide pass. It also
            # spreads the response more evenly over the grit range, which
            # matters more here -- this is a fine-tuning control.
            passes = int(min(_SAND_PASSES, max(1, round(total / 1.2))))
            sr = total / passes
            for _ in range(passes):
                # Direction from a blurred luma: taken per-pixel it would
                # follow the grain and jitter it is meant to remove, and sand
                # in circles.
                #
                # The blur has to scale with the sanding radius, not sit at a
                # fixed width. Where the gradient is weak the tangent is
                # numerically unstable -- it is a ratio of two near-zero
                # numbers -- and a filter reaching 13px along an arbitrary
                # direction samples somewhere entirely different for an
                # imperceptible change in input. That is not merely noisy: it
                # made tiled exports seam from 8px grit upward, because the
                # two tilings hand the gradient marginally different values.
                # Estimating direction over a window comparable to the reach
                # keeps it coherent and the result tile-independent.
                gl = _blur(lum, max(0.6, _SAND_DIR_K * sr))
                px_ = F.pad(gl, (1, 1, 0, 0), mode="replicate")
                gx_ = (px_[..., 2:] - px_[..., :-2]) * 0.5
                py_ = F.pad(gl, (0, 0, 1, 1), mode="replicate")
                gy_ = (py_[..., 2:, :] - py_[..., :-2, :]) * 0.5
                mag = (gx_ * gx_ + gy_ * gy_).sqrt().clamp_min(1e-6)
                # Tangent is the gradient turned 90 degrees.
                tx, ty = -gy_ / mag, gx_ / mag
                # Where the gradient vanishes the tangent is a ratio of two
                # near-zero numbers and its direction is meaningless -- it
                # will swing on floating-point noise alone, and a filter
                # reaching a dozen pixels along it then samples somewhere
                # entirely different. Left ungated this showed up as a handful
                # of isolated pixels per frame disagreeing between a tiled and
                # a single-pass render. Fading the effect out with the
                # gradient fixes it and costs nothing: a region with no
                # gradient has no edge to sand.
                coherent = _smoothstep(0.0, _SAND_MIN_GRAD, mag)

                sanded = None
                wsum = 0.0
                for offv, wgt in _SAND_TAPS:
                    tap = (
                        base if offv == 0.0
                        else _warp(base, tx * (offv * sr), ty * (offv * sr))
                    )
                    sanded = tap * wgt if sanded is None else sanded + tap * wgt
                    wsum += wgt
                # Normalised here rather than trusting the table to sum to one
                # -- truncated gaussian weights do not, and the shortfall would
                # show up as every sanded edge being fractionally darker.
                sanded = sanded / wsum

                # Gated on the pre-ragged mask: the ragged envelope exists to
                # make erosion uneven, and sanding through it would polish in
                # patches.
                base = base + (sanded - base) * (edge_clean * coherent * snd)
                lum = _luma(base)

        # 9. Luminance response: grain is at full strength across the band
        #    [lum_low, lum_high] and eases out over a falloff width on each
        #    side. Band edges and transition widths are independent -- welding
        #    them together forces the ramp to start at pure black or run all
        #    the way to white, which is what makes the boundary visible.
        #
        #    The mask is driven by a spatially blurred luma so the transition is
        #    smooth across the *frame* as well as across the tone curve. Reading
        #    per-pixel luma lets image detail modulate the mask itself, which
        #    speckles the boundary region.
        lum_m = _blur(lum, max(1.0, 3.0 * scale))
        lo = p["lum_low"]
        hi = max(p["lum_high"], lo + 0.05)
        sf = max(p["shadow_falloff"], 1e-3)
        hf = max(p["highlight_falloff"], 1e-3)

        up_ramp = _smootherstep(max(0.0, lo - sf), lo, lum_m)
        dn_ramp = 1.0 - _smootherstep(hi, min(1.0, hi + hf), lum_m)
        m = (1.0 - p["shadow_drop"]) + p["shadow_drop"] * up_ramp
        m = m * ((1.0 - p["highlight_drop"]) + p["highlight_drop"] * dn_ramp)

        # 10. Grain field, weighted toward micro-edges and away from flat areas.
        g = self._grain_field(h, w, y0, x0, lum, p, scale)
        eb = p["edge_bias"]
        weight = m * ((1.0 - eb) + eb * edge)

        # Smooth-area guard. The edge mask only sees micro-edges, so a smooth
        # gradient -- skin, a clear sky, a studio backdrop -- gets no protection
        # from it and takes the full flat-area floor. That is what makes skin
        # read as jagged. Measure local contrast over a medium radius instead:
        # a linear gradient has almost none (blurring a ramp returns the ramp),
        # while fabric, foliage and hair have plenty. Suppress grain where that
        # measure says the region is genuinely featureless.
        sg = p["smooth_guard"]
        if sg > 0.01:
            med_r = max(1.0, hp_r * 2.5)
            # From the reference luma for the same reason as the edge mask: a
            # softened region is not a featureless one, and blurring the frame
            # should not talk the guard into treating fabric as skin.
            tex = _blur((lum_ref - _blur(lum_ref, med_r)).abs(), med_r)
            textured = _smoothstep(_TEX_LO, _TEX_HI, tex)
            weight = weight * ((1.0 - sg) + sg * textured)

        amp = (p["intensity"] / 100.0) * _AMP_SCALE
        out = base + g * weight * amp

        # 11. Structural erosion: modulate the image's own micro-detail by the
        #    grain field. Zero in flat areas, strongest on edges.
        er = p["edge_erosion"]
        if er > 0.01:
            detail = base - _blur(base, hp_r)
            # Per-channel modulation of a high-contrast edge gives each dye
            # layer its own erosion, producing coloured speckle along the edge.
            # ``edge_chroma`` blends between neutral erosion and full fringing.
            mono_g = g.mean(dim=1, keepdim=True)
            eg = mono_g + p["edge_chroma"] * (g - mono_g)
            out = out + eg * detail * weight * (1.6 * er)

        # 12. Adjacency (Eberhard) effect. Developer exhausts faster on the
        #     dense side of an edge and diffuses across it, leaving a local
        #     contrast boost. Extracted from the pre-grain base so it sharpens
        #     the image rather than amplifying the grain we just added.
        acut = p["acutance"]
        if acut > 0.01:
            out = out + (base - _blur(base, hp_r * 1.5)) * (0.35 * acut)

        # 13. Global grain -- a flat overlay, applied last and masked by
        #     nothing.
        #
        #     Everything above is masked: by the luminance band, by the edge
        #     envelope, by the smooth-area guard. That is emulsion behaviour,
        #     and it is why smooth skies and skin stay clean. This layer is
        #     deliberately none of that. It sits on the finished frame at one
        #     amplitude everywhere, the way a scanned print carries grain from
        #     the print stock and the scan itself rather than from the
        #     negative -- so it reaches exactly the areas the masks protect.
        #
        #     Monochrome, and on its own seed offset: sharing the main grain's
        #     seed would lay it directly on top of the same clumps and read as
        #     nothing more than a louder version of the same field.
        gi = p["global_intensity"]
        go = p["global_opacity"]
        if gi > 0.01 and go > 0.001:
            gg = _fbm(
                h, w, y0, x0, max(_MIN_CELL, p["global_size"] * scale),
                int(p["seed"]) + 7717, 1, 2, 0.5, self.device,
            )
            gg = ((gg * 2.0 - 1.0) / _GNORM).clamp(-1.0, 1.0)
            out = out + gg * ((gi / 100.0) * _AMP_SCALE * go)

        # 14. Output sharpening -- deliberately the last thing in the pipeline.
        #
        #     An unsharp mask amplifies whatever high-frequency content it
        #     finds, and by this point that is the grain as much as the image.
        #     That is the entire reason it sits here rather than earlier: it
        #     cranks the noise already present instead of generating any, so
        #     grain gains bite and the picture gains acutance from the same
        #     operation. Run before the grain stages it would sharpen a clean
        #     image and leave the grain flat, which is the opposite of the
        #     intent.
        #
        #     Distinct from `acutance`, which is an edge-local development
        #     effect extracted from the *pre-grain* base specifically so it
        #     sharpens the image without amplifying grain. This one is the
        #     blunt instrument, and it is applied to the unclamped signal so
        #     overshoot keeps its headroom until the final clamp.
        sh = p["sharpen"]
        if sh > 0.01:
            out = out + (out - _blur(out, max(0.3, p["sharpen_radius"] * scale))) * sh

        # 15. Physical damage, after everything including sharpening -- a
        #     speck of dust sits on the film, it was never in the picture, so
        #     it must not be sharpened, grained or masked along with it.
        out = self._film_texture(out, h, w, y0, x0, p, scale, full_hw)

        return out.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def _film_texture(
        self, out: torch.Tensor, h: int, w: int, y0: float, x0: float,
        p: dict, scale: float, full_hw: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Physical damage: dust, scratches, hair, light leaks.

        Everything above this point models what the *emulsion* does. This
        models what happened to the piece of film afterwards -- it got dusty,
        it got dragged through a gate, someone's hair landed on the scanner
        bed, the back came loose. That is why it sits last and is weighted by
        none of the image masks: a scratch does not care what is underneath it.

        All four are drawn by thresholding noise addressed in global
        coordinates rather than by scattering objects. Scattering would need a
        list of positions, and a list is a statistic of the region -- it would
        break tile independence the moment an export split a scratch across
        two tiles. Thresholded noise gives every pixel the same answer no
        matter which tile asks, and it also stops the marks looking stamped:
        their outlines are organic because the field is.
        """
        dev = self.device
        seed = int(p["texture_seed"])
        # Counts are per *frame*, so they need its size. Without it (a caller
        # that did not pass full_hw) the counted marks are skipped rather than
        # guessed at from the tile, which would put N marks on every tile.
        area = None if full_hw is None else max(full_hw[0] * full_hw[1], 1.0)

        # -- light leak ---------------------------------------------------
        # Light that got past a seal, so it is anchored to the frame rather
        # than floating in the image, and it is added in linear light because
        # it is light.
        #
        # Drawn as a handful of discrete *beams*, which is the whole shape of
        # this stage and the thing it got wrong before. The old version was a
        # falloff from the nearest border gated by a slow noise field along it:
        # every leak was therefore a soft inward wash with no direction, no
        # length and no edge, present on all four borders at once -- a chewed-up
        # vignette. Real leaks are streaks with a definite edge limiting their
        # reach; they come from one or two places on the frame, they lean
        # across it, and they stop somewhere.
        #
        # So each leak is a beam: a source on the perimeter, a depth it
        # penetrates (`leak_size_*`), a lean (`shear`), a width that fans out
        # as it travels, and one hard edge where the obstruction's shadow is.
        # Noise now *perturbs* that shape instead of being it.
        ll = p["light_leak"]
        if ll >= 1.0 and full_hw is not None:
            fh = max(float(full_hw[0]), 1.0)
            fw = max(float(full_hw[1]), 1.0)
            Ypx = (torch.arange(h, device=dev, dtype=torch.float32)
                   + float(y0)).view(1, 1, h, 1)
            Xpx = (torch.arange(w, device=dev, dtype=torch.float32)
                   + float(x0)).view(1, 1, 1, w)

            var = p["leak_variation"]
            # Swapped if given the wrong way round, so dragging either slider
            # past the other never makes the leaks vanish.
            s_lo = min(p["leak_size_min"], p["leak_size_max"]) * scale
            s_hi = max(p["leak_size_min"], p["leak_size_max"]) * scale
            # Cap at half the frame's short side over the warp's headroom:
            # that is the depth at which the falloff dies exactly in the
            # middle of the frame, and past it a leak leaves a floor over the
            # whole picture -- centre fog, which reads as a bad exposure
            # rather than as a leak. Geometric, not a taste constant.
            reach_cap = 0.5 * min(fh, fw) / _LEAK_REACH_SAFETY
            # The along-border edges want a softness as a 0..1, and the honest
            # 0..1 is the feather measured against the sizes asked for -- a
            # 50px feather is a rim on a 400px leak and a wash on an 80px one.
            # Derived from the parameters alone, never from the field, so it
            # is a constant per render and tiles cannot disagree about it.
            soft = min(1.0, p["leak_feather"] / max(
                0.5 * (p["leak_size_min"] + p["leak_size_max"]), 1.0))
            bw_soft = 0.12 + 0.75 * soft

            expo_lin = _srgb_to_linear(out)
            # Per-channel exposure, accumulated over the beams. Light adds, so
            # two leaks overlapping is brighter than either -- and it has to be
            # per channel rather than a scalar times one tint, because each
            # leak carries its own hue.
            expos = torch.zeros(1, 3, h, w, device=dev, dtype=torch.float32)

            for k, st in enumerate(_leak_sites(ll, seed, var)):
                border, s0 = _leak_anchor(st["pos"], fh, fw)
                # `u` is the perpendicular depth from this leak's own border
                # and `s` runs along it. Keeping the obliquity in a shear on
                # `s` rather than rotating the whole frame is what lets a leak
                # lean hard across the picture while `reach` stays exactly the
                # depth the slider promises.
                if border == 0:
                    u, s, blen = Ypx, Xpx, fw
                elif border == 1:
                    u, s, blen = fh - Ypx, Xpx, fw
                elif border == 2:
                    u, s, blen = Xpx, Ypx, fh
                else:
                    u, s, blen = fw - Xpx, Ypx, fh

                reach = min(s_lo + (s_hi - s_lo) * st["reach_t"], reach_cap)
                # How far the leak runs *along* its border. Measured against
                # the border, not against the reach -- and that is the second
                # thing the old shape got wrong. A seal fails along a seam, so
                # the leak is a band that runs a long way sideways and comes in
                # a modest depth; sizing its length off its depth instead makes
                # every leak roughly as long as it is deep, which is a blob.
                # Floored against the reach because light through a slot cannot
                # be much narrower than it is deep.
                hw0 = max(blen * st["width"], 0.55 * reach)

                # Two octaves of domain warp. The coarse one wanders the whole
                # beam, the fine one frays its edge; between them the outline is
                # organic while still being an outline -- which is the inversion
                # that matters here. Noise used to *be* the shape and the result
                # was fog; now it perturbs a shape that has a definite edge.
                # The depth amplitudes sum to exactly `_LEAK_WARP * reach`,
                # which is what the reach cap was sized against.
                wn = _value_noise(h, w, y0, x0, max(16.0, 0.80 * reach),
                                  seed + 9137 + k * 37, 3, dev)
                wf = _value_noise(h, w, y0, x0, max(6.0, 0.25 * reach),
                                  seed + 9701 + k * 37, 2, dev)
                warp = (wn[:, 0:1] - 0.5) * 1.5 + (wf[:, 0:1] - 0.5) * 0.5
                # Clamped at zero: the warp may pull the beam *inward*, and
                # the falloff below has to stay defined at the border.
                du = (u + warp * _LEAK_WARP * reach).clamp_min(0.0)
                lat = (wn[:, 1:2] - 0.5) * 1.5 + (wf[:, 1:2] - 0.5) * 0.5
                dv = (s - s0) - st["shear"] * du + lat * 0.18 * hw0

                # Along the beam: the same feather-to-exponent mapping the
                # pixel sizes have always used. Solving (1 - hl/reach)^e = 0.5
                # gives e = ln(0.5) / ln(1 - hl/reach), so the feather is a
                # visible distance -- short is a tight bright rim on the
                # border, half the reach is a straight ramp, most of the reach
                # is a broad wash. Scalars per leak now rather than fields,
                # since a beam has one of each.
                hl = (p["leak_feather"] * scale) * (
                    1.0 + var * 0.45 * (2.0 * st["halo"] - 1.0))
                hl = min(max(hl, 0.5), reach * 0.95)
                expo = math.log(0.5) / math.log1p(-min(hl / reach, 0.95))
                # Floored at *zero*, not at an epsilon: raising a 1e-4 floor
                # to a small exponent gives 0.12, not something small, and
                # that is a fog over the whole beam's footprint.
                along = (1.0 - (du / reach).clamp(0.0, 1.0)).clamp_min(0.0) ** expo

                # Across the beam: narrow at the source and fanning inward,
                # which is what a shaft through a gap does and is most of why
                # this reads as a beam rather than as a band.
                hwid = (hw0 * (0.75 + st["flare"] * du / reach)).clamp_min(1.0)
                q = dv.abs() / hwid
                # One edge is the obstruction's shadow and is much harder than
                # the other. Both soft is haze; both hard is a painted shape.
                bw_hard = max(0.03, bw_soft * (1.0 - 0.95 * st["hard"]))
                on_hard = (dv * st["hard_side"] >= 0.0).to(dv.dtype)
                bw = bw_soft + (bw_hard - bw_soft) * on_hard
                tt = ((1.0 + bw - q) / (2.0 * bw)).clamp(0.0, 1.0)
                across = tt * tt * (3.0 - 2.0 * tt)

                # A beam is not uniform inside itself either -- dust in the
                # chamber, an uneven gap. Mean 1.0, so it modulates without
                # changing the strength the leak was drawn with.
                dens = 0.72 + 0.56 * wn[:, 2:3]

                hue = min(max(p["leak_hue"] + st["hue"], 0.0), 1.0)
                tint = torch.tensor(
                    [1.0, 0.16 + 0.46 * hue, 0.04 + 0.18 * hue],
                    device=dev, dtype=torch.float32,
                ).view(1, 3, 1, 1)
                expos = expos + (along * across * dens * st["strength"]) * tint

            # Saturating response, per channel and per dye layer. A leak's core
            # is *white* with the colour only in its falloff, and no amount of
            # adding a fixed warm ratio can do that -- a fixed ratio stays the
            # same colour at every strength, which is exactly why the old wash
            # read as flat tan everywhere. Each layer saturating separately
            # gives the real progression: deep red where only the red-sensitive
            # layer caught enough light, through orange and yellow, to white
            # where all three are at the top. It also self-limits at 1.0 in
            # linear light, so a hot leak cannot drive a channel past white.
            added = -torch.expm1(-expos * (p["leak_strength"] * _LEAK_GAIN))
            out = _linear_to_srgb(expo_lin + added.to(out.dtype))

        # -- scratches ----------------------------------------------------
        # A gouge through the emulsion lets the light straight through, so on
        # a positive it prints bright. Drawn as noise whose cells are a couple
        # of pixels wide and hundreds tall: that anisotropy *is* the scratch.
        sc = p["scratches"]
        if sc >= 1.0 and area is not None:
            wpx = max(0.4, p["scratch_width"] * scale)
            n = _value_noise(
                h, w, y0, x0, wpx * 2.0, seed + 4409, 1, dev,
                cell_y=max(60.0, 900.0 * scale),
            )
            # A scratch occupies one cell of a very tall, very thin lattice,
            # so its "area" is that cell -- the count then works out the same
            # way as for dust despite the anisotropy.
            cell_x, cell_y = wpx * 2.0, max(60.0, 900.0 * scale)
            th_a = _count_threshold(sc * 2.5, cell_x * cell_y, area, _BLOB_CELLS_SCRATCH)
            th_b = _count_threshold(sc * 0.5, cell_x * cell_y, area, _BLOB_CELLS_SCRATCH)
            line = _smoothstep(th_a, max(th_b, th_a + 1e-4), n)
            # Break them along their length, or every scratch runs the full
            # height of the frame and reads as a printing artifact.
            brk = _value_noise(
                h, w, y0, x0, max(24.0, 300.0 * scale), seed + 4410, 1, dev,
                cell_y=max(8.0, 90.0 * scale),
            )
            line = line * _smoothstep(0.30, 0.72, brk)
            # Variation field shares the scratch's own anisotropy, so softness
            # and density are constant *along* a scratch and differ *between*
            # scratches -- the other way round would make one scratch fade in
            # and out down its length.
            vary = _value_noise(
                h, w, y0, x0, wpx * 6.0, seed + 4411, 2, dev,
                cell_y=max(90.0, 1300.0 * scale),
            )
            out = out + self._weather(
                line, vary, p["scratch_soften"],
                p["scratch_soften"] * 3.0 * max(wpx, 0.6),
            ) * 0.85

        # -- hair ---------------------------------------------------------
        # A hair on the scanner bed is opaque, so it prints as a dark
        # filament. Drawn as the level set of a smooth field: |n - 0.5| below
        # a small epsilon is a thin curve that wanders the way a hair does,
        # which is far more convincing than any curve I would parameterise.
        hr = p["hair"]
        if hr >= 1.0 and area is not None:
            # Length and count are separate controls, and keeping them separate
            # is the whole point here. Deriving the contour cell from the count
            # -- which is what I did first -- means raising the count shortens
            # every hair, so the slider reads as a *length* control instead of
            # a count. The contour field's cell now comes from Hair Length
            # alone, and the count is a threshold on a gate at that same scale.
            length = max(8.0, p["hair_length"] * scale)
            c_cell = length * 1.2
            n = _value_noise(h, w, y0, x0, c_cell, seed + 6607, 1, dev)
            # Width held at ~1.6 full-res px as the cell changes: a level set is
            # about 2 * eps * cell wide, so eps has to track the cell or hairs
            # fatten as they lengthen.
            eps = (1.6 * max(scale, 0.25)) / (2.0 * c_cell)
            fil = 1.0 - _smoothstep(0.0, eps, (n - 0.5).abs())

            # Gate blobs are one hair each, and they are sized by length, so
            # the threshold is free to control only how many there are.
            g_a = _count_threshold(hr * 2.5, length * length, area, _BLOB_CELLS_HAIR)
            g_b = _count_threshold(hr * 0.4, length * length, area, _BLOB_CELLS_HAIR)
            sparse = _value_noise(h, w, y0, x0, length, seed + 6608, 1, dev)
            fil = fil * _smoothstep(g_a, max(g_b, g_a + 1e-4), sparse)

            vary = _value_noise(
                h, w, y0, x0, max(30.0, 200.0 * scale), seed + 6609, 2, dev
            )
            out = out - self._weather(
                fil, vary, p["hair_soften"], p["hair_soften"] * 3.0 * max(scale, 0.25),
            ) * 0.9

        # -- dust ---------------------------------------------------------
        # Two populations: opaque specks that block light and print dark, and
        # the pinholes and lint that print bright. Both are wanted -- dust
        # that is only ever dark reads as sensor dirt, not film.
        du = p["dust"]
        if du >= 1.0 and area is not None:
            cell = max(_MIN_CELL, p["dust_size"] * scale)
            n = _value_noise(h, w, y0, x0, cell, seed + 5501, 2, dev)
            # Two thirds dark motes, one third bright pinholes -- both wanted,
            # dust that is only ever dark reads as sensor dirt.
            # Both ends of the ramp come from the count, a band rather than a
            # threshold plus a fudge. At low counts the threshold sits so deep
            # in the field's tail that `t + (1-t)*0.55` collapses below
            # _smoothstep's degeneracy guard and becomes a hard step -- and a
            # hard step there is knife-edge: a pixel within float epsilon of it
            # landed on opposite sides in a tiled versus a single-pass render.
            # Deriving both ends from counts guarantees a real ramp, and makes
            # specks fade in rather than switch on.
            # Softness widens the band *symmetrically about its midpoint*,
            # which gives each speck a gradual profile instead of a disc edge.
            # Blur alone cannot do this job: blurring a 2px speck by several
            # times its own size does not soften it, it erases it -- energy is
            # conserved so the peak collapses, and what you are left with is
            # fewer specks rather than softer ones. Symmetric expansion keeps
            # roughly the same count.
            spread_k = 1.0 + _DUST_SOFT_BAND * p["dust_soften"]

            def band(n_marks: float) -> tuple[float, float]:
                a_ = _count_threshold(n_marks * 3.0, cell * cell, area, _BLOB_CELLS_DUST)
                b_ = _count_threshold(n_marks * 0.4, cell * cell, area, _BLOB_CELLS_DUST)
                b_ = max(b_, a_ + 1e-4)
                mid, half = 0.5 * (a_ + b_), 0.5 * (b_ - a_) * spread_k
                return max(0.0, mid - half), min(1.0, mid + half)

            dark = _smoothstep(*band(du * 0.66), n[:, 0:1])
            lite = _smoothstep(*band(du * 0.34), n[:, 1:2])
            # Coarser than the dust itself so neighbouring specks differ,
            # rather than every speck being mottled within itself. Four
            # channels: softness and opacity for each population.
            vary = _value_noise(h, w, y0, x0, cell * 3.0, seed + 5502, 4, dev)
            # Capped relative to the speck: past about its own size the blur
            # is removing specks, not softening them.
            r = p["dust_soften"] * _DUST_SOFT_REACH * cell

            # Composited rather than added, which is what separates opacity
            # from luminosity. Additively they are the same number: a fainter
            # speck and a lighter speck are indistinguishable. As a composite,
            # opacity is how much of the photograph the speck hides and
            # luminosity is what colour the speck itself is, so a solid grey
            # mote and a faint black veil are different things.
            o_var, l_var = p["dust_opacity_var"], p["dust_lum_var"]
            base_op = p["dust_opacity"]

            def lay(mask, chans, lum_lo, lum_hi):
                nonlocal out
                v_op = _spread(chans[:, 0:1])
                v_lum = _spread(chans[:, 1:2])
                soft = self._weather(
                    mask, chans, p["dust_soften"], r, lum_floor=1.0
                )
                alpha = (soft * base_op * (1.0 - o_var * (1.0 - v_op))).clamp(0.0, 1.0)
                # Luminosity spreads about the population's own end of the
                # scale, so dark motes stay dark and pinholes stay bright.
                mid = 0.5 * (lum_lo + lum_hi)
                col = mid + (v_lum - 0.5) * (lum_hi - lum_lo) * l_var
                out = out * (1.0 - alpha) + col.clamp(0.0, 1.0) * alpha

            lay(dark, vary[:, 0:2], 0.0, 0.42)
            lay(lite, vary[:, 2:4], 0.72, 1.0)

        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _weather(
        mark: torch.Tensor, vary: torch.Tensor, soften: float, radius: float,
        lum_floor: float = _TEX_LUM_FLOOR,
    ) -> torch.Tensor:
        """Make a field of marks non-uniform in sharpness and in brightness.

        A thresholded noise field gives every mark the same crisp edge and the
        same opacity, which is the tell that they were generated: real debris
        is at different depths, so some of it is in focus and some is not, and
        none of it is equally dark.

        ``vary`` carries two decorrelated fields addressed at mark scale, so a
        whole scratch shares its blur and its density rather than varying
        pixel-to-pixel down its own length. The first drives how far each mark
        blends toward a blurred copy; the second scales its strength.

        Blurring also thins a mark, which is left uncorrected on purpose --
        out-of-focus debris really is both softer and fainter.
        """
        # Spread, not raw. Value noise clusters so tightly around its median
        # (p10-p90 spans 0.41-0.71) that a floor-to-1.0 mapping delivered only
        # a +/-16% spread however wide the range it was given -- which is why
        # the marks still looked uniform. Same fix as the light leaks needed.
        v_soft, v_lum = _spread(vary[:, 0:1]), _spread(vary[:, 1:2])
        if soften > 0.01 and radius > 0.05:
            blurred = _blur(mark, radius)
            # Centre the field so `soften` sets the *average* blur, with marks
            # either side of it, rather than a floor everything sits above.
            b = (soften * _smoothstep(0.15, 0.85, v_soft)).clamp(0.0, 1.0)
            mark = mark * (1.0 - b) + blurred * b
        # Never all the way to zero: a mark that fades out entirely just thins
        # the population rather than varying it.
        if lum_floor >= 1.0:
            return mark
        return mark * (lum_floor + (1.0 - lum_floor) * v_lum)

    # ------------------------------------------------------------------ #
    def render_supersampled(
        self, img: torch.Tensor, p: dict, scale: float, y0: float, x0: float,
        ss: int, full_hw: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Render a tile at ``ss``x linear resolution and area-average back down.

        Grain is a sub-pixel phenomenon: rendering it at the output grid gives
        each clump a hard, aliased pixel footprint, which is exactly the
        synthetic look the project exists to avoid. Rendering above Nyquist and
        integrating down gives clumps genuine partial pixel coverage. Costs
        ss^2 in time and memory, and it is the single biggest realism win in
        the pipeline.
        """
        if ss <= 1:
            return self.render(img, p, scale, y0, x0, full_hw)
        h, w = img.shape[-2:]
        up = F.interpolate(
            img, size=(h * ss, w * ss), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        # Working resolution and tile offset both scale, so the noise lattice
        # still resolves to the same global full-resolution coordinates.
        # Frame size scales with the working resolution exactly as the tile
        # offset does, so a normalised frame position resolves the same.
        fh = None if full_hw is None else (full_hw[0] * ss, full_hw[1] * ss)
        r = self.render(up, p, scale * ss, y0 * ss, x0 * ss, fh)
        return F.avg_pool2d(r, ss)

    # ------------------------------------------------------------------ #
    def pad_for(self, p: dict, scale: float) -> int:
        """Overlap needed so a rendered region matches the full-image render.

        Must cover every blur kernel in the pipeline: the high-pass chain, the
        acutance blur (the widest at 1.5x), the micro-blur, the edge-softening
        blur, the output sharpening blur and halation, plus the displacement
        of every stage that *reads* a pixel from somewhere else rather than
        blurring in place -- the jitter warp, the sanding taps and scatter.
        Miss one and tiled exports seam along its radius -- which no preview
        will ever show.
        """
        hp_r = max(0.3, p["highpass_radius"] * scale)
        mb = p["micro_blur"] * scale
        halo = p["halation_radius"] * scale if p["halation"] > 0.01 else 0.0
        soft = p["edge_soften_radius"] * scale if p["edge_soften"] > 0.01 else 0.0
        shr = p["sharpen_radius"] * scale if p["sharpen"] > 0.01 else 0.0
        if p["pre_sharpen"] > 0.01:
            shr = max(shr, p["pre_sharpen_radius"] * scale)
        # Film-texture softening blurs the mark fields, so it reaches like any
        # other kernel. Widest of the three wins -- they are separate stages,
        # not compounded.
        tex_r = 0.0
        if p["dust"] >= 1.0:
            tex_r = max(tex_r, p["dust_soften"] * 1.6
                        * max(_MIN_CELL, p["dust_size"] * scale))
        if p["scratches"] >= 1.0:
            tex_r = max(tex_r, p["scratch_soften"] * 3.0
                        * max(0.4 * p["scratch_width"] * scale, 0.6))
        if p["hair"] >= 1.0:
            tex_r = max(tex_r, p["hair_soften"] * 3.0 * max(scale, 0.25))
        mask_r = max(1.0, 3.0 * scale)
        # Scatter reads a pixel up to its full reach away. It displaces rather
        # than blurring, so it belongs with the warps below and not in the
        # kernel sum.
        #
        # Reach *plus one pixel*: dx and dy are rounded to whole pixels
        # independently, so two half-pixel roundings the same way lengthen the
        # vector by up to sqrt(2)/2. It would fit inside the +4 at the end of
        # this function either way, but a stage that silently depends on
        # another term's slack is a seam waiting for somebody to tighten it.
        sca = (
            max(0.5, p["scatter_radius"] * scale) + 1.0
            if p["scatter"] > 0.001 else 0.0
        )
        # Jitter warps the image rather than blurring it, so it reads pixels
        # displaced by up to its peak -- which at _JITTER_MAX is no longer the
        # sub-pixel rounding error it was at 0.6.
        # Both the jitter warp and the sanding filter read displaced pixels
        # rather than blurring in place, so the overlap has to cover how far
        # each of them travels.
        jit = _JITTER_MAX * p["edge_jitter"] * max(scale, 0.25) + sca
        if p["edge_sand"] > 0.01:
            # Sanding compounds in two ways at once, and both have to be
            # counted or a tiled export seams while every preview looks fine.
            # Each of its (up to three) passes resamples the previous pass's
            # output, so tap travel accumulates to 2 x total rather than
            # total; and each pass re-derives its direction from a blurred
            # luma, so that blur's reach accumulates too. Counting only the
            # first was enough at the old 4px grit ceiling and seams from 8px
            # up. Passes is pinned at its maximum here rather than recomputed,
            # because pad_for is called at the un-supersampled scale and would
            # otherwise disagree with the renderer about the count.
            total = max(0.5, p["edge_sand_grit"] * scale)
            sr = total / _SAND_PASSES
            dir_reach = 3.0 * max(0.6, _SAND_DIR_K * sr)
            jit += _SAND_PASSES * (2.0 * sr + dir_reach)
        return int(
            math.ceil(
                3.0 * (hp_r * 3.3 + mb + halo + soft + shr + tex_r + mask_r)
                + jit
            )
        ) + 4

    def render_view(
        self, arr: np.ndarray, p: dict, box: tuple[int, int, int, int],
        zoom: float = 1.0, supersample: int = 2,
    ) -> np.ndarray:
        """Render ``box`` = (y, x, h, w) of ``arr`` at a display ``zoom``.

        Reads a padded window so every filter sees its true neighbourhood, then
        trims. This is what makes the inspection view trustworthy: what you see
        is exactly what the export will contain for that region.

        Zoom above 1.0 renders at 1:1 and leaves magnification to the client --
        upsampling before rendering would invent grain that is not in the
        export. Zoom below 1.0 renders at that working scale, which is the
        honest thing to show: at 50% the export's grain really is half-resolved.
        """
        y, x, bh, bw = box
        H, W, _ = arr.shape
        scale = min(float(zoom), 1.0)

        # Padding is needed in source pixels, but pad_for is in working pixels.
        pad = int(math.ceil(self.pad_for(p, scale) / max(scale, 1e-3)))
        ya, yb = max(0, y - pad), min(H, y + bh + pad)
        xa, xb = max(0, x - pad), min(W, x + bw + pad)

        if scale < 0.999:
            # Snap the read origin so that origin*scale is a whole number of
            # working pixels. Downsampling samples at pixel centres, so a crop
            # whose origin lands mid-pixel resolves on a different grid phase
            # than a whole-image downscale would -- a half-pixel shift that is
            # invisible on smooth areas and obvious on hard edges.
            step = next(
                (k for k in range(1, 9) if abs(k * scale - round(k * scale)) < 1e-6),
                1,
            )
            ya = (ya // step) * step
            xa = (xa // step) * step

        chunk = np.ascontiguousarray(arr[ya:yb, xa:xb, :])
        t = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if scale < 0.999:
            ch, cw = t.shape[-2:]
            t = F.interpolate(
                t, size=(max(1, round(ch * scale)), max(1, round(cw * scale))),
                mode="bicubic", antialias=True, align_corners=False,
            ).clamp(0.0, 1.0)

        # Frame size is the whole source at this scale, not the read window --
        # a crop must place the light leak where it falls in the *frame*, or
        # zooming in would drag the leak around with the viewport.
        fh, fw = arr.shape[0] * scale, arr.shape[1] * scale
        r = self.render_supersampled(
            t, p, scale, ya * scale, xa * scale, max(1, int(supersample)),
            (float(fh), float(fw)),
        )
        r = r.squeeze(0).permute(1, 2, 0).cpu().numpy()

        oy, ox = round((y - ya) * scale), round((x - xa) * scale)
        oh, ow = max(1, round(bh * scale)), max(1, round(bw * scale))
        return r[oy: oy + oh, ox: ox + ow, :]

    def render_crop(
        self, arr: np.ndarray, p: dict, box: tuple[int, int, int, int],
        scale: float = 1.0, supersample: int = 2,
    ) -> np.ndarray:
        """1:1 render of ``box``, bit-identical to the same region of a full
        render. Thin wrapper kept for the invariant checks."""
        return self.render_view(arr, p, box, scale, supersample)

    # ------------------------------------------------------------------ #
    def render_image(
        self, arr: np.ndarray, p: dict, scale: float = 1.0,
        tile: int = 1024, supersample: int = 2, progress=None,
    ) -> np.ndarray:
        """Render a whole image, tiling when it is larger than ``tile``.

        ``arr`` is HxWx3 float32 in 0..1. Returns the same shape.
        """
        # Nothing switched on: hand the input straight back. Not merely an
        # optimisation -- see params.is_neutral for why rendering it would
        # *not* return the input.
        if P.is_neutral(p):
            return arr
        ss = max(1, int(supersample))
        h, w, _ = arr.shape
        if max(h, w) <= tile:
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self.render_supersampled(
                t, p, scale, 0.0, 0.0, ss, (float(h), float(w))
            )
            if progress:
                progress(1.0)
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Overlap must cover every blur kernel in the pipeline plus the warp.
        pad = self.pad_for(p, scale)

        out = np.empty_like(arr)
        ny = math.ceil(h / tile)
        nx = math.ceil(w / tile)
        done = 0
        for ty in range(ny):
            for tx in range(nx):
                y_a, y_b = ty * tile, min((ty + 1) * tile, h)
                x_a, x_b = tx * tile, min((tx + 1) * tile, w)
                # padded read window, clamped to the image
                py_a, py_b = max(0, y_a - pad), min(h, y_b + pad)
                px_a, px_b = max(0, x_a - pad), min(w, x_b + pad)

                chunk = arr[py_a:py_b, px_a:px_b, :]
                t = torch.from_numpy(np.ascontiguousarray(chunk))
                t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
                r = self.render_supersampled(
                    t, p, scale, float(py_a), float(px_a), ss, (float(h), float(w))
                )
                r = r.squeeze(0).permute(1, 2, 0).cpu().numpy()

                out[y_a:y_b, x_a:x_b, :] = r[
                    y_a - py_a: y_a - py_a + (y_b - y_a),
                    x_a - px_a: x_a - px_a + (x_b - x_a),
                    :,
                ]
                done += 1
                if progress:
                    progress(done / float(ny * nx))
        return out
