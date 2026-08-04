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

import collections
import math
import os

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

# Byte cap on the Global Grain texture cache (see `_global_grain_field`). Sized
# for a handful of tiles at preview resolution -- one entry is [1,3,h,w] at
# *working* resolution, so 113MB per tile at tile 1536 / supersample 2, or ~38MB
# for the monochrome case. Capped by bytes rather than by entry count for
# exactly that reason: an entry-count cap sized for the chroma case would hold
# three times too much memory when chroma is off, and vice versa.
#
# Shared budget, not an independent one: this competes with the working set that
# `tile_for` sizes tiles against, so raising one means lowering the other.
_GG_CACHE_BYTES = int(
    float(os.environ.get("FILM_GRAIN_GRAIN_CACHE_GB", "0.5")) * (1 << 30)
)

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

# Global-grain smoothing. Value noise is a quilt -- its extrema sit on an
# axis-aligned lattice and each cell is a separable patch -- so past roughly
# 8px the cells read as rectangles. Measured on a cell-20 field, |gradient|
# binned by phase within a cell swings by 1.74x its own mean, which is that
# grid showing up as a number.
#
# _SMOOTH_MAX  peak blur sigma as a fraction of the clump, at Smoothness 1.
#              Half a cell takes that 1.74 to 0.27 on the bare field and
#              1.72 to 0.32 through a full render (the figure `verify.py`
#              pins) -- the quilt is gone rather than merely softened -- and
#              still leaves clump-scale structure behind.
# _SMOOTH_GAIN_K  restores the amplitude the blur costs. Measured, the loss
#              depends on sigma/cell *alone*: cells of 8, 16 and 32px attenuate
#              within 0.5% of each other at every ratio tested, so one closed
#              form covers every size. sqrt(1 + k(sigma/cell)^2) fits it to
#              0.6% over the whole 0..0.5 range this stage can reach. Analytic
#              on purpose -- reading the tile's own std would restore a
#              different amount per tile and seam the export.
#
#              **Fit it against the field it is actually used on.** Calibrated
#              first on single-octave value noise it came out 7.7, but the
#              global layer is a two-octave fBm whose coarse half survives a
#              blur far better, so 7.7 over-restored and made full Smoothness
#              10% louder than no smoothing -- exactly the amplitude coupling
#              the gain exists to prevent.
_SMOOTH_MAX = 0.5
_SMOOTH_GAIN_K = 5.62

# --- colour grading (step -1, above everything) ---------------------------- #
#
# Peak channel gain for the temperature shift, at grade_temp = +/-1. Red and
# blue move by this much in opposite directions and green is left alone; the
# whole vector is then normalised against the luma weights, so the control
# changes the light's colour and not its level. 0.40 puts the extremes at a
# roughly 3200K/8000K feel without any setting driving a channel to a rail.
_GRADE_TEMP_GAIN = 0.40
#
# Peak channel gain for Tint, the other white-balance axis: green against
# magenta. Deliberately *smaller* than Temperature's -- green carries most of
# the luma weight (0.7152 against red's 0.2126 and blue's 0.0722), so the same
# magnitude on this axis costs more level. Measured on an asymmetric plate,
# 0.40 drifts luma up to 2.3%, against Temperature's own 1.8% at its 0.40; 0.30
# brings the worst case to 1.4%, in the same envelope as Temperature.
_GRADE_TINT_GAIN = 0.30
#
# Where the shadow and highlight ramps meet. Each control acts on one side of
# it only -- so every pixel is in exactly one of them, the two are independent
# by construction rather than by bookkeeping, and both curves leave the knee at
# slope exactly 1, which is what makes the join invisible on a gradient without
# a ramp to fade them in. Not exposed: a knee plus a falloff per end would be
# four more sliders for a section that is meant to stay cheap and quick.
_GRADE_TONE_KNEE = 0.5
#
# How far a tone lift can travel at +/-1, as a share of the headroom it has.
#
# Not 1.0, and the difference matters more than it sounds. The lift is a share
# of the distance to the rail, so at 1.0 a setting of +1 takes a black pixel to
# *pure white* -- which means the whole top of the slider is unusable and the
# useful range is squeezed into its first tenth. Measured on a real photograph
# (mean luma 0.21), Shadows at only +0.5 took the frame's mean from 0.19 to
# 0.53; that is not a shadow lift, it is a different exposure. At 0.35 the same
# +1 moves a black pixel to 0.35 -- a thoroughly lifted, faded-print look, which
# is a fair thing to find at the end of the travel -- and +0.3 gives the natural
# one-stop-ish lift you actually reach for. Same lesson as `_JITTER_MAX`, from
# the other direction: a control whose whole range has to be usable.
#
# Applies to the *expanding* half of each control only -- Shadows negative and
# Highlights positive, which push a tone toward the rail it is already nearest.
# The recovering halves (Shadows positive, Highlights negative) do not use it:
# they are asymptotic rolls whose endpoint is fixed by the curve's own shape,
# and a share-of-headroom cap is exactly what made them non-monotonic. See
# `_tone_roll`.
_GRADE_TONE_MAX = 0.35
#
# Where a channel starts counting as clipped, for highlight reconstruction: the
# soft window over which "this is a real measurement" becomes "this hit the
# ceiling and the true value is somewhere above it".
#
# Soft rather than a single threshold because the window doubles as the blend
# weight, and a hard switch would draw a visible contour around every blown
# region. The top is just short of 1.0 rather than at it because an 8-bit
# ceiling is 255/255 and JPEG ringing puts real clipped pixels a code value or
# two either side of it.
_RECON_LO = 0.94
_RECON_HI = 0.999
#
# How much local evidence a channel needs before its reconstruction is trusted,
# as a share of the neighbourhood that still holds a valid measurement of that
# channel. Below this the estimate fades out rather than being divided into
# existence: in the middle of a region blown wide in every channel there is
# genuinely nothing to recover from, and saying so is better than inventing it.
_RECON_MIN_EVIDENCE = 0.02
#
# Ceiling on a reconstructed value, in display-referred units -- two stops above
# white. The estimate divides by the local chromaticity of whichever channels
# survived, so a highlight lit by something the surviving channel barely sees
# (deep blue in tungsten light) has a small denominator and would otherwise run
# away. Two stops is more headroom than any 8-bit source can justify and still
# bounded.
_RECON_CEIL = 4.0
#
# Knee for reconstruction's own roll -- the curve that brings what it recovered
# back inside the visible range, without which the whole stage is invisible.
#
# High, at 0.80, and that is the point: the recovered data lands just above
# white, so only the top fifth of the range has to give way to make room for it.
# A knee at `_GRADE_TONE_KNEE` (0.5) would work too and would be the wrong
# control -- that is a broad highlight roll, i.e. what `grade_highlights` is
# for, and duplicating it here would mean reconstruction quietly graded the
# picture as well as repairing it.
_RECON_ROLL_KNEE = 0.80
#
# How wide to smooth the roll's gate, as a fraction of the reconstruction
# radius. The gate is the reconstruction's own weight field, so the roll engages
# only where something was actually repaired -- but a per-pixel gate would draw
# an edge around every repaired region, so it is dilated and feathered.
#
# 0.25 rather than the 0.5 first tried, and it is better on every axis measured,
# which is worth recording because the larger value looks like the safer one. The
# gate only has to be wide enough not to contour: swept against the second
# derivative of a repaired ramp, 0.05 starts introducing curvature the source
# does not have (3.2e-03 against the source's own 2.2e-03) while 0.10, 0.25 and
# 0.50 all sit at 1.4e-03, so 0.25 clears it with a 5x margin. Going wider then
# only costs -- the recovered span holds flatter across the radius at 0.25
# (0.0736 / 0.0736 / 0.0735 / 0.0657 against 0.5's 0.0736 / 0.0735 / 0.0650 /
# 0.0654 at 8/16/32/64px) and `pad_for` at the top of the radius range is 933px
# against 1233px.
_RECON_ROLL_GATE_FRAC = 0.25
# Gain on the positive side of Clarity. The negative side is pinned at exactly
# 1.0 and cannot be raised: at gain 1 a setting of -1 removes precisely 100% of
# the local-contrast band, and anything past that does not flatten further, it
# *inverts* -- dark halos on the light side of every edge. Positive has no such
# limit, so it gets the headroom to be worth reaching for.
_GRADE_CLARITY_GAIN = 1.6
#
# Gain on Contrast's pivot-about-middle-grey, at +/-1. Floored so the gain can
# never reach 0 or go negative: at -1 it is 0.1, a tenth of the original
# spread rather than a flattened or inverted picture. Deliberately smaller than
# the film characteristic curve's own 1.1 -- this control has no shoulder to
# catch what it steepens, so a gentler reach keeps the top of the slider from
# clipping immediately.
_GRADE_CONTRAST_GAIN = 0.9

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

# Anti-aliasing: a three-tap 1-2-1 along the isophote. Short on purpose -- a
# stair-step is a *pixel-scale* wobble along the contour, so reaching further
# only starts averaging away the shape the contour has. That is the whole
# difference in scale from `edge_sand`, whose taps run to +/-2 sigma because
# the roughness it removes sits at much longer wavelengths.
_AA_TAPS = ((-1.0, 0.25), (0.0, 0.5), (1.0, 0.25))

# Maximum anti-aliasing passes, and therefore the top of `aa_strength`. One
# pass of a three-tap filter is a gentle thing -- measured, 35% of a stair-step
# -- and the way to make it bite is to run it again rather than to lengthen it:
# the taps are short *on purpose*, so a longer reach averages away the shape the
# contour has instead of the wobble on it. Each pass re-estimates the tangent
# from the image it is given, which re-aims along a curving edge where one wide
# pass cuts the corner. Same reasoning and same shape as `_SAND_PASSES`, and
# like that one `pad_for` assumes this count exactly, so the two must not drift.
_AA_PASSES = 3

# Direction-estimate blur for the AA tangent, as a fraction of its radius, and
# a floor. Smaller than `_SAND_DIR_K` against a smaller radius: this filter has
# to follow a contour at the pixel scale, and estimating its direction over a
# wide window would cut the corners off small features. The floor is what keeps
# the tangent from swinging on single-pixel noise, which is the same stability
# problem `_SAND_DIR_K` exists for.
_AA_DIR_K = 0.5
_AA_DIR_MIN = 0.7

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


class RenderCancelled(Exception):
    """Raised out of `render_image` when its `should_cancel` hook says stop.

    Its own type rather than a bool return so a cancelled render cannot be
    mistaken for a finished one by a caller that forgot to check.
    """


# Peak render memory per *working* pixel -- i.e. per pixel of the padded tile
# after supersampling. Measured, not guessed: single-tile `Stock` renders at 512,
# 768, 1024 and 1280 square with supersample 2 gave marginal costs of 763, 530
# and 424 bytes per working pixel as the frame grew (the allocator reserves in
# ~1GB steps, so the small end reads high). 512 is above the worst marginal
# figure, which is the safe direction to be wrong in: over-estimating picks a
# smaller tile and renders slower, under-estimating runs the machine out of
# memory. `defaults` measures roughly half this, so the constant is sized for
# the heaviest preset rather than the typical one.
_WORKING_BYTES_PER_PX = 512

# Fraction of the backend's recommended working set to actually use. The rest is
# headroom for the client, the encoder and whatever else shares the GPU -- the
# render is not the only thing on the machine, and on Apple silicon this is
# system RAM, so overcommitting means swapping rather than a clean failure.
_RENDER_BUDGET_FRACTION = 0.5

# Tile floor and ceiling. The floor is not a memory figure: below it `pad_for`
# overlap dominates the useful area so completely that the extra work costs more
# than the memory it saves, and every supported backend can hold a tile this
# size. The ceiling keeps a single enormous tile from defeating the point of
# tiling on a machine that reports a very large budget.
_TILE_MIN = 768
_TILE_MAX = 8192


def _render_budget_bytes() -> int:
    """Working-set budget for one tile, in bytes.

    `FILM_GRAIN_TILE_BUDGET_GB` overrides it outright, in the same spirit as
    `FILM_GRAIN_DEFAULT_PRESET` -- useful both for forcing large tiles on a big
    machine and for reproducing a small machine's tiling on a large one, which is
    what makes the tile-independence checks in `verify.py` testable here.
    """
    env = os.environ.get("FILM_GRAIN_TILE_BUDGET_GB")
    if env:
        try:
            return int(float(env) * (1 << 30))
        except ValueError:
            pass
    total = 0
    if torch.backends.mps.is_available():
        try:
            total = int(torch.mps.recommended_max_memory())
        except Exception:
            total = 0
    elif torch.cuda.is_available():
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            total = 0
    if total <= 0:
        # CPU, or a backend that will not say. Derive from system RAM, which is
        # the real constraint there too.
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError, AttributeError):
            total = 4 << 30
    return max(1 << 30, int(total * _RENDER_BUDGET_FRACTION))


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


def _isophote(
    lum: torch.Tensor, dir_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit tangent along the contour, plus the gradient magnitude.

    Shared by edge sanding and anti-aliasing, which want the same vector for
    opposite-scale reasons -- one polishes roughness off a contour, the other
    takes stair-steps off one -- and differ only in how far they then reach
    along it and what they gate the result on.

    ``dir_sigma`` is not optional and must not be zero. Taken per-pixel the
    gradient follows whatever noise is present and the tangent sands in
    circles; worse, where the gradient is weak the direction is a ratio of two
    near-zero numbers, so it swings on floating-point alone and a filter
    reaching along it samples somewhere else entirely. That is not just noisy,
    it made tiled exports seam: two tilings hand the gradient marginally
    different values. Callers gate on the returned magnitude for the same
    reason.
    """
    gl = _blur(lum, dir_sigma)
    px_ = F.pad(gl, (1, 1, 0, 0), mode="replicate")
    gx_ = (px_[..., 2:] - px_[..., :-2]) * 0.5
    py_ = F.pad(gl, (0, 0, 1, 1), mode="replicate")
    gy_ = (py_[..., 2:, :] - py_[..., :-2, :]) * 0.5
    mag = (gx_ * gx_ + gy_ * gy_).sqrt().clamp_min(1e-6)
    # The tangent is the gradient turned 90 degrees.
    return -gy_ / mag, gx_ / mag, mag


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _apply_lut(x: torch.Tensor, lut) -> torch.Tensor:
    """Trilinear 3D LUT lookup. ``x`` is [1,3,h,w] display-referred in 0..1.

    ``lut`` is a ``server.lut.Lut``; duck-typed rather than imported so the
    engine keeps no dependency on the file loader. It supplies the table as a
    ``[1, 3, D, H, W]`` volume and its input domain.

    One ``grid_sample`` call, which is trilinear in 3D and runs on the GPU -- so
    a 35-cube and a 65-cube cost the same and neither shows up against the
    stages below. The alternative, gathering eight corners by flat index and
    interpolating by hand, needs int64 index tensors that MPS handles badly and
    eight full-frame gathers of working memory.

    Two things are load-bearing:

    * **``align_corners=True``.** A LUT's first and last samples *are* input 0
      and input 1, not the centres of edge cells. With the default the whole
      table would be read at half a cell's offset -- a small, uniform, entirely
      wrong shift that would look like the LUT being slightly off rather than
      like a bug.
    * **The grid's last dimension is ``(x, y, z)`` mapping to ``(W, H, D)``,**
      and the table is stored ``[c][b][g][r]`` so that maps to ``(r, g, b)``.
      That is why the grid is simply the image's own channels in order. Get it
      backwards and any symmetric LUT still looks fine while every real one is
      channel-swapped, which is what ``verify.py`` uses an asymmetric table to
      pin.

    ``padding_mode="border"`` clamps rather than reflecting, so a value that has
    somehow left 0..1 reads the nearest real entry instead of folding back into
    the middle of the cube.
    """
    tab = lut.tensor(x.device)
    n = x
    # Almost every LUT in the wild declares the 0..1 domain, so the rescale is
    # skipped rather than paid for -- a per-channel multiply-add over the frame
    # for nothing.
    if lut.dmin != (0.0, 0.0, 0.0) or lut.dmax != (1.0, 1.0, 1.0):
        lo = torch.tensor(lut.dmin, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        hi = torch.tensor(lut.dmax, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        n = (n - lo) / (hi - lo).clamp_min(1e-6)
    grid = n.permute(0, 2, 3, 1).unsqueeze(1) * 2.0 - 1.0  # [1,1,h,w,3] as (r,g,b)
    out = F.grid_sample(
        tab, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return out.squeeze(2)


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


def _shoulder(t: torch.Tensor) -> torch.Tensor:
    """``1 - exp(-t)``: the roll every recovery in this file blends toward.

    Slope 1 at the knee, so it joins the identity without a seam; asymptotes at
    1, so an unbounded input lands inside a bounded output; strictly increasing
    everywhere, so ordering -- and therefore detail -- is never lost. Shared by
    `_tone_roll` and by highlight reconstruction's own local roll so the two
    cannot drift apart.
    """
    return 1.0 - torch.exp(-t)


def _tone_roll(t: torch.Tensor, amount: float) -> torch.Tensor:
    """The one monotone curve behind both tone-recovery directions.

    ``t`` is distance from the knee toward a rail, in units of the distance to
    that rail: ``t = 0`` is the knee, ``t = 1`` is the rail itself, and
    ``t > 1`` is a value that has already left the cube -- which is the whole
    reason this exists. Returns the rolled distance, to be mapped back the same
    way it came in.

    ``amount > 0`` recovers: a convex blend of the identity and the exponential
    shoulder ``1 - exp(-t)``, so at 1.0 the rail becomes an *asymptote*. Two
    properties fall out of that and both are the point:

    * **Strictly monotone at every setting.** The slope is
      ``1 - amount * (1 - exp(-t))``, which for ``amount <= 1`` is bounded below
      by ``exp(-t) > 0``. Ordering is never lost, so neither is detail: two
      tones that differ before the curve still differ after it.
    * **Unbounded input, bounded output.** Anything from the knee to infinity
      lands inside the cube, monotonically. That is what makes over-range data
      -- from reconstruction, from exposure, from a bright source -- *visible*
      rather than clipped flat, and it is the difference between recovering a
      highlight and merely dimming it.

    ``amount < 0`` expands instead, and keeps the old share-of-headroom form:
    ``t + |amount| * _GRADE_TONE_MAX * quintic(t) * (1 - t)``. Also monotone
    (measured slope stays above ``1 - _GRADE_TONE_MAX``), and it cannot drive an
    in-gamut value out of the cube, which is the guarantee that half of each
    control has always made.

    The asymmetry is deliberate and is the same shape of decision Clarity's is:
    pushing a tone toward a rail and pulling one back off it are different
    operations, and one formula that did both would do neither well. What made
    the previous single formula fail was precisely that its strength was a
    function of the pixel's own level -- ``x + a * m(x) * (1 - x)`` with ``m``
    rising steeply through the band it was gating -- so in the recovering
    direction the ``m'`` term overwhelmed the ``1`` and the transfer *inverted*:
    measured slope **-0.21 over 16% of the range at 1.0**, which does not
    compress highlight detail, it destroys and flips it. Hence a curve whose
    monotonicity is a property of its own algebra rather than of how far the
    slider happens to be pushed.
    """
    if amount >= 0.0:
        # (1 - a) * t + a * shoulder(t), written so a = 0 returns t exactly.
        return t + amount * (_shoulder(t) - t)
    ramp = _smootherstep(0.0, 1.0, t)
    return t + (-amount) * _GRADE_TONE_MAX * ramp * (1.0 - t)


def _recon_estimate(
    img: torch.Tensor, amount: float, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The estimate half of highlight reconstruction: what was the clipped value?

    Returns ``(out, w)`` -- the image with clipped channels raised to their
    estimated true values, which are **above 1.0** by design, and the per-channel
    weight that raising was applied with. `_reconstruct_highlights` then rolls
    that back into the visible range; the two are split so the estimate's
    accuracy can be checked as an equality against a known unclipped scene
    without the roll's compression in the way, which is exactly what
    ``verify.py`` does.

    An 8-bit file clips per channel, not per pixel, and that asymmetry is the
    opening this works through: a warm highlight reaches the ceiling in red
    long before green and well before blue, so across a blown cloud the red
    channel is a flat plateau while green and blue are still recording the
    scene's own gradient. The detail is *in the file*; it is only missing from
    one channel at a time. Where every channel is at the ceiling there is
    genuinely nothing left, and this says so rather than inventing it.

    Per channel, in display-referred space:

    * ``clipped`` is the soft indicator over ``_RECON_LO.._RECON_HI``, and
      ``valid = 1 - clipped`` marks what is still a real measurement.
    * ``q`` is each channel's local level, blurred over ``radius`` but averaged
      **only over whole clean pixels** -- ones with nothing clipped in any
      channel -- so a plateau of ceiling values contributes nothing to the
      estimate of what its own colour should be, and every channel's mean is a
      mean over the *same* pixels. That second half is load-bearing; see the
      comment in the body for the 6% ratio error that using each channel's own
      mask produced, and for why that error made the whole stage a no-op.
      Normalising ``q`` by its own luma-weighted mean across channels gives
      ``k``, the local chromaticity: the colour of the light around here,
      measured where it could be measured.
    * ``guide`` is this pixel's own brightness read off whichever channels are
      still valid, divided back through ``k`` so a surviving channel that the
      local light happens to be poor in does not read as a darker pixel. With
      one channel valid it reduces exactly to ``x_c / k_c``.
    * ``recon = k * guide`` is then the pixel's brightness wearing the local
      chromaticity -- and it exceeds 1.0 exactly as far as the clipped channel
      really did.

    Only ever *raises* a channel, and only ever a clipped one, weighted by how
    clipped it is and by whether there was evidence to work from -- so an
    unblown photograph is untouched and this half cannot darken anything.

    Tile-independent for the ordinary reason: two fixed-radius blurs and
    per-pixel arithmetic, no statistic of the region anywhere.
    """
    lw = torch.tensor(_LUMA, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    clipped = _smoothstep(_RECON_LO, _RECON_HI, img)
    valid = 1.0 - clipped

    # Local chromaticity, averaged over **whole clean pixels** -- ones with
    # nothing clipped in any channel -- rather than over each channel's own
    # valid mask.
    #
    # That distinction is the difference between this working and this quietly
    # not working, and it took a measurement to see. Averaging each channel over
    # wherever *it* was valid compares means taken over different sets of
    # pixels: red's mask stops at the clip boundary while green's runs on into
    # the brighter region past it, so red's mean is drawn from a darker
    # neighbourhood than green's and the ratio between them comes out
    # compressed. Measured on a warm ramp clipping in red only, the estimate
    # landed at k_R = 1.136 against a true 1.205 -- a 6% underestimate, which is
    # enough to put the reconstruction *below* the ceiling it was recovering
    # from, so the ``clamp_min(0)`` swallowed it and the stage did precisely
    # nothing. One shared mask makes every channel's mean a mean over the same
    # pixels, and the ratio exact.
    #
    # Note the two masks answer different questions and both are needed:
    # `clean` is "is this *pixel* a trustworthy sample of the local colour",
    # `valid` below is "is this *channel* of this pixel a trustworthy reading of
    # its brightness".
    clean = valid.amin(dim=1, keepdim=True)
    den = _blur(clean, radius)
    q = _blur(img * clean, radius) / den.clamp_min(1e-3)
    ev = _smoothstep(0.0, _RECON_MIN_EVIDENCE, den)

    # Normalised so a neutral neighbourhood gives k = 1 in every channel.
    # `_LUMA` sums to 1, so this is a mean and not merely a sum.
    qg = (q * lw).sum(dim=1, keepdim=True)
    k = q / qg.clamp_min(1e-4)

    # This pixel's brightness from its surviving channels, read through the same
    # chromaticity so the two are on one scale. With a single channel valid it
    # reduces to x_c / k_c.
    wv = lw * valid
    guide = (wv * img).sum(dim=1, keepdim=True) / (wv * k).sum(
        dim=1, keepdim=True).clamp_min(1e-4)
    recon = (k * guide).clamp(0.0, _RECON_CEIL)

    # Two more things have to be true before a value is trusted. There has to
    # have been a clean pixel within reach to read the colour from (`ev`), and
    # this pixel has to have at least one channel of its own left to read its
    # brightness off -- one white in all three has nothing to be recovered from
    # and must come through untouched rather than through a division by an
    # epsilon.
    w = amount * clipped * ev
    return img + w * (recon - img).clamp_min(0.0), w


def _reconstruct_highlights(
    img: torch.Tensor, amount: float, radius: float,
) -> torch.Tensor:
    """`_recon_estimate`, then the roll that makes its result *visible*.

    Kept as one stage behind one slider because either half alone is useless: the
    estimate without the roll is invisible, and the roll without the estimate is
    just a highlight dimmer. ``pad_for`` carries all three of the kernels
    involved, in series.
    """
    out, w = _recon_estimate(img, amount, radius)

    # The roll that makes any of that *visible*, which the first version of this
    # stage left out and which made the whole control read as dead.
    #
    # Reconstruction's output is above 1.0 -- that is the entire point, it is
    # where the clipped channel really was -- and the section's final clamp then
    # took it straight back off, so the slider moved 0.0004 of mean level on a
    # real photograph. Reported as "I don't see any effect from those sliders at
    # all", and correct. Pairing it with Highlights did work (0.395 of max change
    # against 0.05) but a control that needs a second, differently-named control
    # to do anything is broken however clearly the help text says so.
    #
    # There is a hard constraint here worth stating, because it rules out the
    # tidier designs: **any curve that brings over-range data into view must move
    # in-gamut highlights too.** A gamut map with its knee exactly at 1.0 has to
    # jump -- it would send v = 1 to 1 - d -- so a smooth one needs its knee
    # below 1, and everything above that knee moves. "Visible on its own" and
    # "bit-exact no-op" are therefore in genuine conflict for a *global* curve.
    #
    # The way out is to make it **local**: gate the roll on reconstruction's own
    # weight field, blurred. Then the conflict dissolves rather than being
    # traded off --
    #
    # * where nothing was repaired the gate is 0, the roll is the identity, and
    #   an unblown photograph comes through **bit-exactly** untouched, which is
    #   the property that keeps this a repair tool and not a second highlight
    #   grade;
    # * where something was repaired the roll engages and the recovered detail
    #   appears;
    # * and the gate is smooth, so there is no contour at the boundary -- a
    #   per-pixel gate would outline every repaired region.
    #
    # Same `_shoulder` as `_tone_roll`, but applied **per channel** -- the
    # opposite of the tone stage's channel-max-and-uniform-scale, and the
    # difference is not stylistic.
    #
    # Uniform scaling holds hue *exactly*, which is a virtue in the tone stage
    # because its input is near the cube already. Here the input can be 2-4x over
    # white in **one** channel, and holding the ratio exact then means dragging
    # the other two down by the same factor. Measured on a real photograph before
    # this was changed: a bright warm highlight at (1.000, 0.871, 0.634) came out
    # (1.000, 0.305, 0.222) -- luma 0.882 -> 0.447, a **dark saturated red where a
    # bright highlight had been**, on about 6% of the frame. Exactly the artifact
    # this stage exists to remove, introduced by the stage itself.
    #
    # Per channel, each rolls against its own headroom: the reconstructed 2.0 in
    # red comes back to just under white while green barely moves and blue, below
    # the knee, is untouched. The highlight stays bright and loses a little
    # saturation -- which is what film does as a dye layer approaches saturation,
    # and is the same behaviour `highlight_desat` models further down the
    # pipeline. Fitting an out-of-gamut brightness into the cube costs either
    # saturation or luminance; for a highlight, saturation is the right one to
    # spend.
    # **Dilate before feathering, or the radius fights the repair.** A plain blur
    # of the weight field dilutes it: a blown region 120px across, gated through
    # a sigma-100 blur, comes out with a peak well under 1, so the roll weakens
    # and *less* of the recovered range becomes visible. Measured before this
    # was added, the recovered span ran 0.069 at a 16px radius down to 0.041 at
    # 200px -- i.e. reaching further to find the colour made the repair fainter,
    # which is not what the control says it does.
    #
    # Growing the mask first and feathering the grown version is the standard fix
    # and it decouples the two: the gate stays saturated across everything that
    # was repaired, whatever the radius, and only its outer ramp widens.
    # Separable, as two 1-D max pools, because a single 2-D pool at a 200px
    # radius is a 401x401 window.
    # Dilate **wider than the feather**, or the gate never saturates: a blur of a
    # mask dilated by exactly its own sigma pulls the peak back below 1 near the
    # mask's edge, so the roll runs at partial strength and leaves over-range
    # values for the hard clamp to flatten -- the original bug, in miniature.
    # Growing by 2x the feather leaves the interior at a clean 1.0.
    rg = max(1, int(round(radius * _RECON_ROLL_GATE_FRAC)))
    rd = 2 * rg
    gate = w.amax(dim=1, keepdim=True)
    gate = F.max_pool2d(gate, (1, 2 * rd + 1), stride=1, padding=(0, rd))
    gate = F.max_pool2d(gate, (2 * rd + 1, 1), stride=1, padding=(rd, 0))
    gate = _blur(gate, float(rg))
    d = 1.0 - _RECON_ROLL_KNEE
    t = ((out - _RECON_ROLL_KNEE) / d).clamp_min(0.0)
    return torch.where(
        out > _RECON_ROLL_KNEE,
        _RECON_ROLL_KNEE + d * (t + gate * (_shoulder(t) - t)),
        out,
    )


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


def _u64(v: int) -> int:
    """Reinterpret a 64-bit constant as the signed int64 with the same bits."""
    return int(np.uint64(v).astype(np.int64))


# splitmix64's mixing constants, as signed int64. Multiplication and addition are
# bit-identical for signed and unsigned two's complement, so the same bit
# patterns give the same hash -- only the *shifts* need care. See `_lsr`.
_HASH_KY = _u64(0xC2B2AE3D27D4EB4F)
_HASH_KX = _u64(0x9E3779B97F4A7C15)
_HASH_M1 = _u64(0xBF58476D1CE4E5B9)
_HASH_M2 = _u64(0x94D049BB133111EB)


def _lsr(n: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (unsigned) right shift on an int64 tensor.

    torch has no uint64, and int64 `>>` is *arithmetic* -- it smears the sign bit
    down instead of shifting zeros in, so it is simply the wrong operator for a
    hash. Masking off the bits above the shift width restores the unsigned
    result: after shifting right by ``k`` only ``64 - k`` bits can be set.

    **The mask has to come from ``k``.** A first pass at this used one wide
    constant for every shift, which produced something that still *looked* like
    noise and was a different field -- exactly the kind of bug that survives an
    eyeball test, which is why ``verify.py`` asserts bit-equality against the
    numpy reference rather than rendering a frame and squinting at it.
    """
    return (n >> k) & ((1 << (64 - k)) - 1)


def _lattice_np(iy0: int, ix0: int, hl: int, wl: int, seed: int, nfields: int) -> np.ndarray:
    """Deterministic hash noise on an integer lattice window.

    Integer hashing, so it runs on the CPU rather than the GPU: 64-bit integer
    ops are poorly supported on MPS, and the exactness is the point -- two tiles
    asking about the same lattice point must agree bit for bit or exports seam.

    **It is not a small amount of work, whatever the shape of the lattice
    suggests.** An older version of this note claimed the lattice is "far smaller
    than the pixel grid, so this is cheap", and that is false at every setting
    the app actually ships. ``cell`` is floored at ``_MIN_CELL`` = 0.8 *working*
    pixels and every preset in ``presets/`` sets ``grain_size`` to 0.1-0.3, so
    the base lattice is *denser* than the pixel grid. Measured lattice points
    hashed per output pixel: defaults 2.5x, Dreamy 5.7x, Dramatic 38x, Subtle
    48x, ExtraGrain 54x, **Stock 58x** -- 291M hashes for one 2400px proxy
    preview. It was 23% of that render's wall time.

    So it runs in **torch on the CPU rather than numpy**, which is worth 2.5x for
    free: numpy's uint64 elementwise ops are single-threaded, torch's int64 ones
    use ``at::parallel_for`` across every core, and the arithmetic is otherwise
    identical. Measured 113ms -> 46ms on a 44M-point lattice, bit-exact. It stays
    on the CPU (rather than moving to the GPU in 32-bit, which would be faster
    still) because that would change every value and reroll every preset's grain.

    Returns numpy for its callers' ``torch.from_numpy(...)``; the conversion
    shares memory and costs nothing.
    """
    # Lattice indices go negative near the origin; the int64 bit pattern *is* the
    # unsigned one, so this needs no reinterpretation -- it is the same wrap the
    # old `.view(np.uint64)` produced.
    yy = torch.arange(iy0, iy0 + hl, dtype=torch.int64).unsqueeze(1)
    xx = torch.arange(ix0, ix0 + wl, dtype=torch.int64).unsqueeze(0)
    out = torch.empty((nfields, hl, wl), dtype=torch.float32)
    for f in range(nfields):
        # Fold the seed in Python ints so the wrap is explicit.
        s = _u64(((seed + f * 7919) * 0x165667B19E3779F9) % (1 << 64))
        n = xx * _HASH_KX + yy * _HASH_KY + s
        n = n ^ _lsr(n, 29)
        n = n * _HASH_M1
        n = n ^ _lsr(n, 32)
        n = n * _HASH_M2
        n = n ^ _lsr(n, 31)
        out[f] = _lsr(n, 40).to(torch.float32) / float(1 << 24)
    return out.numpy()


def _lat_span(
    n: int, origin: float, cell: float, pad_lo: int, pad_hi: int,
) -> tuple[int, int]:
    """First lattice index and count covering ``n`` working pixels from ``origin``.

    Exists to keep this arithmetic off the GPU. The three noise builders used to
    build their coordinate ramp on the device and then read scalars back off it
    (``int(math.floor(float(ys[0])))``), and every such read drains the MPS
    command queue -- counted 32 per ``render_supersampled`` at defaults and 108
    at ``Stock``, each a full pipeline stall for a number Python already had.

    **Computed in float32, and that is not pedantry.** The device ramp is
    ``torch.arange(n, dtype=float32) + origin``, and a Python float scalar takes
    the tensor's dtype, so the whole expression is float32. Doing it in float64
    here would occasionally land on the other side of an integer boundary and
    select a *different* lattice window -- which is a different noise field, not
    a rounding difference. ``verify.py`` pins this against the device path.
    """
    f = np.float32
    lo = f(f(f(0.0) + f(origin)) / f(cell))
    hi = f(f(f(n - 1) + f(origin)) / f(cell))
    i0 = int(math.floor(float(lo))) - pad_lo
    return i0, int(math.floor(float(hi))) + pad_hi - i0 + 1


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
    iy0, hl = _lat_span(h, y0, cy, 1, 2)
    ix0, wl = _lat_span(w, x0, cell, 1, 2)

    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cy
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell

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
    iy0, hl = _lat_span(h, y0, cell, 0, 0)
    ix0, wl = _lat_span(w, x0, cell, 0, 0)

    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cell
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell

    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, nfields)).to(device)
    iy = (torch.floor(ys).long() - iy0).clamp(0, hl - 1)
    ix = (torch.floor(xs).long() - ix0).clamp(0, wl - 1)
    return lat[:, iy][:, :, ix].unsqueeze(0)


# Jitter range for `_variable_cell_noise`'s point placement, as a fraction of
# the cell. Keeping every point in the middle half of its own cell -- never
# closer than a quarter-cell to an edge -- is what lets the neighbour search
# below be *exact* rather than a heuristic, given the search radius below.
_VARCELL_JITTER_LO = 0.25
_VARCELL_JITTER_SPAN = 0.5

# How many rings of neighbouring cells the search below checks, each way --
# 2 means 5x5 (25 candidates). Worked from the far corner case: a point in the
# first excluded ring is at least ``(RINGS + 1 - 0.75)`` cells from a pixel in
# the search centre's own cell, and no point's radius can exceed one cell (see
# ``_variable_cell_noise``), so the domain-warp amplitude below has to leave
# at least that much margin over the 1-cell radius. At 1 ring (3x3) the margin
# is only 0.25 cells, which measurably was not enough room to fix the
# resonance below without also risking the proof; at 2 rings (5x5) it is 1.25,
# comfortable enough that the warp amplitude is the only thing this trades
# against, not correctness.
_VARCELL_RINGS = 2

# A domain warp on the pixel side of the distance computation only -- never on
# which cell a pixel is nominally assigned to, which would need the search
# proof above re-done against a wider slack. It exists for one specific
# failure mode: when the working-pixel cell size lands on (or very near) an
# exact integer, every pixel's position falls at *exactly the same* fractional
# offset within its own cell, for every row and column -- the sampling grid
# and the cell grid are commensurate, so no pixel is ever nearer than
# `_VARCELL_JITTER_LO` of a cell to any point, anywhere, and the field can
# never reach its own full amplitude. Measured on a flat field: cell 1.00
# scored 0.123 std against 0.193 at 1.05 or 0.95, a >35% amplitude hole
# exactly where a user's slider is most likely to land, since round numbers
# are round numbers before and after the working-scale multiply.
#
# A first attempt warped by only 0.12 cells, the most a 3x3 (1-ring) search
# could safely absorb, and it was not enough: measured, cell 1.00 only moved
# 0.123 -> 0.128, because breaking a phase lock that pins *every* pixel at the
# same offset takes a warp comparable to half a cell, not a tenth of one.
# Widening the search to 5x5 buys room for that: swept 0.5 to 1.2 cells at
# cell 1.00 against cell 1.6 as a control, and 0.7 is where the two agree to
# within 0.3% (0.899 -> 0.997 -> back up past 1.0 and settling around 1 as the
# warp overshoots and returns) -- comfortably inside the 1.25-cell budget the
# 5x5 search allows. Sourced from `_value_noise` rather than a second point
# lattice: interpolated noise does not itself suffer this failure mode (its
# output varies smoothly pixel to pixel regardless of whether its own cell
# aligns with the pixel grid), so it cannot reintroduce the same problem at
# one remove.
_VARCELL_WARP_CELL_FRAC = 0.37
_VARCELL_WARP_AMOUNT = 0.7

# How much better a candidate's falloff has to be than the current winner's
# before it takes over. See the tie-break comment where this is used: the
# coordinate arithmetic behind `dist` differs in its last bit or two between
# computation paths, and without a margin comfortably above that noise floor
# a near-tie between two candidates can resolve to a *different* winning
# point -- and therefore a different brightness -- depending on how a tile
# happened to be padded and cropped to get there.
#
# Swept 1e-4 to 1e-2 against a scene that reproduced the failure: 1e-4 still
# left a 4.9e-3 tile-independence gap (too close to the ~2e-5 noise floor to
# fully cover it), 3e-4 to 1e-3 both closed it to 1.4e-4, and 1e-2 reopened a
# much larger one (4.9e-2) -- a margin that wide starts treating pixels with a
# genuine, non-noise difference in falloff as tied, which is a different
# failure than the one this exists to fix. Set in the middle of the working
# range rather than at its edge.
_VARCELL_TIE_MARGIN = 1e-3


def _variable_cell_noise(
    h: int, w: int, y0: float, x0: float, lo: float, hi: float, seed: int,
    device: torch.device, nfields: int = 1,
) -> torch.Tensor:
    """Cellular ("Worley-style") noise with an independently random radius per
    point, drawn uniformly from ``[lo, hi]`` -- genuine per-clump size
    variation, rather than one clump size for the whole field.

    ``_fbm`` cannot do this: its lattice is addressed on a single uniform
    pitch, so every "clump" it produces is the same size by construction --
    Octaves and Roughness change how much *structure* stacks on top of the
    base clump, never the base clump's own size. Getting an independently
    sized clump means giving up the uniform lattice and its cheap
    ``grid_sample`` interpolation trick, and building the field from discrete
    points instead.

    One point per cell of a base lattice pitched at ``hi`` -- the *largest*
    radius any point can have, which is what makes the fixed-ring neighbour
    search below exact rather than approximate (see ``_VARCELL_JITTER_LO`` and
    ``_VARCELL_RINGS`` for the proof). Each point's position is jittered
    within the middle half of its cell, and each draws its own radius
    uniformly from ``[lo, hi]`` and its own brightness, all from one hash per
    cell -- deterministic and addressed in global coordinates, so two tiles
    asking about the same point agree.

    A pixel takes its value from whichever of the candidate points has
    the strongest claim on it -- the largest radial falloff (1 at that point's
    own centre, 0 at its own radius, cubic in between), which is "closest
    *relative to that point's own size*" rather than closest in raw distance.
    That point's own signed brightness, scaled by the falloff, is the output.
    Selecting by falloff rather than combining every candidate is what keeps
    this reading as discrete particles with their own edges rather than as
    fog, and it is what makes coverage gaps -- clear film base between grains,
    where no point's falloff ever rises off zero -- an honest consequence of a
    wide ``lo``-``hi`` range rather than a bug to paper over.

    **Centred on 0.5, the same convention `_fbm` returns, and for the same
    reason `_smooth_noise` needs it**: that function re-centres explicitly
    (``0.5 + blur(n - 0.5, sigma) * gain``), so a field that means something
    different at 0.5 would be blurred around the wrong point. A gap -- every
    candidate's falloff at zero -- must land exactly on 0.5, the shared
    "no deviation" value both the blur and the caller's later ``*2-1`` remap
    already assume; landing gaps on raw zero instead would read as the
    strongest possible *darkening* over most of the frame the moment any
    range wider than a sliver of ``lo``-``hi`` left real gaps, which is
    backwards from "nothing is here". So brightness is drawn signed, in
    ``[-1, 1)``, and the output is ``0.5 + 0.5 * falloff * brightness``: a gap
    (falloff 0) is 0.5 regardless of any candidate's brightness, and a pixel
    sitting in a point's centre reaches its own brightness at full amplitude,
    lighter or darker with equal likelihood -- matching how real grain is not
    one-sided either.

    Returns shape ``[1, nfields, h, w]``, matching `_fbm`'s convention, so it
    drops into the same normalise-and-clamp pipeline the mono and chroma
    global-grain fields already share. Geometry (positions and radii) is
    identical across fields and the winning point is chosen once, shared by
    every field; only brightness is drawn independently per field and read
    from whichever point won -- the same "shared shape, independent value"
    pattern ``_lattice_np`` already gives ``_fbm``'s multi-field calls, and it
    is what lets a chroma variant share every grain's position and size across
    channels while still giving each channel its own intensity.
    """
    cell = hi
    # The ramp is monotonically increasing (cell > 0 always), so its min and max
    # are its first and last entries -- which `_lat_span` computes in Python,
    # saving both the scalar reads and the two full-tensor reductions that used
    # to feed them.
    iy0, hl = _lat_span(h, y0, cell, _VARCELL_RINGS, _VARCELL_RINGS)
    ix0, wl = _lat_span(w, x0, cell, _VARCELL_RINGS, _VARCELL_RINGS)

    # Three geometry channels (jitter y, jitter x, radius fraction) plus one
    # brightness channel per output field, all from the one hash so a single
    # CPU call covers the whole lattice window.
    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, 3 + nfields)).to(device)
    jy = _VARCELL_JITTER_LO + _VARCELL_JITTER_SPAN * lat[0]
    jx = _VARCELL_JITTER_LO + _VARCELL_JITTER_SPAN * lat[1]
    rad = lo + lat[2] * (hi - lo)
    bri = lat[3:] * 2.0 - 1.0  # [nfields, hl, wl], signed -- see the docstring

    cell_iy = torch.arange(iy0, iy0 + hl, device=device, dtype=torch.float32)[:, None]
    cell_ix = torch.arange(ix0, ix0 + wl, device=device, dtype=torch.float32)[None, :]
    py = (cell_iy + jy) * cell  # [hl, wl], this cell's point in working px
    px = (cell_ix + jx) * cell

    Y = (torch.arange(h, device=device, dtype=torch.float32) + float(y0))[:, None]
    X = (torch.arange(w, device=device, dtype=torch.float32) + float(x0))[None, :]
    # Cell assignment uses the true, unwarped position -- the search proof
    # above is against this coordinate, and warping it would need a wider
    # search to stay valid.
    piy = (torch.floor(Y / cell).long() - iy0).clamp(0, hl - 1)  # [h, 1]
    pix = (torch.floor(X / cell).long() - ix0).clamp(0, wl - 1)  # [1, w]

    # See `_VARCELL_WARP_CELL_FRAC` -- breaks the pixel-grid/cell-grid
    # resonance without touching which cell a pixel belongs to.
    warp_cell = max(_MIN_CELL, _VARCELL_WARP_CELL_FRAC * cell)
    wn = _value_noise(h, w, y0, x0, warp_cell, seed + 51, 2, device)
    Yw = Y + (wn[0, 0] * 2.0 - 1.0) * (_VARCELL_WARP_AMOUNT * cell)
    Xw = X + (wn[0, 1] * 2.0 - 1.0) * (_VARCELL_WARP_AMOUNT * cell)

    # The winning candidate is chosen by falloff alone -- shared across every
    # field -- and only afterward is that winner's own (per-field) brightness
    # read out. Selecting by the product instead would let a dim, distant
    # candidate's positive brightness beat a near-zero-falloff gap into
    # registering as real signal, which is exactly backwards: a gap must stay
    # a gap regardless of what any nearby point's brightness happens to be.
    # Carries the *winning cell*, not the winning brightness. Brightness is
    # `[nfields, h, w]`, so keeping it in the loop meant `torch.where` rewriting
    # nfields full-frame planes on all 25 iterations -- 75 plane writes to end up
    # using one. The flat cell index is a single plane whatever `nfields` is, and
    # one gather afterwards reads out the winner. Measured 1.26x on this function
    # at nfields=3 and bit-identical, which it has to be: same candidates, same
    # order, same comparison, so the same point wins and the same brightness is
    # read from it. The saving is mostly peak memory, which is what lets a
    # weaker machine afford a larger tile.
    best_shape = torch.zeros(h, w, device=device)
    best_cell = torch.zeros(h, w, device=device, dtype=torch.long)
    for dy in range(-_VARCELL_RINGS, _VARCELL_RINGS + 1):
        for dx in range(-_VARCELL_RINGS, _VARCELL_RINGS + 1):
            ny = (piy + dy).clamp(0, hl - 1)
            nx = (pix + dx).clamp(0, wl - 1)
            dxp = Xw - px[ny, nx]
            dyp = Yw - py[ny, nx]
            dist = torch.sqrt(dxp * dxp + dyp * dyp)
            d_norm = dist / rad[ny, nx].clamp_min(1e-3)
            shape = 1.0 - _smoothstep(0.0, 1.0, d_norm)  # [h, w]
            # A margin, not a bare `>`: two candidates can be genuinely almost
            # tied (a pixel near the border between two blobs), and there the
            # coordinate arithmetic feeding `dist` differs in its last bit or
            # two depending on how a tile happens to be padded and cropped --
            # around 2e-5 measured. Without a margin comfortably above that
            # noise floor, a near-tie can resolve to a *different* winning
            # point depending on tile layout, which is not a gradual drift but
            # a discrete jump to that other point's own brightness -- the one
            # thing a tile-independent field must never do. The margin is
            # fixed and the candidate order is fixed (same dy, dx sweep every
            # time), so whichever candidate is reached first while within it
            # keeps the win consistently, regardless of how the image was
            # tiled to get here.
            win = shape > best_shape + _VARCELL_TIE_MARGIN
            best_cell = torch.where(win, (ny * wl + nx).expand(h, w), best_cell)
            best_shape = torch.where(win, shape, best_shape)

    # One gather instead of 25 masked writes. `bri` is [nfields, hl, wl] and
    # `best_cell` indexes its flattened lattice, so this reads each pixel's
    # winning point's own per-field brightness.
    best_bri = bri.reshape(nfields, -1).gather(
        1, best_cell.reshape(1, -1).expand(nfields, -1)
    ).reshape(nfields, h, w)
    return (0.5 + 0.5 * best_shape.unsqueeze(0) * best_bri).unsqueeze(0)


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


def _smooth_noise(n: torch.Tensor, cell: float, amount: float) -> torch.Tensor:
    """Blur a 0..1 noise field and put back the amplitude the blur costs.

    The defect this exists for is structural rather than a bad setting: value
    noise is a quilt of axis-aligned cells, invisible at a 1.6px clump and
    plainly rectangular at 20px. No amount of re-seeding or extra octaves moves
    it, because every octave is built on the same kind of lattice -- measured,
    the two-octave global field scores the same gridiness as one octave alone.
    A filter on the field is the cure.

    **Variance is restored, and that is the point.** `_fbm` preserves variance
    so Octaves changes structure at constant strength; this follows the same
    rule for the same reason. A smoothing control that quietly turned the layer
    down would leave Global Intensity fighting it, and "smoother" would be
    indistinguishable from "less".

    The gain is a closed form in ``sigma/cell`` rather than a measurement of
    the field in hand. Normalising against ``n.std()`` would be a statistic of
    the region -- invariant 1 -- and would restore a different amount in every
    tile of an export while every preview looked fine.
    """
    if amount <= 0.001:
        return n
    sigma = amount * _SMOOTH_MAX * cell
    if sigma < 0.05:
        return n
    gain = math.sqrt(1.0 + _SMOOTH_GAIN_K * (sigma / cell) ** 2)
    return 0.5 + _blur(n - 0.5, sigma) * gain


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

class GrainEngine:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        # Global Grain texture cache -- see `_global_grain_field`. An
        # OrderedDict used as an LRU: `move_to_end` on a hit, pop from the front
        # when over `_GG_CACHE_BYTES`.
        #
        # A plain dict needs no lock because `main._RENDER_LOCK` serialises every
        # render. That is a load-bearing assumption borrowed from the caller: if
        # renders ever run concurrently, this needs a lock or a per-render cache.
        self._gg_cache: collections.OrderedDict = collections.OrderedDict()
        self._gg_bytes = 0
        # Hit/miss counters, for `verify.py` to assert *which* parameters miss
        # rather than only that the output is right. A stale-cache bug renders a
        # plausible texture, so "the numbers changed" is not enough of a test.
        self.gg_hits = 0
        self.gg_misses = 0

    def clear_caches(self) -> None:
        """Drop the Global Grain texture cache."""
        self._gg_cache.clear()
        self._gg_bytes = 0

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
    def _global_grain_field(
        self, h: int, w: int, y0: float, x0: float, p: dict,
        gcell: float, gcell_max: float, variable: bool,
    ) -> torch.Tensor:
        """The Global Grain texture layer, normalised and clamped. Cached.

        Shape is ``[1,1,h,w]`` at chroma 0 and ``[1,3,h,w]`` above it; both
        broadcast against the frame, which is why the channel count is allowed to
        depend on a parameter.

        **Cached because it reads no image data at all.** Every input is either a
        parameter or the tile's own global coordinates, so nudging Halation or
        Sharpen currently pays to rebuild a texture that has not changed --
        measured at 1.29s of a 3.70s `Stock` proxy preview, 35%. The two sliders
        in this section anyone actually drags, `global_intensity` and
        `global_opacity`, are applied by the *caller* as a single scalar multiply
        and so sit outside this boundary entirely: they cannot miss the cache.

        The cache key has to cover every input, and the failure mode if it does
        not is the nasty kind -- a stale hit renders a perfectly plausible
        texture that is simply the previous one, so nothing looks broken. What is
        in it, and why:

        * ``y0, x0, h, w`` -- **absolute global coordinates, never a tile
          index.** The field is addressed globally (invariant 1), so keying on
          anything relative would hand one tile another tile's texture and seam
          every export while every preview looked fine.
        * ``gcell, gcell_max, variable`` -- the *derived* working cells, not the
          raw sliders. Two different (size, scale) pairs that floor to the same
          working cell genuinely produce the same field and should share an
          entry, and this folds in both `scale` and the supersample level for
          free, since the caller has already multiplied them in.
        * ``seed`` -- both field draws offset from it.
        * ``global_smooth`` -- `_smooth_noise` is inside this boundary.
        * ``global_chroma`` -- decides whether the second field is built at all,
          and changes the returned channel count.
        * the device -- these are device-resident tensors.

        Deliberately *not* keyed on the image or upload: the field never reads
        either. Two photographs of the same dimensions already get an identical
        global-grain field today, because the lattice is addressed in absolute
        coordinates; caching does not change that, it just stops recomputing it.

        Caching the finished field rather than `_lattice_np` is the deliberate
        choice even though the lattice would be more general. The lattice is the
        one array in the pipeline you least want to hold: at `Stock` it is ~58
        points per output pixel (see `_lattice_np`). This is one plane per
        channel at working resolution.
        """
        key = (
            h, w, float(y0), float(x0), gcell, gcell_max, variable,
            int(p["seed"]), p["global_smooth"], p["global_chroma"],
            str(self.device),
        )
        hit = self._gg_cache.get(key)
        if hit is not None:
            self._gg_cache.move_to_end(key)
            self.gg_hits += 1
            return hit
        self.gg_misses += 1

        def field(seed_off: int, nfields: int) -> torch.Tensor:
            if variable:
                return _variable_cell_noise(
                    h, w, y0, x0, gcell, gcell_max,
                    int(p["seed"]) + seed_off, self.device, nfields,
                )
            return _fbm(
                h, w, y0, x0, gcell,
                int(p["seed"]) + seed_off, nfields, 2, 0.5, self.device,
            )

        gg = field(7717, 1)
        # Before the normalise-and-clamp, not after: the clamp is what
        # gives the field its hard tails, and smoothing a clamped field
        # would leave the plateaus it created and merely round their
        # corners. Smoothed first, the clamp bites on a field that has
        # already lost its extremes, so the rails are reached less often.
        #
        # Referenced against Max in the variable case: that is the field's
        # own characteristic scale now (the base lattice pitch
        # `_variable_cell_noise` places one point per cell of), where Min
        # was the reference for the old single-size field. Smoothness on
        # the variable field mostly softens the base-lattice boundary
        # rather than a value-noise quilt -- there is no axis-aligned quilt
        # here to begin with, since the points are jittered off-grid.
        gg = _smooth_noise(gg, gcell_max if variable else gcell, p["global_smooth"])
        gg = gg * 2.0 - 1.0

        # Chroma: decorrelate the three channels without touching the
        # monochrome field.
        #
        # The obvious construction is `_grain_field`'s -- draw three
        # independent fields, take their rescaled mean as the monochrome
        # component and blend outward. It is not used here for two reasons.
        # It would replace the single field this layer has always been built
        # from, rerolling every existing preset's global grain at chroma 0;
        # and that blend does not hold amplitude, because the mean and the
        # per-channel fields are correlated -- measured pre-clamp, it dips
        # to 88.8% of its own strength at chroma 0.5 and returns to 99.9% by
        # 1.0, so the slider quietly moves loudness as well as colour.
        #
        # Instead the mono field `m` is kept exactly as it was and a
        # *mean-zero* deviation `d` is added on top, from its own seed.
        # Because `d` sums to zero across channels its statistics are fixed
        # -- var 2/3 and covariance -1/3 of a single field -- and the two
        # coefficients can be solved rather than guessed:
        #
        #     g_c = A*m + B*d_c,   A = sqrt(1 - 2/3 c),  B = sqrt(c)
        #
        # gives unit variance and cross-channel correlation exactly `1 - c`
        # at every setting. Measured: correlation 1.000 / 0.501 / 0.001 at
        # chroma 0 / 0.5 / 1, pre-clamp amplitude flat to 0.6%, and chroma 0
        # bit-identical to the old layer (max channel spread 0.0).
        #
        # The one thing that does move is the clamp below. Mixing in `d`
        # gaussianises the field, so it reaches the rails less often --
        # clipping falls 25.4% -> 22.8% across the slider -- and since a
        # clipped sample sits at exactly +-1 rather than wherever it was
        # headed, less clipping means slightly less measured sigma. Rendered
        # amplitude therefore drifts 100% -> 96.8% from chroma 0 to 1. That
        # is the hard tails doing their job, not the blend, and it is a
        # third of the wobble the other construction has.
        gc = p["global_chroma"]
        if gc > 0.001:
            # Same construction as `gg` above (fBm or variable-cell,
            # whichever `variable` selected), through the same closure --
            # the decorrelation below only needs `gs` to be a second field
            # of comparable amplitude, but sharing the geometry generator
            # means a variable-size chroma field reuses each grain's own
            # position and radius across channels and only randomises its
            # per-channel brightness, which is what gives a coloured grain
            # its speckle without moving its shape from channel to channel.
            gs = field(3391, 3)
            gs = _smooth_noise(gs, gcell_max if variable else gcell, p["global_smooth"])
            gs = gs * 2.0 - 1.0
            gd = gs - gs.mean(dim=1, keepdim=True)
            gg = gg * math.sqrt(1.0 - (2.0 / 3.0) * gc) + gd * math.sqrt(gc)

        gg = (gg / _GNORM).clamp(-1.0, 1.0)

        # LRU insert. An entry larger than the whole budget is returned without
        # being cached rather than immediately evicting itself -- otherwise a
        # single-tile render of a large frame would thrash the cache empty on
        # every pass and pay the bookkeeping for nothing.
        nbytes = gg.element_size() * gg.nelement()
        if nbytes <= _GG_CACHE_BYTES:
            self._gg_cache[key] = gg
            self._gg_bytes += nbytes
            while self._gg_bytes > _GG_CACHE_BYTES and len(self._gg_cache) > 1:
                _, old = self._gg_cache.popitem(last=False)
                self._gg_bytes -= old.element_size() * old.nelement()
        return gg

    # ------------------------------------------------------------------ #
    @staticmethod
    def _grade(img: torch.Tensor, p: dict, scale: float) -> torch.Tensor:
        """Colour grading, step -1: the only block above pre-blur.

        Everything below this models an emulsion. This models the *decision*
        about what the photograph is before any of that runs -- which is why it
        sits at the very top and why the LUT is last within it: white balance,
        exposure and the tonal adjustments set the light and the tonal range,
        then the LUT reads the picture it was meant to read.

        Written to be cheap, which was the explicit ask, and two of the twelve
        stages are the only ones that are not free: everything else is pure
        per-pixel arithmetic with no kernel and no neighbourhood at all, so it
        costs a couple of passes over the frame and adds nothing to ``pad_for``.
        Clarity needs a blurred copy to find its band, and it is a
        *single-channel* blur rather than three because the detail it extracts is
        added back to all three channels equally. Highlight reconstruction needs
        two three-channel blurs to find what the light was doing around a blown
        region -- the one stage here that was accepted as expensive on purpose,
        because the alternative is losing the highlight.

        Tile independence comes for free everywhere except those two, for the
        same reason: nothing here reads a statistic of the region. The LUT is a
        fixed table, temperature/tint are constant vectors, exposure is a
        constant multiply, and the tone, contrast, black-point, vibrance and
        saturation stages are all curves or gains on each pixel's own level.
        Both blurs are kernels like any other and are paid for in ``pad_for``.

        Clamping happens **once**, after the tone stage, and not after each
        stage. That is load-bearing rather than tidy: white balance and exposure
        can push a value out of the cube, and clipping it there is precisely
        what made the highlight control a brightness shift over an already-flat
        patch instead of a recovery.
        """
        temp = p["grade_temp"]
        tint = p["grade_tint"]
        ev = p["grade_exposure"]
        sh = p["grade_shadows"]
        hl = p["grade_highlights"]
        ct = p["grade_contrast"]
        bp = p["grade_black_point"]
        cl = p["grade_clarity"]
        vib = p["grade_vibrance"]
        sat = p["grade_saturation"]
        rec = p["grade_recover"]
        lut = p.get("lut")
        mix = p["lut_amount"]

        # -1z. Highlight reconstruction, above everything including white
        #      balance -- the only stage in this section that adds information
        #      rather than rearranging it.
        #
        #      First because every stage below reads the picture and this
        #      changes what the picture *is*: white balance multiplying a
        #      channel that is sitting on the ceiling multiplies a wrong number,
        #      exposure raises a plateau as a plateau, and the tone curve can
        #      only roll off what it was given. Restore the channel first and
        #      all three are working on the scene instead of on the file's
        #      ceiling.
        #
        #      The one kernel here besides Clarity, and `pad_for` carries it.
        if rec > 0.001:
            img = _reconstruct_highlights(
                img, rec, max(1.0, p["grade_recover_radius"] * scale),
            )

        # -1a. White balance: temperature (blue/amber) and tint (green/magenta),
        #      in linear light.
        #
        #      A white balance is a change in the *illuminant*, so it multiplies
        #      light -- and gamma-encoded values are not light. Done encoded the
        #      same gain moves the shadows much further than the highlights,
        #      which is what makes a naive slider read as a tint laid over the
        #      picture instead of a different lamp. Same argument as
        #      `pre_blur`'s, and gated the same way so the transfer round trip
        #      costs nothing when both stages are off.
        #
        #      One round trip for both axes, not two: a change of illuminant
        #      moves along both at once, so temperature and tint are one
        #      physical operation rather than two, and the combined gain vector
        #      is normalised by its own luma exactly once so the control stays
        #      colour-only -- warming or tinting a frame must not also expose
        #      it, or every other tonal control in the app is being fought by
        #      this one.
        if abs(temp) > 0.001 or abs(tint) > 0.001:
            gt = _GRADE_TEMP_GAIN * temp
            gn = _GRADE_TINT_GAIN * tint
            gain = [1.0 + gt + gn, 1.0 - gn, 1.0 - gt + gn]
            norm = sum(w * c for w, c in zip(_LUMA, gain))
            gain = torch.tensor(
                [c / norm for c in gain], device=img.device, dtype=img.dtype,
            ).view(1, 3, 1, 1)
            # Deliberately not clamped here. A clamp would clip whichever
            # channel the new illuminant raises, which both breaks the hue it
            # was setting and throws away the headroom the tone stage below
            # exists to recover; the section clamps once, after that stage.
            img = _linear_to_srgb(_srgb_to_linear(img) * gain)

        # -1b. Exposure, in linear light, ahead of every luma-keyed mask below --
        #      Shadows, Highlights, Clarity, Vibrance and Saturation all measure
        #      *this* image, so raising exposure first means they read the frame
        #      at the light level actually being graded rather than the one that
        #      arrived. Same construction as Tone Response's Brightness: a
        #      stops multiply that lets the sRGB encoding roll the highlights
        #      off on the way back, instead of a display-referred stretch that
        #      would clip them flat.
        #
        #      Not clamped, for the same reason white balance above is not: a
        #      stop of exposure that clips here is a stop of highlight the
        #      recovery curve below can never give back, and clipping it is
        #      exactly the "no recovery, just a brightness shift" failure this
        #      section had. The over-range value survives to the tone stage,
        #      which rolls it back inside the cube with its detail intact -- or
        #      to the single clamp after it, which is bit-identical to clamping
        #      here when the tone controls are off.
        if abs(ev) > 0.001:
            img = _linear_to_srgb(_srgb_to_linear(img) * (2.0 ** ev))

        # -1c/d. Shadows and highlights, display-referred: tone *recovery*, not
        #        a brightness shift over the region that happens to be bright.
        #
        #        Two decisions, and each fixes something the share-of-headroom
        #        version above them got wrong.
        #
        #        **The curve is `_tone_roll`, which is monotone by algebra.**
        #        The previous formula scaled its own strength by the pixel's
        #        level through a steep quintic, and in the recovering
        #        directions -- Shadows up, Highlights down, the two anyone
        #        actually reaches for -- that term overwhelmed the identity and
        #        the transfer inverted: slope **-0.21 across 16% of the range**
        #        at full travel. A control that reverses tonal order does not
        #        recover a highlight, it flattens it into exactly the
        #        textureless patch it was supposed to rescue, which is what was
        #        reported. Here the rail is an asymptote instead, so the whole
        #        of ``[knee, infinity)`` folds into ``[knee, rail)`` with
        #        ordering intact -- over-range data from reconstruction, from
        #        exposure or from a bright source becomes *visible detail*
        #        rather than a clip.
        #
        #        **It keys on the channel maximum and scales all three
        #        together.** The value, not the luma, is the right question for
        #        a control about clipping: a saturated red at (1, 0, 0) has a
        #        channel hard against the ceiling while its luma is 0.21, and
        #        the old luma key called that a shadow. And because the whole
        #        pixel is scaled by one factor, hue and HSV saturation are
        #        preserved *exactly* rather than approximately -- a uniform
        #        scale cannot move a ratio -- while gamut safety is structural:
        #        the curve's output is bounded by the rail, so every channel,
        #        being at or below the maximum, is too.
        #
        #        The two halves need no shared reference and no ordering rule.
        #        Highlights only touches ``v > knee`` and cannot push a value
        #        below it; Shadows only touches ``v < knee`` and cannot push one
        #        above. Their supports are disjoint, so they are independent by
        #        construction -- a stronger version of what one shared luma was
        #        buying, and it cannot be got wrong by a later edit.
        if abs(sh) > 0.001 or abs(hl) > 0.001:
            v = img.amax(dim=1, keepdim=True)
            vt = v
            if abs(hl) > 0.001:
                d = 1.0 - _GRADE_TONE_KNEE
                t = ((vt - _GRADE_TONE_KNEE) / d).clamp_min(0.0)
                vt = torch.where(
                    vt > _GRADE_TONE_KNEE,
                    _GRADE_TONE_KNEE + d * _tone_roll(t, -hl),
                    vt,
                )
            if abs(sh) > 0.001:
                d = _GRADE_TONE_KNEE
                t = ((_GRADE_TONE_KNEE - vt) / d).clamp_min(0.0)
                vt = torch.where(
                    vt < _GRADE_TONE_KNEE,
                    _GRADE_TONE_KNEE - d * _tone_roll(t, sh),
                    vt,
                )
            img = img * (vt / v.clamp_min(1e-4))

        # The section's one gamut clamp, and it is *here* rather than after each
        # stage above so that white balance and exposure can hand their
        # over-range result to the tone stage instead of having it clipped away
        # first. That is the difference between "recover the highlight" and
        # "recover what is left of the highlight after we threw it away": raise
        # exposure a stop and pull Highlights back, and the picture comes back,
        # because nothing between the two ever rounded it off to white.
        #
        # With both tone controls at 0 this lands on exactly the old behaviour,
        # since a monotone brightening followed by a clamp is the same picture
        # whichever end the clamp sits at -- and it is bit-exactly a no-op on
        # in-gamut input, which is what keeps the neutral render untouched.
        img = img.clamp(0.0, 1.0)

        # -1e. Contrast: a two-way pivot about the same middle grey the
        #      (deferred) film characteristic curve uses, done directly rather
        #      than through a toe and shoulder.
        #
        #      The gain is floored at 0 so no setting inverts the picture
        #      through grey -- at -1 the spread is a tenth of the original
        #      rather than crossing zero. Unlike the film curve, nothing here
        #      asymptotes, so a strong positive setting clips outright: that is
        #      what a quick contrast control is expected to do, and it runs
        #      after Shadows/Highlights precisely so the clip-free version is
        #      available first.
        if abs(ct) > 0.001:
            gain_ct = max(0.0, 1.0 + _GRADE_CONTRAST_GAIN * ct)
            img = (_MID_GREY + (img - _MID_GREY) * gain_ct).clamp(0.0, 1.0)

        # -1f. Black point: the blunt Levels-style remap, not a masked lift.
        #
        #      Every value at or below `bp` is driven to 0 and 1 stays exactly
        #      at 1, so this genuinely crushes shadow detail rather than easing
        #      it -- the opposite trade from Shadows above, and the reason both
        #      exist. One-directional: there is nothing below 0 to lift from.
        if bp > 0.001:
            img = ((img - bp) / max(1.0 - bp, 1e-4)).clamp(0.0, 1.0)

        # -1g. Clarity: two-way local contrast on one band.
        #
        #      The band is the luma's own high-pass at the chosen radius, added
        #      back to all three channels. Doing it on luminance rather than per
        #      channel is both the cheaper and the better choice -- one blur
        #      instead of three, and because the same signed amount goes to R, G
        #      and B the hue is held exactly, so a saturated area cannot be
        #      pushed out of gamut by a control that is meant to be about
        #      structure.
        #
        #      The negative side is capped at gain 1.0 while the positive side
        #      gets _GRADE_CLARITY_GAIN. That asymmetry is deliberate: at gain 1
        #      a setting of -1 subtracts exactly the band it measured, i.e. the
        #      local contrast is gone. Past that it does not keep flattening, it
        #      reverses -- a dark halo on the light side of every edge, which is
        #      an artifact rather than a look. There is no such ceiling going
        #      the other way.
        if abs(cl) > 0.001:
            r = max(0.5, p["grade_clarity_radius"] * scale)
            lum_c = _luma(img)
            detail = lum_c - _blur(lum_c, r)
            gain_c = cl * (_GRADE_CLARITY_GAIN if cl > 0 else 1.0)
            img = (img + gain_c * detail).clamp(0.0, 1.0)

        # -1h. Vibrance: saturation weighted against how saturated a pixel
        #      already is, so muted colour comes up while colour that is
        #      already strong is left alone. Identical construction to Tone
        #      Response's own Vibrance, kept as its own parameter because this
        #      section runs before the film pipeline and the two must stay
        #      independent -- see params.py for why sharing one slider would be
        #      wrong. Saturation is chroma-over-value, the HSV definition, so a
        #      deep, dark red still reads as fully saturated.
        if abs(vib) > 0.001:
            mx = img.amax(dim=1, keepdim=True)
            mn = img.amin(dim=1, keepdim=True)
            s_ = (mx - mn) / mx.clamp_min(1e-4)
            lum_v = _luma(img)
            gain_v = (1.0 + vib * (1.0 - s_)).clamp_min(0.0)
            img = (lum_v + (img - lum_v) * gain_v).clamp(0.0, 1.0)

        # -1i. Saturation: a flat multiply about the same luma axis, unweighted
        #      by how saturated a pixel already is -- the blunt control
        #      Vibrance is deliberately not. -1 lands exactly on the luma
        #      (monochrome), +1 doubles chroma.
        if abs(sat) > 0.001:
            lum_s = _luma(img)
            gain_s = max(0.0, 1.0 + sat)
            img = (lum_s + (img - lum_s) * gain_s).clamp(0.0, 1.0)

        # -1j. The 3D LUT, last in the section, on the graded frame.
        #
        #      Applied display-referred because that is the space a .cube is
        #      authored in -- its axes are code values, not light. Mixed as a
        #      straight cross-fade, which is the honest reading of "50% of this
        #      LUT" and is what every grading application means by it.
        #
        #      `lut` is absent from `p` unless a request resolved one, and the
        #      service zeroes `lut_amount` when it could not, so this gate and
        #      `params.is_neutral` can never disagree about whether the stage
        #      runs.
        if lut is not None and mix > 0.001:
            graded = _apply_lut(img, lut).clamp(0.0, 1.0)
            img = graded if mix >= 0.999 else img + (graded - img) * mix

        return img

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
    def _antialias(
        self, lin: torch.Tensor, p: dict, scale: float,
    ) -> torch.Tensor:
        """Take stair-stepping off hard edges without softening them.

        A stair-step is a *pixel-scale wobble along* a contour, not a hard
        transition across one -- so the cure is to filter along the isophote
        tangent and never across it. Averaging across is what a blur does, and
        it would take the edge with it; that is the whole reason this is a
        directional filter rather than a soften.

        It overlaps `edge_sand` in mechanism and is deliberately not the same
        control, on three counts. **Position**: this runs at step 1c on the
        source, in the optical block, where the aliasing that came in with the
        file lives; sanding runs at 8b to polish roughness the *jitter stage
        just added*, and cannot reach back to fix the input. **Scale**: three
        taps at about a pixel against sanding's five to +/-2 sigma, because
        the two are removing different wavelengths -- measured, 92% of a
        jittered contour's roughness sits above 8px, while a stair-step is one
        pixel by definition. **Gate**: this one fires on the luma *step*, so
        it finds hard borders and leaves texture alone, where sanding follows
        wherever the grit is dialled.

        In linear light, with the block it sits in. It averages light, and
        averaging gamma-encoded values holds the encoded mean rather than the
        light's -- the same reason `pre_blur` does its transfer round trip.

        Above strength 1 the filter is **repeated** rather than widened, up to
        ``_AA_PASSES``. One three-tap pass is a gentle thing -- 35% of a
        stair-step -- and a single pass was reported as doing "little to none".
        Reaching further is the wrong lever for the reason the taps are short in
        the first place: a stair-step is one pixel wide by definition, so a
        longer filter starts averaging away the shape the contour has rather
        than the wobble on it. Repeating attacks only the wobble, and because
        each pass re-estimates the tangent from the frame it is handed, it
        re-aims along a curving edge where one wide pass cuts the corner. Same
        idiom, and same reasoning, as ``edge_sand``'s ``_SAND_PASSES``.
        """
        st = p["aa_strength"]
        radius = max(0.2, p["aa_radius"] * scale)
        edge_only = p["aa_edge_only"]

        # Whole passes plus a fractional last one, so the control stays
        # continuous and strength <= 1 is bit-for-bit the single pass it always
        # was. Capped: pad_for reserves for _AA_PASSES exactly.
        passes = min(_AA_PASSES, int(math.ceil(st - 1e-6)))

        for i in range(passes):
            # The last pass carries the remainder -- strength 2.5 is two full
            # passes and one at half. Earlier passes are full strength.
            amt = min(1.0, st - i)

            # Display-referred for the detector, encode-then-luma rather than
            # the other way round: the transfer curve does not commute with a
            # weighted sum, and the step thresholds below are shared with edge
            # softening, which measures the same quantity the same way.
            #
            # Re-measured every pass, on the current frame rather than the
            # original. That is the entire value of iterating: the tangent
            # follows the contour as the previous pass left it, so a curve gets
            # followed instead of chorded.
            lum_d = _luma(_linear_to_srgb(lin))
            tx, ty, mag = _isophote(lum_d, max(_AA_DIR_MIN, _AA_DIR_K * radius))
            # Fade out where the tangent is meaningless -- see _isophote.
            # Without it a flat region's direction swings on float noise and
            # tiled and untiled renders disagree by a scatter of single pixels.
            m = _smoothstep(0.0, _SAND_MIN_GRAD, mag)

            # The aliasing gate: how far the luma actually steps across this
            # neighbourhood. `_STEP_LO`/`_STEP_HI` already separate a real
            # transition from fine texture -- fine texture measures an order of
            # magnitude below a hard border -- so a jagged border is found and
            # fabric is left alone. Reused rather than re-derived: two constants
            # for one discrimination would be two things to keep in step.
            if edge_only > 0.001:
                # Measured exactly as edge softening measures it -- same
                # high-pass, same radius convention, no scale factor. The
                # thresholds are calibrated against that quantity, so a fudge
                # here would silently put this control on a different scale from
                # the constants it borrows. An earlier ×2 did precisely that and
                # left the gate firing on fabric.
                step = (lum_d - _blur(lum_d, radius)).abs()
                hard = _smoothstep(_STEP_LO, _STEP_HI, step)
                # Smoothed, or the mask is as ragged as the staircase it is
                # selecting and the filter switches on and off down the edge.
                hard = _blur(hard, radius * 0.6)
                # At 0 the filter runs everywhere, at 1 only on hard edges. A
                # mix rather than a switch, because a CG render aliases on
                # gentler steps than a photograph does.
                m = m * ((1.0 - edge_only) + edge_only * hard)

            out = None
            wsum = 0.0
            for offv, wgt in _AA_TAPS:
                tap = (
                    lin if offv == 0.0
                    else _warp(lin, tx * (offv * radius), ty * (offv * radius))
                )
                out = tap * wgt if out is None else out + tap * wgt
                wsum += wgt
            # Normalised from the weights actually used, not trusted to the
            # table.
            out = out / wsum
            lin = lin + (out - lin) * (amt * m)

        return lin

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

        # -1. Colour grading, above everything -- temperature, shadows,
        #     highlights, clarity, then the 3D LUT. See _grade.
        #
        #     Above `pre_blur` rather than anywhere else because this is the
        #     decision about what the photograph *is*; every stage below it is
        #     the emulsion's response to that photograph. Putting it after the
        #     film stages would mean grading grain, halation and dust along with
        #     the picture, and a LUT built to be fed a photograph would be being
        #     fed a rendered negative instead.
        img = self._grade(img, p, scale)

        # 0. Pre-blur, on the untouched input, in linear light.
        #
        #    The same gaussian as `micro_blur` and a different stage, because
        #    a blur's effect here is not only what it does to the pixels:
        #
        #    * It runs before `lum_ref` is taken, so the edge mask, the
        #      hard-edge step mask and the smooth-area guard all measure the
        #      *softened* frame. Micro-blur is deliberately excluded from that
        #      -- the masks read the untouched tile input so diffusing the
        #      frame cannot quietly talk the grain amount down. Here that
        #      coupling is the point: soften the source and the grain follows
        #      the softer edges and backs off where detail has gone.
        #    * It runs before the pre-sharpen below, so a broad radius here
        #      against a tight one there is a detail-killing pair the pipeline
        #      could not otherwise express. The other order would just throw
        #      the sharpening away.
        #
        #    In linear light for micro-blur's reason: a blur is light spreading
        #    sideways, and averaging gamma-encoded values instead darkens every
        #    edge it crosses. Gated so the transfer round trip costs nothing
        #    when the stage is off.
        pb = p["pre_blur"] * scale
        if pb >= 0.05:
            img = _linear_to_srgb(_blur(_srgb_to_linear(img), pb)).clamp(0.0, 1.0)

        # 0b. Pre-sharpen, on the input.
        #
        #    Placed before every film stage so it sharpens the *photograph* and
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

        # 1. Diffusion resolved as discrete deflections rather than as an
        #    average -- see _scatter for why that is a different operation and
        #    not a slower blur. In linear light because it happens to the light,
        #    before the emulsion records anything.
        #
        #    **Ahead of micro-blur, and the order is deliberate** (changed
        #    2026-08-03, on request). Both model the same physical event from
        #    opposite ends, and which runs first changes the result a long way --
        #    measured on separate plates at scatter 0.85 / reach 3 / blur 1px,
        #    against the same stages alone:
        #
        #    | | fine texture | hard edge |
        #    |---|---|---|
        #    | scatter alone | 100% | 100% |
        #    | micro-blur alone | 28% | 34% |
        #    | blur then scatter (old) | 28% | **60%** |
        #    | scatter then blur (new) | 32% | **28%** |
        #
        #    The edge column is the whole story, and the old order's number is
        #    the surprising one: **scatter was undoing the blur.** Displacing a
        #    blurred gradient by whole pixels drops a hard step back into it, so
        #    the pair came out *harder* on borders than the blur alone -- 60%
        #    against 34% -- which is not a thing either stage claims to do.
        #
        #    This way round each stage does its own job. Scatter gets the
        #    source's own detail at full contrast and shreds the border into
        #    raggedness; the blur then averages that raggedness into a genuinely
        #    soft transition, ending *below* blur-alone at 28%. Fine texture
        #    barely notices the swap (28% -> 32%) because scatter does not touch
        #    texture sigma either way. It is also the physical order: light
        #    deflects off a grain and then goes on diffusing.
        #
        #    Note the masks below are measured from the *untouched* tile input,
        #    so scattering the frame does not talk the edge mask or the
        #    smooth-area guard into turning grain down -- the same independence
        #    micro-blur has, and for the same reason.
        if p["scatter"] > 0.001:
            lin = self._scatter(lin, h, w, y0, x0, p, scale)

        # 1b. Light diffusing sideways through the gel layers, as an average.
        #     Last in the light path -- see above.
        lin = _blur(lin, mb)

        # 1c. Anti-aliasing -- stair-stepping off the incoming file's hard
        #     edges, filtered along the contour so the edge itself stays put.
        #
        #     Here rather than at the top of the pipeline because an
        #     anti-alias filter is an *optical* element: on a camera it is a
        #     birefringent plate in front of the sensor, so it belongs in the
        #     light path beside the other two optical stages and ahead of
        #     anything the emulsion does. It also has to run before the masks
        #     are measured, or the grain would keep keying on the jaggies this
        #     just removed.
        if p["aa_strength"] > 0.001:
            lin = self._antialias(lin, p, scale)

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
            add = glow * tint * (hal * 0.9)

            # 2b. Highlight recovery: add the bloom into the headroom that is
            #     actually there, instead of adding it flat and letting the
            #     total clip.
            #
            #     The stage exists because halation adds light in linear space
            #     with no ceiling until display space, so a highlight already
            #     near white gets pushed the rest of the way to a flat,
            #     textureless clip -- reported as halation burning highlights
            #     out. Holding the *glow* back was the first answer and it is
            #     the wrong one: it buys headroom by deleting the bloom, so the
            #     highlights stop burning because the effect stopped happening
            #     there. It also cannot restore anything, because two pixels
            #     that both clipped are still both at 1.0 afterwards.
            #
            #     What this does instead is meter the light against the room
            #     that is actually left. With ``H = 1 - lin`` the headroom each
            #     channel still has,
            #
            #         add' = add * (H + add * (1 - r)) / (H + add)
            #
            #     which at ``r = 1`` is ``add * H / (H + add)``. Three
            #     properties, and each is why it is this expression and not one
            #     of the others tried:
            #
            #     * **Free where there is room.** For ``add << H`` it is ``add``
            #       to first order, so an ordinary highlight with headroom to
            #       spare gets the whole bloom at full strength and the control
            #       costs nothing there. Only a pixel being asked to take more
            #       light than it can hold is metered at all.
            #     * **Cannot reach white at r = 1**, since ``add' < H`` strictly.
            #     * **Strictly increasing in ``lin``**: d(out)/d(lin) =
            #       ``1 - r * a^2 / (H + a)^2``, bounded below by ``1 - r`` and
            #       positive throughout. Nothing flattens, so nothing is lost.
            #
            #     An exponential soft-add -- ``lin + H(1 - exp(-add/H))``, the
            #     obvious tone-mapping answer -- was built and measured first
            #     and is *worse*, which is worth recording because it looks
            #     better on paper. It bends from the origin, so it compresses
            #     hard even where the bloom was modest: on a bright plate
            #     carrying real fine texture it held only 51% of that texture
            #     against this expression's 60%, at less bloom retained. Bending
            #     late beats bending smoothly when what you are protecting is
            #     local contrast rather than the peak value.
            #
            #     Measured on that plate (mean 0.93, fine texture, halation 0.9
            #     at threshold 0.6), against holding the glow back at the same
            #     setting -- highlight texture kept / bloom light kept:
            #
            #     | recovery | hold the glow back | meter against headroom |
            #     |---|---|---|
            #     | 0.5 | 56% / 82% | 53% / 91% |
            #     | 1.0 | 55% / 52% | **60% / 68%** |
            #
            #     At full strength it is better on both axes at once, which is
            #     the whole claim: more of the highlight's detail survives *and*
            #     more of the bloom does.
            #
            #     Keyed on real per-channel headroom rather than on `hi`, the
            #     threshold field the old version used. `hi` answers "is this
            #     pixel bright enough to bloom", which is not the question: a
            #     saturated highlight can sit far above the threshold in luma
            #     while one of its channels still has most of its range free,
            #     and only that channel's own headroom knows so.
            #
            #     What is left on the table, measured: the remaining loss is
            #     compression, not clipping, and it is forced -- red here is
            #     asked to absorb 0.63 of linear light into 0.15 of headroom, so
            #     no metering can be free. The way past it is not a better curve
            #     but a better *model*: real halation is light that *left* the
            #     highlight to re-expose its surroundings, so an
            #     energy-conserving bloom would darken the core as it lights the
            #     halo and the core's texture would survive intact. That is a
            #     change to what halation *is* rather than to this dial, and it
            #     would move every preset that uses the stage, so it is not done
            #     here.
            #
            #     Still purely per-pixel, so `pad_for` is unaffected -- same as
            #     blue compensation above.
            recover = p["halation_recovery"]
            if recover > 0.001:
                head = (1.0 - lin).clamp_min(1e-4)
                add = add * (head + add * (1.0 - recover)) / (head + add)
            lin = lin + add

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
                tx, ty, mag = _isophote(lum, max(0.6, _SAND_DIR_K * sr))
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
        #     On its own seed offset: sharing the main grain's seed would lay it
        #     directly on top of the same clumps and read as nothing more than a
        #     louder version of the same field. Monochrome unless
        #     `global_chroma` asks otherwise -- see below for why that is built
        #     as a separate mean-zero field rather than by the main grain's
        #     recipe.
        #
        #     `global_size_max` (2026-08-04, on request) is the one thing here
        #     that is not just another knob on this same field -- it swaps the
        #     construction entirely. Reported as "too digital a job": `_fbm`
        #     stacks octaves of a *single* clump size, so no matter how rough or
        #     how many octaves, every grain is the same diameter. Above Min,
        #     `_variable_cell_noise` replaces it with discrete points that each
        #     draw their own radius uniformly between Min and Max, which is
        #     what makes clumps genuinely differ from their neighbours rather
        #     than all sharing one stacked structure. See that function for the
        #     construction and `pad_for` for why its reach is Max, not Min.
        gi = p["global_intensity"]
        go = p["global_opacity"]
        if gi > 0.01 and go > 0.001:
            gcell = max(_MIN_CELL, p["global_size"] * scale)
            # Max can never pull the effective ceiling *below* Min: clamped up
            # to it rather than swapped with it, so a preset that raises Min
            # alone (global_size_max still at its own untouched default) can
            # never accidentally cross into the variable-size construction --
            # the two are not a symmetric pair the way the light-leak sizes
            # are, because Min already has an established meaning on its own
            # and Max is purely "how much further can it stretch".
            gcell_max = max(gcell, p["global_size_max"] * scale)
            variable = (gcell_max - gcell) > 1e-3

            gg = self._global_grain_field(
                h, w, y0, x0, p, gcell, gcell_max, variable,
            )
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

        **`master_opacity` is applied here, and the position is the whole
        point.** It cross-fades the finished frame back over the untouched
        input, so it has to see an input that has been through nothing at all
        -- and inside ``render`` there is no such thing at ss > 1, because what
        that method receives is already a bicubic *upsample*. Blending there
        and pooling down would make opacity 0 return the up-then-down round
        trip, which is measurably 1.0e-01 softer than the source on hard edges
        (see ``params.is_neutral``): "no effect" would quietly cost sharpness.
        Blending after the pool, against ``img``, is bit-exact at both ends at
        every supersample.

        Sitting here also means every entry point inherits it -- ``render_image``
        for the export, ``render_view`` for the preview -- so the two cannot
        disagree about what half strength looks like. It is per-pixel against
        the tile's own input, so it touches no statistic of the region and
        ``pad_for`` is unchanged.
        """
        op = p["master_opacity"]
        # Nothing of the render survives, so do not pay for it. The early exit
        # matters most on the slider itself: dragging toward 0 gets cheaper
        # rather than costing a full render to throw away.
        if op <= 0.0:
            return img

        if ss <= 1:
            r = self.render(img, p, scale, y0, x0, full_hw)
        else:
            h, w = img.shape[-2:]
            up = F.interpolate(
                img, size=(h * ss, w * ss), mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            # Working resolution and tile offset both scale, so the noise
            # lattice still resolves to the same global full-resolution
            # coordinates. Frame size scales with the working resolution
            # exactly as the tile offset does, so a normalised frame position
            # resolves the same.
            fh = None if full_hw is None else (full_hw[0] * ss, full_hw[1] * ss)
            r = self.render(up, p, scale * ss, y0 * ss, x0 * ss, fh)
            r = F.avg_pool2d(r, ss)

        # Cross-faded display-referred, where both images already live, rather
        # than round-tripping through linear. This is a compositing control --
        # "how much of the edit do I keep" -- not a physical average of light,
        # so the reasoning that puts `pre_blur` and halation in linear does not
        # carry over.
        #
        # The two are not interchangeable and the difference is not where you
        # would guess. Measured on a grained frame at half strength: mean
        # deviation 5.4e-04 and overall brightness within +0.05%, but a
        # worst-case 0.146 on individual pixels -- concentrated in the shadows,
        # where the transfer curve is steepest. That is exactly why encoded
        # wins here. Blending in the space the eye reads makes the slider
        # linear in *visible* deviation, so 0.5 is half the grain everywhere;
        # in linear the same 0.5 would take more than half out of the shadows
        # and less out of the highlights, which is an opacity control that
        # changes the look's balance as you dial it back. It also costs no
        # transfer round trip.
        if op < 1.0:
            r = img + (r - img) * op
        return r

    # ------------------------------------------------------------------ #
    def pad_for(self, p: dict, scale: float) -> int:
        """Overlap needed so a rendered region matches the full-image render.

        Must cover every blur kernel in the pipeline: the clarity high-pass at
        the very top, the high-pass chain, the
        acutance blur (the widest at 1.5x), the pre-blur and micro-blur, the
        edge-softening blur, the global-grain smoothing blur, the output
        sharpening blur and halation, plus the
        displacement of every stage that *reads* a pixel from somewhere else
        rather than blurring in place -- the jitter warp, the sanding taps and
        scatter. Miss one and tiled exports seam along its radius -- which no
        preview will ever show.
        """
        hp_r = max(0.3, p["highpass_radius"] * scale)
        # Pre-blur and micro-blur are two kernels in series, not alternatives:
        # micro-blur reads pixels the pre-blur has already spread, so their
        # reaches add rather than the widest winning.
        mb = (p["micro_blur"] + p["pre_blur"]) * scale
        # Clarity's high-pass and highlight reconstruction's neighbourhood are
        # the only two kernels in the whole colour-grading section -- the other
        # ten stages are per-pixel and reserve nothing. Both are real reaches
        # even though the section runs first, and for the same reason: what they
        # measure over their radius feeds a value that then propagates through
        # every stage below, so a tile that cannot see far enough is wrong from
        # the top of the pipeline down rather than only at its own border.
        #
        # Summed rather than the widest winning: they are stages in series, and
        # reconstruction runs *above* clarity, so clarity's band is measured on
        # pixels reconstruction has already changed from up to its own radius
        # away.
        clar = (
            p["grade_clarity_radius"] * scale
            if abs(p["grade_clarity"]) > 0.001 else 0.0
        )
        if p["grade_recover"] > 0.001:
            # Two kernels in series inside the one stage: the chromaticity
            # estimate reads `radius`, and its own weight field is then blurred
            # again by `radius * _RECON_ROLL_GATE_FRAC` to gate the roll. The
            # second reads pixels the first already spread, so they add.
            # Three kernels in series inside the one stage: the chromaticity
            # estimate reads `radius`, then its weight field is dilated by
            # `radius * _RECON_ROLL_GATE_FRAC` and feathered by the same again to
            # gate the roll. Each reads pixels the previous one already spread,
            # so all three add. The dilation is a hard reach like a warp rather
            # than a kernel, but it is counted in here with the others because it
            # sits between two blurs and the sum is what has to be covered.
            rr = max(1.0, p["grade_recover_radius"] * scale)
            clar += rr * (1.0 + 2.0 * _RECON_ROLL_GATE_FRAC)
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
        # Anti-aliasing reads two ways at once and both have to be counted:
        # its taps travel a radius along the tangent (a displacement, like the
        # warps below), and it derives that tangent -- and its step gate --
        # from blurred luma, which is a kernel.
        #
        # Both terms are multiplied by the pass count, for the reason sanding
        # documents below: each pass resamples the previous pass's output, so tap
        # travel accumulates, and each pass re-derives its direction from a fresh
        # blurred luma, so that reach accumulates too. Pinned at _AA_PASSES
        # rather than recomputed from the strength, because pad_for is called at
        # the un-supersampled scale and would otherwise disagree with the
        # renderer about the count.
        aa_r = 0.0
        aa_tap = 0.0
        if p["aa_strength"] > 0.001:
            aa_rad = max(0.2, p["aa_radius"] * scale)
            aa_r = _AA_PASSES * max(
                max(_AA_DIR_MIN, _AA_DIR_K * aa_rad), aa_rad * 1.5)
            aa_tap = _AA_PASSES * aa_rad
        # Global-grain smoothing is a blur on the noise field, so it reaches
        # like every other kernel here. It is gated on the layer being on --
        # with intensity or opacity at zero the field is never built.
        #
        # Referenced against the *effective* cell -- max(Min, Max) after the
        # same up-clamp `render()` applies -- not against Min alone. Above Min,
        # the field itself is built on a lattice pitched at Max, and the blur
        # is measured against that same reference, so a tile computed here
        # with only Min in view would under-reserve and the export would seam
        # exactly where Max exceeds Min.
        gsm = 0.0
        # Same floored min and up-clamped max render() computes, not the raw
        # slider values -- matched exactly, including the floor, so this gate
        # and render()'s own `variable` decision can never disagree about
        # whether the field switched construction.
        gcell_lo = max(_MIN_CELL, p["global_size"] * scale)
        g_eff = max(gcell_lo, p["global_size_max"] * scale)
        variable_gate = (g_eff - gcell_lo) > 1e-3
        if (p["global_intensity"] > 0.01 and p["global_opacity"] > 0.001
                and p["global_smooth"] > 0.001):
            gsm = p["global_smooth"] * _SMOOTH_MAX * g_eff
        # `_variable_cell_noise` reads lattice cells up to `_VARCELL_RINGS` away
        # from a pixel's own -- a real read reach, not merely an addressing
        # convenience, because the render pipeline only ever hands it a padded
        # window rather than the whole image. Under-reserve this and a pixel
        # near a tile edge silently substitutes a clamped boundary cell for the
        # true neighbour, which two different tile splits do differently --
        # invisible in a single preview, a seam in a tiled export.
        varcell_r = 0.0
        if (p["global_intensity"] > 0.01 and p["global_opacity"] > 0.001
                and variable_gate):
            varcell_r = _VARCELL_RINGS * g_eff
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
                3.0 * (hp_r * 3.3 + mb + clar + halo + soft + shr + tex_r
                       + mask_r + gsm + aa_r + varcell_r)
                + jit + aa_tap
            )
        ) + 4

    # ------------------------------------------------------------------ #
    def tile_for(
        self, p: dict, scale: float, h: int, w: int, ss: int,
    ) -> int:
        """Largest tile whose working set fits the memory budget.

        Tiling is pure overhead: `pad_for` overlap is read, rendered and thrown
        away on all four sides, so a smaller tile does strictly more work for the
        same output. Measured on a 2400x1600 `Stock` proxy at supersample 2,
        fresh process each, best of 3:

        | tile | tiles | overdraw | time  |
        |------|-------|----------|-------|
        | 1024 |   6   |  1.59x   | 4.46s |
        | 1536 |   4   |  1.32x   | 3.70s |
        | 2048 |   2   |  1.15x   | 3.30s |
        | 4096 |   1   |  1.00x   | 2.77s |

        Interior *export* tiles are the worst case, since they pad on all four
        sides: 1024 + 2*178 = 1380 square rendered for 1024 square kept, 1.82x.

        **So why not simply always use one tile?** Because memory is the binding
        constraint and it is the thing this codebase's own "quality beats speed"
        licence does not cover -- an out-of-memory render is not slow, it is
        broken. Peak driver-allocated memory on the sweep above went 6.0GB at
        tile 1536 to 8.0GB at 2048. On an 8GB machine that swaps or dies, and an
        8GB machine is exactly where the tiling matters most.

        Hence a budget rather than a constant. Note the coupling this creates
        with `pad_for`, which is the right one and which the old hard-coded 1024
        and 1536 got wrong in both directions: a wide-kernel preset pads more, so
        it *gets a smaller tile*, because its working set per tile is larger for
        the same nominal tile.

        `_WORKING_BYTES_PER_PX` is measured, not guessed -- see its comment. The
        answer is clamped into `_TILE_MIN`..`_TILE_MAX` and never exceeds what
        the image actually needs, so a small frame still renders in one pass.
        """
        pad = self.pad_for(p, scale)
        budget = _render_budget_bytes()
        ss = max(1, int(ss))
        longest = max(h, w)

        def fits(tile: int) -> bool:
            # The padded read window is *clamped to the image* (see
            # `render_image`), so the worst tile is bounded by the frame, not by
            # `tile + 2 * pad`. Solving the square upper bound in closed form
            # instead over-predicts badly once the tile approaches the image
            # size -- it wanted 2 tiles for a proxy that comfortably fits in one.
            th = min(h, min(tile, h) + 2 * pad)
            tw = min(w, min(tile, w) + 2 * pad)
            return (th * ss) * (tw * ss) * _WORKING_BYTES_PER_PX <= budget

        # Descending search rather than closed form: `fits` is monotonic in
        # `tile`, the candidate list is short, and this keeps the memory model in
        # one readable place instead of inverted through algebra.
        tile = _TILE_MIN
        for cand in range(min(_TILE_MAX, longest), _TILE_MIN, -128):
            if fits(cand):
                tile = cand
                break
        # Never below _TILE_MIN even if the budget says so: there the overlap
        # dominates the useful area so completely that the extra work costs more
        # than the memory it saves, and every supported backend can hold a tile
        # this size.
        tile = max(_TILE_MIN, tile)
        # No point in a tile larger than the image -- `render_image`
        # short-circuits to a single untiled pass when `tile >= max(h, w)`.
        return min(tile, longest)

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
        should_cancel=None,
    ) -> np.ndarray:
        """Render a whole image, tiling when it is larger than ``tile``.

        ``arr`` is HxWx3 float32 in 0..1. Returns the same shape.

        ``should_cancel``, if given, is polled once per tile and once before the
        single-tile path; returning true raises `RenderCancelled`. Tile
        granularity is deliberate: it needs no plumbing inside `render`, and it
        bounds the wasted work at one tile. It matters because the caller cannot
        interrupt this any other way -- a Starlette threadpool worker runs to
        completion whatever the client does, so an abandoned preview would
        otherwise keep the render lock for its full duration and every request
        behind it would queue on work nobody is waiting for.
        """
        # Nothing switched on: hand the input straight back. Not merely an
        # optimisation -- see params.is_neutral for why rendering it would
        # *not* return the input.
        if P.is_neutral(p):
            return arr
        if should_cancel is not None and should_cancel():
            raise RenderCancelled()
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
                if should_cancel is not None and should_cancel():
                    raise RenderCancelled()
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
