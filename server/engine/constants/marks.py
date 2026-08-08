from __future__ import annotations


# Floor on a mark's per-mark brightness multiplier. Marks vary in density from
# this to full; taking it to zero would just delete marks rather than vary them,
# which thins the population instead of making it look weathered.
_TEX_LUM_FLOOR = 0.25

# -- dust ------------------------------------------------------------------ #
# Every constant below describes one speck's *shape*. See `_dust_sites` for why
# dust is drawn one speck at a time rather than thresholded out of a field.
#
# Eccentricity: a speck's two semi-axes are `r * (1 +- e)` with `e` drawn up to
# this. **Not zero, and not much larger.** A population of exact circles is the
# single clearest tell that a texture was generated -- real debris is a chip or
# a fibre-end seen at some angle, so it is a little oval and pointing somewhere.
# Past about 0.4 the specks start reading as short scratches instead.
_DUST_ECCENT = 0.35

# Amplitudes of the 3rd, 4th and 5th angular harmonics perturbing the ellipse's
# radius, each with its own random phase. This is what "imperfect" means here
# and it is deliberately built on top of the ellipse rather than instead of it:
# the 2nd harmonic *is* an elongation, so it would only fight the eccentricity
# draw above, where 3-5 dent the outline without changing its overall shape.
#
# **These are the amplitudes at `dust_irregular` = 1, not the amplitudes that
# ship** (rewritten 2026-08-08). They used to be unconditional and summed to
# 0.22, so every speck was a little lumpy and nothing could ask for a clean one;
# the report was the other way round -- "dust is not rounded" -- so the whole
# perturbation is now scaled by the slider and the slider ships at 0. At 0 the
# radius is exactly 1 and the outline is exactly the ellipse.
#
# The sum is the number that matters -- the radius is
# `1 + sum(a_k cos(k phi + p_k))`, so a sum at or above 1 can fold the outline
# through its own centre and draw a shape with a bite out of it. 0.53 here times
# the top of `_DUST_ROUGH_SPREAD` is 0.795, which is the worst case any setting
# can reach: a deep notch, still a single closed outline, never a fold.
_DUST_HARMONICS = (0.24, 0.17, 0.12)

# Per-speck multiplier on that amplitude, so a frame at one irregularity setting
# is a *population* of shapes rather than one lumpiness stamped out N times --
# the same argument `_DUST_SIZE_SPREAD` makes about diameter. The floor is not 0
# because a speck with no perturbation at all next to one with a notch in it
# reads as two different textures mixed rather than as debris.
_DUST_ROUGH_SPREAD = (0.45, 1.5)

# Spread of speck diameter about `dust_size`, as a multiplier range. Real debris
# does not come in one size, and drawing every speck at the slider's exact value
# reads as a stamped population. Geometric-ish rather than symmetric, so the
# mean stays near 1.
_DUST_SIZE_SPREAD = (0.55, 1.55)

# Edge width of a speck at Dust Softness 0, as a fraction of its own radius, and
# the width reached at Softness 1. The floor is not decoration: a hard analytic
# edge aliases at any speck size, and it is what supersampling is left to clean
# up when it is too tight to resolve.
#
# `_DUST_EDGE_MAX` is no longer a ceiling -- Dust Softness runs to 5 as of
# 2026-08-08 and the mapping stays linear past 1, so the edge reaches 3.85 radii
# at the top. The slider used to clamp there and the complaint was that it was
# too weak; the clamp was the reason. Past an edge of 1 the inner smoothstep
# bound goes negative, which is not a bug but the point: the speck stops having
# a solid core and becomes the diffuse smudge that badly out-of-focus debris
# actually is.
_DUST_EDGE_MIN = 0.10
_DUST_EDGE_MAX = 0.85

# Absolute floor on that edge, in working pixels. A 1px speck's 10% edge is a
# hundredth of a pixel, which is a hard step in the output whatever the analytic
# profile says. Half a pixel is the smallest edge the grid can carry.
_DUST_EDGE_PX = 0.5

# The narrowest half-width the pixel grid can carry, in working pixels. Below
# it a mark is drawn at this width and *faded* by how much of it is really
# there, instead of being drawn thinner.
#
# **This is not a nicety, it is the difference between a hair and a dashed
# line.** A filament narrower than a pixel only registers where its centre
# happens to pass near a pixel centre, so it renders as a row of dots with gaps
# between them -- which is exactly what a hair's tapered tip did before this
# existed: measured, one hair came out as a 394-pixel filament plus a detached
# one-pixel speck at its end. Fading by the area (or, for a filament, the width)
# that fell below the floor is what area-averaging would have done anyway, so
# the mark thins the honest way: it gets fainter, not dotted.
_MARK_MIN_PX = 0.5

# How much of a speck's opacity softness takes away. Out-of-focus debris really
# is both softer and fainter -- the same light is spread over a wider footprint
# -- and leaving this at 0 makes Dust Softness read as "the specks got bigger".
#
# Applied over the *first unit* of softness only, and that matters now that the
# slider runs to 5. Extended linearly it crosses zero at 2.2 and the specks would
# simply vanish somewhere in the top half of the slider's travel; past 1 the
# widening profile dims its own peak anyway, so the geometry carries the fade
# from there and this does not need to.
_DUST_SOFT_FADE = 0.45

# Luminosity ranges the two populations spread across: opaque motes from black
# to mid-grey, pinholes and lint from off-white to white. `dust_lum_var` spreads
# each about its own midpoint, so a population varies within itself without the
# two ever swapping places.
_DUST_DARK_LUM = (0.0, 0.42)
_DUST_LITE_LUM = (0.72, 1.0)

# -- hair ------------------------------------------------------------------ #
# Filament width at full resolution, in pixels, before the per-hair draw. A hair
# is about this on a 24MP scan; the value is inherited from the level-set
# construction this replaced, where it had to be *solved* for rather than picked
# (a level set is `2 * eps * cell` wide, and the first attempt at 0.35px drew
# literally nothing). Drawn directly now, so the width is simply the width.
_HAIR_WIDTH = 1.6
_HAIR_WIDTH_SPREAD = (0.7, 1.5)

# Spread of hair length about `hair_length`, as a multiplier range.
_HAIR_LEN_SPREAD = (0.65, 1.4)

# How far a hair bends over its own length: the quadratic sag and the two
# sinusoidal wobbles, all as fractions of the half-length. A hair lies in a
# curve, and a straight one reads as a scratch -- which is the other mark type,
# and the two must not converge.
_HAIR_CURVE = 0.45
_HAIR_WOBBLE = (0.18, 0.07)

# Ceiling on each wobble's steepest lateral slope. See `_hair_sites`: a wobble
# steep enough to double back within a pixel breaks the perpendicular-distance
# approximation the renderer draws the filament with, and the hair comes out in
# pieces. Capping the slope rather than the amplitude lets a slow wobble be
# wide and forces a fast one to be shallow, which is what a fibre does anyway.
# With the quadratic sag's own 2 * `_HAIR_CURVE` this holds the total under 1.8.
_HAIR_SLOPE = (0.55, 0.30)

# Where the taper starts, as a fraction of the half-length, and how thin the tip
# gets. A real fibre comes to a point; a filament of constant width with two
# blunt ends reads as a line segment somebody drew.
_HAIR_TAPER = 0.55
_HAIR_TIP = 0.15

# Luminosity range a hair composites toward. A hair on the glass is opaque, so
# it prints near black -- but not *at* black, or every hair is the same hair.
_HAIR_LUM = (0.02, 0.30)
_HAIR_ALPHA = (0.45, 1.0)

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
# measurement wins. Tuned against delivered counts on a 1.5MP frame, and
# accurate to roughly a factor of 1.5 across the range -- this is a count you
# steer by, not a guarantee.
#
# **Scratches are the only mark type left that needs one.** Dust and hair used
# to have their own (14.0 and 0.5) and the calibration was never shared, because
# a compact speck and a level-set filament turn a given coverage fraction into
# quite different numbers of countable marks. Both are drawn from lists now and
# their counts are exact, so the constants went with the construction.
_BLOB_CELLS_SCRATCH = 26.0

# -- light leaks ---------------------------------------------------------- #
# A leak is a *shaft* of light past an obstruction, so it is drawn as a small
# number of discrete oriented beams anchored on the perimeter, not as a wash
# gated along the whole border. See `_leak_sites` for why a list of them does
# not break tile independence the way a list of dust specks would.

# Step used to place leaks around the perimeter, as a fraction of it. Golden
# ratio, i.e. a low-discrepancy sequence rather than a stratification, so leak
# k lands in the same place whatever the count is -- raising the count must add
# a leak, not reshuffle the ones already on the frame.
# Reciprocal powers of the plastic number: the 2-D low-discrepancy step
# `_mark_spread` places dust and hair on, and the direct analogue of the
# golden-ratio step `_leak_sites` places leaks on one dimension with.
_R2_A1 = 1.0 / 1.32471795724474602596
_R2_A2 = _R2_A1 * _R2_A1

# How far a mark jitters off its low-discrepancy slot, as a fraction of the
# frame. Small on purpose, for `_leak_sites`' reason: the sequence already
# spreads the marks, and a large jitter only lets two of them land on top of
# each other. At the top of the dust count the R2 spacing is 0.05, so this is
# larger than the spacing and the placement goes locally random; at a count of
# three it is far smaller and the sequence wins.
_MARK_JITTER = 0.06

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

__all__ = [
    '_TEX_LUM_FLOOR',
    '_DUST_ECCENT',
    '_DUST_HARMONICS',
    '_DUST_SIZE_SPREAD',
    '_DUST_EDGE_MIN',
    '_DUST_EDGE_MAX',
    '_DUST_EDGE_PX',
    '_MARK_MIN_PX',
    '_DUST_SOFT_FADE',
    '_DUST_DARK_LUM',
    '_DUST_LITE_LUM',
    '_HAIR_WIDTH',
    '_HAIR_WIDTH_SPREAD',
    '_HAIR_LEN_SPREAD',
    '_HAIR_CURVE',
    '_HAIR_WOBBLE',
    '_HAIR_SLOPE',
    '_HAIR_TAPER',
    '_HAIR_TIP',
    '_HAIR_LUM',
    '_HAIR_ALPHA',
    '_NOISE_ICDF',
    '_BLOB_CELLS_SCRATCH',
    '_R2_A1',
    '_R2_A2',
    '_MARK_JITTER',
    '_LEAK_PHI',
    '_LEAK_CORNER_BIAS',
    '_LEAK_WARP',
    '_LEAK_REACH_SAFETY',
    '_LEAK_GAIN',
]
