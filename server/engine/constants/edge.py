from __future__ import annotations


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

__all__ = [
    '_JITTER_MAX',
    '_SAND_TAPS',
    '_SAND_PASSES',
    '_SAND_DIR_K',
    '_SAND_MIN_GRAD',
    '_AA_TAPS',
    '_AA_PASSES',
    '_AA_DIR_K',
    '_AA_DIR_MIN',
]
