from __future__ import annotations

from .core import _LUMA

# -- split toning ---------------------------------------------------------- #
# The warm/cool axis Highlight Warmth and Shadow Warmth push along, and the
# amplitude of a full-strength push.
#
# **The axis is projected onto the luma-null plane, and that is the whole
# construction.** The raw direction below is a warm shift -- red up, a little
# green, blue down -- and pushing a pixel along it as written *also brightens
# it*, because its luma is 0.248 rather than 0. So warming the highlights would
# lift them as well, fighting Shoulder and Brightness for the same range and
# making the two controls impossible to set independently. Subtracting the axis'
# own luma from every channel lands it exactly on the plane where the luma
# weights sum to zero, so the shift is a pure change of colour at every setting
# and in both directions. `_WARM_AXIS` below is that projection, normalised so
# its largest component is 1.
#
# **Amplitude, and why it is nearly three times what it replaced.** This was
# `warm_highlights` and `cool_shadows`, two 0..1 sliders adding a fixed
# [0.055, 0.012, -0.040] and [-0.030, 0.002, 0.050]. The user reported both as
# doing nothing visible, and the arithmetic agrees: the peak shift was 0.055 in
# one channel, and the weighting only reached 1.0 at pure white, so an ordinary
# highlight at luma 0.7 got 0.019 -- under two 8-bit levels, which is a
# rounding error and not a look. At 0.14 a full-strength push moves the blue
# channel by 36 levels at the top of the range, which is a visible cast without
# being a colour filter; the pair at opposite signs is a split tone you can see
# at a glance and still dial back to nothing.
_WARM_RAW = (1.0, 0.15, -1.0)
_WARM_NULL = tuple(
    c - sum(k * v for k, v in zip(_LUMA, _WARM_RAW)) for c in _WARM_RAW
)
# Normalised on the largest component rather than on the vector's length, so
# `_WARM_GAIN` reads directly as "how far the worst-shifted channel moves".
_WARM_AXIS = tuple(c / max(abs(v) for v in _WARM_NULL) for c in _WARM_NULL)
_WARM_GAIN = 0.14

# Where the two weightings reach full strength. They overlap through the
# mid-tones deliberately -- disjoint bands leave an untinted stripe across the
# middle of the range, so setting both sliders the same way would tint the top
# and the bottom of a gradient and miss its centre. Widened from the old
# (0.45, 1.0) / (0.0, 0.5): those only reached full weight at pure white and
# pure black, so most of a real photograph took a fraction of the setting.
_WARM_HI_BAND = (0.30, 0.85)
_WARM_LO_BAND = (0.15, 0.70)

__all__ = [
    '_WARM_RAW',
    '_WARM_NULL',
    '_WARM_AXIS',
    '_WARM_GAIN',
    '_WARM_HI_BAND',
    '_WARM_LO_BAND',
]
