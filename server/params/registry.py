"""Lookups derived from ``PARAMS``, and the neutral/rescale helpers that
read them.
"""
from __future__ import annotations

from .definitions import PARAMS
from .param import Param


PARAM_BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}

DEFAULTS: dict[str, float] = {p.key: p.default for p in PARAMS}


# Every parameter that *does* something when raised. Setting all of these to
# zero makes render() a pass-through -- the pipeline is off and the output is
# the input. The rest are shapes, sizes, radii and seeds: they describe how a
# stage behaves, not whether it runs, so they stay at their defaults where they
# are harmless and remember what you had dialled in.
#
# Kept as an explicit list rather than inferred: "is this an amount?" is not
# something the schema can work out, and a stage silently missing from here
# would leave the Original button showing a not-quite-original image, which is
# a worse failure than any of them being over-zealous. `verify.py` renders with
# these and asserts the output is the input.
NEUTRAL_ZERO: tuple[str, ...] = (
    # Colour grading. `lut_amount` belongs here and the LUT *name* deliberately
    # does not: this list is what "Original" applies, and it has to be a set of
    # numbers the engine can be handed. Zeroing the mix switches the LUT off as
    # completely as unselecting it would, so the name can stay put and be there
    # again when the section is switched back on -- the same reasoning that
    # keeps sizes, radii and seeds out of this list. `grade_clarity_radius` and
    # `grade_recover_radius` are radii, not amounts, so they stay out for the
    # same reason -- as does `grade_black_point`'s partner-in-spirit `base_fog`
    # below.
    "grade_recover",
    "grade_temp", "grade_tint", "grade_exposure", "grade_shadows",
    "grade_highlights", "grade_contrast", "grade_black_point", "grade_clarity",
    "grade_vibrance", "grade_saturation", "lut_amount",
    "pre_blur", "pre_sharpen",
    "contrast", "toe", "shoulder", "highlight_desat", "brightness",
    "vibrance", "base_fog",
    "intensity",
    "edge_erosion", "acutance", "edge_soften", "edge_sand", "edge_jitter",
    "halation", "halation_blue", "halation_recovery",
    "micro_blur", "scatter", "aa_strength",
    # Both directions of these are an effect, so it is the *magnitude* that has
    # to be zero -- `is_neutral` takes an absolute value, which is why a
    # bidirectional control can live in this list at all.
    "highlight_warmth", "shadow_warmth",
    "global_intensity",
    "global_src_r", "global_src_g", "global_src_b", "global_src_l",
    "sharpen",
    "dust", "scratches", "hair", "light_leak",
)


def neutral_values() -> dict[str, float]:
    """Values that switch every stage off, leaving the image untouched."""
    out = dict(DEFAULTS)
    for k in NEUTRAL_ZERO:
        if k in out:
            out[k] = 0.0
    return out


def rescale(values: dict[str, float], k: float) -> dict[str, float]:
    """Rescale a value set authored at one image size for another.

    ``k`` is the ratio of *linear* dimensions, not of pixel counts. That
    distinction is the whole thing: every parameter marked ``spatial`` is a
    length in full-resolution pixels, and a 16MP frame is 0.816x the width of a
    24MP one, not 0.667x. Scaling lengths by the megapixel ratio overshoots by
    the square root -- a 2px clump would come out at 1.33px instead of 1.63px,
    and at the other end a 40MP frame would get 3.3px clumps where it wants
    2.6px.

    Deliberately *not* rescaled:

    * Amounts and blend weights (intensity, halation, sharpen, vibrance...).
      They are per-pixel and dimensionless, so the same number means the same
      thing at any size.
    * Mark counts (dust, scratches, hair, leaks). Those already resolve against
      the frame's area inside the engine, so 50 specks is 50 specks whatever
      the resolution -- which is what keeps the look constant.
    * Discrete choices (``scatter_pattern``). It is an index into a list of
      stencils, not a quantity -- scaling it would silently swap the shape.

    Leak sizes and the leak feather *are* rescaled, because they became
    lengths in pixels. They used to be fractions of the frame and so were
    exempt; a preset written against the old fraction will read its number as
    pixels and produce a hairline leak, which is why the shipped ones were
    migrated in place.

    Values are clamped back into range afterwards, so a large upscale can
    saturate a parameter rather than escaping its slider.
    """
    if abs(k - 1.0) < 1e-6:
        return dict(values)
    out = dict(values)
    for prm in PARAMS:
        if not prm.spatial or prm.key not in out:
            continue
        out[prm.key] = max(prm.min, min(prm.max, out[prm.key] * k))
    return out


def scale_factor(reference_mp: float | None, current_mp: float) -> float:
    """Linear scale between a preset's authored size and the current image."""
    if not reference_mp or reference_mp <= 0 or current_mp <= 0:
        return 1.0
    return float((current_mp / reference_mp) ** 0.5)


def is_neutral(p: dict) -> bool:
    """True when no stage is active, so a render would return its input.

    Worth testing for rather than just rendering: the supersample round trip
    is *not* itself a pass-through -- a bicubic upsample followed by a box
    downsample softens hard edges, measured at 1.0e-01 max deviation -- so a
    render with every stage off still comes back visibly softer than the
    original at 2x. Callers short-circuit on this so "show me the original"
    means the original, whatever the quality setting.
    """
    return all(abs(float(p.get(k, 0.0))) < 1e-9 for k in NEUTRAL_ZERO)
