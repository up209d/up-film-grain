"""Step -2: auto exposure, auto white balance and range compression.

The only stage in the pipeline whose settings are *measured from the
photograph* rather than dialled in, which is the whole reason it is written
in two halves that live in one file:

``meter`` runs **once per upload**, on the whole frame, on the CPU, in numpy,
and returns six plain floats. ``_normalize`` runs **per tile**, on the device,
and is pure per-pixel arithmetic on those six numbers.

That split is not organisational, it is invariant 1. A stage that measured
the region it was handed would meter every tile against its own crop, and a
tiled export would come apart at the seams while every preview looked
perfect -- `pad_for` can reserve a finite reach and a whole-image statistic
is an infinite one. Measuring the *frame* once and handing every tile the
identical numbers is the same carve-out the light-leak site list and the
dust and hair mark lists already use.

The two halves must also stay in step across *tiers*: the proxy preview and
the 1:1 export are handed the same cached floats, so they normalise
identically. Metering each tier separately would quietly break "export what
I am looking at".
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..colour import _linear_to_srgb, _srgb_to_linear, _tone_roll
from ..constants.core import _LUMA
from ..primitives import _smoothstep
from ..constants.normalize import (
    _NORM_EV_MAX, _NORM_HP_HI, _NORM_HP_LO, _NORM_MAX_SAMPLES,
    _NORM_TARGET_LIN, _NORM_TOE_KNEE, _NORM_TONE_MAX, _NORM_VALID_HI,
    _NORM_VALID_LO, _NORM_WB_DIV_HI, _NORM_WB_DIV_LO, _NORM_WB_MAX, _NORM_WB_P,
)

#: What `meter` returns when it declines to correct, and what `_normalize`
#: treats as "do nothing". Named rather than repeated so the identity is one
#: fact: a frame with nothing measurable in it and a frame with the stage
#: switched off must produce byte-identical output.
NORM_IDENTITY: dict[str, float] = {
    "norm_ev": 0.0,
    "norm_gain_r": 1.0,
    "norm_gain_g": 1.0,
    "norm_gain_b": 1.0,
    "norm_toe": 0.0,
    # 1.0 rather than 0.0: it is a *white point*, and at 1.0 the tone map is
    # algebraically the identity. Zero would be a division by nothing.
    "norm_white": 1.0,
}

#: Below this many valid samples the frame is not metered at all. A photograph
#: that is almost entirely clipped or almost entirely black has no trustworthy
#: measurement in it, and inventing one from a handful of surviving pixels is
#: how an auto control produces confident nonsense.
_MIN_SAMPLES = 256


def _to_linear_np(x: np.ndarray) -> np.ndarray:
    """sRGB -> linear, in numpy. The exact curve `colour._srgb_to_linear` uses.

    Duplicated in numpy rather than run through torch because the metering is a
    once-per-upload CPU pass over a whole frame and has no business allocating
    device memory. The two must agree, and `verify.py` pins that they do.
    """
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb_np(x: np.ndarray) -> np.ndarray:
    """linear -> sRGB, in numpy. The inverse of `_to_linear_np`."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def meter(arr: np.ndarray) -> dict[str, float]:
    """Measure a frame and return the correction it needs, as six floats.

    ``arr`` is HxWx3 float in 0..1, display-referred -- the whole photograph,
    not a tile. Called once per upload and cached; see `models/upload.py`.

    Every measurement here is deliberately *robust* rather than exact, because
    the failure mode of an auto control is not being slightly off, it is being
    confidently wrong on the photograph where the estimator's assumption does
    not hold. Each of the three has its own guard and each degrades toward
    doing nothing.
    """
    flat = arr.reshape(-1, 3).astype(np.float64, copy=False)

    # A uniform stride rather than a random sample: the same photograph has to
    # produce the same correction every time it is opened, or re-uploading a
    # file would silently regrade it.
    stride = max(1, flat.shape[0] // _NORM_MAX_SAMPLES)
    s = flat[::stride]

    # Validity is per *pixel*, not per channel, and shared by both estimators.
    # A pixel with one channel on the ceiling is not a sample of the local
    # colour at all -- its ratios are set by the clip, not by the light -- so
    # letting it into the white balance drags the result toward the clipped hue.
    # The same shared-mask reasoning highlight reconstruction had to be fixed to
    # use: two estimators averaging over *different* sets of pixels compare
    # means from different neighbourhoods and the ratio between them is wrong.
    mx = s.max(axis=1)
    mn = s.min(axis=1)
    dark = mn < _NORM_VALID_LO
    valid = (mx <= _NORM_VALID_HI) & ~dark
    v = s[valid]

    # **Exposure gets a different mask, and the asymmetry is a bug fix rather
    # than a refinement.** Excluding clipped pixels is right for colour and
    # exactly wrong for level: a blown frame is mostly clipped, so the surviving
    # samples are its *dark* pixels, and metering the log average over those
    # says "this photograph is dark, brighten it". Measured on a frame
    # deliberately over-exposed by 1.4 stops, the shared mask asked for
    # **+1.38 stops brighter** -- the correct answer with the sign inverted, on
    # the one input the control exists to fix.
    #
    # So the top end stays in. A clipped pixel understates its own luminance,
    # which biases the key down a little; dropping it biases the key down by the
    # whole bright half of the picture. The bottom end still goes, because a log
    # average is hypersensitive to values near zero and a letterbox border or a
    # black backdrop would otherwise drag the key down on its own.
    ve = s[~dark]

    # **The two guards are separate, and folding them into one was a real hole
    # in exactly this stage's target case.** A frame blown across almost all of
    # itself leaves very few pixels inside the colour window, and a single
    # early return keyed on that count abandoned the *exposure* correction with
    # it -- on the most over-exposed input there is, which is precisely what
    # this control exists for. The exposure mask is far larger on that frame
    # and has plenty to work with.
    #
    # So each half degrades on its own: too little trustworthy colour means no
    # white balance, too little trustworthy anything means no correction at all.
    luma = np.asarray(_LUMA, dtype=np.float64)
    if ve.shape[0] < _MIN_SAMPLES:
        return dict(NORM_IDENTITY)

    # ---- white balance -------------------------------------------------- #
    gain = np.ones(3, dtype=np.float64)
    if v.shape[0] >= _MIN_SAMPLES:
        lin = _to_linear_np(v)
        # Minkowski p-norm: grey-world at p=1, white-patch at p=inf, and
        # neither is usable alone. See `_NORM_WB_P`.
        m = np.power(np.mean(np.power(lin, _NORM_WB_P), axis=0), 1.0 / _NORM_WB_P)
        m = np.maximum(m, 1e-6)
        gain = float(m.mean()) / m

        # How much do the hues in this frame actually vary? A real cast shifts
        # every hue together and leaves the spread intact; a scene that is
        # legitimately one colour -- a sunset, blue hour, a close-up of red
        # fabric -- has no spread to begin with and is indistinguishable from a
        # cast by the channel means alone. Measured on each pixel's own colour,
        # normalised by its own brightness so the answer is about hue rather
        # than exposure.
        tot = lin.sum(axis=1, keepdims=True)
        chroma = lin / np.maximum(tot, 1e-6)
        diversity = float(np.mean(np.abs(chroma - chroma.mean(axis=0))))
        damp = (diversity - _NORM_WB_DIV_LO) / (_NORM_WB_DIV_HI - _NORM_WB_DIV_LO)
        damp = min(1.0, max(0.0, damp))
        gain = 1.0 + (gain - 1.0) * damp

        # Clamp each channel before the luma normalisation, then normalise -- in
        # that order. A clamp applied afterwards would break the luma neutrality
        # it is meant to preserve, which is the "two controls fighting" failure
        # the split tone already taught this codebase once.
        gain = np.clip(gain, 1.0 / _NORM_WB_MAX, _NORM_WB_MAX)
        gain = gain / max(float(np.dot(luma, gain)), 1e-6)

    # ---- exposure ------------------------------------------------------- #
    # The *log* average, not the mean. The mean of a frame with a bright sky in
    # it is the sky; the log average is dominated by the bulk of the tones,
    # which is what "how bright is this photograph" means to a person.
    #
    # Metered after white balance because that is the frame the exposure will
    # actually be applied to -- the gain above is luma-neutral by construction
    # so this is a small correction, but "small" is not "zero" and the two
    # halves are cheaper to reason about when neither depends on the other's
    # error.
    lum = np.maximum(_to_linear_np(ve) * gain, 0.0) @ luma
    key = math.exp(float(np.mean(np.log(np.maximum(lum, 1e-6)))))
    ev = math.log2(_NORM_TARGET_LIN / max(key, 1e-6))
    ev = min(_NORM_EV_MAX, max(-_NORM_EV_MAX, ev))

    # ---- highlight compression ------------------------------------------- #
    #
    # **Rewritten 2026-08-16 because the first version destroyed highlights**,
    # which is the one thing the user said mattered most. It applied the gain and
    # then squashed whatever came out above a fixed knee at 0.82. Measured on a
    # real photograph needing +2 EV, the source band 0.70..1.00 -- 77 8-bit
    # levels of highlight -- came out as **3.2 levels**. Everything above source
    # 0.5 rendered as white.
    #
    # Two things were wrong and neither was the curve's shape:
    #
    # * **A fixed knee is in the wrong place for a large lift.** At +2 EV every
    #   source value above 0.29 already lands above 0.82, so 70% of the tonal
    #   range had to be crammed into 18% of the output. No shoulder shape
    #   survives that.
    # * **Sizing from the frame's true maximum let clipped pixels set the
    #   curve.** 0.84% of that photograph was already blown -- flat white,
    #   carrying no detail at all -- and those pixels forced the compression for
    #   the 99.16% that still had something in them.
    #
    # The replacement is the extended Reinhard tone map, in **linear light**,
    # with the frame's own gained maximum as its white point:
    #
    #     y = x * (1 + x / Lw**2) / (1 + x)
    #
    # It compresses gradually across the whole top end instead of gating at a
    # knee, so the same photograph keeps **35.3 of those 77 levels** -- 11x the
    # old figure. Three properties, none of them tuned:
    #
    # * `Lw` maps to exactly 1.0, so the brightest pixel lands on white and
    #   nothing exceeds it. Already-clipped pixels *should* come out white; what
    #   was wrong before was letting them drag everything else with them.
    # * **At `Lw = 1` it is algebraically the identity** -- `x(1+x)/(1+x)`. A
    #   frame that already fits is untouched, with no special case and no knee.
    # * It is strictly increasing, so ordering still cannot be lost.
    #
    # This is why there is no `_NORM_SHOULDER_KNEE` or shoulder amount any more:
    # one measured white point replaced a knee, a target and a fitted strength.
    ch_max = flat.max(axis=0)
    white = float(np.max(_to_linear_np(ch_max)) * np.max(gain) * (2.0 ** ev))
    white = max(1.0, white)

    # The roll pulls the mid-tones down a little on its way past, so the gain is
    # corrected once against where the key *actually* lands rather than where it
    # would have landed without it. One pass, not a solve: the roll is nearly
    # linear down here (the correction measures under 2%), and iterating a
    # fixed point to convergence would be answering a question nobody asked.
    keyed = key * (2.0 ** ev)
    rolled = keyed * (1.0 + keyed / (white * white)) / (1.0 + keyed)
    if rolled > 1e-6:
        ev += math.log2(max(1e-6, _NORM_TARGET_LIN / rolled))
        ev = min(_NORM_EV_MAX, max(-_NORM_EV_MAX, ev))
        white = max(1.0, float(np.max(_to_linear_np(ch_max)) * np.max(gain)
                               * (2.0 ** ev)))

    # The toe answers "how much shadow separation did *this correction* cost",
    # which is a different question from "how dark is this photograph" -- and
    # getting those two confused is what the first version did. Sized from the
    # frame's own black level, a well-exposed picture with genuine deep shadows
    # measured a toe of 0.216 and had its blacks lifted for no reason: the
    # photograph was fine, the deep shadows were the photographer's, and the
    # control was washing them out to fix a problem that did not exist.
    #
    # Darkening by `ev` stops compresses everything below the knee by exactly
    # `2**ev`, so the cost is a function of the correction alone. Brightening
    # cannot crush a shadow -- it moves tones *away* from black -- so the toe is
    # one-directional and a frame that needs no exposure change gets none of it,
    # which is what keeps "an already-good photograph comes back untouched"
    # true at the bottom end as well as the top.
    toe = min(1.0, max(0.0, -ev / _NORM_EV_MAX)) * _NORM_TONE_MAX

    return {
        "norm_ev": float(ev),
        "norm_gain_r": float(gain[0]),
        "norm_gain_g": float(gain[1]),
        "norm_gain_b": float(gain[2]),
        "norm_toe": float(toe),
        "norm_white": float(white),
    }


class NormalizeMixin:
    """Step -2: the measured correction, applied per pixel.

    Above Colour Grading and therefore above everything. Ships off, so with
    nothing selected the pipeline is still a pass-through.
    """

    @staticmethod
    def _normalize(img: torch.Tensor, p: dict) -> torch.Tensor:
        """Apply the metered correction. ``img`` is [1,3,h,w] display-referred.

        Pure per-pixel arithmetic on six constants, so it reserves **nothing**
        in `pad_for` and is tile-independent by construction. No ``scale``
        argument for the same reason -- there is no length in here to scale.

        The six floats are read with ``.get`` rather than a direct subscript,
        which is the one place this stage departs from the house rule that
        `sanitize` guarantees every key. They are not `Param`s: they are
        measured from the image and attached in `models/upload.py:params_for`,
        beside `p["lut"]` and for the same reason -- a measurement is not a
        quantity a user dials. A caller that never went through `params_for`
        (every check in `verify.py`, for one) therefore gets the identity, which
        is both correct and what makes the stage testable with injected numbers.
        """
        if p.get("normalize", 0.0) < 0.5:
            return img

        ev = float(p.get("norm_ev", 0.0))
        gr = float(p.get("norm_gain_r", 1.0))
        gg = float(p.get("norm_gain_g", 1.0))
        gb = float(p.get("norm_gain_b", 1.0))
        toe = float(p.get("norm_toe", 0.0))
        white = float(p.get("norm_white", 1.0))
        # The one *dialled* value in the stage. Everything else here was
        # measured from the photograph; this is the user settling a trade the
        # measurement cannot settle for them.
        hp = float(p.get("highlight_priority", 0.0))

        # Kept before anything touches it: Highlight Priority blends back toward
        # what the file actually recorded, so the untouched frame has to survive
        # to the bottom of the method. Free -- torch does not copy here, and the
        # correction below rebinds `img` rather than writing into it.
        src = img

        # -2a. White balance, exposure and the highlight roll, in one
        #      linear-light round trip.
        #
        #      In linear because both are properties of *light*: a white balance
        #      is a change of illuminant and an exposure is a change of how much
        #      of it reached the film, and gamma-encoded values are not light.
        #      Done encoded, the same gain moves the shadows much further than
        #      the highlights, which is what makes a naive correction read as a
        #      wash laid over the picture rather than a different lamp.
        #
        #      One round trip rather than two, the way Temperature and Tint
        #      share theirs: they are one physical operation resolved into two
        #      factors, so paying the transfer cost twice would buy nothing.
        #
        #      **The roll happens inside this round trip, not after it**, which
        #      is the correction to the first version. Compressing display-
        #      referred values above a fixed knee meant a large lift pushed most
        #      of the picture past that knee before the compression ever saw it,
        #      and 77 8-bit levels of highlight came out as 3. In linear the
        #      tone map is operating on light, where the excess actually is.
        if abs(ev) > 1e-6 or abs(gr - 1.0) > 1e-6 or abs(gg - 1.0) > 1e-6 \
                or abs(gb - 1.0) > 1e-6 or white > 1.0 + 1e-6:
            gain = torch.tensor(
                [gr, gg, gb], device=img.device, dtype=img.dtype,
            ).view(1, 3, 1, 1)
            lin = _srgb_to_linear(img) * gain * (2.0 ** ev)

            # The extended Reinhard tone map, keyed on the channel maximum and
            # applied as one scale to all three. Keying on the max holds hue
            # *exactly* -- a uniform scale cannot move a ratio -- where a
            # per-channel roll would desaturate every highlight it touched.
            #
            # `white` is the frame's own gained maximum, measured once by
            # `meter`, so it maps to exactly 1.0 and nothing can exceed it: every
            # other channel is at or below the max and comes down by the same
            # factor. At `white == 1.0` the expression is algebraically `x`, so
            # a frame that already fits passes through untouched -- which is why
            # this needs no knee, no target and no fitted strength, and why the
            # three constants those took are gone.
            if white > 1.0 + 1e-6:
                v = lin.amax(dim=1, keepdim=True)
                vt = v * (1.0 + v / (white * white)) / (1.0 + v)
                lin = lin * (vt / v.clamp_min(1e-6))

            img = _linear_to_srgb(lin)

        # -2c. The toe: shadow separation kept when the frame had to be darkened.
        #
        #      The mirror of the shoulder, and it exists for a different failure.
        #      Nothing clips at the bottom -- a positive gain moves away from 0
        #      and cannot cross it -- so in float the information is all still
        #      there. What it loses is *levels*: darken by two stops and a shadow
        #      at 0.10 lands at 0.03, which is eight 8-bit codes where it had
        #      forty, and the export is where that becomes permanent.
        #
        #      Capped at `_NORM_TONE_MAX` where the highlight roll is not, and
        #      that asymmetry is the "a bit of HDR, not a log profile" decision
        #      made concrete: a lifted black point is the single most
        #      recognisable thing about a log picture, while a rolled highlight
        #      is what every film stock already does.
        if toe > 0.001:
            v = img.amax(dim=1, keepdim=True)
            u = ((_NORM_TOE_KNEE - v) / _NORM_TOE_KNEE).clamp_min(0.0)
            vt = torch.where(
                v < _NORM_TOE_KNEE,
                _NORM_TOE_KNEE - _NORM_TOE_KNEE * _tone_roll(u, toe),
                v,
            )
            img = img * (vt / v.clamp_min(1e-4))

        # -2d. Highlight Priority: hand the bright end back to the original file.
        #
        #      The one control here the user sets rather than the image, and it
        #      exists because the trade underneath it is real rather than a bug.
        #      Lifting a dark frame's mid-tones leaves the bright end nowhere to
        #      go -- it is already near white in the file -- so the tone map has
        #      to compress it, and compressed highlights lose the fine separation
        #      that reads as texture. No curve escapes that; something between
        #      the lifted mids and the fixed ceiling has to give.
        #
        #      So this does not try to be cleverer, it lets you choose: weight
        #      each pixel by how bright it was **in the source** and blend that
        #      far back toward the source's own value. At 1 the highlights come
        #      back at their original tonal spacing -- measured on a real
        #      photograph lifted two stops, all 256 of the bright region's
        #      distinct levels, against 250 at 0 -- while the mid-tones keep most
        #      of the correction (0.280 against the source's 0.172).
        #
        #      Keyed on the *source* rather than the corrected frame, because the
        #      question is "was this a highlight in the photograph", and the
        #      corrected frame has already moved. `_NORM_HP_LO` is low for a
        #      reason that is not obvious and is measured in its own comment: a
        #      narrow band high up flattens the curve rather than steepening it,
        #      because the blend and the correction end up pulling against each
        #      other.
        #
        #      Blended per channel against the source directly, so at 1 the
        #      result *is* the source there -- including its hue, and including
        #      any highlight that was already blown, which stays blown because
        #      there is nothing in the file to bring back.
        if hp > 0.001:
            sv = src.amax(dim=1, keepdim=True)
            w = _smoothstep(_NORM_HP_LO, _NORM_HP_HI, sv) * hp
            img = img + (src - img) * w

        # The stage's one clamp, and by construction it is a no-op on everything
        # the tone map touched: the roll lands the frame's own maximum on 1.0
        # rather than past it, and the blend above only ever moves toward a value
        # that was already in range. It is here for arithmetic slop, not to do
        # the compression's job -- if this line is ever what is bounding the
        # highlights, the metering is wrong and `verify.py` says so.
        return img.clamp(0.0, 1.0)
