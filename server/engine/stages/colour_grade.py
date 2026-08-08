from __future__ import annotations

import torch

from ... import params as P
from ..colour import (
    _MID_GREY, _apply_lut, _linear_to_srgb, _reconstruct_highlights, _srgb_to_linear, _tone_roll,
)
from ..constants.core import _LUMA
from ..constants.grade import (
    _GRADE_CLARITY_GAIN, _GRADE_CONTRAST_GAIN, _GRADE_TEMP_GAIN, _GRADE_TINT_GAIN, _GRADE_TONE_KNEE,
)
from ..primitives import _blur, _luma

class ColourGradeMixin:
    """Step -1: 3D LUT plus the twelve adjustments, above the emulsion.

    The only block that runs before pre-blur. Ships neutral, so with
    nothing selected the whole section is a colour pass-through.
    """

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
