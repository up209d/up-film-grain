from __future__ import annotations

import torch

from ..colour import _linear_to_srgb
from ..constants.halation import _BLUE_HUE, _BLUE_RANGE, _BLUE_SAT_FLOOR
from ..primitives import (
    _blur, _hsv_to_rgb, _hue_sat, _luma, _rotate_hue, _smootherstep, _smoothstep,
)

class HalationMixin:
    """Halation, and the blue compensation that puts the sky back before the
    wash takes it."""

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

    def _halation(
        self, lin: torch.Tensor, p: dict, scale: float,
    ) -> torch.Tensor:
        """Light reaching the film base, reflecting, and re-exposing the
        emulsion from behind. In linear light, on the whole frame.

        Extracted from `render()` on 2026-08-08 so the pipeline body reads as
        a sequence of sections. Bit-identical to the inline version.
        """
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
        return lin
