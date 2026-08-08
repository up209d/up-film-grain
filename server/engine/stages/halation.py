from __future__ import annotations

import torch

from ..colour import _linear_to_srgb
from ..constants.halation import _BLUE_HUE, _BLUE_RANGE, _BLUE_SAT_FLOOR
from ..primitives import _hue_sat, _luma, _rotate_hue, _smootherstep, _smoothstep

class HalationMixin:
    """Blue compensation -- put the sky back before the wash takes it."""

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
