"""Tone Response: brightness, the characteristic curve, dye desaturation,
vibrance, the split tone and base fog.

Extracted from `render()` on 2026-08-08. The section ships neutral -- every
slider defaults to 0 -- but the shipped presets do use it, so this is not a
dead stage: `brightness` runs -0.6..0.1 across `presets/`, `toe` to 1.0,
`shoulder` to 0.3 and `vibrance` to 0.45.
"""

from __future__ import annotations

import torch

from ..colour import _characteristic_curve, _linear_to_srgb, _srgb_to_linear
from ..constants.tone import (
    _WARM_AXIS, _WARM_GAIN, _WARM_HI_BAND, _WARM_LO_BAND,
)
from ..primitives import _luma, _smoothstep


class ToneMixin:
    """The development stage's tonal controls, in pipeline order."""

    def _tone(self, base: torch.Tensor, p: dict) -> torch.Tensor:
        """Bit-identical to the inline version it replaced.

        Not a `staticmethod`, despite touching no engine state of its own: the
        body polls `self._poll_cancel` between its sub-stages, which is what
        keeps a superseded preview interruptible inside a section as well as
        between them.
        """
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

        self._poll_cancel()
        # 4. Dye layers desaturate as they approach saturation, rather than
        #    clipping to a hue-shifted edge the way a sensor does.
        hd = p["highlight_desat"]
        if hd > 0.01:
            lum_h = _luma(base)
            wgt = _smoothstep(0.62, 1.0, lum_h) * hd
            base = base + wgt * (lum_h - base)

        self._poll_cancel()
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

        self._poll_cancel()
        # 5. Split tone: a cross-channel bias on each end of the range. Most of
        #    what reads as "a film palette" lives here, not in the grain.
        #
        #    **Both controls are signed** (rewritten 2026-08-06, on request).
        #    They were `warm_highlights` and `cool_shadows`, each 0..1 and each
        #    locked to one direction, so the panel could describe warm-over-cool
        #    and nothing else -- not tungsten stock's cool highlights, not a
        #    cross-process, not warm shadows under a cold sky. Now each end of
        #    the range runs cool at -1 through neutral at 0 to warm at +1, which
        #    is the same two stages with the sign let out.
        #
        #    They were also reported as invisible, and they were: see
        #    `_WARM_GAIN` for the arithmetic. Both the amplitude and the two
        #    weighting bands were widened along with the sign.
        #
        #    One axis for both, in opposite directions, rather than a separate
        #    "warm" and "cool" vector. Two hand-written vectors are two things
        #    that can drift apart; a signed push along one axis is warm and cool
        #    by construction, and it is what makes 0 exactly neutral rather than
        #    approximately so.
        hw_, sw_ = p["highlight_warmth"], p["shadow_warmth"]
        if abs(hw_) > 0.001 or abs(sw_) > 0.001:
            lum_s = _luma(base)
            axis = torch.tensor(_WARM_AXIS, device=base.device,
                                dtype=base.dtype).view(1, 3, 1, 1) * _WARM_GAIN
            w_hi = _smoothstep(*_WARM_HI_BAND, lum_s) * hw_
            w_lo = (1.0 - _smoothstep(*_WARM_LO_BAND, lum_s)) * sw_
            base = base + (w_hi + w_lo) * axis

        self._poll_cancel()
        # 6. Base fog: the film base has a minimum density, so there is no true
        #    black. Lifts the floor without touching the white point.
        fog = p["base_fog"]
        if fog > 0.001:
            base = fog + (1.0 - fog) * base

        base = base.clamp(0.0, 1.0)
        return base
