from __future__ import annotations


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

__all__ = [
    '_BLUE_HUE',
    '_BLUE_RANGE',
    '_BLUE_SAT_FLOOR',
]
