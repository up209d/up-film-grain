from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ----------------------------------------------------------------- tone
    Param(
        "contrast", "Contrast", "Tone Response",
        0.0, 1.0, 0.01, 0.0, "",
        "Steepness of the straight-line section of the characteristic curve.",
    ),
    Param(
        "toe", "Toe", "Tone Response",
        0.0, 1.0, 0.01, 0.0, "",
        "Shadow compression. Film's response flattens as exposure falls, so "
        "shadows compress and lift rather than clipping to black.",
    ),
    Param(
        "shoulder", "Shoulder", "Tone Response",
        0.0, 1.0, 0.01, 0.0, "",
        "Highlight rolloff. The single biggest tell between film and digital: "
        "film shoulders gracefully where a sensor clips flat.",
    ),
    Param(
        "highlight_desat", "Highlight Desaturation", "Tone Response",
        0.0, 1.0, 0.01, 0.0, "",
        "Colour drains toward neutral as dye layers approach saturation, "
        "instead of clipping to a hue-shifted edge.",
    ),
    Param(
        "brightness", "Brightness", "Tone Response",
        -2.0, 2.0, 0.01, 0.0, "EV",
        "Exposure, in stops: +1 is twice the light, -1 is half. The multiply "
        "happens in linear light, so highlights roll off through the sRGB "
        "encoding instead of stretching into a flat clip, and it runs before "
        "the characteristic curve so Shoulder can catch whatever it lifts. "
        "0 = off, and the pipeline stays a pass-through.",
    ),
    Param(
        "vibrance", "Vibrance", "Tone Response",
        -1.0, 1.0, 0.01, 0.0, "",
        "Saturation weighted against how saturated a pixel already is: muted "
        "colour comes up, colour that is already strong is left where it is. "
        "That is the difference from a plain saturation control, which drags "
        "everything together and pushes already-vivid areas out of gamut. "
        "Negative drains the muted colour instead and leaves the vivid, which "
        "reads as a bleached look. 0 = off, and the pipeline stays a colour "
        "pass-through.",
    ),
    Param(
        "base_fog", "Base Fog", "Tone Response",
        0.0, 0.08, 0.002, 0.0, "",
        "Minimum density of the film base. There is no true black on film.",
    ),
    Param(
        "highlight_warmth", "Highlight Warmth", "Tone Response",
        -1.0, 1.0, 0.01, 0.0, "",
        "Cross-channel bias on the top of the range: **positive is warm, "
        "negative is cool**, and 0 leaves the highlights alone. The three dye "
        "layers reach saturation at different rates, so a stock's highlights "
        "carry a cast of their own -- warm on most colour negative, cool on "
        "tungsten stock and on a lot of reversal film. Both directions are "
        "reachable because both are real; a warm-only control could only ever "
        "describe half the stocks.\n"
        "\n"
        "The shift is **luma-neutral by construction** -- the colour axis is "
        "projected onto the plane where the luma weights sum to zero -- so this "
        "changes the colour of the highlights without brightening or darkening "
        "them, and it cannot fight Shoulder or Brightness for the same range.",
    ),
    Param(
        "shadow_warmth", "Shadow Warmth", "Tone Response",
        -1.0, 1.0, 0.01, 0.0, "",
        "The same control for the bottom of the range: positive warms the "
        "shadows, negative cools them, 0 leaves them alone. Set against "
        "Highlight Warmth this is a split tone, and it is most of what reads "
        "as a film colour palette -- cool shadows under warm highlights is the "
        "classic daylight negative look, and the opposite pairing is the "
        "cross-processed one.\n"
        "\n"
        "Luma-neutral like its partner. The two weightings overlap through the "
        "mid-tones on purpose, so setting both to the same sign tints the whole "
        "frame rather than leaving a band untouched in the middle.",
    ),
]
