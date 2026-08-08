from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ------------------------------------------------------------- halation
    Param(
        "halation", "Halation", "Halation",
        0.0, 1.0, 0.01, 0.35, "",
        "Warm bloom around bright highlights, from light reflecting off the "
        "film base and re-exposing the emulsion from behind.",
    ),
    Param(
        "halation_radius", "Halation Spread", "Halation",
        2.0, 80.0, 0.5, 24.0, "px",
        "How far the bloom spreads, at full resolution.",
        spatial=True,
    ),
    Param(
        "halation_threshold", "Halation Threshold", "Halation",
        0.3, 1.0, 0.01, 0.72, "",
        "Luminance above which highlights start to bloom.",
    ),
    Param(
        "halation_recovery", "Highlight Recovery", "Halation",
        0.0, 1.0, 0.01, 0.0, "",
        "Adds the bloom into the headroom that is actually there, instead of "
        "adding it flat and letting the total clip. The bloom is added as light, "
        "with nothing stopping it, so a highlight already near white gets "
        "pushed the rest of the way to a flat, textureless patch -- the usual "
        "complaint that halation burns highlights out.\n"
        "\n"
        "The bloom is metered against the room each channel has left, so a "
        "highlight with headroom to spare still gets the whole thing at full "
        "strength and only one being asked to take more light than it can hold "
        "is held back at all. At 1.0 no channel can be driven to white by the "
        "bloom, and the sum stays strictly ordered -- two highlights that "
        "differed by a hair before still differ after, which is exactly what a "
        "clip destroys and what dimming a flat patch cannot give back.\n"
        "\n"
        "**This is not a strength control.** Measured on a bright plate "
        "carrying fine texture, 1.0 keeps 60% of that texture against 40% with "
        "recovery off, and does it while keeping 68% of the bloom's light -- "
        "more detail *and* more bloom than simply turning Halation down to the "
        "same effect. Measured per channel, so a saturated highlight far over "
        "the threshold in luma but with most of one channel still free gets "
        "that channel's full share.\n"
        "\n"
        "0 = off, and the bloom burns exactly as much as Halation and Halation "
        "Threshold alone say it should.",
    ),
    Param(
        "halation_hue", "Halation Hue", "Halation",
        0.0, 360.0, 1.0, 11.0, "deg",
        "Colour of the bloom, as a hue angle. 0 = red, 30 = amber, 60 = "
        "yellow, 120 = green, 240 = blue. Real halation is red -- the "
        "antihalation layer and the red-sensitive layer are what produce it, "
        "so 0-40 is the physically honest region -- but the whole wheel is "
        "here because this is a look tool. Was a 0-1 red-to-amber ramp that "
        "only spanned 25 degrees.",
    ),
    # Blue compensation. Applied to the image *before* the wash lands, so the
    # blue is strengthened on clean data and the wash is left alone -- see
    # step 2a in engine.render() for why that beats correcting afterwards.
    Param(
        "halation_blue", "Blue Compensation", "Halation",
        0.0, 3.0, 0.01, 0.0, "",
        "Strengthens blue *before* the bloom lands on it, so it survives the "
        "wash instead of being greyed by it. Halation adds warm light, and "
        "adding light to a colour desaturates it -- a red bloom over a blue "
        "sky lifts the red channel far more than the blue one, so the sky "
        "loses its colour and drifts toward grey. This puts the colour back "
        "in the exposure rather than repainting it afterwards, which is what "
        "a punchier blue-sensitive stock or a polariser would do.\n"
        "\n"
        "It is self-limiting, which is the main reason it runs here rather "
        "than after the wash: whatever you add, the wash eats the same share "
        "of it, so the recovered saturation flattens off instead of running "
        "away. Measured on a sky the bloom had cost 16% of its colour, 0.5 "
        "puts it back to within 1% of untouched and everything from 1.0 "
        "upward sits at 3% past it -- 3.0 included. The same correction "
        "applied *after* the wash has no such brake: it is 9% past by 0.5 and "
        "by 1.0 it has driven a channel to black, pinning the sky at fully "
        "saturated. Only does anything while Halation is above 0: with no "
        "wash there is nothing to compensate for, and this is not a grading "
        "control. 0 = off.",
    ),
    Param(
        "halation_blue_level", "Blue Level", "Halation",
        0.0, 1.0, 0.005, 0.45, "",
        "How light a blue has to be before it is worth saving. The wash only "
        "reaches what is near the light, so pale sky loses colour and deep "
        "sky loses none -- measured on a sky gradient, the loss is 23% at the "
        "bright end and flat 0% below about half brightness. Compensating "
        "everything regardless is what makes a deep blue go lurid: it was "
        "never damaged, so every bit of the correction is overshoot. Blue "
        "above this brightness is compensated and blue below it is left as "
        "it was. Read on the picture's own brightness scale, the same one the "
        "Shadow and Highlight Knees under Grain Structure use.",
    ),
    Param(
        "halation_blue_falloff", "Blue Level Falloff", "Halation",
        0.02, 0.5, 0.005, 0.25, "",
        "How wide the fade is below the level, so the change from saved to "
        "left-alone is a ramp rather than a line across the sky. Independent "
        "of the level itself on purpose -- deriving the width from the knee "
        "would mean moving the knee also changed the softness, and a sky is "
        "exactly the smooth gradient that shows up a hard switch-on.",
    ),
    Param(
        "halation_blue_shift", "Blue Hue Shift", "Halation",
        -45.0, 45.0, 1.0, 0.0, "deg",
        "Rotates the selected blue before the wash. Saturation alone cannot "
        "fix the *hue*: measured, a red bloom swings an ordinary sky about 6 "
        "degrees toward purple, and scaling saturation about the luma axis "
        "leaves that swing exactly where it is. Negative turns the sky toward "
        "cyan, which is the direction that cancels a warm bloom. Applied only "
        "where Blue Range selects, so the rest of the frame keeps its hue.",
    ),
    Param(
        "halation_sat", "Halation Saturation", "Halation",
        0.0, 1.0, 0.01, 0.86, "",
        "How coloured the bloom is. 0 = a neutral white glow, which the old "
        "red-to-amber control could not reach at all; 1 = fully saturated. "
        "Lower it when halation is strong and the tint starts to read as a "
        "colour cast rather than as light.",
    ),
]
