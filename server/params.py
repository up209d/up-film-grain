"""Parameter schema for the grain engine.

This module is the single source of truth for every tunable knob. The engine
reads defaults from here and the web client builds its slider panel from the
same schema (served by ``GET /api/params``), so the UI can never drift out of
sync with what the renderer actually accepts.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    group: str
    min: float
    max: float
    step: float
    default: float
    unit: str = ""
    help: str = ""
    #: True when the value is a *length* in full-resolution pixels, so it has
    #: to be rescaled when a preset authored on one image size is applied to
    #: another. See ``rescale``.
    spatial: bool = False
    #: Names for a *discrete* parameter, indexed by the value. Non-empty turns
    #: the control into a menu instead of a slider -- the value is still a
    #: number, so nothing else in the schema, the engine or a preset file has
    #: to know the difference. Only for genuine either/or choices: a stencil
    #: shape has no midpoint between "cross" and "diagonal", and a slider that
    #: pretends otherwise invites you to leave it at 2.5.
    choices: tuple[str, ...] = ()


# Groups are rendered in this order by the client.
GROUPS: list[str] = [
    "Pre Sharpen",
    "Grain Structure",
    "Luminance Response",
    "Edge Destruction",
    "Halation",
    "Optical",
    "Color",
    "Tone Response",
    "Global Grain",
    "Sharpening",
    "Film Texture",
]


PARAMS: list[Param] = [
    # ---------------------------------------------------------- pre sharpen
    # Runs before everything, on the untouched input -- see step 0 in
    # engine.render().
    Param(
        "pre_sharpen", "Pre Sharpen", "Pre Sharpen",
        0.0, 30.0, 0.01, 0.0, "",
        "Unsharp mask on the source, before any of the film pipeline. This is "
        "the opposite end from the Sharpening section: there is no grain yet, "
        "so it can only crisp the photograph -- and everything downstream then "
        "keys off the sharpened image, so edges read as harder to the edge "
        "mask and grain follows them. Use it to bring a soft scan up before "
        "the emulsion goes on. 0 = off.",
    ),
    Param(
        "pre_sharpen_radius", "Pre Sharpen Radius", "Pre Sharpen",
        0.3, 8.0, 0.05, 1.0, "px",
        "Radius of the pre-sharpen unsharp mask, at full resolution.",
        spatial=True,
    ),
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
    # ---------------------------------------------------------------- grain
    Param(
        "intensity", "Intensity", "Grain Structure",
        0.0, 100.0, 0.5, 32.0, "%",
        "Overall grain amplitude before luminance and edge weighting.",
    ),
    Param(
        "grain_size", "Clump Size", "Grain Structure",
        0.1, 10.0, 0.05, 1.6, "px",
        "Silver-halide clump diameter, measured at full resolution -- the "
        "finest structure in the grain. Octaves stack coarser scales on top of "
        "it. Held in full-res units, so it means the same thing at any zoom.",
        spatial=True,
    ),
    Param(
        "shadow_size", "Shadow Clumping", "Grain Structure",
        0.0, 1.0, 0.01, 0.45, "",
        "Enlarges clumps in shadows, where halide crystals grow larger and "
        "less densely packed.",
    ),
    Param(
        "clump", "Clump Hardness", "Grain Structure",
        0.0, 1.0, 0.01, 0.5, "",
        "Sharpens the grain distribution toward discrete clumps instead of "
        "smooth gaussian noise.",
    ),
    Param(
        "octaves", "Octaves", "Grain Structure",
        1.0, 10.0, 1.0, 3.0, "",
        "Number of noise scales stacked to build the emulsion structure. Clump "
        "Size is the finest scale and each octave adds one twice as coarse, so "
        "this builds clumps-of-clumps and then broad mottling over the top. "
        "How far up the stack still matters is set by Roughness: at 0.5 the "
        "effect has largely landed by 3-4 octaves, at 1.0 all ten count.",
    ),
    Param(
        "roughness", "Roughness", "Grain Structure",
        0.0, 1.0, 0.01, 0.5, "",
        "Amplitude falloff between successive octaves, and so how much of the "
        "coarse structure survives. 0 = the base clump alone and Octaves does "
        "nothing; 1 = every octave weighs the same and the grain goes visibly "
        "clumpy and mottled. Total grain strength stays put either way -- this "
        "moves structure around, Intensity sets how much of it there is.",
    ),
    # ------------------------------------------------------------ luminance
    Param(
        "lum_low", "Shadow Knee", "Luminance Response",
        0.0, 0.5, 0.005, 0.15, "",
        "Lower edge of the peak-grain band. Below this, density falls off.",
    ),
    Param(
        "lum_high", "Highlight Knee", "Luminance Response",
        0.3, 1.0, 0.005, 0.65, "",
        "Upper edge of the peak-grain band. Above this, tightly packed "
        "silver suppresses visible grain.",
    ),
    Param(
        "shadow_falloff", "Shadow Falloff", "Luminance Response",
        0.02, 0.5, 0.005, 0.15, "",
        "How wide the fade-out is below the shadow knee. Independent of the "
        "knee position, so you can place the band anywhere and still control "
        "how gradual the transition into it is.",
    ),
    Param(
        "highlight_falloff", "Highlight Falloff", "Luminance Response",
        0.02, 0.5, 0.005, 0.25, "",
        "How wide the fade-out is above the highlight knee. Widen it for a "
        "gentler hand-off into clean highlights.",
    ),
    Param(
        "highlight_drop", "Highlight Suppression", "Luminance Response",
        0.0, 1.0, 0.01, 0.85, "",
        "How far grain is cut in dense highlights. 0.85 = 85% reduction.",
    ),
    Param(
        "shadow_drop", "Black Suppression", "Luminance Response",
        0.0, 1.0, 0.01, 0.6, "",
        "How far grain is cut in deep blacks.",
    ),
    # ----------------------------------------------------------------- edge
    Param(
        "edge_bias", "Edge Bias", "Edge Destruction",
        0.0, 1.0, 0.01, 0.75, "",
        "Pushes grain onto high-contrast micro-edges and away from flat, "
        "smooth areas such as skies.",
    ),
    Param(
        "smooth_guard", "Smooth-Area Guard", "Edge Destruction",
        0.0, 1.0, 0.01, 0.85, "",
        "Keeps grain out of genuinely featureless regions -- skin, clear sky, "
        "studio backdrops -- by measuring local contrast over a medium radius "
        "rather than brightness. 0 = off, 1 = smooth areas left clean.",
    ),
    Param(
        "highpass_radius", "High-Pass Radius", "Edge Destruction",
        0.5, 5.0, 0.05, 2.0, "px",
        "Radius used to isolate micro-edges, at full resolution.",
        spatial=True,
    ),
    Param(
        "edge_erosion", "Edge Erosion", "Edge Destruction",
        0.0, 1.0, 0.01, 0.5, "",
        "Modulates existing micro-detail by the grain field so grain erodes "
        "edge structure rather than sitting on top of it.",
    ),
    Param(
        "acutance", "Acutance", "Edge Destruction",
        0.0, 1.0, 0.01, 0.25, "",
        "Adjacency (Eberhard) effect: developer exhausts differently on either "
        "side of an edge, leaving a local contrast boost. It is why film reads "
        "as sharp despite resolving less detail than a sensor.",
    ),
    Param(
        "edge_soften", "Edge Softening", "Edge Destruction",
        0.0, 1.0, 0.01, 0.0, "",
        "Takes the digital snap off hard transitions without touching flat "
        "areas or fine texture. This is the one to reach for when the image "
        "wants to be softer -- Micro-Blur diffuses the whole frame, texture "
        "and all, which reads as out of focus rather than as film. Grain is "
        "added afterwards and its amount is measured from the unsoftened "
        "image, so softening never costs you noise. 0 = off.",
    ),
    Param(
        "edge_soften_radius", "Softening Radius", "Edge Destruction",
        0.3, 8.0, 0.05, 1.5, "px",
        "How far a softened edge spreads, at full resolution. Kept separate "
        "from the amount so you can set how soft independently of how wide.",
        spatial=True,
    ),
    Param(
        "edge_jitter", "Edge Jitter", "Edge Destruction",
        0.0, 5.0, 0.01, 0.3, "",
        "Warps edges along a noise field so a border wanders instead of "
        "running dead straight, which is most of what stops a rendered edge "
        "reading as vector art. Displacement is in full-resolution pixels and "
        "peaks at 3px; the default 0.3 makes a straight border wander about "
        "±0.4px. Flat areas are untouched — it is weighted by the edge mask.",
        spatial=True,
    ),
    Param(
        "jitter_aniso", "Jitter Direction", "Edge Destruction",
        0.0, 1.0, 0.01, 0.0, "",
        "Concentrates Edge Jitter onto one axis instead of displacing edges "
        "every way at once. 0 = isotropic, the default, and the angle below "
        "then does nothing -- rotating a field that is already the same in "
        "every direction changes nothing. 1 = edges only ever move parallel "
        "to that angle, which reads as a directional slip rather than a "
        "wobble.",
    ),
    Param(
        "jitter_angle", "Jitter Angle", "Edge Destruction",
        0.0, 180.0, 1.0, 0.0, "deg",
        "Axis the jitter is biased along, once Jitter Direction is above 0. "
        "0 = horizontal, 90 = vertical, 45 = diagonal. Only 0-180 is needed: "
        "the displacement is symmetric, so 200 degrees is 20 degrees.",
    ),
    Param(
        "edge_sand", "Edge Sanding", "Edge Destruction",
        0.0, 5.0, 0.01, 0.0, "",
        "Polishes the jaggedness back off a roughened border, the way "
        "sandpaper does -- the counterpart to Edge Jitter rather than more of "
        "it. It averages each pixel with its neighbours *along* the edge, "
        "never across it, so the burrs and stair-stepping smooth out while "
        "the transition stays exactly as sharp. Raise it when jitter or "
        "erosion has left an edge looking harsh. 0 = off.",
    ),
    Param(
        "edge_sand_grit", "Sanding Grit", "Edge Destruction",
        0.3, 20.0, 0.05, 0.8, "px",
        "How far along the edge the polish reaches, at full resolution. Small "
        "is a fine grit: it takes off pixel-scale jaggies and leaves the "
        "border's shape alone. Large flattens broader undulations too, so the "
        "wander Edge Jitter added starts going with them.",
        spatial=True,
    ),
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
        "Luminance Response knees use.",
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
    # -------------------------------------------------------------- optical
    Param(
        "micro_blur", "Micro-Blur", "Optical",
        0.0, 3.0, 0.01, 0.45, "px",
        "Light diffusion through the gel layers, as an average: every pixel "
        "is mixed with its neighbours. That is the smooth half of diffusion, "
        "and it costs texture along with the edges -- Scatter below is the "
        "same physics without the averaging. Applied to the base image before "
        "grain injection so grain stays sharp against a soft base.",
        spatial=True,
    ),
    # Scatter: diffusion resolved as discrete deflections instead of as an
    # average. See step 1b in engine.render() for why that is not a blur.
    Param(
        "scatter", "Scatter", "Optical",
        0.0, 1.0, 0.01, 0.0, "",
        "Spreads detail into the neighbouring pixels *without* averaging "
        "anything, so the picture loses its digital exactness while keeping "
        "its bite. Every displaced pixel is an exact copy of a real pixel "
        "nearby -- no in-between values are invented, so contrast, grit and "
        "texture come through at full strength where a blur of the same reach "
        "would have flattened them. The number is the fraction of the frame "
        "that moves: 0.3 relocates three pixels in ten and leaves the rest "
        "exactly where they were. It is deliberately a coverage, not a blend "
        "-- blending a moved pixel with its original *is* averaging, and that "
        "is the one thing this stage must never do. Smooth regions are "
        "untouched for free: shuffling pixels that already match their "
        "neighbours changes nothing, so skies and skin stay clean while "
        "detail is the only thing that comes apart. 0 = off.",
    ),
    Param(
        "scatter_radius", "Scatter Reach", "Optical",
        0.5, 24.0, 0.1, 3.0, "px",
        "How far a displaced pixel travels, at full resolution. Small reads "
        "as an emulsion that will not quite resolve; large tears detail into "
        "streaks and crumbs. It is also what decides *which* structure comes "
        "apart, because moving a pixel only changes anything where the "
        "picture varies over the distance travelled: a short reach disorders "
        "fine texture and leaves shapes standing, a long one starts taking "
        "the shapes with it.",
        spatial=True,
    ),
    Param(
        "scatter_pattern", "Scatter Pattern", "Optical",
        0.0, 8.0, 1.0, 0.0, "",
        "Where a displaced pixel is allowed to land -- the stencil. Restricting "
        "the choice is what makes the result read as a *structure* rather than "
        "as noise: detail smears the way the shape says and nowhere else.\n"
        "\n"
        "Any is isotropic and reads as plain diffusion. Cross, Diagonal and Box "
        "are the 4-, 45- and 8-neighbour stencils. Diamond keeps every angle "
        "but reaches furthest along the axes and pulls in on the diagonals, so "
        "detail spreads as a rhombus rather than a disc. Donut holds a hole "
        "open in the middle -- nothing lands near where it started, so detail "
        "is thrown outward and hollowed out, and it stays hollow whatever Reach "
        "Spread is set to. Star is eight spokes with every other one running "
        "short, which is the shape a cross filter flares into. Horizontal and "
        "Vertical are the extreme case, a one-axis slip that leaves edges "
        "running along that axis completely untouched.",
        choices=("Any", "Cross", "Diagonal", "Box", "Diamond", "Donut",
                 "Star", "Horizontal", "Vertical"),
    ),
    Param(
        "scatter_spread", "Reach Spread", "Optical",
        0.0, 1.0, 0.01, 1.0, "",
        "Whether every displaced pixel travels the full reach or a share of "
        "it. 0 is a shell -- everything lands on the edge of the pattern's "
        "shape, which hollows detail out into an outline and is the harshest "
        "setting here. 1 fills the shape inward, with distances spread evenly "
        "from nothing up to the reach, which reads as diffusion rather than as "
        "an outline. Donut is the exception by design: it holds its hole open "
        "at any setting, so this only decides how thick its ring is.",
    ),
    Param(
        "scatter_cell", "Scatter Clump", "Optical",
        1.0, 16.0, 0.1, 1.0, "px",
        "How big a piece of the picture moves as one. At 1 every pixel "
        "chooses for itself and the image crumbles; larger values move whole "
        "tiles of detail intact, so structure survives the trip and lands "
        "somewhere else. Past about 4px the tiles start reading as tiles -- "
        "which is a look, a shattered plate rather than a soft one, but it is "
        "no longer subtle. Held in full-res pixels like every other length.",
        spatial=True,
    ),
    # ---------------------------------------------------------------- color
    Param(
        "chroma_grain", "Chroma Grain", "Color",
        0.0, 1.0, 0.01, 0.35, "",
        "0 = monochrome grain shared across channels. 1 = independent dye "
        "cloud noise per layer.",
    ),
    Param(
        "edge_chroma", "Edge Colour Fringing", "Color",
        0.0, 1.0, 0.01, 0.5, "",
        "Runs edge erosion independently per colour layer, so eroded edges "
        "pick up coloured speckle. 0 = neutral erosion, 1 = full dye-layer "
        "fringing.",
    ),
    Param(
        "warm_highlights", "Warm Highlights", "Color",
        0.0, 1.0, 0.01, 0.0, "",
        "Cross-channel bias pushing highlights warm, as the three dye layers "
        "reach saturation at different rates.",
    ),
    Param(
        "cool_shadows", "Cool Shadows", "Color",
        0.0, 1.0, 0.01, 0.0, "",
        "Complementary cool cast in the shadows. Together with warm "
        "highlights this is most of what reads as a film colour palette.",
    ),
    Param(
        "seed", "Seed", "Color",
        0.0, 9999.0, 1.0, 1234.0, "",
        "Deterministic seed for the grain lattice.",
    ),
    # --------------------------------------------------------- global grain
    # Applied last and masked by nothing -- see step 13 in engine.render().
    # Ships at 0 so it never alters an existing look until asked for.
    Param(
        "global_intensity", "Global Intensity", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "A flat grain layer over the finished frame, at one strength "
        "everywhere. Unlike the main grain it ignores the luminance band, the "
        "edge bias and the smooth-area guard, so it reaches skies, skin and "
        "blown highlights that those masks deliberately keep clean. 0 = off. "
        "Because nothing holds it back it bites far harder than the main "
        "Intensity slider at the same number -- 32 here measures 8% luminance "
        "sigma against 3.5% there. 5-20 is the usable range.",
    ),
    Param(
        "global_size", "Global Size", "Global Grain",
        0.1, 10.0, 0.05, 1.6, "px",
        "Clump diameter of the global layer, at full resolution. Set it apart "
        "from Clump Size and the two layers read as separate structures; match "
        "them and it just thickens the main grain.",
        spatial=True,
    ),
    Param(
        "global_opacity", "Global Opacity", "Global Grain",
        0.0, 1.0, 0.01, 1.0, "",
        "How much of the global layer is mixed in. It multiplies with Global "
        "Intensity -- intensity is how coarse and strong the layer is in its "
        "own right, opacity is how far it is dialled back over the image.",
    ),
    # ------------------------------------------------------------ sharpening
    # The last stage in the pipeline -- see step 14 in engine.render().
    Param(
        "sharpen", "Sharpen", "Sharpening",
        0.0, 30.0, 0.01, 0.0, "",
        "Unsharp mask over the finished frame. Because it runs last, the "
        "high-frequency detail it amplifies is the grain as much as the "
        "image -- it cranks the noise already there rather than adding any, "
        "so grain gains bite and the picture gains acutance together. "
        "Measured on textured detail: 1 puts grain at 150% of unsharpened, 2 "
        "at 204%, 10 at 601%, 20 at 877% -- the top of the range compresses "
        "as overshoot starts clipping, but never stops responding. The usual "
        "unsharp halos show on hard borders past about 1.2, so nearly all of "
        "this range is a deliberate effect rather than a correction. 0 = off.",
    ),
    Param(
        "sharpen_radius", "Sharpen Radius", "Sharpening",
        0.3, 8.0, 0.05, 1.0, "px",
        "Radius of the unsharp mask, at full resolution. Keep it near the "
        "clump size to bite on grain; widen it to work on image structure "
        "instead, which fattens halos as it goes.",
        spatial=True,
    ),
    # ---------------------------------------------------------- film texture
    # Physical damage to the film, not emulsion behaviour -- applied dead last
    # and masked by nothing. See step 15 in engine.render().
    Param(
        "dust", "Dust Count", "Film Texture",
        0.0, 400.0, 1.0, 0.0, "",
        "Roughly how many specks land on the frame -- a count, not a "
        "strength, so it means the same thing whatever the image size. Two "
        "thirds print dark (opaque motes) and one third bright (pinholes "
        "and lint). Approximate by nature: specks merge and the frame edge "
        "clips some. 0 = none.",
    ),
    Param(
        "dust_size", "Dust Size", "Film Texture",
        0.5, 120.0, 0.05, 2.0, "px",
        "Speck diameter at full resolution. Small is scanner dust; large is "
        "lint and debris on the negative.",
        spatial=True,
    ),
    Param(
        "dust_opacity", "Dust Opacity", "Film Texture",
        0.0, 1.0, 0.01, 0.85, "",
        "How much of the photograph a speck hides at full strength. Separate "
        "from luminosity: opacity is coverage, luminosity is what colour the "
        "speck is, so a solid grey mote and a faint black veil are different "
        "things rather than the same number twice.",
    ),
    Param(
        "dust_opacity_var", "Dust Opacity Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.6, "",
        "How much opacity differs between specks. At 1 the faintest are "
        "barely there while the strongest are solid -- real dust sits at "
        "different depths and in different thicknesses.",
    ),
    Param(
        "dust_lum_var", "Dust Luminosity Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.5, "",
        "How much the specks differ in tone. Dark motes spread across black "
        "to mid-grey and bright pinholes across off-white to white, so each "
        "population varies within itself without the two swapping places.",
    ),
    Param(
        "dust_soften", "Dust Softness", "Film Texture",
        0.0, 1.0, 0.01, 0.35, "",
        "How far out of focus the specks are. Debris sits at different "
        "depths, so this is a *spread* rather than a uniform blur -- some "
        "specks stay crisp and others go soft at any setting. Blurred "
        "specks also come out fainter, which is what out-of-focus debris "
        "actually does. 0 = all crisp.",
    ),
    Param(
        "scratches", "Scratch Count", "Film Texture",
        0.0, 60.0, 1.0, 0.0, "",
        "Roughly how many gouges run down the frame. They follow the "
        "direction of travel and print bright, because a scratch through "
        "the emulsion lets light straight through, and they break along "
        "their length rather than ruling the whole frame. 0 = none.",
    ),
    Param(
        "scratch_width", "Scratch Width", "Film Texture",
        0.3, 20.0, 0.05, 1.0, "px",
        "Width of a scratch at full resolution. Hairline values are the "
        "convincing ones; wide reads as damage rather than wear.",
        spatial=True,
    ),
    Param(
        "scratch_soften", "Scratch Softness", "Film Texture",
        0.0, 1.0, 0.01, 0.35, "",
        "Softens the gouges. A perfectly crisp scratch is the clearest sign "
        "the texture was generated -- real ones are cut at different depths "
        "and the scanner only focuses on one plane. Softness varies between "
        "scratches but stays constant along each one, so a scratch never "
        "fades in and out down its own length. 0 = all crisp.",
    ),
    Param(
        "hair", "Hair Count", "Film Texture",
        0.0, 40.0, 1.0, 0.0, "",
        "Roughly how many hairs and fibres are lying on the frame, printing "
        "as dark wandering filaments. Each follows a contour of a noise "
        "field, so it curves the way a hair actually lies rather than along "
        "a curve somebody chose. 0 = none.",
    ),
    Param(
        "hair_length", "Hair Length", "Film Texture",
        20.0, 600.0, 5.0, 160.0, "px",
        "How long each hair is, at full resolution -- independent of how many "
        "there are. It also sets how much a hair wanders over that length, "
        "because a longer filament follows a broader contour.",
        spatial=True,
    ),
    Param(
        "hair_soften", "Hair Softness", "Film Texture",
        0.0, 1.0, 0.01, 0.35, "",
        "Softens the filaments. A hair lying on the glass is sharp; one on "
        "the negative a layer away is not, so the population wants both. 0 "
        "= all crisp.",
    ),
    Param(
        "light_leak", "Leak Count", "Film Texture",
        0.0, 12.0, 1.0, 0.0, "",
        "Roughly how many light leaks reach in from the frame edges. "
        "Counted against the perimeter rather than the area, because that "
        "is where they happen. Added in linear light so it behaves like "
        "light falling on the emulsion, not a gradient painted over the "
        "picture. 0 = none -- and so is anything below 1, because you "
        "cannot render a fraction of a leak. A hand-edited file holding "
        "0.05 here renders nothing at all rather than a faint leak.",
    ),
    Param(
        "leak_strength", "Leak Strength", "Film Texture",
        0.0, 3.0, 0.01, 1.0, "",
        "How much light each leak lets in. The response saturates one dye "
        "layer at a time, so this is not just an opacity: a faint leak is deep "
        "red because only the red-sensitive layer caught enough light, and "
        "pushing it up takes the core through orange and yellow to white while "
        "leaving the colour in the falloff. Past about 1.5 most leaks have a "
        "blown white core, which is the 'sun got in the back' look.",
    ),
    Param(
        "leak_size_min", "Leak Size Min", "Film Texture",
        5.0, 3000.0, 1.0, 250.0, "px",
        "How far the *smallest* leak reaches in from the frame edge, at full "
        "resolution. Each leak picks its own reach somewhere between this and "
        "the maximum, so the two together are what makes a frame of leaks look "
        "accidental rather than stamped -- set them equal and every leak comes "
        "in exactly as far as the next.",
        spatial=True,
    ),
    Param(
        "leak_size_max", "Leak Size Max", "Film Texture",
        5.0, 3000.0, 1.0, 850.0, "px",
        "How far the *largest* leak reaches in. Given below the minimum the "
        "two simply swap, so you can drag either one past the other without "
        "the leaks disappearing. Corners bloom further than edge midpoints "
        "whatever this says -- that is where the cassette mouth and the film "
        "gate actually let light past. Reach is capped at half the frame's "
        "short side, which is the distance at which a leak just dies in the "
        "middle: past that it would leave a floor over the whole frame, and a "
        "leak that fogs the centre reads as a bad exposure rather than a leak.",
        spatial=True,
    ),
    Param(
        "leak_feather", "Leak Feather", "Film Texture",
        1.0, 1500.0, 1.0, 180.0, "px",
        "How far in from the border a leak has faded to *half* strength, at "
        "full resolution -- so it is a distance you can see rather than an "
        "abstract softness. Small against the size gives a tight bright rim "
        "hugging the edge; around half the size gives a straight ramp; most of "
        "the way to the size gives a broad wash that hardly falls off until it "
        "ends. Because it is absolute, the same feather is a wash on a small "
        "leak and a rim on a large one, which is what stops a frame of "
        "differently-sized leaks looking like one shape at several scales.\n"
        "\n"
        "It softens the leak's *other* edge too -- the transition along the "
        "border where one leak stops. A leak has two visible edges and "
        "softening only one still reads as a painted shape.",
        spatial=True,
    ),
    Param(
        "leak_variation", "Leak Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.7, "",
        "How much one leak differs from the next in everything *except* size, "
        "which Leak Size Min and Max now set directly: how hard its edge is, "
        "how broad or tight its halo is, and how strong it arrives. 0 makes "
        "every leak identical in those respects; 1 is a wide spread. Light "
        "gets in through whatever gap it finds, and no two gaps are alike.",
    ),
    Param(
        "leak_hue", "Leak Hue", "Film Texture",
        0.0, 1.0, 0.01, 0.3, "",
        "0 = deep red, the classic 35mm canister leak. 1 = amber, closer to "
        "daylight getting in around a loose back.",
    ),
    Param(
        "texture_seed", "Texture Seed", "Film Texture",
        0.0, 9999.0, 1.0, 77.0, "",
        "Re-rolls where every mark lands. Separate from the grain Seed on "
        "purpose: you will want to reshuffle the damage without disturbing "
        "grain you have already dialled in.",
    ),
]


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
    "pre_sharpen",
    "contrast", "toe", "shoulder", "highlight_desat", "brightness",
    "vibrance", "base_fog",
    "intensity",
    "edge_erosion", "acutance", "edge_soften", "edge_sand", "edge_jitter",
    "halation", "halation_blue",
    "micro_blur", "scatter",
    "warm_highlights", "cool_shadows",
    "global_intensity",
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


def schema() -> dict:
    """JSON-serialisable schema for the client."""
    presets = load_presets()
    names = {p["name"] for p in presets}
    return {
        "groups": GROUPS,
        "params": [asdict(p) for p in PARAMS],
        "presets": presets,
        # The preset the client starts on, or None to start on the raw
        # parameter defaults. Reported rather than assumed, so a missing file
        # degrades to "no starting preset" instead of erroring.
        "default_preset": DEFAULT_PRESET if DEFAULT_PRESET in names else None,
        # What the client's "Original" button applies.
        "neutral": neutral_values(),
        "default_reference_mp": DEFAULT_REFERENCE_MP,
    }


def sanitize(raw: dict | None) -> dict[str, float]:
    """Clamp incoming values into range and fill any missing key with its default.

    Unknown keys are dropped. This is the only place params enter the engine,
    so the renderer can assume every value is present and in range.
    """
    out = dict(DEFAULTS)
    if not raw:
        return out
    for key, value in raw.items():
        p = PARAM_BY_KEY.get(key)
        if p is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        out[key] = max(p.min, min(p.max, v))
    return out


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #

# Presets are files on disk, not a table in this module. Anything you save from
# the UI can be dropped in here and it is a preset; nothing has to be edited or
# redeployed to add one. The directory sits beside the server package, so the
# same relative layout works in the source tree and in a built distribution.
PRESET_DIR = Path(
    os.environ.get("FILM_GRAIN_PRESETS")
    or Path(__file__).resolve().parent.parent / "presets"
)

# Preset applied when the app opens, and the one Reset returns to -- by name,
# which is to say by filename. It is only a *preferred* name: if no such file
# exists the client falls back to the parameter defaults, so deleting it is a
# supported way to start from neutral rather than a way to break startup.
DEFAULT_PRESET = os.environ.get("FILM_GRAIN_DEFAULT_PRESET", "Stock")

# Fallback size for presets that do not record one. Unset by default: a preset
# with no `reference_mp` scales by 1.0, which is exactly how it behaved before
# rescaling existed. Guessing a size here would silently change the look of
# every legacy preset, and a wrong guess is worse than no scaling.
#
# Set it if you know your existing presets were all dialled in on the same
# camera -- FILM_GRAIN_DEFAULT_REFERENCE_MP=24 retrofits the lot in one go.
try:
    DEFAULT_REFERENCE_MP: float | None = (
        float(os.environ.get("FILM_GRAIN_DEFAULT_REFERENCE_MP", "") or 0) or None
    )
except ValueError:
    DEFAULT_REFERENCE_MP = None


def load_presets() -> list[dict]:
    """Read every ``*.json`` in ``PRESET_DIR``, sorted by name.

    Read on each call rather than cached, so dropping a file into the folder
    shows up on the next page load without restarting the server. It is a
    handful of small files.

    A preset is named by its **filename**, not by whatever ``name`` the file
    carries inside. The files are the interface here -- renaming one in Finder
    should rename it in the dropdown, and a file saved under one name and
    renamed later should not keep announcing the old one.

    Values go through ``sanitize`` like any other input: unknown keys dropped,
    everything clamped into range, anything missing filled from defaults. So a
    hand-edited file, or one written before a slider's range changed, still
    loads instead of poisoning the engine.
    """
    out: list[dict] = []
    try:
        files = sorted(PRESET_DIR.glob("*.json"), key=lambda f: f.name.lower())
    except OSError:
        return out

    for f in files:
        try:
            raw = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            # Say so rather than silently omitting it -- a typo in one file
            # should not make a preset quietly cease to exist.
            print(f"[presets] skipping {f.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(raw, dict):
            print(f"[presets] skipping {f.name}: not a JSON object", file=sys.stderr)
            continue
        # Accept our own wrapper or a bare {key: value} map, matching what the
        # client's file loader accepts.
        values = raw.get("values")
        if not isinstance(values, dict):
            values = raw
        ref = raw.get("reference_mp") or DEFAULT_REFERENCE_MP
        out.append({
            "name": f.stem,
            "values": sanitize(values),
            # Size the preset was dialled in on, so it can be rescaled onto a
            # different photo. Absent in older files -> no scaling, which is
            # the pre-existing behaviour rather than a guess.
            "reference_mp": float(ref) if isinstance(ref, (int, float)) and ref else None,
        })
    return out
