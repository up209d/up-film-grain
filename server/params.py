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


# The Global Grain blend modes, indexed by ``global_blend``. Defined here rather
# than in the engine because this module is the single source of truth and the
# engine imports it -- two literal tuples that had to agree on an *index* would
# be a silent renderer bug the first time one of them was reordered.
#
# Order is therefore load-bearing in preset files: appending is safe, reordering
# is not. Index 0 is Add, which is what this section did before the menu
# existed, so an old preset with no `global_blend` key sanitizes to the historic
# behaviour rather than to something new.
GLOBAL_BLENDS: tuple[str, ...] = (
    "Add", "Overlay", "Soft Light", "Hard Light", "Multiply", "Screen",
)


# Groups are rendered in this order by the client.
#
# `Luminance Response` is **gone as a section** (2026-08-06, on request): its six
# parameters moved into `Grain Structure`, under the controls that build the
# field they mask. It was never a stage of its own -- it says which densities
# carry the grain the section above it makes -- and a heading of its own read as
# a second thing to set up rather than as the tail of the first. See
# `docs/panel-layout.md`, which also has why this list is not, and cannot be,
# pipeline-ordered as a whole.
GROUPS: list[str] = [
    "Colour Grading",
    "Pre Blur",
    "Pre Sharpen",
    "Grain Structure",
    "Edge Destruction",
    "Anti Aliasing",
    "Global Grain",
    "Sharpening",
    "Halation",
    "Tone Response",
    "Film Texture",
    "Output",
]


PARAMS: list[Param] = [
    # ------------------------------------------------------- colour grading
    # Step -1: the only block above pre-blur, and the whole section runs on the
    # source before anything films it. Ships at 0, so the pipeline is still a
    # colour pass-through until something here is asked for.
    #
    # Panel order matches pipeline order, the way Edge Destruction's does:
    # highlight reconstruction, white balance, exposure, the tonal range,
    # contrast and black point, clarity, then vibrance and saturation, then the
    # LUT they all feed. The LUT *file* is not a parameter -- see server/lut.py
    # for why a name cannot be an index -- so it travels beside these values and
    # the client renders its picker directly above `lut_amount`.
    Param(
        "grade_recover", "Highlight Reconstruction", "Colour Grading",
        0.0, 1.0, 0.01, 0.0, "",
        "Rebuilds a blown highlight's clipped channels from the ones that "
        "survived, so the detail comes back instead of being dimmed. An 8-bit "
        "file clips per *channel*, not per pixel: a warm highlight hits the "
        "ceiling in red long before green and well before blue, so across a "
        "blown cloud red is a flat plateau while green and blue are still "
        "recording the scene's own gradient. This reads the colour of the light "
        "around the blown area from wherever it was still measurable and uses "
        "it to work out what the flattened channel was doing -- putting the "
        "value back above white, where it really was.\n"
        "\n"
        "The recovered value is above white, so the stage then rolls it back "
        "into view -- but only where it actually repaired something, so the rest "
        "of the frame is untouched. That makes this a self-contained repair: "
        "raise it and blown highlights regain their texture, with nothing else "
        "to set up. Highlights below still stacks on top if you want a broader, "
        "stronger roll across the whole top of the range.\n"
        "\n"
        "Only ever raises a clipped channel, never darkens anything, and it is "
        "a no-op on a photograph with nothing blown in it. Where *every* "
        "channel is at the ceiling -- a specular hit, a blown sky at noon -- "
        "there is genuinely nothing left in the file to work from and it leaves "
        "the pixel alone rather than inventing texture. The one expensive "
        "stage in this section: it costs two blurs of the frame. 0 = off.",
    ),
    Param(
        "grade_recover_radius", "Reconstruction Radius", "Colour Grading",
        4.0, 200.0, 1.0, 32.0, "px",
        "How far Highlight Reconstruction looks for a valid measurement of a "
        "clipped channel, at full resolution. This is the size of the blown "
        "area it can work across: a highlight wider than the radius has no "
        "surviving sample of its own colour anywhere in reach, so the estimate "
        "fades out toward the middle of it rather than being extrapolated from "
        "nothing. Larger reaches across bigger blown regions and costs more, "
        "and it borrows the local colour from further away -- which is only "
        "right while the light out there is the same light.\n"
        "\n"
        "It also sets how softly the repair blends into the surrounding frame.\n"
        "\n"
        "**This is by far the most expensive control in the app, and the cost "
        "grows faster than the number does** -- the tile overlap grows with it "
        "too, so a large radius spends most of its time rendering overlap it "
        "throws away. Measured on a 2400px proxy against a render that costs "
        "0.57s with this off: 1.6s at the 32px default, 3.6s at 80px, and 14s at "
        "200px. A full-resolution export multiplies all of that. Reach for the "
        "smallest radius that covers your blown areas, not the largest.",
        spatial=True,
    ),
    Param(
        "grade_temp", "Temperature", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "White balance, as a warm/cool shift. Positive is warmer (more red, "
        "less blue), negative cooler. Done as channel gains in *linear* light, "
        "which is where a white balance physically happens -- the same reason "
        "Pre Blur does its transfer round trip. Applied in gamma-encoded space "
        "instead it drags the shadows further than the highlights and reads as "
        "a tint painted over the picture rather than a different light.\n"
        "\n"
        "The gains are normalised against the luma weights, so warming a frame "
        "does not also brighten it -- measured, overall luminance holds to "
        "within 1% across the whole slider. Use it to set the light before the "
        "LUT below sees it; a LUT built for daylight has nothing sensible to do "
        "with a tungsten frame, and correcting afterwards fights the look. "
        "0 = off.",
    ),
    Param(
        "grade_tint", "Tint", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "The other half of white balance: green against magenta, at right "
        "angles to Temperature's blue/amber axis. Positive pushes toward "
        "magenta (red and blue up, green down), negative toward green. Same "
        "construction as Temperature and applied in the very same linear-light "
        "round trip -- a change of illuminant is a shift on both axes at once, "
        "so this and Temperature are one physical operation split across two "
        "sliders rather than two separate operations paying for the transfer "
        "twice.\n"
        "\n"
        "The gain is normalised against the luma weights the same way "
        "Temperature's is, so tinting a frame does not also expose it. "
        "0 = off.",
    ),
    Param(
        "grade_exposure", "Exposure", "Colour Grading",
        -2.0, 2.0, 0.01, 0.0, "EV",
        "A stops-based exposure multiply in linear light, ahead of Shadows "
        "and Highlights so their masks measure the frame at the light level "
        "actually being graded rather than the one that arrived -- raise this "
        "first and the two knees below still read the picture correctly. +1 "
        "is twice the light, -1 is half, and the sRGB encoding on the way "
        "back rolls the highlights off by itself rather than stretching them "
        "into a flat clip.\n"
        "\n"
        "Same construction as Tone Response's own Brightness, and kept as a "
        "separate control here rather than shared with it: that section is "
        "deferred and ships at 0, and this one exists so the light can be set "
        "before the film pipeline -- and the LUT below -- ever sees the "
        "picture. 0 = off.",
    ),
    Param(
        "grade_shadows", "Shadows", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "Opens or crushes the bottom half of the tonal range. Positive opens "
        "the shadows; negative takes them down toward black.\n"
        "\n"
        "Opening is a genuine recovery, not a brightness shift over the region "
        "that happens to be dark. The curve makes black an *asymptote*, so the "
        "whole of the range below the knee -- including anything that had "
        "already gone under zero on the way here -- is folded back into view "
        "with its tonal order intact, and two tones that differed before still "
        "differ after. It is strictly monotone at every setting, which is the "
        "property that separates recovering shadow detail from flattening it "
        "into a grey patch.\n"
        "\n"
        "It cannot clip and it cannot break a hue: the curve's output is bounded "
        "by the rail it approaches, and the whole pixel is scaled by one factor "
        "so hue and saturation are held exactly rather than approximately. It "
        "keys on the pixel's brightest channel, and it and Highlights touch "
        "opposite sides of the knee, so the two cannot reach into each other's "
        "range at all. 0 = off.",
    ),
    Param(
        "grade_highlights", "Highlights", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "The same control for the top half of the range, and negative is the "
        "direction that matters: it is the highlight recovery for the whole "
        "app. White becomes an asymptote instead of a wall, so everything from "
        "the knee upward -- including values *above* white, whether they came "
        "from Highlight Reconstruction, from Exposure, or from a bright source "
        "-- is rolled back into the visible range monotonically. Nothing "
        "flattens: two highlights that differed by a hair still differ "
        "afterwards, which is exactly what a clip destroys and what dimming a "
        "clipped patch cannot give back. Positive pushes highlights up instead.\n"
        "\n"
        "**This is the stage that makes Highlight Reconstruction visible.** "
        "Reconstruction puts the clipped channel's real value back above white; "
        "this is what brings it inside the range you can see. Reach for the "
        "pair together when a highlight is blown, and for this alone when it is "
        "merely bright.\n"
        "\n"
        "Gamut-safe, monotone and hue-exact for the same reasons Shadows is. "
        "0 = off.",
    ),
    Param(
        "grade_contrast", "Contrast", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "Steepness of the tonal range about the same middle grey the "
        "(deferred) film characteristic curve pivots on, but two-way and "
        "applied directly here rather than through a toe and shoulder: "
        "positive steepens, negative flattens toward the pivot. The gain is "
        "floored at 0 so no setting can invert the picture through grey -- at "
        "-1 the spread is reduced to a tenth of the original rather than "
        "crossing zero.\n"
        "\n"
        "Unlike the film curve further down, nothing here rolls off "
        "asymptotically, so a strong positive setting will clip highlights "
        "and shadows outright -- that is what a quick contrast control is "
        "expected to do, and Shadows/Highlights above exist for the clip-free "
        "version. 0 = off.",
    ),
    Param(
        "grade_black_point", "Black Point", "Colour Grading",
        0.0, 0.3, 0.005, 0.0, "",
        "Where the black clips. Unlike Shadows above, which is a broad, "
        "clip-free lift, this is the blunt Levels-style remap: every value at "
        "or below the chosen point is driven to 0 and 1 stays exactly at 1, "
        "so it genuinely crushes shadow detail rather than easing it -- that "
        "is the point of a black-point control. Deliberately one-directional: "
        "there is nothing below 0 to lift from, and a floor lift belongs to "
        "Shadows or to the (deferred) Base Fog instead.\n"
        "\n"
        "Reach for Shadows for a gentle, reversible lift and this for a hard, "
        "printable black. 0 = off.",
    ),
    Param(
        "grade_clarity", "Clarity", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "Two-way local contrast: positive adds it, negative takes it away. "
        "Above 0 it is the usual mid-frequency punch -- structure and texture "
        "come forward without the edge halos a small-radius sharpen leaves. "
        "Below 0 it flattens that same band, which reads as the soft, hazy, "
        "lifted look of light bouncing around inside the lens. Both are one "
        "band at one radius, so this is a different thing from Pre Blur (which "
        "destroys detail outright) and from Edge Softening (which only touches "
        "hard transitions).\n"
        "\n"
        "The two directions are deliberately not the same strength. Negative "
        "stops at exactly the point where the band is *gone*: -1 removes 100% "
        "of the local contrast at this radius and no setting can push past it "
        "into inverted contrast, which would put dark halos on the light side "
        "of every edge. Positive is free to go further and does. It runs on "
        "luminance only -- the detail it adds or removes goes to all three "
        "channels equally -- so it holds hue exactly and cannot drive a "
        "saturated area out of gamut, and it costs one single-channel blur "
        "instead of three. 0 = off.",
    ),
    Param(
        "grade_clarity_radius", "Clarity Radius", "Colour Grading",
        2.0, 80.0, 0.5, 14.0, "px",
        "Which band Clarity works on, as a radius at full resolution. Small is "
        "fine texture and starts competing with the grain further down the "
        "pipeline; large is broad shaping that reads as light rather than as "
        "detail. This is the one length in this section, so it scales with the "
        "photo like every other radius in the app -- and it is the only thing "
        "here that needs tile overlap, which is why Clarity is the only part of "
        "this section that costs anything measurable.",
        spatial=True,
    ),
    Param(
        "grade_vibrance", "Vibrance", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "The same saturation-weighted-against-itself construction as Tone "
        "Response's own Vibrance -- muted colour comes up, colour that is "
        "already strong is left alone -- kept as its own control here because "
        "this section runs before the film pipeline and the two have to stay "
        "independent: grading the picture and grading the negative are "
        "different jobs done at different points, and sharing one slider "
        "between them would mean the (deferred) Tone Response section could "
        "never be switched on later without re-touching a grade that was "
        "already finished. Negative drains the muted colour and leaves the "
        "vivid, which reads as bleached. 0 = off, and the pipeline stays a "
        "colour pass-through.",
    ),
    Param(
        "grade_saturation", "Saturation", "Colour Grading",
        -1.0, 1.0, 0.01, 0.0, "",
        "A flat saturation scale about each pixel's own luma. Unlike "
        "Vibrance, every pixel gains or loses the same proportion regardless "
        "of how saturated it already is, which is the classic blunt "
        "saturation control -- it will push an already-vivid area out of "
        "gamut before a muted one has caught up. -1 is fully neutral "
        "(equivalent to a monochrome conversion at this point in the "
        "pipeline), +1 doubles the existing chroma. Reach for Vibrance "
        "instead when skin and sky need to stay untouched while muted colour "
        "comes up. 0 = off.",
    ),
    Param(
        "lut_amount", "LUT Mix", "Colour Grading",
        0.0, 1.0, 0.01, 0.0, "",
        "How much of the selected 3D LUT is mixed in, as a straight cross-fade "
        "between the graded frame and its LUT'd self. 1 is the LUT as its "
        "author built it; part-way is the standard way to use a film LUT that "
        "is stronger than the photograph wants.\n"
        "\n"
        "0 = off, and with no LUT selected this does nothing whatever it says "
        "-- the server zeroes it in that case so 'show me the original' stays "
        "bit-exact. Picking a LUT raises it to 1 for you if it was sitting at "
        "0, because a picker that appears to do nothing is worse than one that "
        "commits.\n"
        "\n"
        "The LUT is applied display-referred, on the source, before every film "
        "stage -- which is what a LUT expects, and it means the grain, halation "
        "and texture below all land on the graded picture rather than being "
        "graded themselves.",
    ),
    # ------------------------------------------------------------- pre blur
    # The very first thing that touches the image -- ahead of pre-sharpen and
    # of every film stage. See step 0 in engine.render().
    Param(
        "pre_blur", "Pre Blur", "Pre Blur",
        0.0, 10.0, 0.05, 0.0, "px",
        "Gaussian blur on the source, at the top of the pipeline: before "
        "pre-sharpen and before anything films it. Radius at full resolution. "
        "It is not a second Micro-Blur despite being the same kernel -- this "
        "one runs before the masks are measured, so it also tells the grain "
        "where the detail went: edges read as softer, the smooth-area guard "
        "sees more smooth frame, and grain backs off with them. Micro-Blur is "
        "deliberately invisible to those masks. Use this to take a "
        "digital-sharp source down before the emulsion goes on, and pair it "
        "with Pre Sharpen at a tighter radius to put the bite back only where "
        "you want it. 0 = off.",
        spatial=True,
    ),
    # ---------------------------------------------------------- pre sharpen
    # Runs before every film stage, on the (optionally pre-blurred) input --
    # see step 0b in engine.render().
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
    Param(
        "chroma_grain", "Chroma Grain", "Grain Structure",
        0.0, 1.0, 0.01, 0.35, "",
        "0 = monochrome grain shared across channels. 1 = independent dye "
        "cloud noise per layer.",
    ),
    # ---------------------------------------------- luminance response
    # Was a section of its own until 2026-08-06, merged in here on request:
    # these six do not describe a stage, they describe *where the grain built
    # above them lands*, so a heading of their own read as a second thing to
    # set up rather than as the tail of this one. Panel order is the pipeline's:
    # build the field, then say which densities carry it.
    Param(
        "lum_low", "Shadow Knee", "Grain Structure",
        0.0, 0.5, 0.005, 0.15, "",
        "Lower edge of the peak-grain band. Below this, density falls off.",
    ),
    Param(
        "lum_high", "Highlight Knee", "Grain Structure",
        0.3, 1.0, 0.005, 0.65, "",
        "Upper edge of the peak-grain band. Above this, tightly packed "
        "silver suppresses visible grain.",
    ),
    Param(
        "shadow_falloff", "Shadow Falloff", "Grain Structure",
        0.02, 0.5, 0.005, 0.15, "",
        "How wide the fade-out is below the shadow knee. Independent of the "
        "knee position, so you can place the band anywhere and still control "
        "how gradual the transition into it is.",
    ),
    Param(
        "highlight_falloff", "Highlight Falloff", "Grain Structure",
        0.02, 0.5, 0.005, 0.25, "",
        "How wide the fade-out is above the highlight knee. Widen it for a "
        "gentler hand-off into clean highlights.",
    ),
    Param(
        "highlight_drop", "Highlight Suppression", "Grain Structure",
        0.0, 1.0, 0.01, 0.85, "",
        "How far grain is cut in dense highlights. 0.85 = 85% reduction.",
    ),
    Param(
        "shadow_drop", "Black Suppression", "Grain Structure",
        0.0, 1.0, 0.01, 0.6, "",
        "How far grain is cut in deep blacks.",
    ),
    # `seed` sits under everything it rerolls -- the same place `Texture Seed`
    # and `Global Seed` sit in theirs -- so the six above go in front of it.
    # It is no longer *last* in the group: the three placement controls below
    # it were moved here on request 2026-08-06 and the ask named this position.
    Param(
        "seed", "Seed", "Grain Structure",
        0.0, 9999.0, 1.0, 1234.0, "",
        "Deterministic seed for the grain lattice. Every other noise field in "
        "the pipeline -- the global layer, the edge envelope, the jitter "
        "displacement, the film-texture marks -- is offset from this one, so "
        "moving it rerolls the whole frame without changing any look.",
    ),
    # Where the grain lands, as opposed to what it is made of. All three
    # multiply the grain field at step 10 and none of them destroy an edge, so
    # they read as the tail of Grain Structure the way the luminance band does.
    # `highpass_radius` is the one with a foot in both camps -- the edge mask it
    # sizes also feeds Edge Erosion and Acutance -- and it follows the two
    # sliders that are its main consumers rather than staying behind. See
    # `docs/panel-layout.md`.
    Param(
        "edge_bias", "Edge Bias", "Grain Structure",
        0.0, 1.0, 0.01, 0.75, "",
        "Pushes grain onto high-contrast micro-edges and away from flat, "
        "smooth areas such as skies.",
    ),
    Param(
        "smooth_guard", "Smooth-Area Guard", "Grain Structure",
        0.0, 1.0, 0.01, 0.85, "",
        "Keeps grain out of genuinely featureless regions -- skin, clear sky, "
        "studio backdrops -- by measuring local contrast over a medium radius "
        "rather than brightness. 0 = off, 1 = smooth areas left clean.",
    ),
    Param(
        "highpass_radius", "High-Pass Radius", "Grain Structure",
        0.5, 5.0, 0.05, 2.0, "px",
        "Radius used to isolate micro-edges, at full resolution. It also sizes "
        "the mask Edge Erosion and Acutance work through, so widening it "
        "coarsens those two as well as the grain's edge bias.",
        spatial=True,
    ),
    # ----------------------------------------------------------------- edge
    Param(
        "edge_erosion", "Edge Erosion", "Edge Destruction",
        0.0, 1.0, 0.01, 0.5, "",
        "Modulates existing micro-detail by the grain field so grain erodes "
        "edge structure rather than sitting on top of it.",
    ),
    Param(
        "edge_chroma", "Edge Colour Fringing", "Edge Destruction",
        0.0, 1.0, 0.01, 0.5, "",
        "Runs edge erosion independently per colour layer, so eroded edges "
        "pick up coloured speckle. 0 = neutral erosion, 1 = full dye-layer "
        "fringing. It modulates the slider above and does nothing without it.",
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
    # ----------------------------------------------------- edge destruction
    # (scatter and micro-blur -- formerly their own "Optical" group, merged in
    # here 2026-08-04 on request; the engine's step numbering is unaffected,
    # this is a UI grouping only.)
    #
    # Scatter first, micro-blur last, in the panel and in the pipeline alike --
    # see step 1 in engine.render(). The order is the point: scatter gets the
    # source's own detail to take apart, and the blur then averages what is
    # left rather than handing scatter a frame that is already smooth.
    #
    # Scatter: diffusion resolved as discrete deflections instead of as an
    # average. See _scatter for why that is not a blur.
    Param(
        "scatter", "Scatter", "Edge Destruction",
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
        "scatter_radius", "Scatter Reach", "Edge Destruction",
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
        "scatter_pattern", "Scatter Pattern", "Edge Destruction",
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
        "scatter_spread", "Reach Spread", "Edge Destruction",
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
        "scatter_cell", "Scatter Clump", "Edge Destruction",
        0.1, 5.0, 0.1, 1.0, "px",
        "How big a piece of the picture moves as one. At 1 every pixel "
        "chooses for itself and the image crumbles; larger values move whole "
        "tiles of detail intact, so structure survives the trip and lands "
        "somewhere else. Past about 4px the tiles start reading as tiles -- "
        "which is a look, a shattered plate rather than a soft one, but it is "
        "no longer subtle. Held in full-res pixels like every other length.\n"
        "\n"
        "Below one *working* pixel there is nothing left to resolve -- one "
        "choice per pixel is already the finest this can be -- so the bottom "
        "of the range is only reachable through supersampling, which is what "
        "makes a working pixel smaller than a real one. At supersample 2 that "
        "puts the floor at 0.5; below it every setting renders identically.",
        spatial=True,
    ),
    Param(
        "micro_blur", "Micro-Blur", "Edge Destruction",
        0.0, 3.0, 0.01, 0.45, "px",
        "Light diffusion through the gel layers, as an average: every pixel "
        "is mixed with its neighbours. That is the smooth half of diffusion, "
        "and it costs texture along with the edges -- Scatter above is the "
        "same physics without the averaging. Last in the light path, so it "
        "averages whatever scatter has already pulled apart rather than "
        "handing scatter a frame that is smooth before it starts. Applied to "
        "the base image before grain injection so grain stays sharp against a "
        "soft base.",
        spatial=True,
    ),
    # -------------------------------------------------------- anti aliasing
    # Step 1c, in the optical block -- an anti-alias filter is a plate in the
    # light path, not a retouch. Ships at 0 like every other optional stage.
    Param(
        "aa_strength", "AA Strength", "Anti Aliasing",
        0.0, 3.0, 0.01, 0.0, "",
        "Removes stair-stepping from hard edges in the source -- the ragged "
        "diagonal you get from an upscaled JPEG, a screenshot or a CG render. "
        "It filters *along* each edge rather than across it, so the jaggies "
        "average out while the edge stays as sharp as it was. That is what "
        "separates it from Micro-Blur and Edge Softening, which both work "
        "across the edge and cost sharpness. 0 = off.\n"
        "\n"
        "Past 1 it runs the filter again, re-aiming along the contour each "
        "time, which is what makes it bite on aliasing a single pass barely "
        "touches: measured on a deliberately-aliased diagonal, 1 removes 34% "
        "of the contour's raggedness, 2 removes 52% and 3 removes 64%, while "
        "across-edge sharpness falls only from 86% to 70% over that whole "
        "range. Repeating is the right lever rather than a longer AA Radius -- "
        "a stair-step is one pixel wide by definition, so reaching further "
        "averages away the shape the contour actually has instead of the "
        "wobble on it. Whole numbers are whole passes and anything between "
        "fades the last one in.",
    ),
    Param(
        "aa_radius", "AA Radius", "Anti Aliasing",
        0.2, 4.0, 0.05, 1.0, "px",
        "How far along the edge each pixel is averaged, at full resolution. "
        "A stair-step is one pixel by definition, so around 1 is the honest "
        "setting and the default. Larger values start rounding off genuine "
        "corners and small detail along with the jaggies -- useful if the "
        "source was upscaled and its steps are several pixels wide, wrong "
        "otherwise.",
        spatial=True,
    ),
    Param(
        "aa_edge_only", "Edge Only", "Anti Aliasing",
        0.0, 1.0, 0.01, 0.7, "",
        "How strictly the filter is held to hard edges. At 1 it only touches "
        "borders that step a long way in brightness, so fabric, foliage and "
        "grain are untouched -- fine texture measures an order of magnitude "
        "below a real border, which is the gap this keys on. At 0 it runs "
        "everywhere, which suits a CG render that aliases on gentle steps and "
        "will visibly soften a photograph's texture.",
    ),
    Param(
        "global_smooth", "Global Smoothness", "Anti Aliasing",
        0.0, 1.0, 0.01, 0.0, "",
        "Blurs the Global Grain layer by up to half a clump, rounding its "
        "grains off and softening the boundaries between them. It used to be "
        "the cure for that layer breaking into rectangular blocks at large "
        "sizes; the field is no longer built on a lattice that does that, so "
        "this is now a shape control rather than a repair -- reach for it "
        "when the grain reads as too crisp. Strength is held constant as you "
        "raise it, so it changes the shape of the grain and not how much "
        "there is. Scaled to Global Size, so one setting stays right as you "
        "resize the clumps. Here rather than under Global Grain because it is "
        "the same job as the sliders above it: taking the pixel grid back out.",
    ),
    # --------------------------------------------------------- global grain
    # Applied last -- see step 13 in engine.render(). Five layers, built from
    # one set of shape controls (Size Min, Size Max, Smoothness, Chroma) on five
    # different seeds, each with its own amount slider and its own mask. Every
    # amount ships at 0 so the section never alters an existing look until asked
    # for, and the blend mode ships on Add, which is what it has always done.
    #
    # The menu goes first because it governs all five sliders under it.
    Param(
        "global_blend", "Blend Mode", "Global Grain",
        0.0, float(len(GLOBAL_BLENDS) - 1), 1.0, 0.0, "",
        "How all five layers in this section are combined with the picture "
        "underneath, the way a layer's blend mode works in an image editor. "
        "The grain is treated as an image that is mid grey where there is no "
        "grain, and each layer's amount and mask together act as its "
        "opacity -- so every mode still fades to nothing as you take the "
        "sliders down.\n"
        "\n"
        "Add is the plain sum this section has always used, and the only one "
        "that is completely even-handed: it lifts and drops every tone by the "
        "same amount, which is why it can lift a black. Overlay and Soft Light "
        "are the two that behave like film -- both leave mid tones grainiest "
        "and taper off toward black and white on their own, Overlay firmly and "
        "Soft Light gently, so grain settles into the picture instead of "
        "sitting on it. Hard Light is Overlay driven from the grain rather "
        "than the image; it is much stronger and clips readily.\n"
        "\n"
        "Multiply and Screen are the odd pair, and worth knowing before you "
        "reach for them: neither has a neutral grey, so they darken or lighten "
        "the whole frame as well as texturing it, and the amount sliders are "
        "the only thing holding that back. Multiply grains the highlights and "
        "leaves shadows alone; Screen does the reverse. Use them low.",
        choices=GLOBAL_BLENDS,
    ),
    Param(
        "global_intensity", "Global Intensity", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "A flat grain layer over the finished frame, at one strength "
        "everywhere. Unlike the main grain it ignores the luminance band, the "
        "edge bias and the smooth-area guard, so it reaches skies, skin and "
        "blown highlights that those masks deliberately keep clean. 0 = off. "
        "Because nothing holds it back it bites far harder than the main "
        "Intensity slider at the same number -- 32 here measures 8% luminance "
        "sigma against 3.5% there. 5-20 is the usable range.\n"
        "\n"
        "This is the unmasked layer of the five. The four below it are the "
        "same grain following the picture instead of ignoring it; because this "
        "one goes on first and everywhere, nothing they mask away is ever left "
        "perfectly clean.",
    ),
    Param(
        "global_size", "Global Size Min", "Global Grain",
        0.1, 20.0, 0.05, 1.6, "px",
        "Clump diameter of the global layer, at full resolution -- the "
        "smallest a clump can be, and the only size that exists while Global "
        "Size Max stays at or below it. Set it apart from Clump Size and the "
        "two layers read as separate structures; match them and it just "
        "thickens the main grain. Large sizes are safe at any setting now: "
        "the layer is drawn as scattered grains on a lattice tilted off the "
        "pixel grid, so it no longer breaks into rectangular blocks the way "
        "it did past about 8px.",
        spatial=True,
    ),
    Param(
        "global_size_max", "Global Size Max", "Global Grain",
        0.1, 20.0, 0.05, 1.6, "px",
        "The largest a global-grain clump can be. At or below Global Size Min "
        "every clump renders at exactly Min; raised above it, each clump "
        "independently draws its own diameter somewhere between the two, so "
        "the layer reads as real crystals of differing sizes rather than one "
        "uniform grain. It is a range, not a switch -- widening it changes "
        "how much the sizes vary and nothing else about how the layer is "
        "drawn. A very wide gap leaves visible clear patches between clumps; "
        "real grain has them too, but narrow the gap if it reads as sparse.",
        spatial=True,
    ),
    Param(
        "global_chroma", "Global Chroma Grain", "Global Grain",
        0.0, 1.0, 0.01, 0.0, "",
        "The same job as Chroma Grain under Grain Structure, for this layer: "
        "0 = one monochrome field shared by all three channels, 1 = an "
        "independent field per channel so the layer carries colour speckle "
        "rather than pure luminance noise. Unlike that slider this one holds "
        "the layer's amplitude to within 3% across its whole range, so it "
        "changes colour without changing loudness. Its own slider because the "
        "two "
        "layers model different things -- the main grain is the negative's "
        "emulsion, where the dye layers are genuinely separate, while this one "
        "stands in for print stock and scanner noise and is often wanted "
        "neutral over a chromatic main grain. Ships at 0, which is what this "
        "layer has always been.\n"
        "\n"
        "Governs all five layers in the section, not just Global Intensity's. "
        "Note what it does and does not do: it colours each *grain*, keeping "
        "one grain's edge in the same place in all three channels. The four "
        "sliders below are the other thing -- separate grains in separate "
        "places, picked out by what colour the picture already is.",
    ),
    # The source-masked set -- see step 13 in engine.render(). Four more layers
    # of the same grain on four more seeds, each multiplied by an envelope read
    # off the picture. They stack on the flat layer above rather than replacing
    # it, so all four at 0 is exactly the layer every preset was dialled in
    # against, and with them up no area is ever left grain-free.
    #
    # **The colour names are the mask, never the output channel.** Each of these
    # is a full-colour grain field written into all three channels and taking
    # Chroma Grain like every other layer here; "Red" says only that it shows up
    # where the picture is red. Saying otherwise in a help string would be the
    # easiest thing in this section to get wrong.
    Param(
        "global_src_r", "Source Red", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "A grain layer that only shows up in the red parts of the picture -- "
        "stronger the redder and the brighter an area is, and absent from "
        "everything neutral. It is the *mask* that is red, not the grain: this "
        "is the same full-colour field as Global Intensity's, on its own seed, "
        "and it takes Chroma Grain like the rest.\n"
        "\n"
        "Its own seed is the point of having four of these -- a red-masked "
        "grain and a blue-masked grain are separate particles in separate "
        "places, the way three emulsion layers are. That is the other half of "
        "Global Chroma Grain, which colours one shared grain in place. Reach "
        "for these when you want grain to pick out what is already in the "
        "frame.\n"
        "\n"
        "On the same scale as Global Intensity before the mask takes its "
        "share, and the mask takes a lot: hue rarely dominates by more than "
        "0.3-0.5 in a real photograph, so expect to run these well above the "
        "number you would use up there. 0 = off, and off is free -- the field "
        "is never built.",
    ),
    Param(
        "global_src_g", "Source Green", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The green member of the source-masked set: the same grain again, on "
        "its own seed, showing up in the green parts of the picture. Foliage, "
        "in other words, and not much else in most frames -- it is the one of "
        "the three that tends to find a single subject rather than spreading "
        "over the whole image.\n"
        "\n"
        "The three colour masks are mutually exclusive by construction: only "
        "one channel can be the dominant one at any pixel, so no two of them "
        "ever land on the same spot and turning all three up cannot pile them "
        "into a hot patch.",
    ),
    Param(
        "global_src_b", "Source Blue", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The blue member of the source-masked set: the same grain again, on "
        "its own seed, showing up in the blue parts of the picture -- skies "
        "and water, mostly, which is exactly where the flat layer is most "
        "obvious and hardest to place. The blue-sensitive layer is the "
        "grainiest one in most real colour stocks, so running this above the "
        "other two is the closest thing here to that behaviour.",
    ),
    Param(
        "global_src_l", "Source Lightness", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The odd one out, and the one to try first. Instead of a colour it "
        "follows *exposure*, and not as a ramp: it is loudest at mid grey and "
        "fades away toward both ends, so highlights stay clean and shadows "
        "stay clean and everything between them takes the grain. Full strength "
        "at mid grey, about a tenth of it near black and near white, nothing "
        "at all at either extreme.\n"
        "\n"
        "That is where film grain actually lives -- a blown highlight has no "
        "silver left to be grainy and a solid black has none developed -- so "
        "this is the layer that reads as emulsion rather than as noise laid "
        "over a photograph. Unlike the three above it, it does not care what "
        "colour anything is, so it works on a monochrome frame.",
    ),
    Param(
        "global_opacity", "Global Opacity", "Global Grain",
        0.0, 1.0, 0.01, 1.0, "",
        "How much of the global layer is mixed in. It multiplies with Global "
        "Intensity -- intensity is how coarse and strong the layer is in its "
        "own right, opacity is how far it is dialled back over the image. It "
        "governs the whole section, the four source-masked layers included, so "
        "it is the one dial that takes all five down together.",
    ),
    Param(
        "global_seed", "Global Seed", "Global Grain",
        0.0, 9999.0, 1.0, 0.0, "",
        "Reshuffles where every grain in this section falls, without touching "
        "the main grain you have already dialled in. All five layers reroll "
        "together and each stays independent of the other four -- what changes "
        "is the whole set at once, not their relationship to each other.\n"
        "\n"
        "It is an *offset* on the Seed slider under Grain Structure rather than "
        "a seed in its own right, which is why it starts at 0 and not at some "
        "arbitrary number. Two things fall out of that and both are wanted: "
        "moving Seed still rerolls the entire frame including this section, the "
        "way its help promises; and 0 is the exact layer every preset was "
        "dialled in against, whatever seed that preset happens to use.\n"
        "\n"
        "Reach for it when the global layer has landed a clump somewhere "
        "unhelpful -- across an eye, along a horizon -- and everything else is "
        "already right.",
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
        "How many specks land on the frame -- a count, not a strength, so it "
        "means the same thing whatever the image size. **Exact**: the specks "
        "are drawn one at a time from a list anchored to the frame, so 20 here "
        "is twenty specks, not roughly twenty. Raising it adds specks and "
        "leaves the ones already there where they were. 0 = none.",
    ),
    Param(
        "dust_balance", "Dust Dark / Light", "Film Texture",
        -1.0, 1.0, 0.01, 0.0, "",
        "Which way the population leans. **-1 is every speck dark, +1 is every "
        "speck bright, 0 is an even mix.** Dark specks are opaque motes sitting "
        "on the emulsion; bright ones are pinholes in it and lint on the "
        "scanner glass, and a frame of only dark specks reads as sensor dirt "
        "rather than as film.\n"
        "\n"
        "The split is exact and it converts specks *in place*: moving this "
        "changes which of the specks are bright without moving any of them, so "
        "you can find the ratio you want without the frame reshuffling under "
        "you. Dust Count stays the total either way.",
    ),
    Param(
        "dust_size", "Dust Size", "Film Texture",
        0.5, 120.0, 0.05, 2.0, "px",
        "Mean speck diameter at full resolution. Small is scanner dust; large "
        "is lint and debris on the negative. Individual specks are drawn "
        "around this rather than all cut to it -- real debris comes in a range "
        "of sizes, and a frame of identically-sized specks is the clearest "
        "sign the texture was generated.",
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
        "specks stay crisp and others go soft at any setting. Soft specks "
        "also come out fainter, which is what out-of-focus debris actually "
        "does. 0 = all crisp.",
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
        "How many hairs and fibres are lying on the frame, printing as dark "
        "wandering filaments. **Exact**: one hair is one hair. Each is drawn "
        "as its own filament with its own direction, curl and taper, so "
        "raising the count adds hairs and leaves the ones already there "
        "untouched. 0 = none.",
    ),
    Param(
        "hair_length", "Hair Length", "Film Texture",
        20.0, 600.0, 5.0, 160.0, "px",
        "Mean hair length at full resolution -- independent of how many there "
        "are, and drawn around rather than cut to, so a frame carries long "
        "fibres and short ones. It also sets how far a hair curls and wanders "
        "over its own length, because a longer filament bends more.",
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
    # --------------------------------------------------------------- output
    # The master blend, applied after literally everything -- and after the
    # supersample pool, which is the only place it can be bit-exact at 0.
    # Defaults to 1.0, so it is the one parameter whose neutral value is not
    # zero and the one that must stay out of NEUTRAL_ZERO.
    Param(
        "master_opacity", "Overall Opacity", "Output",
        0.0, 1.0, 0.01, 1.0, "",
        "How much of the finished result is laid over the untouched photo. "
        "1 = the full effect, 0 = the original returned bit for bit, and "
        "anything between is a straight cross-fade -- so it dials back "
        "everything at once: grain, halation, softening, marks, the lot. "
        "Reach for it when a preset is right in character but too strong, "
        "instead of walking a dozen sliders down together.\n"
        "\n"
        "Not to be confused with Global Opacity under Global Grain, which "
        "only mixes that one noise layer. This one is the whole pipeline.",
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
        lut = raw.get("lut")
        out.append({
            "name": f.stem,
            "values": sanitize(values),
            # Which 3D LUT the look wants, by name. A sibling key rather than a
            # value, for the reason server/lut.py sets out -- it is a resource,
            # not a quantity. Unresolvable names (a renamed file, an upload from
            # a previous run) degrade to no LUT rather than erroring.
            "lut": lut if isinstance(lut, str) and lut else None,
            # Size the preset was dialled in on, so it can be rescaled onto a
            # different photo. Absent in older files -> no scaling, which is
            # the pre-existing behaviour rather than a guess.
            "reference_mp": float(ref) if isinstance(ref, (int, float)) and ref else None,
        })
    return out
