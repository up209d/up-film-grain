from __future__ import annotations

from ..param import Param

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
        # Topped at 64, not 200 (2026-08-08). **The old ceiling was
        # unsupportable at any tile size on any machine**, not merely slow.
        # `pad_for` triples the kernel sum and `tile_for` then shrinks the tile
        # to fit the memory budget, so a radius costs three times over: kernel
        # work, a smaller tile, and overdraw on every one of them. Measured at
        # 12MP with reconstruction on, one fresh process each:
        #
        #   32 (default)  pad  252  tile 2208   4 tiles  1.32x   3.12s
        #   100           pad  558  tile 1568   6 tiles  2.14x  10.45s
        #   200           pad 1008  tile  768  24 tiles  9.14x  79.67s
        #
        # 25x from one slider, and 11.0x overdraw at 24MP. Holding overdraw at
        # 2x would need `tile >= 4.83 * pad`, i.e. a 5072 tile whose working set
        # at supersample 2 is ~221GB -- there is no tiling that rescues it. On
        # the CPU it is worse still: 100 takes 56.5s for a *proxy* against 5.7s
        # at the default, and 200 did not finish in four minutes.
        #
        # 64 holds pad at ~409. No shipped preset goes near either number, so
        # nothing in `presets/` changes; `sanitize` clamps anything that did.
        4.0, 64.0, 1.0, 32.0, "px",
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
        # Widened to +/-5 stops on request (2026-08-16), from +/-2. Nothing in
        # the stage needed changing -- it is a multiply in linear light, so the
        # range is a question about how far the control should reach rather
        # than about what it can express. +5 is 32x the light, which will clip
        # most frames outright; the sRGB encoding still rolls the top rather
        # than stretching it flat, and Highlights below is still the clip-free
        # way to bring a bright frame back.
        -5.0, 5.0, 0.01, 0.0, "EV",
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
        # Topped at 48, not 80 (2026-08-08), for `grade_recover_radius`' reason
        # in a milder form: at 80 the pad goes 108 -> 492 and a 24MP export
        # renders 2.23x its own area. Measured at 12MP, 1.76s -> 2.66s. 48 keeps
        # it near 1.5x. No shipped preset exceeds the default 14.
        2.0, 48.0, 0.5, 14.0, "px",
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
]
