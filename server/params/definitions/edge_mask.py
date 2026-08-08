"""The edge mask every Edge Destruction control is weighted by.

Moved out of `Grain Structure` on 2026-08-09, on request. These three define
*what counts as an edge* -- the scale it is measured at, how hard it has to
step, and whether a colour boundary counts -- and every control below them in
the section reads the result.

**They are consumed before the section they now sit in.** The mask is built in
the Grain Structure stage because the grain's own `edge_bias` needs it there,
so these are the one place the panel deliberately reads ahead of execution.
That is the right trade: they are edge controls to anyone using the app, and
the alternative is three sliders in a section whose heading does not mention
edges. `edge_bias` stays behind in Grain Structure because it is the opposite
question -- not what an edge is, but how much the grain should care.
"""

from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    Param(
        "highpass_radius", "High-Pass Radius", "Edge Destruction",
        # Topped at 24, not 5 (2026-08-09). A high-pass only responds to
        # structure *finer* than its radius, so a soft edge -- a portrait's
        # skin-against-background boundary, anything shot wide open -- was
        # invisible to the mask however hard it stepped. Measured on a 0.26-luma
        # skin boundary, the mask peaks at 0.16 for a 30px ramp at radius 2 and
        # 0.97 at radius 12; jitter on that same edge goes 0.09 -> 2.89 levels.
        # 24 costs 2.09x overdraw at 24MP, which is where the other radii in
        # this app are capped.
        0.5, 24.0, 0.05, 2.0, "px",
        "The scale edges are measured at, in full-resolution pixels -- and so "
        "**which edges the section can see at all**. A high-pass only responds "
        "to structure finer than its radius, so a transition that ramps over "
        "more pixels than this reads as flat ground however hard it steps.\n"
        "\n"
        "2 finds micro-edges and film grain's own scale. Raise it when Jitter, "
        "Sanding or Erosion ignore boundaries you can see: a soft, shallow-"
        "depth-of-field subject needs 12-24. It also sizes the mask the grain's "
        "Edge Bias reads, so widening it coarsens where grain lands as well.\n"
        "\n"
        "**Edge Softening does not use this** -- it measures at its own "
        "Softening Radius, which is the one to raise for that stage.",
        spatial=True,
    ),
    Param(
        "edge_sensitivity", "Edge Sensitivity", "Edge Destruction",
        0.1, 4.0, 0.05, 1.0, "x",
        "How hard a transition has to step before it counts as a full-strength "
        "edge. This is the reference the whole edge family is measured "
        "against -- Edge Softening, Jitter, Sanding, Erosion and the grain's "
        "own edge bias all read the same mask -- and it was a fixed internal "
        "number until 2026-08-09.\n"
        "\n"
        "1 is that number and changes nothing. Raise it and gentler edges reach "
        "full strength, so the section acts on far more of the picture; lower "
        "it and only the hardest borders qualify. Reach for it when Edge "
        "Destruction appears to be skipping edges you can plainly see: a "
        "soft-lit or low-contrast frame can sit almost entirely under the "
        "default reference.",
    ),
    Param(
        "edge_chroma_sense", "Edge Colour Sensitivity", "Edge Destruction",
        0.0, 1.0, 0.01, 1.0, "",
        "Whether a boundary between two *colours* counts as an edge when both "
        "sides are equally bright. The masks used to be built from luminance "
        "alone, so a red-to-green transition at matched luma was flat to them "
        "and every control in Edge Destruction skipped it -- measured 17-21x "
        "weaker than the same-size luminance edge. Photographs are full of "
        "them: foliage against sky, skin against fabric.\n"
        "\n"
        "At 1 an edge is as strong as its strongest single channel, which is "
        "what the eye sees. At 0 the mask is the old luminance-only one, bit "
        "for bit. Neutral, greyscale content renders identically at every "
        "setting -- there is no colour difference to find -- so this only ever "
        "adds edges, never moves the ones already there.",
    ),
]
