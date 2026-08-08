from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
        "smooth areas such as skies.\n"
        "\n"
        "**What counts as an edge is set under Edge Destruction** -- High-Pass "
        "Radius, Edge Sensitivity and Edge Colour Sensitivity build one mask "
        "and both sections read it. This slider is the other half of the "
        "question: not what an edge is, but how much the grain should care. At "
        "0 it ignores the mask entirely, which is also the setting at which "
        "those three stop affecting grain at all.",
    ),
    Param(
        "smooth_guard", "Smooth-Area Guard", "Grain Structure",
        0.0, 1.0, 0.01, 0.85, "",
        "Keeps grain out of genuinely featureless regions -- skin, clear sky, "
        "studio backdrops -- by measuring local contrast over a medium radius "
        "rather than brightness. 0 = off, 1 = smooth areas left clean.",
    ),
]
