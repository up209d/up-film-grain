from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
