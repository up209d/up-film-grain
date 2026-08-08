from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
