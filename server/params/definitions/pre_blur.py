from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
