"""The invariant checks, one module per area of the engine.

Importing this package registers every module into `harness.SUITES`. The order
below is the order they ran in as one function, and it is the order the runner
prints them in -- a diff against an older log should be a diff about the engine,
not about scheduling.

Grouping is by *what you would re-run after touching one thing*. Touch
`engine/stages/global_*` and `verify.py global_grain global_layers global_field
global_mix` is the honest re-run; the whole suite is not.
"""

from __future__ import annotations

from tests.checks import (  # noqa: F401
    tiling, colour, presets, grading, response,
    global_grain, global_layers, global_field, global_mix,
    edges, scatter, halation, sharpen, film_texture, film_tiling,
    internals, imageio,
)

ORDER = [
    "tiling", "colour", "presets", "grading", "response",
    "global_grain", "global_layers", "global_field", "global_mix",
    "edges", "scatter", "halation", "sharpen", "film_texture", "film_tiling",
    "internals", "imageio",
]
