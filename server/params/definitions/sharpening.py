from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
