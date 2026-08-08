from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
