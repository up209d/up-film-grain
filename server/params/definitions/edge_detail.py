"""Edge Erosion, Colour Fringing and Acutance.

Split out of `edge.py` on 2026-08-08 so the panel can list them *after* the
softening controls, which is the order they run in. All three add fine
high-frequency structure and every other control in the section removes it,
so running them first made them do nothing measurable -- 0.01% of pixels
moved by more than one 8-bit level, against 2.63% when they run last.

Still the `Edge Destruction` group: they are the same section to a user,
and the group name is what the panel heading reads.
"""

from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
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
]
