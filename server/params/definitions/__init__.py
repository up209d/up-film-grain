"""Every control the app has, one module per panel section.

Adding a parameter means adding one ``Param`` to the section it belongs to
and reading ``p["key"]`` in the engine -- the UI builds itself from this.
The concatenation order below is the order controls appear *within* a
group; ``GROUPS`` in ``..param`` orders the groups themselves.
"""
from __future__ import annotations

from ..param import Param
from .colour_grading import PARAMS as _colour_grading
from .pre_blur import PARAMS as _pre_blur
from .pre_sharpen import PARAMS as _pre_sharpen
from .tone import PARAMS as _tone
from .grain import PARAMS as _grain
from .luminance_response import PARAMS as _luminance_response
from .edge import PARAMS as _edge
from .edge_mask import PARAMS as _edge_mask
from .edge_detail import PARAMS as _edge_detail
from .halation import PARAMS as _halation
from .edge_destruction import PARAMS as _edge_destruction
from .anti_aliasing import PARAMS as _anti_aliasing
from .global_grain import PARAMS as _global_grain
from .sharpening import PARAMS as _sharpening
from .film_texture import PARAMS as _film_texture
from .output import PARAMS as _output

PARAMS: list[Param] = [
    *_colour_grading,
    *_pre_blur,
    *_pre_sharpen,
    *_tone,
    *_grain,
    *_luminance_response,
    # `_edge_destruction` before `_edge`, so the panel lists Edge Destruction's
    # softening controls above Edge Erosion, Colour Fringing and Acutance --
    # which is the order they run in. Those three add fine high-frequency
    # structure and every other control in the section removes it, so they run
    # last or they do nothing at all: measured 0.01% of pixels moved when they
    # ran first, 2.63% when they run last.
    # The mask definition first: everything after it is weighted by it.
    *_edge_mask,
    *_edge,
    *_edge_destruction,
    *_edge_detail,
    *_halation,
    *_anti_aliasing,
    *_global_grain,
    *_sharpening,
    *_film_texture,
    *_output,
]
