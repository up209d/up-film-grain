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
    *_edge,
    *_halation,
    *_edge_destruction,
    *_anti_aliasing,
    *_global_grain,
    *_sharpening,
    *_film_texture,
    *_output,
]
