"""One mixin per pipeline stage. ``GrainEngine`` composes them.

They are mixins rather than standalone collaborators because every stage
reads the same engine state (device, caches) and the test-suite calls them
as engine methods. Stage *order* is not defined here -- it lives in
``RenderMixin.render``, and docs/pipeline-order.md says what moving one
breaks.
"""
from __future__ import annotations

from .colour_grade import ColourGradeMixin
from .edge import EdgeMixin
from .film_texture import FilmTextureMixin
from .global_grain import GlobalGrainMixin
from .halation import HalationMixin
from .render import RenderMixin
from .sharpen import SharpenMixin
from .tone import ToneMixin

__all__ = [
    "ColourGradeMixin",
    "EdgeMixin",
    "FilmTextureMixin",
    "GlobalGrainMixin",
    "HalationMixin",
    "RenderMixin",
    "SharpenMixin",
    "ToneMixin",
]
