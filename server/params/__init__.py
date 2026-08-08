"""Parameter schema for the grain engine.

This module is the single source of truth for every tunable knob. The engine
reads defaults from here and the web client builds its slider panel from the
same schema (served by ``GET /api/params``), so the UI can never drift out of
sync with what the renderer actually accepts.

Split into a package on 2026-08-08: ``param`` holds the record type,
``definitions/`` holds the controls one module per panel section, and the
rest is derived from them. Import from ``server.params`` as before -- this
module re-exports the whole surface.
"""
from __future__ import annotations

from .definitions import PARAMS
from .param import GLOBAL_BLENDS, GROUPS, Param
from .presets import (
    DEFAULT_PRESET, DEFAULT_REFERENCE_MP, PRESET_DIR, load_presets,
)
from .registry import (
    DEFAULTS, NEUTRAL_ZERO, PARAM_BY_KEY, is_neutral, neutral_values,
    rescale, scale_factor,
)
from .sanitize import sanitize
from .schema import schema

__all__ = [
    "DEFAULTS",
    "DEFAULT_PRESET",
    "DEFAULT_REFERENCE_MP",
    "GLOBAL_BLENDS",
    "GROUPS",
    "NEUTRAL_ZERO",
    "PARAMS",
    "PARAM_BY_KEY",
    "PRESET_DIR",
    "Param",
    "is_neutral",
    "load_presets",
    "neutral_values",
    "rescale",
    "sanitize",
    "scale_factor",
    "schema",
]
