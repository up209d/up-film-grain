"""The only door parameters enter the engine through."""
from __future__ import annotations

from .definitions import PARAMS
from .registry import DEFAULTS, PARAM_BY_KEY


def sanitize(raw: dict | None) -> dict[str, float]:
    """Clamp incoming values into range and fill any missing key with its default.

    Unknown keys are dropped. This is the only place params enter the engine,
    so the renderer can assume every value is present and in range.
    """
    out = dict(DEFAULTS)
    if not raw:
        return out
    for key, value in raw.items():
        p = PARAM_BY_KEY.get(key)
        if p is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        out[key] = max(p.min, min(p.max, v))
    return out
