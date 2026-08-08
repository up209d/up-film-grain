"""The JSON the client builds its whole panel from."""
from __future__ import annotations

from dataclasses import asdict

from .definitions import PARAMS
from .param import GROUPS
from .presets import DEFAULT_PRESET, DEFAULT_REFERENCE_MP, load_presets
from .registry import neutral_values


def schema() -> dict:
    """JSON-serialisable schema for the client."""
    presets = load_presets()
    names = {p["name"] for p in presets}
    return {
        "groups": GROUPS,
        "params": [asdict(p) for p in PARAMS],
        "presets": presets,
        # The preset the client starts on, or None to start on the raw
        # parameter defaults. Reported rather than assumed, so a missing file
        # degrades to "no starting preset" instead of erroring.
        "default_preset": DEFAULT_PRESET if DEFAULT_PRESET in names else None,
        # What the client's "Original" button applies.
        "neutral": neutral_values(),
        "default_reference_mp": DEFAULT_REFERENCE_MP,
    }
