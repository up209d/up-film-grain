"""The JSON the client builds its whole panel from."""
from __future__ import annotations

from dataclasses import asdict

from .definitions import PARAMS
from .param import GROUPS
from .presets import DEFAULT_PRESET, DEFAULT_REFERENCE_MP, load_presets
from .registry import NEUTRAL_ZERO, neutral_values


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
        # *Which* of those keys are the amounts -- the ones that decide whether
        # a stage runs at all. Shipped rather than inferred for the reason
        # `NEUTRAL_ZERO` itself is an explicit list: "is this an amount?" is not
        # something a value map can be read for. The client used to answer it by
        # comparing every key against `neutral`, which quietly made a *seed*
        # count as a change to the picture -- so a rerolled seed made the
        # untouched photo stop reading as untouched. `global_seed` is the proof
        # that guessing does not work either: its default is 0 and it is not an
        # amount.
        "neutral_zero": list(NEUTRAL_ZERO),
        "default_reference_mp": DEFAULT_REFERENCE_MP,
    }
