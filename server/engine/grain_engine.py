from __future__ import annotations

import collections

import torch

from .device import pick_device
from .stages import (
    ColourGradeMixin, EdgeMixin, FilmTextureMixin, GlobalGrainMixin,
    HalationMixin, RenderMixin,
)
from .tiling import TilingMixin


class GrainEngine(
    ColourGradeMixin,
    HalationMixin,
    EdgeMixin,
    GlobalGrainMixin,
    FilmTextureMixin,
    RenderMixin,
    TilingMixin,
):
    """The pipeline, composed from one mixin per stage.

    Holds only what the stages share: the device and the two texture caches.
    The stage bodies live in ``.stages``; the tiling and supersampling entry
    points live in ``.tiling``.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        # Global Grain texture cache -- see `_global_grain_field`. An
        # OrderedDict used as an LRU: `move_to_end` on a hit, pop from the front
        # when over `_GG_CACHE_BYTES`.
        #
        # A plain dict needs no lock because `main._RENDER_LOCK` serialises every
        # render. That is a load-bearing assumption borrowed from the caller: if
        # renders ever run concurrently, this needs a lock or a per-render cache.
        self._gg_cache: collections.OrderedDict = collections.OrderedDict()
        self._gg_bytes = 0
        # Hit/miss counters, for `verify.py` to assert *which* parameters miss
        # rather than only that the output is right. A stale-cache bug renders a
        # plausible texture, so "the numbers changed" is not enough of a test.
        self.gg_hits = 0
        self.gg_misses = 0
        # Layers 1-4 are counted apart from the flat one. All five are the same
        # construction through the same function and share the one LRU and the
        # one byte budget -- they compete for the same memory, and a second
        # budget would silently double it -- but a test that wants to say "the
        # flat layer hit while a source layer missed" needs two counters to say
        # it with.
        self.gs_hits = 0
        self.gs_misses = 0

    def clear_caches(self) -> None:
        """Drop the Global Grain texture cache."""
        self._gg_cache.clear()
        self._gg_bytes = 0
