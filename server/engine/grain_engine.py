from __future__ import annotations

import collections

import torch

from .checkpoint import CheckpointCache
from .device import _checkpoint_bytes, pick_device
from .stages import (
    ColourGradeMixin, EdgeMixin, FilmTextureMixin, GlobalGrainMixin,
    HalationMixin, NormalizeMixin, RenderMixin, SharpenMixin, ToneMixin,
)
from .tiling import TilingMixin


class GrainEngine(
    NormalizeMixin,
    ColourGradeMixin,
    HalationMixin,
    ToneMixin,
    EdgeMixin,
    GlobalGrainMixin,
    FilmTextureMixin,
    SharpenMixin,
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
        # The parameter state the cache's live entries belong to, and the one
        # before it. Anything older is unreachable and is dropped on sight
        # rather than waiting for memory pressure -- see `_global_grain_field`.
        self._gg_gen = None
        self._gg_prev_gen = None
        # Counted for the same reason the hits are: dropping too much is as
        # invisible as dropping too little, and only a number can tell them
        # apart.
        self.gg_evicted = 0
        # Pipeline checkpoints: the finished frame at a section boundary, so an
        # edit below that boundary restores it instead of re-running everything
        # above. See `checkpoint.py` for which boundaries are usable and why the
        # answer is a property of the pipeline rather than a choice.
        #
        # `None` until a caller opts in by setting `_ckpt_id` -- the id of the
        # image and tier being rendered. Without one there is nothing to key on
        # that distinguishes two photographs of the same size, so the cache
        # stays off rather than guessing.
        self.ckpt = CheckpointCache(_checkpoint_bytes())
        self._ckpt_id = None
        # The in-flight render's "has a newer request arrived" hook, or None.
        #
        # On the instance rather than threaded through every signature, and that
        # borrows the same assumption `_gg_cache` above already rests on:
        # `runtime.RENDER_LOCK` serialises every render, so there is only ever
        # one in flight. `render_image` sets it and clears it in a `finally`;
        # `render_view` and `render_crop` leave it None and so never poll.
        self._cancel = None

    def clear_caches(self) -> None:
        """Drop the Global Grain texture cache."""
        self._gg_cache.clear()
        self._gg_bytes = 0
        # The generation markers go with it. Left set, the next render would
        # compare against a state whose entries no longer exist and skip the
        # sweep it should have done -- harmless today because the dict is empty,
        # and exactly the kind of stale pairing that stops being harmless the
        # first time somebody clears the cache for a reason other than a test.
        self._gg_gen = None
        self._gg_prev_gen = None
