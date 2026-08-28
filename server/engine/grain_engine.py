from __future__ import annotations

import os
import threading

import torch

from .checkpoint import CheckpointCache
from .device import pick_device, try_release_cache
from .diskcache import _CHECKPOINT_DISK_SHARE, _GRAIN_DISK_SHARE, DiskStore, Spill
from .stages import (
    ColourGradeMixin, EdgeMixin, FilmTextureMixin, GlobalGrainMixin,
    HalationMixin, NormalizeMixin, RenderMixin, SharpenMixin, ToneMixin,
)
from .tiling import TilingMixin


def _idle_seconds() -> float:
    """How long after the last render the flush fires. See `arm_flush`.

    Two seconds: long enough that a slider drag (a render every 1-2s, back to
    back) never trips it, short enough that putting the mouse down and looking
    at the picture gives the memory back before you have finished looking.
    """
    env = os.environ.get("FILM_GRAIN_FLUSH_IDLE")
    if env:
        try:
            return max(0.0, float(env))
        except ValueError:
            pass
    return 2.0


_FLUSH_IDLE = _idle_seconds()


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

    **Neither cache holds a tensor any more** (2026-08-29). Both are now indexes
    over files on the SSD -- see ``engine/diskcache.py`` for why that is a good
    trade here specifically and not in general. Between renders this object is a
    device handle, two dictionaries of paths, and some counters.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        # Global Grain texture cache -- see `_global_grain_field`. Disk-backed:
        # five layers at a 2400px proxy tile are 922MB and 1.6GB at a 24MP
        # export tile, which is the second largest thing the app used to hold
        # resident, and it held it for a texture that reads no image data and is
        # therefore *always* recomputable.
        #
        # The store takes its own lock, but the reasoning that made a plain dict
        # safe here still holds and is still load-bearing elsewhere:
        # `runtime.RENDER_LOCK` serialises every render.
        self._gg_cache = DiskStore("grain-textures", _GRAIN_DISK_SHARE)
        # Hit/miss counters, for `verify.py` to assert *which* parameters miss
        # rather than only that the output is right. A stale-cache bug renders a
        # plausible texture, so "the numbers changed" is not enough of a test.
        #
        # Kept on the engine rather than read off the store because the store
        # counts one number and the tests need two: layers 1-4 are counted apart
        # from the flat one, so that a check can say "the flat layer hit while a
        # source layer missed". All five still share the one store and the one
        # byte budget -- they compete for the same disk, and a second budget
        # would silently double it.
        self.gg_hits = 0
        self.gg_misses = 0
        self.gs_hits = 0
        self.gs_misses = 0
        # The parameter state the cache's live entries belong to, and the one
        # before it. Anything older is unreachable and is dropped on sight
        # rather than waiting for pressure -- see `_global_grain_field`.
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
        self.ckpt = CheckpointCache(_CHECKPOINT_DISK_SHARE)
        self.ckpt.device = self.device
        self._ckpt_id = None
        # The in-flight render's "has a newer request arrived" hook, or None.
        #
        # On the instance rather than threaded through every signature, and that
        # borrows the same assumption the caches above already rest on:
        # `runtime.RENDER_LOCK` serialises every render, so there is only ever
        # one in flight. `render_image` sets it and clears it in a `finally`;
        # `render_view` and `render_crop` leave it None and so never poll.
        self._cancel = None
        # The idle flush fires on a timer thread -- see `arm_flush`. What
        # protects it from firing into a live render is `device.device_work` /
        # `device.try_release_cache`, not anything held here; see those for the
        # measured reason it has to be the *device* that is guarded rather than
        # the callers.
        self._flush_timer: threading.Timer | None = None
        self._flush_lk = threading.Lock()

    def clear_caches(self) -> None:
        """Drop the Global Grain texture cache, files included."""
        self._gg_cache.clear()
        # The generation markers go with it. Left set, the next render would
        # compare against a state whose entries no longer exist and skip the
        # sweep it should have done -- harmless today because the store is
        # empty, and exactly the kind of stale pairing that stops being harmless
        # the first time somebody clears the cache for a reason other than a
        # test.
        self._gg_gen = None
        self._gg_prev_gen = None

    # -- giving memory back ------------------------------------------------ #

    def arm_flush(self) -> None:
        """Schedule a flush for when rendering has actually stopped.

        Called from `render_image`'s `finally`, so it runs on the cancelled and
        failed paths too -- and it *schedules* rather than flushes, which is the
        one decision in this file worth arguing for.

        The naive version releases everything the moment a render returns. It is
        wrong for the case that matters, and measurably so: a slider drag is a
        stream of single-tile proxy renders, and `release_cache` on that path
        costs `Stock` 1.13s -> 1.45s and `VintageDarkGrainy` 1.64s -> 1.94s (the
        figures `_render_image` records for the same call between tiles) to hand
        back blocks the very next render immediately asks for again. Nothing is
        saved: during a drag the free list is not idle memory, it is the next
        frame's working set arriving early.

        What is actually worth reclaiming is the free list held while the user
        is *not* rendering -- reading the picture, choosing a preset, or away
        from the machine -- which on a 24MP export is many gigabytes sitting
        against a render that already finished. So the timer resets on every
        render and fires once, `_FLUSH_IDLE` seconds after the last one.

        `FILM_GRAIN_FLUSH_IDLE` tunes it, and **0 flushes immediately after
        every render** -- the literal reading of "free it when the pipeline is
        done", available for anyone who would rather pay the 0.3s than hold the
        memory for a couple of seconds.
        """
        if _FLUSH_IDLE <= 0.0:
            self.flush_ram()
            return
        with self._flush_lk:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
            t = threading.Timer(_FLUSH_IDLE, self.flush_ram)
            t.daemon = True
            self._flush_timer = t
            t.start()

    def flush_ram(self) -> None:
        """Give back every byte the finished render was still holding.

        Two things are released, held by two different owners, which is why this
        is one call rather than two scattered ones:

        * **Mapped source frames.** `Spill.unmap()` drops the page tables for
          every uploaded photograph, which is what lets the kernel reclaim the
          clean pages the render faulted in. A 24MP source is 288MB and the full
          tier touches all of it; left mapped, that stays resident until pressure
          forces the issue, which on a machine with plenty of RAM is never.
          Re-mapping on the next render costs a page fault per page, served from
          the SSD.
        * **The allocator's free list.** On unified memory this is system RAM, so
          holding it starves everything else on the machine -- see
          `device.release_cache`, which measures the same call between tiles at
          1.5x faster for 2.1x less memory.

        **The caches themselves have nothing to release**, by construction; that
        is the whole of `diskcache.py`. They are named here only so the next
        person looking for "where does the cache get freed" finds the answer
        rather than assuming it was forgotten.

        Called from the idle timer, from `/api/upload` and from
        `/api/cache/clear`, none of which is a render thread -- hence
        `try_release_cache`, which does nothing if the device is busy. Skipping
        is the right answer in every one of those cases: a render is in flight,
        so the free list is not idle memory, and that render will re-arm the
        timer when it finishes.
        """
        Spill.unmap_all()
        try_release_cache(self.device)
