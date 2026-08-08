from __future__ import annotations

import os

import torch


# Peak render memory per *working* pixel -- i.e. per pixel of the padded tile
# after supersampling. Measured, not guessed: single-tile `Stock` renders at 512,
# 768, 1024 and 1280 square with supersample 2 gave marginal costs of 763, 530
# and 424 bytes per working pixel as the frame grew (the allocator reserves in
# ~1GB steps, so the small end reads high). 512 is above the worst marginal
# figure, which is the safe direction to be wrong in: over-estimating picks a
# smaller tile and renders slower, under-estimating runs the machine out of
# memory. `defaults` measures roughly half this, so the constant is sized for
# the heaviest preset rather than the typical one.
#
# **This figure only holds because `release_cache` runs between tiles**, and it
# is worth knowing how nearly it did not. Measured 2026-08-08 on a 24MP `Stock`
# export at tile 2288: driver-allocated peaked at **27.2GB against 9.9GB of live
# tensors** -- 1075 bytes per working pixel, more than double this constant,
# because the caching allocator was sitting on ~17GB of freed blocks. `tile_for`
# had sized that tile believing it would use 14GB. Freeing the blocks per tile
# brings the peak to 13.0GB, i.e. 514 bytes per working pixel, which is what this
# number describes. Remove the `release_cache` call and this constant is wrong by
# 2.1x in the direction that runs a machine out of memory.
#
# Raised 512 -> 640 in the same change. With the blocks freed, a 24MP `Stock`
# export measures **601** bytes per working pixel, so 512 was under the real
# figure by 17% and `tile_for` was quietly overrunning its share. 640 puts it
# back above the measurement, which is the direction that costs a smaller tile
# rather than a dead machine. Note the figure is not constant in frame size --
# the same render measures 563-630 at a 2400px proxy and 601 at 24MP -- so it
# must be set from the *export*, which is the case that runs out of memory.
_WORKING_BYTES_PER_PX = 640

# Fraction of the backend's recommended working set to actually use. The rest is
# headroom for the client, the encoder and whatever else shares the GPU -- the
# render is not the only thing on the machine, and on Apple silicon this is
# system RAM, so overcommitting means swapping rather than a clean failure.
_RENDER_BUDGET_FRACTION = 0.5

# How the budget above is split between the two things that spend it. They are
# genuinely in competition -- the Global Grain texture cache holds device
# tensors that the renderer then cannot have -- and the old arrangement left
# that competition implicit: `tile_for` took the whole budget and the cache took
# a flat 0.5GB decided by a comment that described a tile size and a layer count
# neither of which is true any more. Splitting it here is what makes raising one
# visibly lower the other.
#
# **The renderer's 0.7 is the number that must not drop**, and the two caches
# were sized around it rather than out of it. Taking the checkpoint share off
# the tile budget instead was tried and measured: 0.6 puts a 2400px proxy over
# the line into two tiles, and `SuperPortra` went 1.75s -> 2.72s on the GPU and
# 7.9s -> 9.5s on the CPU. A cache that makes the uncached path slower is a bad
# trade however good its hit rate.
#
# 0.15 each to the two caches, which is what they actually need rather than
# what was left over. The texture cache wants 922MB for five layers at one tile
# of a 2400px proxy and 1.6GB at a 24MP export tile; the checkpoints want 184MB
# per boundary per tile. On this machine's 14GB pool that is 2.1GB apiece and
# everything fits; on an 8GB machine it is 300MB and each holds part of its set,
# which is a partial hit rate rather than the 0% a too-small budget produces.
# Generation-aware eviction keeps both *live* sets to two parameter states
# regardless, so the caps bound the worst case rather than the usual one.
_TILE_BUDGET_SHARE = 0.7
_GRAIN_CACHE_SHARE = 0.15
_CHECKPOINT_SHARE = 0.15

# How large one tile's working set has to be, as a share of the tile budget,
# before `release_cache` is worth calling between tiles.
#
# The call is not free -- it forces a synchronise, and the next tile then
# re-acquires blocks it could have reused. On a tile whose working set is a small
# fraction of the budget there is no peak to bound, so the stall buys nothing:
# `verify.py`, which renders many small frames at tiles from 256 up, went from
# 35.7s to 96.4s with the release firing unconditionally. Gating it at half the
# budget keeps the suite at its old speed while still firing on every tile large
# enough to matter -- a 24MP export tile is 100% of the budget, an 8GB machine's
# proxy tile is 171% of its own.
_RELEASE_MIN_SHARE = 0.5

# Tile floor and ceiling. The floor is not a memory figure: below it `pad_for`
# overlap dominates the useful area so completely that the extra work costs more
# than the memory it saves, and every supported backend can hold a tile this
# size. The ceiling keeps a single enormous tile from defeating the point of
# tiling on a machine that reports a very large budget.
_TILE_MIN = 768
_TILE_MAX = 8192


def _render_budget_bytes() -> int:
    """Total device-memory budget for a render, in bytes.

    The **pool**, not the tile's share of it -- `_tile_budget_bytes` and
    `_grain_cache_bytes` divide this between them. It used to be the tile budget
    outright while the texture cache took a flat 0.5GB on top, so the two could
    not both be honoured and the real ceiling was whatever they happened to sum
    to.

    `FILM_GRAIN_TILE_BUDGET_GB` overrides it outright, in the same spirit as
    `FILM_GRAIN_DEFAULT_PRESET` -- useful both for forcing large tiles on a big
    machine and for reproducing a small machine's tiling on a large one, which is
    what makes the tile-independence checks in `verify.py` testable here. It
    keeps its name and overrides the *pool*: a machine reproduced by it should
    reproduce both halves of the split, not just one.
    """
    env = os.environ.get("FILM_GRAIN_TILE_BUDGET_GB")
    if env:
        try:
            return int(float(env) * (1 << 30))
        except ValueError:
            pass
    total = 0
    if torch.backends.mps.is_available():
        try:
            total = int(torch.mps.recommended_max_memory())
        except Exception:
            total = 0
    elif torch.cuda.is_available():
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            total = 0
    if total <= 0:
        # CPU, or a backend that will not say. Derive from system RAM, which is
        # the real constraint there too.
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError, AttributeError):
            total = 4 << 30
    return max(1 << 30, int(total * _RENDER_BUDGET_FRACTION))


def _tile_budget_bytes() -> int:
    """The renderer's share of the pool -- what `tile_for` sizes a tile against."""
    return int(_render_budget_bytes() * _TILE_BUDGET_SHARE)


def _checkpoint_bytes() -> int:
    """The pipeline checkpoint cache's share of the pool.

    `FILM_GRAIN_CHECKPOINT_GB` overrides it; 0 switches checkpointing off
    entirely, which is the honest way to measure what it is worth.
    """
    env = os.environ.get("FILM_GRAIN_CHECKPOINT_GB")
    if env:
        try:
            return int(float(env) * (1 << 30))
        except ValueError:
            pass
    return int(_render_budget_bytes() * _CHECKPOINT_SHARE)


def _grain_cache_bytes() -> int:
    """The Global Grain texture cache's share of the pool.

    Derived rather than constant, and that is the fix for a measured 0% hit rate.
    The old flat 0.5GB was sized by a comment reading "113MB per tile at tile
    1536 / supersample 2" -- written when the tile was a hard-coded constant and
    the section had **one** layer. It now has five and `tile_for` computes the
    tile, so `SuperPortra` at a 2400px proxy wants 5 x 184MB = 922MB, the LRU
    held two entries, and every render missed all five. Measured 2026-08-08: the
    same render was 7.36s with the flat budget and 1.78s with a budget that fits,
    on identical parameters, 0 hits vs 5.

    `FILM_GRAIN_GRAIN_CACHE_GB` still overrides it outright, which is how the
    starved case can be reproduced deliberately.
    """
    env = os.environ.get("FILM_GRAIN_GRAIN_CACHE_GB")
    if env:
        try:
            return int(float(env) * (1 << 30))
        except ValueError:
            pass
    return int(_render_budget_bytes() * _GRAIN_CACHE_SHARE)


def release_cache(dev: torch.device) -> None:
    """Return the allocator's cached blocks to the system.

    Called between tiles. **This buys time as well as memory, which is not the
    usual shape of such a call**: on unified memory the allocator's free list is
    system RAM, so holding it starves everything else on the machine and the
    render slows down. Measured on a 24MP `Stock` export, tile 2288, idle:
    30.6s / 27.2GB driver-allocated without it, **20.1s / 13.0GB with it** --
    1.5x faster for 2.1x less memory, against an unchanged 10.0GB of live
    tensors.

    Per tile rather than deeper in: it needs a synchronise to be meaningful, and
    a tile boundary is already one. `_WORKING_BYTES_PER_PX` describes the world
    where this runs -- see its comment.
    """
    if dev.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    return {"cuda": "CUDA", "mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(dev.type, dev.type)
