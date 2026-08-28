from __future__ import annotations

import contextlib
import os
import sys
import threading

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

# The renderer's share of the pool.
#
# It used to be a three-way split -- 0.7 to `tile_for`, 0.15 each to the Global
# Grain texture cache and the pipeline checkpoints -- because those two caches
# held *device* tensors that the renderer then could not have. **Both moved to
# the SSD on 2026-08-29** (`engine/diskcache.py`), so nothing competes for this
# pool any longer and the two shares are gone rather than set to zero.
#
# **0.7 is kept anyway, and is the number that must not drop.** It was measured
# rather than left over: at 0.6 a 2400px proxy tips into two tiles and
# `SuperPortra` goes 1.75s -> 2.72s on the GPU and 7.9s -> 9.5s on the CPU.
# Raising it toward 1.0 is now free of the old objection and is deliberately not
# done -- larger tiles mean a larger *peak* working set, and the change that
# removed these caches was made to lower the app's memory use, not to spend the
# saving somewhere else. The remaining 0.3 is headroom for the client, the
# encoder and whatever else shares the machine, which is what it was always
# described as.
_TILE_BUDGET_SHARE = 0.7

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


# Backends the engine knows how to drive. `FILM_GRAIN_DEVICE` is checked against
# this rather than passed to `torch.device` directly, so a typo produces a
# warning and the normal auto-detection instead of a crash at import time --
# this runs before uvicorn binds, so raising here means the app never starts.
_DEVICES = ("cpu", "mps", "cuda")

# Warnings are emitted once per distinct message. `_resolve_device` is called
# from `_render_budget_bytes`, which the Global Grain cache calls on *every*
# insert, so an un-deduplicated warning would print thousands of times per
# render.
_warned: set[str] = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[device] {msg}", file=sys.stderr)


def _system_ram_bytes() -> int:
    """Total physical RAM, or a conservative guess.

    `os.sysconf` covers macOS and Linux and **does not exist on Windows**, where
    the old code caught the `AttributeError` and fell through to a hardcoded 4GB.
    That did not crash, which is exactly why it was worth finding: it silently
    told a 64GB machine it had 4GB, and since this figure drives
    `_render_budget_bytes` -> `_tile_budget_bytes` -> `tile_for`, every tile
    clamped to `_TILE_MIN` and both caches landed in the starved regime that
    the Global Grain texture cache measured at 7.36s against 1.78s when it was
    starved the same way. Wrong by 16x in the direction that costs performance,
    reported by nothing.

    `GlobalMemoryStatusEx` is the Windows equivalent and needs no dependency --
    `ctypes` is stdlib. **Unverified until the windows target is actually built**;
    it is written now while the reasoning is fresh, and it is dead code on macOS.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemoryStatusEx()
            st.dwLength = ctypes.sizeof(_MemoryStatusEx)
            # Returns 0 on failure; ullTotalPhys would then be garbage.
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                if st.ullTotalPhys > 0:
                    return int(st.ullTotalPhys)
        except Exception:
            pass

    _warn("cannot read system RAM; assuming 4GB. Set FILM_GRAIN_TILE_BUDGET_GB "
          "to override.")
    return 4 << 30


def _resolve_device() -> torch.device:
    """The device the engine will actually render on.

    The single source of truth for that question, and it has two callers that
    must never disagree: `pick_device`, which the engine renders on, and
    `_render_budget_bytes`, which sizes tiles and caches for it. They used to
    answer independently -- and in a different order, this one testing MPS first
    while `pick_device` tested CUDA first. Harmless only because no machine has
    both; not harmless once `FILM_GRAIN_DEVICE` exists, because forcing CPU on a
    Mac would still have sized the memory pool from
    `torch.mps.recommended_max_memory()`, i.e. from a GPU nothing was using.

    `FILM_GRAIN_DEVICE` (`cpu`/`mps`/`cuda`) forces the choice, in the same
    spirit as the other `FILM_GRAIN_*` overrides: it is how the CPU path gets
    exercised on a machine that has a GPU, and how a user works around a driver
    that misbehaves without waiting for a new build. Asking for a backend that is
    not present warns and falls back to auto-detection rather than failing --
    a bundled app that refuses to start is worse than a slow one.

    Read per call rather than cached, matching `FILM_GRAIN_TILE_BUDGET_GB`: the
    probes are the same ones this function already made on every call, so this
    costs nothing new, and a test that sets the variable gets the behaviour it
    asked for.
    """
    want = (os.environ.get("FILM_GRAIN_DEVICE") or "").strip().lower()
    if want:
        if want not in _DEVICES:
            _warn(f"FILM_GRAIN_DEVICE={want!r} is not one of "
                  f"{'/'.join(_DEVICES)}; detecting automatically.")
        elif want == "cpu":
            # Always available, and the only forced value that cannot fail.
            return torch.device("cpu")
        elif want == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif want == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            _warn(f"FILM_GRAIN_DEVICE={want} was requested but that backend is "
                  "not available; detecting automatically.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _render_budget_bytes() -> int:
    """Total device-memory budget for a render, in bytes.

    The **pool**, not the tile's share of it -- `_tile_budget_bytes` takes
    `_TILE_BUDGET_SHARE` of it and the rest is headroom. It used to be divided
    three ways, with the texture cache and the checkpoints holding device
    tensors out of it; both live on the SSD now (`engine/diskcache.py`), so this
    pool has one claimant again.

    `FILM_GRAIN_TILE_BUDGET_GB` overrides it outright, in the same spirit as
    `FILM_GRAIN_DEFAULT_PRESET` -- useful both for forcing large tiles on a big
    machine and for reproducing a small machine's tiling on a large one, which is
    what makes the tile-independence checks in `verify.py` testable here. It
    keeps its name and overrides the *pool* rather than the tile's share of it,
    so a machine reproduced by it reproduces the headroom too.
    """
    env = os.environ.get("FILM_GRAIN_TILE_BUDGET_GB")
    if env:
        try:
            return int(float(env) * (1 << 30))
        except ValueError:
            pass
    total = 0
    # Keyed off the device the engine will actually use -- see `_resolve_device`
    # for why asking the hardware directly is the wrong question here.
    dev = _resolve_device().type
    if dev == "cuda":
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            total = 0
    elif dev == "mps":
        try:
            total = int(torch.mps.recommended_max_memory())
        except Exception:
            total = 0
    if total <= 0:
        # CPU, or a backend that will not say. Derive from system RAM, which is
        # the real constraint there too.
        total = _system_ram_bytes()
    return max(1 << 30, int(total * _RENDER_BUDGET_FRACTION))


def _tile_budget_bytes() -> int:
    """The renderer's share of the pool -- what `tile_for` sizes a tile against."""
    return int(_render_budget_bytes() * _TILE_BUDGET_SHARE)


# How many threads are currently submitting work to the device.
#
# **This exists so that nothing ever calls `release_cache` while another thread
# has device work in flight**, which on MPS is not a race that produces a wrong
# answer -- it aborts the process:
#
#     -[IOGPUMetalCommandBuffer validate]: failed assertion
#     `commit an already committed command buffer'
#
# Measured, not theorised: the idle flush was first written to fire on its timer
# thread under the render lock, and the check suite died on exactly that
# assertion, because a check calling `render()` directly holds no such lock and
# an upload resampling on a request thread holds nothing at all. Guarding the
# *callers* meant enumerating every path that touches the GPU and being right
# about all of them; guarding the *device* means one counter that every path
# increments and a releaser that waits for zero.
#
# A counter rather than a mutex because the thing being excluded is asymmetric:
# any number of threads may use the device at once (that was already true and
# already fine), and only the release must be alone. A plain lock held for the
# length of a render would have made an upload during a 24MP export wait for the
# export, which is a regression nobody asked for.
_busy = 0
_busy_lk = threading.Lock()


@contextlib.contextmanager
def device_work():
    """Mark this thread as using the device for the duration of the block.

    Wrap any entry point that submits work to the GPU. Nesting is free -- it is
    a counter, not a lock -- so an outer `render_image` and the `render` inside
    it may both take it, which is what lets the direct-`render()` calls in
    `tests/checks/` be covered without the engine's own call graph caring.
    """
    global _busy
    with _busy_lk:
        _busy += 1
    try:
        yield
    finally:
        with _busy_lk:
            _busy -= 1


def try_release_cache(dev: torch.device) -> bool:
    """`release_cache`, but only if the device is idle. True if it ran.

    For callers that are **not** on a render thread -- today, the engine's idle
    flush. Everything on a render thread should call `release_cache` directly:
    it is already inside `device_work`, so this would always answer False, and
    the between-tiles release that keeps `_WORKING_BYTES_PER_PX` honest would
    silently stop happening.

    The check and the release are one critical section on purpose. Testing the
    counter and then releasing outside the lock would leave exactly the window
    this function exists to close.
    """
    with _busy_lk:
        if _busy:
            return False
        release_cache(dev)
        return True


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
    """The device the engine renders on.

    A thin wrapper over `_resolve_device` on purpose: sharing one implementation
    is what guarantees the renderer and the memory budget cannot pick different
    backends. Kept as its own name because it is the public one -- `runtime.py`,
    `GrainEngine` and the test harness all call it.
    """
    return _resolve_device()


def device_name(dev: torch.device) -> str:
    return {"cuda": "CUDA", "mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(dev.type, dev.type)
