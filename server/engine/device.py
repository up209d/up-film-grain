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
_WORKING_BYTES_PER_PX = 512

# Fraction of the backend's recommended working set to actually use. The rest is
# headroom for the client, the encoder and whatever else shares the GPU -- the
# render is not the only thing on the machine, and on Apple silicon this is
# system RAM, so overcommitting means swapping rather than a clean failure.
_RENDER_BUDGET_FRACTION = 0.5

# Tile floor and ceiling. The floor is not a memory figure: below it `pad_for`
# overlap dominates the useful area so completely that the extra work costs more
# than the memory it saves, and every supported backend can hold a tile this
# size. The ceiling keeps a single enormous tile from defeating the point of
# tiling on a machine that reports a very large budget.
_TILE_MIN = 768
_TILE_MAX = 8192


def _render_budget_bytes() -> int:
    """Working-set budget for one tile, in bytes.

    `FILM_GRAIN_TILE_BUDGET_GB` overrides it outright, in the same spirit as
    `FILM_GRAIN_DEFAULT_PRESET` -- useful both for forcing large tiles on a big
    machine and for reproducing a small machine's tiling on a large one, which is
    what makes the tile-independence checks in `verify.py` testable here.
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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    return {"cuda": "CUDA", "mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(dev.type, dev.type)
