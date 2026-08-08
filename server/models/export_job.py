"""Export jobs: the registry, the three scales, and the worker body."""

from __future__ import annotations

from .. import imageio as iio
from ..runtime import DEVICE, ENGINE, RENDER_LOCK
from ..services.render import render_tier
from .upload import Upload

JOBS: dict[str, dict] = {}

EXPORT_SCALES = ("full", "preview", "preview_full")


def run_export(job_id: str, up: Upload, p: dict, fmt: str, ss: int,
               quality: int, mode: str) -> None:
    job = JOBS[job_id]
    try:
        def progress(f: float) -> None:
            job["progress"] = round(float(f), 3)

        with RENDER_LOCK:
            job["status"] = "rendering"
            if mode in ("preview", "preview_full"):
                # Byte-for-byte the live preview's render, guaranteed by going
                # through the same function it does rather than by two call
                # sites agreeing about their arguments. Anything different here
                # and "export what I am looking at" stops being true.
                out = render_tier(up, p, ss, False, progress=progress)
                if mode == "preview_full":
                    # Blown up to the source's own pixel dimensions -- not a
                    # fresh full-resolution render. This adds no detail; it
                    # exists so "the look I am seeing" can leave as a
                    # full-size file without silently becoming a different,
                    # finer-grained picture the way a real full-res render
                    # would. See imageio.upscale and CLAUDE.md.
                    job["status"] = "upscaling"
                    out = iio.upscale(out, up.h, up.w, DEVICE)
            else:
                tile = ENGINE.tile_for(p, 1.0, up.h, up.w, ss)
                out = ENGINE.render_image(
                    up.arr, p, 1.0, tile=tile, supersample=ss, progress=progress
                )
            job["status"] = "encoding"
            data = iio.encode(out, fmt, quality)

        job["bytes"] = data
        job["size"] = len(data)
        job["status"] = "done"
        job["progress"] = 1.0
    except Exception as e:  # surfaced to the client rather than swallowed
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
