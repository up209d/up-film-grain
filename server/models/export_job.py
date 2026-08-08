"""Export jobs: the registry and the worker body.

**Every export is full size** (2026-08-08, on request). What used to be a choice
of three *scales* -- the 1:1 render, the proxy render at its own size, and the
proxy render blown up -- is a choice of **supersample** now, and the output is
always the source's own pixel dimensions.

The old menu asked the wrong question. Its three entries differed in two things
at once, resolution and look, and only one of those is what anyone was choosing
between: "As previewed" wrote a smaller file *and* a coarser grain, because
every length scales with the frame. Picking the supersample separates them --
the file is always full size, and the number says how finely it was rendered.
"""

from __future__ import annotations

from .. import imageio as iio
from ..runtime import ENGINE, RENDER_LOCK
from .upload import Upload

JOBS: dict[str, dict] = {}


def run_export(job_id: str, up: Upload, p: dict, fmt: str, ss: float,
               quality: int) -> None:
    job = JOBS[job_id]
    try:
        def progress(f: float) -> None:
            job["progress"] = round(float(f), 3)

        with RENDER_LOCK:
            job["status"] = "rendering"
            # Always `up.arr` at scale 1.0. `ss` below 1 renders the frame
            # smaller than its output and resamples back up inside
            # `render_supersampled`, so the file is full size at every setting
            # -- which is the whole point of the menu change.
            tile = ENGINE.tile_for(p, 1.0, up.h, up.w, ss)
            out = ENGINE.render_image(
                up.arr, p, 1.0, tile=tile, supersample=ss, progress=progress,
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
