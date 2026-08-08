"""Rendering a download, and polling it."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from .. import imageio as iio
from ..models import upload as up_model
from ..models.export_job import EXPORT_SCALES, JOBS, run_export

router = APIRouter(prefix="/api")


@router.post("/export")
def export(body: dict = Body(...)) -> dict:
    """Render and encode a download.

    ``scale`` picks which of three renders is written:

      * ``"full"`` (default) -- the whole source at scale 1.0, the same pixels
        the Render 1:1 button shows.
      * ``"preview"`` -- the working proxy, identical to what a slider change
        renders. Every length is multiplied by ``proxy_scale`` like everything
        else, so this is not a downscale of the full render: the grain is
        resolved at the proxy's own pixel grid, which is exactly why it looks
        like the preview and the full render does not.
      * ``"preview_full"`` -- the same proxy render as ``"preview"``, then
        blown back up to the source's full pixel dimensions with a plain
        bicubic upsample (``imageio.upscale``). Written for "export exactly
        what I am looking at, but as a full-size file": it guarantees a pixel
        match to the on-screen preview (just enlarged), which a fresh
        full-resolution render cannot, because grain is resolved on a
        different, finer grid at full scale -- see CLAUDE.md. It adds no
        detail; it is the proxy's own look, magnified, not a substitute for
        ``"full"``.
    """
    up = up_model.get(body.get("id", ""))
    p = up_model.params_for(up, body)
    fmt = body.get("format", "jpeg")
    if fmt not in iio.FORMATS:
        raise HTTPException(400, f"Unknown format {fmt!r}.")
    ss = max(1, min(3, int(body.get("supersample", 2))))
    quality = max(60, min(100, int(body.get("quality", 95))))
    mode = str(body.get("scale", "full")).lower()
    if mode not in EXPORT_SCALES:
        raise HTTPException(400, f"Unknown scale {mode!r}.")

    # "preview_full" writes the source's own dimensions -- it is the "preview"
    # render upscaled to them, not the proxy's own (smaller) size.
    h, w = (up.h, up.w) if mode != "preview" else up.proxy.shape[:2]
    job_id = uuid.uuid4().hex[:12]
    stem = Path(up.name).stem or "image"
    downscaled = up.proxy_scale < 0.999
    if mode == "preview" and downscaled:
        # Preview-scale exports carry their long edge in the name. Two files
        # from one photo that differ only in resolution are otherwise
        # indistinguishable in a folder, and the smaller one is the
        # surprising one.
        tag = f"_grain_{max(w, h)}px"
    elif mode == "preview_full" and downscaled:
        # Same pixel dimensions as "full", so the *size* cannot tell these
        # two apart in a folder -- the look is what differs, so that is what
        # the name says instead.
        tag = "_grain_previewlook"
    else:
        # Either "full", or the source was never bigger than the proxy in the
        # first place, in which case every mode renders the same pixels and
        # tagging one as different from another would be a lie.
        tag = "_grain"
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "progress": 0.0,
        "filename": f"{stem}{tag}.{iio.FORMATS[fmt][1]}",
        "mime": iio.FORMATS[fmt][0], "created": time.time(),
        "width": int(w), "height": int(h),
    }
    threading.Thread(
        target=run_export, args=(job_id, up, p, fmt, ss, quality, mode),
        daemon=True,
    ).start()
    return {"job": job_id}


@router.get("/export/{job_id}")
def export_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return {k: v for k, v in job.items() if k != "bytes"}


@router.get("/export/{job_id}/download")
def export_download(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if job.get("status") != "done":
        raise HTTPException(409, f"Job is {job.get('status')}.")
    return Response(
        content=job["bytes"],
        media_type=job["mime"],
        headers={"Content-Disposition": f'attachment; filename="{job["filename"]}"'},
    )
