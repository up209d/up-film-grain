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
from ..models.upload import _clamp_ss
from ..models.export_job import JOBS, run_export

router = APIRouter(prefix="/api")


@router.post("/export")
def export(body: dict = Body(...)) -> dict:
    """Render and encode a download. **Always at the source's full size.**

    ``supersample`` is the only quality choice now -- 0.5, 1, 1.5, 2 (default)
    or 3 -- and it picks how finely the frame is rendered, not how big the file
    is. Below 1 the render happens smaller than the output and is resampled back
    up; above it, above the output grid and integrated down.

    It replaced a three-way ``scale`` menu (2026-08-08, on request) whose
    entries differed in resolution *and* look at the same time: "As previewed"
    wrote the proxy render, which is a smaller file **and** a coarser grain,
    because every length scales with the frame. Two things on one control, and
    only one of them was ever the question being asked. A ``scale`` key in the
    body is now ignored rather than rejected, so a stale client degrades to a
    full-size export instead of a 400.
    """
    up = up_model.get(body.get("id", ""))
    p = up_model.params_for(up, body)
    fmt = body.get("format", "jpeg")
    if fmt not in iio.FORMATS:
        raise HTTPException(400, f"Unknown format {fmt!r}.")
    ss = _clamp_ss(body.get("supersample", 2))
    quality = max(60, min(100, int(body.get("quality", 95))))
    h, w = up.h, up.w
    job_id = uuid.uuid4().hex[:12]
    stem = Path(up.name).stem or "image"
    # Every export is the same pixel dimensions now, so the filename cannot use
    # size to tell two apart -- it carries the supersample instead, and only
    # when it is not the default. Same reasoning the old `_grain_2400px` tag
    # had: two files from one photo that differ in a way a folder listing
    # cannot show are worth naming apart.
    tag = "_grain" if abs(ss - 2.0) < 1e-6 else f"_grain_ss{ss:g}".replace(".", "_")
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "progress": 0.0,
        "filename": f"{stem}{tag}.{iio.FORMATS[fmt][1]}",
        "mime": iio.FORMATS[fmt][0], "created": time.time(),
        "width": int(w), "height": int(h),
    }
    threading.Thread(
        target=run_export, args=(job_id, up, p, fmt, ss, quality),
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
