"""Rendering a download, and polling it."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, Response

from .. import imageio as iio
from ..models import upload as up_model
from ..models.upload import _clamp_ss
from ..models.export_job import JOBS, run_export

router = APIRouter(prefix="/api")


@router.post("/export")
def export(body: dict = Body(...)) -> dict:
    """Render and encode a download, at the working frame's full size.

    ``supersample`` is the only quality choice -- 0.5, 1, 1.5, 2 (default) or 3
    -- and it picks how finely the frame is rendered, not how big the file is.

    **What gets rendered is the preview tier** (2026-08-09, on request): the
    proxy at the requested supersample, the identical call ``/api/preview``
    makes, enlarged afterwards to the source's own dimensions. So the file is
    the picture that was on screen when the settings were dialled in, rather
    than a fresh 1:1 render of the same numbers -- which is a *different*
    picture, finer and denser, because every length scales with the frame. See
    ``models/export_job.py`` for what that trade costs.

    ``full`` (2026-08-09, on request) opts back out of that for the one caller
    that wants it: true renders the source itself at scale 1.0 and there is
    nothing to enlarge. It is the menu's sixth entry and always asks for
    ``supersample: 1``, but nothing here forces that -- a full-tier render at
    any factor is a coherent request, it is just not one the UI offers, and
    hard-coding the pair would make the API narrower than the engine.

    It replaced a three-way ``scale`` menu (2026-08-08, on request) whose
    entries differed in resolution *and* look at the same time. A ``scale`` key
    in the body is still ignored rather than rejected, so a stale client
    degrades to a full-size export instead of a 400.

    **"Full size" means the frame that was rendered** (2026-08-29), which is the
    file's own dimensions only while Prescaling Source is off. When it is on, the
    photograph has been resampled to a working resolution and that is what every
    tier of this endpoint renders; ``prescale_output`` in the parameters decides
    whether the file is written at that resolution or resampled back to the
    photograph's own. Note this is the one export decision that lives in the
    parameters rather than in this body: it travels with a preset, on request.
    """
    up = up_model.get(body.get("id", ""))
    # The frame as well as the values -- see `models/upload.py:params_for`.
    fr, p = up_model.params_for(up, body)
    fmt = body.get("format", "jpeg")
    if fmt not in iio.FORMATS:
        raise HTTPException(400, f"Unknown format {fmt!r}.")
    ss = _clamp_ss(body.get("supersample", 2))
    full = bool(body.get("full", False))
    quality = max(60, min(100, int(body.get("quality", 95))))
    # Which dimensions the file is written at, and it is the one question the
    # supersample menu deliberately does *not* answer. `prescale_output` picks:
    # 0 writes the frame that was rendered, 1 resamples it back to the file's
    # own dimensions. With prescaling off the two are the same number, so this
    # is the historic behaviour by construction rather than by a branch.
    prescaled = fr is not up
    at_photo_size = prescaled and float(p.get("prescale_output", 0.0)) >= 0.5
    h, w = (up.h, up.w) if at_photo_size else (fr.h, fr.w)
    job_id = uuid.uuid4().hex[:12]
    stem = Path(up.name).stem or "image"
    # Every export is the same pixel dimensions now, so the filename cannot use
    # size to tell two apart -- it carries the supersample instead, and only
    # when it is not the default. Same reasoning the old `_grain_2400px` tag
    # had: two files from one photo that differ in a way a folder listing
    # cannot show are worth naming apart.
    #
    # The 1:1 render needs its own word for exactly that reason and it is the
    # sharpest case of it: `_grain_ss1` and a full-tier render at 1x are the
    # same dimensions and the same factor, and differ in the one thing anybody
    # would keep both files for.
    if full:
        tag = ("_grain_full" if abs(ss - 1.0) < 1e-6
               else f"_grain_full_ss{ss:g}".replace(".", "_"))
    else:
        tag = ("_grain" if abs(ss - 2.0) < 1e-6
               else f"_grain_ss{ss:g}".replace(".", "_"))
    # A prescaled export says so, and this one is not optional in the way the
    # supersample tag is. With `prescale_output` writing the photograph's own
    # size, a prescaled export and a plain one are the **same dimensions** and a
    # different picture -- the exact case where a folder listing cannot tell two
    # files apart, which is what every tag in this block exists for.
    if prescaled:
        tag += f"_pre{fr.target_mp:g}mp".replace(".", "_")
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "progress": 0.0,
        "filename": f"{stem}{tag}.{iio.FORMATS[fmt][1]}",
        "mime": iio.FORMATS[fmt][0], "created": time.time(),
        "width": int(w), "height": int(h),
    }
    threading.Thread(
        target=run_export,
        args=(job_id, fr, p, fmt, ss, quality, full, (int(h), int(w))),
        daemon=True,
    ).start()
    return {"job": job_id}


@router.get("/exports")
def exports_active() -> dict:
    """The exports that have not finished yet.

    Added for the desktop shell's quit guard, which needs to answer "is anything
    still rendering?" before letting the window close. Nothing else could answer
    it: `/api/export/{job_id}` needs an id the shell never sees, since the
    renderer is the one that starts the job.

    That guard matters more than it looks. The worker is a `daemon=True` thread
    (see `export` above), so process exit kills an in-flight export with no drain
    and no error. In a browser tab that is a shrug -- the tab is still there. As a
    desktop app with a close button it is silent data loss, and the user's only
    clue is a file that never appears.

    Read-only and cheap, so it is safe to poll. `list()` first rather than
    iterating `JOBS` live: a concurrent POST can insert while this runs, and
    `dict` iteration would raise rather than merely race.
    """
    live = [j for j in list(JOBS.values())
            if j.get("status") in ("queued", "rendering")]
    return {
        "active": len(live),
        "jobs": [
            {
                "id": j.get("id"),
                "status": j.get("status"),
                "progress": j.get("progress", 0.0),
                "filename": j.get("filename"),
            }
            for j in live
        ],
    }


@router.get("/export/{job_id}")
def export_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    # `blob` is a file handle, not JSON -- and `size` beside it already says
    # everything a client wants to know about it.
    return {k: v for k, v in job.items() if k not in ("bytes", "blob")}


@router.get("/export/{job_id}/download")
def export_download(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if job.get("status") != "done":
        raise HTTPException(409, f"Job is {job.get('status')}.")
    blob = job.get("blob")
    if blob is None:
        raise HTTPException(410, "That export has been cleared.")
    headers = {
        "Content-Disposition": f'attachment; filename="{job["filename"]}"'
    }
    if blob.path is not None:
        # Streamed off the disk by the OS rather than read into a `bytes` and
        # handed to Starlette -- a 24MP 16-bit PNG is ~140MB, and materialising
        # it here would put the whole file back in the memory this change took
        # it out of, at exactly the moment the user is least willing to wait.
        return FileResponse(
            blob.path, media_type=job["mime"], headers=headers,
            filename=job["filename"],
        )
    # No writable cache directory: the bytes were kept in memory as the
    # fallback, so serve them the way this always did.
    return Response(content=blob.data, media_type=job["mime"], headers=headers)
