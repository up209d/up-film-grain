"""FastAPI service for the film grain engine.

Endpoints are deliberately synchronous (``def``, not ``async def``) where they
do tensor work, so Starlette runs them in its threadpool and one slow render
cannot block the event loop.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import imageio as iio
from . import params as P
from .engine import GrainEngine, device_name, pick_device

# Production is the default. Dev mode is the special case -- it is the one that
# needs CORS holes and an interactive schema browser, so it has to be asked for
# rather than assumed, or a distribution ships with both.
IS_DEV = os.environ.get("APP_ENV", "production").lower() in ("dev", "development")

app = FastAPI(
    title="Film Grain Engine",
    # No interactive docs on a production build: this binds to loopback and has
    # no auth, so the smaller the surface the better.
    docs_url="/docs" if IS_DEV else None,
    redoc_url=None,
    openapi_url="/openapi.json" if IS_DEV else None,
)

if IS_DEV:
    # Only needed when Vite serves the client from its own origin on :5173.
    # In production FastAPI serves the client itself, so every request is
    # same-origin and this would be a hole for nothing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

DEVICE = pick_device()
ENGINE = GrainEngine(DEVICE)

# Renders are serialised: concurrent tensor work on one GPU just thrashes, and
# the UI only ever needs the newest preview anyway.
_RENDER_LOCK = threading.Lock()


# Long edge of the working proxy used for live preview renders. Measured on a
# 24MP source: a full-resolution 2x pass is 7.8s, this is 1.3s. That gap is the
# whole reason the proxy exists -- it is what makes dragging a slider feel like
# editing rather than batch processing.
PROXY_LONG_EDGE = 2400


class Upload:
    __slots__ = ("id", "name", "arr", "h", "w", "proxy", "proxy_scale",
                 "src_png", "touched")

    def __init__(self, uid: str, name: str, arr: np.ndarray) -> None:
        self.id = uid
        self.name = name
        self.arr = arr
        self.h, self.w = arr.shape[:2]
        self.proxy_scale = min(1.0, PROXY_LONG_EDGE / float(max(self.h, self.w)))
        self.proxy = iio.downscale(arr, self.proxy_scale, DEVICE)
        # The untouched image never changes, so it is encoded once and served
        # from here rather than re-encoded on every parameter change.
        self.src_png: bytes | None = None
        self.touched = time.time()


UPLOADS: dict[str, Upload] = {}
JOBS: dict[str, dict] = {}
_MAX_UPLOADS = 12


def _reap() -> None:
    """Drop the oldest uploads; full-resolution arrays are large."""
    if len(UPLOADS) <= _MAX_UPLOADS:
        return
    for uid in sorted(UPLOADS, key=lambda k: UPLOADS[k].touched)[:-_MAX_UPLOADS]:
        UPLOADS.pop(uid, None)


def _get(uid: str) -> Upload:
    up = UPLOADS.get(uid)
    if up is None:
        raise HTTPException(404, "Unknown image id -- it may have been evicted.")
    up.touched = time.time()
    return up


def _params_for(up: Upload, body: dict) -> dict[str, float]:
    """Sanitised parameters, rescaled if they were authored at another size.

    ``reference_mp`` is the megapixel count of the image a preset was dialled
    in on. Resolved per request rather than baked into the values when a preset
    loads, so switching to a different photo re-scales on its own instead of
    leaving the last photo's numbers behind.
    """
    p = P.sanitize(body.get("params"))
    ref = body.get("reference_mp")
    if ref:
        k = P.scale_factor(float(ref), up.w * up.h / 1e6)
        p = P.rescale(p, k)
    return p


# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "device": device_name(DEVICE)}


@app.get("/api/params")
def get_params() -> dict:
    return P.schema()


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        arr = iio.load_image(data)
    except iio.UploadTooLarge as e:
        raise HTTPException(413, str(e))
    except iio.UnsupportedFormat as e:
        raise HTTPException(415, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    uid = uuid.uuid4().hex[:12]
    UPLOADS[uid] = Upload(uid, file.filename or "image", arr)
    _reap()
    up = UPLOADS[uid]
    return {
        "id": uid,
        "name": up.name,
        "width": up.w,
        "height": up.h,
        "megapixels": round(up.w * up.h / 1e6, 1),
        "proxy_width": int(up.proxy.shape[1]),
        "proxy_height": int(up.proxy.shape[0]),
    }


@app.post("/api/preview")
def preview(body: dict = Body(...)) -> Response:
    """Render the whole frame -- at proxy scale by default, full res on demand.

    There is no view geometry in this request at all: no mode, no zoom, no
    viewport, no crop. The client scales what it gets for display, so one
    render answers every zoom level and panning or zooming never comes back
    here.

    ``full`` picks which of the two renders you get, and that is the only
    difference between them:

      * ``false`` (default) -- the working proxy. Fast enough to sit behind a
        slider. Magnified past its own resolution it is soft, so it cannot be
        used to judge grain.
      * ``true`` -- the whole source at scale 1.0. The preview *is* the export
        at this point: same pixels, same coordinates, differing only in bit
        depth.

    Both go through the identical pipeline at their working scale, which is
    what scale invariance buys -- the proxy predicts the full render's
    structure, it just cannot resolve its finest detail.
    """
    up = _get(body.get("id", ""))
    p = _params_for(up, body)
    ss = max(1, min(3, int(body.get("supersample", 2))))
    full = bool(body.get("full", False))

    t0 = time.time()
    with _RENDER_LOCK:
        # Larger tiles than the export's: the per-tile overlap is fixed
        # padding, so wider tiles amortise it. Measured on a 24MP source,
        # 1536 beats 1024 by ~5% at the default halation radius and ~12% at
        # the widest. Past 2048 it turns around again as the tensors stop
        # fitting comfortably.
        if full:
            out = ENGINE.render_image(up.arr, p, 1.0, tile=1536, supersample=ss)
        else:
            out = ENGINE.render_image(
                up.proxy, p, up.proxy_scale, tile=1536, supersample=ss
            )
    ms = int((time.time() - t0) * 1000)

    return Response(
        content=iio.encode_preview(out),
        media_type="image/png",
        headers={
            "X-Render-Ms": str(ms),
            "X-Render-Full": "1" if full else "0",
            "X-Render-Device": device_name(DEVICE),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/source")
def source(body: dict = Body(...)) -> Response:
    """The untouched image at full resolution, for before/after compare.

    Same pixel grid as ``/api/preview``, so the two line up under the wipe and
    in the side-by-side view at any zoom. Encoded once per upload and cached --
    the source does not change when a slider does.
    """
    up = _get(body.get("id", ""))
    if up.src_png is None:
        up.src_png = iio.encode_preview(up.arr)
    return Response(
        content=up.src_png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ------------------------------------------------------------------ export --

def _run_export(job_id: str, up: Upload, p: dict, fmt: str, ss: int, quality: int) -> None:
    job = JOBS[job_id]
    try:
        def progress(f: float) -> None:
            job["progress"] = round(float(f), 3)

        with _RENDER_LOCK:
            job["status"] = "rendering"
            out = ENGINE.render_image(
                up.arr, p, 1.0, tile=1024, supersample=ss, progress=progress
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


@app.post("/api/export")
def export(body: dict = Body(...)) -> dict:
    up = _get(body.get("id", ""))
    p = _params_for(up, body)
    fmt = body.get("format", "jpeg")
    if fmt not in iio.FORMATS:
        raise HTTPException(400, f"Unknown format {fmt!r}.")
    ss = max(1, min(3, int(body.get("supersample", 2))))
    quality = max(60, min(100, int(body.get("quality", 95))))

    job_id = uuid.uuid4().hex[:12]
    stem = Path(up.name).stem or "image"
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "progress": 0.0,
        "filename": f"{stem}_grain.{iio.FORMATS[fmt][1]}",
        "mime": iio.FORMATS[fmt][0], "created": time.time(),
    }
    threading.Thread(
        target=_run_export, args=(job_id, up, p, fmt, ss, quality), daemon=True
    ).start()
    return {"job": job_id}


@app.get("/api/export/{job_id}")
def export_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return {k: v for k, v in job.items() if k != "bytes"}


@app.get("/api/export/{job_id}/download")
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


# ------------------------------------------------------------------- static --

_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

elif IS_DEV:
    # In dev the client is normally served by Vite on :5173, so a missing
    # build here is expected and only matters if you opened the API port.
    @app.get("/")
    def index_missing() -> JSONResponse:
        return JSONResponse(
            {"error": "Web client not built. Run `npm run build` in web/."},
            status_code=503,
        )

else:
    # A production process with no client is broken, not degraded -- there is
    # nothing else serving the UI. Fail at import rather than boot happily and
    # hand every visitor a 503.
    raise RuntimeError(
        f"No web client at {_DIST}. Build it first (./build.sh), or set "
        "APP_ENV=development to run the API on its own."
    )
