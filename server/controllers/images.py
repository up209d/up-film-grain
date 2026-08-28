"""Taking an image in, and handing the untouched one back out."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import Response

from .. import imageio as iio
from ..models import upload as up_model

router = APIRouter(prefix="/api")


@router.post("/upload")
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
    up_model.UPLOADS[uid] = up_model.Upload(uid, file.filename or "image", arr)
    # A new photograph invalidates every cache in the app -- see
    # `upload.reset`. Called before `reap`, which is now the fallback for the
    # case this misses rather than the thing that bounds the registry.
    up_model.reset(uid)
    up_model.reap()
    up = up_model.UPLOADS[uid]
    return {
        "id": uid,
        "name": up.name,
        "width": up.w,
        "height": up.h,
        "megapixels": round(up.w * up.h / 1e6, 1),
        "proxy_width": int(up.proxy.shape[1]),
        "proxy_height": int(up.proxy.shape[0]),
        # The constant itself, not just this photograph's proxy. The client
        # mirrors the prescale arithmetic to label the export and to size the
        # stage, and a prescaled frame has a proxy of its own -- which it cannot
        # work out from `proxy_width` alone, because on a photograph smaller
        # than the long edge that number is the photograph's own size and says
        # nothing about the ceiling. See `web/src/models/prescale.ts`.
        "proxy_long_edge": up_model.PROXY_LONG_EDGE,
    }


@router.post("/source")
def source(body: dict = Body(...)) -> Response:
    """The untouched image at full resolution, for before/after compare.

    Same pixel grid as ``/api/preview``, so the two line up under the wipe and
    in the side-by-side view at any zoom. Encoded once per upload and cached --
    the source does not change when a slider does.
    """
    up = up_model.get(body.get("id", ""))
    if up.src_enc is None:
        up.src_enc = iio.encode_preview(up.arr)
    return Response(
        content=up.src_enc,
        media_type=iio.PREVIEW_MEDIA_TYPE,
        headers={"Cache-Control": "no-store"},
    )
