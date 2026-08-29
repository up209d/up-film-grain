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
        # The *bounds*, not this photograph's proxy dimensions. There is no
        # longer one proxy to report: the long edge is a property of each render
        # request, so a size measured here would be a guess about a value the
        # client has not chosen yet. The client already derives proxy dimensions
        # from the edge it is asking for -- `proxyOf` in
        # `web/src/models/prescale.ts` -- because on a photograph smaller than
        # the ceiling a measured `proxy_width` is the photograph's own size and
        # says nothing about where the ceiling is. So it needs the ceiling, and
        # now the range it may move it over, and never the measurement.
        #
        # Reporting them rather than hard-coding them client-side keeps the
        # slider's range and the server's clamp from drifting apart, which is
        # the same bargain `default_preset` and the parameter schema make.
        "proxy_edge_default": up_model.PROXY_LONG_EDGE,
        "proxy_edge_min": up_model.PROXY_EDGE_MIN,
        "proxy_edge_max": up_model.PROXY_EDGE_MAX,
        "proxy_edge_step": up_model.PROXY_EDGE_STEP,
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
