"""The live preview render."""

from __future__ import annotations

import time

from fastapi import APIRouter, Body
from fastapi.responses import Response

from .. import imageio as iio
from ..engine import RenderCancelled, device_name
from ..models import upload as up_model
from ..runtime import DEVICE, RENDER_LOCK, is_superseded, next_preview_gen
from ..services.render import render_tier

router = APIRouter(prefix="/api")


@router.post("/preview")
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
    up = up_model.get(body.get("id", ""))
    p = up_model.params_for(up, body)
    ss = max(1, min(3, int(body.get("supersample", 2))))
    full = bool(body.get("full", False))

    # Take a ticket *before* waiting on the lock, so a request already queued
    # here is superseded by a newer one arriving behind it rather than after it.
    mine = next_preview_gen()

    def superseded() -> bool:
        return is_superseded(mine)

    t0 = time.time()
    try:
        with RENDER_LOCK:
            out = render_tier(up, p, ss, full, should_cancel=superseded)
    except RenderCancelled:
        # 499, nginx's "client closed request". The client aborted this fetch the
        # moment it issued the newer one, so nothing is waiting for this body --
        # `api.ts` never sees it. Returning a status rather than an empty 200
        # keeps it out of the success path if anything ever does look.
        return Response(status_code=499, headers={"Cache-Control": "no-store"})
    ms = int((time.time() - t0) * 1000)

    return Response(
        content=iio.encode_preview(out),
        media_type=iio.PREVIEW_MEDIA_TYPE,
        headers={
            "X-Render-Ms": str(ms),
            "X-Render-Full": "1" if full else "0",
            "X-Render-Device": device_name(DEVICE),
            "Cache-Control": "no-store",
        },
    )
