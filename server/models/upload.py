"""The uploaded image: its proxy, the in-memory registry, and the per-request
parameter resolution that depends on the image's own size.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import HTTPException

from .. import imageio as iio
from .. import lut as lutlib
from .. import params as P
from ..runtime import DEVICE

# Long edge of the working proxy used for live preview renders. Measured on a
# 24MP source: a full-resolution 2x pass is 7.8s, this is 1.3s. That gap is the
# whole reason the proxy exists -- it is what makes dragging a slider feel like
# editing rather than batch processing.
PROXY_LONG_EDGE = 2400


class Upload:
    __slots__ = ("id", "name", "arr", "h", "w", "proxy", "proxy_scale",
                 "src_enc", "touched")

    def __init__(self, uid: str, name: str, arr: np.ndarray) -> None:
        self.id = uid
        self.name = name
        self.arr = arr
        self.h, self.w = arr.shape[:2]
        self.proxy_scale = min(1.0, PROXY_LONG_EDGE / float(max(self.h, self.w)))
        self.proxy = iio.downscale(arr, self.proxy_scale, DEVICE)
        # The untouched image never changes, so it is encoded once and served
        # from here rather than re-encoded on every parameter change. Named for
        # "encoded" rather than a format: it follows `iio.encode_preview`, which
        # is JPEG now, and a full-resolution PNG of a 24MP source was ~76MB.
        self.src_enc: bytes | None = None
        self.touched = time.time()


UPLOADS: dict[str, Upload] = {}
_MAX_UPLOADS = 12


def reap() -> None:
    """Drop the oldest uploads; full-resolution arrays are large."""
    if len(UPLOADS) <= _MAX_UPLOADS:
        return
    for uid in sorted(UPLOADS, key=lambda k: UPLOADS[k].touched)[:-_MAX_UPLOADS]:
        UPLOADS.pop(uid, None)


def get(uid: str) -> Upload:
    up = UPLOADS.get(uid)
    if up is None:
        raise HTTPException(404, "Unknown image id -- it may have been evicted.")
    up.touched = time.time()
    return up


def params_for(up: Upload, body: dict) -> dict[str, float]:
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

    # The 3D LUT rides alongside the values rather than in them -- it is a
    # resource identified by name, not a quantity (see server/lut.py). Attached
    # here, after sanitize and rescale, both of which only ever touch keys that
    # are in PARAMS and so leave this alone.
    #
    # An unresolvable name is not an error: a preset can reference a LUT that
    # has since been renamed, or an upload from a previous run. But the mix has
    # to be zeroed with it, because `params.is_neutral` decides whether to
    # short-circuit the render from the numbers alone and cannot see the LUT --
    # leave a nonzero mix with no table and "show me the original" would return
    # a full render of a neutral pipeline, which is measurably softer than the
    # source rather than equal to it.
    lut = lutlib.get(body.get("lut"))
    p["lut"] = lut
    if lut is None:
        p["lut_amount"] = 0.0
    return p
