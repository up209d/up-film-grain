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
from ..engine.stages import normalize
from ..runtime import DEVICE

# Long edge of the working proxy used for live preview renders. Measured on a
# 24MP source: a full-resolution 2x pass is 7.8s, this is 1.3s. That gap is the
# whole reason the proxy exists -- it is what makes dragging a slider feel like
# editing rather than batch processing.
PROXY_LONG_EDGE = 2400

# The supersample factors the UI offers, and the only ones a request may ask
# for. A menu rather than a free number because each is a different bargain and
# there is no useful midpoint: 2 is the default and the look every preset was
# dialled in against, 3 costs 2.25x that for a little more clump resolution,
# 1 renders at the output grid and gives grain a hard pixel footprint, and the
# two below 1 render *smaller than the output* and scale up -- genuinely lossy,
# and there for machines that cannot afford anything else.
#
# Clamped to the list rather than to a range: a request for 2.7 is a client bug,
# and rounding it to the nearest offered value is a more useful answer than
# either honouring it or refusing it.
SUPERSAMPLES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


def _clamp_ss(v) -> float:
    """Nearest offered supersample factor. Junk falls back to the default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 2.0
    return min(SUPERSAMPLES, key=lambda s: abs(s - f))



class Upload:
    __slots__ = ("id", "name", "arr", "h", "w", "proxy", "proxy_scale",
                 "src_enc", "norm", "touched")

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
        # What Normalize measured from this photograph, or None until something
        # asks. Six floats; see `engine/stages/normalize.py`.
        #
        # Cached here rather than computed per render for two separate reasons,
        # and only the first is about speed. It is a pass over the whole frame,
        # so doing it per preview would put it in the drag loop -- but more
        # importantly the numbers have to be *the same* for every render of this
        # image, or the proxy preview and the 1:1 export would normalise
        # differently and "export what I am looking at" would stop being true.
        # One measurement per photograph makes that structural.
        #
        # Lazy rather than computed in `__init__` like `proxy`, because the
        # control ships off: a session that never switches it on never pays.
        self.norm: dict[str, float] | None = None
        self.touched = time.time()

    def norm_stats(self) -> dict[str, float]:
        """The metered correction for this photograph, measured once."""
        if self.norm is None:
            self.norm = normalize.meter(self.arr)
        return self.norm


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

    # Normalize's six measured floats ride alongside the values for the same
    # reason the LUT does: they are not quantities anyone dials, they are what
    # the stage measured from *this photograph*. Attached after sanitize and
    # rescale, both of which only touch keys in PARAMS and so leave these alone.
    #
    # **Plain floats, deliberately.** `checkpoint.upstream_signature` walks
    # `sorted(p)` and keeps anything that is an int or a float, so these land in
    # the checkpoint key automatically and two photographs can never share a
    # cached frame. A tuple or an array would be silently dropped by that filter
    # -- the LUT needs its own line in that function for exactly this reason --
    # and the symptom would be one photograph rendering with another's
    # correction, which is a plausible and wrong picture.
    #
    # Measured only when the control is on. The metering is a pass over the full
    # frame and the stage ships off, so an untouched session never pays for it.
    if p["normalize"] >= 0.5:
        p.update(up.norm_stats())
    return p
