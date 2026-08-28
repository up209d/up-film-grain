"""The uploaded image: its proxy, the prescaled frames of it, the in-memory
registry, and the per-request parameter resolution that depends on the size the
image is being rendered at.

Two things in here resize a photograph and they are answering different
questions. ``Upload.proxy`` is a *fidelity* decision -- the same photograph,
rendered smaller so a slider stays responsive. ``Frame`` is a *look* decision
-- a different working resolution for the photograph itself, so that the
pipeline's lengths mean the same thing whatever came out of the camera. Every
``Frame`` has a proxy of its own for exactly that reason.
"""

from __future__ import annotations

import math
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


def prescale_dims(h: int, w: int, target_mp: float) -> tuple[int, int]:
    """The photograph's dimensions at ``target_mp`` megapixels.

    Only the pixel *count* is set; the aspect ratio is the photograph's own and
    both axes take the same factor, so it is preserved to within half a pixel.
    That half pixel is a hard requirement rather than a nicety: ``/api/source``
    still serves the untouched photograph at its original resolution, and the
    before/after wipe lines the two up by scaling both into one box on the
    client. An aspect ratio that drifted would show as the two layers parting
    company at one edge.

    Rounding is ``floor(x + 0.5)`` written out rather than ``round``, because
    the client mirrors this arithmetic to label the export and Python's
    ``round`` is banker's rounding where JavaScript's ``Math.round`` is not --
    they disagree on exactly the half-pixel case, which is the one case this
    function is built around. See ``web/src/models/prescale.ts``.
    """
    k = math.sqrt(target_mp * 1e6 / float(h * w))
    return max(1, int(math.floor(h * k + 0.5))), max(1, int(math.floor(w * k + 0.5)))


def prescale_target(p: dict) -> float | None:
    """The working size these parameters ask for, or ``None`` for the file's own.

    The one place "is prescaling actually doing anything" is answered, so the
    switch and a target that happens to equal the photograph's own size are the
    same answer everywhere downstream rather than in each caller.
    """
    if float(p.get("prescale", 0.0)) < 0.5:
        return None
    mp = float(p.get("prescale_mp", 0.0))
    return mp if mp > 0.0 else None


class Frame:
    """An upload resampled to a working resolution, resampled once.

    Everything downstream reads the same handful of attributes off this that it
    used to read off ``Upload`` -- ``id``, ``h``, ``w``, ``arr``, ``proxy``,
    ``proxy_scale``, ``norm_stats()`` -- so ``params_for`` and ``render_tier``
    neither know nor care which of the two they were handed. ``Upload.at(None)``
    returns the upload *itself* for exactly that reason: the original-resolution
    path is then not a parallel implementation that can drift out of step, it is
    the same object it has always been.

    Not a stage, and that is the point. Resampling here rather than inside the
    engine is what keeps both invariants untouched by this feature: no stage
    ever learns that the frame it is rendering is not the file, so tile
    independence and scale invariance are exactly the properties they were.
    """

    __slots__ = ("id", "up", "target_mp", "h", "w", "proxy_scale",
                 "_arr", "_proxy")

    def __init__(self, up: "Upload", target_mp: float) -> None:
        self.up = up
        self.target_mp = target_mp
        self.h, self.w = prescale_dims(up.h, up.w, target_mp)
        # From the *prescaled* dimensions, so the working scale the engine is
        # handed is exactly the one a real photograph of this size would give
        # it. This is most of what the feature buys: `proxy_scale` stops being
        # a function of what came out of the camera, so the proxy's `_MIN_CELL`
        # divergence from the 1:1 render is the same on every photograph.
        self.proxy_scale = min(1.0, PROXY_LONG_EDGE / float(max(self.h, self.w)))
        # The checkpoint id has to say which working resolution this is: the
        # engine's cache also keys on `h`, `w` and `scale`, so this is belt and
        # braces -- which is the right amount for a cache whose stale hit is a
        # plausible but wrong *photograph*.
        self.id = f"{up.id}@{target_mp:g}mp"
        self._arr: np.ndarray | None = None
        self._proxy: np.ndarray | None = None

    @property
    def arr(self) -> np.ndarray:
        """The whole photograph at the working resolution.

        Lazy, unlike the proxy's eager cousin on ``Upload``, because only the
        full tier and the 1:1 export ever ask: at 24MP this is ~288MB, and a
        session that never leaves the proxy must not pay for it.
        """
        if self._arr is None:
            self._arr = iio.resize_to(self.up.arr, self.h, self.w, DEVICE)
        return self._arr

    @property
    def proxy(self) -> np.ndarray:
        """The working proxy for this resolution, in **one** resample.

        Deliberately not `downscale(self.arr, self.proxy_scale)`. Going via
        `arr` would upscale a 6MP photograph to 24MP and immediately throw
        23.4MP of it away again -- 288MB and a second interpolation, for pixels
        within rounding of what one pass from the original gives. So the target
        size is computed from the prescaled dimensions and the pixels come
        straight from the source.
        """
        if self._proxy is None:
            ph = max(1, int(round(self.h * self.proxy_scale)))
            pw = max(1, int(round(self.w * self.proxy_scale)))
            self._proxy = iio.resize_to(self.up.arr, ph, pw, DEVICE)
        return self._proxy

    def norm_stats(self) -> dict[str, float]:
        """Normalize's metering -- the upload's, shared, and not re-measured.

        Deliberately measured on the photograph rather than on this frame.
        Normalize answers a *tonal* question, and a colour correction that
        moved when you changed working resolution would be a surprise nobody
        asked for. It also keeps `norm_white` -- a channel maximum, the one
        metered value a resample genuinely does move -- reading the real
        photograph, and keeps a 24MP array from being materialised purely to
        meter it.
        """
        return self.up.norm_stats()



class Upload:
    __slots__ = ("id", "name", "arr", "h", "w", "proxy", "proxy_scale",
                 "src_enc", "norm", "frame", "touched")

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
        # The one prescaled frame of this photograph that is currently live,
        # as `(target_mp, Frame)`. See `at()`.
        self.frame: tuple[float, Frame] | None = None
        self.touched = time.time()

    def norm_stats(self) -> dict[str, float]:
        """The metered correction for this photograph, measured once."""
        if self.norm is None:
            self.norm = normalize.meter(self.arr)
        return self.norm

    def at(self, target_mp: float | None) -> "Upload | Frame":
        """This photograph at a working resolution. Resampled once and kept.

        This cache is the whole feature working. A slider drag re-renders dozens
        of times and every one of those renders has to resample **nothing** --
        prescaling is a property of the photograph, not of the parameters being
        dragged, so paying for it per render would put a full-frame
        interpolation inside the drag loop for no change in its result.

        A single slot rather than a dict, because at most one is ever needed:
        `at(None)` and a photograph that is already the target size both return
        `self`, so the only thing worth holding is the one resampled frame.
        Changing the target is a deliberate act taken rarely, next to moving a
        slider, so rebuilding on a change is the right side of that trade -- and
        a dict would grow one full-resolution array per target the user tried.
        Dropped along with the upload by `reap()`.

        Returning `self` rather than an identity `Frame` is not an optimisation.
        It is what makes "Prescaling off behaves exactly as it did before this
        existed" structural: there is no second code path to keep in step,
        because there is no second object.
        """
        if target_mp is None:
            return self
        if prescale_dims(self.h, self.w, target_mp) == (self.h, self.w):
            # Already this size. Take the original path bit-for-bit rather than
            # a no-op resample -- `resize_to` would hand the array straight
            # back anyway, but the checkpoint id would say `@24mp` and split the
            # cache in two for one photograph.
            return self
        if self.frame is not None and self.frame[0] == target_mp:
            return self.frame[1]
        fr = Frame(self, target_mp)
        self.frame = (target_mp, fr)
        return fr


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


def params_for(up: Upload, body: dict) -> tuple[Upload | Frame, dict]:
    """The frame to render and the parameters to render it with.

    Returns both because the second depends on the first: prescaling decides
    what size the photograph *is*, and ``reference_mp`` rescaling then asks how
    that size compares to the one a preset was authored at. Resolving them in
    one place is what stops the two from being answered against different
    numbers -- see the order below.

    ``reference_mp`` is the megapixel count of the image a preset was dialled
    in on. Resolved per request rather than baked into the values when a preset
    loads, so switching to a different photo re-scales on its own instead of
    leaving the last photo's numbers behind.
    """
    p = P.sanitize(body.get("params"))
    # Prescaling read *after* sanitize, so a junk or out-of-range target has
    # already been clamped to the slider's own range and this cannot be asked
    # for a 4000MP frame.
    fr = up.at(prescale_target(p))
    ref = body.get("reference_mp")
    if ref:
        # `fr`, not `up`. Measuring against the file's own megapixels here would
        # resample the photograph to 24MP *and* rescale every length for the
        # 50MP frame it no longer is -- two corrections for one problem, and the
        # picture would come back with grain sized for a photograph that is not
        # being rendered. With prescaling on and a preset stamped at the same
        # size, this factor is exactly 1.0, which is the whole point of the
        # section.
        k = P.scale_factor(float(ref), fr.w * fr.h / 1e6)
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
        p.update(fr.norm_stats())
    return fr, p
