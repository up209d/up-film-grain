"""Export jobs: the registry and the worker body.

**Every export is full size, and every export is the preview's look**
(2026-08-09, on request). The menu is unchanged -- 0.5, 1, 1.5, 2 (default) or
3 -- and so are its labels; what changed is underneath. Each entry now renders
the **proxy tier at that supersample**, exactly as ``/api/preview`` does, and
then blows the result up to the source's own pixel dimensions with
``imageio.upscale``.

Why: the file has to match the picture it was judged on. Every spatial length
scales with the frame, so a 1:1 render of the same numbers is not a sharper
version of the preview -- it is a *different* picture, with finer, denser grain
(see `docs/preview-and-export.md` for the two reasons it diverges). Rendering
the tier you were looking at and enlarging it is the only construction that
guarantees the match.

What it gives up, and it is worth naming: the enlargement adds no detail. Zoomed
in, the file carries the proxy's texture, just bigger. That is the deliberate
trade -- the supersample still buys real quality *within* that tier, it just no
longer changes which tier is rendered.

**Plus a sixth entry that opts out** (2026-08-09, also on request): ``full``
renders the source itself at scale 1.0 -- the real full-resolution render, at 1x
-- for when the frame's own finest grain is the thing wanted. It is deliberately
not the default, because it is the one file the preview cannot show you.

The output size is unchanged at every setting, so the menu still asks one
question rather than two: the file is always the working frame's own dimensions,
and the number says how finely it was rendered.

**Prescaling Source (2026-08-29) moved what "the source" means**, and it is the
only thing that can. With it on the photograph has been resampled to a working
resolution before any of this, so the frame every tier renders -- and therefore
the file every entry writes -- is that resolution rather than the file's.
``prescale_output`` is the one opt-out: it resamples the finished render back to
the photograph's own dimensions, which is a resample of grain and says so in its
help text.
"""

from __future__ import annotations

from .. import imageio as iio
from ..engine.diskcache import Blob
from ..runtime import DEVICE, RENDER_LOCK
from ..services.render import render_tier
from .upload import PROXY_LONG_EDGE, Frame, Upload

JOBS: dict[str, dict] = {}

#: How many finished exports keep their file. Older ones are deleted.
#:
#: There was no limit at all before 2026-08-29: a finished job kept its encoded
#: bytes on the dict for the life of the process, and a 24MP 16-bit PNG is
#: ~140MB, so ten exports in a session was 1.4GB of RAM that nothing would ever
#: free. That is a leak rather than a cache with a bad policy -- the bytes were
#: not being kept against a future hit, they were simply never dropped.
#:
#: The bytes live on the SSD now (`engine.diskcache.Blob`), so what this bounds
#: is disk rather than memory, and four is chosen against what the *client*
#: does: it downloads a job as soon as it reports done. A finished file is
#: therefore live only for the moment between those two, and the depth exists
#: for the browser that was closed mid-download or the user who wants the last
#: one again -- not as a library.
_KEEP_FINISHED = 4


def _reap_finished() -> None:
    """Delete all but the newest `_KEEP_FINISHED` finished exports.

    Only finished ones are considered. A queued or rendering job holds no file
    yet, and removing its record would strand the worker writing into it.
    """
    done = sorted(
        (j for j in list(JOBS.values())
         if j.get("status") in ("done", "error")),
        key=lambda j: j.get("created", 0.0),
    )
    for j in done[:-_KEEP_FINISHED] if len(done) > _KEEP_FINISHED else []:
        blob = j.pop("blob", None)
        if blob is not None:
            blob.release()
        JOBS.pop(j.get("id"), None)


def run_export(job_id: str, fr: Upload | Frame, p: dict, fmt: str, ss: float,
               quality: int, full: bool = False,
               out_hw: tuple[int, int] | None = None,
               preset: dict | None = None,
               edge: int = PROXY_LONG_EDGE) -> None:
    """Render, resize to the requested output size, encode.

    ``preset`` is the `load_presets` record the look came from, or ``None``.
    It reaches the file as EXIF (JPEG) or a text chunk (PNG) and nothing else
    uses it.

    ``fr`` is the photograph at its working resolution and ``out_hw`` is the
    size the *file* is written at -- the same numbers unless Prescaling Source is
    on and set to write the photograph's own dimensions. The caller decides,
    because it is the caller that has the `Upload` to compare against; this only
    has the frame.

    ``edge`` is the proxy long edge the preview was rendered at, passed straight
    through so this renders the tier that was on screen. Unused when ``full``.
    """
    job = JOBS[job_id]
    try:
        def progress(f: float) -> None:
            job["progress"] = round(float(f), 3)

        with RENDER_LOCK:
            job["status"] = "rendering"
            # `render_tier(..., full=False)` -- the *identical* call
            # `/api/preview` makes, which is the whole point and not a
            # convenience. "Export what I am looking at" is only literally true
            # while both go through one call site; two sites carrying the same
            # literals is exactly how they drift apart, and nothing on screen
            # would show it.
            #
            # `full=True` is the one entry that opts out and renders the source
            # at 1.0. It goes through the same call rather than round the side
            # of it, so the 1:1 export and the `Render 1:1` preview are the same
            # pixels for the same reason the other five are.
            out = render_tier(fr, p, ss, full, edge, progress=progress)
            # Then to the frame's own dimensions -- or the photograph's, if
            # `prescale_output` asked for that. A pass-through returning the
            # array itself when the size already matches, which is the common
            # case: a source no bigger than the proxy, and the full tier, both
            # land here already at size.
            #
            # `resize_to` rather than `upscale`, and the difference is not
            # cosmetic. Writing a prescaled-up photograph back at its own size
            # is a **reduction** of a finished, grainy frame, and reducing grain
            # without antialias folds it above Nyquist into visible crawl.
            # `upscale` is right when the only direction is up, which stopped
            # being true when prescaling could enlarge the input.
            th, tw = out_hw if out_hw is not None else (fr.h, fr.w)
            out = iio.resize_to(out, th, tw, DEVICE)
            job["status"] = "encoding"
            # Stamped with the preset that made it, when the caller named
            # one. A hand-dialled export writes no metadata at all -- see
            # `imageio._description`.
            data = iio.encode(out, fmt, quality, preset)

        # Straight to the SSD. The download route serves the file with
        # `FileResponse`, so these bytes never come back through Python's heap
        # -- the encode is the only moment the whole export exists in memory.
        job["blob"] = Blob(job_id, data)
        job["size"] = len(data)
        job["status"] = "done"
        job["progress"] = 1.0
        _reap_finished()
    except Exception as e:  # surfaced to the client rather than swallowed
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        _reap_finished()
