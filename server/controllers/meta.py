"""Read-only description of what the service can do: health, the parameter
schema the client builds its panel from, and the LUT registry.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import lut as lutlib
from .. import params as P
from ..engine import device_name, diskcache
from ..runtime import DEVICE, ENGINE

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"ok": True, "device": device_name(DEVICE)}


@router.get("/cache")
def cache() -> dict:
    """What the caches are holding, and where.

    Its own endpoint rather than a field on ``/api/health`` because the client
    polls this one on a timer while the picture is up, and health is asked once
    at boot. Everything in it is assembled by `engine.diskcache.stats` -- what
    counts as "the cache" is that module's business, and a controller that
    itemised the stores would need editing every time one is added.

    Read-only and cheap: the byte totals are counters the stores already keep,
    so nothing here walks the directory.
    """
    return diskcache.stats()


@router.post("/cache/clear")
def cache_clear() -> dict:
    """Throw away every cache that can be rebuilt. The photograph survives.

    What it drops is the checkpoint frames and the Global Grain textures, plus
    the mapped pages and the allocator's free list that `flush_ram` would have
    given back on its own a couple of seconds later. What it deliberately does
    **not** drop is the uploaded photograph's own arrays or a finished export --
    neither can be rebuilt from anything the process still has, so clearing them
    would not be freeing a cache, it would be losing the user's work.

    Opening a new photograph clears rather more than this (see
    `models.upload.reset`), because a different photograph makes the frames
    themselves unreachable and this does not.
    """
    ENGINE.ckpt.clear()
    ENGINE.clear_caches()
    ENGINE.flush_ram()
    return diskcache.stats()


@router.get("/params")
def get_params() -> dict:
    return P.schema()


@router.get("/luts")
def get_luts() -> dict:
    """Every 3D LUT the client can pick: the ``luts/`` tree plus any uploads.

    Its own endpoint rather than a field on ``/api/params`` because the list
    changes during a session -- uploading one has to add to it -- while the
    parameter schema never does. Nothing is parsed to answer this; the tree is
    only walked, so a library of 300 64-cubes costs nothing to browse.

    Subfolders are walked and each entry carries the ``group`` it came from, so
    the client can offer 300 LUTs as collapsed folders rather than one flat
    list. The ``id`` is the path relative to ``luts/`` without the extension.
    """
    return {"luts": lutlib.list_luts()}


@router.post("/lut")
def upload_lut(file: UploadFile = File(...)) -> dict:
    """Take a ``.cube`` from the user and hold it for this session.

    Parsed here rather than at render time so a malformed file is reported while
    the user is still looking at the file picker, instead of turning into a
    failed render several gestures later.
    """
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        lut = lutlib.add_upload(file.filename or "lut.cube", data)
    except lutlib.LutError as e:
        raise HTTPException(415, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not read that LUT: {e}")
    return lut.info()
