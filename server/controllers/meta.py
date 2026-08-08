"""Read-only description of what the service can do: health, the parameter
schema the client builds its panel from, and the LUT registry.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import lut as lutlib
from .. import params as P
from ..engine import device_name
from ..runtime import DEVICE

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"ok": True, "device": device_name(DEVICE)}


@router.get("/params")
def get_params() -> dict:
    return P.schema()


@router.get("/luts")
def get_luts() -> dict:
    """Every 3D LUT the client can pick: the ``luts/`` folder plus any uploads.

    Its own endpoint rather than a field on ``/api/params`` because the list
    changes during a session -- uploading one has to add to it -- while the
    parameter schema never does. Nothing is parsed to answer this; the folder is
    only listed, so a directory of 64-cubes costs nothing to browse.
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
