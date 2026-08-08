"""Serving the built web client.

Mounted rather than routed, and the missing-build case is deliberately fatal in
production: a production process with no client is broken, not degraded, so it
fails at import rather than booting happily and handing every visitor a 503.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..runtime import IS_DEV

DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


def mount(app: FastAPI) -> None:
    if DIST.is_dir():
        # The whole build directory, not just /assets: Vite copies web/public/
        # to the root of dist, so the favicons and the touch icon are siblings
        # of index.html rather than hashed assets. Mounting only /assets left
        # them 404ing. html=True is what serves index.html for "/".
        app.mount("/", StaticFiles(directory=DIST, html=True), name="client")

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
        raise RuntimeError(
            f"No web client at {DIST}. Build it first (./build.sh), or set "
            "APP_ENV=development to run the API on its own."
        )
