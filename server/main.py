"""FastAPI service for the film grain engine.

Assembly only: this module builds the app, applies the dev-only middleware, and
registers the routers from ``.controllers``. The endpoints themselves live one
per area in that package, the domain objects in ``.models``, and the shared
singletons (device, engine, locks) in ``.runtime``.

Endpoints are deliberately synchronous (``def``, not ``async def``) where they
do tensor work, so Starlette runs them in its threadpool and one slow render
cannot block the event loop.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .controllers import ROUTERS, client
from .runtime import IS_DEV

app = FastAPI(
    title="Film Grain Engine",
    # No interactive docs on a production build: this binds to loopback and has
    # no auth, so the smaller the surface the better.
    docs_url="/docs" if IS_DEV else None,
    redoc_url=None,
    openapi_url="/openapi.json" if IS_DEV else None,
)

if IS_DEV:
    # Only needed when Vite serves the client from its own origin on :5173.
    # In production FastAPI serves the client itself, so every request is
    # same-origin and this would be a hole for nothing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

for _router in ROUTERS:
    app.include_router(_router)

# Last: the client mount claims "/" and, in production, refuses to boot without
# a build. Registering it after the API routes keeps that failure about the
# client rather than about the endpoints.
client.mount(app)
