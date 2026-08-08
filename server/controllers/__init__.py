"""HTTP layer: one router per area, assembled by ``server.main``.

Controllers stay thin on purpose -- they validate the request, call a model or
the render service, and shape the response. Anything that survives a request
lives in ``..models``; anything a second caller would need lives in
``..services``.
"""

from __future__ import annotations

from . import client, export, images, meta, preview

ROUTERS = (meta.router, images.router, preview.router, export.router)

__all__ = ["ROUTERS", "client", "export", "images", "meta", "preview"]
