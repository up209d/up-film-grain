"""The single place either preview tier is rendered."""

from __future__ import annotations

from ..models.upload import Upload
from ..runtime import ENGINE


def render_tier(up: Upload, p: dict, ss: int, full: bool, progress=None,
                should_cancel=None):
    """Render one of the two tiers. The single place either tier is rendered.

    ``/api/preview`` and the proxy branch of ``/api/export`` both come through
    here, and that is deliberate rather than tidy: a preview-scale export is
    supposed to be **byte-for-byte** the live preview, so "export what I am
    looking at" stays literally true, and the only way to guarantee that is for
    both to make the identical call. They used to be two call sites carrying
    duplicated literals, which is exactly the drift the invariant warns about --
    and it became load-bearing the moment tile size stopped being a constant,
    because now the two would have to agree about a *computed* value.
    """
    src, sc = (up.arr, 1.0) if full else (up.proxy, up.proxy_scale)
    # Tile size from `ENGINE.tile_for`, not a constant. Per-tile overlap is fixed
    # padding that gets rendered and thrown away, so wider tiles are strictly
    # less work -- but the ceiling is memory, and the right ceiling differs per
    # machine, per preset (a wide kernel pads more) and per supersample. See
    # `tile_for` for the measurements.
    tile = ENGINE.tile_for(p, sc, int(src.shape[0]), int(src.shape[1]), ss)
    return ENGINE.render_image(
        src, p, sc, tile=tile, supersample=ss, progress=progress,
        should_cancel=should_cancel,
    )
