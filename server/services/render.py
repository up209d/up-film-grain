"""The single place either preview tier is rendered."""

from __future__ import annotations

from ..models.upload import Frame, Upload
from ..runtime import ENGINE


def render_tier(fr: Upload | Frame, p: dict, ss: float, full: bool,
                progress=None, should_cancel=None):
    """Render one of the two tiers. The single place either tier is rendered.

    ``fr`` is what ``models.upload.params_for`` handed back: the photograph at
    the working resolution these parameters asked for. It is the ``Upload``
    itself when Prescaling Source is off and a ``Frame`` when it is on, and
    nothing here can tell the difference -- both carry ``arr``, ``proxy``,
    ``proxy_scale`` and an ``id``, which is the whole interface this needs.

    ``/api/preview`` and ``/api/export`` both come through here, and that is
    deliberate rather than tidy: an export is supposed to be **byte-for-byte**
    the live preview before it is enlarged, so "export what I am looking at"
    stays literally true, and the only way to guarantee that is for both to make
    the identical call. They used to be two call sites carrying duplicated
    literals, which is exactly the drift the invariant warns about -- and it
    became load-bearing the moment tile size stopped being a constant, because
    now the two would have to agree about a *computed* value.

    ``ss`` is a float, not an int: the menu offers 0.5 and 1.5 alongside the
    whole factors, and ``render_supersampled`` handles the fractional ones by
    rounding the working grid to whole pixels rather than the factor.
    """
    src, sc = (fr.arr, 1.0) if full else (fr.proxy, fr.proxy_scale)
    # Tile size from `ENGINE.tile_for`, not a constant. Per-tile overlap is fixed
    # padding that gets rendered and thrown away, so wider tiles are strictly
    # less work -- but the ceiling is memory, and the right ceiling differs per
    # machine, per preset (a wide kernel pads more) and per supersample. See
    # `tile_for` for the measurements.
    tile = ENGINE.tile_for(p, sc, int(src.shape[0]), int(src.shape[1]), ss)
    # The checkpoint id has to distinguish the *image*, the *working resolution*
    # and the *tier*: two photographs of the same dimensions with the same
    # parameters would otherwise key identically, the same photograph prescaled
    # and not is two different frames, and the proxy and the 1:1 render are two
    # more. The image and the resolution both come from `fr.id`, which is the
    # upload's id with the prescale target appended when there is one. The
    # working scale is in the key too, but only via `sc`, which is 1.0 for both
    # an untouched full-res source and the full tier -- hence naming the tier
    # outright.
    return ENGINE.render_image(
        src, p, sc, tile=tile, supersample=ss, progress=progress,
        should_cancel=should_cancel,
        checkpoint_id=f"{fr.id}:{'full' if full else 'proxy'}",
    )
