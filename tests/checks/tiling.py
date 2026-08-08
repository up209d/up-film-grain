"""tile independence, crop fidelity and zoom fidelity

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from tests.harness import Ctx, check, suite


@suite("tiling", "tile independence, crop fidelity and zoom fidelity")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 1. tile independence ------------------------------------------------
    print("tile independence (tiled render == single-pass render)")
    # Fractional factors included since 2026-08-08. Invariant 2 is the one at
    # risk there: a request like 1.5x cannot give a whole working grid on every
    # tile, so `render_supersampled` rounds to whole pixels and then derives
    # `scale`, `y0`, `x0` and `full_hw` from the grid it *actually* rendered.
    # Get that wrong and the noise lattice resolves to different global
    # coordinates than the geometry does, which shows up here and nowhere else.
    for ss in (0.5, 1, 1.5, 2, 3):
        a = eng.render_image(img, p, 1.0, tile=4096, supersample=ss)
        b = eng.render_image(img, p, 1.0, tile=128, supersample=ss)
        d = float(np.abs(a - b).max())
        check(f"supersample {ss}x", d < 2e-3, f"max delta {d:.2e}")

    # -- 2. crop render matches the full render ------------------------------
    print("\ncrop fidelity (1:1 preview == same region of the export)")
    full = eng.render_image(img, p, 1.0, supersample=2)
    crop = eng.render_crop(img, p, (180, 240, 220, 300), 1.0, 2)
    d = float(np.abs(full[180:400, 240:540] - crop).max())
    check("render_crop", d < 2e-3, f"max delta {d:.2e}")

    # -- 2b. zoomed-out view agrees with a full render at the same scale -----
    print("\nzoom fidelity (zoomed view == full render at that scale)")
    for z in (0.5, 0.25):
        small = eng.render_image(
            np.ascontiguousarray(iio.downscale(img, z)), p, z, tile=4096, supersample=2
        )
        sh, sw, _ = small.shape
        # Align the box to the zoom step. An origin that lands on a half
        # pixel at this scale resamples on a different grid phase, which is a
        # property of the test's box, not of the renderer.
        q = int(round(1 / z))
        by, bx = (int(0.25 * img.shape[0]) // q) * q, (int(0.25 * img.shape[1]) // q) * q
        bh, bw = (int(0.4 * img.shape[0]) // q) * q, (int(0.4 * img.shape[1]) // q) * q
        view = eng.render_view(img, p, (by, bx, bh, bw), z, 2)
        vy, vx = round(by * z), round(bx * z)
        ref = small[vy: vy + view.shape[0], vx: vx + view.shape[1]]
        n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
        d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
        check(f"zoom {int(z * 100)}%", d < 6e-3, f"max delta {d:.2e}")
