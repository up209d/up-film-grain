"""output sharpening

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import params as P
from tests.harness import Ctx, check, suite


@suite("sharpen", "output sharpening")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    big = cx.big
    sy, sx, ph, pw, ty, tx, i = cx.patches
    # -- 5f. output sharpening cranks existing grain, invents none -----------
    # The stage is an unsharp mask placed last precisely so the detail it
    # amplifies is the grain. Two things have to hold: with grain on it must
    # raise the grain, and with grain off it must add nothing of its own --
    # on a flat field there is no high-frequency content, so a pure amplifier
    # has nothing to amplify and must be a no-op.
    print("\noutput sharpening (amplifies existing grain, generates none)")
    sharp = {"sharpen": 0.8, "sharpen_radius": 1.0}

    def grain_sigma(over: dict) -> float:
        on = eng.render_image(big, P.sanitize(over), 1.0, supersample=2)
        flat_p = P.sanitize({**over, "intensity": 0, "global_intensity": 0})
        off = eng.render_image(big, flat_p, 1.0, supersample=2)
        return float((on - off)[ty + i:ty + ph - i, tx + i:tx + pw - i].std())

    gs0, gs1 = grain_sigma({"sharpen": 0.0}), grain_sigma(sharp)
    check("grain gets cranked", gs1 > gs0 * 1.2, f"grain {gs1 / gs0 * 100:.0f}% of unsharpened")

    plain = np.full((256, 256, 3), 0.5, np.float32)
    quiet = P.sanitize({
        **sharp, "intensity": 0, "global_intensity": 0, "sharpen": 1.5,
        "edge_erosion": 0, "edge_jitter": 0, "acutance": 0, "micro_blur": 0,
        "halation": 0, "edge_soften": 0,
    })
    d = float(np.abs(eng.render_image(plain, quiet, 1.0, supersample=1) - plain).max())
    check("invents no noise on a flat field", d < 1e-5, f"max delta {d:.2e}")

    a = eng.render_image(img, P.sanitize(sharp), 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, P.sanitize(sharp), 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
