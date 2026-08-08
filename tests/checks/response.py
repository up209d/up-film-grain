"""luminance response, and grain on edges rather than flat areas

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
import torch
from server import params as P
from server.engine import (
    _blur,
)
from tests.harness import Ctx, check, suite


@suite("response", "luminance response, and grain on edges rather than flat areas")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 4. luminance response peaks in the 15-65% band ----------------------
    print("\nluminance response (grain must peak in midtones/shadows)")
    grad = np.zeros((256, 1024, 3), np.float32)
    grad[:] = np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None]
    # Halation and acutance are separate features that also live in the
    # residual; leaving them on measures their bloom as if it were grain and
    # masks the highlight falloff entirely. Isolate the grain.
    grain_only = P.sanitize({**p, "halation": 0.0, "acutance": 0.0})
    res = eng.render_image(grad, grain_only, 1.0, supersample=2) - grad
    band = []
    for i in range(0, 100, 5):
        a_, b_ = int(i / 100 * 1024), int((i + 5) / 100 * 1024)
        band.append((i, float(res[:, a_:b_].std())))
    peak = max(s for _, s in band)
    peak_at = [i for i, s in band if s == peak][0]
    hi = [s for i, s in band if i >= 95][0]
    check("peak inside 15-65%", 15 <= peak_at <= 65, f"peak at {peak_at}-{peak_at + 5}%")
    check("highlight suppression", hi / peak < 0.30, f"95-100% is {hi / peak * 100:.0f}% of peak")
    for i, s in band:
        print(f"      {i:3d}-{i + 5:3d}%  {'#' * int(s / peak * 34)}")

    # -- 4b. the luminance mask is measured off density, not off the picture --
    # The mask moved to step 6b on 2026-08-06 -- directly after the
    # characteristic curve and base fog, and *above* edge softening, jitter and
    # sanding. What that buys is testable and the test is a strong one.
    #
    # Put a hard black-to-white border on a frame and set the grain band to
    # mid-tones only, so both sides of the border are suppressed and the frame
    # should carry no grain anywhere. Then soften the border hard. Softening
    # invents a mid-tone ramp across it that was never in the photograph, and a
    # mask read *after* that stage believes it -- laying a ribbon of grain along
    # a border whose two sides are both meant to be clean.
    #
    # The reference number is not hypothetical: feeding the engine that same
    # softened frame as its *input* (which is exactly what the old order's mask
    # saw) puts 0.095 sigma of grain in the ribbon. Here it must be nothing.
    print("\nluminance response is keyed on density, not on the softened frame")
    step_plate = np.zeros((400, 400, 3), np.float32)
    step_plate[:, :200] = 0.03
    step_plate[:, 200:] = 0.97
    mid_only = {
        "intensity": 60, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
        # Both off, so what is measured is `m` alone rather than the edge mask
        # or the smooth-area guard also having an opinion about the border.
        "edge_bias": 0.0, "smooth_guard": 0.0,
        "lum_low": 0.35, "lum_high": 0.65, "shadow_falloff": 0.02,
        "highlight_falloff": 0.02, "shadow_drop": 1.0, "highlight_drop": 1.0,
        "edge_soften_radius": 30.0,
    }

    def ribbon_sigma(arr: np.ndarray, over: dict) -> float:
        q = P.sanitize({**mid_only, **over})
        got = eng.render_image(arr, q, 1.0, supersample=1)
        ref_ = eng.render_image(
            arr, P.sanitize({**mid_only, **over, "intensity": 0}), 1.0, supersample=1,
        )
        r_ = got - ref_
        return max(float(r_[:, 175:195].std()), float(r_[:, 205:225].std()))

    # The control: a frame that really does have mid-tones across the border
    # must be grainy there, or the check below is passing for the wrong reason.
    t_ = torch.from_numpy(step_plate).permute(2, 0, 1).unsqueeze(0)
    pre_softened = _blur(t_, 30.0).squeeze(0).permute(1, 2, 0).numpy()
    ctrl = ribbon_sigma(np.ascontiguousarray(pre_softened), {"edge_soften": 0.0})
    check(
        "a real mid-tone ramp is grainy", ctrl > 0.02,
        f"sigma {ctrl:.5f} where the frame itself carries the mid-tones",
    )
    for es in (0.0, 1.0):
        got = ribbon_sigma(step_plate, {"edge_soften": es})
        check(
            f"softening at {es:.0f} invents no grain at the border",
            got < ctrl * 0.05,
            f"sigma {got:.5f} against the mid-tone control's {ctrl:.5f}",
        )

    # -- 5. grain concentrates on edges rather than flat areas ---------------
    print("\nedge bias (grain onto micro-edges, not flat areas)")
    e = np.full((512, 512, 3), 0.45, np.float32)
    e[:, 256:] = 0.55
    r = eng.render_image(e, p, 1.0, supersample=2) - e
    flat_s = float(r[:, 20:200].std())
    edge_s = float(r[:, 246:266].std())
    check("edge > flat", edge_s > flat_s * 1.1, f"edge {edge_s:.4f} vs flat {flat_s:.4f} ({edge_s / flat_s:.2f}x)")

    # -- 5b. smooth areas must stay clean ------------------------------------
    print("\nsmooth-area guard (skin/sky must not be invaded)")
    # `big`, the patch geometry and the smooth-patch sigma are fixtures now:
    # global grain, edge destruction and sharpening are all measured against
    # this same sigma, and a second copy of it computed slightly differently
    # would quietly weaken all three.
    big = cx.big
    res = cx.big_residual
    sy, sx, ph, pw, ty, tx, i = cx.patches
    sm = cx.smooth_sigma
    tx_s = float(res[ty + i:ty + ph - i, tx + i:tx + pw - i].std())
    check(
        "smooth patch is quiet", sm < tx_s * 0.20,
        f"smooth sigma {sm:.4f} vs textured {tx_s:.4f} ({sm / tx_s * 100:.0f}% of textured)",
    )
    check("smooth patch near-clean", sm < 0.006, f"smooth sigma {sm:.4f} (want < 0.006)")
