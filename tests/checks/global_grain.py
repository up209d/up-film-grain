"""global grain: unmasked, tile-independent, and its chroma

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from server import params as P
from tests.harness import Ctx, check, suite


@suite("global_grain", "global grain: unmasked, tile-independent, and its chroma")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    big = cx.big
    sy, sx, ph, pw, ty, tx, i = cx.patches
    sm = cx.smooth_sigma
    # -- 5c. global grain: unmasked, and still tile-independent ---------------
    # This layer exists precisely to reach what the masks above protect, so it
    # is checked against the opposite expectation: the smooth patch must now be
    # as grainy as the textured one. It is applied last and touches no
    # statistic of the region, so it must not cost tile independence either --
    # which the default-parameter checks cannot see, since it ships at 0.
    print("\nglobal grain (unmasked overlay)")
    gp = P.sanitize({"global_intensity": 40, "global_size": 1.6, "global_opacity": 1.0})
    gout = eng.render_image(big, gp, 1.0, supersample=2)
    gres = gout - big
    g_sm = float(gres[sy + i:sy + ph - i, sx + i:sx + pw - i].std())
    g_tx = float(gres[ty + i:ty + ph - i, tx + i:tx + pw - i].std())
    check(
        "reaches smooth areas", g_sm > sm * 3.0 and g_sm > 0.6 * g_tx,
        f"smooth sigma {g_sm:.4f} vs masked-only {sm:.4f}, textured {g_tx:.4f}",
    )
    a = eng.render_image(img, gp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, gp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # Scale invariance: the overlay is specified in full-res pixels like every
    # other spatial quantity, so a zoomed view must still predict the export.
    view = eng.render_view(img, gp, (100, 100, 400, 400), 0.5, 2)
    small = eng.render_image(
        np.ascontiguousarray(iio.downscale(img, 0.5)), gp, 0.5, tile=4096, supersample=2
    )
    ref = small[50: 50 + view.shape[0], 50: 50 + view.shape[1]]
    n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
    d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
    check("scale invariance", d < 6e-3, f"max delta {d:.2e}")

    # -- 5b-i. global chroma: decorrelate the channels, hold the amplitude ----
    # Built as a mean-zero deviation added to the *existing* mono field rather
    # than by the main grain's mean-of-three recipe, so that chroma 0 is
    # bit-identical to the layer every current preset was dialled in against.
    # Three things to pin, and the first is the one a rewrite would break.
    print("\nglobal chroma grain")
    gc_grey = np.full((360, 520, 3), 0.5, dtype=np.float32)

    def gc_render(gc: float, smooth: float = 0.0) -> np.ndarray:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 40.0, "global_size": 4.0,
                     "global_opacity": 1.0, "global_smooth": smooth,
                     "global_chroma": gc})
        out = eng.render_image(gc_grey, P.sanitize(over), 1.0, tile=1024,
                               supersample=1)
        return out.astype(np.float64) - 0.5

    def gc_corr(d: np.ndarray) -> float:
        return float(np.corrcoef(d[:, :, 0].ravel(), d[:, :, 1].ravel())[0, 1])

    # At 0 the three channels must be the same field to the bit. Anything else
    # means the mono component was rebuilt and every shipped preset's global
    # layer just changed pattern.
    d0 = gc_render(0.0)
    spread = float(np.abs(d0[:, :, 0] - d0[:, :, 1]).max())
    check("chroma 0 is exactly monochrome", spread == 0.0,
          f"max channel spread {spread:.2e}")
    # Correlation is `1 - chroma` by construction, not by tuning -- which is the
    # whole reason the coefficients are solved rather than lerped.
    for gc, want in ((0.25, 0.75), (0.5, 0.5), (1.0, 0.0)):
        rho = gc_corr(gc_render(gc))
        check(f"chroma {gc} decorrelates to {want}", abs(rho - want) < 0.05,
              f"channel correlation {rho:+.3f}, wanted {want:+.2f}")
    # Amplitude must not ride along with it, or the slider is a second loudness
    # control. The residual drift is the +-1 clamp meeting a more gaussian
    # field, not the blend: pre-clamp the construction is flat to 0.6%.
    s0 = d0.std()
    for gc in (0.5, 1.0):
        r = gc_render(gc).std() / s0
        check(f"amplitude holds at chroma {gc}", abs(r - 1.0) < 0.05,
              f"{r * 100:.1f}% of monochrome")
    # A second noise field is a second thing `pad_for` has to cover, and it is
    # smoothed by the same kernel -- so re-run tile independence with both on.
    gcp = P.sanitize({"global_intensity": 40, "global_size": 12.0,
                      "global_opacity": 1.0, "global_smooth": 1.0,
                      "global_chroma": 1.0})
    a = eng.render_image(img, gcp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, gcp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # It only exists to colour the global layer, so with that layer off it must
    # be inert -- otherwise it is a colour grade, which is deferred.
    off = {k: 0.0 for k in P.NEUTRAL_ZERO}
    a = eng.render_image(img, P.sanitize({**off, "intensity": 32.0,
                                         "global_chroma": 0.0}), 1.0, tile=1024)
    b = eng.render_image(img, P.sanitize({**off, "intensity": 32.0,
                                         "global_chroma": 1.0}), 1.0, tile=1024)
    d = float(np.abs(a.astype(float) - b.astype(float)).max())
    check("inert with the global layer off", d == 0.0, f"max delta {d:.2e}")
