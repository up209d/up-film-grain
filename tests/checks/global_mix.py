"""master opacity and global smoothness

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import params as P
from tests.harness import gridiness
from tests.harness import Ctx, check, suite


@suite("global_mix", "master opacity and global smoothness")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 5b-ii. master opacity: a cross-fade over the untouched source --------
    # The two ends have to be *bit* exact, not close. Opacity 0 is a second
    # "show me the original" path and inherits that stage's trap: the
    # supersample round trip is not the identity, so blending anywhere inside
    # render() would hand back an image 1e-01 softer than the source and call
    # it "no effect". Checked at every supersample for exactly that reason.
    print("\nmaster opacity (cross-fade over the untouched source)")
    mo_ref = P.sanitize(None)
    for ss in (1, 2, 3):
        z = eng.render_image(img, P.sanitize({"master_opacity": 0.0}), 1.0,
                             supersample=ss)
        d = float(np.abs(z - img).max())
        check(f"0 returns the source at {ss}x", d == 0.0, f"max delta {d:.2e}")
    for ss in (1, 2):
        a = eng.render_image(img, P.sanitize({"master_opacity": 1.0}), 1.0,
                             supersample=ss)
        b = eng.render_image(img, mo_ref, 1.0, supersample=ss)
        d = float(np.abs(a - b).max())
        check(f"1 is the untouched pipeline at {ss}x", d == 0.0,
              f"max delta {d:.2e}")
    # Linear in between, or it is not a cross-fade. Measured against the two
    # ends rather than against a formula, so a stage that quietly re-ran at
    # partial strength would show up.
    full = eng.render_image(img, mo_ref, 1.0, supersample=2)
    worst = 0.0
    for op in (0.25, 0.5, 0.75):
        got = eng.render_image(img, P.sanitize({"master_opacity": op}), 1.0,
                               supersample=2)
        worst = max(worst, float(np.abs(got - (img + (full - img) * op)).max()))
    check("the middle is a straight cross-fade", worst < 1e-6,
          f"worst deviation from the exact blend {worst:.2e}")
    # It dials the *whole* pipeline back, not just the grain, so a stage that
    # was applied after the blend would keep its full strength here.
    heavy = P.sanitize({"halation": 0.8, "intensity": 60, "sharpen": 4.0,
                        "dust": 40, "light_leak": 4})
    h1 = eng.render_image(img, heavy, 1.0, supersample=2)
    h5 = eng.render_image(img, {**heavy, "master_opacity": 0.5}, 1.0,
                          supersample=2)
    r = float(np.abs(h5 - img).mean()) / float(np.abs(h1 - img).mean())
    check("everything scales together", abs(r - 0.5) < 0.02,
          f"mean deviation from the source is {r * 100:.1f}% of full strength "
          "with halation, grain, sharpening, dust and leaks all on")
    # Per-pixel against the tile's own input, so this must be free -- but it is
    # the last thing in the pipeline and a mistake here would seam every export.
    mop = P.sanitize({"master_opacity": 0.4})
    a = eng.render_image(img, mop, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, mop, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # -- 5c-ii. global smoothness: soften the grain, keep the amplitude -------
    # This was built to remove the value-noise quilt, and there is no longer a
    # quilt for it to remove -- `_grain_points` scores under 0.05 on `gridiness`
    # before any smoothing, against that field's 1.4-1.7. So what it has to be
    # now is what it always claimed to be for the strength half: a *shape*
    # control that visibly softens the grain at constant loudness.
    print("\nglobal grain smoothing (the shape, not the strength)")
    gs_grey = np.full((512, 512, 3), 0.5, np.float32)
    GS_CELL = 20.0

    def gs_render(sm: float) -> np.ndarray:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 20.0, "global_size": GS_CELL,
                     "global_opacity": 1.0, "global_smooth": sm})
        return eng.render_image(
            gs_grey, P.sanitize(over), 1.0, tile=1024, supersample=1)

    gs_off, gs_on = gs_render(0.0), gs_render(1.0)
    # It starts ungridded and must stay that way -- pinned at both ends so a
    # future field that reintroduced a lattice could not hide behind the blur.
    for tag, im in (("unsmoothed", gs_off), ("smoothed", gs_on)):
        q = gridiness(im.mean(axis=2), GS_CELL)
        check(f"the field is free of the lattice grid ({tag})", q < 0.35,
              f"phase-binned gridiness {q:.3f} (value noise scored 1.4-1.7)")
    # Softening means the gradient falls while the amplitude does not -- the two
    # together are what separate a shape control from a volume control, and
    # either alone would pass on a stage that did the wrong thing.
    g0 = float(np.abs(np.diff(gs_off.mean(axis=2), axis=1)).mean())
    g1 = float(np.abs(np.diff(gs_on.mean(axis=2), axis=1)).mean())
    check("it visibly softens the grain", g1 < 0.7 * g0,
          f"mean |gradient| {g0:.5f} -> {g1:.5f} ({g1 / g0 * 100:.0f}%)")
    # The whole reason the blur carries an analytic gain: a structure control
    # that quietly turns the layer down leaves Global Intensity fighting it,
    # and "smoother" becomes indistinguishable from "less". Same rule `_fbm`
    # follows for Octaves.
    s0 = float(gs_off.mean(axis=2).std())
    worst = 0.0
    for sm in (0.25, 0.5, 0.75, 1.0):
        r = float(gs_render(sm).mean(axis=2).std()) / s0
        worst = max(worst, abs(r - 1.0))
    check(
        "strength is held constant across the slider", worst < 0.06,
        f"worst amplitude drift {worst * 100:.1f}% over smoothness 0.25-1.0",
    )
    # The gain is a closed form in sigma/cell, so it has to hold at every clump
    # size and not just the one it was fitted at.
    for cell in (4.0, 12.0):
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 20.0, "global_size": cell,
                     "global_opacity": 1.0})
        a = eng.render_image(gs_grey, P.sanitize({**over, "global_smooth": 0.0}),
                             1.0, tile=1024, supersample=1)
        b = eng.render_image(gs_grey, P.sanitize({**over, "global_smooth": 1.0}),
                             1.0, tile=1024, supersample=1)
        r = float(b.mean(axis=2).std()) / float(a.mean(axis=2).std())
        check(f"the gain holds at a {cell:.0f}px clump", abs(r - 1.0) < 0.08,
              f"amplitude {r * 100:.1f}% of unsmoothed")
    # It is a blur on the noise field, so it is a kernel `pad_for` has to
    # cover -- and a kernel missing from pad_for seams a tiled export along
    # exactly its radius while every preview looks fine.
    smp = P.sanitize({"global_intensity": 40, "global_size": 12.0,
                      "global_opacity": 1.0, "global_smooth": 1.0})
    a = eng.render_image(img, smp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, smp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence with smoothing on", d < 2e-3, f"max delta {d:.2e}")
