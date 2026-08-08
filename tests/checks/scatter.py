"""scatter: destroying detail without averaging anything

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from server import params as P
from tests.harness import Ctx, check, suite


@suite("scatter", "scatter: destroying detail without averaging anything")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    big = cx.big
    sy, sx, ph, pw, ty, tx, i = cx.patches
    # -- 5e3. scatter destroys detail without averaging anything -------------
    # The stage's whole claim is that it is not a filter: it displaces a share
    # of the pixels onto their neighbours and averages nothing, so the frame
    # loses its exactness and keeps its micro-contrast. That claim is only
    # worth anything measured *against a blur of the same reach*, which is the
    # tool it exists to replace -- so every reading here is a ratio to one.
    #
    # It also ships at 0, so like global grain and edge softening it needs its
    # own tile-independence run: it reads a pixel a full reach away, and a
    # displacement missing from pad_for seams a tiled export along exactly
    # that distance while every preview looks fine.
    print("\nscatter (displaces detail, averages nothing)")
    sc_rng = np.random.default_rng(11)
    # Half fine texture, half smooth ramp. The two halves test opposite
    # claims: texture must survive, and the ramp must come through untouched
    # without any mask in the code doing it.
    sc_img = np.zeros((400, 800, 3), np.float32)
    sc_img[:, :400] = np.clip(
        0.5 + 0.09 * sc_rng.standard_normal((400, 400, 1)).repeat(3, 2), 0.0, 1.0
    )
    sc_img[:, 400:] = np.linspace(0.3, 0.7, 400, dtype=np.float32)[None, :, None]
    sc_img = np.ascontiguousarray(sc_img)
    sc_off = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
        "edge_soften": 0,
    }

    def sc_run(over: dict, im: np.ndarray = sc_img, ss: int = 1) -> np.ndarray:
        return eng.render_image(im, P.sanitize({**sc_off, **over}), 1.0, supersample=ss)

    def sc_texture(o: np.ndarray) -> tuple[float, float]:
        """Fine-texture sigma and local contrast over the textured half."""
        t = o[:, 20:380]
        return float(t.std()), float(np.abs(np.diff(t.mean(2), axis=1)).mean())

    REACH = 3.0
    t0, c0 = sc_texture(sc_img)
    tb, cb = sc_texture(sc_run({"micro_blur": REACH}))
    ts, cs = sc_texture(sc_run({"scatter": 1.0, "scatter_radius": REACH}))
    check(
        "texture survives where an equal blur destroys it",
        ts > t0 * 0.95 and tb < t0 * 0.5,
        f"sigma {ts / t0 * 100:.0f}% kept vs {tb / t0 * 100:.0f}% for micro-blur {REACH}",
    )
    check(
        "micro-contrast survives too", cs > c0 * 0.85 and cb < c0 * 0.2,
        f"local contrast {cs / c0 * 100:.0f}% kept vs {cb / c0 * 100:.0f}% for the blur",
    )
    # No value may be invented. Every output pixel has to be a copy of a real
    # input pixel from within the reach -- that is what "averages nothing"
    # means, and it is the property a bilinear resample would quietly break.
    def sc_nearest(o: np.ndarray, im: np.ndarray, r: int) -> float:
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = np.abs(o - np.roll(np.roll(im, dy, 0), dx, 1)).max(2)
                best = d if best is None else np.minimum(best, d)
        k = r + 2
        return float(best[k:-k, k:-k].max())

    sc_full = sc_run({"scatter": 1.0, "scatter_radius": REACH})
    d_sc = sc_nearest(sc_full, sc_img, int(REACH))
    d_bl = sc_nearest(sc_run({"micro_blur": REACH}), sc_img, int(REACH))
    check(
        "every pixel is a copy of a real neighbour", d_sc < 1e-5 < d_bl,
        f"worst deviation {d_sc:.2e} (a blur of the same reach: {d_bl:.2e})",
    )
    # Self-masking, with no mask anywhere in the code: shuffling pixels that
    # already match their neighbours cannot change them, so the smooth half
    # comes through at the ramp's own slope times the travel and no more.
    smooth_d = float(np.abs(sc_full[:, 420:780] - sc_img[:, 420:780]).max())
    check(
        "smooth areas mask themselves", smooth_d < 0.004,
        f"max change on the ramp {smooth_d:.4f} (slope x reach = "
        f"{0.4 / 400 * REACH:.4f})",
    )
    # Amount is coverage, not opacity -- it moves a threshold on a uniform
    # field, so it has to scale the *number* of pixels that travel. A control
    # that faded the moved pixel in would be an average by another name.
    moved = [
        float((np.abs(sc_run({"scatter": a, "scatter_radius": 6.0})[:, :400]
                      - sc_img[:, :400]).max(2) > 1e-6).mean())
        for a in (0.0, 0.25, 0.5, 0.75)
    ]
    check(
        "amount is coverage", moved[0] == 0.0
        and all(moved[k] < moved[k + 1] for k in range(len(moved) - 1)),
        "fraction moved " + ", ".join(f"{m:.2f}" for m in moved),
    )
    # The stencils. The engine's table and the parameter's menu are one list in
    # two places, and the value a preset stores is the *index* -- so the first
    # thing to pin is that they line up, and every check below looks its
    # pattern up by name rather than hard-coding a number that would silently
    # come to mean something else.
    from server.engine import _SCATTER_NAMES  # noqa: E402

    menu = P.PARAM_BY_KEY["scatter_pattern"].choices
    check(
        "the stencil menu matches the engine's table", tuple(menu) == _SCATTER_NAMES,
        f"{len(menu)} stencils: {', '.join(menu)}",
    )
    check(
        "the slider spans exactly the stencils that exist",
        P.PARAM_BY_KEY["scatter_pattern"].max == len(_SCATTER_NAMES) - 1,
        f"max {P.PARAM_BY_KEY['scatter_pattern'].max:.0f} for "
        f"{len(_SCATTER_NAMES)} entries",
    )
    pat = {n: i for i, n in enumerate(_SCATTER_NAMES)}

    # A displacement parallel to an edge cannot move it, so a one-axis pattern
    # must leave an edge running along that axis *bit-exact* -- the sharpest
    # available statement that the stencil is real and not a relabelled
    # isotropic field.
    sc_h = np.zeros((300, 300, 3), np.float32)
    sc_h[150:] = 0.4
    sc_h += 0.3
    sc_h = np.ascontiguousarray(sc_h)
    sc_v = np.ascontiguousarray(np.transpose(sc_h, (1, 0, 2)).copy())
    st = {"scatter": 1.0, "scatter_radius": 4.0}
    par = float(np.abs(sc_run({**st, "scatter_pattern": pat["Horizontal"]}, sc_h)
                       - sc_h).max())
    perp = float(np.abs(sc_run({**st, "scatter_pattern": pat["Horizontal"]}, sc_v)
                        - sc_v).max())
    # Not `== 0`: the pipeline's own sRGB round trip has a ~6e-08 floor, which
    # is the same number the sharpening check calls "invents nothing". The
    # claim is that the stencil contributes exactly none of it.
    check(
        "a one-axis stencil cannot move an edge along that axis",
        par < 1e-6 and perp > 0.1,
        f"horizontal border {par:.2e} (float floor), vertical border {perp:.3f}",
    )

    # Every stencil's footprint -- the complete set of places a displaced pixel
    # can land. Enumerated straight out of _scatter_offsets by sweeping its two
    # uniform inputs, rather than inferred from a rendered frame: the choice
    # field gives each cell *one* direction, so a rendered probe samples the
    # shape a few points at a time and a sparse stencil can miss it entirely.
    # This is the only way to tell a diamond from a disc, or a star from a box,
    # without eyeballing it.
    import torch  # noqa: E402

    from server.engine import _scatter_offsets  # noqa: E402

    def footprint(name: str, spread: float = 1.0, reach: float = 12.0):
        n = 512
        sel = torch.linspace(0.0, 1.0, n).view(1, 1, 1, n)
        magn = torch.linspace(0.0, 1.0, n).view(1, 1, n, 1)
        dxf, dyf = _scatter_offsets(sel, magn, reach, spread, pat[name])
        dxf = np.unique(np.stack([
            dyf.expand(1, 1, n, n).reshape(-1).numpy(),
            dxf.expand(1, 1, n, n).reshape(-1).numpy(),
        ], 1), axis=0)
        return dxf[:, 0], dxf[:, 1]

    # Peak travel is what pad_for reserves tile overlap for, so no stencil may
    # exceed it. The bound is the reach plus one pixel, not the reach: dx and
    # dy are rounded to whole pixels *independently*, and two half-pixel
    # roundings in the same direction lengthen the vector by up to sqrt(2)/2.
    # pad_for carries the same +1 rather than leaning on its trailing slack.
    for name in _SCATTER_NAMES:
        dyf, dxf = footprint(name)
        rad = np.hypot(dyf, dxf)
        check(
            f"{name} never travels past the reach", rad.max() <= 12.0 + 1.0,
            f"peak travel {rad.max():.1f}px of a 12px reach + 1px rounding, "
            f"{len(rad)} landing sites",
        )

    # Diamond: |dx| + |dy| is the constant, so it reaches the full 12px on the
    # axes and only ~8.5px on the diagonals. A disc would be 12 both ways.
    dyf, dxf = footprint("Diamond", spread=0.0)
    l1 = np.abs(dyf) + np.abs(dxf)
    on_ax = np.hypot(dyf, dxf)[(np.abs(dyf) < 2) | (np.abs(dxf) < 2)].max()
    on_di = np.hypot(dyf, dxf)[(np.abs(np.abs(dyf) - np.abs(dxf)) < 2)].max()
    check(
        "Diamond is a rhombus, not a disc",
        l1.max() - l1.min() <= 2 and on_di < on_ax * 0.80,
        f"|dx|+|dy| held at {l1.min()}-{l1.max()}; reaches {on_ax:.1f}px on the "
        f"axes vs {on_di:.1f}px on the diagonals",
    )
    # Donut: the hole is the feature, and it has to survive Reach Spread 1 --
    # that is the setting that fills every other stencil solid.
    for spread in (0.0, 1.0):
        dyf, dxf = footprint("Donut", spread=spread)
        hole = np.hypot(dyf, dxf).min()
        solid = np.hypot(*footprint("Any", spread=spread)).min()
        check(
            f"Donut keeps its hole at spread {spread:.0f}",
            hole > 12.0 * 0.5 and (spread == 0.0 or solid < hole * 0.5),
            f"nearest landing {hole:.1f}px (Any at the same setting: {solid:.1f}px)",
        )
    # Star: eight spokes with alternate ones running short. Measured as the
    # reach along the axes against the reach on the diagonals -- a Box stencil
    # has the same eight directions and no such split, which is the thing this
    # has to distinguish itself from.
    def spoke_ratio(name: str) -> float:
        dyf, dxf = footprint(name, spread=0.0)
        r = np.hypot(dyf, dxf)
        ax = r[(np.abs(dyf) < 2) | (np.abs(dxf) < 2)].max()
        di = r[np.abs(np.abs(dyf) - np.abs(dxf)) < 2].max()
        return float(di / max(ax, 1e-6))
    star, box = spoke_ratio("Star"), spoke_ratio("Box")
    check(
        "Star runs long and short spokes where Box does not",
        star < 0.6 and box > 0.9,
        f"diagonal/axis reach {star:.2f} for Star, {box:.2f} for Box",
    )
    # Clump size. On a linear ramp the residual *is* slope times displacement,
    # so its lag-1 correlation reads the field's coherence directly: 0 when
    # every pixel chooses for itself, approaching 1 as whole tiles travel
    # together. The uniformity of the choice field is what this really pins --
    # quantising interpolated value noise would bias the stencil, which is why
    # _cell_noise exists at all.
    ramp = np.ascontiguousarray(
        np.repeat(np.linspace(0.2, 0.8, 600, dtype=np.float32)[None, :, None], 400, 0)
        .repeat(3, 2)
    )

    def sc_coherence(cell: float) -> float:
        r = (sc_run({"scatter": 1.0, "scatter_radius": 6.0, "scatter_cell": cell},
                    ramp) - ramp)[40:-40, 40:-40].mean(2)
        return float(np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1])

    co1, co8 = sc_coherence(1.0), sc_coherence(8.0)
    check(
        "clump size sets how much moves as one", co1 < 0.1 and co8 > 0.7,
        f"lag-1 correlation {co1:.2f} at 1px, {co8:.2f} at 8px",
    )
    # Reach Spread: 0 must put every displaced pixel on the shell at exactly
    # the reach, 1 must fill the disc. Measured on the ramp with a one-axis
    # stencil, where residual/slope is literally the travel in pixels.
    def sc_travel(spread: float) -> np.ndarray:
        r = np.abs((sc_run({"scatter": 1.0, "scatter_radius": 6.0,
                            "scatter_spread": spread,
                            "scatter_pattern": pat["Horizontal"]}, ramp)
                    - ramp)[40:-40, 40:-40].mean(2)) / (0.6 / 600)
        return r[r > 0.5]

    sh, disc = sc_travel(0.0), sc_travel(1.0)
    check(
        "spread 0 is a shell, 1 is a disc",
        sh.std() < 0.1 and disc.std() > 1.0,
        f"travel {sh.mean():.1f}+/-{sh.std():.2f}px at 0, "
        f"{disc.mean():.1f}+/-{disc.std():.2f}px at 1",
    )
    # Scatter must not be the reason grain changes, the same independence
    # micro-blur has: the masks are measured from the untouched tile input.
    def sc_grain(over: dict) -> float:
        on = eng.render_image(big, P.sanitize(over), 1.0, supersample=2)
        flat_p = P.sanitize({**over, "intensity": 0, "global_intensity": 0})
        off_ = eng.render_image(big, flat_p, 1.0, supersample=2)
        return float((on - off_)[ty + i:ty + ph - i, tx + i:tx + pw - i].std())

    sg0 = sc_grain({})
    sg1 = sc_grain({"scatter": 0.8, "scatter_radius": 4.0})
    check(
        "scatter does not cost grain", abs(sg1 - sg0) < sg0 * 0.05,
        f"grain {sg1 / sg0 * 100:.0f}% of unscattered",
    )
    for name, over in (
        ("fine", {"scatter": 1.0, "scatter_radius": 6.0}),
        ("clumped box", {"scatter": 0.6, "scatter_radius": 12.0,
                         "scatter_cell": 6.0, "scatter_pattern": 3}),
    ):
        q = P.sanitize(over)
        a = eng.render_image(img, q, 1.0, tile=4096, supersample=2)
        b = eng.render_image(img, q, 1.0, tile=128, supersample=2)
        d = float(np.abs(a - b).max())
        check(f"tile independence ({name})", d < 2e-3, f"max delta {d:.2e}")
    # Reach and clump are full-resolution lengths like every other spatial
    # quantity, so a zoomed view still has to predict the export.
    scp = P.sanitize({"scatter": 0.7, "scatter_radius": 6.0, "scatter_cell": 3.0})
    view = eng.render_view(img, scp, (100, 100, 400, 400), 0.5, 2)
    small = eng.render_image(
        np.ascontiguousarray(iio.downscale(img, 0.5)), scp, 0.5, tile=4096, supersample=2
    )
    ref = small[50: 50 + view.shape[0], 50: 50 + view.shape[1]]
    n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
    d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
    check("scale invariance", d < 6e-3, f"max delta {d:.2e}")
    # Scatter runs *before* micro-blur, and the order is not cosmetic -- it is
    # worth a check because swapping it back would look plausible in a diff and
    # change every preset that has both stages on.
    #
    # The tell is a hard edge. Blurring first and scattering second came out
    # *harder* on a border than the blur alone (60% of the original slope
    # against 34%), because displacing a blurred gradient by whole pixels drops
    # a hard step back into it -- scatter was undoing the blur. In this order
    # scatter shreds the border and the blur then averages the raggedness, so
    # the pair must land *below* blur-alone instead of above it.
    ord_edge = np.ascontiguousarray(np.repeat(
        np.where(np.arange(600)[None, :].repeat(300, 0) < 300, 0.2, 0.8)
        .astype(np.float32)[:, :, None], 3, axis=2))

    def ord_slope(over: dict) -> float:
        o = sc_run(over, ord_edge).mean(axis=2)
        return float(np.abs(np.diff(o, axis=1)).max(axis=1).mean())

    e_ref = ord_slope({})
    e_blur = ord_slope({"micro_blur": 1.0})
    e_both = ord_slope({"micro_blur": 1.0, "scatter": 0.85,
                        "scatter_radius": 3.0, "scatter_cell": 1.2})
    check(
        "scatter runs before micro-blur", e_both < e_blur * 0.95,
        f"hard-edge slope {e_blur / e_ref * 100:.0f}% of untouched for the blur "
        f"alone, {e_both / e_ref * 100:.0f}% with scatter ahead of it "
        f"(the old order measured 60%, above the blur)",
    )
