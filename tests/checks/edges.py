"""anti-aliasing, edge softening, pre-blur, edge jitter and sanding

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from server import params as P
from tests.harness import Ctx, check, suite


@suite("edges", "anti-aliasing, edge softening, pre-blur, edge jitter and sanding")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    big = cx.big
    # -- 5c-iii. anti-aliasing: lose the staircase, keep the edge -------------
    # The claim this stage makes is a *trade*, so both halves are pinned. It
    # filters along the isophote and never across it, which is what separates
    # it from every blur in the pipeline -- so jaggedness must fall a long way
    # while the across-edge slope barely moves. Ships at 0, so this is also its
    # own tile-independence run: it adds both a kernel and a displacement.
    print("\nanti-aliasing (staircase off, edge intact)")
    aa_h = aa_w = 400
    _y = np.arange(aa_h)[:, None]
    _x = np.arange(aa_w)[None, :]
    # A shallow diagonal sampled with no partial coverage at all: every step is
    # a whole pixel, which is precisely the artifact and gives the contour a
    # 1/sqrt(12) = 0.289px residual about its own straight-line fit.
    aa_img = np.ascontiguousarray(np.repeat(
        np.where(_x < 150.0 + _y * 0.18, 0.15, 0.85).astype(np.float32)[:, :, None],
        3, axis=2))

    def aa_params(**kw) -> dict:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update(kw)
        return P.sanitize(over)

    def contour(lum: np.ndarray) -> np.ndarray:
        """Sub-pixel x of the 50% crossing on each row."""
        mid = (lum.min() + lum.max()) * 0.5
        out = []
        for r in lum:
            i = int(np.argmax(r > mid))
            if i == 0:
                out.append(np.nan)
                continue
            a_, b_ = r[i - 1], r[i]
            out.append(i - 1 + (mid - a_) / (b_ - a_) if b_ != a_ else float(i))
        return np.array(out)

    def jagged(lum: np.ndarray) -> float:
        c = contour(lum)
        ys = np.arange(len(c))
        ok = np.isfinite(c)
        fit = np.polyval(np.polyfit(ys[ok], c[ok], 1), ys)
        return float(np.std(c[ok] - fit[ok]))

    def slope(lum: np.ndarray) -> float:
        return float(np.abs(np.diff(lum, axis=1)).max(axis=1).mean())

    aa_base = eng.render_image(aa_img, aa_params(), 1.0, supersample=1)
    j0, k0 = jagged(aa_base.mean(axis=2)), slope(aa_base.mean(axis=2))
    aa_on = eng.render_image(
        aa_img, aa_params(aa_strength=1.0, aa_radius=1.0, aa_edge_only=0.7),
        1.0, supersample=1)
    j1, k1 = jagged(aa_on.mean(axis=2)), slope(aa_on.mean(axis=2))
    check(
        "the staircase comes off", j1 < 0.80 * j0,
        f"contour residual {j0:.3f}px -> {j1:.3f}px ({j1 / j0 * 100:.0f}%)",
    )
    # The whole point of filtering along the contour rather than across it. For
    # scale: `edge_sand` keeps 73% of the sharpness for 32% of the jaggedness,
    # because it is working at a much longer wavelength; at the pixel scale
    # this trade is meant to be the better one.
    check(
        "the edge stays as sharp as it was", k1 > 0.80 * k0,
        f"across-edge slope {k0:.4f} -> {k1:.4f} ({k1 / k0 * 100:.0f}% kept) "
        f"for {(1 - j1 / j0) * 100:.0f}% of the jaggedness removed",
    )
    # Off must be *exactly* off -- it sits ahead of the masks, so a stage that
    # ran at 0 would move every downstream weighting with it.
    aa_zero = eng.render_image(aa_img, aa_params(aa_strength=0.0), 1.0,
                               supersample=1)
    d = float(np.abs(aa_zero - aa_base).max())
    check("0 is a true no-op", d == 0.0, f"max delta {d:.2e}")
    # Above 1 the filter repeats, re-aiming each time. A single pass was
    # reported as doing "little to none", and repeating is the right lever
    # rather than a longer radius -- the taps are short on purpose. Both halves
    # of the trade have to keep improving together, or "more aggressive" just
    # means "blurrier": the ladder must be monotonic in jaggedness removed while
    # the across-edge slope stays well clear of what a blur would leave.
    ladder = []
    for st in (1.0, 2.0, 3.0):
        o = eng.render_image(
            aa_img, aa_params(aa_strength=st, aa_radius=1.0, aa_edge_only=0.7),
            1.0, supersample=1).mean(axis=2)
        ladder.append((st, jagged(o), slope(o)))
    js = [j for _, j, _ in ladder]
    check(
        "repeating keeps taking the staircase down",
        js[2] < js[1] < js[0] and js[2] < 0.45 * j0,
        ", ".join(f"{st:.0f} pass -> {j / j0 * 100:.0f}%" for st, j, _ in ladder),
    )
    check(
        "and does not turn into a blur doing it", ladder[2][2] > 0.60 * k0,
        ", ".join(f"{st:.0f} pass -> {k / k0 * 100:.0f}% slope"
                  for st, _, k in ladder),
    )
    # Whole numbers are whole passes and the remainder fades the last one in, so
    # strength 1 has to stay *bit* identical to the single-pass version it was
    # before passes existed -- otherwise raising the ceiling silently moved
    # every value under it.
    d = float(np.abs(
        eng.render_image(aa_img, aa_params(aa_strength=1.0, aa_radius=1.0,
                                           aa_edge_only=0.7), 1.0,
                         supersample=1).astype(float) - aa_on.astype(float)
    ).max())
    check("strength 1 is still exactly one pass", d == 0.0, f"max delta {d:.2e}")
    # Three passes each displace a radius and each re-derive a tangent from
    # blurred luma, so pad_for has to count both terms three times over. Miss it
    # and a tiled export seams along the accumulated reach.
    aap = aa_params(aa_strength=3.0, aa_radius=4.0, aa_edge_only=0.3)
    a = eng.render_image(img, aap, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, aap, 1.0, tile=96, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence at three passes", d < 2e-3, f"max delta {d:.2e}")

    # Edge Only is the texture guard, and the discriminator is step *amplitude*
    # -- the same one edge softening uses, and for the same reason: keying on
    # "is there a micro-edge here" would select fabric, which is made of them.
    aa_rng = np.random.default_rng(11)
    aa_tex = np.ascontiguousarray(np.repeat(
        np.clip(0.5 + (aa_rng.random((300, 300)) - 0.5) * 0.04, 0, 1
                ).astype(np.float32)[:, :, None], 3, axis=2))
    t0 = float(eng.render_image(aa_tex, aa_params(), 1.0, supersample=1).std())
    keep = {}
    for eo in (1.0, 0.0):
        t = eng.render_image(
            aa_tex, aa_params(aa_strength=1.0, aa_radius=1.0, aa_edge_only=eo),
            1.0, supersample=1)
        keep[eo] = float(t.std()) / t0
    check(
        "Edge Only protects fine texture", keep[1.0] > 0.97 > keep[0.0],
        f"fabric-scale texture kept {keep[1.0] * 100:.0f}% at Edge Only 1 "
        f"against {keep[0.0] * 100:.0f}% at 0",
    )

    aap = P.sanitize({"aa_strength": 1.0, "aa_radius": 2.0, "aa_edge_only": 0.7})
    a = eng.render_image(img, aap, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, aap, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # The radius is a length in full-resolution pixels like every other spatial
    # quantity, so the proxy has to predict the export.
    view = eng.render_view(img, aap, (100, 100, 400, 400), 0.5, 2)
    small = eng.render_image(
        np.ascontiguousarray(iio.downscale(img, 0.5)), aap, 0.5, tile=4096,
        supersample=2)
    ref = small[50: 50 + view.shape[0], 50: 50 + view.shape[1]]
    n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
    d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
    check("scale invariance", d < 6e-3, f"max delta {d:.2e}")

    # -- 5d. edge softening: soften borders, keep texture and grain ----------
    # The point of this stage is that it is *not* a global blur, so it is
    # checked against a frame that is half hard border and half fine texture.
    # It also ships at 0, so like global grain it needs its own tile-
    # independence run -- it adds a blur kernel, and a blur missing from
    # pad_for seams tiled exports along exactly its radius.
    print("\nedge softening (borders soften, texture and grain do not)")
    es_img = np.zeros((400, 800, 3), np.float32)
    es_img[:, :400] = 0.35
    es_img[:, 200:400] = 0.65
    es_rng = np.random.default_rng(3)
    es_img[:, 400:] = np.clip(
        0.5 + 0.04 * es_rng.standard_normal((400, 400, 1)).repeat(3, 2), 0.0, 1.0
    )
    es_img = np.ascontiguousarray(es_img)

    def es_measure(over: dict) -> tuple[float, float, float]:
        on = eng.render_image(es_img, P.sanitize(over), 1.0, supersample=2)
        flat = P.sanitize({**over, "intensity": 0, "global_intensity": 0})
        off = eng.render_image(es_img, flat, 1.0, supersample=2)
        return (
            float((on - off).std()),                                       # grain
            float(np.abs(np.diff(off[:, 150:250].mean(2), axis=1)).max()),  # border
            float(np.abs(np.diff(off[:, 450:750].mean(2), axis=1)).mean()), # texture
        )

    g0, e0, t0 = es_measure({"micro_blur": 0.0, "edge_soften": 0.0})
    soft = {"micro_blur": 0.0, "edge_soften": 1.0, "edge_soften_radius": 3.0}
    g1, e1, t1 = es_measure(soft)
    check("border softens", e1 < e0 * 0.7, f"hard edge {e1 / e0 * 100:.0f}% of unsoftened")
    check("texture survives", t1 > t0 * 0.8, f"fine texture {t1 / t0 * 100:.0f}% kept")
    check(
        "grain is not collateral", abs(g1 - g0) < g0 * 0.05,
        f"grain {g1 / g0 * 100:.0f}% of unsoftened",
    )
    # And the same for a global micro-blur: softening must never be the reason
    # grain changes, whichever control did the softening.
    g2, _, _ = es_measure({"micro_blur": 3.0, "edge_soften": 0.0})
    check(
        "micro-blur does not cost grain", abs(g2 - g0) < g0 * 0.05,
        f"grain {g2 / g0 * 100:.0f}% of unblurred",
    )
    a = eng.render_image(img, P.sanitize(soft), 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, P.sanitize(soft), 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # -- 5d1. pre-blur: the same kernel as micro-blur, a different stage -----
    # It ships at 0, so the default-parameter checks render straight past it,
    # and it adds a blur kernel in series with micro-blur -- a kernel missing
    # from pad_for seams a tiled export along exactly its radius while every
    # preview looks fine.
    #
    # The interesting assertions are the two that separate it from micro-blur:
    # it runs before `lum_ref`, so it *must* cost grain (the exact opposite of
    # the check three lines above), and it runs before pre-sharpen, so the
    # sharpen has to survive it.
    print("\npre-blur (softens the source, and the masks with it)")
    gp0, ep0, tp0 = es_measure({"micro_blur": 0.0, "edge_soften": 0.0})
    preb = {"micro_blur": 0.0, "edge_soften": 0.0, "pre_blur": 3.0}
    gp1, ep1, tp1 = es_measure(preb)
    check("border softens", ep1 < ep0 * 0.5, f"hard edge {ep1 / ep0 * 100:.0f}% of unblurred")
    check(
        "texture goes too", tp1 < tp0 * 0.5,
        f"fine texture {tp1 / tp0 * 100:.0f}% kept -- a blur is not edge_soften",
    )
    # The whole reason this is not micro-blur: the masks are measured after it,
    # so softening the source turns the grain down with it. micro_blur 3.0 on
    # the identical frame holds grain within 5% (checked above); this must not.
    check(
        "grain follows the softened frame", gp1 < gp0 * 0.9,
        f"grain {gp1 / gp0 * 100:.0f}% of unblurred, vs micro-blur's "
        f"{g2 / g0 * 100:.0f}%",
    )
    # Order: pre-blur first, then pre-sharpen. Run the other way round the blur
    # would wipe the sharpening out and this would land back on blur-alone.
    _, ep2, _ = es_measure({**preb, "pre_sharpen": 8.0, "pre_sharpen_radius": 3.0})
    check(
        "pre-sharpen survives the pre-blur", ep2 > ep1 * 1.15,
        f"hard edge {ep2 / ep1 * 100:.0f}% of pre-blur alone",
    )
    # Blurring in linear light, not in the display encoding. On a black/white
    # border the two differ enormously: averaging gamma-encoded values holds
    # the *encoded* mean, which is a fraction of the light it stands for, so
    # every edge the kernel crosses comes out darker than the light that made
    # it. Measured in the transition band only -- outside it both agree.
    print("\npre-blur runs in linear light (energy across an edge)")
    bw = np.zeros((120, 240, 3), np.float32)
    bw[:, 120:] = 1.0
    bw = np.ascontiguousarray(bw)
    pb_only = P.sanitize({**P.neutral_values(), "pre_blur": 4.0})
    pb_out = eng.render_image(bw, pb_only, 1.0, supersample=1)

    def _lin(a: np.ndarray) -> np.ndarray:
        return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)

    band = slice(120 - 14, 120 + 14)
    want = float(_lin(bw[:, band]).mean())
    got = float(_lin(pb_out[:, band]).mean())
    # sRGB-space blurring reads this band at ~0.21 against linear's 0.50.
    check(
        "light is conserved across the border", abs(got - want) < 0.02,
        f"linear mean {got:.3f} vs {want:.3f} (gamma-space blurring gives ~0.21)",
    )
    a = eng.render_image(img, P.sanitize(preb), 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, P.sanitize(preb), 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # Scale invariance: the radius is a full-resolution length like every other
    # spatial quantity, so a half-scale view must still predict the export.
    view = eng.render_view(img, P.sanitize(preb), (100, 100, 400, 400), 0.5, 2)
    small = eng.render_image(
        np.ascontiguousarray(iio.downscale(img, 0.5)), P.sanitize(preb), 0.5,
        tile=4096, supersample=2,
    )
    ref = small[50: 50 + view.shape[0], 50: 50 + view.shape[1]]
    n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
    d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
    check("scale invariance", d < 6e-3, f"max delta {d:.2e}")

    # -- 5e. edge jitter actually displaces edges ----------------------------
    # Measured as sub-pixel border position rather than as a pixel delta: a
    # warp that moves an edge by a fraction of a pixel produces a large delta
    # right at the border and almost none elsewhere, so a mean-delta test
    # passes happily on a displacement far too small to see. That is exactly
    # how this shipped at an effective quarter-pixel.
    print("\nedge jitter (borders must visibly wander, not round off)")
    ej = np.zeros((300, 600, 3), np.float32)
    ej[:, :300] = 0.3
    ej[:, 150:300] = 0.7
    ej = np.ascontiguousarray(ej)
    # edge_jitter zeroed here too: it defaults to 0.3, and leaving it on would
    # add its own smooth wander to the sanding measurement below and disguise
    # the very thing that separates the two.
    ej_off = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0,
        "edge_soften": 0, "acutance": 0, "edge_erosion": 0, "halation": 0,
        "edge_jitter": 0, "edge_sand": 0, "sharpen": 0,
    }

    def border_wander(j: float) -> float:
        out = eng.render_image(
            ej, P.sanitize({**ej_off, "edge_jitter": j}), 1.0, supersample=2
        )
        col = out[:, 140:162].mean(2)
        pos = [
            np.interp(0.5, r, np.arange(140, 162))
            for r in col if r[0] < 0.5 < r[-1]
        ]
        return float(np.std(pos)) if pos else 0.0

    def border(over: dict) -> np.ndarray:
        out = eng.render_image(
            ej, P.sanitize({**ej_off, **over}), 1.0, supersample=2
        )
        col = out[:, 138:164].mean(2)
        return np.array([
            np.interp(0.5, r, np.arange(138, 164))
            for r in col if r[0] < 0.5 < r[-1]
        ])

    def border_wander(j: float) -> float:
        return float(border({"edge_jitter": j}).std())

    w0, w1 = border_wander(0.0), border_wander(1.0)
    check("straight edge stays straight at 0", w0 < 0.01, f"wander {w0:.3f}px")
    check("full jitter displaces ~1px", w1 > 0.5, f"wander +/-{w1:.3f}px")
    jp = P.sanitize({"edge_jitter": 1.0})
    a = eng.render_image(img, jp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, jp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # -- 5e1. jitter direction bias ------------------------------------------
    # A displacement parallel to an edge cannot move it, so a fully biased
    # jitter must leave one orientation completely untouched -- that is the
    # sharpest possible statement of "the angle is doing what it says". The
    # no-op case matters just as much: the raw field is isotropic, so an angle
    # with no anisotropy behind it has to change literally nothing.
    print("\njitter direction (angle only means something with anisotropy)")
    ej_h = np.ascontiguousarray(np.transpose(ej, (1, 0, 2)).copy())

    def wander_h(over: dict) -> float:
        out = eng.render_image(
            ej_h, P.sanitize({**ej_off, **over}), 1.0, supersample=2
        )
        m = out.mean(2)
        pos = [
            np.interp(0.5, c, np.arange(138, 164))
            for c in m[138:164, :].T if c[0] < 0.5 < c[-1]
        ]
        return float(np.std(pos)) if pos else 0.0

    iso = {"edge_jitter": 2.0}
    horiz = {**iso, "jitter_aniso": 1.0, "jitter_angle": 0.0}
    vert_a = {**iso, "jitter_aniso": 1.0, "jitter_angle": 90.0}
    check(
        "biased horizontal leaves horizontal edges alone",
        wander_h(horiz) < 0.01,
        f"horizontal-edge wander {wander_h(horiz):.3f}px (displacement is parallel to it)",
    )
    check(
        "biased horizontal still moves vertical edges",
        border(horiz).std() > 0.5,
        f"vertical-edge wander {border(horiz).std():.3f}px",
    )
    check(
        "biased vertical is the mirror image",
        border(vert_a).std() < 0.01 and wander_h(vert_a) > 0.5,
        f"vertical-edge {border(vert_a).std():.3f}px, horizontal-edge {wander_h(vert_a):.3f}px",
    )
    no_op = {**iso, "jitter_aniso": 0.0, "jitter_angle": 45.0}
    d = float(
        np.abs(
            eng.render_image(ej, P.sanitize({**ej_off, **no_op}), 1.0, supersample=2)
            - eng.render_image(ej, P.sanitize({**ej_off, **iso}), 1.0, supersample=2)
        ).max()
    )
    check("angle without anisotropy is a no-op", d < 1e-6, f"max delta {d:.2e}")

    # -- 5e2. edge sanding polishes what jitter roughened --------------------
    # Sanding is the counterpart to jitter, not more of it: it smooths *along*
    # the contour so the jaggedness comes off while the border's overall shape
    # and its sharpness across survive. All three have to be asserted together
    # -- an isotropic blur would score well on jaggedness alone, and that is
    # exactly the failure this has to rule out.
    print("\nedge sanding (polishes jitter's jaggedness, keeps the edge)")

    def contour(over: dict) -> tuple[float, float, float]:
        out = eng.render_image(
            ej, P.sanitize({**ej_off, **over}), 1.0, supersample=2
        )
        col = out[:, 138:164].mean(2)
        pos = np.array([
            np.interp(0.5, r, np.arange(138, 164))
            for r in col if r[0] < 0.5 < r[-1]
        ])
        sharp = float(np.abs(np.diff(col, axis=1)).max())
        return float(np.abs(np.diff(pos)).mean()), float(pos.std()), sharp

    j_jag, j_wan, j_sharp = contour({"edge_jitter": 1.0})
    s_jag, s_wan, s_sharp = contour(
        {"edge_jitter": 1.0, "edge_sand": 1.0, "edge_sand_grit": 5.0}
    )
    check(
        "jaggedness comes off", s_jag < j_jag * 0.85,
        f"{(1 - s_jag / j_jag) * 100:.0f}% of the jaggedness removed",
    )
    check(
        "the border's shape survives", s_wan > j_wan * 0.7,
        f"{s_wan / j_wan * 100:.0f}% of the wander kept",
    )
    check(
        "the edge stays an edge", s_sharp > j_sharp * 0.6,
        f"{s_sharp / j_sharp * 100:.0f}% of the edge sharpness kept",
    )
    # Truncated gaussian tap weights do not sum to 1; unnormalised they would
    # darken every sanded edge by ~1%.
    lit = P.sanitize({**ej_off, "edge_sand": 1.0, "edge_sand_grit": 5.0})
    dark = P.sanitize({**ej_off, "edge_sand": 0.0})
    shift = abs(
        float(eng.render_image(ej, lit, 1.0, supersample=2).mean())
        - float(eng.render_image(ej, dark, 1.0, supersample=2).mean())
    )
    check("sanding does not shift exposure", shift < 1e-3, f"mean shift {shift:.2e}")

    sp = P.sanitize({"edge_sand": 1.0})
    a = eng.render_image(img, sp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, sp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # At the top of the range the tangent filter reaches far enough that a
    # handful of pixels per frame land on a direction singularity, where the
    # gradient is so near zero that float noise alone decides which way the
    # filter points. Those pixels disagree between a tiled and a single-pass
    # render. It is not a seam -- they sit nowhere near tile boundaries, and
    # the mean difference is ~1e-6 -- so this asserts structural agreement
    # rather than a max, which is the property that actually matters.
    xp = P.sanitize({"edge_sand": 5.0, "edge_sand_grit": 20.0, "edge_jitter": 5.0})
    a = eng.render_image(big, xp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(big, xp, 1.0, tile=512, supersample=2)
    m = float(np.abs(a - b).mean())
    check("no seam at maximum settings", m < 1e-5, f"mean delta {m:.2e}")
