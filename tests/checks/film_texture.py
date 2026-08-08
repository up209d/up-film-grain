"""film texture: dust, hair, scratches, light leaks and speck shape

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import math
import numpy as np
from server import params as P
from tests.harness import TEX_OFF, Ctx, check, suite


@suite("film_texture", "film texture: dust, hair, scratches, light leaks and speck shape")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 5g. film texture: physical damage, drawn without a list of objects --
    # Dust and scratches are the classic way to break tile independence:
    # scatter a list of specks and an export splits one across two tiles, or
    # draws a different list per tile. These are thresholded noise fields in
    # global coordinates instead, so the checks are that each mark type has
    # the geometry it claims, and that the whole section survives tiling.
    print("\nfilm texture (dust, scratches, hair, light leak)")
    tex_off = dict(TEX_OFF)
    plain = cx.plain

    def marks(over: dict) -> tuple[float, float, float]:
        o = eng.render_image(plain, P.sanitize({**tex_off, **over}), 1.0, supersample=2)
        m = np.abs(o - plain).max(2) > 0.04

        def run(mask: np.ndarray) -> float:
            tot = cnt = 0
            for row in mask:
                d = np.diff(np.concatenate(([0], row.astype(int), [0])))
                s, e = np.where(d == 1)[0], np.where(d == -1)[0]
                if len(s):
                    tot += (e - s).sum()
                    cnt += len(s)
            return tot / max(cnt, 1)

        return float(m.mean()) * 100, run(m), run(m.T)

    d_cov, d_h, d_v = marks({"dust": 120.0})
    check(
        "dust is sparse and compact",
        d_cov < 1.5 and max(d_h, d_v) / max(min(d_h, d_v), 0.01) < 2.0,
        f"{d_cov:.2f}% of frame, {d_v / max(d_h, 0.01):.1f}:1 aspect",
    )
    s_cov, s_h, s_v = marks({"scratches": 20.0})
    check(
        "scratches run along the film", s_cov < 1.0 and s_v / max(s_h, 0.01) > 8.0,
        f"{s_cov:.2f}% of frame, {s_v / max(s_h, 0.01):.0f}:1 aspect ({s_h:.1f}px wide)",
    )
    h_cov, h_h, h_v = marks({"hair": 12.0})
    check(
        "hair is a thin filament", 0.02 < h_cov < 1.0 and min(h_h, h_v) < 4.0,
        f"{h_cov:.2f}% of frame, {min(h_h, h_v):.1f}px wide",
    )
    # A gating field coarser than the frame is a constant, not a mask -- that
    # bug rendered hair as nothing at all. Every mark type must actually
    # appear on a frame of ordinary size.
    for name, over in (
        ("dust", {"dust": 120.0}), ("scratches", {"scratches": 20.0}),
        ("hair", {"hair": 12.0}), ("light leak", {"light_leak": 6.0}),
    ):
        cov = marks(over)[0]
        # Counts of small marks cover very little area -- 150 one-pixel specks
        # on a 1.3MP frame is ~0.01%. The floor only has to catch "nothing at
        # all", which is the failure that actually happened.
        check(f"{name} actually renders", cov > 0.004, f"{cov:.3f}% of frame")

    # Softening must make marks softer *on average* while leaving the
    # population varied -- a uniform blur would pass a mean test and be
    # exactly the artificial-looking result this exists to avoid. So both the
    # mean slope and the spread are asserted.
    def slope_and_spread(over: dict) -> tuple[float, float]:
        o = eng.render_image(plain, P.sanitize({**tex_off, **over}), 1.0, supersample=2)
        d = np.abs(o - plain).max(2)
        g = np.abs(np.diff(d, axis=1))
        e = g[g > 0.01]
        peaks = []
        for row in d:
            idx = np.where(row > 0.02)[0]
            if not len(idx):
                continue
            for s in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                peaks.append(row[s].max())
        pk = np.array(peaks)
        return float(e.mean()), float(pk.std() / max(pk.mean(), 1e-9))

    for key, soft_key in (
        ("dust", "dust_soften"),
        ("scratches", "scratch_soften"),
        ("hair", "hair_soften"),
    ):
        name = key
        n = {"dust": 200.0, "scratches": 30.0, "hair": 16.0}[key]
        crisp, _ = slope_and_spread({key: n, soft_key: 0.0})
        sof, _ = slope_and_spread({key: n, soft_key: 1.0})
        # Spread is measured at *default* softness, not maximum. Softening
        # blends neighbouring marks together by design, so at 1.0 the marks
        # are deliberately less distinct -- asserting non-uniformity there is
        # asserting against the feature.
        _, spread1 = slope_and_spread({key: n})
        check(
            f"{name} softening actually softens", sof < crisp * 0.9,
            f"mean edge slope {crisp:.4f} -> {sof:.4f} ({(1 - sof / crisp) * 100:.0f}% softer)",
        )
        check(
            f"{name} stays non-uniform", spread1 > 0.3,
            f"per-mark brightness spread {spread1 * 100:.0f}% of mean",
        )

    # -- 5g-2. dust and hair are drawn, so their counts are exact ------------
    # Both were rewritten on 2026-08-06 from a threshold on a noise field to a
    # list of drawn marks. The reported bug was "I set Hair Count to 1 and I see
    # more than one hair", and it was structural: a level set of a field is not
    # one curve, so one unit of "hair" drew however many separate arcs the field
    # happened to cross 0.5 along. Dust had the same problem in a milder form --
    # its count went through a fitted blob-per-cell constant good to about a
    # factor of 1.5.
    #
    # So the check is the count itself, and it has to be exact rather than
    # approximate, because "roughly N" is precisely what was wrong.
    def components(mask: np.ndarray, min_px: int = 1) -> list[int]:
        """Sizes of the 8-connected components of ``mask``, largest first."""
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ys, xs = np.nonzero(mask)
        pts = list(zip(ys.tolist(), xs.tolist()))
        for q in pts:
            parent[q] = q
        for y, x in pts:
            for dy, dx in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
                nb = (y + dy, x + dx)
                if nb in parent:
                    ra, rb = find(nb), find((y, x))
                    if ra != rb:
                        parent[rb] = ra
        sizes: dict[tuple[int, int], int] = {}
        for q in pts:
            r_ = find(q)
            sizes[r_] = sizes.get(r_, 0) + 1
        return sorted((v for v in sizes.values() if v >= min_px), reverse=True)

    def mark_mask(over: dict, thr: float = 0.04) -> np.ndarray:
        o = eng.render_image(plain, P.sanitize({**tex_off, **over}), 1.0, supersample=2)
        return np.abs(o - plain).max(2) > thr

    # Small counts across several seeds, because "1 means 1" has to hold for
    # every frame rather than on average.
    for key, extra, min_px in (
        ("dust", {"dust_size": 8.0}, 3),
        # 20px floor for hair: a filament fades to nothing at its tapered tip,
        # and the last pixel or two of that fade sits within a hair's breadth of
        # any threshold you pick. Those are not extra hairs -- measured, the
        # detached remnant is a single pixel at delta 0.0405 on a flat plate,
        # which is a 1% shift. The filament itself runs to several hundred.
        ("hair", {"hair_length": 200.0}, 20),
    ):
        worst = None
        for sd in (1234, 77, 909, 42, 5, 31):
            for n in (1, 2, 3, 5):
                got = len(components(
                    mark_mask({key: float(n), "texture_seed": float(sd), **extra}),
                    min_px,
                ))
                if got != n and worst is None:
                    worst = f"seed {sd} asked {n} drew {got}"
        check(
            f"{key} count 1-5 is exact over six seeds",
            worst is None, worst or "every frame drew exactly what it was asked for",
        )

    # At high counts marks genuinely merge, which is the one honest source of
    # error -- two specks that overlap are one blob and there is nothing to fix
    # about that. It is a few percent, not the factor of 1.5 the fitted constant
    # used to cost.
    for n in (20, 120, 400):
        got = len(components(mark_mask({"dust": float(n), "dust_size": 8.0}), 3))
        check(
            f"dust count {n} lands within 3%", abs(got - n) <= max(1, 0.03 * n),
            f"drew {got} of {n} ({(got - n) / n * 100:+.1f}%, merges only)",
        )

    # -- 5g-3. a speck is a round shape, and not a circle --------------------
    # Both halves were asked for: "dots need to be in round form, some dot I
    # found is not round -- don't make them circle, a shape form of imperfect
    # circle or imperfect ellipse".
    #
    # Roundness is measured as the isoperimetric quotient 4*pi*A/P^2, which is
    # 1.0 for a circle and falls away for anything ragged or elongated -- and it
    # is measured *against a rendered disc of the same size* rather than against
    # 1.0, because a rasterised outline over-counts its own perimeter by a
    # factor that depends on the radius. The old thresholded-noise specks score
    # far below this because their outlines are whatever the field did.
    #
    # The other half is asserted from the second moments: a population whose
    # mean axis ratio is 1.0 is a population of circles, which is the thing that
    # was explicitly not wanted.
    def speck_stats(over: dict) -> tuple[float, float, float]:
        m = mark_mask({"dust": 60.0, "dust_soften": 0.0, **over}, thr=0.05)
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ys, xs = np.nonzero(m)
        pts = list(zip(ys.tolist(), xs.tolist()))
        for q in pts:
            parent[q] = q
        for y, x in pts:
            for dy, dx in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
                nb = (y + dy, x + dx)
                if nb in parent:
                    ra, rb = find(nb), find((y, x))
                    if ra != rb:
                        parent[rb] = ra
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for q in pts:
            groups.setdefault(find(q), []).append(q)
        iso, ratio = [], []
        for px in groups.values():
            if len(px) < 60:
                continue
            s = set(px)
            area = float(len(px))
            per = float(sum(
                1 for (y, x) in px
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (y + dy, x + dx) not in s
            ))
            iso.append(4.0 * math.pi * area / max(per * per, 1e-9))
            a = np.array(px, np.float64)
            a -= a.mean(0)
            ev = np.linalg.eigvalsh(a.T @ a / len(px))
            ratio.append(float(math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))))
        return float(np.mean(iso)), float(np.mean(ratio)), float(np.std(ratio))

    # A rendered disc of the same nominal size, as the yardstick. Zero
    # eccentricity and no harmonics is exactly a circle, so this is the score a
    # perfect speck of this radius gets through the same rasteriser and the same
    # component finder.
    # Both constants now live on the rasteriser: since 2026-08-08 the site list
    # records the raw 0..1 eccentricity draw and `_film_texture` scales it by a
    # ceiling that `dust_irregular` slides between `_DUST_ECCENT_LO` and `_HI`.
    # Flattening the ceiling to zero is what makes a perfect disc here.
    import server.engine.stages.film_texture as _raster_mod
    _keep = (_raster_mod._DUST_ECCENT_LO, _raster_mod._DUST_ECCENT_HI,
             _raster_mod._DUST_HARMONICS)
    try:
        _raster_mod._DUST_ECCENT_LO = 0.0
        _raster_mod._DUST_ECCENT_HI = 0.0
        _raster_mod._DUST_HARMONICS = (0.0, 0.0, 0.0)
        disc_iso, disc_ratio, _ = speck_stats({"dust_size": 24.0})
    finally:
        (_raster_mod._DUST_ECCENT_LO, _raster_mod._DUST_ECCENT_HI,
         _raster_mod._DUST_HARMONICS) = _keep
    # **The two are measured at opposite ends of `dust_irregular` since
    # 2026-08-08**, because the slider now controls how oval a speck may get and
    # not merely how dented its outline is. At 0 the population is 91% round by
    # axis ratio, which is what "round" has to mean; at 1 it reaches 29%, which
    # is what "not a circle" has to mean. Measuring both at one setting is how
    # the old fixed 0.35 eccentricity hid -- a third of the specks were obvious
    # ellipses at *every* setting and both assertions still passed.
    iso, _, _ = speck_stats({"dust_size": 24.0})
    check(
        "a speck is round at irregularity 0", iso > disc_iso * 0.82,
        f"isoperimetric {iso:.3f} against a rendered disc's {disc_iso:.3f} "
        f"({iso / disc_iso * 100:.0f}% of a circle)",
    )
    rough_iso, ratio, spread = speck_stats(
        {"dust_size": 24.0, "dust_irregular": 1.0}
    )
    check(
        "a speck is not a circle at irregularity 1",
        ratio > 1.10 and spread > 0.05,
        f"mean axis ratio {ratio:.2f} (a disc renders {disc_ratio:.2f}), "
        f"spread {spread:.2f}",
    )

    # -- 5g-3b. `dust_irregular` --------------------------------------------
    # Reported as "the specks are not rounded". The harmonics used to be
    # unconditional, so no setting drew a clean speck at all -- the shape was
    # never a control. Two assertions, and the first is the one that matters:
    # at 0 the outline is *exactly* the ellipse, not nearly it. A "roundness"
    # slider whose zero still dents the outline is the same bug reported again.
    def dust_frame(over: dict) -> np.ndarray:
        return eng.render_image(
            plain,
            P.sanitize({**tex_off, "dust": 60.0, "dust_size": 24.0,
                        "dust_soften": 0.0, **over}),
            1.0, supersample=2,
        )

    try:
        _raster_mod._DUST_HARMONICS = (0.0, 0.0, 0.0)
        flat = dust_frame({})
    finally:
        _raster_mod._DUST_HARMONICS = _keep[2]
    d0 = float(np.abs(dust_frame({"dust_irregular": 0.0}) - flat).max())
    check(
        "irregularity 0 is exactly an ellipse", d0 == 0.0,
        f"max delta {d0:.2e} against a build with the harmonics zeroed out",
    )
    check(
        "irregularity 1 dents the outline", rough_iso < iso * 0.9,
        f"isoperimetric {iso:.3f} at 0 -> {rough_iso:.3f} at 1 "
        f"({(1 - rough_iso / iso) * 100:.0f}% rougher)",
    )

    # -- 5g-3c. Dust Softness past 1 ----------------------------------------
    # The slider stopped at 1 and the report was that it was too weak; the stop
    # was the reason, and it stopped in three places at once -- the per-speck
    # clamp, the 0.9 on the edge fraction, and the range of the slider itself.
    # So the check is that the added travel is *real* on both counts: the edge
    # keeps getting softer step by step, and the specks are still there at the
    # top rather than faded away by an opacity term that, extended linearly,
    # would have crossed zero at 2.2.
    softs = [
        slope_and_spread({"dust": 200.0, "dust_soften": s})[0]
        for s in (1.0, 2.5, 5.0)
    ]
    check(
        "dust softness keeps softening past 1",
        softs[1] < softs[0] * 0.9 and softs[2] < softs[1] * 0.9,
        "mean edge slope " + " -> ".join(f"{v:.4f}" for v in softs)
        + " at softness 1 / 2.5 / 5",
    )
    cov1 = marks({"dust": 200.0, "dust_size": 8.0, "dust_soften": 1.0})[0]
    cov5 = marks({"dust": 200.0, "dust_size": 8.0, "dust_soften": 5.0})[0]
    check(
        "soft specks spread rather than vanish", cov5 > cov1,
        f"{cov1:.2f}% of frame at softness 1 -> {cov5:.2f}% at 5",
    )

    # -- 5g-4. the dark/light balance ---------------------------------------
    # Reported as "too many dark dots, I want more light dots" -- the split was
    # hard-coded at two thirds dark. It is a slider now, and the split is exact
    # because it is a prefix of the list rather than a per-speck coin flip.
    def populations(bal: float) -> tuple[int, int]:
        o = eng.render_image(
            plain,
            P.sanitize({**tex_off, "dust": 100.0, "dust_size": 8.0,
                        "dust_balance": bal, "dust_lum_var": 0.0}),
            1.0, supersample=2,
        )
        d = o.max(2) - 0.5
        return int(len(components(d < -0.04, 3))), int(len(components(d > 0.04, 3)))

    for bal, want in ((-1.0, "all dark"), (1.0, "all light")):
        dk, lt = populations(bal)
        ok = (lt == 0 and dk > 90) if bal < 0 else (dk == 0 and lt > 90)
        check(f"balance {bal:+.0f} is {want}", ok, f"{dk} dark, {lt} light")
    dk, lt = populations(0.0)
    check(
        "balance 0 is an even mix", abs(dk - lt) <= max(4, 0.08 * (dk + lt)),
        f"{dk} dark, {lt} light",
    )
