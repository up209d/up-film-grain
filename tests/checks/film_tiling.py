"""film texture drawn from absolute coordinates, tiled against a single pass

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from server import params as P
from tests.harness import TEX_OFF, Ctx, check, suite
from server.engine import (
    _leak_anchor, _leak_sites,
)


@suite("film_tiling", "film texture drawn from absolute coordinates, tiled against a single pass")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    big = cx.big
    tex_off = dict(TEX_OFF)
    plain = cx.plain
    # -- 5g-5. drawn marks reserve no tile overlap --------------------------
    # `pad_for` stopped counting dust and hair when they stopped blurring. That
    # is the claim to be paranoid about: a stage missing from `pad_for` seams a
    # tiled export along exactly its own radius while every preview looks fine.
    # It is safe here only because a drawn mark is clipped to *its own*
    # footprint in absolute frame coordinates rather than to the tile's, so a
    # speck straddling a boundary is drawn identically by both tiles.
    heavy = P.sanitize({
        **tex_off, "dust": 300.0, "dust_size": 40.0, "dust_soften": 5.0,
        "dust_irregular": 1.0,
        "hair": 20.0, "hair_length": 400.0, "hair_soften": 1.0,
    })
    whole = eng.render_image(plain, heavy, 1.0, tile=4096, supersample=1)
    split = eng.render_image(plain, heavy, 1.0, tile=128, supersample=1)
    d_ = float(np.abs(whole - split).max())
    check(
        "no seam from drawn marks at zero overlap", d_ < 1e-6,
        f"max delta {d_:.2e} over 300 specks and 20 hairs, tiled at 128px, "
        "at the widest softness and irregularity either mark can reach",
    )

    # Light leaks: the failure this guards is every leak coming out identical,
    # which reads as stamped rather than accidental. Two things had to be
    # right -- the wash field must be fine enough that a frame has several
    # leaks to differ from each other, and the variation fields must be
    # stretched, because raw value noise clusters so tightly around its median
    # that a 9x available range still produced near-identical leaks.
    # Bigger plate than the rest of this section uses, deliberately. The leak
    # field's cells are a fixed size in pixels, so a small frame holds only a
    # handful of leaks and several of those are truncated by its corners --
    # measured, the statistic then ranks the two settings *backwards*. At
    # 2000x1400 the result is stable across backgrounds (1-3% at variation 0,
    # 11-20% at 1); at 1400x900 it is noise.
    leak_plate = np.full((1400, 2000, 3), 0.5, np.float32)

    # Three leaks over many seeds rather than eight over a few. A run of lit
    # border is only one leak's peak if no other leak overlaps it, and eight
    # beams on one frame merge into each other -- measured, the same probe
    # reports a spread ratio of 1.19 at eight leaks and 2.07 at three, because
    # at eight it is mostly measuring the brightest member of each merged pair.
    def leak_peaks(var: float) -> np.ndarray:
        got = []
        for sd in (11, 77, 404, 909, 1234, 5678, 31, 42):
            o = eng.render_image(
                leak_plate,
                # Low strength, or the measurement is taken on a clipped
                # signal: at the shipped strength every leak drives its
                # brightest channel to 1.0 on this plate, so all of them peak
                # at exactly the same number and the spread reads as zero
                # whatever the variation is set to.
                P.sanitize({**tex_off, "light_leak": 3.0, "leak_strength": 0.15,
                            "leak_variation": var, "texture_seed": sd}),
                1.0, supersample=1,
            )
            d = (o - leak_plate).max(2)
            # All four borders, not just the top and the bottom (2026-08-16).
            # Where a leak lands answers to the seed now; it used to come off a
            # constant, which put a lone leak on the right-hand border of every
            # frame and made "read the top and bottom rows" a sample of one
            # border's worth of the population dressed up as a sample of the
            # frame. Reading all four is both more of the population and an
            # unbiased slice of it.
            for strip, perp in ((d[0, :], d), (d[-1, :], d[::-1]),
                                (d[:, 0], d.T), (d[:, -1], d.T[::-1])):
                on = strip > 0.02
                k = 0
                while k < len(on):
                    if on[k]:
                        j = k
                        while j < len(on) and on[j]:
                            j += 1
                        if j - k > 12:
                            got.append(float(perp[:, k:j].max()))
                        k = j
                    else:
                        k += 1
        return np.array(got)

    # Where a leak lands has to answer to the seed, and until 2026-08-16 it did
    # not: the run of leaks started at the constant 0.37 of the perimeter with a
    # +-0.05 jitter on top, and 0.37 of the perimeter is on the right-hand
    # border at *every* aspect ratio -- 16:9, 3:2, square and 2:3 all -- with
    # nowhere near enough jitter to leave it. So `light_leak 1` put its leak on
    # the right of the frame for every seed there is, which is what was
    # reported. Counted over the whole seed space rather than eyeballed on one
    # render, because a fixed placement with a small jitter is indistinguishable
    # from a random one until you look at more than one frame.
    borders = {
        f"{int(fw)}x{int(fh)}": len({
            _leak_anchor(_leak_sites(1.0, sd, 0.0)[0]["pos"], fh, fw)[0]
            for sd in range(400)
        })
        for fh, fw in ((1000.0, 1500.0), (1500.0, 1000.0), (1000.0, 1000.0))
    }
    check(
        "a lone leak can land on any of the four borders",
        all(n == 4 for n in borders.values()),
        ", ".join(f"{k}: {n} of 4 over 400 seeds" for k, n in borders.items()),
    )
    # ...and the seeded start must not cost what the golden step buys. Every
    # leak sits at `base + phi*k` off a base drawn from the seed *once for the
    # whole list*, so raising the count still adds leaks rather than rerolling
    # the frame. Drawing the position per leak from that leak's own generator
    # would look identical on a single render and break this.
    check(
        "adding a leak does not move the ones already there",
        [s["pos"] for s in _leak_sites(3, 77, 0.5)]
        == [s["pos"] for s in _leak_sites(9, 77, 0.5)][:3],
        "leaks 0-2 are unmoved when the count goes 3 -> 9",
    )

    # Leak sizes and the feather are lengths in pixels now, so they have to
    # deliver the pixels they claim -- and the two frame axes have to agree,
    # which the old normalised distance did not do: X divided by the width and
    # Y by the height, so on a 3:2 frame the same size came in 1.5x deeper
    # from a side border than from the top.
    lk_plate = np.ascontiguousarray(np.full((1000, 1500, 3), 0.5, np.float32))
    lk_off = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
        "scatter": 0, "edge_soften": 0,
    }

    def lk_run(over: dict) -> np.ndarray:
        return eng.render_image(
            lk_plate, P.sanitize({**lk_off, "light_leak": 12.0, **over}),
            1.0, supersample=1,
        )

    def lone_leaks(over: dict, seeds=range(20, 56)) -> list[tuple[int, np.ndarray]]:
        """Every *unclipped* single leak over a seed sweep, each rotated so its
        own border is row 0 and its length runs along the columns.

        A sweep, and a median taken over it, rather than one leak on one seed.
        Where a leak lands answers to `texture_seed` -- since 2026-08-16 it
        does, at any rate; it used to come off the constant 0.37, which put a
        lone leak on the **right-hand border of every frame at every seed and
        every aspect ratio**, and that is what these probes were unwittingly
        calibrated against. Two of them read a fixed window of the frame and
        assumed a leak would be sitting in it; with the placement free they
        measured whatever tail happened to reach the window instead, and a
        240px leak came back as 126px deep. So the probe finds its leak now
        rather than assuming one, which is what makes the number about the
        geometry rather than about the arrangement.

        Leaks a corner has truncated are dropped: one has no full length to
        measure and no second edge to compare it against.
        """
        got = []
        var = float(over.get("leak_variation", 0.0))
        for sd in seeds:
            border, _ = _leak_anchor(
                _leak_sites(1.0, sd, var)[0]["pos"], 1000.0, 1500.0)
            g = np.abs(lk_run({**over, "light_leak": 1.0,
                               "texture_seed": float(sd)}) - lk_plate).max(2)
            g = {0: g, 1: g[::-1], 2: g.T, 3: g.T[::-1]}[border]
            xs = np.where(g[0] > 0.01)[0]
            if len(xs) and xs.min() > 0 and xs.max() < g.shape[1] - 1:
                got.append((border, g))
        return got

    def lk_depths(size: int, feather: int) -> dict[str, list[int]]:
        """How deep a lone leak of this size comes in, by which axis it is on."""
        out: dict[str, list[int]] = {}
        for border, g in lone_leaks({"leak_size_min": size, "leak_size_max": size,
                                     "leak_feather": feather,
                                     "leak_variation": 0.0}):
            col = g[:, int(g[0].argmax())]
            axis = "top/bottom" if border in (0, 1) else "left/right"
            out.setdefault(axis, []).append(int(np.where(col > 0.01)[0].max()) + 1)
        return out

    def lk_depth(size: int) -> int:
        d = lk_depths(size, max(2, size // 4))
        return int(np.median([v for vs in d.values() for v in vs] or [0]))

    grew = [lk_depth(s) for s in (60, 120, 240, 400)]
    check(
        "leak size is a distance in pixels",
        all(grew[k] < grew[k + 1] for k in range(len(grew) - 1)) and grew[0] < 120,
        "sizes 60/120/240/400px reach " + ", ".join(f"{g}" for g in grew) + "px",
    )
    axes = lk_depths(240, 60)
    hz = [int(np.median(axes[a])) if axes.get(a) else 0
          for a in ("top/bottom", "left/right")]
    check(
        "both frame axes agree",
        min(len(axes.get(a, ())) for a in ("top/bottom", "left/right")) >= 4
        and abs(hz[0] - hz[1]) < max(hz) * 0.45,
        f"240px reaches {hz[1]}px from the side and {hz[0]}px from the top "
        f"on a 3:2 frame",
    )
    # The feather is the distance to *half* strength, which is the whole claim
    # of putting it in pixels rather than on an abstract 0-1.
    #
    # Probed at a low strength on purpose. The falloff is defined on the light
    # the leak *deposits*, and the response that turns that into pixels
    # saturates one dye layer at a time -- which is what gives a hot leak its
    # white core, and which also means the visible half-way point on a blown
    # leak sits deeper than the exposure's does. Measured at full strength the
    # same 150px feather reads as 227px, and neither number is wrong: this
    # check is about the falloff, so it measures where the response is linear.
    #
    # And on *one* leak, walked down its own centre line. Light adds, so a
    # profile drawn through a frame of twelve overlapping beams keeps being
    # propped up by the next leak along and reads the falloff as longer than
    # it is -- measured, a 20px feather came back as 37px that way.
    #
    # The median of a sweep, not the first leak that fits. One leak is one
    # sample of a spread: over the seeds swept here the same 20px feather
    # measures anywhere from 22px to 55px depending on which leak is asked,
    # so taking whichever seed happened to come first was reading the noise.
    # See `lone_leaks`.
    lone = {"leak_size_min": 300, "leak_size_max": 300,
            "leak_variation": 0.0, "leak_strength": 0.1}

    def half_at(f: int) -> int:
        got = []
        for _, g in lone_leaks({**lone, "leak_feather": f}, range(20, 50)):
            col = g[:, int(g[0].argmax())]
            below = np.where(col < col[:4].max() * 0.5)[0]
            if len(below):
                got.append(int(below.min()))
        return int(np.median(got)) if got else -1

    asked = (20, 80, 150, 285)
    got_half = [half_at(f) for f in asked]
    check(
        "leak feather is the distance to half strength",
        all(abs(g - a) < max(12, a * 0.25) for g, a in zip(got_half, asked)),
        "asked " + "/".join(str(a) for a in asked) + "px, measured "
        + "/".join(str(g) for g in got_half) + "px",
    )
    # A leak must still not fog the middle. The reach is capped at half the
    # short side, which is exactly where `edge_d` tops out -- so the falloff
    # reaches zero *at* the centre however large a number is typed in. Without
    # the cap, `1 - edge_d/reach` never gets to zero and the leak leaves a
    # floor over the whole picture.
    fog = max(
        float(np.abs(lk_run({"leak_size_min": s, "leak_size_max": s,
                             "leak_feather": f})[500, 750] - 0.5).max())
        for s in (60, 300, 1200, 3000) for f in (2, 50, 1500)
    )
    check("no leak can fog the frame centre", fog == 0.0, f"worst centre lift {fog:.2e}")
    # The defaults have to be usable on a *full-resolution photograph*, not
    # just on the small plate the rest of this section renders. Sizes are
    # absolute pixels now, so a number tuned against a 1500px test frame is
    # three to six times too small on a 6000px one -- which is exactly how the
    # first pixel defaults shipped as a few thin lines hugging the border.
    big_leak = np.ascontiguousarray(np.full((1400, 2100, 3), 0.5, np.float32))
    dflt = eng.render_image(
        big_leak, P.sanitize({**lk_off, "light_leak": 6.0}), 1.0, supersample=1
    )
    dl = np.abs(dflt - big_leak).max(2)
    on = np.where(dl[466:934, :1050].max(0) > 0.01)[0]
    deep = (int(on.max()) + 1) if len(on) else 0
    check(
        "the defaults reach into a full-size frame",
        deep > 1050 * 0.15 and float((dl > 0.01).mean()) > 0.02,
        f"deepest {deep}px of a 1050px half-width, "
        f"{float((dl > 0.01).mean()) * 100:.1f}% coverage",
    )
    # And a leak is a *length* now, so the proxy has to predict the export --
    # the same scale invariance every other spatial quantity owes.
    half = np.ascontiguousarray(iio.downscale(big_leak, 0.5))
    small_leak = eng.render_image(
        half, P.sanitize({**lk_off, "light_leak": 6.0}), 0.5, supersample=1
    )
    cov_full = float((dl > 0.01).mean())
    cov_half = float((np.abs(small_leak - half).max(2) > 0.01).mean())
    check(
        "leaks hold their size at proxy scale",
        abs(cov_full - cov_half) < 0.02,
        f"coverage {cov_full * 100:.1f}% at 1:1 vs {cov_half * 100:.1f}% at half scale",
    )
    # Min and max given the wrong way round must swap rather than collapse.
    a_ = lk_run({"leak_size_min": 80, "leak_size_max": 320, "leak_feather": 40})
    b_ = lk_run({"leak_size_min": 320, "leak_size_max": 80, "leak_feather": 40})
    check(
        "the two sizes swap if crossed", float(np.abs(a_ - b_).max()) == 0.0,
        "min 80/max 320 renders identically to min 320/max 80",
    )

    flat_leaks, varied = leak_peaks(0.0), leak_peaks(1.0)
    # Coefficient of variation, not brightest-over-dimmest: with a dozen leaks
    # sampled, min/max turns on whichever corner sliver happened to clip the
    # frame edge, and it ranked the two settings backwards on this scene.
    cv0 = float(flat_leaks.std() / max(flat_leaks.mean(), 1e-9))
    cv1 = float(varied.std() / max(varied.mean(), 1e-9))
    check("several leaks per frame", len(varied) >= 8, f"{len(varied)} leaks sampled")
    # The count has to be a count. Under the old perimeter wash it was not:
    # the gate thresholds were quantiles of a noise field, so asking for two
    # leaks still washed most of the border and the control only really moved
    # how ragged that wash was. Beams are placed one per leak now, so coverage
    # tracks the number asked for.
    cov_n = [float((np.abs(lk_run({"light_leak": float(n),
                                   "leak_size_min": 200, "leak_size_max": 200,
                                   "leak_feather": 60}) - lk_plate).max(2)
                    > 0.01).mean())
             for n in (1, 2, 4, 8)]
    check(
        "the leak count controls how many there are",
        all(cov_n[k] < cov_n[k + 1] for k in range(3)) and cov_n[0] < cov_n[3] * 0.4,
        "1/2/4/8 leaks cover " + ", ".join(f"{c * 100:.1f}%" for c in cov_n),
    )

    # The two things the old perimeter wash could not do, and the reason this
    # stage was rewritten: a leak has to run *along* its border rather than
    # radiate inward from all of them, and it has to have a definite edge.
    # Both are measured on one leak at a time, rotated so its own border is
    # row 0, and only on leaks a corner has not truncated -- a clipped leak
    # has no length to measure and no second edge to compare. `lone_leaks` is
    # what supplies them, so the sweep is over leaks on all four borders rather
    # than over the one border the old fixed placement could produce.
    aspect, hardness = [], []
    for _, g in lone_leaks({"leak_strength": 0.2, "leak_size_min": 240,
                            "leak_size_max": 240, "leak_feather": 120,
                            "leak_variation": 1.0}):
        m = g > 0.01
        xs = np.where(m[0])[0]
        a_, b_ = int(xs.min()), int(xs.max())
        aspect.append((b_ - a_ + 1) / max(int(m[:, a_:b_ + 1].max(1).sum()), 1))
        # Steepest step just inside each end of the run at the border. One end
        # is the obstruction's shadow and one is the penumbra, and a leak soft
        # on both sides reads as haze rather than as light getting past
        # something.
        gl = float(np.abs(np.diff(g[0, max(a_ - 8, 0):a_ + 40])).max())
        gr = float(np.abs(np.diff(g[0, max(b_ - 40, 0):b_ + 8])).max())
        hardness.append(max(gl, gr) / max(min(gl, gr), 1e-9))
    asp, hard = float(np.median(aspect)), float(np.median(hardness))
    check(
        "a leak runs along its border, not inward from every border",
        len(aspect) >= 8 and asp > 1.15,
        f"median length/depth {asp:.2f} over {len(aspect)} leaks",
    )
    check(
        "a leak has one hard edge and one soft one",
        hard > 2.0, f"median steepest-edge ratio {hard:.2f} between a leak's two sides",
    )
    check(
        "variation spreads the leaks apart", cv1 > cv0 * 1.4,
        f"strength spread {cv0 * 100:.0f}% at 0 -> {cv1 * 100:.0f}% at 1",
    )

    tex_all = P.sanitize({
        "dust": 150.0, "scratches": 25.0, "hair": 14.0, "light_leak": 8.0,
        "dust_soften": 0.6, "scratch_soften": 0.6, "hair_soften": 0.6,
    })
    a = eng.render_image(big, tex_all, 1.0, tile=4096, supersample=2)
    b = eng.render_image(big, tex_all, 1.0, tile=128, supersample=2)
    diff = np.abs(a - b)
    # Structural agreement, not a max. Counted marks are selected by a
    # threshold on a noise field, and at sparse counts that threshold sits deep
    # in the field's tail -- so a couple of pixels per frame whose noise value
    # is within float epsilon of it resolve differently between a tiled and a
    # single-pass render, and one flipped speck is a 0.02 delta on its own.
    # Measured: 2 pixels out of 1.4M, sitting ~53px from any tile boundary,
    # mean difference ~3e-07. That is not a seam, and asserting the max here
    # would be asserting float determinism at a knife edge.
    m = float(diff.mean())
    check("no seam from film texture", m < 1e-5, f"mean delta {m:.2e}")
