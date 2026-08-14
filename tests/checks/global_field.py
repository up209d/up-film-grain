"""the global grain point field

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
import torch
from server import params as P
from tests.harness import gridiness
from server.engine import (
    _GRAIN_CLUSTER_REF, _grain_gain, _grain_points,
)
from tests.harness import Ctx, check, suite


@suite("global_field", "the global grain point field")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 5b-i-b. the global grain point field ---------------------------------
    # One construction at every setting since 2026-08-05 (`_grain_points`), so
    # Min and Max are the two ends of one size distribution and nothing else.
    # What is pinned here is the three defects the rewrite exists to fix -- a
    # visible axis-aligned grid, an evenly-spaced mesh with no structure above
    # the clump, and an amplitude that depended on where the sliders happened to
    # sit -- plus the properties the old construction had that must survive.
    print("\nglobal grain point field")
    gv_grey = np.full((400, 560, 3), 0.5, dtype=np.float32)

    def gv_render(lo: float, hi: float, smooth: float = 0.0,
                  im: np.ndarray = gv_grey, tile: int = 1024) -> np.ndarray:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 40.0, "global_size": lo,
                     "global_size_max": hi, "global_opacity": 1.0,
                     "global_smooth": smooth})
        return eng.render_image(im, P.sanitize(over), 1.0, tile=tile,
                                supersample=1).astype(np.float64) - 0.5

    # Max is clamped *up* to Min, never swapped with it -- the two are not a
    # symmetric pair, because Min has a meaning on its own and Max is only "how
    # much further can it stretch". So a Max below Min must render exactly as
    # Max == Min...
    a = gv_render(4.0, 4.0)
    b = gv_render(4.0, 3.5)
    check("max below min renders as max == min",
          float(np.abs(a - b).max()) == 0.0, "max delta 0.00e+00 required")
    # ...and raising Min alone, with Max left at its own untouched default, must
    # not quietly become a Min-to-default range because the default happens to
    # be the smaller number.
    c = eng.render_image(gv_grey, P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO}
                         | {"global_intensity": 40.0, "global_size": 9.0,
                            "global_opacity": 1.0}), 1.0, tile=1024, supersample=1)
    d = eng.render_image(gv_grey, P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO}
                         | {"global_intensity": 40.0, "global_size": 9.0,
                            "global_size_max": 1.6, "global_opacity": 1.0}),
                         1.0, tile=1024, supersample=1)
    check("raising min alone leaves max pinned to it",
          float(np.abs(c.astype(float) - d.astype(float)).max()) == 0.0,
          "max delta 0.00e+00 required")

    # A range has to actually be a range -- clumps of differing size, not a
    # cosmetic nudge on one size.
    var = gv_render(4.0, 14.0)
    check("a min-max range visibly differs from a single size",
          float(np.abs(a - var).max()) > 0.05,
          f"max delta {float(np.abs(a - var).max()):.3f}")

    # -- the grid. This is the complaint the rewrite exists for --------------
    # `gridiness` (defined with the smoothing checks below, and the metric this
    # codebase already used for the quilt) bins the field's |gradient| by phase
    # within a cell. A lattice-addressed field swings a long way between phases,
    # because its extrema sit *on* the lattice and the gradient vanishes there;
    # a field that does not care where the cell boundaries fall does not swing.
    #
    # The value-noise fBm this replaced -- what every Max == Min setting used to
    # render, including three shipped presets at a 5px clump -- scores **1.41 to
    # 1.49** on it, and that number is its visible quilt. The bar here is 0.35,
    # four times under the worst reading below and thirty times under the field
    # it replaced. Checked at Max == Min especially, since that is the setting
    # that used to take the value-noise path.
    for lo, hi in ((1.0, 3.0), (8.0, 8.0), (4.0, 8.0), (12.0, 12.0),
                   (20.0, 20.0)):
        f = _grain_points(768, 768, 0, 0, lo, hi, 4242,
                          torch.device("cpu"), 1)[0, 0].numpy().astype(np.float64)
        q = gridiness(f, hi)
        check(f"no lattice grid at min={lo:g} max={hi:g}", q < 0.35,
              f"phase-binned gridiness {q:.3f} (the value-noise field it "
              "replaced scores 1.4-1.5)")

    # -- structure above the clump -------------------------------------------
    # A point process with uniform density averages to a featureless screen as
    # you step back, and a featureless screen at a distance is precisely what
    # "repetitive pattern when zooming out" describes. `_grain_cluster` is what
    # supplies variation at scales far above one clump.
    #
    # Measured as the spread of *local contrast* -- the coefficient of variation
    # of block standard deviations, at 16 clumps to a block -- and not as the
    # spread of local means, which is the obvious metric and is blind to this.
    # Clustering scales each grain's brightness *magnitude*, and brightness is
    # signed, so it leaves every local mean alone and moves only how grainy one
    # region is against another. A first version of this check measured block
    # means, read a flat 0.99x, and was measuring nothing.
    #
    # A plain argument since 2026-08-13, where the depth became `global_mottle`.
    # It used to have to patch `noise.grain`'s own binding of the constant --
    # rebinding it on `server.engine` changed nothing the generator could see --
    # and that whole hazard is gone with the constant.
    def gv_contrast_cv(cluster: float) -> float:
        f = _grain_points(1024, 1024, 0, 0, 2.0, 4.0, 77,
                          torch.device("cpu"), 1, cluster)[0, 0].numpy()
        f = f.astype(np.float64) - 0.5
        blk = 64                                   # 16 clumps at max size 4
        b = f.reshape(1024 // blk, blk, 1024 // blk, blk).std(axis=(1, 3))
        return float(b.std() / b.mean())

    # Held against `_GRAIN_CLUSTER_REF` explicitly, not against the parameter's
    # default: the default is 0 now, and reading it here would make this check
    # assert that no clustering produces clustering.
    flat, clustered = gv_contrast_cv(0.0), gv_contrast_cv(_GRAIN_CLUSTER_REF)
    check("clustering gives the layer structure above the clump",
          clustered > flat * 3.0 and clustered > 0.10,
          f"local-contrast spread {clustered:.4f} clustered vs {flat:.4f} "
          f"unclustered ({clustered / flat:.1f}x)")

    # -- Global Mottling, the slider on that depth (2026-08-13) ---------------
    # The check above says clustering *can* give the layer structure. These say
    # the new control governs how much, in both directions, and that it does
    # nothing else -- reported against a blank white plate, where 0.6 read as
    # "not organic, not digital", with grain amplitude varying 11.8% across the
    # frame and 4.4% of that surviving a blur to 33px.
    #
    # Measured on the **amplitude envelope at a scale far above one clump**,
    # which is the metric the complaint is actually about. Local sd over 8px
    # blocks is the envelope; averaging those in 4x4 groups throws away the
    # per-block estimator noise and leaves only mottle 32px and coarser. The
    # obvious metric -- the field's own sigma -- is blind to this by
    # construction, since clustering scales a *signed* brightness and so leaves
    # every mean and very nearly every global statistic alone. That is the same
    # trap the block-means version of the check above fell into.
    def gv_mottle_cv(cluster: float, blk: int = 8, grp: int = 4) -> float:
        f = _grain_points(1024, 1024, 0, 0, 0.8, 0.8, 8460,
                          torch.device("cpu"), 1, cluster)[0, 0].numpy()
        f = f.astype(np.float64) - 0.5
        n = 1024 // blk
        env = f.reshape(n, blk, n, blk).std(axis=(1, 3))          # the envelope
        m = n // grp
        env = env.reshape(m, grp, m, grp).mean(axis=(1, 3))       # >= 32px only
        return float(env.std() / env.mean())

    # The white-noise control: what a *perfectly* even field measures on this
    # estimator. Without it the numbers below are unreadable -- some spread is
    # unavoidable when you estimate a local sd from finitely many samples, and
    # the whole question is whether the field beats that floor or not.
    _rs = np.random.RandomState(0).standard_normal((1024, 1024))
    _n = 1024 // 8
    _e = _rs.reshape(_n, 8, _n, 8).std(axis=(1, 3))
    _e = _e.reshape(_n // 4, 4, _n // 4, 4).mean(axis=(1, 3))
    gv_white = float(_e.std() / _e.mean())

    gv_mot = {d: gv_mottle_cv(d) for d in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)}
    gv_seq = [gv_mot[d] for d in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)]
    check("mottling is monotone in the amplitude envelope",
          all(b > a for a, b in zip(gv_seq, gv_seq[1:])),
          "envelope spread at >=32px: "
          + ", ".join(f"{d:g}:{v:.4f}" for d, v in gv_mot.items()))
    # The user-facing promise, and the reason the default is 0: at that setting
    # the layer must be *at least as even as white noise*, not merely evener
    # than it was. It comfortably is -- grains overlap, so neighbouring blocks
    # are correlated where independent samples are not.
    check("mottling at 0 is as even as noise gets",
          gv_mot[0.0] < gv_white,
          f"{gv_mot[0.0]:.4f} against the white-noise floor {gv_white:.4f}")
    # ...and the far end has to be worth having, or the slider is one-sided.
    check("mottling at the pre-slider depth still mottles",
          gv_mot[_GRAIN_CLUSTER_REF] > gv_mot[0.0] * 3.0,
          f"{gv_mot[_GRAIN_CLUSTER_REF]:.4f} at {_GRAIN_CLUSTER_REF:g} against "
          f"{gv_mot[0.0]:.4f} at 0 ({gv_mot[_GRAIN_CLUSTER_REF]/gv_mot[0.0]:.1f}x)")

    # Mottling must move texture and *not* loudness, the property Global Chroma
    # Grain is built around and for the same reason: a slider that moves two
    # things at once cannot be dialled in, because you cannot tell which one you
    # are hearing. Uncorrected the field runs 6.0% louder from 0 to 1, which is
    # exactly the size of effect that reads as "this made the grain stronger".
    gv_amp = {}
    for d in (0.0, 0.3, 0.6, 1.0):
        f = _grain_points(1024, 1024, 0, 0, 1.6, 1.6, 4242,
                          torch.device("cpu"), 1, d)
        gv_amp[d] = float((f - 0.5).std())
    lo_m, hi_m = min(gv_amp.values()), max(gv_amp.values())
    check("mottling does not change the layer's loudness",
          (hi_m - lo_m) / (0.5 * (hi_m + lo_m)) < 0.01,
          f"spread {(hi_m - lo_m) / (0.5 * (hi_m + lo_m)) * 100:.2f}% over "
          + ", ".join(f"{d:g}:{v:.4f}" for d, v in gv_amp.items()))

    # The anchor. `_GRAIN_STD_FIT` was fitted with the depth pinned at
    # `_GRAIN_CLUSTER_REF`, so the correction has to be exactly 1 there -- that
    # is what makes "put 0.6 back to get the old look" a true statement rather
    # than an approximate one, and it is the only reason the constant still has
    # a name now that it is not a default.
    for lo, hi in ((1.0, 3.0), (2.0, 2.0), (1.0, 20.0)):
        check(f"the pre-slider depth is the gain's anchor (min={lo:g} max={hi:g})",
              _grain_gain(lo, hi, _GRAIN_CLUSTER_REF) == _grain_gain(lo, hi),
              "the correction must be identically 1 at "
              f"{_GRAIN_CLUSTER_REF:g}, not merely close")

    # -- grains are full density, never randomly faded -----------------------
    # A grain's brightness draw decides its *direction* and nothing else (see
    # `bri` in `_grain_points`). The draw used to be uniform on [-1, 1), which
    # handed every grain its own random opacity on top of its random size, so
    # 55% of all grains rendered at under half strength -- a veil of near-nothing
    # particles that the amplitude normaliser then had to amplify, pushing the
    # full-strength ones into the clamp to pay for the faint ones.
    #
    # Measured from the field rather than from the draw, because the draw is not
    # the property -- what matters is that a grain *arrives* at full density.
    # With clustering forced off, every present grain has magnitude exactly 1, so
    # `|2(f - 0.5) / gain|` must reach 1 at any pixel a single grain dominates,
    # and the population up near 1 must be a real population rather than a tail.
    # Both halves are needed: the maximum alone would pass on a construction
    # where one lucky grain in the frame drew a strength near 1, and the
    # population alone would pass on one that fades grains only slightly.
    #
    # The old uniform draw fails both by a wide margin -- it peaks at 0.994 and
    # puts 0.12-0.21% above 0.9, against 6.3-6.9% here, a 30-58x separation --
    # so this is what stops the opacity draw being reintroduced as a tidy-up.
    for lo, hi in ((2.0, 4.0), (1.0, 3.0), (6.0, 6.0)):
        f = _grain_points(1024, 1024, 0, 0, lo, hi, 4242,
                          torch.device("cpu"), 1, 0.0)[0, 0].numpy()
        a = np.abs(2.0 * (f.astype(np.float64) - 0.5) / _grain_gain(lo, hi, 0.0))
        top, mx = float((a > 0.9).mean()), float(a.max())
        check(f"grains are full density at min={lo:g} max={hi:g}",
              mx > 0.999 and top > 0.03,
              f"peak {mx:.4f} (want 1.0), {top:.2%} of the field above 0.9 "
              "(the uniform-opacity draw gave 0.994 and 0.2%)")

    # Coverage gaps are the honest consequence of a wide min-max range, not a
    # bug -- but the frame's mean must stay unbiased regardless of how wide the
    # gaps get. A first version failed this: gaps read as the raw field's zero
    # rather than its neutral 0.5, which after the shared *2-1 remap put every
    # gap at the fully negative rail -- a frame-wide dark cast, not "nothing
    # here". Swept from a narrow to a very wide range.
    for hi in (4.0, 10.0, 18.0):
        v = gv_render(1.0, hi)
        check(f"gaps stay neutral at max={hi:.0f}", abs(v.mean()) < 0.01,
              f"frame mean {v.mean():+.4f}")

    # The resonance -- "sometimes it does a good job, sometimes it does not,
    # even with the same config". When the working cell lands on a whole number
    # of pixels an axis-aligned lattice phase-locks with the pixel grid: every
    # pixel sits at the same fractional offset inside its own cell, so none is
    # ever near a point and the field cannot reach its own amplitude. Measured
    # on the old construction, cell 1.00 scored 65% of what 1.05 and 0.95 did.
    #
    # The rotation is what fixes it, and the fix is structural rather than a
    # tuned warp: an irrational slope leaves the two grids incommensurate at
    # *every* size, so this is swept across whole numbers and their neighbours
    # rather than spot-checked at one. Tested against the construction directly,
    # so the check pins the mechanism and not a render that happens to look ok.
    def gv_std(cell: float) -> float:
        f = _grain_points(600, 600, 0, 0, cell, cell, 555,
                          torch.device("cpu"), 1)
        return float((f - 0.5).std())

    sizes = (0.95, 1.0, 1.05, 1.6, 1.95, 2.0, 2.05, 3.0, 4.0, 5.0)
    stds = {c: gv_std(c) for c in sizes}
    lo_s, hi_s = min(stds.values()), max(stds.values())
    check("no cell size is a dead zone",
          (hi_s - lo_s) / (0.5 * (hi_s + lo_s)) < 0.08,
          "std spread "
          f"{(hi_s - lo_s) / (0.5 * (hi_s + lo_s)) * 100:.1f}% over "
          + ", ".join(f"{c:g}:{stds[c]:.3f}" for c in sizes))

    # Amplitude must not depend on where the size sliders sit. It used to,
    # badly: the two old constructions disagreed by 43%, so `global_intensity`
    # meant one thing below Max and another above it. `_grain_gain` normalises
    # it with a closed form in the Min/Max ratio -- closed form because a
    # measured `std()` would be a statistic of the region (invariant 1) and
    # would normalise every tile of an export differently.
    amps = {}
    for lo, hi in ((0.8, 0.8), (1.0, 3.0), (2.0, 2.0), (4.0, 8.0),
                   (1.0, 20.0), (10.0, 20.0), (20.0, 20.0)):
        f = _grain_points(1024, 1024, 0, 0, lo, hi, 4242, torch.device("cpu"), 1)
        amps[(lo, hi)] = float((f - 0.5).std())
    lo_a, hi_a = min(amps.values()), max(amps.values())
    check("amplitude is flat across the whole size range",
          (hi_a - lo_a) / (0.5 * (hi_a + lo_a)) < 0.12,
          f"spread {(hi_a - lo_a) / (0.5 * (hi_a + lo_a)) * 100:.1f}% over "
          + ", ".join(f"{a:g}-{b:g}:{v:.3f}" for (a, b), v in amps.items()))

    # Tile independence. `pad_for` reserves **nothing** for this field -- it
    # derives its own lattice window from whichever window it is handed, with a
    # ring of slack on every side, so a pixel always sees its true neighbouring
    # cells however the frame was split. That is a claim worth testing directly
    # rather than trusting: with the padding term gone there is no slack left
    # over to hide a reach nobody accounted for.
    gv_pad = _grain_points(384, 384, 0, 0, 1.0, 4.0, 31, torch.device("cpu"), 3)
    gv_worst = 0.0
    for oy, ox, th, tw in ((0, 0, 97, 61), (137, 89, 103, 151),
                           (300, 217, 84, 84), (41, 300, 61, 79)):
        sub = _grain_points(th, tw, oy, ox, 1.0, 4.0, 31,
                            torch.device("cpu"), 3)
        gv_worst = max(gv_worst, float(
            (sub - gv_pad[:, :, oy:oy + th, ox:ox + tw]).abs().max()))
    check("the field itself needs no tile overlap", gv_worst < 2e-3,
          f"max delta {gv_worst:.2e} over 4 sub-windows at zero padding")
    # ...and `pad_for` says so. Turning the layer on at any size must not widen
    # the overlap by a pixel; only `global_smooth`, which is a real kernel, may.
    gv_base = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO}), 1.0)
    for lo, hi in ((1.6, 1.6), (20.0, 20.0), (1.0, 20.0)):
        n = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO} | {
            "global_intensity": 40.0, "global_size": lo,
            "global_size_max": hi, "global_opacity": 1.0}), 1.0)
        check(f"pad_for reserves nothing for the field (min={lo:g} max={hi:g})",
              n == gv_base, f"{n}px against {gv_base}px with the layer off")
    gv_sm = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO} | {
        "global_intensity": 40.0, "global_size": 20.0, "global_opacity": 1.0,
        "global_smooth": 1.0}), 1.0)
    check("pad_for still reserves for the smoothing kernel", gv_sm > gv_base,
          f"{gv_base}px -> {gv_sm}px at smoothness 1, clump 20px")
    # Mottling is not a kernel either. It is a per-*cell* multiplier evaluated
    # on the lattice by absolute cell index, so it reads nothing outside the
    # window `_grain_points` already derives -- but it ships at 0 now, which
    # means the checks above run straight past it, and a stage nobody pads for
    # seams a tiled export along exactly its reach while every preview looks
    # fine. So it gets its own row rather than an argument.
    gv_mt = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO} | {
        "global_intensity": 40.0, "global_size": 20.0, "global_opacity": 1.0,
        "global_mottle": 1.0}), 1.0)
    check("pad_for reserves nothing for mottling", gv_mt == gv_base,
          f"{gv_mt}px against {gv_base}px with the layer off")

    # And the same thing end to end through the renderer, at the sizes and
    # smoothing settings where a reach bug would actually bite.
    gv_scene = np.ascontiguousarray(
        (np.random.RandomState(9).rand(320, 480, 3) * 0.5 + 0.25).astype(np.float32)
    )
    # The last row carries mottling, for the reason the `pad_for` row above
    # does: it ships at 0, so every other row here renders straight past it, and
    # the cluster field is the one part of this construction that reads a
    # *second* lattice on its own derived window.
    for lo, hi, smooth, mottle, tile in (
        (0.4, 1.0, 0.0, 0.0, 64), (1.0, 2.0, 0.6, 0.0, 64),
        (0.8, 20.0, 1.0, 0.0, 96), (1.0, 3.0, 0.0, 1.0, 64),
    ):
        p = P.sanitize({"global_intensity": 40, "global_size": lo,
                        "global_size_max": hi, "global_opacity": 1.0,
                        "global_smooth": smooth, "global_mottle": mottle})
        wide = eng.render_image(gv_scene, p, 1.0, tile=4096, supersample=1)
        narrow = eng.render_image(gv_scene, p, 1.0, tile=tile, supersample=1)
        d = float(np.abs(wide.astype(float) - narrow.astype(float)).max())
        check(f"tile independence (min={lo}, max={hi}, smooth={smooth}, "
              f"mottle={mottle})", d < 2e-3, f"max delta {d:.2e}")

    # Chroma shares the geometry generator, so a colour variant of a wide
    # size range must still decorrelate the channels.
    over = {k: 0.0 for k in P.NEUTRAL_ZERO}
    over.update({"global_intensity": 40.0, "global_size": 4.0,
                 "global_size_max": 14.0, "global_opacity": 1.0,
                 "global_chroma": 1.0})
    gcv = eng.render_image(gv_grey, P.sanitize(over), 1.0, tile=1024,
                           supersample=1).astype(np.float64) - 0.5
    spread = float(np.abs(gcv[:, :, 0] - gcv[:, :, 1]).max())
    check("chroma decorrelates a wide size range", spread > 0.05,
          f"max channel spread {spread:.3f}")
