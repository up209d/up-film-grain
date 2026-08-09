"""the Colour Grading section, highlight reconstruction and .cube parsing

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import imageio as iio
from server import params as P
from tests.harness import Ctx, check, suite


@suite("grading", "the Colour Grading section, highlight reconstruction and .cube parsing")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 3e. colour grading: the section above pre-blur -----------------------
    # Five stages, and every one of them ships at 0 -- so the checks above,
    # which render at the defaults, walk straight past all of it. Each of these
    # switches one thing on and measures the specific property that would be
    # silently wrong otherwise.
    #
    # The LUT ones are the sharp end. A 3D LUT is read by three coordinates into
    # a volume, and *any* mix-up of those axes -- the storage order, the grid
    # order, the sample alignment -- still produces a plausible-looking graded
    # image. So the tables here are deliberately asymmetric and exactly linear:
    # trilinear interpolation of a linear function is exact, which turns "did
    # the LUT land right" from an eyeball question into an equality.
    print("\ncolour grading (temperature, tone, clarity, 3D LUT)")
    import torch  # noqa: E402

    from server import lut as lutlib  # noqa: E402
    from server.engine import (  # noqa: E402
        _blur, _luma, _linear_to_srgb, _srgb_to_linear, _MID_GREY,
    )

    def cube(fn) -> lutlib.Lut:
        """A LUT built from ``fn(r, g, b) -> (R, G, B)`` over an 8-cube.

        Stored ``[b][g][r]`` because that is what the .cube format's
        red-varies-fastest ordering means after a C-order reshape.
        """
        n = 8
        ax = np.linspace(0.0, 1.0, n, dtype=np.float32)
        bb, gg, rr = np.meshgrid(ax, ax, ax, indexing="ij")
        tab = np.stack(fn(rr, gg, bb), axis=-1).astype(np.float32)
        return lutlib.Lut(id="t", name="t", size=n, table=np.ascontiguousarray(tab))

    ident = cube(lambda r, g, b: (r, g, b))
    # Output red = input blue, green = red, blue = green. Nothing about this
    # survives a transposed axis or a half-cell misalignment.
    rot = cube(lambda r, g, b: (b, r, g))

    def graded(over: dict, im: np.ndarray, lut=None, ss: int = 1) -> np.ndarray:
        p_ = P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, **over})
        p_["lut"] = lut
        return eng.render_image(im, p_, 1.0, tile=4096, supersample=ss)

    # A mid-grey plate with pixel-scale-free structure and three distinct
    # channels, so hue and local contrast are both measurable on it without
    # anything clipping at either rail.
    gy, gx = np.mgrid[0:192, 0:192].astype(np.float32)
    pat = np.sin(gx / 3.0) * np.sin(gy / 3.4)
    plate = np.stack([0.55, 0.45, 0.35], -1)[None, None, :] * np.ones(
        (192, 192, 1), np.float32
    ) * (1.0 + 0.2 * pat[..., None])
    plate = np.ascontiguousarray(np.clip(plate, 0, 1).astype(np.float32))

    o = graded({"lut_amount": 1.0}, plate, ident)
    d = float(np.abs(o - plate).max())
    check(
        "an identity LUT is a pass-through", d < 1e-5,
        f"max delta {d:.2e} (align_corners=False would shift the whole table)",
    )

    o = graded({"lut_amount": 1.0}, plate, rot)
    want = plate[..., [2, 0, 1]]
    d = float(np.abs(o - want).max())
    check(
        "LUT axes are (r, g, b) -> (W, H, D)", d < 1e-5,
        f"a channel-rotating LUT rotates them, max delta {d:.2e}",
    )

    o = graded({"lut_amount": 0.5}, plate, rot)
    half = plate + (want - plate) * 0.5
    d = float(np.abs(o - half).max())
    check("LUT mix is a straight cross-fade", d < 1e-5, f"at 0.5, max delta {d:.2e}")

    o = graded({"lut_amount": 0.0}, plate, rot)
    d = float(np.abs(o - plate).max())
    check("mix 0 is bit-exactly off", d == 0.0, f"max delta {d:.2e}")

    # A selected LUT with a nonzero mix has to defeat the neutral short-circuit,
    # and a zero mix has to leave it intact -- that is what keeps the Original
    # button bit-exact whether or not a LUT happens to be picked.
    nz = P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, "lut_amount": 1.0})
    z = P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO})
    check(
        "the mix is what is_neutral sees",
        not P.is_neutral(nz) and P.is_neutral(z),
        "mix 1 renders, mix 0 short-circuits",
    )

    # Temperature. Warm must move red up and blue down while leaving the *level*
    # alone -- a white balance that also exposes the frame is fighting every
    # tonal control below it.
    warm = graded({"grade_temp": 1.0}, plate)
    cool = graded({"grade_temp": -1.0}, plate)
    lum0 = float((plate * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
    lw = float((warm * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
    lc = float((cool * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
    check(
        "temperature moves red against blue",
        warm[..., 0].mean() > plate[..., 0].mean()
        and warm[..., 2].mean() < plate[..., 2].mean()
        and cool[..., 0].mean() < plate[..., 0].mean()
        and cool[..., 2].mean() > plate[..., 2].mean(),
        f"warm R {plate[..., 0].mean():.3f}->{warm[..., 0].mean():.3f}, "
        f"B {plate[..., 2].mean():.3f}->{warm[..., 2].mean():.3f}",
    )
    check(
        "temperature holds the level",
        abs(lw / lum0 - 1.0) < 0.02 and abs(lc / lum0 - 1.0) < 0.02,
        f"luma {lum0:.4f} -> {lw:.4f} warm, {lc:.4f} cool "
        f"({100 * (lw / lum0 - 1):+.1f}% / {100 * (lc / lum0 - 1):+.1f}%)",
    )

    # Tint. The other white-balance axis, at right angles to temperature:
    # green against magenta. Same two properties, and the same round trip --
    # the check is that its own constant was tuned smaller than temperature's
    # to land in the same level-holding envelope, not that it is identical.
    magenta = graded({"grade_tint": 1.0}, plate)
    green = graded({"grade_tint": -1.0}, plate)
    lm = float((magenta * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
    lg = float((green * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
    check(
        "tint moves green against red and blue",
        magenta[..., 1].mean() < plate[..., 1].mean()
        and magenta[..., 0].mean() > plate[..., 0].mean()
        and magenta[..., 2].mean() > plate[..., 2].mean()
        and green[..., 1].mean() > plate[..., 1].mean()
        and green[..., 0].mean() < plate[..., 0].mean()
        and green[..., 2].mean() < plate[..., 2].mean(),
        f"magenta G {plate[..., 1].mean():.3f}->{magenta[..., 1].mean():.3f}, "
        f"green G {plate[..., 1].mean():.3f}->{green[..., 1].mean():.3f}",
    )
    check(
        "tint holds the level",
        abs(lm / lum0 - 1.0) < 0.02 and abs(lg / lum0 - 1.0) < 0.02,
        f"luma {lum0:.4f} -> {lm:.4f} magenta, {lg:.4f} green "
        f"({100 * (lm / lum0 - 1):+.1f}% / {100 * (lg / lum0 - 1):+.1f}%)",
    )

    # Exposure. A stops multiply in linear light, ahead of every luma-keyed
    # mask below it -- so the check is an exact match against a direct linear
    # 2x, not merely a mean brightness change the sRGB encoding could fake.
    ev1 = graded({"grade_exposure": 1.0}, plate)
    t = torch.from_numpy(plate).permute(2, 0, 1).unsqueeze(0)
    want_ev1 = (
        _linear_to_srgb(_srgb_to_linear(t) * 2.0).clamp(0.0, 1.0)
        .squeeze(0).permute(1, 2, 0).numpy()
    )
    d = float(np.abs(ev1 - want_ev1).max())
    check(
        "exposure is an exact linear-light stop", d < 1e-5,
        f"max delta from a direct 2x linear multiply: {d:.2e}",
    )
    evm2 = graded({"grade_exposure": -2.0}, plate)
    check(
        "exposure -2 is markedly darker",
        float(evm2.mean()) < float(plate.mean()) * 0.6,
        f"mean {float(plate.mean()):.3f} -> {float(evm2.mean()):.3f}",
    )

    # Shadows and highlights. Four properties, and the first is the one the
    # whole rewrite exists for.
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    ramp = np.repeat(np.repeat(ramp[None, :, None], 32, 0), 3, 2)
    ramp = np.ascontiguousarray(ramp)
    lo_i, hi_i = slice(0, 40), slice(216, 256)

    # 1. **Strictly monotone at every setting.** This is what separates
    #    recovering tonal detail from destroying it, and it is not a tolerance
    #    to be relaxed: the previous share-of-headroom construction scaled its
    #    strength by the pixel's own level through a steep quintic, and in the
    #    two *recovering* directions -- Shadows up, Highlights down, the ones
    #    anyone actually reaches for -- that term overwhelmed the identity and
    #    the transfer inverted, measured slope -0.21 over 16% of the range. A
    #    control that reverses tonal order does not pull a highlight back, it
    #    flattens it into the textureless patch it was meant to rescue. So the
    #    check is on the *slope of the transfer*, not on the mean level: a
    #    mean-only test passes happily on a curve that has folded over.
    worst_slope = 1.0
    where = ""
    for k in ("grade_shadows", "grade_highlights"):
        for a in (-1.0, -0.7, -0.4, 0.4, 0.7, 1.0):
            prof = graded({k: a}, ramp)[0, :, 0].astype(np.float64)
            sl = float(np.diff(prof).min() * (len(prof) - 1))
            if sl < worst_slope:
                worst_slope, where = sl, f"{k} {a:+.1f}"
    check(
        "the tone curves are strictly monotone -- no setting flattens or "
        "inverts detail",
        worst_slope > 0.0,
        f"worst transfer slope over 12 settings: {worst_slope:+.3f} at {where} "
        "(the share-of-headroom construction this replaced measured -0.211)",
    )

    # 2. Neither can leave the cube from in-gamut input: the recovering halves
    #    approach their rail asymptotically, the expanding halves are a share of
    #    the headroom that is there.
    worst = 0.0
    for a in (-1.0, -0.5, 0.5, 1.0):
        for k in ("grade_shadows", "grade_highlights"):
            r_ = graded({k: a}, ramp)
            worst = max(worst, float(r_.min() * -1.0), float(r_.max() - 1.0))
    check(
        "tone lifts cannot clip", worst <= 1e-6,
        f"worst excursion outside 0..1 over 8 settings: {worst:.2e}",
    )

    # 3. Each end still keys on its own end. Their supports are disjoint about
    #    the knee now, so the far end must be *bit-exactly* untouched rather
    #    than merely close -- a stronger claim than the shared-luma version
    #    could make.
    s_up = graded({"grade_shadows": 1.0}, ramp)
    h_dn = graded({"grade_highlights": -1.0}, ramp)
    ds_lo = float((s_up[:, lo_i] - ramp[:, lo_i]).mean())
    ds_hi = float(np.abs(s_up[:, hi_i] - ramp[:, hi_i]).max())
    dh_hi = float((h_dn[:, hi_i] - ramp[:, hi_i]).mean())
    dh_lo = float(np.abs(h_dn[:, lo_i] - ramp[:, lo_i]).max())
    check(
        "each end keys on its own end",
        ds_lo > 0.10 and ds_hi < 1e-6 and dh_hi < -0.10 and dh_lo < 1e-6,
        f"shadows +1: {ds_lo:+.3f} low / {ds_hi:.2e} worst high; "
        f"highlights -1: {dh_lo:.2e} worst low / {dh_hi:+.3f} high",
    )

    # 4. Hue and HSV saturation are held *exactly*, not approximately, because
    #    the whole pixel is scaled by one factor taken from its brightest
    #    channel -- a uniform scale cannot move a ratio. The old per-channel
    #    lift keyed on a shared luma could not make this claim.
    rng_t = np.random.default_rng(7)
    cols = np.ascontiguousarray(rng_t.random((48, 48, 3)).astype(np.float32))

    def hsv_sat(a: np.ndarray) -> np.ndarray:
        mx, mn = a.max(-1), a.min(-1)
        return (mx - mn) / np.clip(mx, 1e-4, None)

    worst_hs = 0.0
    for k, a in (("grade_highlights", -1.0), ("grade_shadows", 1.0)):
        worst_hs = max(worst_hs, float(
            np.abs(hsv_sat(graded({k: a}, cols)) - hsv_sat(cols)).max()))
    check(
        "tone recovery holds hue and saturation exactly", worst_hs < 1e-5,
        f"worst HSV-saturation change at full travel: {worst_hs:.2e}",
    )

    # 5. And the point of it all: an exposure that pushed the frame past white
    #    is *recoverable*, because nothing between white balance and the tone
    #    stage clips it away first. Without the recovery the top of the frame is
    #    a flat plateau; with it, the gradient is back.
    up = np.linspace(0.30, 0.92, 400, dtype=np.float32)
    up = np.ascontiguousarray(np.repeat(np.repeat(up[None, :, None], 32, 0), 3, 2))
    ev_flat = graded({"grade_exposure": 1.0}, up)
    ev_rec = graded({"grade_exposure": 1.0, "grade_highlights": -1.0}, up)

    def flat_frac(a: np.ndarray) -> float:
        return float((a >= 0.999).mean())

    check(
        "an over-exposed highlight is recoverable, not merely dimmable",
        flat_frac(ev_flat) > 0.2 and flat_frac(ev_rec) < 1e-6,
        f"+1 stop leaves {flat_frac(ev_flat) * 100:.1f}% of the frame flat at "
        f"white; Highlights -1 takes it to {flat_frac(ev_rec) * 100:.2f}%",
    )

    # Contrast. Unlike Shadows/Highlights above, this one is allowed to clip --
    # the check is that the pivot itself never moves and that the gain floors
    # at 0 rather than crossing into an inversion.
    mg = np.full((8, 8, 3), _MID_GREY, np.float32)
    mg_hi = graded({"grade_contrast": 1.0}, mg)
    mg_lo = graded({"grade_contrast": -1.0}, mg)
    d_piv = max(float(np.abs(mg_hi - _MID_GREY).max()), float(np.abs(mg_lo - _MID_GREY).max()))
    check(
        "contrast pivots exactly at the mid grey it claims",
        d_piv < 1e-4, f"a flat mid-grey field moves by {d_piv:.2e} at +-1",
    )
    c_hi = graded({"grade_contrast": 1.0}, ramp)
    c_lo = graded({"grade_contrast": -1.0}, ramp)
    std0 = float(ramp[:, :, 0].std())
    std_hi, std_lo = float(c_hi[:, :, 0].std()), float(c_lo[:, :, 0].std())
    check(
        "contrast steepens the spread and floors it, never inverts",
        std_hi > std0 * 1.3 and 0.0 < std_lo < std0 * 0.15,
        f"std {std0:.3f} -> {std_hi:.3f} at +1, {std_lo:.3f} at -1 "
        f"(gain floor is 0.1x, measured {std_lo / std0:.3f}x)",
    )

    # Black point. Deliberately the odd one out in this section: it is
    # *supposed* to clip, so the check is that it clips exactly at the chosen
    # level and nowhere else, and holds white untouched.
    lvl = 0.2  # lands exactly on ramp index 51 (51/255 == 0.2)
    bp1 = graded({"grade_black_point": lvl}, ramp)
    below = float(bp1[:, :52].max())
    white = float(bp1[16, -1, 0])
    mono_bp = bool(np.all(np.diff(bp1[16, 52:, 0]) >= -1e-6))
    check(
        "black point clips at and below the chosen level, holds white",
        below < 1e-5 and white > 0.999 and mono_bp,
        f"max at/below {lvl}: {below:.2e}, white holds at {white:.5f}, "
        f"monotonic above it: {mono_bp}",
    )

    # Clarity. The band is measured at the stage's own radius, so the metric
    # has to use the same kernel the stage does or it is measuring something
    # else.
    def band(a: np.ndarray, r: float) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0)
        lm = _luma(t)
        return (lm - _blur(lm, r)).squeeze().numpy()

    CR = 6.0
    b0 = band(plate, CR)
    e0 = float(b0.std())
    ladder = []
    for cl in (-1.0, -0.5, 0.5, 1.0):
        oc = graded({"grade_clarity": cl, "grade_clarity_radius": CR}, plate)
        bc = band(oc, CR)
        ladder.append((cl, float(bc.std()) / e0, float(np.mean(bc * b0) / (e0 * e0))))
    check(
        "clarity is two-way and monotonic",
        all(ladder[i][1] < ladder[i + 1][1] for i in range(len(ladder) - 1))
        and ladder[0][1] < 0.1 and ladder[-1][1] > 1.5,
        ", ".join(f"{c:+.1f} -> {s * 100:.0f}%" for c, s, _ in ladder),
    )
    check(
        "negative clarity flattens, never inverts",
        all(corr > -0.02 for _, _, corr in ladder),
        f"band correlation with the source at -1 is {ladder[0][2]:+.3f} "
        "(a negative number would be reversed halos)",
    )
    # Clarity adds one signed luminance value to all three channels, so the
    # channel *differences* -- which is what hue is -- come through untouched.
    oc = graded({"grade_clarity": 1.0, "grade_clarity_radius": CR}, plate)
    dif0 = np.stack([plate[..., 0] - plate[..., 1], plate[..., 1] - plate[..., 2]], -1)
    dif1 = np.stack([oc[..., 0] - oc[..., 1], oc[..., 1] - oc[..., 2]], -1)
    d = float(np.abs(dif0 - dif1).max())
    check(
        "clarity holds hue exactly", d < 1e-6,
        f"channel differences move by {d:.2e}",
    )

    # Saturation. A flat scale about each pixel's own luma, so chroma -- each
    # channel's offset from that luma -- must scale by *exactly* the gain,
    # which is a stronger and more precise claim than a saturation-ratio test
    # can make (that ratio also moves because the max channel shifts).
    def chroma(a: np.ndarray) -> np.ndarray:
        return a.max(-1) - a.min(-1)

    c0 = chroma(plate)
    s_hi = graded({"grade_saturation": 1.0}, plate)
    s_lo = graded({"grade_saturation": -1.0}, plate)
    d_sat = float(np.abs(chroma(s_hi) - 2.0 * c0).max())
    check(
        "saturation scales chroma exactly by its gain",
        d_sat < 1e-4, f"max deviation from an exact 2x chroma at +1: {d_sat:.2e}",
    )
    check(
        "saturation -1 is exactly monochrome",
        float(chroma(s_lo).max()) < 1e-4,
        f"max chroma remaining: {float(chroma(s_lo).max()):.2e}",
    )

    # Vibrance. The same saturation-weighted-against-itself construction as
    # Tone Response's own vibrance, on its own key -- the check is the same
    # defining property: gain must fall as starting saturation rises.
    import colorsys as _cs2  # noqa: E402
    gsats = [0.15, 0.35, 0.55, 0.75, 0.95]
    gsw = np.zeros((32, 32 * len(gsats), 3), np.float32)
    for k, s_ in enumerate(gsats):
        gsw[:, k * 32:(k + 1) * 32] = _cs2.hsv_to_rgb(0.05, s_, 0.75)
    gsw = np.ascontiguousarray(gsw)

    def gvib_sat_of(v: float) -> list[float]:
        o = graded({"grade_vibrance": v}, gsw)
        got = []
        for k in range(len(gsats)):
            px = o[16, k * 32 + 16]
            mx, mn = float(px.max()), float(px.min())
            got.append((mx - mn) / max(mx, 1e-4))
        return got

    gb0 = gvib_sat_of(0.0)
    gd0 = max(abs(x - s_) for x, s_ in zip(gb0, gsats))
    check("grade_vibrance neutral at 0", gd0 < 5e-3, f"max saturation drift {gd0:.2e}")
    gup = gvib_sat_of(0.8)
    ggains = [a / b - 1.0 for a, b in zip(gup, gb0)]
    check(
        "grade_vibrance weights muted over vivid",
        all(ggains[k] > ggains[k + 1] for k in range(len(ggains) - 1)),
        " > ".join(f"{g * 100:.0f}%" for g in ggains),
    )

    # -- highlight reconstruction ------------------------------------------
    # An 8-bit file clips per *channel*, so a warm highlight is a flat plateau
    # in red while green and blue are still recording the scene's gradient --
    # the detail is in the file, missing from one channel at a time. This
    # rebuilds the flattened channel from the local chromaticity measured
    # wherever it was still measurable, and puts it back *above* white where it
    # really was; Highlights then rolls it into view.
    #
    # The plate is the whole test: a warm ramp (1.00, 0.80, 0.62) running from
    # 0.40 to 1.45, so `truth` is what the scene was and `shot` is what an
    # 8-bit file could hold of it. That gives an exact answer to check against
    # rather than a "looks recovered" judgement.
    print("\ncolour grading: highlight reconstruction")
    rh_h, rh_w, rh_edge = 400, 400, 280
    rt = np.concatenate([
        np.linspace(0.40, 1.00, rh_edge), np.linspace(1.00, 1.45, rh_w - rh_edge),
    ]).astype(np.float32)[None, :]
    truth = np.ascontiguousarray(
        (np.stack([rt * 1.00, rt * 0.80, rt * 0.62], -1)
         * np.ones((rh_h, 1, 1), np.float32)).astype(np.float32))
    shot = np.ascontiguousarray(np.clip(truth, 0.0, 1.0))
    blown = shot[..., 0] >= 0.999

    # Trilinear-exact in spirit: the reconstruction of a channel whose true
    # value is known must *equal* it, not merely move toward it, wherever there
    # is a clean sample of the local colour within reach. Checked at 0/20/40px
    # into the blown zone at a 64px radius, which reaches all of it.
    from server.engine import _recon_estimate  # noqa: E402
    from server.engine import _RECON_ROLL_KNEE as _RECON_ROLL_KNEE_V  # noqa: E402
    ten = torch.from_numpy(shot).permute(2, 0, 1).unsqueeze(0).to(eng.device)
    prof = _recon_estimate(ten, 1.0, 64.0)[0].cpu().numpy()[0, 0].mean(0)
    tprof = truth[..., 0].mean(0)
    errs = [abs(prof[rh_edge + d] - tprof[rh_edge + d]) / tprof[rh_edge + d]
            for d in (0, 20, 40, 60, 90)]
    check(
        "reconstruction recovers the true clipped value, not an approximation",
        max(errs) < 0.01,
        "worst error against the unclipped scene at 0/20/40/60/90px into the "
        "blown zone: " + " / ".join(f"{e * 100:.1f}%" for e in errs),
    )

    # Beyond the radius there is no surviving sample of the channel's own colour
    # anywhere in reach, and the honest answer is to stop rather than
    # extrapolate. Checked as a *degradation*, so a future change that silently
    # started inventing values out there would fail here.
    prof16 = _recon_estimate(ten, 1.0, 16.0)[0].cpu().numpy()[0, 0].mean(0)
    deep = rh_edge + 100
    check(
        "reconstruction fades out past its radius instead of extrapolating",
        abs(prof16[deep] - shot[..., 0].mean(0)[deep]) < 1e-4,
        f"at 100px into the blown zone with a 16px radius it leaves the pixel "
        f"at {prof16[deep]:.4f}, the file's own value",
    )

    # **The slider has to do something on its own.** The first version left the
    # estimate above white and relied on Highlights to bring it into view, so it
    # measured 0.0004 of mean change on a real photograph and was reported as
    # having no effect at all. The stage now finishes with its own locally-gated
    # roll, and this is the regression test for that: reconstruction *alone*, no
    # other control touched, must put real spread back into a channel the file
    # has flat.
    rec_p = {"grade_recover": 1.0, "grade_recover_radius": 32.0}

    def red_span(a: np.ndarray) -> float:
        r = a[..., 0][blown]
        return float(r.max() - r.min())

    spans = [red_span(graded({"grade_recover": a}, shot))
             for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    check(
        "reconstruction alone recovers visible detail, monotonically in the slider",
        spans[0] < 1e-6 and spans[-1] > 0.02
        and all(b >= a - 1e-6 for a, b in zip(spans, spans[1:])),
        "red's span inside the blown region across the slider: "
        + " -> ".join(f"{s:.4f}" for s in spans)
        + f" (the file itself: {red_span(shot):.4f})",
    )
    hl_only = graded({"grade_highlights": -1.0}, shot)
    check(
        "and Highlights alone cannot -- it can only dim the flat patch",
        red_span(hl_only) < 1e-5,
        f"Highlights -1 leaves the blown region flat at {red_span(hl_only):.2e} "
        "spread, which is the whole reason reconstruction exists",
    )

    # Safety. Two properties survive from the estimate-only version and one had
    # to be replaced.
    #
    # It can no longer be "never darkens": the roll pulls the top of the range
    # down to make room for what was recovered, and *any* curve that brings
    # over-range data into view must move in-gamut values too. What replaces it
    # is a bound on how far -- and that bound is a real regression test, because
    # the first roll applied a uniform scale to the channel maximum (holding hue
    # exactly, as the tone stage does) and a pixel whose red had been rebuilt to
    # 2x its neighbours therefore had its other channels dragged down with it:
    # (1.000, 0.871, 0.634) came out (1.000, 0.305, 0.222), a **dark saturated red
    # where a bright highlight had been**, on 6% of the frame. Rolling per channel
    # instead spends saturation rather than luminance, which is what film does.
    unblown = np.ascontiguousarray(np.clip(truth * 0.55, 0, 1).astype(np.float32))
    d_noop = float(np.abs(graded(rec_p, unblown) - graded({}, unblown)).max())
    white = np.ascontiguousarray(np.full((rh_h, rh_w, 3), 1.0, np.float32))
    d_white = float(np.abs(graded(rec_p, white) - white).max())
    check(
        "reconstruction is a bit-exact no-op unblown, and invents nothing at white",
        d_noop < 1e-6 and d_white < 1e-6,
        f"unblown frame {d_noop:.2e}, all-white frame {d_white:.2e}",
    )
    # The exact form of that claim, and the one that pins the mechanism: a channel
    # sitting **below the roll's knee** must come out bit-identical, however far
    # the channel beside it was rebuilt. Under the uniform scale it did not -- it
    # was divided by the same factor as the rebuilt channel however dark it was,
    # which is precisely how a bright warm highlight became a dark red one.
    #
    # Below the knee rather than merely unclipped, because the roll is a genuine
    # per-channel highlight roll-off: a channel above 0.80 inside a repaired
    # region is legitimately compressed a little (blue peaks at 0.90 here and
    # moves 2.4e-03), which is the stage rolling the top of the range off, not
    # the pixel being dragged.
    lw_v = np.array([0.2126, 0.7152, 0.0722], np.float32)
    base_r = graded({}, shot)
    on_r = graded(rec_p, shot)
    below = shot[..., 2] < _RECON_ROLL_KNEE_V
    d_blue = float(np.abs(on_r[..., 2][below] - base_r[..., 2][below]).max())
    drop = (base_r * lw_v).sum(-1) - (on_r * lw_v).sum(-1)
    check(
        "the roll spends saturation, not luminance -- a channel under the knee is "
        "bit-exact",
        d_blue < 1e-6 and float(drop.max()) < 0.10,
        f"blue below the knee moves {d_blue:.2e} beside a red rebuilt to 1.45; "
        f"worst luminance the roll spends lowering the rebuilt channel "
        f"{float(drop.max()):.3f} (the channel-max uniform scale this replaced "
        "dragged every channel down together and cost 0.435)",
    )

    a = graded({**rec_p, "grade_highlights": -0.6}, shot)
    b = eng.render_image(
        shot, {**P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, **rec_p,
                             "grade_highlights": -0.6}), "lut": None},
        1.0, tile=256, supersample=1)
    d = float(np.abs(a - b).max())
    check("tile independence with reconstruction on", d < 2e-3, f"max delta {d:.2e}")

    # pad_for has to know about the two kernels in this section. Everything else
    # here is per-pixel and must reserve nothing.
    pad_off = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO}), 1.0)
    pad_cl = eng.pad_for(
        P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO},
                    "grade_clarity": 1.0, "grade_clarity_radius": 40.0}), 1.0)
    pad_rc = eng.pad_for(
        P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO},
                    "grade_recover": 1.0, "grade_recover_radius": 40.0}), 1.0)
    pad_both = eng.pad_for(
        P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO},
                    "grade_clarity": 1.0, "grade_clarity_radius": 40.0,
                    "grade_recover": 1.0, "grade_recover_radius": 40.0}), 1.0)
    pad_rest = eng.pad_for(
        P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, "grade_temp": 1.0,
                    "grade_tint": 1.0, "grade_exposure": 1.0,
                    "grade_shadows": 1.0, "grade_highlights": -1.0,
                    "grade_contrast": 1.0, "grade_black_point": 0.2,
                    "grade_vibrance": 1.0, "grade_saturation": 1.0,
                    "lut_amount": 1.0}), 1.0)
    # Reconstruction reserves more than its nominal radius: the estimate reads
    # `radius`, then its weight field is dilated and feathered to gate the roll,
    # and all three are in series. Under-reserve any of them and a tiled export
    # seams along exactly that reach while every preview looks fine.
    check(
        "pad_for reserves for clarity and reconstruction, and nothing else here",
        pad_cl >= pad_off + 3 * 40 and pad_rc >= pad_off + 3 * 1.5 * 40
        and pad_both >= pad_off + 3 * 2.5 * 40 and pad_rest == pad_off,
        f"{pad_off}px off, {pad_cl}px at 40px clarity, {pad_rc}px at 40px "
        f"reconstruction (its three kernels in series), {pad_both}px with both, "
        f"{pad_rest}px with the per-pixel stages on",
    )

    # And the seam check that all of that is for: the whole section on, at
    # default everything else, tiled against a single pass.
    cg_all = P.sanitize({
        "grade_temp": 0.5, "grade_tint": -0.3, "grade_exposure": 0.4,
        "grade_shadows": 0.35, "grade_highlights": -0.4,
        "grade_contrast": 0.5, "grade_black_point": 0.1,
        "grade_clarity": 0.8, "grade_clarity_radius": 24.0,
        "grade_vibrance": 0.4, "grade_saturation": -0.3, "lut_amount": 1.0,
    })
    cg_all["lut"] = rot
    a = eng.render_image(img, cg_all, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, cg_all, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("no seam from colour grading", d < 2e-3, f"max delta {d:.2e}")

    # Scale invariance: clarity is a length, so the proxy and the export have to
    # agree about *the picture* -- the same relative band energy at half scale.
    def clar_ratio(sc: float) -> float:
        im = np.ascontiguousarray(iio.downscale(plate, sc)) if sc < 1 else plate
        pc = {"grade_clarity": 1.0, "grade_clarity_radius": CR}
        on = graded(pc, im) if sc >= 1 else eng.render_image(
            im, {**P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, **pc}),
                 "lut": None}, sc, tile=4096, supersample=1)
        return float(band(on, CR * sc).std() / max(band(im, CR * sc).std(), 1e-9))

    r1, r2 = clar_ratio(1.0), clar_ratio(0.5)
    check(
        "clarity is scale-invariant", abs(r2 / r1 - 1.0) < 0.12,
        f"band gain {r1:.2f}x at 1:1 vs {r2:.2f}x at half scale",
    )

    # -- 3f. .cube parsing ----------------------------------------------------
    print("\n.cube parsing (header keywords, comments, storage order)")
    txt = (
        "# a comment\nTITLE \"t\"\nVENDOR_JUNK 3\n\nDOMAIN_MIN 0 0 0\n"
        "DOMAIN_MAX 1 1 1\nLUT_3D_SIZE 2\n\n"
        # red fastest, then green, then blue -- 8 entries. Each entry is tagged
        # with its own (r, g, b) index so a transposed reshape cannot pass.
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
    )
    pl = lutlib.parse_cube(txt, "t", "t", "folder")
    ok = pl.size == 2 and all(
        abs(pl.table[bi, gi, ri, c] - v) < 1e-6
        for bi in (0, 1) for gi in (0, 1) for ri in (0, 1)
        for c, v in enumerate((ri, gi, bi))
    )
    check("red varies fastest", ok, f"size {pl.size}, table indexed [b][g][r]")

    bad = 0
    for probe, why in (
        ("LUT_1D_SIZE 4\n0 0 0\n", "1D"),
        ("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n", "truncated"),
        ("TITLE \"x\"\n0 0 0\n", "no size"),
    ):
        try:
            lutlib.parse_cube(probe, "p", "p", "folder")
        except lutlib.LutError:
            bad += 1
    check("bad files are refused with a reason", bad == 3, f"{bad}/3 rejected")

    # -- the folder is a tree (2026-08-09) ------------------------------------
    # `luts/` was flat and held 7 files until a library of ~300 arrived
    # organised into subfolders. An id is a *path* relative to `luts/` from that
    # point on, which turned the traversal guard from "reject anything with a
    # separator" into a real one -- see `lut.resolve_path`.
    shipped = lutlib.list_luts()
    on_disk = sorted(
        f for f in lutlib.LUT_DIR.rglob("*") if f.suffix.lower() == ".cube"
    )
    check(
        "every .cube on disk is listed",
        len(shipped) == len(on_disk) and bool(on_disk),
        f"{len(shipped)} listed, {len(on_disk)} files on disk in "
        f"{len({x['group'] for x in shipped})} groups",
    )

    loaded = [(x["id"], lutlib.get(x["id"])) for x in shipped]
    failed = [i for i, l in loaded if l is None]
    check(
        "every LUT in luts/ loads",
        bool(loaded) and not failed,
        # Names, not a roll-call: 303 of them would bury the log. The failures
        # are the part worth printing, and there are normally none.
        f"{len(loaded) - len(failed)}/{len(loaded)} parsed"
        + (f"; FAILED {', '.join(failed[:5])}" if failed else ""),
    )

    # A nested LUT is reachable by its path and *only* by its path. Two folders
    # may hold the same filename -- `gmic/negative_new` and `gmic/negative_old`
    # both ship a `kodak_portra_400` -- so a bare stem that still resolved would
    # hand a preset whichever one the directory walk happened to reach first.
    nested = next((x for x in shipped if x["group"]), None)
    check(
        "a nested LUT resolves by path, not by bare name",
        nested is not None
        and lutlib.get(nested["id"]) is not None
        and lutlib.get(nested["name"]) is None,
        f"{nested['id']} loads; bare {nested['name']!r} does not"
        if nested else "no nested LUT present to test",
    )

    escapes = (
        "../presets/Stock", "..", ".", "",
        "gmic/../../presets/Stock",      # climbs out mid-path
        "/etc/passwd",                   # absolute
        "gmic\\bw\\agfa_apx_100",        # Windows separators
        "gmic//bw/agfa_apx_100",         # empty segment
    )
    escaped = [e for e in escapes if lutlib.get(e) is not None]
    check(
        "a path cannot escape the LUT folder",
        not escaped,
        f"{len(escapes)} traversal attempts resolve to no LUT"
        if not escaped else f"RESOLVED: {escaped}",
    )
