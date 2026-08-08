"""halation blue compensation and highlight recovery

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import colorsys as _cs
import numpy as np
from server import params as P
from tests.harness import Ctx, check, suite


@suite("halation", "halation blue compensation and highlight recovery")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 5e4. halation blue compensation -------------------------------------
    # Halation adds warm light, and adding light desaturates whatever it lands
    # on. On a blue sky that reads as the colour draining toward grey, which
    # is what this stage puts back -- *before* the wash, on the recorded
    # colour, rather than repainting the result. The checks are that it
    # recovers the loss, that it stops there instead of running away, that it
    # touches nothing outside its hue window, and that it leaves the bloom
    # itself alone.
    print("\nhalation blue compensation (put the sky back before the wash)")
    bh, bw = 400, 900
    bl_sky = np.zeros((bh, bw, 3), np.float32)
    bl_sky[:] = np.array([0.18, 0.38, 0.78], np.float32)
    byy, bxx = np.mgrid[0:bh, 0:bw]
    bl_sun = np.clip(1.0 - np.hypot(byy - 200, bxx - 200) / 70.0, 0, 1).astype(np.float32)
    bl_img = np.ascontiguousarray(np.clip(
        bl_sky + bl_sun[..., None] * np.array([1.0, 0.95, 0.85], np.float32), 0, 1
    ).astype(np.float32))
    bl_off = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "edge_jitter": 0, "sharpen": 0, "edge_soften": 0,
        "scatter": 0,
    }
    # A threshold *under* the sky's own luma (0.366), so the whole sky blooms
    # onto itself and the loss is uniform rather than a rim around the sun.
    # That is the case the control exists for, and the one a shipped preset
    # actually hits.
    bl_hal = {"halation": 0.6, "halation_radius": 7.5, "halation_threshold": 0.30}

    def bl_run(over: dict, im: np.ndarray = bl_img) -> np.ndarray:
        return eng.render_image(im, P.sanitize({**bl_off, **over}), 1.0, supersample=1)

    def bl_sat(px) -> float:
        px = np.asarray(px, float)
        return float((px.max() - px.min()) / max(px.max(), 1e-6))

    def bl_hue(px) -> float:
        px = np.asarray(px, float)
        return float(_cs.rgb_to_hsv(*px)[0] * 360.0)

    sky_at = (350, 800)  # far from the sun: pure wash, no local bloom
    tgt = bl_run({"halation": 0.0})[sky_at]
    washed = bl_run(bl_hal)[sky_at]
    check(
        "the wash really does grey the sky",
        bl_sat(washed) < bl_sat(tgt) * 0.9,
        f"sat {bl_sat(tgt):.3f} -> {bl_sat(washed):.3f} "
        f"({(1 - bl_sat(washed) / bl_sat(tgt)) * 100:.0f}% of the colour gone), "
        f"hue {bl_hue(tgt):.1f} -> {bl_hue(washed):.1f} deg",
    )
    fixed = bl_run({**bl_hal, "halation_blue": 0.5})[sky_at]
    check(
        "compensation puts it back",
        abs(bl_sat(fixed) - bl_sat(tgt)) < bl_sat(tgt) * 0.05,
        f"sat {bl_sat(washed):.3f} -> {bl_sat(fixed):.3f} against a target of "
        f"{bl_sat(tgt):.3f}",
    )
    # The reason this runs before the wash rather than after it. Anything
    # added here is eaten by the wash in the same proportion, so the control
    # cannot run away; the identical correction applied afterwards has no such
    # brake and pins the sky at fully saturated.
    runaway = [bl_sat(bl_run({**bl_hal, "halation_blue": k})[sky_at])
               for k in (1.0, 2.0, 3.0)]
    check(
        "and stops there -- the wash is its brake",
        max(runaway) - min(runaway) < 0.01 and max(runaway) < bl_sat(tgt) * 1.10,
        "sat at amount 1/2/3 = " + ", ".join(f"{s:.3f}" for s in runaway)
        + f" (target {bl_sat(tgt):.3f})",
    )
    # Saturation scales chroma about the luma axis, which by construction
    # cannot rotate anything -- so the hue swing needs its own control. This
    # is the check that says the second slider is not decoration.
    sat_only = bl_run({**bl_hal, "halation_blue": 1.0})[sky_at]
    shifted = bl_run({**bl_hal, "halation_blue": 1.0, "halation_blue_shift": -8})[sky_at]
    check(
        "the hue shift fixes what saturation cannot",
        abs(bl_hue(sat_only) - bl_hue(tgt)) > 4.0
        and abs(bl_hue(shifted) - bl_hue(tgt)) < 1.5,
        f"hue error {bl_hue(sat_only) - bl_hue(tgt):+.1f} deg on saturation alone, "
        f"{bl_hue(shifted) - bl_hue(tgt):+.1f} deg with -8 deg of shift",
    )
    # The brightness gate, and the bug it exists for. The wash only reaches
    # what is near the light, so a deep blue is never damaged -- and
    # compensating it anyway is pure overshoot. Measured before the gate
    # existed, amount 2.0 took an untouched deep sky from 0.872 saturation to
    # 1.000, which is a channel clamped to black.
    bl_grad = np.zeros((600, 900, 3), np.float32)
    gt = np.linspace(0, 1, 600, dtype=np.float32)[:, None, None]
    bl_grad[:] = (np.array([0.06, 0.18, 0.58], np.float32) * (1 - gt)
                  + np.array([0.52, 0.68, 0.90], np.float32) * gt)
    gyy2, gxx2 = np.mgrid[0:600, 0:900]
    bl_grad = np.ascontiguousarray(np.clip(
        bl_grad + np.clip(1.0 - np.hypot(gyy2 - 520, gxx2 - 450) / 90.0, 0, 1)
        .astype(np.float32)[..., None] * np.array([1.0, 0.95, 0.85], np.float32),
        0, 1).astype(np.float32))
    g_hal2 = {"halation": 0.8, "halation_radius": 30.0, "halation_threshold": 0.55}
    grad_ref = bl_run({"halation": 0.0}, bl_grad)
    grad_hal = bl_run(g_hal2, bl_grad)
    deep = (20, 450)   # top of the frame, far from the sun: provably undamaged
    check(
        "deep blue is not damaged in the first place",
        abs(bl_sat(grad_hal[deep]) - bl_sat(grad_ref[deep])) < 0.005,
        f"sat {bl_sat(grad_ref[deep]):.3f} -> {bl_sat(grad_hal[deep]):.3f} "
        f"at display luma "
        f"{float(np.dot(grad_ref[deep], (0.2126, 0.7152, 0.0722))):.3f}",
    )
    hot = bl_run({**g_hal2, "halation_blue": 2.0}, bl_grad)[deep]
    ungated = bl_run({**g_hal2, "halation_blue": 2.0, "halation_blue_level": 0.0,
                      "halation_blue_falloff": 0.02}, bl_grad)[deep]
    check(
        "so the gate leaves it alone even when cranked",
        abs(bl_sat(hot) - bl_sat(grad_ref[deep])) < 0.01
        and bl_sat(ungated) > bl_sat(grad_ref[deep]) * 1.1,
        f"sat {bl_sat(grad_ref[deep]):.3f} -> {bl_sat(hot):.3f} gated, "
        f"{bl_sat(ungated):.3f} with the gate open",
    )
    # Raising the level must progressively exclude darker blue, monotonically.
    mid = (260, 450)
    ladder = [bl_sat(bl_run({**g_hal2, "halation_blue": 1.0,
                             "halation_blue_level": L}, bl_grad)[mid])
              for L in (0.30, 0.45, 0.60, 0.75)]
    check(
        "the level progressively excludes darker blue",
        all(ladder[k] >= ladder[k + 1] - 1e-4 for k in range(len(ladder) - 1))
        and ladder[0] > ladder[-1] + 0.05,
        "sat at level 0.30/0.45/0.60/0.75 = "
        + ", ".join(f"{s:.3f}" for s in ladder)
        + f" (untouched {bl_sat(grad_ref[mid]):.3f})",
    )
    # Knee and falloff are separate controls -- the lesson the Luminance
    # Response band already learned. Widening the falloff must move the *foot*
    # of the ramp and leave its top where the knee put it, so changing one
    # never silently changes the other.
    def foot(fall: float) -> float:
        """Lowest display luma still getting any compensation."""
        col = bl_run({**g_hal2, "halation_blue": 1.5, "halation_blue_level": 0.55,
                      "halation_blue_falloff": fall}, bl_grad)[:, 450]
        ref = grad_ref[:, 450]
        moved = np.where(np.abs(col - ref).max(1) > 2e-3)[0]
        return float(np.dot(ref[moved.min()], (0.2126, 0.7152, 0.0722))) if len(moved) else 1.0

    f_narrow, f_wide = foot(0.05), foot(0.45)
    check(
        "knee and falloff are independent",
        f_wide < f_narrow - 0.10,
        f"ramp foot at display luma {f_narrow:.2f} with a 0.05 falloff, "
        f"{f_wide:.2f} with 0.45 -- same 0.55 knee",
    )
    # Nothing outside the hue window may move, at any setting. Grey especially:
    # a compensation that lifts colour where there is none puts a cast on every
    # neutral in the frame, which is the failure `vibrance` is written against.
    hard = {**bl_hal, "halation_blue": 3.0, "halation_blue_shift": 45.0}
    for name, swatch in (
        ("grey", (0.5, 0.5, 0.5)),
        ("red", (0.75, 0.22, 0.18)),
        ("foliage green", (0.24, 0.52, 0.20)),
    ):
        sw_ = np.ascontiguousarray(
            np.tile(np.array(swatch, np.float32), (64, 64, 1)).astype(np.float32)
        )
        d = float(np.abs(bl_run(hard, sw_) - bl_run(bl_hal, sw_)).max())
        check(f"{name} is left alone", d < 1e-6, f"max delta {d:.2e} at maximum settings")
    # The glow is computed *before* the compensation runs, so the two controls
    # are independent: dialling blue must not move the bloom. Probed on a grey
    # field lit by a saturated blue source -- grey is untouched by the
    # compensation (checked above), so any change there is the glow moving.
    gh = 300
    gp = np.full((gh, gh, 3), 0.5, np.float32)
    gyy, gxx = np.mgrid[0:gh, 0:gh]
    gr = np.hypot(gyy - 150, gxx - 150)
    gp[gr < 40] = np.array([0.30, 0.55, 0.99], np.float32)
    gp = np.ascontiguousarray(gp)
    g_hal = {"halation": 0.9, "halation_radius": 25.0, "halation_threshold": 0.45}
    g_ring = (gr > 55) & (gr < 95)
    g_off = bl_run(g_hal, gp)
    g_on = bl_run({**g_hal, "halation_blue": 3.0, "halation_blue_shift": -45.0}, gp)
    moved = float(np.abs(g_on[g_ring] - g_off[g_ring]).max())
    lit = float(np.abs(g_on[gr < 40] - g_off[gr < 40]).max())
    check(
        "compensation does not move the bloom", moved < 1e-6 < lit,
        f"glow on the grey ring moved {moved:.2e} while the blue source itself "
        f"moved {lit:.3f}",
    )
    # With no wash there is nothing to compensate, and this must not become a
    # general blue grade -- colour grading is deferred.
    #
    # Both sides carry a live micro_blur so both actually render. Without it
    # the halation_blue = 0 side has every NEUTRAL_ZERO parameter at zero, so
    # render_image short-circuits and hands back the input bit-exactly, and
    # the comparison measures the sRGB round trip rather than this stage.
    dead = {"halation": 0.0, "micro_blur": 0.45}
    d = float(np.abs(bl_run({**dead, "halation_blue": 3.0, "halation_blue_shift": 45.0})
                     - bl_run(dead)).max())
    check("inert with halation off", d == 0.0, f"max delta {d:.2e}")
    # Purely per-pixel, so it must not widen the tile overlap.
    check(
        "costs no tile overlap",
        eng.pad_for(P.sanitize({**bl_hal, "halation_blue": 3.0}), 1.0)
        == eng.pad_for(P.sanitize(bl_hal), 1.0),
        f"pad_for unchanged at {eng.pad_for(P.sanitize(bl_hal), 1.0)}px",
    )

    # -- 5e5. halation highlight recovery ------------------------------------
    # The bloom is additive in linear light with no upper clamp until display
    # space, so a highlight already close to white gets pushed the rest of the
    # way to a flat clip -- reported as burning highlights out. Recovery meters
    # the added light against the headroom each channel *actually* has left:
    # free where there is room, and at 1.0 unable to reach white at all.
    #
    # Keyed on real per-channel headroom rather than on `hi`, the threshold
    # field the first version used -- `hi` answers "is this bright enough to
    # bloom", which is not the same question as "how much more light can this
    # take".
    print("\nhalation highlight recovery (meter the bloom against headroom)")

    def rec_iso(over: dict, im: np.ndarray) -> np.ndarray:
        return eng.render_image(
            im, P.sanitize({**{k: 0.0 for k in P.NEUTRAL_ZERO}, **over}),
            1.0, supersample=1,
        )

    rh, rw = 48, 240
    rx = np.linspace(0.55, 0.99, rw, dtype=np.float32)
    rec_plate = np.ascontiguousarray(
        np.repeat(np.repeat(rx[None, :, None], rh, 0), 3, 2).astype(np.float32)
    )
    rec_hal = {"halation": 0.9, "halation_threshold": 0.6, "halation_radius": 8.0}

    def clipped_frac(a: np.ndarray) -> float:
        return float((a >= 0.999).mean())

    burned = clipped_frac(rec_iso(rec_hal, rec_plate))
    half = clipped_frac(rec_iso({**rec_hal, "halation_recovery": 0.5}, rec_plate))
    full = clipped_frac(rec_iso({**rec_hal, "halation_recovery": 1.0}, rec_plate))
    check(
        "recovery reduces the burned (clipped) fraction, monotonically",
        burned > half > full,
        f"clipped fraction {burned * 100:.1f}% at 0 -> {half * 100:.1f}% at 0.5 "
        f"-> {full * 100:.1f}% at 1.0",
    )
    d = float(np.abs(
        rec_iso({**rec_hal, "halation_recovery": 0.0}, rec_plate)
        - rec_iso(rec_hal, rec_plate)
    ).max())
    check("recovery 0 is bit-exactly the old behaviour", d == 0.0, f"max delta {d:.2e}")

    # The exact claim, on a flat plate where every quantity is a scalar: the
    # added light must be metered by exactly (H + a(1-r)) / (H + a), with H the
    # channel's own linear headroom and `a` the light the bloom wanted to add.
    # Solved from the recovery-off render's own added light rather than from the
    # glow field, so the check does not have to reimplement the bloom -- and it
    # is an equality, not a "less clipping" judgement.
    flat_val = 0.68
    flat = np.full((256, 256, 3), flat_val, np.float32)
    off = rec_iso({"halation": 0.0}, flat)
    added0 = rec_iso(rec_hal, flat) - off
    lin_of = ((flat_val + 0.055) / 1.055) ** 2.4
    head = 1.0 - lin_of
    worst = 0.0
    for r in (0.35, 0.7, 1.0):
        added_r = rec_iso({**rec_hal, "halation_recovery": r}, flat) - off
        for c in range(3):
            # `a` in linear light, from the recovery-off result for this channel.
            a_lin = (((flat_val + added0[..., c].mean() + 0.055) / 1.055) ** 2.4
                     - lin_of)
            want_lin = lin_of + a_lin * (head + a_lin * (1.0 - r)) / (head + a_lin)
            want = 1.055 * want_lin ** (1.0 / 2.4) - 0.055
            got = flat_val + float(added_r[..., c].mean())
            worst = max(worst, abs(got - want))
    check(
        "recovery meters the added light by exactly (H + a(1-r)) / (H + a)",
        worst < 2e-3,
        f"worst deviation from the closed form over 3 settings x 3 channels: "
        f"{worst:.2e}",
    )

    # The property that makes it a recovery rather than a dimmer: at 1.0 the
    # bloom cannot flatten anything. Measured as the transfer slope along the
    # ramp -- with recovery off the top of it is a clipped plateau, and a
    # plateau is precisely what destroys highlight detail.
    def min_step(a: np.ndarray) -> float:
        return float(np.diff(a.mean(-1).mean(0)).min())

    off_ramp = rec_iso(rec_hal, rec_plate)
    on_ramp = rec_iso({**rec_hal, "halation_recovery": 1.0}, rec_plate)
    check(
        "at full recovery the bloom cannot flatten a highlight gradient",
        min_step(off_ramp) <= 0.0 < min_step(on_ramp),
        f"minimum step along the ramp: {min_step(off_ramp):+.2e} at recovery 0 "
        f"(flat, i.e. detail gone) -> {min_step(on_ramp):+.2e} at 1.0",
    )

    # And that it buys that by *metering* the light rather than deleting it: on
    # a bright plate carrying real fine texture, full recovery has to keep more
    # of the texture than recovery off -- which is the complaint -- while still
    # depositing most of the bloom. Both halves matter: holding the glow back
    # would pass a texture-only test by simply turning the effect off.
    hry, hrx = np.mgrid[0:256, 0:256].astype(np.float32)
    hfine = (np.sin(hrx / 2.3) * np.sin(hry / 2.7)
             + 0.6 * np.sin(hrx / 5.1 + 1.0)) / 1.6
    bright = np.ascontiguousarray(
        (np.clip(0.93 + 0.035 * hfine[..., None], 0, 1)
         * np.ones((1, 1, 3), np.float32)).astype(np.float32))

    def hf(a: np.ndarray) -> float:
        m = a.mean(-1)
        return float((m[1:-1, 1:-1] - 0.25 * (m[:-2, 1:-1] + m[2:, 1:-1]
                                              + m[1:-1, :-2] + m[1:-1, 2:])).std())

    b_off = rec_iso({"halation": 0.0, "micro_blur": 0.3}, bright)
    b_0 = rec_iso(rec_hal, bright)
    b_1 = rec_iso({**rec_hal, "halation_recovery": 1.0}, bright)
    t_src, t_0, t_1 = hf(bright), hf(b_0), hf(b_1)
    light = float((b_1 - b_off).mean()) / max(float((b_0 - b_off).mean()), 1e-9)
    check(
        "full recovery keeps highlight texture *and* most of the bloom",
        t_1 / t_src > t_0 / t_src * 1.3 and light > 0.6,
        f"fine texture kept {t_0 / t_src * 100:.1f}% at recovery 0 -> "
        f"{t_1 / t_src * 100:.1f}% at 1.0, with {light * 100:.1f}% of the "
        "bloom's light still deposited",
    )

    d = float(np.abs(
        rec_iso({"halation": 0.0, "halation_recovery": 1.0, "micro_blur": 0.3}, rec_plate)
        - rec_iso({"halation": 0.0, "micro_blur": 0.3}, rec_plate)
    ).max())
    check("inert with halation off", d == 0.0, f"max delta {d:.2e}")

    check(
        "costs no tile overlap",
        eng.pad_for(P.sanitize({**rec_hal, "halation_recovery": 1.0}), 1.0)
        == eng.pad_for(P.sanitize(rec_hal), 1.0),
        f"pad_for unchanged at {eng.pad_for(P.sanitize(rec_hal), 1.0)}px",
    )

    a = eng.render_image(
        rec_plate, P.sanitize({**rec_hal, "halation_recovery": 1.0}), 1.0,
        tile=4096, supersample=1,
    )
    b = eng.render_image(
        rec_plate, P.sanitize({**rec_hal, "halation_recovery": 1.0}), 1.0,
        tile=64, supersample=1,
    )
    d = float(np.abs(a - b).max())
    check("tile independence with recovery on", d < 2e-3, f"max delta {d:.2e}")
