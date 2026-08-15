"""Normalize: auto exposure, auto white balance and range compression

Step -2, above everything. The section ships off, so every default-parameter
check in the suite renders straight past it -- which is why tile independence
and `pad_for` are re-run here with the stage switched on, the same way the
global-grain, colour-grading and scatter modules do it.

The measurements that matter here are of two kinds the rest of the suite has
learned to insist on. *Did it correct the right way* is asserted against a
known-wrong input rather than as "something changed", because an auto control's
failure mode is producing a confident answer with the sign inverted -- which is
exactly what the first version did on an over-exposed frame. And *did it keep
the information* is asserted as strict monotonicity of the transfer over a
4096-step ramp, because "the picture still looks fine" cannot tell a rolled
highlight from a flattened one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from server import params as P
from server.engine.colour import _linear_to_srgb, _srgb_to_linear
from server.engine.constants.core import _LUMA
from server.engine.constants.normalize import _NORM_EV_MAX, _NORM_TONE_MAX
from server.engine.grain_engine import GrainEngine
from server.engine.stages.normalize import (
    NORM_IDENTITY, _to_linear_np, _to_srgb_np, meter,
)
from tests.harness import Ctx, check, suite
from tests.scene import scene

#: A fixed, deliberately lopsided correction, for the checks that need the
#: stage to *do* something without depending on what `meter` decides. Not the
#: identity in any component, so a stage that silently dropped one of the six
#: would show up.
_PROBE: dict[str, float] = {
    "norm_ev": 0.45,
    "norm_gain_r": 1.08,
    "norm_gain_g": 0.97,
    "norm_gain_b": 1.05,
    "norm_toe": 0.30,
    "norm_white": 1.9,
}


def _expose(a: np.ndarray, stops: float) -> np.ndarray:
    """A real exposure error: a multiply in *linear* light, then clipped.

    Deliberately not a display-referred multiply, which is what a first pass at
    these fixtures reaches for and is not the same thing at all -- scaling an
    encoded value by 0.25 is nearer one stop than two, so a test built on it
    measures the metering against an input whose true error it has misstated.
    """
    return np.clip(_to_srgb_np(_to_linear_np(a) * (2.0 ** stops)), 0.0, 1.0).astype(
        np.float32
    )


@suite("normalize", "auto exposure, auto white balance and range compression")
def run(cx: Ctx) -> None:
    eng, img = cx.eng, cx.img
    base = scene(600, 800).astype(np.float32)

    def params(over: dict | None = None, on: bool = True) -> dict:
        p = P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO})
        p["normalize"] = 1.0 if on else 0.0
        p["lut"] = None
        if on:
            p.update(_PROBE)
        if over:
            p.update(over)
        return p

    def render(arr, over=None, on=True, tile=4096, ss=1):
        return eng.render_image(arr, params(over, on), 1.0, tile=tile, supersample=ss)

    # -- off is off ----------------------------------------------------------
    print("\nnormalize: the stage ships off")
    off = render(base, on=False)
    d = float(np.abs(off - base).max())
    check("switched off, the frame is untouched", d == 0.0, f"max delta {d:.2e}")

    # Even with the six measured floats present. They arrive from `params_for`
    # whenever a photograph has been metered, and the *checkbox* is what decides
    # whether the stage runs -- not whether the numbers happen to be there.
    stray = P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO})
    stray["lut"] = None
    stray.update(_PROBE)
    d = float(np.abs(eng.render_image(base, stray, 1.0, tile=4096, supersample=1)
                     - base).max())
    check("the measurement alone does not switch it on", d == 0.0,
          f"max delta {d:.2e} with all six floats present and the box unticked")

    check("is_neutral sees the checkbox",
          not P.is_neutral({**{k: 0.0 for k in P.NEUTRAL_ZERO}, "normalize": 1.0})
          and P.is_neutral({k: 0.0 for k in P.NEUTRAL_ZERO}),
          "on renders, off short-circuits")

    # A caller that never went through `params_for` gets the identity, which is
    # what makes the stage safe to call from anywhere in the suite.
    d = float(np.abs(render(base, over=dict(NORM_IDENTITY)) - base).max())
    check("an unmetered frame passes through", d == 0.0, f"max delta {d:.2e}")

    # -- tile independence ---------------------------------------------------
    #
    # The standing requirement for any stage that ships at 0: the default checks
    # render past it, so the seam test has to be re-run with it switched on. It
    # is per-pixel and reserves nothing, so the bar is exact equality rather
    # than the 2e-3 a kernel gets.
    print("\ntile independence with the stage on")
    a = render(img, tile=4096)
    b = render(img, tile=128)
    d = float(np.abs(a - b).max())
    check("no seam from normalize", d == 0.0, f"max delta {d:.2e} at tile 128")

    pad_off = eng.pad_for(params(on=False), 1.0)
    pad_on = eng.pad_for(params(), 1.0)
    check("it reserves nothing in pad_for", pad_off == pad_on,
          f"{pad_off}px off, {pad_on}px on -- per-pixel, so no kernel to cover")

    # -- it corrects, and in the right direction -----------------------------
    #
    # Asserted as a *result* -- where the mid-tones land -- rather than as "the
    # frame changed". A control that moved the picture the wrong way passes any
    # did-something test, and moving it the wrong way is precisely what the
    # first version of the metering did.
    print("\nthe correction lands the mid-tones on target")
    ref = float(np.mean(render(base, over=meter(base))))
    worst_n, worst_e = None, 0.0
    for st in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        src = _expose(base, st)
        got = float(np.mean(render(src, over=meter(src))))
        if abs(got - ref) > worst_e:
            worst_n, worst_e = f"{st:+.1f} stops", abs(got - ref)
    check("a +/-2 stop error converges on the same level", worst_e < 0.06,
          f"worst {worst_n} at {worst_e:.3f} from the correctly-exposed frame's "
          f"{ref:.3f}")

    # The sign. This is the regression test for a real bug: the validity mask
    # excluded clipped pixels from *both* estimators, so on a blown frame the
    # only samples left were its dark ones and the metering asked for +1.38
    # stops brighter on a frame that was 1.4 stops over.
    over_ev = meter(_expose(base, 1.5))["norm_ev"]
    under_ev = meter(_expose(base, -1.5))["norm_ev"]
    check("an over-exposed frame is darkened and an under-exposed one lifted",
          over_ev < -0.5 and under_ev > 0.5,
          f"over {over_ev:+.2f} EV, under {under_ev:+.2f} EV")

    check("the correction is capped rather than unbounded",
          abs(meter(_expose(base, -6.0))["norm_ev"]) <= _NORM_EV_MAX + 1e-9,
          f"a 6-stop-under frame asks for {meter(_expose(base, -6.0))['norm_ev']:+.2f}"
          f" EV, capped at {_NORM_EV_MAX}")

    # A frame blown across nearly all of itself leaves almost nothing inside the
    # *colour* window, and the two guards used to be one early return keyed on
    # that count -- which abandoned the exposure correction along with the white
    # balance, on the most over-exposed input there is. They degrade separately
    # now, so the half that still has samples still runs.
    blown = np.clip(base * 6.0, 0.0, 1.0).astype(np.float32)
    m_blown = meter(blown)
    frac = float((blown.max(axis=2) > 0.92).mean()) * 100.0
    check("a nearly-blown frame still gets its exposure corrected",
          m_blown["norm_ev"] < -0.5,
          f"{frac:.0f}% of it is past the colour window, and it still asks for "
          f"{m_blown['norm_ev']:+.2f} EV")

    # -- and it is visible, in units a human reads ---------------------------
    #
    # `docs/lessons.md`: the split tone shipped for weeks doing something real
    # and invisible. A control whose plausible failure is being too faint has to
    # assert a floor in 8-bit levels.
    src = _expose(base, -1.5)
    lv = float(np.abs(render(src, over=meter(src)) - src).mean()) * 255.0
    check("the correction is visible, not merely real", lv > 2.0,
          f"mean {lv:.1f} 8-bit levels on a 1.5-stop-under frame")

    # -- and it leaves a good photograph alone -------------------------------
    #
    # The other half of the same claim, and the one that decides whether this is
    # safe to leave switched on. A control that fights every well-exposed frame
    # is worse than no control.
    flat = np.zeros((256, 256, 3), np.float32)
    flat[:] = _to_srgb_np(np.array([0.179, 0.179, 0.179]))[None, None, :]
    m_flat = meter(flat)
    check("a neutral, correctly-exposed frame meters as a no-op",
          abs(m_flat["norm_ev"]) < 0.05
          and max(abs(m_flat[f"norm_gain_{c}"] - 1.0) for c in "rgb") < 0.02,
          f"ev {m_flat['norm_ev']:+.3f}, gains within "
          f"{max(abs(m_flat[f'norm_gain_{c}'] - 1.0) for c in 'rgb'):.4f} of 1")

    # -- highlight detail survives, measured in 8-bit levels ----------------
    #
    # **This replaces a check that passed while the stage was destroying
    # highlights, and the way it failed is the lesson.** It asserted that the
    # transfer was strictly increasing over a 4096-step ramp and got 0
    # non-increasing steps -- true, and worthless. Strictly increasing permits
    # increasing by a millionth per step, which is a flat white patch at 8-bit.
    # Measured properly on a real photograph, the source band 0.70..1.00 -- 77
    # levels of highlight -- was arriving as **3.2 levels**.
    #
    # A control whose failure is "the detail is technically there and invisible"
    # has to be measured in the units the eye works in. Same lesson the split
    # tone taught, one layer up: assert the thing a human would notice.
    print("\nhighlight detail survives, in 8-bit levels")
    ramp = np.linspace(0.0, 1.0, 4096, dtype=np.float32)[None, :, None]
    ramp = np.ascontiguousarray(ramp.repeat(3, axis=2).repeat(8, axis=0))

    def hi_levels(over):
        o = render(ramp, over=over)[0, :, 0].astype(np.float64)
        band = o[int(0.70 * 4096):]
        return len(np.unique((band * 255).round())), o

    # A big lift with the frame's own white point, the way `meter` derives it:
    # source 1.0 gained by 2**1.5 lands at that value in linear.
    lift = {**NORM_IDENTITY, "norm_ev": 1.5, "norm_white": 2.0 ** 1.5}
    n_roll, out_roll = hi_levels(lift)
    # The same lift with no roll at all, which is what clipping looks like.
    n_clip, out_clip = hi_levels({**NORM_IDENTITY, "norm_ev": 1.5})

    check("a 1.5-stop lift keeps most of the highlight band",
          n_roll >= 40,
          f"source 0.70..1.00 is 77 8-bit levels; {n_roll} survive "
          f"({100.0 * n_roll / 77:.0f}%)")
    check("and without the roll the same lift flattens it into white",
          n_clip < 5,
          f"{n_clip} levels survive unrolled -- this is the failure the roll "
          f"exists to prevent, and the old fixed-knee version measured 3")

    check("nothing reaches white", float(out_roll.max()) <= 1.0,
          f"brightest output {float(out_roll.max()):.6f}")
    check("the transfer is still strictly increasing",
          int((np.diff(out_roll) <= 0).sum()) == 0,
          "0 non-increasing steps of 4095 -- necessary, and on its own not "
          "sufficient, which is why the level count above is the real check")

    # At `norm_white == 1` the extended Reinhard is algebraically the identity,
    # so a frame that already fits is untouched with no knee and no special
    # case. Asserted as an equality because it is one.
    d = float(np.abs(render(base, over={**NORM_IDENTITY, "norm_white": 1.0})
                     - base).max())
    check("a white point of 1 is exactly the identity", d == 0.0,
          f"max delta {d:.2e} -- x(1+x)/(1+x) = x, so a frame that fits is "
          f"untouched by construction")

    # The real case: a frame carrying more range than fits. The source is
    # already clipped across a fifth of itself, and the promise is not that the
    # detail comes back -- it was never recorded -- but that nothing *new* is
    # lost and the plateau comes off the rail.
    h, w = 300, 400
    lin = np.full((h, w, 3), 0.012, np.float64)
    lin[:, :90] = 2.4
    lin[90:200, 140:320] = np.linspace(0.02, 0.09, 180)[None, :, None]
    hdr = np.clip(_to_srgb_np(lin), 0.0, 1.0).astype(np.float32)
    m_hdr = meter(hdr)
    out_hdr = render(hdr, over=m_hdr)
    # The claim is **no *newly* clipped pixel**, not "no pixel at the rail".
    # Those are different, and the difference is the honest half: a source pixel
    # already at pure white was blown at capture, carries no detail, and pure
    # white is the truthful place for it to land. The tone map maps the frame's
    # own maximum to exactly 1.0 for that reason. What would be a defect is a
    # pixel that *had* detail arriving flat -- so the check asks about pixels
    # that were below the rail on the way in.
    src_rail = hdr.max(axis=2) >= 0.999999
    out_rail = out_hdr.max(axis=2) >= 0.999999
    newly = int((out_rail & ~src_rail).sum())
    check("nothing that had detail is newly clipped",
          float(src_rail.mean()) > 0.10 and newly == 0,
          f"{float(src_rail.mean()) * 100:.1f}% of the source was already at the "
          f"rail; {newly} further pixels reach it, white point "
          f"{m_hdr['norm_white']:.2f}")

    subj_in = float(hdr[90:200, 140:320].mean())
    subj_out = float(out_hdr[90:200, 140:320].mean())
    check("while the subject is actually exposed", subj_out > subj_in * 1.5,
          f"subject mean {subj_in:.3f} -> {subj_out:.3f}")

    # And the same claim on a *real* photograph rather than a synthetic one,
    # because the synthetic frames are the ones that passed while the shipped
    # stage was destroying highlights. Levels in the bright region, in and out.
    for shot in ("film-grain-16x9.jpg", "screenshot.jpg"):
        path = Path(__file__).resolve().parents[2] / shot
        if not path.exists():
            continue
        real = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        o = render(real, over=meter(real))
        bright = real.max(axis=2) > 0.70
        if bright.sum() < 1000:
            continue
        lv_in = len(np.unique((real[bright] * 255).round()))
        lv_out = len(np.unique((o[bright] * 255).round()))
        check(f"a real photograph keeps its highlight detail ({shot})",
              lv_out >= lv_in * 0.8,
              f"{lv_in} distinct levels in the bright region -> {lv_out} out "
              f"({100.0 * lv_out / lv_in:.0f}% kept), "
              f"{float((o >= 0.999).mean()) * 100:.2f}% of the output at white")

    # -- Highlight Priority --------------------------------------------------
    #
    # The one dialled control in the section, and it settles a trade rather than
    # fixing a defect: lifting a dark frame's mid-tones leaves the bright end
    # nowhere to go, so the tone map must compress it. This hands the bright end
    # back to the original file, weighted by how bright it was there.
    print("\nHighlight Priority hands the bright end back to the source")

    def at(hp, over=None, im=None):
        o = dict(over or {})
        o["highlight_priority"] = hp
        return render(im if im is not None else base, over=o)

    # A ramp shows the transfer directly, but it has to be driven by a metering
    # that actually *demands* compression. A ramp's own metering barely corrects
    # anything -- it is uniformly distributed by construction -- so measuring it
    # against itself would report a control that does nothing. The correction a
    # real dark photograph asks for is the case this exists for, so that is the
    # one the ramp is pushed through.
    ramp2 = np.linspace(0.0, 1.0, 4096, dtype=np.float32)[None, :, None]
    ramp2 = np.ascontiguousarray(ramp2.repeat(3, axis=2).repeat(4, axis=0))
    _demanding = Path(__file__).resolve().parents[2] / "film-grain-16x9.jpg"
    m_ramp = meter(
        np.asarray(Image.open(_demanding).convert("RGB"), np.float32) / 255.0
    ) if _demanding.exists() else {**NORM_IDENTITY, "norm_ev": 2.0,
                                   "norm_white": 4.0}

    def band_levels(hp):
        o = at(hp, over=m_ramp, im=ramp2)[0, :, 0].astype(np.float64)
        return len(np.unique((o[int(0.70 * 4096):] * 255).round())), o

    lv0, out0 = band_levels(0.0)
    lv1, out1 = band_levels(1.0)
    check("priority 1 restores the highlight band the correction compressed",
          lv1 > lv0 * 2.5,
          f"the 0.70..1.00 band carries {lv0} of 77 8-bit levels at priority 0 "
          f"and {lv1} at priority 1")

    check("and the curve gets better behaved, not worse",
          int((np.diff(out1) < 0).sum()) == 0
          and float(np.diff(out1).min()) > float(np.diff(out0).min()),
          f"0 non-monotone steps; minimum slope {float(np.diff(out0).min()) * 4095:+.3f} "
          f"at priority 0 against {float(np.diff(out1).min()) * 4095:+.3f} at 1 -- "
          f"a narrow blend band would flatten it instead, which is what sets "
          f"_NORM_HP_LO")

    # On a real photograph: at 1 the bright region should carry the source's own
    # local contrast back, which is the strongest form of "restore the detail".
    for shot in ("film-grain-16x9.jpg", "film-grain-1x1.jpg"):
        path = Path(__file__).resolve().parents[2] / shot
        if not path.exists():
            continue
        real = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        m_r = meter(real)
        hi = real.max(axis=2) > 0.70
        if hi.sum() < 1000:
            continue
        s_sd = float(real[hi].std())
        c0 = float(at(0.0, over=m_r, im=real)[hi].std())
        c1 = float(at(1.0, over=m_r, im=real)[hi].std())
        check(f"priority 1 recovers the source's own highlight contrast ({shot})",
              abs(c1 - s_sd) < abs(c0 - s_sd) and abs(c1 - s_sd) < 0.01,
              f"source {s_sd:.4f}; corrected {c0:.4f} at priority 0, "
              f"{c1:.4f} at 1")

    # The other half of the ask: the rest of the photograph stays corrected.
    real = np.asarray(
        Image.open(Path(__file__).resolve().parents[2] / "film-grain-16x9.jpg")
        .convert("RGB"), np.float32) / 255.0
    m_r = meter(real)
    mid = (real.max(axis=2) > 0.25) & (real.max(axis=2) < 0.45)
    src_mid = float(real[mid].mean())
    mid0 = float(at(0.0, over=m_r, im=real)[mid].mean())
    mid1 = float(at(1.0, over=m_r, im=real)[mid].mean())
    check("while the mid-tones keep most of the correction",
          (mid1 - src_mid) > 0.6 * (mid0 - src_mid),
          f"mid-tones {src_mid:.3f} -> {mid0:.3f} at priority 0, {mid1:.3f} at 1 "
          f"({100.0 * (mid1 - src_mid) / (mid0 - src_mid):.0f}% of the lift kept)")

    # It is a modifier, not a stage: with the box unticked it must do nothing at
    # all, or "Original" would stop being the original.
    d = float(np.abs(render(base, over={"highlight_priority": 1.0}, on=False)
                     - base).max())
    check("it does nothing with Normalize off", d == 0.0, f"max delta {d:.2e}")

    d = float(np.abs(at(0.0) - render(base)).max())
    check("and nothing at 0", d == 0.0,
          f"max delta {d:.2e} against the same render without the key")

    # -- the white balance is colour only ------------------------------------
    #
    # `docs/lessons.md`: a colour shift that is not luma-neutral is two controls
    # fighting. Normalising the gain vector against the luma weights makes this
    # exact rather than approximate, so it is asserted as an equality.
    print("\nwhite balance is luma-neutral by construction")
    luma = np.asarray(_LUMA, dtype=np.float64)
    worst = 0.0
    for nm, cast in (
        ("tungsten", (1.35, 1.0, 0.62)),
        ("shade", (0.72, 0.95, 1.30)),
        ("green", (0.9, 1.25, 0.9)),
    ):
        m = meter(np.clip(base * np.array(cast, np.float32), 0, 1))
        g = np.array([m["norm_gain_r"], m["norm_gain_g"], m["norm_gain_b"]])
        worst = max(worst, abs(float(np.dot(luma, g)) - 1.0))
    check("the gain vector has unit luma", worst < 1e-9,
          f"worst |dot(LUMA, gain) - 1| = {worst:.2e}")

    m_t = meter(np.clip(base * np.array((1.35, 1.0, 0.62), np.float32), 0, 1))
    check("and it corrects the cast rather than following it",
          m_t["norm_gain_r"] < 0.95 and m_t["norm_gain_b"] > 1.05,
          f"a warm cast asks for R {m_t['norm_gain_r']:.3f}, B {m_t['norm_gain_b']:.3f}")

    # A scene that is legitimately one colour must be left alone. Grey-world
    # cannot tell it from a cast by the channel means, so the discriminator is
    # how much the hues vary -- and the check is an equality, not a tolerance.
    sh, sw = 300, 400
    sunset = np.zeros((sh, sw, 3), np.float32)
    sunset[..., 0] = np.linspace(0.55, 0.95, sw)[None, :]
    sunset[..., 1] = np.linspace(0.18, 0.42, sw)[None, :]
    sunset[..., 2] = np.linspace(0.08, 0.16, sw)[None, :]
    m_s = meter(sunset)
    check("a scene that is genuinely one colour is not neutralised",
          max(abs(m_s[f"norm_gain_{c}"] - 1.0) for c in "rgb") < 1e-9,
          "the diversity damper backs the correction off to exactly identity")

    # -- the toe is one-directional ------------------------------------------
    print("\nthe toe engages only where darkening cost shadow separation")
    check("brightening asks for no toe", meter(_expose(base, -1.5))["norm_toe"] == 0.0,
          "a lifted frame moves shadows away from black, so there is nothing to lift")
    toe_over = meter(_expose(base, 1.5))["norm_toe"]
    check("darkening asks for one, capped", 0.0 < toe_over <= _NORM_TONE_MAX + 1e-9,
          f"{toe_over:.3f}, capped at {_NORM_TONE_MAX} -- above that it reads as log")

    # -- the two transfer implementations agree ------------------------------
    #
    # The metering runs in numpy on the CPU and the stage runs in torch on the
    # device, and each carries its own copy of the sRGB curve. They are used on
    # the two halves of one calculation, so a drift between them would make the
    # stage apply a correction measured against a slightly different picture.
    print("\nthe numpy and torch transfer curves are one curve")
    xs = np.linspace(0.0, 1.0, 8192, dtype=np.float64)
    import torch as _t
    tt = _t.tensor(xs, dtype=_t.float64)
    d_lin = float(np.abs(_to_linear_np(xs) - _srgb_to_linear(tt).numpy()).max())
    d_srg = float(np.abs(_to_srgb_np(xs) - _linear_to_srgb(tt).numpy()).max())
    check("sRGB -> linear agrees", d_lin < 1e-12, f"max delta {d_lin:.2e}")
    check("linear -> sRGB agrees", d_srg < 1e-12, f"max delta {d_srg:.2e}")

    # -- metering is deterministic and frame-wide ----------------------------
    #
    # Both halves of invariant 1 as it applies here. The same photograph has to
    # meter identically every time it is opened, or re-uploading a file would
    # silently regrade it; and a *crop* has to meter differently, because if it
    # did not the statistic would not be a property of the frame at all and the
    # per-tile version of this stage would have been safe -- which it is not.
    print("\nthe measurement is a property of the photograph")
    m1, m2 = meter(base), meter(base.copy())
    check("metering the same frame twice gives the same numbers",
          all(m1[k] == m2[k] for k in m1), f"{len(m1)} values, bit-identical")

    crop = np.ascontiguousarray(base[:200, :200])
    m_crop = meter(crop)
    check("a crop meters differently, which is why this cannot run per tile",
          any(abs(m_crop[k] - m1[k]) > 1e-6 for k in m1),
          f"crop ev {m_crop['norm_ev']:+.3f} against the frame's {m1['norm_ev']:+.3f}")

    # -- the six floats reach the checkpoint key -----------------------------
    #
    # They are not `Param`s, so they ride in `p` beside the LUT -- and
    # `upstream_signature` keeps only what is an int or a float. Plain floats
    # land in the key automatically; a tuple would be dropped silently and two
    # photographs would share a cached frame, which is the worst failure in this
    # codebase. This is that filter, asserted.
    print("\nthe measured floats are part of the checkpoint signature")
    warm, cold = GrainEngine(cx.dev), GrainEngine(cx.dev)
    small = scene(240, 360)
    p_a = params()
    warm.render_image(small, p_a, 1.0, tile=4096, supersample=1, checkpoint_id="n:proxy")
    p_b = params(over={"norm_ev": -0.6, "norm_gain_r": 0.9, "norm_gain_b": 1.12})
    got = warm.render_image(small, p_b, 1.0, tile=4096, supersample=1,
                            checkpoint_id="n:proxy")
    want = cold.render_image(small, p_b, 1.0, tile=4096, supersample=1)
    d = float(np.abs(got - want).max())
    check("a different measurement invalidates the warm checkpoint", d == 0.0,
          f"max delta {d:.2e} against a cold engine")

    # And the checkbox itself, which is a `Param` and so is covered by
    # `_downstream` -- but it is the first parameter in the first section above
    # the shallowest boundary, so it is worth pinning that the boundary named
    # for the section *below* it does not swallow it.
    warm2 = GrainEngine(cx.dev)
    p_off = params(on=False)
    warm2.render_image(small, p_off, 1.0, tile=4096, supersample=1,
                       checkpoint_id="n2:proxy")
    got = warm2.render_image(small, p_a, 1.0, tile=4096, supersample=1,
                             checkpoint_id="n2:proxy")
    want = GrainEngine(cx.dev).render_image(small, p_a, 1.0, tile=4096, supersample=1)
    d = float(np.abs(got - want).max())
    check("and so does ticking the box", d == 0.0, f"max delta {d:.2e}")
