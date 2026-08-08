"""colour pass-through, the Original button, vibrance and split tone

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
from server import params as P
from server.engine import (
    _LUMA,
)
from tests.harness import Ctx, check, suite


@suite("colour", "colour pass-through, the Original button, vibrance and split tone")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 3. colour is a pass-through with the deferred grading group neutral --
    print("\ncolour pass-through (grading deferred -- must not tint anything)")
    flat = np.random.default_rng(1).random((256, 256, 3)).astype(np.float32)
    neutral = P.sanitize({
        "intensity": 0, "edge_erosion": 0, "edge_jitter": 0,
        "acutance": 0, "micro_blur": 0, "halation": 0,
    })
    o = eng.render_image(flat, neutral, 1.0, supersample=1)
    d = float(np.abs(o - flat).max())
    check("neutral defaults", d < 1e-5, f"max delta {d:.2e}")

    # -- 3a. the Original button really is original ----------------------------
    # The client's "Original" applies params.neutral_values(), so that has to
    # be exactly a pass-through -- and not just at supersample 1. A bicubic
    # upsample followed by a box downsample is not the identity, so a neutral
    # render at 2x came back 1.0e-01 softer than the input until render_image
    # learned to short-circuit. Bit-exact is the bar: "show me the original"
    # showing something almost-original is the whole failure.
    print("\noriginal (every stage off must return the input)")
    nv = P.sanitize(P.neutral_values())
    check("nothing is active", P.is_neutral(nv), "is_neutral() agrees")
    check("defaults are NOT neutral", not P.is_neutral(p), "so the button has work to do")
    for ss in (1, 2, 3):
        d = float(np.abs(eng.render_image(img, nv, 1.0, supersample=ss) - img).max())
        check(f"pass-through at {ss}x", d == 0.0, f"max delta {d:.2e}")

    # -- 3b. vibrance is saturation-weighted, not flat ------------------------
    # The defining property is the *gradient*: muted colour must gain more
    # than vivid colour. A flat saturation control would raise everything by
    # the same proportion and pass any single-swatch test, so the check is
    # that the gain falls monotonically as starting saturation rises.
    print("\nvibrance (weighted against existing saturation)")
    import colorsys as _cs
    sats = [0.15, 0.35, 0.55, 0.75, 0.95]
    sw = np.zeros((32, 32 * len(sats), 3), np.float32)
    for k, s_ in enumerate(sats):
        sw[:, k * 32:(k + 1) * 32] = _cs.hsv_to_rgb(0.05, s_, 0.75)
    sw = np.ascontiguousarray(sw)
    quiet = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
    }

    def sat_of(v: float) -> list[float]:
        o = eng.render_image(sw, P.sanitize({**quiet, "vibrance": v}), 1.0, supersample=1)
        got = []
        for k in range(len(sats)):
            px = o[16, k * 32 + 16]
            mx, mn = float(px.max()), float(px.min())
            got.append((mx - mn) / max(mx, 1e-4))
        return got

    b0 = sat_of(0.0)
    d0 = max(abs(x - s_) for x, s_ in zip(b0, sats))
    check("neutral at 0", d0 < 5e-3, f"max saturation drift {d0:.2e}")
    up = sat_of(0.8)
    gains = [a / b - 1.0 for a, b in zip(up, b0)]
    check(
        "muted gains more than vivid",
        all(gains[k] > gains[k + 1] for k in range(len(gains) - 1)),
        " > ".join(f"{g * 100:.0f}%" for g in gains),
    )
    dn = sat_of(-0.8)
    check(
        "negative drains muted, spares vivid",
        dn[0] < b0[0] * 0.6 and dn[-1] > b0[-1] * 0.85,
        f"sat {b0[0]:.2f}->{dn[0]:.2f} muted, {b0[-1]:.2f}->{dn[-1]:.2f} vivid",
    )

    # -- 3b2. split tone: both directions, visible, and luma-neutral -----------
    # `highlight_warmth` and `shadow_warmth` replaced `warm_highlights` and
    # `cool_shadows` on 2026-08-06, and both halves of that change need pinning.
    #
    # The direction half is obvious. The *visibility* half is the one worth
    # having: the old pair peaked at a 0.055 shift in one channel, reached only
    # at pure white, so an ordinary highlight moved by under two 8-bit levels --
    # which is why they were reported as doing nothing. "It moved" would have
    # passed on the old code too, so the assertion is a floor in 8-bit levels.
    #
    # And luma-neutrality, because that is what lets these be set independently
    # of Shoulder and Brightness rather than fighting them for the same range.
    print("\nsplit tone (highlight and shadow warmth, both directions)")
    ramp = np.zeros((64, 512, 3), np.float32)
    ramp[:] = np.linspace(0.02, 0.98, 512, dtype=np.float32)[None, :, None]
    lw_ = np.array(_LUMA, np.float32)

    def tone_at(over: dict, at: float) -> np.ndarray:
        o = eng.render_image(ramp, P.sanitize({**quiet, **over}), 1.0, supersample=1)
        return o[32, int(at * 511)].astype(np.float64)

    # Probes sit at 0.74 and 0.21 rather than at the extremes: a full-strength
    # push at 0.95 drives blue through 1.0 and clips, which would make the two
    # directions look asymmetric for a reason that has nothing to do with them.
    for key, probe, other in (
        ("highlight_warmth", 0.75, 0.20),
        ("shadow_warmth", 0.20, 0.75),
    ):
        mid = tone_at({}, probe)
        dw = tone_at({key: 1.0}, probe) - mid
        dc = tone_at({key: -1.0}, probe) - mid
        check(
            f"{key} +1 warms and -1 cools",
            dw[0] > 0 and dw[2] < 0 and dc[0] < 0 and dc[2] > 0,
            f"+1 gives r{dw[0]:+.3f} b{dw[2]:+.3f}, -1 gives r{dc[0]:+.3f} b{dc[2]:+.3f}",
        )
        check(
            f"{key} is symmetric about 0",
            float(np.abs(dw + dc).max()) < 5e-3,
            f"worst |(+1) + (-1)| {float(np.abs(dw + dc).max()):.2e}",
        )
        levels = float(np.abs(dw).max()) * 255.0
        check(
            f"{key} is actually visible", levels > 12.0,
            f"{levels:.0f} 8-bit levels at full strength (the old pair managed 2)",
        )
        check(
            f"{key} holds luma",
            abs(float((dw * lw_).sum())) < 2e-3 and abs(float((dc * lw_).sum())) < 2e-3,
            f"luma moves {float((dw * lw_).sum()):+.2e} warm, {float((dc * lw_).sum()):+.2e} cool",
        )
        far = float(np.abs(tone_at({key: 1.0}, other) - tone_at({}, other)).max())
        check(
            f"{key} stays at its own end of the range",
            far < float(np.abs(dw).max()) * 0.35,
            f"{far * 255:.1f} levels at the other end against {levels:.0f} at its own",
        )
