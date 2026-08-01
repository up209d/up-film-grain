"""Engine invariant checks.

    pipenv run python tests/verify.py

These are the properties that, if broken, produce bugs you will not see in a
preview -- seams that only appear in the export, a preview that stops
predicting the output, colour drift in a build that is meant to be a colour
pass-through. Run this after touching engine.py.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import imageio as iio  # noqa: E402
from server import params as P  # noqa: E402
from server.engine import GrainEngine, device_name, pick_device  # noqa: E402
from tests.scene import patch as scene_patch  # noqa: E402
from tests.scene import scene  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def area_downsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w, _ = a.shape
    h -= h % f
    w -= w % f
    return a[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))


def main() -> int:
    dev = pick_device()
    eng = GrainEngine(dev)
    p = P.sanitize(None)
    print(f"device: {device_name(dev)}\n")

    img = scene(700, 900)

    # -- 1. tile independence ------------------------------------------------
    print("tile independence (tiled render == single-pass render)")
    for ss in (1, 2):
        a = eng.render_image(img, p, 1.0, tile=4096, supersample=ss)
        b = eng.render_image(img, p, 1.0, tile=128, supersample=ss)
        d = float(np.abs(a - b).max())
        check(f"supersample {ss}x", d < 2e-3, f"max delta {d:.2e}")

    # -- 2. crop render matches the full render ------------------------------
    print("\ncrop fidelity (1:1 preview == same region of the export)")
    full = eng.render_image(img, p, 1.0, supersample=2)
    crop = eng.render_crop(img, p, (180, 240, 220, 300), 1.0, 2)
    d = float(np.abs(full[180:400, 240:540] - crop).max())
    check("render_crop", d < 2e-3, f"max delta {d:.2e}")

    # -- 2b. zoomed-out view agrees with a full render at the same scale -----
    print("\nzoom fidelity (zoomed view == full render at that scale)")
    for z in (0.5, 0.25):
        small = eng.render_image(
            np.ascontiguousarray(iio.downscale(img, z)), p, z, tile=4096, supersample=2
        )
        sh, sw, _ = small.shape
        # Align the box to the zoom step. An origin that lands on a half
        # pixel at this scale resamples on a different grid phase, which is a
        # property of the test's box, not of the renderer.
        q = int(round(1 / z))
        by, bx = (int(0.25 * img.shape[0]) // q) * q, (int(0.25 * img.shape[1]) // q) * q
        bh, bw = (int(0.4 * img.shape[0]) // q) * q, (int(0.4 * img.shape[1]) // q) * q
        view = eng.render_view(img, p, (by, bx, bh, bw), z, 2)
        vy, vx = round(by * z), round(bx * z)
        ref = small[vy: vy + view.shape[0], vx: vx + view.shape[1]]
        n = min(ref.shape[0], view.shape[0]), min(ref.shape[1], view.shape[1])
        d = float(np.abs(ref[: n[0], : n[1]] - view[: n[0], : n[1]]).max())
        check(f"zoom {int(z * 100)}%", d < 6e-3, f"max delta {d:.2e}")

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

    # -- 3c. preset rescaling across image sizes -------------------------------
    # A preset dialled in on one size has to hold its look on another. The
    # scale is the ratio of *linear* dimensions, not of pixel counts -- a 24MP
    # frame is 1.29x the width of a 16MP one, not 1.5x -- and only lengths move.
    print("\npreset rescaling (same look on a different-sized photo)")
    check(
        "linear, not area", abs(P.scale_factor(24.0, 96.0) - 2.0) < 1e-6,
        f"24MP -> 96MP is 4x the pixels, {P.scale_factor(24.0, 96.0):.2f}x the width",
    )
    src = P.sanitize({"grain_size": 2.0, "halation_radius": 20.0,
                      "intensity": 45.0, "dust": 50.0, "leak_size": 0.5})
    got = P.rescale(src, 1.6)
    check(
        "lengths scale", abs(got["grain_size"] - 3.2) < 1e-4
        and abs(got["halation_radius"] - 32.0) < 1e-4,
        f"grain_size {src['grain_size']} -> {got['grain_size']:.2f}, "
        f"halation_radius {src['halation_radius']} -> {got['halation_radius']:.1f}",
    )
    check(
        "amounts and counts do not",
        got["intensity"] == src["intensity"] and got["dust"] == src["dust"]
        and got["leak_size"] == src["leak_size"],
        "intensity, dust count and leak_size all unchanged",
    )
    check(
        "every spatial param is marked",
        {x.key for x in P.PARAMS if x.spatial} == {
            "dust_size", "edge_jitter", "edge_sand_grit", "edge_soften_radius",
            "global_size", "grain_size", "hair_length", "halation_radius",
            "highpass_radius", "micro_blur", "pre_sharpen_radius",
            "scatter_cell", "scatter_radius",
            "scratch_width", "sharpen_radius"},
        f"{sum(1 for x in P.PARAMS if x.spatial)} marked spatial",
    )
    check(
        "no reference means no change", P.rescale(src, 1.0) == src,
        "scale_factor(None, x) = " f"{P.scale_factor(None, 40.0):.1f}",
    )

    # -- 4. luminance response peaks in the 15-65% band ----------------------
    print("\nluminance response (grain must peak in midtones/shadows)")
    grad = np.zeros((256, 1024, 3), np.float32)
    grad[:] = np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None]
    # Halation and acutance are separate features that also live in the
    # residual; leaving them on measures their bloom as if it were grain and
    # masks the highlight falloff entirely. Isolate the grain.
    grain_only = P.sanitize({**p, "halation": 0.0, "acutance": 0.0})
    res = eng.render_image(grad, grain_only, 1.0, supersample=2) - grad
    band = []
    for i in range(0, 100, 5):
        a_, b_ = int(i / 100 * 1024), int((i + 5) / 100 * 1024)
        band.append((i, float(res[:, a_:b_].std())))
    peak = max(s for _, s in band)
    peak_at = [i for i, s in band if s == peak][0]
    hi = [s for i, s in band if i >= 95][0]
    check("peak inside 15-65%", 15 <= peak_at <= 65, f"peak at {peak_at}-{peak_at + 5}%")
    check("highlight suppression", hi / peak < 0.30, f"95-100% is {hi / peak * 100:.0f}% of peak")
    for i, s in band:
        print(f"      {i:3d}-{i + 5:3d}%  {'#' * int(s / peak * 34)}")

    # -- 5. grain concentrates on edges rather than flat areas ---------------
    print("\nedge bias (grain onto micro-edges, not flat areas)")
    e = np.full((512, 512, 3), 0.45, np.float32)
    e[:, 256:] = 0.55
    r = eng.render_image(e, p, 1.0, supersample=2) - e
    flat_s = float(r[:, 20:200].std())
    edge_s = float(r[:, 246:266].std())
    check("edge > flat", edge_s > flat_s * 1.1, f"edge {edge_s:.4f} vs flat {flat_s:.4f} ({edge_s / flat_s:.2f}x)")

    # -- 5b. smooth areas must stay clean ------------------------------------
    print("\nsmooth-area guard (skin/sky must not be invaded)")
    big = scene(1000, 1400)
    out = eng.render_image(big, p, 1.0, supersample=2)
    res = out - big
    sy, sx, ph, pw = scene_patch(1000, 1400, "smooth")
    ty, tx, _, _ = scene_patch(1000, 1400, "textured")
    # Inset well past the patch borders: the guard's medium-radius blur
    # reaches across a hard patch edge and would inflate the reading.
    i = 25
    sm = float(res[sy + i:sy + ph - i, sx + i:sx + pw - i].std())
    tx_s = float(res[ty + i:ty + ph - i, tx + i:tx + pw - i].std())
    check(
        "smooth patch is quiet", sm < tx_s * 0.20,
        f"smooth sigma {sm:.4f} vs textured {tx_s:.4f} ({sm / tx_s * 100:.0f}% of textured)",
    )
    check("smooth patch near-clean", sm < 0.006, f"smooth sigma {sm:.4f} (want < 0.006)")

    # -- 5c. global grain: unmasked, and still tile-independent ---------------
    # This layer exists precisely to reach what the masks above protect, so it
    # is checked against the opposite expectation: the smooth patch must now be
    # as grainy as the textured one. It is applied last and touches no
    # statistic of the region, so it must not cost tile independence either --
    # which the default-parameter checks cannot see, since it ships at 0.
    print("\nglobal grain (unmasked overlay)")
    gp = P.sanitize({"global_intensity": 40, "global_size": 1.6, "global_opacity": 1.0})
    gout = eng.render_image(big, gp, 1.0, supersample=2)
    gres = gout - big
    g_sm = float(gres[sy + i:sy + ph - i, sx + i:sx + pw - i].std())
    g_tx = float(gres[ty + i:ty + ph - i, tx + i:tx + pw - i].std())
    check(
        "reaches smooth areas", g_sm > sm * 3.0 and g_sm > 0.6 * g_tx,
        f"smooth sigma {g_sm:.4f} vs masked-only {sm:.4f}, textured {g_tx:.4f}",
    )
    a = eng.render_image(img, gp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, gp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # Scale invariance: the overlay is specified in full-res pixels like every
    # other spatial quantity, so a zoomed view must still predict the export.
    view = eng.render_view(img, gp, (100, 100, 400, 400), 0.5, 2)
    small = eng.render_image(
        np.ascontiguousarray(iio.downscale(img, 0.5)), gp, 0.5, tile=4096, supersample=2
    )
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

    # -- 5f. output sharpening cranks existing grain, invents none -----------
    # The stage is an unsharp mask placed last precisely so the detail it
    # amplifies is the grain. Two things have to hold: with grain on it must
    # raise the grain, and with grain off it must add nothing of its own --
    # on a flat field there is no high-frequency content, so a pure amplifier
    # has nothing to amplify and must be a no-op.
    print("\noutput sharpening (amplifies existing grain, generates none)")
    sharp = {"sharpen": 0.8, "sharpen_radius": 1.0}

    def grain_sigma(over: dict) -> float:
        on = eng.render_image(big, P.sanitize(over), 1.0, supersample=2)
        flat_p = P.sanitize({**over, "intensity": 0, "global_intensity": 0})
        off = eng.render_image(big, flat_p, 1.0, supersample=2)
        return float((on - off)[ty + i:ty + ph - i, tx + i:tx + pw - i].std())

    gs0, gs1 = grain_sigma({"sharpen": 0.0}), grain_sigma(sharp)
    check("grain gets cranked", gs1 > gs0 * 1.2, f"grain {gs1 / gs0 * 100:.0f}% of unsharpened")

    plain = np.full((256, 256, 3), 0.5, np.float32)
    quiet = P.sanitize({
        **sharp, "intensity": 0, "global_intensity": 0, "sharpen": 1.5,
        "edge_erosion": 0, "edge_jitter": 0, "acutance": 0, "micro_blur": 0,
        "halation": 0, "edge_soften": 0,
    })
    d = float(np.abs(eng.render_image(plain, quiet, 1.0, supersample=1) - plain).max())
    check("invents no noise on a flat field", d < 1e-5, f"max delta {d:.2e}")

    a = eng.render_image(img, P.sanitize(sharp), 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, P.sanitize(sharp), 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # -- 5g. film texture: physical damage, drawn without a list of objects --
    # Dust and scratches are the classic way to break tile independence:
    # scatter a list of specks and an export splits one across two tiles, or
    # draws a different list per tile. These are thresholded noise fields in
    # global coordinates instead, so the checks are that each mark type has
    # the geometry it claims, and that the whole section survives tiling.
    print("\nfilm texture (dust, scratches, hair, light leak)")
    tex_off = {
        "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
        "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
    }
    plain = np.full((900, 1400, 3), 0.5, np.float32)

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

    def leak_peaks(var: float) -> np.ndarray:
        got = []
        for sd in (11, 77, 404, 909):
            o = eng.render_image(
                leak_plate,
                P.sanitize({**tex_off, "light_leak": 8.0,
                            "leak_variation": var, "texture_seed": sd}),
                1.0, supersample=1,
            )
            d = (o - leak_plate).max(2)
            for strip, perp in ((d[0, :], d), (d[-1, :], d[::-1])):
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

    flat_leaks, varied = leak_peaks(0.0), leak_peaks(1.0)
    # Coefficient of variation, not brightest-over-dimmest: with a dozen leaks
    # sampled, min/max turns on whichever corner sliver happened to clip the
    # frame edge, and it ranked the two settings backwards on this scene.
    cv0 = float(flat_leaks.std() / max(flat_leaks.mean(), 1e-9))
    cv1 = float(varied.std() / max(varied.mean(), 1e-9))
    check("several leaks per frame", len(varied) >= 8, f"{len(varied)} leaks sampled")
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

    # -- 6. 16-bit PNG export is genuinely 16-bit ----------------------------
    print("\n16-bit PNG writer (Pillow cannot write these; we emit them by hand)")
    small = np.random.default_rng(0).random((37, 53, 3)).astype(np.float32)
    blob = iio.encode(small, "png16")
    pos, chunks = 8, {}
    ok_crc = True
    while pos < len(blob):
        ln = struct.unpack(">I", blob[pos:pos + 4])[0]
        tag = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + ln]
        crc = struct.unpack(">I", blob[pos + 8 + ln:pos + 12 + ln])[0]
        ok_crc &= crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        chunks[tag] = chunks.get(tag, b"") + data
        pos += 12 + ln
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    raw = zlib.decompress(chunks[b"IDAT"])
    px = np.frombuffer(raw, np.uint8).reshape(h, w * 6 + 1)[:, 1:].reshape(h, w, 3, 2).astype(np.uint16)
    back = ((px[..., 0] << 8) | px[..., 1]).astype(np.float32) / 65535.0
    d = float(np.abs(back - small).max())
    check("chunk CRCs", ok_crc, "all valid")
    check("bit depth", depth == 16 and ctype == 2, f"depth={depth} colourtype={ctype}")
    check("precision", d < 5e-5, f"roundtrip {d:.2e} (8-bit floor would be 2e-3)")

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
