"""Engine invariant checks.

    pipenv run python tests/verify.py

These are the properties that, if broken, produce bugs you will not see in a
preview -- seams that only appear in the export, a preview that stops
predicting the output, colour drift in a build that is meant to be a colour
pass-through. Run this after touching engine.py.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import imageio as iio  # noqa: E402
from server import params as P  # noqa: E402
from server.engine import (  # noqa: E402
    _GRAIN_CLUSTER, _GRAIN_COS, _GRAIN_FILL, _GRAIN_SHARE, _GRAIN_SIN,
    _GRAIN_SLOTS, GrainEngine, RenderCancelled, _grain_cluster, _grain_gain,
    _grain_points, _lat_span, _lattice_np, _leak_anchor, _leak_sites,
    _smoothstep, device_name, pick_device,
)
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


def gridiness(lum: np.ndarray, cell: float) -> float:
    """|gradient| binned by phase within a cell -- how much a field's structure
    lines up with its own lattice, and therefore with the pixel grid.

    The metric the Global Grain quilt was diagnosed with and the one its
    replacement is held to. A lattice-addressed field swings a long way between
    phases, because its extrema sit *on* the lattice and the gradient vanishes
    there; a field that does not care where the cell boundaries fall does not
    swing. Value noise at a 20px cell scores 1.74; `_grain_points` scores 0.03.
    """
    gx = np.abs(np.diff(lum, axis=1))
    xs = (np.arange(lum.shape[1] - 1) + 0.5) / cell
    ph = np.floor((xs % 1.0) * 8).astype(int)
    # Only bins that actually contain samples. A whole-number cell puts every
    # pixel at one of exactly `cell` phases, so with a cell under 8 some bins
    # are empty and averaging them is a nan, not a zero -- which silently makes
    # the metric useless rather than making it fail.
    m = np.array([gx[:, ph == b].mean() for b in range(8) if (ph == b).any()])
    assert m.size >= 3, f"too few distinct phases at cell {cell}"
    return float((m.max() - m.min()) / m.mean())


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
                      "intensity": 45.0, "dust": 50.0, "leak_variation": 0.5})
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
        and got["leak_variation"] == src["leak_variation"],
        "intensity, dust count and leak_variation all unchanged",
    )
    check(
        "every spatial param is marked",
        {x.key for x in P.PARAMS if x.spatial} == {
            "aa_radius",
            "dust_size", "edge_jitter", "edge_sand_grit", "edge_soften_radius",
            "global_size", "global_size_max", "grade_clarity_radius",
            "grade_recover_radius",
            "grain_size", "hair_length",
            "halation_radius", "highpass_radius", "leak_feather",
            "leak_size_max", "leak_size_min",
            "micro_blur", "pre_blur", "pre_sharpen_radius",
            "scatter_cell", "scatter_radius",
            "scratch_width", "sharpen_radius"},
        f"{sum(1 for x in P.PARAMS if x.spatial)} marked spatial",
    )
    check(
        "no reference means no change", P.rescale(src, 1.0) == src,
        "scale_factor(None, x) = " f"{P.scale_factor(None, 40.0):.1f}",
    )

    # -- 3d. no shipped preset may sit in a mark count's dead zone -----------
    # dust/scratches/hair/light_leak are *counts*, and the engine gates each on
    # `>= 1.0` -- you cannot render a third of a scratch. So a value in (0, 1)
    # renders nothing at all while reading, in the panel and in the file, as
    # though the section were slightly on. Three shipped presets sat there:
    # they carried 0-1 amounts from before these became counts and were never
    # migrated, so their entire Film Texture section had been silently inert.
    # It is invisible from the code and invisible from the UI, which is exactly
    # the kind of thing that only a check catches.
    print("\npreset sanity (mark counts must not sit in the dead zone)")
    dead = [
        (q["name"], k, q["values"][k])
        for q in P.load_presets()
        for k in ("dust", "scratches", "hair", "light_leak")
        if 0.0 < q["values"][k] < 1.0
    ]
    check(
        "no count between 0 and 1", not dead,
        "all counts are 0 or a real number of marks" if not dead
        else ", ".join(f"{n}.{k}={x}" for n, k, x in dead),
    )

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

    shipped = lutlib.list_luts()
    loaded = [(x["name"], lutlib.get(x["id"])) for x in shipped]
    check(
        "every LUT in luts/ loads",
        bool(loaded) and all(l is not None for _, l in loaded),
        ", ".join(f"{n} ({l.size}^3)" if l else f"{n} FAILED" for n, l in loaded)
        or "none present",
    )
    check(
        "a path cannot escape the LUT folder",
        lutlib.get("../presets/Stock") is None and lutlib.get("..") is None,
        "traversal attempts resolve to no LUT",
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

    # -- 5b-i. global chroma: decorrelate the channels, hold the amplitude ----
    # Built as a mean-zero deviation added to the *existing* mono field rather
    # than by the main grain's mean-of-three recipe, so that chroma 0 is
    # bit-identical to the layer every current preset was dialled in against.
    # Three things to pin, and the first is the one a rewrite would break.
    print("\nglobal chroma grain")
    gc_grey = np.full((360, 520, 3), 0.5, dtype=np.float32)

    def gc_render(gc: float, smooth: float = 0.0) -> np.ndarray:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 40.0, "global_size": 4.0,
                     "global_opacity": 1.0, "global_smooth": smooth,
                     "global_chroma": gc})
        out = eng.render_image(gc_grey, P.sanitize(over), 1.0, tile=1024,
                               supersample=1)
        return out.astype(np.float64) - 0.5

    def gc_corr(d: np.ndarray) -> float:
        return float(np.corrcoef(d[:, :, 0].ravel(), d[:, :, 1].ravel())[0, 1])

    # At 0 the three channels must be the same field to the bit. Anything else
    # means the mono component was rebuilt and every shipped preset's global
    # layer just changed pattern.
    d0 = gc_render(0.0)
    spread = float(np.abs(d0[:, :, 0] - d0[:, :, 1]).max())
    check("chroma 0 is exactly monochrome", spread == 0.0,
          f"max channel spread {spread:.2e}")
    # Correlation is `1 - chroma` by construction, not by tuning -- which is the
    # whole reason the coefficients are solved rather than lerped.
    for gc, want in ((0.25, 0.75), (0.5, 0.5), (1.0, 0.0)):
        rho = gc_corr(gc_render(gc))
        check(f"chroma {gc} decorrelates to {want}", abs(rho - want) < 0.05,
              f"channel correlation {rho:+.3f}, wanted {want:+.2f}")
    # Amplitude must not ride along with it, or the slider is a second loudness
    # control. The residual drift is the +-1 clamp meeting a more gaussian
    # field, not the blend: pre-clamp the construction is flat to 0.6%.
    s0 = d0.std()
    for gc in (0.5, 1.0):
        r = gc_render(gc).std() / s0
        check(f"amplitude holds at chroma {gc}", abs(r - 1.0) < 0.05,
              f"{r * 100:.1f}% of monochrome")
    # A second noise field is a second thing `pad_for` has to cover, and it is
    # smoothed by the same kernel -- so re-run tile independence with both on.
    gcp = P.sanitize({"global_intensity": 40, "global_size": 12.0,
                      "global_opacity": 1.0, "global_smooth": 1.0,
                      "global_chroma": 1.0})
    a = eng.render_image(img, gcp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, gcp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")
    # It only exists to colour the global layer, so with that layer off it must
    # be inert -- otherwise it is a colour grade, which is deferred.
    off = {k: 0.0 for k in P.NEUTRAL_ZERO}
    a = eng.render_image(img, P.sanitize({**off, "intensity": 32.0,
                                         "global_chroma": 0.0}), 1.0, tile=1024)
    b = eng.render_image(img, P.sanitize({**off, "intensity": 32.0,
                                         "global_chroma": 1.0}), 1.0, tile=1024)
    d = float(np.abs(a.astype(float) - b.astype(float)).max())
    check("inert with the global layer off", d == 0.0, f"max delta {d:.2e}")

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
    import server.engine as _E  # noqa: E402

    def gv_contrast_cv(cluster: float) -> float:
        keep = _E._GRAIN_CLUSTER
        _E._GRAIN_CLUSTER = cluster
        try:
            f = _grain_points(1024, 1024, 0, 0, 2.0, 4.0, 77,
                              torch.device("cpu"), 1)[0, 0].numpy()
        finally:
            _E._GRAIN_CLUSTER = keep
        f = f.astype(np.float64) - 0.5
        blk = 64                                   # 16 clumps at max size 4
        b = f.reshape(1024 // blk, blk, 1024 // blk, blk).std(axis=(1, 3))
        return float(b.std() / b.mean())

    flat, clustered = gv_contrast_cv(0.0), gv_contrast_cv(_GRAIN_CLUSTER)
    check("clustering gives the layer structure above the clump",
          clustered > flat * 3.0 and clustered > 0.10,
          f"local-contrast spread {clustered:.4f} clustered vs {flat:.4f} "
          f"unclustered ({clustered / flat:.1f}x)")

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

    # And the same thing end to end through the renderer, at the sizes and
    # smoothing settings where a reach bug would actually bite.
    gv_scene = np.ascontiguousarray(
        (np.random.RandomState(9).rand(320, 480, 3) * 0.5 + 0.25).astype(np.float32)
    )
    for lo, hi, smooth, tile in (
        (0.4, 1.0, 0.0, 64), (1.0, 2.0, 0.6, 64), (0.8, 20.0, 1.0, 96),
    ):
        p = P.sanitize({"global_intensity": 40, "global_size": lo,
                        "global_size_max": hi, "global_opacity": 1.0,
                        "global_smooth": smooth})
        wide = eng.render_image(gv_scene, p, 1.0, tile=4096, supersample=1)
        narrow = eng.render_image(gv_scene, p, 1.0, tile=tile, supersample=1)
        d = float(np.abs(wide.astype(float) - narrow.astype(float)).max())
        check(f"tile independence (min={lo}, max={hi}, smooth={smooth})",
              d < 2e-3, f"max delta {d:.2e}")

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

    # -- 5b-ii. master opacity: a cross-fade over the untouched source --------
    # The two ends have to be *bit* exact, not close. Opacity 0 is a second
    # "show me the original" path and inherits that stage's trap: the
    # supersample round trip is not the identity, so blending anywhere inside
    # render() would hand back an image 1e-01 softer than the source and call
    # it "no effect". Checked at every supersample for exactly that reason.
    print("\nmaster opacity (cross-fade over the untouched source)")
    mo_ref = P.sanitize(None)
    for ss in (1, 2, 3):
        z = eng.render_image(img, P.sanitize({"master_opacity": 0.0}), 1.0,
                             supersample=ss)
        d = float(np.abs(z - img).max())
        check(f"0 returns the source at {ss}x", d == 0.0, f"max delta {d:.2e}")
    for ss in (1, 2):
        a = eng.render_image(img, P.sanitize({"master_opacity": 1.0}), 1.0,
                             supersample=ss)
        b = eng.render_image(img, mo_ref, 1.0, supersample=ss)
        d = float(np.abs(a - b).max())
        check(f"1 is the untouched pipeline at {ss}x", d == 0.0,
              f"max delta {d:.2e}")
    # Linear in between, or it is not a cross-fade. Measured against the two
    # ends rather than against a formula, so a stage that quietly re-ran at
    # partial strength would show up.
    full = eng.render_image(img, mo_ref, 1.0, supersample=2)
    worst = 0.0
    for op in (0.25, 0.5, 0.75):
        got = eng.render_image(img, P.sanitize({"master_opacity": op}), 1.0,
                               supersample=2)
        worst = max(worst, float(np.abs(got - (img + (full - img) * op)).max()))
    check("the middle is a straight cross-fade", worst < 1e-6,
          f"worst deviation from the exact blend {worst:.2e}")
    # It dials the *whole* pipeline back, not just the grain, so a stage that
    # was applied after the blend would keep its full strength here.
    heavy = P.sanitize({"halation": 0.8, "intensity": 60, "sharpen": 4.0,
                        "dust": 40, "light_leak": 4})
    h1 = eng.render_image(img, heavy, 1.0, supersample=2)
    h5 = eng.render_image(img, {**heavy, "master_opacity": 0.5}, 1.0,
                          supersample=2)
    r = float(np.abs(h5 - img).mean()) / float(np.abs(h1 - img).mean())
    check("everything scales together", abs(r - 0.5) < 0.02,
          f"mean deviation from the source is {r * 100:.1f}% of full strength "
          "with halation, grain, sharpening, dust and leaks all on")
    # Per-pixel against the tile's own input, so this must be free -- but it is
    # the last thing in the pipeline and a mistake here would seam every export.
    mop = P.sanitize({"master_opacity": 0.4})
    a = eng.render_image(img, mop, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, mop, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence", d < 2e-3, f"max delta {d:.2e}")

    # -- 5c-ii. global smoothness: soften the grain, keep the amplitude -------
    # This was built to remove the value-noise quilt, and there is no longer a
    # quilt for it to remove -- `_grain_points` scores under 0.05 on `gridiness`
    # before any smoothing, against that field's 1.4-1.7. So what it has to be
    # now is what it always claimed to be for the strength half: a *shape*
    # control that visibly softens the grain at constant loudness.
    print("\nglobal grain smoothing (the shape, not the strength)")
    gs_grey = np.full((512, 512, 3), 0.5, np.float32)
    GS_CELL = 20.0

    def gs_render(sm: float) -> np.ndarray:
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 20.0, "global_size": GS_CELL,
                     "global_opacity": 1.0, "global_smooth": sm})
        return eng.render_image(
            gs_grey, P.sanitize(over), 1.0, tile=1024, supersample=1)

    gs_off, gs_on = gs_render(0.0), gs_render(1.0)
    # It starts ungridded and must stay that way -- pinned at both ends so a
    # future field that reintroduced a lattice could not hide behind the blur.
    for tag, im in (("unsmoothed", gs_off), ("smoothed", gs_on)):
        q = gridiness(im.mean(axis=2), GS_CELL)
        check(f"the field is free of the lattice grid ({tag})", q < 0.35,
              f"phase-binned gridiness {q:.3f} (value noise scored 1.4-1.7)")
    # Softening means the gradient falls while the amplitude does not -- the two
    # together are what separate a shape control from a volume control, and
    # either alone would pass on a stage that did the wrong thing.
    g0 = float(np.abs(np.diff(gs_off.mean(axis=2), axis=1)).mean())
    g1 = float(np.abs(np.diff(gs_on.mean(axis=2), axis=1)).mean())
    check("it visibly softens the grain", g1 < 0.7 * g0,
          f"mean |gradient| {g0:.5f} -> {g1:.5f} ({g1 / g0 * 100:.0f}%)")
    # The whole reason the blur carries an analytic gain: a structure control
    # that quietly turns the layer down leaves Global Intensity fighting it,
    # and "smoother" becomes indistinguishable from "less". Same rule `_fbm`
    # follows for Octaves.
    s0 = float(gs_off.mean(axis=2).std())
    worst = 0.0
    for sm in (0.25, 0.5, 0.75, 1.0):
        r = float(gs_render(sm).mean(axis=2).std()) / s0
        worst = max(worst, abs(r - 1.0))
    check(
        "strength is held constant across the slider", worst < 0.06,
        f"worst amplitude drift {worst * 100:.1f}% over smoothness 0.25-1.0",
    )
    # The gain is a closed form in sigma/cell, so it has to hold at every clump
    # size and not just the one it was fitted at.
    for cell in (4.0, 12.0):
        over = {k: 0.0 for k in P.NEUTRAL_ZERO}
        over.update({"global_intensity": 20.0, "global_size": cell,
                     "global_opacity": 1.0})
        a = eng.render_image(gs_grey, P.sanitize({**over, "global_smooth": 0.0}),
                             1.0, tile=1024, supersample=1)
        b = eng.render_image(gs_grey, P.sanitize({**over, "global_smooth": 1.0}),
                             1.0, tile=1024, supersample=1)
        r = float(b.mean(axis=2).std()) / float(a.mean(axis=2).std())
        check(f"the gain holds at a {cell:.0f}px clump", abs(r - 1.0) < 0.08,
              f"amplitude {r * 100:.1f}% of unsmoothed")
    # It is a blur on the noise field, so it is a kernel `pad_for` has to
    # cover -- and a kernel missing from pad_for seams a tiled export along
    # exactly its radius while every preview looks fine.
    smp = P.sanitize({"global_intensity": 40, "global_size": 12.0,
                      "global_opacity": 1.0, "global_smooth": 1.0})
    a = eng.render_image(img, smp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, smp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence with smoothing on", d < 2e-3, f"max delta {d:.2e}")

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

    def lk_depth(o: np.ndarray, side: str) -> int:
        """Deepest pixel a leak reaches from one border, ignoring the others.

        Every probe here has to be walled off from the *other* three borders,
        which the perimeter wash never needed: leaks are discrete beams now and
        a top-border one leans a long way sideways. Each window is chosen to
        sit past the reach any leak on a perpendicular border could have, so
        what it measures is only ever the border it is aimed at.
        """
        d = np.abs(o - lk_plate).max(2)
        strip = (d[400:600, :700].max(0) if side == "left"
                 else d[:400, 400:1100].max(1))
        on = np.where(strip > 0.01)[0]
        return int(on.max()) + 1 if len(on) else 0

    grew = [lk_depth(lk_run({"leak_size_min": s, "leak_size_max": s,
                             "leak_feather": max(2, s // 4),
                             "leak_variation": 0.0}), "left")
            for s in (60, 120, 240, 400)]
    check(
        "leak size is a distance in pixels",
        all(grew[k] < grew[k + 1] for k in range(len(grew) - 1)) and grew[0] < 120,
        "sizes 60/120/240/400px reach " + ", ".join(f"{g}" for g in grew) + "px",
    )
    iso = lk_run({"leak_size_min": 240, "leak_size_max": 240, "leak_feather": 60,
                  "leak_variation": 0.0})
    lft, tp = lk_depth(iso, "left"), lk_depth(iso, "top")
    check(
        "both frame axes agree", abs(lft - tp) < max(lft, tp) * 0.45,
        f"240px reaches {lft}px from the side and {tp}px from the top on a 3:2 frame",
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
    lone = {"light_leak": 1.0, "leak_size_min": 300, "leak_size_max": 300,
            "leak_variation": 0.0, "leak_strength": 0.1}

    def lone_profile(f: int) -> np.ndarray:
        """One leak's falloff, rotated so it runs inward from row 0."""
        for sd in range(20, 80):
            border, _ = _leak_anchor(_leak_sites(1.0, sd, 0.0)[0]["pos"],
                                     1000.0, 1500.0)
            g = np.abs(lk_run({**lone, "leak_feather": f,
                               "texture_seed": float(sd)}) - lk_plate).max(2)
            g = {0: g, 1: g[::-1], 2: g.T, 3: g.T[::-1]}[border]
            xs = np.where(g[0] > 0.01)[0]
            if len(xs) and xs.min() > 0 and xs.max() < g.shape[1] - 1:
                return g[:, int(g[0].argmax())]
        return np.zeros(1)

    def half_at(f: int) -> int:
        col = lone_profile(f)
        below = np.where(col < col[:4].max() * 0.5)[0]
        return int(below.min()) if len(below) else -1

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
    # has no length to measure and no second edge to compare.
    aspect, hardness = [], []
    for sd in range(20, 44):
        st = _leak_sites(1.0, sd, 1.0)[0]
        border, _ = _leak_anchor(st["pos"], 1000.0, 1500.0)
        o = lk_run({"light_leak": 1.0, "leak_strength": 0.2,
                    "leak_size_min": 240, "leak_size_max": 240,
                    "leak_feather": 120, "leak_variation": 1.0,
                    "texture_seed": float(sd)})
        g = np.abs(o - lk_plate).max(2)
        g = {0: g, 1: g[::-1], 2: g.T, 3: g.T[::-1]}[border]
        m = g > 0.01
        xs = np.where(m[0])[0]
        if not len(xs) or xs.min() == 0 or xs.max() == m.shape[1] - 1:
            continue
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

    # -- 6. 16-bit PNG export is genuinely 16-bit ----------------------------
    # -- performance rewrites: every one of these must be bit-exact -----------
    #
    # These four changes exist purely to make the render faster and are only
    # correct if they change nothing at all. Each is checked against a reference
    # implementation of the code it replaced rather than against a tolerance,
    # because "close enough" is not the contract -- a re-rolled noise field would
    # silently restyle every preset.
    print("\nperformance rewrites are bit-exact")

    # `_lattice_np` moved from numpy uint64 to torch int64 for the threading.
    # torch has no uint64, so the logical right shift is emulated; a wrong mask
    # produces something that still looks like noise, which is exactly why this
    # is an equality check and not a render comparison.
    def lattice_ref(iy0, ix0, hl, wl, seed, nfields):
        yy = np.arange(iy0, iy0 + hl, dtype=np.int64).view(np.uint64)[:, None]
        xx = np.arange(ix0, ix0 + wl, dtype=np.int64).view(np.uint64)[None, :]
        out = np.empty((nfields, hl, wl), dtype=np.float32)
        for f in range(nfields):
            s = np.uint64(((seed + f * 7919) * 0x165667B19E3779F9) % (1 << 64))
            n = (xx * np.uint64(0x9E3779B97F4A7C15)
                 + yy * np.uint64(0xC2B2AE3D27D4EB4F))
            n = n + s
            n = n ^ (n >> np.uint64(29))
            n = n * np.uint64(0xBF58476D1CE4E5B9)
            n = n ^ (n >> np.uint64(32))
            n = n * np.uint64(0x94D049BB133111EB)
            n = n ^ (n >> np.uint64(31))
            out[f] = (n >> np.uint64(40)).astype(np.float32) / float(1 << 24)
        return out

    lat_bad = 0
    lat_n = 0
    for nf in (1, 2, 3, 4, 6):
        for iy0, ix0 in ((0, 0), (-1, -1), (-7, 13), (-3841, -2), (5000, 9999)):
            for hl, wl in ((1, 1), (2, 3), (17, 29), (64, 64)):
                for seed in (0, 1, 7717, 3391, 4241, 2 ** 31 - 1):
                    lat_n += 1
                    if not np.array_equal(
                        _lattice_np(iy0, ix0, hl, wl, seed, nf),
                        lattice_ref(iy0, ix0, hl, wl, seed, nf),
                    ):
                        lat_bad += 1
    check("the torch lattice hash equals the numpy one", lat_bad == 0,
          f"{lat_n} windows, {lat_bad} differ (negative origins included)")

    # `_lat_span` replaced four `float(<device tensor>)` reads per noise call
    # with Python arithmetic. It has to agree with the device path *exactly*: a
    # float64 version would occasionally land the other side of an integer
    # boundary and select a different lattice window, which is a different field.
    def span_ref(n, origin, cell, pad_lo, pad_hi):
        t = (torch.arange(n, device=dev, dtype=torch.float32)
             + float(origin)) / cell
        i0 = int(math.floor(float(t[0]))) - pad_lo
        return i0, int(math.floor(float(t[-1]))) + pad_hi - i0 + 1

    span_bad = 0
    span_n = 0
    for cell in (0.8, 1.0, 1.6, 2.0, 2.22, 3.2, 6.0, 110.0, 900.0):
        for pl, ph in ((1, 2), (0, 0), (2, 2)):
            for n in (1, 17, 512, 1536, 3072, 4800):
                for origin in (0.0, 1.0, 7.0, 13.0, 178.0, 1023.0, 4096.0):
                    span_n += 1
                    if _lat_span(n, origin, cell, pl, ph) != span_ref(
                        n, origin, cell, pl, ph
                    ):
                        span_bad += 1
    check("lattice bounds computed in Python match the device ramp",
          span_bad == 0, f"{span_n} cases, {span_bad} differ")

    # `_grain_points` searches only the 3x3 ring of cells around a pixel's own,
    # and unrolls its `falloff ** _GRAIN_SHARE` weight into repeated multiplies.
    # Both are shortcuts, and both are only correct if they change nothing --
    # which "the render still looks like grain" cannot tell you. So this is a
    # plain, deliberately slow reference: a wider 5x5 search and a real `pow`.
    #
    # The 5x5 half is the interesting one. It is the *proof* behind
    # `_GRAIN_RINGS` written out as a measurement: a point two cells away is
    # further than one cell from any pixel in the centre cell, and no radius can
    # exceed one cell, so those candidates must contribute exactly nothing. If
    # the jitter range or the lattice pitch is ever changed without redoing that
    # argument, this is what fails.
    def grain_ref(h, w, y0, x0, lo, hi, seed, device, nfields=1):
        rings = 2
        cell = hi
        ca, sa = _GRAIN_COS, _GRAIN_SIN
        Y = (torch.arange(h, device=device, dtype=torch.float32)
             + float(y0))[:, None]
        X = (torch.arange(w, device=device, dtype=torch.float32)
             + float(x0))[None, :]
        Yr = (Y * ca + X * sa) / cell
        Xr = (X * ca - Y * sa) / cell
        ys = (float(y0), float(y0) + h - 1)
        xs = (float(x0), float(x0) + w - 1)
        vs = [(yy * ca + xx * sa) / cell for yy in ys for xx in xs]
        us = [(xx * ca - yy * sa) / cell for yy in ys for xx in xs]
        pad = rings + 1
        iy0 = int(math.floor(min(vs))) - pad
        hl = int(math.floor(max(vs))) + pad + 1 - iy0
        ix0 = int(math.floor(min(us))) - pad
        wl = int(math.floor(max(us))) + pad + 1 - ix0
        per = 3 + nfields
        lat = torch.from_numpy(
            _lattice_np(iy0, ix0, hl, wl, seed, _GRAIN_SLOTS * per)).to(device)
        ciy = torch.arange(iy0, iy0 + hl, device=device,
                           dtype=torch.float32)[:, None]
        cix = torch.arange(ix0, ix0 + wl, device=device,
                           dtype=torch.float32)[None, :]
        camp = _grain_cluster(iy0, ix0, hl, wl, seed + 991, device)
        piy = (torch.floor(Yr).long() - iy0).clamp(0, hl - 1)
        pix = (torch.floor(Xr).long() - ix0).clamp(0, wl - 1)
        peak = torch.zeros(h, w, device=device)
        num = torch.zeros(nfields, h, w, device=device)
        den = torch.zeros(h, w, device=device)
        for s in range(_GRAIN_SLOTS):
            b = s * per
            u = lat[b + 2]
            rad = torch.where(
                u < _GRAIN_FILL,
                (lo + (hi - lo) * (u / _GRAIN_FILL)) / cell,
                torch.zeros_like(u),
            )
            bri = (lat[b + 3: b + 3 + nfields] * 2.0 - 1.0) * camp
            py, px = ciy + lat[b], cix + lat[b + 1]
            for dy in range(-rings, rings + 1):
                for dx in range(-rings, rings + 1):
                    ny = (piy + dy).clamp(0, hl - 1)
                    nx = (pix + dx).clamp(0, wl - 1)
                    dyp, dxp = Yr - py[ny, nx], Xr - px[ny, nx]
                    sh = 1.0 - _smoothstep(
                        0.0, 1.0,
                        (torch.sqrt(dyp * dyp + dxp * dxp) + 1e-7)
                        / rad[ny, nx].clamp_min(1e-12))
                    wgt = sh ** _GRAIN_SHARE
                    num = num + wgt * bri[:, ny, nx]
                    den = den + wgt
                    peak = torch.maximum(peak, sh)
        val = num / den.clamp_min(1e-12)
        return (0.5 + (0.5 * _grain_gain(lo, hi))
                * peak.unsqueeze(0) * val).unsqueeze(0)

    gr_worst = 0.0
    # Integer and non-integer cells both. The integer sizes are where the old
    # construction phase-locked against the pixel grid, and they are the ones a
    # slider actually lands on.
    for h_, w_, nf, lo_, hi_, oy, ox in (
        (256, 256, 1, 1.0, 3.0, 0.0, 0.0),
        (256, 256, 3, 1.0, 3.0, 17.0, 29.0),
        (192, 288, 3, 2.0, 6.0, 101.0, 7.0),
        (160, 160, 3, 0.8, 0.8, 5.0, 11.0),
        (160, 160, 3, 1.0, 2.0, 0.0, 0.0),
        (160, 240, 1, 0.5, 20.0, 63.0, 41.0),
    ):
        d = float((
            _grain_points(h_, w_, oy, ox, lo_, hi_, 7717, dev, nf)
            - grain_ref(h_, w_, oy, ox, lo_, hi_, 7717, dev, nf)
        ).abs().max())
        gr_worst = max(gr_worst, d)
    check("the 3x3 search and unrolled weight change nothing",
          gr_worst < 1e-6, f"worst deviation {gr_worst:.2e} over 6 configurations")

    # -- the Global Grain texture cache --------------------------------------
    #
    # The one failure mode here is a key that misses an input, and it fails
    # *silently*: a stale hit renders a perfectly plausible texture that happens
    # to be the previous one. So this tests it as a cache -- which parameters
    # miss, and whether reverting one returns the original bytes -- rather than
    # only checking that some render looks right.
    print("\nGlobal Grain texture cache (a stale hit is invisible, so test the cache)")
    gg_eng = GrainEngine(dev)
    gp = P.sanitize({
        "global_intensity": 12.0, "global_opacity": 0.8, "global_size": 1.6,
        "global_size_max": 4.0, "global_chroma": 0.6, "global_smooth": 0.3,
        "intensity": 0.0, "halation": 0.0, "micro_blur": 0.0,
    })
    gimg = scene(220, 300)

    def gg_render(pp):
        return gg_eng.render_image(pp and pp, pp, 1.0, tile=4096, supersample=1)

    gg_eng.clear_caches()
    cold = gg_eng.render_image(gimg, gp, 1.0, tile=4096, supersample=1)
    m_after_cold = gg_eng.gg_misses
    warm = gg_eng.render_image(gimg, gp, 1.0, tile=4096, supersample=1)
    check("a warm cache returns the identical frame",
          float(np.abs(cold - warm).max()) == 0.0
          and gg_eng.gg_misses == m_after_cold,
          f"maxdiff {float(np.abs(cold - warm).max()):.2e}, "
          f"{gg_eng.gg_misses - m_after_cold} further misses")

    # Every input the field is built from must miss, must change the frame, and
    # must come back bit-exact when reverted.
    for key, delta in (("seed", 1.0), ("global_size", 0.6),
                       ("global_size_max", 2.0), ("global_smooth", 0.4),
                       ("global_chroma", -0.5)):
        q = P.sanitize({**gp, key: gp[key] + delta})
        before = gg_eng.gg_misses
        other = gg_eng.render_image(gimg, q, 1.0, tile=4096, supersample=1)
        missed = gg_eng.gg_misses > before
        moved = float(np.abs(other - cold).max())
        back = gg_eng.render_image(gimg, gp, 1.0, tile=4096, supersample=1)
        exact = float(np.abs(back - cold).max()) == 0.0
        check(f"{key} invalidates the cache", missed and moved > 1e-6 and exact,
              f"missed={missed} moved {moved:.2e} revert bit-exact={exact}")

    # The two amplitude sliders are applied outside the cached field, which is
    # the whole reason this cache is worth having: they are what a user drags.
    for key, delta in (("global_intensity", 6.0), ("global_opacity", -0.3)):
        q = P.sanitize({**gp, key: gp[key] + delta})
        before = gg_eng.gg_misses
        other = gg_eng.render_image(gimg, q, 1.0, tile=4096, supersample=1)
        moved = float(np.abs(other - cold).max())
        check(f"{key} reuses the cached texture",
              gg_eng.gg_misses == before and moved > 1e-6,
              f"misses={gg_eng.gg_misses - before} (want 0), moved {moved:.2e}")

    # Tile independence again, but with the cache *warm* -- the key carries
    # absolute (y0, x0), so a tiled render must not be able to pick up a
    # neighbouring tile's texture.
    a = gg_eng.render_image(gimg, gp, 1.0, tile=4096, supersample=1)
    b = gg_eng.render_image(gimg, gp, 1.0, tile=96, supersample=1)
    d = float(np.abs(a - b).max())
    check("the cache is keyed on absolute coordinates", d < 2e-3,
          f"tiled vs whole-image, warm cache: {d:.2e}")

    # -- tile size is chosen, not fixed --------------------------------------
    #
    # `tile_for` now derives the tile from a memory budget, so the renderer sees
    # sizes nobody hard-coded. Tile independence is what makes that safe, and it
    # is only safe if it holds at whatever size the budget picks.
    print("\ntile size is derived from a memory budget")
    # Explicit fresh parameter sets rather than reusing `p`, which earlier
    # sections rebind -- these checks compare *pads*, so they must not inherit
    # whatever the previous section left behind.
    narrow = P.sanitize(None)
    ref = eng.render_image(img, narrow, 1.0, tile=4096, supersample=2)
    worst = 0.0
    for tile in (256, 384, 512, 1024):
        d = float(np.abs(
            eng.render_image(img, narrow, 1.0, tile=tile, supersample=2) - ref
        ).max())
        worst = max(worst, d)
    check("any tile size gives the same picture", worst < 2e-3,
          f"worst deviation {worst:.2e} over tiles 256-1024 vs single-pass")

    hi_tile = eng.tile_for(narrow, 1.0, 4000, 6000, 2)
    _prev = os.environ.get("FILM_GRAIN_TILE_BUDGET_GB")
    os.environ["FILM_GRAIN_TILE_BUDGET_GB"] = "2"
    lo_tile = eng.tile_for(narrow, 1.0, 4000, 6000, 2)
    if _prev is None:
        os.environ.pop("FILM_GRAIN_TILE_BUDGET_GB", None)
    else:
        os.environ["FILM_GRAIN_TILE_BUDGET_GB"] = _prev
    check("a smaller budget picks a smaller tile", lo_tile < hi_tile,
          f"{hi_tile}px at this machine's budget, {lo_tile}px at 2GB")
    # A wider kernel pads more, so it must get a *smaller* tile for the same
    # budget -- that coupling is the point, and the old constants had none. The
    # search steps in 128px increments, so this needs a pad difference wider than
    # that to show; halation at full radius is 276px against the default 108px.
    wide = P.sanitize({"halation": 1.0, "halation_radius": 400.0})
    wide_tile = eng.tile_for(wide, 1.0, 4000, 6000, 2)
    check("a wider kernel gets a smaller tile", wide_tile < hi_tile,
          f"pad {eng.pad_for(wide, 1.0)}px -> tile {wide_tile}px, against pad "
          f"{eng.pad_for(narrow, 1.0)}px -> {hi_tile}px")

    # -- a superseded render stops ------------------------------------------
    print("\nsuperseded renders stop instead of running to completion")
    polls: list[int] = []

    def cancel_on(nth):
        def f():
            polls.append(1)
            return len(polls) > nth
        return f

    cancelled = True
    try:
        eng.render_image(img, p, 1.0, tile=128, supersample=1,
                         should_cancel=cancel_on(2))
        cancelled = False
    except RenderCancelled:
        pass
    check("cancellation raises rather than returning a partial frame",
          cancelled, f"stopped after {len(polls)} polls")
    polls.clear()
    never = eng.render_image(img, p, 1.0, tile=128, supersample=1,
                             should_cancel=lambda: False)
    plain = eng.render_image(img, p, 1.0, tile=128, supersample=1)
    check("a hook that never fires costs nothing",
          float(np.abs(never - plain).max()) == 0.0,
          f"maxdiff {float(np.abs(never - plain).max()):.2e}")

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

    print("\nupscale (blow a render up to the source's own dimensions)")
    # A small gradient plate rather than noise -- noise has no structure for a
    # round trip to preserve, and a gradient can show an axis swap or an
    # off-by-one a shape-only check would miss.
    gy2, gx2 = np.mgrid[0:60, 0:90].astype(np.float32)
    small_plate = np.stack(
        [gx2 / 89.0, gy2 / 59.0, (gx2 + gy2) / 148.0], -1
    ).astype(np.float32)
    small_plate = np.ascontiguousarray(small_plate)

    up_same = iio.upscale(small_plate, 60, 90)
    check(
        "a no-op at the target size returns the same array",
        up_same is small_plate, "identity, not merely equal",
    )

    big = iio.upscale(small_plate, 240, 360)
    check(
        "upscale hits the exact requested size",
        big.shape == (240, 360, 3), f"got {big.shape}",
    )
    check(
        "upscale stays inside 0..1",
        float(big.min()) >= 0.0 and float(big.max()) <= 1.0,
        f"range {float(big.min()):.3f}..{float(big.max()):.3f} "
        "(bicubic can ring past the source's own range without the clamp)",
    )

    # Matches a direct call with the same arguments -- pins the choice of
    # bicubic/no-antialias/align_corners=False against a silent drift in any
    # one of them, since a symmetric gradient could pass a looser check with
    # any of the three wrong.
    import torch.nn.functional as _F  # noqa: E402
    t = torch.from_numpy(small_plate).permute(2, 0, 1).unsqueeze(0)
    ref = _F.interpolate(t, size=(240, 360), mode="bicubic", align_corners=False)
    ref = ref.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    d = float(np.abs(big - ref).max())
    check(
        "upscale matches a direct bicubic call", d < 1e-6,
        f"max delta {d:.2e}",
    )

    # A downscale/upscale round trip cannot recover detail, but on a smooth
    # gradient with no fine structure it should land close to the original --
    # a coarse sanity check that nothing is transposed, flipped or scaled
    # wrong, not a claim about image quality.
    down = iio.downscale(small_plate, 0.5)
    back_up = iio.upscale(down, 60, 90)
    d = float(np.abs(back_up - small_plate).max())
    check(
        "a downscale/upscale round trip approximates a smooth plate",
        d < 0.05, f"max delta {d:.2e}",
    )

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
