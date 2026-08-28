"""Prescaling Source: the input resampled to a fixed working resolution

Step -3, above everything including Normalize -- and the only section that is
not a stage. It resamples the frame the pipeline is *handed* rather than the
pixels in it, so it appears nowhere in `render()` and reserves nothing in
`pad_for`; the resample lives in `server/models/upload.py`, above the engine.

That placement is the reason this module looks unlike the other seventeen. There
is no transfer function to measure and no seam that could open, because no stage
ever learns the frame is not the file. What there *is* to get wrong is
bookkeeping, and every one of these checks is aimed at a specific way it could
be got wrong silently:

* a resample per render instead of per photograph -- correct pixels, a slider
  that stutters, and nothing on screen to say why;
* the parameters rescaled for the file's size while the image is rendered at
  another -- two corrections for one problem, and a plausible wrong picture;
* a checkpoint shared between two working resolutions of one photograph, which
  is the worst failure this codebase has;
* an aspect ratio drifting by a pixel, which shows up nowhere until the
  before/after wipe parts company at one edge.

**Everything here is deliberately sub-megapixel, and that is not laziness.**
Every branch this module tests -- the target arithmetic, the identity paths, the
cache, `proxy_scale`, the checkpoint id -- is scale-free, so a 0.24MP source
exercises exactly the code a 50MP one would. The first version used realistic
22MP and 24MP scenes and held about 1.1GB of them live at once, which is fine on
its own and **crashed the machine** under the parallel runner, where this lands
beside `edges` and `global_layers` holding fixtures of their own. A check module
is a unit of parallelism, so its peak footprint is multiplied by the pool: big
arrays here have to be earned, and none of these need one. Where a branch really
does need a frame wider than `PROXY_LONG_EDGE`, one 4MP frame is built and
dropped again.
"""

from __future__ import annotations

import time

import numpy as np

from server import imageio as iio
from server import params as P
from server.controllers import export as export_ctl
from server.engine.grain_engine import GrainEngine
from server.models import upload as up_model
from server.models.export_job import JOBS
from server.models.upload import PROXY_LONG_EDGE, Upload, prescale_dims
from server.services.render import render_tier
from tests.harness import Ctx, check, suite
from tests.scene import scene


def _upload(h: int, w: int, uid: str = "t") -> Upload:
    """An `Upload` built straight from a scene, bypassing the HTTP layer."""
    return Upload(uid, f"{uid}.jpg", scene(h, w))


@suite("prescale", "prescaling the source to a fixed working resolution")
def run(cx: Ctx) -> None:
    # -- the arithmetic ------------------------------------------------------
    # Only the pixel count is chosen; the aspect ratio is the photograph's. The
    # ratio matters more than it looks: `/api/source` still serves the untouched
    # photograph at its original resolution, and the client lines the two up
    # under the wipe by scaling both into one box, so a ratio that drifted would
    # show as the layers parting at an edge rather than as an error.
    print("\nprescale geometry (megapixels chosen, aspect ratio preserved)")
    shapes = [(4000, 6000), (5792, 8688), (2000, 3000), (3000, 3000),
              (800, 6000), (4032, 3024), (1080, 1920), (2848, 4288)]
    targets = (24.0, 12.0, 61.0, 1.0, 120.0)
    worst_mp, worst_mp_at = 0.0, None
    worst_ax, worst_ax_at = 0.0, None
    for h, w in shapes:
        for target in targets:
            nh, nw = prescale_dims(h, w, target)
            k = (target * 1e6 / (h * w)) ** 0.5
            err = abs(nh * nw / 1e6 - target) / target
            # Both axes take the *same* factor and each is rounded to a whole
            # pixel, so the only deviation either one may carry is that
            # rounding. Asserted per axis rather than as a ratio error, because
            # this is exactly what the function promises and it is scale-free --
            # a ratio tolerance would have to be loosened for a panorama, where
            # half a pixel on the short axis is a larger share of it.
            axis = max(abs(nh - h * k), abs(nw - w * k))
            if err > worst_mp:
                worst_mp, worst_mp_at = err, (h, w, target)
            if axis > worst_ax:
                worst_ax, worst_ax_at = axis, (h, w, target)
    check(
        "lands on the requested megapixels", worst_mp < 0.002,
        f"worst {worst_mp * 100:.3f}% at {worst_mp_at} over "
        f"{len(shapes) * len(targets)} shape/target pairs",
    )
    check(
        "neither axis deviates by more than the rounding",
        worst_ax <= 0.5 + 1e-9,
        f"worst {worst_ax:.4f}px at {worst_ax_at}",
    )
    check(
        "a bigger target is a bigger frame",
        prescale_dims(4000, 6000, 61.0) > prescale_dims(4000, 6000, 24.0)
        > prescale_dims(4000, 6000, 12.0),
        "12MP < 24MP < 61MP on the same photograph",
    )

    # -- off, and already-at-size, are the *same object* --------------------
    # Not merely equal pixels. `Upload.at()` hands the upload itself back, so
    # "Prescaling off behaves exactly as it did before this existed" is
    # structural: there is no second code path to keep in step, because there is
    # no second object. The already-at-size case matters for the same reason
    # plus one more -- an identity `Frame` would carry a `@24mp` checkpoint id
    # and split one photograph's cache in two for no gain.
    print("\nthe identity paths return the upload itself")
    # 0.24MP, and every target below is scaled to match -- see the note at the
    # top of this module. `at()` branches on the ratio, never on the size.
    up = _upload(400, 600)
    check("prescaling off is the upload", up.at(None) is up, "at(None) is up")
    check(
        "a photograph already at the target is the upload",
        up.at(0.24) is up and prescale_dims(400, 600, 0.24) == (400, 600)
        and up.frame is None,
        "a source already at the target is up, and no Frame is built",
    )
    check(
        "a target of zero is off", up.at(None) is up
        and up_model.prescale_target({"prescale": 1.0, "prescale_mp": 0.0}) is None
        and up_model.prescale_target({"prescale": 0.0, "prescale_mp": 24.0}) is None
        and up_model.prescale_target({"prescale": 1.0, "prescale_mp": 24.0}) == 24.0,
        "the switch and a zero target are the same answer",
    )

    # -- the resample happens once ------------------------------------------
    # This is the check the feature turns on, and it deliberately measures *that
    # the work did not happen* rather than that the answer is right -- the same
    # construction the Global Grain cache checks use, for the same reason: a
    # second resample renders a perfectly plausible frame, so a correctness-only
    # test passes while every slider drag pays for a full-frame interpolation.
    print("\nthe photograph is resampled once, not once per render")
    calls: list[tuple[int, int]] = []
    real = iio.resize_to

    def counting(arr, h, w, device=None):
        calls.append((h, w))
        return real(arr, h, w, device)

    iio.resize_to = counting
    try:
        fr = up.at(1.0)
        again = up.at(1.0)
        for _ in range(4):
            _ = up.at(1.0).proxy
        proxy_calls = len(calls)
        for _ in range(3):
            _ = up.at(1.0).arr
        full_calls = len(calls) - proxy_calls
    finally:
        iio.resize_to = real
    check(
        "the frame is cached on the upload", again is fr and up.frame[0] == 1.0,
        "at(1MP) twice is one Frame",
    )
    check(
        "four proxy reads are one resample", proxy_calls == 1,
        f"{proxy_calls} resample(s) for 4 reads, at {calls[0]}",
    )
    check(
        "three full reads are one resample", full_calls == 1,
        f"{full_calls} resample(s) for 3 reads",
    )
    check(
        "changing the target rebuilds", up.at(0.5) is not fr
        and up.at(1.0) is not fr,
        "a single slot, so switching away and back is a fresh Frame",
    )

    # -- the frame is what a real photograph of that size would be ----------
    # `proxy_scale` is the whole point: it stops being a function of what came
    # out of the camera, so the proxy's divergence from the 1:1 render is the
    # same on every photograph. And the proxy comes from the source in ONE pass
    # -- going via `arr` would upscale a small photograph to 24MP and throw
    # 23MP of it away again.
    print("\nthe frame is indistinguishable from an upload of that size")
    # `Frame.__init__` is pure arithmetic -- dimensions, `proxy_scale` and the
    # id, no pixels -- so the whole relationship can be asserted without
    # materialising anything. Checked over a spread of targets rather than one,
    # because the property is that `proxy_scale` follows the *prescaled* size
    # and a single case cannot tell that from a coincidence.
    worst = 0.0
    for target in (0.5, 4.0, 16.0, 61.0):
        f = up.at(target)
        want = min(1.0, PROXY_LONG_EDGE / float(max(f.h, f.w)))
        worst = max(worst, abs(f.proxy_scale - want))
    check(
        "proxy_scale is measured from the prescaled size, not the file's",
        worst < 1e-12 and up.at(16.0).proxy_scale < 0.5 < up.proxy_scale,
        f"worst {worst:.1e} over 4 targets; a 16MP frame proxies at "
        f"{up.at(16.0).proxy_scale:.3f} where the 0.24MP file is "
        f"{up.proxy_scale:.3f}",
    )
    check(
        "the frame names its working resolution",
        up.at(24.0).id == f"{up.id}@24mp" and up.at(0.5).id.endswith("@0.5mp")
        and up.at(24.0).id != up.id,
        f"{up.at(24.0).id}",
    )
    # The one place a real frame is needed: `proxy_scale` below 1 means the long
    # edge is past PROXY_LONG_EDGE, and the claim is that those 2400px come from
    # the *source* in one pass rather than via a materialised full frame. 4MP is
    # the smallest target on this source that gets there. Dropped immediately.
    big = up.at(4.0)
    proxy_shape = big.proxy.shape[:2]
    built_arr = big._arr is not None
    check(
        "the proxy is that scale applied to the frame",
        proxy_shape == (round(big.h * big.proxy_scale),
                        round(big.w * big.proxy_scale)),
        f"{proxy_shape[1]}x{proxy_shape[0]} at {big.proxy_scale:.3f} "
        f"of a {big.w}x{big.h} frame",
    )
    check(
        "an upscaled frame proxies from the source, not from itself",
        not built_arr and max(proxy_shape) == PROXY_LONG_EDGE,
        f"{proxy_shape[1]}x{proxy_shape[0]} proxy built with no full array",
    )
    del big
    up.frame = None
    check(
        "metering is the photograph's, not the frame's",
        up.at(1.0).norm_stats() is up.norm_stats(),
        "a resample must not move the colour correction",
    )

    # -- the double-scaling guard -------------------------------------------
    # Prescaling and `reference_mp` are the same correction from opposite ends.
    # Applying both is not twice as good, it is wrong: the image is rendered at
    # 24MP with lengths sized for the 22MP file it no longer is.
    print("\nprescaling and preset rescaling must not both fire")
    # The preset's authored size and the target are the same number, which is
    # the case that matters: it is what every shipped preset does. That the
    # number is 1MP rather than 24 changes nothing -- `scale_factor` sees a
    # ratio -- and keeps this off a 24MP array.
    body = {"params": {"prescale": 1, "prescale_mp": 1.0}, "reference_mp": 1.0}
    f_on, p_on = up_model.params_for(up, body)
    f_off, p_off = up_model.params_for(up, {"params": {"prescale": 0},
                                            "reference_mp": 1.0})
    spatial = [x.key for x in P.PARAMS if x.spatial]
    # Relative, not absolute, and the reason is a real property of the feature
    # rather than a slack tolerance. Pixel dimensions are integers, so a frame
    # prescaled to 1MP is 816x1225 = 0.9996MP -- there is no integer pair at
    # exactly the target on most aspect ratios. The factor therefore lands
    # *within a rounding* of 1.0 rather than on it, and `rescale`'s own
    # `abs(k - 1) < 1e-6` early-out does not catch it. What matters is whether
    # that is perceptible, and it is not: 0.02% of a 1.6px clump is 0.0003px.
    # It shrinks as the frame grows -- a 3:2 frame at 24MP is exactly 24.0MP --
    # so this small source is the worst case, deliberately.
    worst_on = max(abs(p_on[k] / P.DEFAULTS[k] - 1.0)
                   for k in spatial if P.DEFAULTS[k])
    moved_off = [k for k in spatial if abs(p_off[k] - P.DEFAULTS[k]) > 1e-9]
    check(
        "prescaling to the preset's own size scales nothing perceptible",
        f_on is not up and worst_on < 1e-3,
        f"factor {P.scale_factor(1.0, f_on.w * f_on.h / 1e6):.6f}, worst length "
        f"moved {worst_on * 100:.3f}% over {len(spatial)} lengths",
    )
    check(
        "without it the same preset still rescales",
        f_off is up and len(moved_off) > 10,
        f"factor {P.scale_factor(1.0, up.w * up.h / 1e6):.4f}, "
        f"{len(moved_off)} of {len(spatial)} lengths moved",
    )
    check(
        "the rescale is measured against the frame, not the file",
        # The sharpest form of it: prescale to something *other* than the
        # preset's size and the factor has to follow the frame. 4x the
        # megapixels is 2x the width, so every length has to double.
        abs(P.scale_factor(1.0, 4.0) - 2.0) < 1e-9
        and abs(up_model.params_for(up, {"params": {"prescale": 1, "prescale_mp": 4.0},
                                        "reference_mp": 1.0})[1]["grain_size"]
                - P.DEFAULTS["grain_size"] * 2.0) < 1e-3,
        "a 4x-megapixel target doubles every length against the same preset",
    )
    up.frame = None

    # -- no checkpoint is shared between two working resolutions ------------
    # The engine's cache keys on the id, `h`, `w` and the scale, so this is
    # over-determined -- which is the right amount for a cache whose stale hit
    # renders a plausible but wrong *photograph*. Asserted through `render_tier`
    # rather than against `_ckpt_key` directly, because the id is assembled
    # there and that assembly is the part that could go wrong.
    print("\nswitching resolution cannot hit another resolution's checkpoint")
    tiny = _upload(400, 600, "c")
    warm = GrainEngine(cx.dev)
    # 1MP is the slider's floor, so this is a target a user can actually ask
    # for -- and on a 0.24MP source it is an upscale, which is the direction
    # that did not exist before this section.
    q = P.sanitize({"prescale": 1, "prescale_mp": 1.0, "intensity": 40.0})
    a = render_tier(tiny.at(None), q, 1.0, True).shape[:2]
    b = render_tier(tiny.at(1.0), q, 1.0, True).shape[:2]
    check(
        "the two frames are different sizes to begin with",
        a == (400, 600) and b != a,
        f"{a[1]}x{a[0]} vs {b[1]}x{b[0]}",
    )
    order: list[float] = []
    for target in (None, 1.0, None, 1.0):
        f = tiny.at(target)
        got = ENG_render(warm, f, q)
        want = ENG_render(GrainEngine(cx.dev), f, q)
        order.append(float(np.abs(got - want).max()))
    check(
        "a warm cache renders each resolution as a cold one does",
        max(order) == 0.0 and warm.ckpt.hits > 0,
        f"worst {max(order):.2e} over four switches, {warm.ckpt.hits} cache hits",
    )
    del warm
    tiny.frame = None

    # -- the resample is outside the engine, so nothing seams ---------------
    # This cannot fail by construction: the engine is handed an array and never
    # learns where it came from. It is here because *that claim* is the thing
    # worth pinning -- the same reason `verify.py` tiles dust and hair to prove
    # they reserve nothing in `pad_for`. A future version that resampled inside
    # a stage would break invariant 1, and this is the check that would say so.
    print("\na prescaled frame tiles exactly like any other")
    # `at()` directly rather than through a sanitized `prescale_mp`, and below
    # the slider's own 1MP floor: what is being tiled is a *frame*, the engine
    # never reads the target, and a 0.09MP one keeps 70 tiles' worth of stress
    # inside this module's few seconds.
    seam_src = _upload(300, 200, "m").at(0.09)
    sp = P.sanitize({"intensity": 45.0, "halation": 0.8, "halation_radius": 24.0,
                     "micro_blur": 1.2, "sharpen": 6.0})
    one = cx.eng.render_image(seam_src.proxy, sp, seam_src.proxy_scale,
                              tile=4096, supersample=1)
    many = cx.eng.render_image(seam_src.proxy, sp, seam_src.proxy_scale,
                               tile=128, supersample=1)
    d = float(np.abs(one - many).max())
    # 2e-3, which is `tests/checks/tiling.py`'s tolerance and deliberately not a
    # tighter one invented here. Padded tile windows are clamped at the image
    # border and summed in a different order, so tile independence has always
    # been a float-noise bound rather than bit-exactness -- the same parameters
    # measure 1.53e-04 on an ordinary upload of this size. Asserting 0.0 here
    # would be asserting something the invariant never claimed, and it would
    # fail for a reason that has nothing to do with prescaling.
    check(
        "tiled equals single-pass on a prescaled frame", d < 2e-3,
        f"{d:.2e} at tile 128 on a "
        f"{seam_src.proxy.shape[1]}x{seam_src.proxy.shape[0]} frame",
    )

    # -- the schema contract -------------------------------------------------
    # Three ways the section could be wired wrongly and still look fine.
    print("\nthe controls are declared the way the plumbing assumes")
    mp = P.PARAM_BY_KEY["prescale_mp"]
    out = P.PARAM_BY_KEY["prescale_output"]
    check(
        "the target is not a length",
        not mp.spatial,
        # If it were, `rescale` would multiply the target by a factor derived
        # from the target -- circular, and the working size would drift every
        # render while every individual number looked reasonable.
        "prescale_mp is megapixels, so `rescale` must not touch it",
    )
    check(
        "the switch is what Original turns off",
        "prescale" in P.NEUTRAL_ZERO and "prescale_mp" not in P.NEUTRAL_ZERO
        and "prescale_output" not in P.NEUTRAL_ZERO
        and P.neutral_values()["prescale"] == 0.0
        and P.neutral_values()["prescale_mp"] == mp.default,
        "prescale zeroes, the target and the export size keep their setting",
    )
    check(
        "the export menu's index 0 is the prescaled size",
        out.choices == ("Prescaled size", "Photo's own size"),
        "reordering these silently rewrites every saved preset",
    )
    check(
        "the section is first in GROUPS", P.GROUPS[0] == "Prescaling Source",
        f"GROUPS[:2] = {P.GROUPS[:2]}",
    )
    stamped = [q["name"] for q in P.load_presets()
               if q["values"].get("prescale") == 1.0
               and q["values"].get("prescale_mp") == 24.0]
    allp = P.load_presets()
    check(
        "every shipped preset is stamped at 24MP",
        len(stamped) == len(allp),
        f"{len(stamped)} of {len(allp)} presets carry prescale 1 at 24MP",
    )

    # -- the file is written at the size the panel promised -----------------
    # The one decision `prescale_output` makes, and it is invisible on screen:
    # the preview is browser-scaled either way, so only the written file differs.
    print("\nthe exported file's dimensions follow prescale_output")
    ex = _upload(400, 600, "e")
    up_model.UPLOADS[ex.id] = ex
    frame_hw = prescale_dims(ex.h, ex.w, 1.0)
    for out_mode, want in ((0, frame_hw), (1, (ex.h, ex.w))):
        body = {
            "id": ex.id, "format": "png8", "supersample": 1, "quality": 95,
            "params": {"prescale": 1, "prescale_mp": 1.0,
                       "prescale_output": out_mode, "intensity": 30.0},
        }
        job_id = export_ctl.export(body)["job"]
        deadline = time.time() + 120
        while JOBS[job_id]["status"] not in ("done", "error") and time.time() < deadline:
            time.sleep(0.05)
        job = JOBS[job_id]
        # `blob.data` rather than a `bytes` on the job: a finished export is
        # written to the SSD and streamed from there (`engine.diskcache.Blob`),
        # so this reads the file back the way the download route serves it. That
        # is the stronger check of the two anyway -- it proves the bytes reached
        # the disk intact, which reading them out of a dict never could.
        blob = job.get("blob")
        arr = (iio.load_image(blob.data)
               if job["status"] == "done" and blob is not None else None)
        check(
            f"prescale_output {out_mode} writes {want[1]}x{want[0]}",
            job["status"] == "done" and (job["height"], job["width"]) == want
            and arr is not None and arr.shape[:2] == want,
            f"job says {job['width']}x{job['height']}, file is "
            f"{'-' if arr is None else f'{arr.shape[1]}x{arr.shape[0]}'}"
            f"{'' if job['status'] == 'done' else ' :: ' + str(job.get('error'))}",
        )
        # Both modes tag, and mode 1 is why the tag exists at all: it writes
        # the photograph's own dimensions, so size alone cannot tell it from a
        # plain un-prescaled export of the same photograph.
        check(
            f"prescale_output {out_mode} names the file apart",
            "_pre1mp" in job["filename"],
            job["filename"],
        )
    up_model.UPLOADS.pop(ex.id, None)


def ENG_render(eng: GrainEngine, fr, q: dict) -> np.ndarray:
    """`render_tier`'s full tier on one engine, for the checkpoint comparison.

    Not `render_tier` itself: that reaches for the module-level `ENGINE`, and
    these checks need two engines -- one warm, one cold -- rendering the
    identical call.
    """
    tile = eng.tile_for(q, 1.0, fr.h, fr.w, 1.0)
    return eng.render_image(fr.arr, q, 1.0, tile=tile, supersample=1,
                            checkpoint_id=f"{fr.id}:full")
