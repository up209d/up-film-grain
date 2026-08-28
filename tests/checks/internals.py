"""bit-exact performance rewrites, the texture cache, tile size and cancellation

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
import os
import torch
from server import params as P
from tests.checks.normalize import _PROBE as _NORM_PROBE
from tests.refs import blur_ref
from tests.refs import grain_ref
from tests.refs import lattice_ref
from tests.refs import span_ref
from tests.scene import scene
from server.engine import (
    GrainEngine, RenderCancelled, _BLUR_EXACT_MAX_SIGMA, _blur, _grain_points,
    _lat_span, _lattice_np,
)
from tests.harness import Ctx, check, suite


@suite("internals", "bit-exact performance rewrites, the texture cache, tile size and cancellation")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    dev = cx.dev
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

    span_bad = 0
    span_n = 0
    for cell in (0.8, 1.0, 1.6, 2.0, 2.22, 3.2, 6.0, 110.0, 900.0):
        for pl, ph in ((1, 2), (0, 0), (2, 2)):
            for n in (1, 17, 512, 1536, 3072, 4800):
                for origin in (0.0, 1.0, 7.0, 13.0, 178.0, 1023.0, 4096.0):
                    span_n += 1
                    if _lat_span(n, origin, cell, pl, ph) != span_ref(
                        n, origin, cell, pl, ph, dev
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

    # -- the large-sigma blur is exact below its threshold --------------------
    #
    # `_blur` drops to a decimated grid above `_BLUR_EXACT_MAX_SIGMA`, which is
    # the one approximation in the pipeline. **The threshold is the whole safety
    # argument**, so it is checked from both sides: everything below it must be
    # bit-identical to the exact convolution, and everything above it must stay
    # inside a stated tolerance rather than merely "look blurred".
    #
    # The bit-identity half is the important one. It is what lets every other
    # check in this suite keep asserting 0.00e+00 -- if the exact path ever
    # drifted, the failure would show up here as a threshold bug rather than as
    # a dozen unrelated tolerances quietly needing to be loosened.
    tb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev)
    exact_worst = 0.0
    for s in (0.5, 2.0, 8.0, 20.0, _BLUR_EXACT_MAX_SIGMA - 0.1):
        exact_worst = max(exact_worst, float(
            (_blur(tb, s, (0.0, 0.0)) - blur_ref(tb, s)).abs().max()
        ))
    # Passing an origin is what *enables* decimation, so these pass one -- a
    # caller that omits it gets the exact path at every sigma, which is the
    # default and is separately worth stating.
    no_origin = float((_blur(tb, 80.0) - blur_ref(tb, 80.0)).abs().max())
    check("without an origin the blur is exact at any sigma",
          no_origin == 0.0, f"sigma 80 with no origin: {no_origin:.2e}")
    check("below the threshold the blur is the exact convolution",
          exact_worst == 0.0,
          f"worst deviation {exact_worst:.2e} over sigma 0.5-{_BLUR_EXACT_MAX_SIGMA - 0.1}")

    # Above it, measured as a share of the blurred field's own peak -- an
    # absolute delta says nothing useful about a field whose amplitude falls as
    # the kernel widens.
    #
    # 1.5% is a real bound rather than a rubber stamp, and it is set by *this
    # scene* rather than by the pipeline: the fixture is deliberately built out
    # of hard synthetic edges (a black block, a 0.99 disc), which is the worst
    # possible input for a decimated resample. Measured here it peaks at 1.11%
    # at sigma 80; on the scratch mark field the stage actually blurs it is
    # 0.7%, and on noise 0.22-0.64%. For scale, the whole scratch layer moves
    # 0.06% of a frame's pixels by more than one 8-bit level, so an error of
    # 1% *of that layer's peak* is orders below anything visible.
    decim_worst = 0.0
    for s in (_BLUR_EXACT_MAX_SIGMA + 0.1, 40.0, 54.0, 80.0, 120.0):
        ref_b = blur_ref(tb, s)
        peak = float(ref_b.abs().max())
        decim_worst = max(decim_worst, float(
            (_blur(tb, s, (0.0, 0.0)) - ref_b).abs().max()
        ) / max(peak, 1e-9))
    check("above the threshold it stays within 1.5% of peak",
          decim_worst < 0.015,
          f"worst {decim_worst * 100:.2f}% of peak over sigma "
          f"{_BLUR_EXACT_MAX_SIGMA + 0.1}-120 on the hard-edged fixture")

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
                       ("global_chroma", -0.5), ("global_mottle", 0.6)):
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

    # A superseded parameter state can never be asked for again, so it is
    # dropped on sight rather than left for memory pressure to evict. Two
    # generations are kept -- one would make dragging a slider back and forth
    # rebuild every time.
    #
    # Tested as *held bytes*, not as a hit rate: the failure mode is the cache
    # quietly holding the whole budget in fields nothing can reach, which
    # renders perfectly and simply wastes the allowance.
    #
    # Those bytes are on the SSD rather than in RAM since 2026-08-29
    # (`engine/diskcache.py`), and the check is unchanged by that on purpose.
    # Generation eviction was never about which medium the waste sat in -- it is
    # about entries no render can ever ask for again -- and a version of this
    # that started tolerating unbounded growth because the growth is cheaper now
    # would be measuring the budget instead of the policy.
    ev_eng = GrainEngine(dev)
    for i in range(6):
        q = P.sanitize({**gp, "global_size": 1.6 + 0.1 * i})
        ev_eng.render_image(gimg, q, 1.0, tile=4096, supersample=1)
    gens = len({k[5:] for k in ev_eng._gg_cache})
    check("superseded cache generations are dropped, not merely aged",
          gens <= 2 and ev_eng.gg_evicted > 0,
          f"{gens} generations resident after 6 distinct parameter states, "
          f"{ev_eng.gg_evicted} entries evicted, "
          f"{ev_eng._gg_cache.nbytes / 1e6:.0f} MB held")

    # And the previous generation really is kept: going back one step must hit.
    back = P.sanitize({**gp, "global_size": 1.6 + 0.1 * 4})
    h0 = ev_eng.gg_hits + ev_eng.gs_hits
    ev_eng.render_image(gimg, back, 1.0, tile=4096, supersample=1)
    check("the previous generation survives, so A/B does not rebuild",
          ev_eng.gg_hits + ev_eng.gs_hits > h0,
          f"stepping back one value hit {ev_eng.gg_hits + ev_eng.gs_hits - h0} entries")

    # -- pipeline checkpoints ------------------------------------------------
    #
    # The failure mode is the worst one in the codebase: a key that misses an
    # input returns a frame rendered from *different parameters*, and it will
    # look like a perfectly good photograph. So this does not test that
    # checkpointing is fast -- it tests that a warm cache is indistinguishable
    # from no cache, one parameter at a time, against an engine that has never
    # seen a checkpoint.
    print("\npipeline checkpoints (a stale hit renders the wrong photograph)")
    ck_eng = GrainEngine(dev)
    cold_eng = GrainEngine(dev)
    cimg = scene(240, 360)
    cbase = P.sanitize({"global_intensity": 9.0, "halation": 0.5,
                        "sharpen": 4.0, "dust": 20.0, "grade_exposure": 0.1,
                        "edge_jitter": 0.4, "intensity": 30.0})
    # Warm it.
    ck_eng.render_image(cimg, cbase, 1.0, tile=4096, supersample=1,
                        checkpoint_id="t:proxy")
    same = ck_eng.render_image(cimg, cbase, 1.0, tile=4096, supersample=1,
                               checkpoint_id="t:proxy")
    ref = cold_eng.render_image(cimg, cbase, 1.0, tile=4096, supersample=1)
    check("a warm checkpoint returns the identical frame",
          float(np.abs(same - ref).max()) == 0.0 and ck_eng.ckpt.hits > 0,
          f"maxdiff {float(np.abs(same - ref).max()):.2e}, "
          f"{ck_eng.ckpt.hits} hits")

    # One parameter from every section, changed against a warm cache. Each must
    # come back exactly as a cold engine renders it. A key that failed to cover
    # an upstream section shows up here as a frame that did not move.
    #
    # `Prescaling Source` is deliberately absent and must stay absent. Its keys
    # never reach a stage -- they resample the frame `render_image` is *handed*,
    # in `models/upload.py` -- so a probe on one would compare two identical
    # renders and pass as a tautology, and it cannot be written the honest way
    # here because the two frames would be different shapes. That boundary is
    # checked in `tests/checks/prescale.py` instead, where the switch is made
    # through `render_tier` and each resolution is compared against its own cold
    # render.
    probes = [
        ("normalize", 1.0),
        ("grade_exposure", 0.35), ("pre_blur", 1.2), ("pre_sharpen", 0.6),
        ("intensity", 55.0), ("grain_size", 2.5), ("lum_low", 0.3),
        ("edge_jitter", 1.4), ("scatter", 0.5), ("micro_blur", 1.1),
        ("aa_strength", 1.5), ("halation", 0.95), ("halation_radius", 60.0),
        ("brightness", -0.3), ("toe", 0.8), ("base_fog", 0.05),
        ("global_intensity", 40.0), ("global_size", 4.0),
        ("sharpen", 12.0), ("dust", 90.0), ("scratches", 4.0), ("seed", 99.0),
    ]
    worst_k, worst_d = None, 0.0
    for k, v in probes:
        q = P.sanitize({**{kk: vv for kk, vv in cbase.items()
                           if kk in P.PARAM_BY_KEY}, k: v})
        # Normalize's six measured floats are not `Param`s -- `sanitize` has
        # never seen them, so they have to be attached the way `params_for`
        # does it. Without them the stage runs with the identity correction and
        # the probe would pass by changing nothing, which is the one way a
        # checkpoint probe can lie.
        if q["normalize"] >= 0.5:
            q.update(_NORM_PROBE)
        got = ck_eng.render_image(cimg, q, 1.0, tile=4096, supersample=1,
                                  checkpoint_id="t:proxy")
        want = cold_eng.render_image(cimg, q, 1.0, tile=4096, supersample=1)
        d = float(np.abs(got - want).max())
        if d > worst_d:
            worst_k, worst_d = k, d
    check("every section's parameters invalidate what they should",
          worst_d == 0.0,
          f"worst {worst_k or '-'} at {worst_d:.2e} over {len(probes)} probes")

    # Two photographs of the same size with the same parameters must not share
    # a checkpoint -- the id is what separates them, and it is the one part of
    # the key that cannot be derived from `p`.
    other = np.clip(cimg * 0.6 + 0.2, 0.0, 1.0)
    a2 = ck_eng.render_image(other, cbase, 1.0, tile=4096, supersample=1,
                             checkpoint_id="u:proxy")
    b2 = cold_eng.render_image(other, cbase, 1.0, tile=4096, supersample=1)
    check("a different image does not reuse another's checkpoint",
          float(np.abs(a2 - b2).max()) == 0.0,
          f"maxdiff {float(np.abs(a2 - b2).max()):.2e}")

    # Warm cache plus tiling: the key carries absolute (y0, x0), so a tiled
    # render must not pick up a neighbouring tile's frame.
    t_a = ck_eng.render_image(cimg, cbase, 1.0, tile=4096, supersample=1,
                              checkpoint_id="t:proxy")
    t_b = ck_eng.render_image(cimg, cbase, 1.0, tile=96, supersample=1,
                              checkpoint_id="t:proxy")
    check("checkpoints are keyed on absolute coordinates",
          float(np.abs(t_a - t_b).max()) < 2e-3,
          f"tiled vs whole-image, warm: {float(np.abs(t_a - t_b).max()):.2e}")

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

    # A *single-tile* render must be interruptible too, and this is the check
    # that would have caught the real bug. `tile_for` returns 2400 for a 2400px
    # proxy, so every live preview takes the single-pass branch -- where the hook
    # was polled exactly once, before the pass, and then not again. Measured on
    # `SuperPortra`: one poll in 7.91s on the GPU and 21.5s on the CPU, all of it
    # spent on a frame the client had already abandoned. Polls now happen at
    # every stage boundary inside `render`.
    #
    # Asserted as "many more than one" rather than an exact count, so adding or
    # removing a pipeline stage does not fail this.
    polls.clear()
    eng.render_image(img, p, 1.0, tile=4096, supersample=1,
                     should_cancel=cancel_on(10 ** 9))
    check("a single-tile render polls inside the pipeline, not just before it",
          len(polls) > 5, f"{len(polls)} polls in one untiled pass (was 1)")

    polls.clear()
    never = eng.render_image(img, p, 1.0, tile=128, supersample=1,
                             should_cancel=lambda: False)
    plain = eng.render_image(img, p, 1.0, tile=128, supersample=1)
    check("a hook that never fires costs nothing",
          float(np.abs(never - plain).max()) == 0.0,
          f"maxdiff {float(np.abs(never - plain).max()):.2e}")
