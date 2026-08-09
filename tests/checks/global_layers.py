"""the source-masked global layers and the blend modes

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
import torch
from server import params as P
from server.engine import (
    _grain_delta, _source_masks,
)
from tests.harness import Ctx, check, suite


@suite("global_layers", "the source-masked global layers and the blend modes")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 5b-i-c. the source-masked global layers ------------------------------
    # Four more layers of the same grain on four more seeds, each multiplied by
    # an envelope read off the picture. They ship at 0, so every
    # default-parameter check above renders straight past them -- everything
    # they could break has to be pinned here or not at all.
    print("\nglobal grain, source-masked layers")

    # The masks first, as arithmetic, because two of the three claims they make
    # are exact and a render can only ever say "about right" about them.
    gs_cols = {
        "white": (1.0, 1.0, 1.0), "grey": (0.5, 0.5, 0.5),
        "black": (0.0, 0.0, 0.0), "bright red": (0.9, 0.1, 0.1),
        "dark red": (0.3, 0.05, 0.05), "foliage": (0.3, 0.55, 0.2),
        "sky": (0.4, 0.6, 0.9),
    }

    def gs_mask(rgb):
        t = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
        return [float(x) for x in _source_masks(t)]

    # A hue mask, not a channel value. This is the whole difference between the
    # set as built and the first attempt at it: `mask = R` puts all three colour
    # layers at full strength on white and on grey, where they pile up into a
    # brightness mask wearing three sliders. Neutral has to be exactly zero.
    for name in ("white", "grey", "black"):
        m = gs_mask(gs_cols[name])
        check(f"the colour masks are silent on {name}", max(m[:3]) == 0.0,
              f"r/g/b {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}")
    # ... and it grows with hue dominance *and* with brightness, which is what
    # `C - max(others)` says: it factors into "how red" x "how bright".
    mr_bright, mr_dark = gs_mask(gs_cols["bright red"]), gs_mask(gs_cols["dark red"])
    check("a red area masks in red", abs(mr_bright[0] - 0.8) < 1e-6,
          f"bright red -> {mr_bright[0]:.3f}")
    check("and a darker red masks in less", mr_dark[0] < mr_bright[0] * 0.5,
          f"{mr_dark[0]:.3f} against {mr_bright[0]:.3f}")
    check("foliage picks green, sky picks blue",
          gs_mask(gs_cols["foliage"])[1] > 0.2 and gs_mask(gs_cols["sky"])[2] > 0.2,
          f"green {gs_mask(gs_cols['foliage'])[1]:.3f}, "
          f"blue {gs_mask(gs_cols['sky'])[2]:.3f}")
    # Mutually exclusive by construction -- only one channel can be the largest
    # -- so the three can never stack into a hot patch however many are up.
    # Asserted over a random volume rather than the seven samples above.
    rr = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(9))
    mm = _source_masks(rr)
    both = ((mm[0] > 0).float() + (mm[1] > 0).float() + (mm[2] > 0).float()).max()
    check("at most one colour mask fires per pixel", float(both) <= 1.0,
          f"max simultaneous {float(both):.0f}")

    # Lightness is a mid-tone bell, not a ramp: loudest at grey, gone at both
    # ends. A ramp would pass a "follows the picture" test just as well, so the
    # shape is measured at both ends rather than the middle only.
    ramp = torch.linspace(0.0, 1.0, 257).view(1, 1, -1, 1).expand(1, 3, -1, 1)
    bell = _source_masks(ramp.contiguous())[3].flatten()
    check("the lightness bell is zero at black and white",
          float(bell[0]) == 0.0 and float(bell[-1]) == 0.0,
          f"{float(bell[0]):.4f} at black, {float(bell[-1]):.4f} at white")
    check("and peaks at mid grey", abs(float(bell[128]) - 1.0) < 1e-5
          and int(bell.argmax()) == 128, f"peak {float(bell.max()):.4f} "
          f"at L={int(bell.argmax()) / 256:.3f}")
    rise = bell[:129]
    fall = bell[128:]
    check("rising to it and falling away, monotonically",
          bool((rise[1:] >= rise[:-1]).all()) and bool((fall[1:] <= fall[:-1]).all()),
          "monotone on both sides")
    check("and it is symmetric about grey",
          float((bell - bell.flip(0)).abs().max()) < 1e-6,
          f"max asymmetry {float((bell - bell.flip(0)).abs().max()):.2e}")

    # The pivot moves that peak. Measured as *where the maximum is*, not as a
    # mean: a mean-only test cannot tell a moved bell from a taller one, and the
    # two ends have to stay at zero either way or the layer starts graining
    # solid black. Both directions, because a sign error passes one of them.
    for pv, want in ((0.25, 64), (0.75, 192)):
        pb = _source_masks(ramp.contiguous(), pv)[3].flatten()
        check(f"the bell peaks at the pivot ({pv})",
              int(pb.argmax()) == want and abs(float(pb[want]) - 1.0) < 1e-5
              and float(pb[0]) == 0.0 and float(pb[-1]) == 0.0,
              f"peak {float(pb.max()):.4f} at L={int(pb.argmax()) / 256:.3f}, "
              f"ends {float(pb[0]):.4f}/{float(pb[-1]):.4f}")
    # And the default is the shape that shipped before the control existed.
    check("pivot 0.5 is the original mid-tone bell",
          float((_source_masks(ramp.contiguous(), 0.5)[3].flatten()
                 - bell).abs().max()) == 0.0, "bit-identical")

    # A four-patch plate: a red, a mid grey, a dark and a light. Between them
    # they separate every claim the four layers make -- and in particular the
    # dark *and* light patches together are what tell a mid-tone bell from a
    # plain brightness ramp, which the dark one alone cannot.
    #
    # Levels chosen so that nothing clips at the final `clamp(0, 1)`: the flat
    # layer at 20 swings +-0.076 and a source layer at 60 adds up to +-0.228 of
    # its mask on top, so a patch at 0.88 with a bell of 0.145 peaks at 0.989.
    # Clipping would not fail these checks loudly -- it quietly eats one tail
    # and every sigma below would be measuring the clamp instead of the layer.
    gs_plate = np.zeros((240, 480, 3), dtype=np.float32)
    gs_plate[:, :120] = (0.65, 0.25, 0.25)
    gs_plate[:, 120:240] = (0.50, 0.50, 0.50)
    gs_plate[:, 240:360] = (0.12, 0.12, 0.12)
    gs_plate[:, 360:] = (0.88, 0.88, 0.88)
    GS_RED = (slice(None), slice(20, 100))
    GS_GREY = (slice(None), slice(140, 220))
    GS_DARK = (slice(None), slice(260, 340))
    GS_LIT = (slice(None), slice(380, 460))

    # The baseline carries the *flat* layer, and it has to carry something:
    # `render_image` hands a neutral parameter set straight back without
    # rendering (`P.is_neutral`), so a zeroed baseline would be the input array
    # while every layer under test went through the pipeline -- and the two
    # differ by a few ULP of colour round trip, which is not what any check here
    # is about. The flat layer is bit-identical on both sides of every
    # subtraction below, so it cancels exactly. It is also how these layers are
    # meant to be used.
    def gs_render(**over: float) -> np.ndarray:
        base = {k: 0.0 for k in P.NEUTRAL_ZERO}
        base.update({"global_intensity": 20.0, "global_opacity": 1.0,
                     "global_size": 3.0, "global_size_max": 3.0})
        base.update(over)
        return eng.render_image(gs_plate, P.sanitize(base), 1.0, tile=1024,
                                supersample=1).astype(np.float64)

    gs_off = gs_render()
    gs_r = gs_render(global_src_r=60.0) - gs_off
    gs_g = gs_render(global_src_g=60.0) - gs_off
    gs_l = gs_render(global_src_l=60.0) - gs_off

    # **The colour names are the mask, never the output channel**, and this is
    # the check that says so. An earlier build of this set confined each layer
    # to its own channel; it renders as something perfectly plausible and is a
    # different feature. All three channels have to move together.
    ch = [float(np.abs(gs_r[GS_RED + (c,)]).max()) for c in range(3)]
    check("Source Red writes to all three channels", min(ch) > 0.0,
          f"max |delta| per channel {ch[0]:.4f}/{ch[1]:.4f}/{ch[2]:.4f}")
    # At chroma 0 that means the same value in all three -- one field, three
    # channels, and a mask that is one scalar per pixel. Green and blue sit at
    # the same level on this patch, so for them it is exact to the bit; red sits
    # at 0.65, where float32 rounds the sum on a coarser grid, so it agrees to
    # 1e-7 rather than to 0. This is the check that would catch the mask being
    # applied per channel, which is the mistake the naming invites.
    exact = float(np.abs(gs_r[GS_RED + (1,)] - gs_r[GS_RED + (2,)]).max())
    spread = float(np.abs(gs_r[GS_RED + (0,)] - gs_r[GS_RED + (2,)]).max())
    check("and at chroma 0 it is monochrome", exact == 0.0 and spread < 1e-6,
          f"green/blue spread {exact:.2e}, red {spread:.2e}")
    # Raise chroma and it must decorrelate, like every other layer in the
    # section -- the four take that slider too.
    gs_rc = gs_render(global_src_r=60.0, global_chroma=1.0) - \
        gs_render(global_chroma=1.0)
    spread = float(np.abs(gs_rc[..., 0] - gs_rc[..., 2])[GS_RED].max())
    check("Chroma Grain reaches the source layers", spread > 0.01,
          f"max channel spread {spread:.4f}")

    # The mask in a render: loud where the picture is red, *silent* where it is
    # neutral. Silence on grey is exact, which is the hue mask's whole claim.
    check("Source Red fires on red and nowhere else",
          gs_r[GS_RED].std() > 0.01 and float(np.abs(gs_r[GS_GREY]).max()) == 0.0,
          f"red sigma {gs_r[GS_RED].std():.5f}, grey max |delta| "
          f"{float(np.abs(gs_r[GS_GREY]).max()):.2e}")
    check("and Source Green does not", float(np.abs(gs_g[GS_RED]).max()) == 0.0,
          f"max |delta| on the red patch {float(np.abs(gs_g[GS_RED]).max()):.2e}")
    # And the bell in a render. **Both** ends, which is the point: a plain
    # brightness mask would pass a dark-end check just as well and then be
    # *louder* than mid grey at the light end. Only measuring both says "bell".
    check("Source Lightness peaks in the mid tones",
          gs_l[GS_DARK].std() < gs_l[GS_GREY].std() / 4.0
          and gs_l[GS_LIT].std() < gs_l[GS_GREY].std() / 4.0,
          f"grey sigma {gs_l[GS_GREY].std():.5f}, dark {gs_l[GS_DARK].std():.5f}, "
          f"light {gs_l[GS_LIT].std():.5f}")

    # The requirement the whole design is arranged around: these stack on the
    # flat layer rather than replacing it, so a region they mask away is still
    # not clean. Stated exactly, because it can be: the three colour masks are
    # *zero* on a neutral patch, so with all three up and Global Intensity at 0
    # the dark patch comes out perfectly clean -- and that is the failure the
    # flat layer underneath exists to prevent, not a tolerance.
    trio = {"global_src_r": 60.0, "global_src_g": 60.0, "global_src_b": 60.0}
    alone = float((gs_render(global_intensity=0.0, **trio) - 0.12)[GS_DARK].std())
    stacked = float((gs_render(**trio) - 0.12)[GS_DARK].std())
    check("a masked-out region still carries the flat layer",
          alone == 0.0 and stacked > 0.01,
          f"sigma {alone:.2e} from the masked set alone, {stacked:.5f} with the "
          f"flat layer under it")

    # Different seeds per layer -- the reason these are five `_grain_points`
    # calls and not one geometry with five brightness fields. Two layers sharing
    # an offset would put their grains in exactly the same places and the pair
    # would read as one louder layer. Measured on the patch both are live on.
    gs_lg = gs_render(global_src_l=60.0, global_size=3.0) - gs_off
    rho = float(np.corrcoef(gs_lg[GS_GREY + (1,)].ravel(),
                            (gs_off - 0.5)[GS_GREY + (1,)].ravel())[0, 1])
    check("a source layer is independent of the flat layer", abs(rho) < 0.05,
          f"correlation {rho:+.3f}")
    # The Seed slider has to reach them too -- it is the one control that
    # rerolls all five together, which is the other half of "different seed per
    # layer": different from each other, and all movable at once.
    gs_reseed = (gs_render(global_src_r=60.0, seed=4321.0)
                 - gs_render(seed=4321.0))
    rho = float(np.corrcoef(gs_r[GS_RED + (0,)].ravel(),
                            gs_reseed[GS_RED + (0,)].ravel())[0, 1])
    check("and the Seed slider rerolls it", abs(rho) < 0.05,
          f"correlation {rho:+.3f}")

    # Global Seed does the same for this section alone. It is an *offset* on
    # Seed, so three separate things have to hold and each would be a different
    # bug: it rerolls the layers, it is inert at 0, and it leaves the rest of
    # the frame -- the main grain above all -- exactly where it was.
    gs_gseed = (gs_render(global_src_r=60.0, global_seed=77.0)
                - gs_render(global_seed=77.0))
    rho = float(np.corrcoef(gs_r[GS_RED + (0,)].ravel(),
                            gs_gseed[GS_RED + (0,)].ravel())[0, 1])
    check("Global Seed rerolls the section", abs(rho) < 0.05,
          f"correlation {rho:+.3f}")
    d = float(np.abs(gs_render(global_src_r=60.0, global_seed=0.0)
                     - gs_render(global_src_r=60.0)).max())
    check("and 0 is bit-identical to no slider at all", d == 0.0,
          f"max delta {d:.2e}")
    # The main grain must not move. Rendered with the whole global section off,
    # so anything that did move could only have come from the shared `seed`
    # being disturbed -- which is exactly the mistake an absolute seed here
    # would invite someone into later.
    mg = {k: 0.0 for k in P.NEUTRAL_ZERO}
    mg.update({"intensity": 40.0, "global_opacity": 1.0})
    a = eng.render_image(gs_plate, P.sanitize(mg), 1.0, tile=1024, supersample=1)
    b = eng.render_image(gs_plate, P.sanitize({**mg, "global_seed": 4242.0}),
                         1.0, tile=1024, supersample=1)
    d = float(np.abs(a.astype(float) - b.astype(float)).max())
    check("and it leaves the main grain untouched", d == 0.0,
          f"max delta {d:.2e}")
    # It changes the field, so unlike the amounts it *must* miss the cache --
    # the opposite assertion from the one three checks below, and the reason
    # both are worth making.
    eng.clear_caches()
    gs_render(global_src_r=60.0)
    m_gs = eng.gs_misses
    gs_render(global_src_r=60.0, global_seed=99.0)
    check("Global Seed misses the cache", eng.gs_misses > m_gs,
          f"{eng.gs_misses - m_gs} miss(es)")

    # Opacity governs the whole section, not just the flat layer. Both sides
    # keep a non-zero `global_intensity` so both render rather than one of them
    # taking the neutral short-circuit.
    z0 = gs_render(global_opacity=0.0)
    z1 = gs_render(global_opacity=0.0, global_src_r=100.0, global_src_l=100.0)
    d = float(np.abs(z1 - z0).max())
    check("Global Opacity 0 silences the source layers", d == 0.0,
          f"max delta {d:.2e}")

    # The mask is clamped, and this is what that buys. `out` is unclamped at
    # step 13 -- halation can drive a channel well past 1.0 -- so without the
    # clamp the envelope would exceed 1 and these layers would run louder than
    # their own sliders. Bounded by the flat layer at the same amount, which is
    # exactly what an envelope of at most 1 means.
    hal = {"halation": 1.0, "halation_threshold": 0.3, "halation_radius": 8.0}
    hal_base = gs_render(global_intensity=0.0, **hal)
    hal_flat = float((gs_render(global_intensity=40.0, **hal) - hal_base)[GS_GREY].std())
    hal_src = float((gs_render(global_intensity=0.0, global_src_l=40.0, **hal)
                     - hal_base)[GS_GREY].std())
    check("the mask cannot amplify past the slider", hal_src <= hal_flat * 1.02,
          f"masked sigma {hal_src:.5f} against unmasked {hal_flat:.5f}")

    # Tile independence with the flat layer *off*, which is the case `pad_for`
    # would have got wrong: the smoothing blur is shared, and a gate that only
    # knew about `global_intensity` would reserve nothing here and seam the
    # export along exactly the smoothing radius.
    gsp = P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO} | {
        "global_opacity": 1.0, "global_size": 12.0, "global_size_max": 12.0,
        "global_smooth": 1.0, "global_src_r": 50.0, "global_src_g": 50.0,
        "global_src_b": 50.0, "global_src_l": 50.0})
    a = eng.render_image(img, gsp, 1.0, tile=4096, supersample=2)
    b = eng.render_image(img, gsp, 1.0, tile=128, supersample=2)
    d = float(np.abs(a - b).max())
    check("tile independence with the flat layer off", d < 2e-3,
          f"max delta {d:.2e}")
    # Pinned directly as well as through the render, so the reason the render
    # passes stays visible if someone re-tightens the gate.
    gs_pad_off = eng.pad_for(P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO}), 1.0)
    gs_pad_on = eng.pad_for(gsp, 1.0)
    check("pad_for reserves the smoothing blur with the flat layer off",
          gs_pad_on > gs_pad_off,
          f"{gs_pad_on}px against {gs_pad_off}px with the section off")

    # The fields read no image data, so they cache on geometry and parameters
    # alone; the mask is applied outside that boundary. A stale hit here renders
    # a plausible texture, so the counters are what is tested, not the pixels.
    eng.clear_caches()
    m0 = eng.gs_misses
    gs_render(global_src_l=40.0)
    built = eng.gs_misses - m0
    check("one field built per active layer", built == 1,
          f"{built} miss(es) for one layer")
    h1, m1 = eng.gs_hits, eng.gs_misses
    gs_render(global_src_l=70.0)
    check("the amount slider stays outside the cache",
          eng.gs_misses == m1 and eng.gs_hits > h1,
          f"{eng.gs_misses - m1} miss(es), {eng.gs_hits - h1} hit(s)")
    h2, m2 = eng.gs_hits, eng.gs_misses
    gs_render(global_src_l=70.0, global_blend=1.0)
    check("and so does the blend mode",
          eng.gs_misses == m2 and eng.gs_hits > h2,
          f"{eng.gs_misses - m2} miss(es), {eng.gs_hits - h2} hit(s)")
    m3 = eng.gs_misses
    gs_render(global_src_l=70.0, global_size=7.0, global_size_max=7.0)
    check("a size change misses it", eng.gs_misses > m3,
          f"{eng.gs_misses - m3} miss(es)")
    # The flat layer counts separately, which is what lets the two claims above
    # be about the source layers specifically rather than about the section.
    g0, s0 = eng.gg_misses, eng.gs_misses
    eng.clear_caches()
    gs_render(global_src_r=40.0, global_src_b=40.0)
    check("the flat layer is counted apart from them",
          eng.gg_misses == g0 + 1 and eng.gs_misses == s0 + 2,
          f"{eng.gg_misses - g0} flat, {eng.gs_misses - s0} source")

    # -- 5b-i-d. the blend modes ----------------------------------------------
    # One menu over all five layers. Add is the default and has to stay exactly
    # what the section did before the menu existed; the rest are new behaviour
    # and each is pinned by the property that distinguishes it.
    print("\nglobal grain, blend modes")

    # A grain-free layer is mid grey, so four of the six leave the picture
    # alone -- that is what "neutral" means for a blend mode and it is what
    # makes the amount slider a fade. Multiply and Screen have no neutral value
    # in 0..1 at all, and that is documented rather than fixed.
    gb_base = torch.linspace(0.02, 0.98, 49).view(1, 1, -1, 1).expand(1, 3, -1, 1)
    gb_base = gb_base.contiguous()
    for i, name in enumerate(P.GLOBAL_BLENDS):
        d = float(_grain_delta(gb_base, torch.zeros_like(gb_base), i).abs().max())
        if name in ("Multiply", "Screen"):
            check(f"{name} has no neutral grey, as documented", d > 0.4,
                  f"max |delta| {d:.4f} with no grain")
        else:
            check(f"{name} is neutral with no grain", d == 0.0,
                  f"max |delta| {d:.2e}")
    # Add returns the field itself, bit for bit. Not an optimisation: rebuilding
    # it as `(base + g) - base` is a different float, and every shipped preset
    # renders through this branch.
    gb_g = torch.rand(1, 3, 49, 1, generator=torch.Generator().manual_seed(4)) * 2 - 1
    check("Add is the field itself, to the bit",
          bool((_grain_delta(gb_base, gb_g, 0) == gb_g).all()), "exact")
    # Nothing is out of range where it should not be: a full-swing layer over a
    # 0..1 base stays in 0..1 for every mode except Add, which is the one that
    # can lift a black or blow a white and is why it reads as "on top of".
    for i, name in enumerate(P.GLOBAL_BLENDS[1:], start=1):
        o = gb_base + _grain_delta(gb_base, gb_g, i)
        check(f"{name} stays inside 0..1", float(o.min()) >= -1e-6
              and float(o.max()) <= 1 + 1e-6, f"[{float(o.min()):.4f}, "
              f"{float(o.max()):.4f}]")

    # What each mode does *across tones* -- the whole reason to offer a choice,
    # and the part of the help text that would otherwise be unverified prose.
    #
    # Measured on three flat plates rather than one ramp, and that is not
    # incidental: a 24-column slice of a ramp samples a different patch of the
    # grain field at each end, and the field's own local sigma varies enough
    # between patches (measured 0.029 against 0.042 on neighbouring slices) to
    # swamp the effect being measured. Three flat plates at the same coordinates
    # read the *identical* field and differ only in what is underneath it, which
    # is exactly the one variable these checks are about.
    def gb_sigma(mode: int, level: float) -> float:
        plate = np.full((200, 200, 3), level, dtype=np.float32)

        def r(amt: float) -> np.ndarray:
            base = {k: 0.0 for k in P.NEUTRAL_ZERO}
            base.update({"global_opacity": 1.0, "global_size": 2.5,
                         "global_size_max": 2.5, "global_blend": float(mode),
                         "global_intensity": amt})
            return eng.render_image(plate, P.sanitize(base), 1.0, tile=1024,
                                    supersample=1).astype(np.float64)
        return float((r(25.0) - r(0.0)).std())

    #                    what distinguishes this mode from the other five
    gb_shape = {
        # Ignores what is underneath entirely -- the only even-handed one, and
        # the reason it is the one that can lift a black. Equal to 0.1% rather
        # than to the bit: the delta is identical, but float32 rounds the sum
        # onto a coarser grid at 0.88 than at 0.12.
        "Add":        lambda d, m, l: max(d, m, l) / min(d, m, l) < 1.001,
        # Tapers at both ends. This is the film-like one.
        "Overlay":    lambda d, m, l: d < 0.4 * m and l < 0.4 * m,
        # The same shape, gentler, and lopsided toward the shadows.
        "Soft Light": lambda d, m, l: d < 0.85 * m and l < 0.5 * m,
        # Overlay driven by the grain instead of the image, which means it does
        # *not* taper -- worth pinning, because it is the one people assume does.
        "Hard Light": lambda d, m, l: d > 0.9 * m and l > 0.9 * m,
        # Grains the highlights, leaves the shadows alone.
        "Multiply":   lambda d, m, l: l > 1.5 * m and d < 0.4 * m,
        # And the exact reverse.
        "Screen":     lambda d, m, l: d > 1.5 * m and l < 0.4 * m,
    }
    for i, name in enumerate(P.GLOBAL_BLENDS):
        dk, md, lt = (gb_sigma(i, v) for v in (0.12, 0.50, 0.88))
        check(f"{name} responds to the tone underneath as documented",
              gb_shape[name](dk, md, lt),
              f"sigma {dk:.5f} dark / {md:.5f} mid / {lt:.5f} light")
    # The mean shift Multiply and Screen carry, which is the caveat in their
    # help text and the reason they are not simply "grain, but nicer".
    def gb_mean(mode: int) -> float:
        plate = np.full((200, 200, 3), 0.5, dtype=np.float32)

        def r(amt: float) -> np.ndarray:
            base = {k: 0.0 for k in P.NEUTRAL_ZERO}
            base.update({"global_opacity": 1.0, "global_size": 2.5,
                         "global_size_max": 2.5, "global_blend": float(mode),
                         "global_intensity": amt})
            return eng.render_image(plate, P.sanitize(base), 1.0, tile=1024,
                                    supersample=1).astype(np.float64)
        return float((r(25.0) - r(0.0)).mean())

    for i, name in enumerate(P.GLOBAL_BLENDS):
        mv = gb_mean(i)
        if name == "Multiply":
            check("Multiply darkens as well as grains", mv < -0.02,
                  f"mean shift {mv:+.4f} at 25")
        elif name == "Screen":
            check("Screen lightens as well as grains", mv > 0.02,
                  f"mean shift {mv:+.4f} at 25")
        else:
            check(f"{name} shifts no exposure", abs(mv) < 0.002,
                  f"mean shift {mv:+.4f} at 25")

    # Every mode is a real change and they are all different from each other --
    # a menu whose entries render the same thing is worse than no menu.
    gb_ramp = np.linspace(0.10, 0.90, 320, dtype=np.float32)
    gb_plate = np.repeat(gb_ramp[None, :, None], 160, axis=0).repeat(3, axis=2)

    def gb_render(mode: int, amt: float) -> np.ndarray:
        base = {k: 0.0 for k in P.NEUTRAL_ZERO}
        base.update({"global_opacity": 1.0, "global_size": 2.5,
                     "global_size_max": 2.5, "global_blend": float(mode),
                     "global_intensity": amt})
        return eng.render_image(gb_plate, P.sanitize(base), 1.0, tile=1024,
                                supersample=1).astype(np.float64)

    gb_out = [gb_render(i, 30.0) for i in range(len(P.GLOBAL_BLENDS))]
    worst = min(float(np.abs(gb_out[i] - gb_out[j]).max())
                for i in range(len(gb_out)) for j in range(i + 1, len(gb_out)))
    check("the six modes all render differently", worst > 1e-3,
          f"closest pair differs by {worst:.4f}")
    # Inert with the section off, the same rule every other shape control here
    # follows -- otherwise it is a colour grade, which is deferred.
    gb_z = {k: 0.0 for k in P.NEUTRAL_ZERO}
    a = eng.render_image(img, P.sanitize({**gb_z, "intensity": 32.0,
                                          "global_blend": 0.0}), 1.0, tile=1024)
    b = eng.render_image(img, P.sanitize({**gb_z, "intensity": 32.0,
                                          "global_blend": 4.0}), 1.0, tile=1024)
    d = float(np.abs(a.astype(float) - b.astype(float)).max())
    check("inert with the global layer off", d == 0.0, f"max delta {d:.2e}")

    # Tile independence per mode. They are pointwise, so none of them should
    # need padding -- but they are also the only stage that reads `out` back to
    # blend against it, and "pointwise" is exactly the assumption worth pinning
    # rather than asserting. All five layers up, so the masks are live too.
    for i, name in enumerate(P.GLOBAL_BLENDS):
        gbp = P.sanitize({k: 0.0 for k in P.NEUTRAL_ZERO} | {
            "global_opacity": 1.0, "global_size": 9.0, "global_size_max": 12.0,
            "global_smooth": 1.0, "global_blend": float(i),
            "global_intensity": 20.0, "global_src_r": 50.0,
            "global_src_g": 50.0, "global_src_b": 50.0, "global_src_l": 50.0})
        a = eng.render_image(img, gbp, 1.0, tile=4096, supersample=1)
        b = eng.render_image(img, gbp, 1.0, tile=64, supersample=1)
        d = float(np.abs(a - b).max())
        check(f"tile independence on {name}", d < 2e-3, f"max delta {d:.2e}")
