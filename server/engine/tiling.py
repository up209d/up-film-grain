from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import params as P
from .constants.core import _GSRC_KEYS, _MIN_CELL, _SMOOTH_MAX
from .constants.edge import (
    _AA_DIR_K, _AA_DIR_MIN, _AA_PASSES, _JITTER_MAX, _SAND_DIR_K, _SAND_PASSES,
)
from .constants.grade import _RECON_ROLL_GATE_FRAC
from .device import (
    _RELEASE_MIN_SHARE, _TILE_MAX, _TILE_MIN, _WORKING_BYTES_PER_PX,
    _tile_budget_bytes, release_cache,
)
from .checkpoint import upstream_signature
from .exceptions import RenderCancelled

class TilingMixin:
    """Supersampling, padding, tile sizing, and the public render entries.

    ``pad_for`` is the load-bearing one: a kernel missing from it seams a
    tiled export along exactly its radius while every preview looks fine.
    """

    def render_supersampled(
        self, img: torch.Tensor, p: dict, scale: float, y0: float, x0: float,
        ss: int, full_hw: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Render a tile at ``ss``x linear resolution and area-average back down.

        Grain is a sub-pixel phenomenon: rendering it at the output grid gives
        each clump a hard, aliased pixel footprint, which is exactly the
        synthetic look the project exists to avoid. Rendering above Nyquist and
        integrating down gives clumps genuine partial pixel coverage. Costs
        ss^2 in time and memory, and it is the single biggest realism win in
        the pipeline.

        **`master_opacity` is applied here, and the position is the whole
        point.** It cross-fades the finished frame back over the untouched
        input, so it has to see an input that has been through nothing at all
        -- and inside ``render`` there is no such thing at ss > 1, because what
        that method receives is already a bicubic *upsample*. Blending there
        and pooling down would make opacity 0 return the up-then-down round
        trip, which is measurably 1.0e-01 softer than the source on hard edges
        (see ``params.is_neutral``): "no effect" would quietly cost sharpness.
        Blending after the pool, against ``img``, is bit-exact at both ends at
        every supersample.

        Sitting here also means every entry point inherits it -- ``render_image``
        for the export, ``render_view`` for the preview -- so the two cannot
        disagree about what half strength looks like. It is per-pixel against
        the tile's own input, so it touches no statistic of the region and
        ``pad_for`` is unchanged.
        """
        op = p["master_opacity"]
        # Nothing of the render survives, so do not pay for it. The early exit
        # matters most on the slider itself: dragging toward 0 gets cheaper
        # rather than costing a full render to throw away.
        if op <= 0.0:
            return img

        if abs(ss - 1.0) < 1e-6:
            r = self.render(img, p, scale, y0, x0, full_hw)
        else:
            h, w = img.shape[-2:]
            # Rounded to whole pixels, and the *rounded* factor is what the rest
            # of the call uses. A fractional request like 1.5 cannot give a whole
            # working grid on every tile, and `scale`, `y0`, `x0` and `full_hw`
            # all have to agree with the grid actually rendered or the noise
            # lattice resolves to different global coordinates than the geometry
            # does -- which is invariant 2, and it seams.
            sh, sw = max(1, round(h * ss)), max(1, round(w * ss))
            eff_y, eff_x = sh / h, sw / w
            up = F.interpolate(
                img, size=(sh, sw), mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            # Working resolution and tile offset both scale, so the noise
            # lattice still resolves to the same global full-resolution
            # coordinates. Frame size scales with the working resolution
            # exactly as the tile offset does, so a normalised frame position
            # resolves the same.
            fh = (None if full_hw is None
                  else (full_hw[0] * eff_y, full_hw[1] * eff_x))
            r = self.render(up, p, scale * eff_x, y0 * eff_y, x0 * eff_x, fh)
            # **`avg_pool2d` whenever the factor is a whole number**, which is
            # every setting that existed before fractional supersampling did.
            # This is not a preference: an antialiased `interpolate` at 2x is a
            # 4-tap triangular filter, not a 2x2 box, so swapping it in
            # unconditionally would quietly reroll the look of every shipped
            # preset. `verify.py` pins 2x and 3x bit-exact against the old path.
            #
            # Fractional factors have no whole pooling window, so they take an
            # antialiased bilinear resample -- a triangular filter rather than a
            # box, which is not what 2x does and does not need to be: 1.5x is a
            # new setting with no shipped look to preserve. (`mode="area"` would
            # be the closer analogue and is deliberately not used: MPS refuses
            # adaptive pooling at non-divisible sizes, so it would work on CPU
            # and raise on the GPU.) Below 1 the frame was rendered *smaller*
            # than its output and has to come back up instead, which is bicubic
            # like every other upsample here.
            k = sh // h if (sh % h == 0 and sw % w == 0 and sh // h == sw // w) else 0
            if k >= 2:
                r = F.avg_pool2d(r, k)
            elif sh >= h:
                r = F.interpolate(
                    r, size=(h, w), mode="bilinear", antialias=True,
                    align_corners=False,
                )
            else:
                r = F.interpolate(
                    r, size=(h, w), mode="bicubic", align_corners=False,
                ).clamp(0.0, 1.0)

        # Cross-faded display-referred, where both images already live, rather
        # than round-tripping through linear. This is a compositing control --
        # "how much of the edit do I keep" -- not a physical average of light,
        # so the reasoning that puts `pre_blur` and halation in linear does not
        # carry over.
        #
        # The two are not interchangeable and the difference is not where you
        # would guess. Measured on a grained frame at half strength: mean
        # deviation 5.4e-04 and overall brightness within +0.05%, but a
        # worst-case 0.146 on individual pixels -- concentrated in the shadows,
        # where the transfer curve is steepest. That is exactly why encoded
        # wins here. Blending in the space the eye reads makes the slider
        # linear in *visible* deviation, so 0.5 is half the grain everywhere;
        # in linear the same 0.5 would take more than half out of the shadows
        # and less out of the highlights, which is an opacity control that
        # changes the look's balance as you dial it back. It also costs no
        # transfer round trip.
        if op < 1.0:
            r = img + (r - img) * op
        return r

    # ------------------------------------------------------------------ #
    def _ckpt_key(self, boundary: str, p: dict, scale: float,
                  y0: float, x0: float, h: int, w: int):
        """Key for a checkpoint at ``boundary``, or None if checkpointing is off.

        Everything the frame at that boundary depends on: which image and tier
        (`_ckpt_id`), where in it (`y0, x0, h, w` -- absolute, like every other
        cache here, so invariant 1 is untouched), at what working scale, and
        every parameter above the boundary. Miss one and the app renders a
        plausible but wrong photograph.
        """
        if self._ckpt_id is None or self.ckpt.cap <= 0:
            return None
        return (
            self._ckpt_id, boundary, float(scale), float(y0), float(x0), h, w,
            str(self.device), upstream_signature(p, boundary),
        )

    # ------------------------------------------------------------------ #
    def _poll_cancel(self) -> None:
        """Give up if a newer render has superseded this one.

        Called at every stage boundary inside `render()`. **Per tile was not
        enough and the reason is the preview**: `tile_for` returns 2400 for a
        2400px proxy, so a proxy render takes `render_image`'s single-pass
        branch, where `should_cancel` was checked exactly once before the pass
        began. Measured 2026-08-08 on `SuperPortra`: **one poll in 7.91s** on the
        GPU, 21.5s on the CPU. Every one of those seconds was spent on a frame
        the client had already abandoned, holding the render lock throughout.

        A stage boundary is the right granularity for the same reason a tile
        boundary was: no plumbing inside the stages, and the wasted work is
        bounded by the longest single stage rather than by the whole render.
        """
        if self._cancel is not None and self._cancel():
            raise RenderCancelled()

    # ------------------------------------------------------------------ #
    def pad_for(self, p: dict, scale: float) -> int:
        """Overlap needed so a rendered region matches the full-image render.

        Must cover every blur kernel in the pipeline: the clarity high-pass at
        the very top, the high-pass chain, the
        acutance blur (the widest at 1.5x), the pre-blur and micro-blur, the
        edge-softening blur, the global-grain smoothing blur, the output
        sharpening blur and halation, plus the
        displacement of every stage that *reads* a pixel from somewhere else
        rather than blurring in place -- the jitter warp, the sanding taps and
        scatter. Miss one and tiled exports seam along its radius -- which no
        preview will ever show.
        """
        hp_r = max(0.3, p["highpass_radius"] * scale)
        # Pre-blur and micro-blur are two kernels in series, not alternatives:
        # micro-blur reads pixels the pre-blur has already spread, so their
        # reaches add rather than the widest winning.
        mb = (p["micro_blur"] + p["pre_blur"]) * scale
        # Clarity's high-pass and highlight reconstruction's neighbourhood are
        # the only two kernels in the whole colour-grading section -- the other
        # ten stages are per-pixel and reserve nothing. Both are real reaches
        # even though the section runs first, and for the same reason: what they
        # measure over their radius feeds a value that then propagates through
        # every stage below, so a tile that cannot see far enough is wrong from
        # the top of the pipeline down rather than only at its own border.
        #
        # Summed rather than the widest winning: they are stages in series, and
        # reconstruction runs *above* clarity, so clarity's band is measured on
        # pixels reconstruction has already changed from up to its own radius
        # away.
        clar = (
            p["grade_clarity_radius"] * scale
            if abs(p["grade_clarity"]) > 0.001 else 0.0
        )
        if p["grade_recover"] > 0.001:
            # Two kernels in series inside the one stage: the chromaticity
            # estimate reads `radius`, and its own weight field is then blurred
            # again by `radius * _RECON_ROLL_GATE_FRAC` to gate the roll. The
            # second reads pixels the first already spread, so they add.
            # Three kernels in series inside the one stage: the chromaticity
            # estimate reads `radius`, then its weight field is dilated by
            # `radius * _RECON_ROLL_GATE_FRAC` and feathered by the same again to
            # gate the roll. Each reads pixels the previous one already spread,
            # so all three add. The dilation is a hard reach like a warp rather
            # than a kernel, but it is counted in here with the others because it
            # sits between two blurs and the sum is what has to be covered.
            rr = max(1.0, p["grade_recover_radius"] * scale)
            clar += rr * (1.0 + 2.0 * _RECON_ROLL_GATE_FRAC)
        halo = p["halation_radius"] * scale if p["halation"] > 0.01 else 0.0
        soft = p["edge_soften_radius"] * scale if p["edge_soften"] > 0.01 else 0.0
        shr = p["sharpen_radius"] * scale if p["sharpen"] > 0.01 else 0.0
        if p["pre_sharpen"] > 0.01:
            shr = max(shr, p["pre_sharpen_radius"] * scale)
        # Scratch softening blurs the mark field, so it reaches like any other
        # kernel.
        #
        # **Dust and hair reserve nothing at all**, and that is not an oversight
        # (changed 2026-08-06). They used to blur their mark fields and had to be
        # counted here; both are drawn one mark at a time now, from absolute
        # frame coordinates with an analytic soft edge and no kernel anywhere, so
        # a tile that can see its own pixels can draw every speck that touches
        # them -- including the ones whose centres sit in the next tile, because
        # `_mark_window` clips the mark's own footprint rather than the tile's.
        # Exactly the position light leaks have always been in, and `verify.py`
        # pins it by tiling a frame at maximum dust and hair with no overlap.
        tex_r = 0.0
        if p["scratches"] >= 1.0:
            tex_r = max(tex_r, p["scratch_soften"] * 3.0
                        * max(0.4 * p["scratch_width"] * scale, 0.6))
        # Anti-aliasing reads two ways at once and both have to be counted:
        # its taps travel a radius along the tangent (a displacement, like the
        # warps below), and it derives that tangent -- and its step gate --
        # from blurred luma, which is a kernel.
        #
        # Both terms are multiplied by the pass count, for the reason sanding
        # documents below: each pass resamples the previous pass's output, so tap
        # travel accumulates, and each pass re-derives its direction from a fresh
        # blurred luma, so that reach accumulates too. Pinned at _AA_PASSES
        # rather than recomputed from the strength, because pad_for is called at
        # the un-supersampled scale and would otherwise disagree with the
        # renderer about the count.
        aa_r = 0.0
        aa_tap = 0.0
        if p["aa_strength"] > 0.001:
            aa_rad = max(0.2, p["aa_radius"] * scale)
            aa_r = _AA_PASSES * max(
                max(_AA_DIR_MIN, _AA_DIR_K * aa_rad), aa_rad * 1.5)
            aa_tap = _AA_PASSES * aa_rad
        # Global-grain smoothing is a blur on the noise field, so it reaches
        # like every other kernel here. It is gated on the layer being on --
        # with intensity or opacity at zero the field is never built.
        #
        # Referenced against the *effective* cell -- max(Min, Max) after the
        # same up-clamp `render()` applies -- not against Min alone. Above Min,
        # the field itself is built on a lattice pitched at Max, and the blur
        # is measured against that same reference, so a tile computed here
        # with only Min in view would under-reserve and the export would seam
        # exactly where Max exceeds Min.
        gsm = 0.0
        # Same floored min and up-clamped max render() computes, not the raw
        # slider values -- matched exactly, including the floor, so the two can
        # never disagree about the field's reference scale.
        gcell_lo = max(_MIN_CELL, p["global_size"] * scale)
        g_eff = max(gcell_lo, p["global_size_max"] * scale)
        #
        # The gate covers the source-masked layers too, and that is not
        # cosmetic: they run through the same `_smooth_noise` against the same
        # reference cell, so with Global Intensity at 0 and a source layer up
        # the blur still happens. A gate that only knew about the flat layer
        # would reserve nothing there and seam the export along exactly the
        # smoothing radius, while every preview looked fine.
        g_on = (p["global_intensity"] > 0.01
                or any(p[k] > 0.01 for k in _GSRC_KEYS))
        if (g_on and p["global_opacity"] > 0.001
                and p["global_smooth"] > 0.001):
            gsm = p["global_smooth"] * _SMOOTH_MAX * g_eff
        # The grain field itself reserves **nothing**, and that is a real
        # change: the old cellular path carried a `_VARCELL_RINGS * cell` term
        # here. `_grain_points` derives its own lattice window from whatever
        # window it is handed, with a ring of slack on every side, so a pixel
        # always sees its true neighbouring cells however the frame was split
        # -- there is no boundary cell for it to substitute. Measured at
        # 1.2e-06 between a whole-frame render and arbitrary sub-windows with
        # zero padding, against the 2e-03 every other tile-independence check
        # here is held to. `verify.py` pins that directly rather than trusting
        # this comment, which is what has to stay true if the field ever grows
        # a kernel of its own. `global_smooth` above is that kernel today and
        # is reserved for separately.
        mask_r = max(1.0, 3.0 * scale)
        # Scatter reads a pixel up to its full reach away. It displaces rather
        # than blurring, so it belongs with the warps below and not in the
        # kernel sum.
        #
        # Reach *plus one pixel*: dx and dy are rounded to whole pixels
        # independently, so two half-pixel roundings the same way lengthen the
        # vector by up to sqrt(2)/2. It would fit inside the +4 at the end of
        # this function either way, but a stage that silently depends on
        # another term's slack is a seam waiting for somebody to tighten it.
        sca = (
            max(0.5, p["scatter_radius"] * scale) + 1.0
            if p["scatter"] > 0.001 else 0.0
        )
        # Jitter warps the image rather than blurring it, so it reads pixels
        # displaced by up to its peak -- which at _JITTER_MAX is no longer the
        # sub-pixel rounding error it was at 0.6.
        # Both the jitter warp and the sanding filter read displaced pixels
        # rather than blurring in place, so the overlap has to cover how far
        # each of them travels.
        jit = _JITTER_MAX * p["edge_jitter"] * max(scale, 0.25) + sca
        if p["edge_sand"] > 0.01:
            # Sanding compounds in two ways at once, and both have to be
            # counted or a tiled export seams while every preview looks fine.
            # Each of its (up to three) passes resamples the previous pass's
            # output, so tap travel accumulates to 2 x total rather than
            # total; and each pass re-derives its direction from a blurred
            # luma, so that blur's reach accumulates too. Counting only the
            # first was enough at the old 4px grit ceiling and seams from 8px
            # up. Passes is pinned at its maximum here rather than recomputed,
            # because pad_for is called at the un-supersampled scale and would
            # otherwise disagree with the renderer about the count.
            total = max(0.5, p["edge_sand_grit"] * scale)
            sr = total / _SAND_PASSES
            dir_reach = 3.0 * max(0.6, _SAND_DIR_K * sr)
            jit += _SAND_PASSES * (2.0 * sr + dir_reach)
        return int(
            math.ceil(
                3.0 * (hp_r * 3.3 + mb + clar + halo + soft + shr + tex_r
                       + mask_r + gsm + aa_r)
                + jit + aa_tap
            )
        ) + 4

    # ------------------------------------------------------------------ #
    def tile_for(
        self, p: dict, scale: float, h: int, w: int, ss: int,
    ) -> int:
        """Largest tile whose working set fits the memory budget.

        Tiling is pure overhead: `pad_for` overlap is read, rendered and thrown
        away on all four sides, so a smaller tile does strictly more work for the
        same output. Measured on a 2400x1600 `Stock` proxy at supersample 2,
        fresh process each, best of 3:

        | tile | tiles | overdraw | time  |
        |------|-------|----------|-------|
        | 1024 |   6   |  1.59x   | 4.46s |
        | 1536 |   4   |  1.32x   | 3.70s |
        | 2048 |   2   |  1.15x   | 3.30s |
        | 4096 |   1   |  1.00x   | 2.77s |

        Interior *export* tiles are the worst case, since they pad on all four
        sides: 1024 + 2*178 = 1380 square rendered for 1024 square kept, 1.82x.

        **So why not simply always use one tile?** Because memory is the binding
        constraint and it is the thing this codebase's own "quality beats speed"
        licence does not cover -- an out-of-memory render is not slow, it is
        broken. Peak driver-allocated memory on the sweep above went 6.0GB at
        tile 1536 to 8.0GB at 2048. On an 8GB machine that swaps or dies, and an
        8GB machine is exactly where the tiling matters most.

        Hence a budget rather than a constant. Note the coupling this creates
        with `pad_for`, which is the right one and which the old hard-coded 1024
        and 1536 got wrong in both directions: a wide-kernel preset pads more, so
        it *gets a smaller tile*, because its working set per tile is larger for
        the same nominal tile.

        `_WORKING_BYTES_PER_PX` is measured, not guessed -- see its comment. The
        answer is clamped into `_TILE_MIN`..`_TILE_MAX` and never exceeds what
        the image actually needs, so a small frame still renders in one pass.

        The budget is `_tile_budget_bytes`, the renderer's *share* of the pool,
        not the whole of it. That is a real change and it costs tiles: the
        texture cache used to take a flat 0.5GB on top of whatever this claimed,
        so the two together overran the pool by design. Sharing it explicitly
        makes a tile ~20% smaller and the cache large enough to hit -- measured
        7.36s -> 1.78s on a `SuperPortra` proxy, against a few percent of extra
        overdraw here.
        """
        pad = self.pad_for(p, scale)
        budget = _tile_budget_bytes()
        ss = max(0.25, float(ss))
        longest = max(h, w)

        def fits(tile: int) -> bool:
            # The padded read window is *clamped to the image* (see
            # `render_image`), so the worst tile is bounded by the frame, not by
            # `tile + 2 * pad`. Solving the square upper bound in closed form
            # instead over-predicts badly once the tile approaches the image
            # size -- it wanted 2 tiles for a proxy that comfortably fits in one.
            th = min(h, min(tile, h) + 2 * pad)
            tw = min(w, min(tile, w) + 2 * pad)
            return (th * ss) * (tw * ss) * _WORKING_BYTES_PER_PX <= budget

        # Descending search rather than closed form: `fits` is monotonic in
        # `tile`, the candidate list is short, and this keeps the memory model in
        # one readable place instead of inverted through algebra.
        tile = _TILE_MIN
        for cand in range(min(_TILE_MAX, longest), _TILE_MIN, -128):
            if fits(cand):
                tile = cand
                break
        # Never below _TILE_MIN even if the budget says so: there the overlap
        # dominates the useful area so completely that the extra work costs more
        # than the memory it saves, and every supported backend can hold a tile
        # this size.
        tile = max(_TILE_MIN, tile)
        # No point in a tile larger than the image -- `render_image`
        # short-circuits to a single untiled pass when `tile >= max(h, w)`.
        return min(tile, longest)

    def render_view(
        self, arr: np.ndarray, p: dict, box: tuple[int, int, int, int],
        zoom: float = 1.0, supersample: float = 2.0,
    ) -> np.ndarray:
        """Render ``box`` = (y, x, h, w) of ``arr`` at a display ``zoom``.

        Reads a padded window so every filter sees its true neighbourhood, then
        trims. This is what makes the inspection view trustworthy: what you see
        is exactly what the export will contain for that region.

        Zoom above 1.0 renders at 1:1 and leaves magnification to the client --
        upsampling before rendering would invent grain that is not in the
        export. Zoom below 1.0 renders at that working scale, which is the
        honest thing to show: at 50% the export's grain really is half-resolved.
        """
        y, x, bh, bw = box
        H, W, _ = arr.shape
        scale = min(float(zoom), 1.0)

        # Padding is needed in source pixels, but pad_for is in working pixels.
        pad = int(math.ceil(self.pad_for(p, scale) / max(scale, 1e-3)))
        ya, yb = max(0, y - pad), min(H, y + bh + pad)
        xa, xb = max(0, x - pad), min(W, x + bw + pad)

        if scale < 0.999:
            # Snap the read origin so that origin*scale is a whole number of
            # working pixels. Downsampling samples at pixel centres, so a crop
            # whose origin lands mid-pixel resolves on a different grid phase
            # than a whole-image downscale would -- a half-pixel shift that is
            # invisible on smooth areas and obvious on hard edges.
            step = next(
                (k for k in range(1, 9) if abs(k * scale - round(k * scale)) < 1e-6),
                1,
            )
            ya = (ya // step) * step
            xa = (xa // step) * step

        chunk = np.ascontiguousarray(arr[ya:yb, xa:xb, :])
        t = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if scale < 0.999:
            ch, cw = t.shape[-2:]
            t = F.interpolate(
                t, size=(max(1, round(ch * scale)), max(1, round(cw * scale))),
                mode="bicubic", antialias=True, align_corners=False,
            ).clamp(0.0, 1.0)

        # Frame size is the whole source at this scale, not the read window --
        # a crop must place the light leak where it falls in the *frame*, or
        # zooming in would drag the leak around with the viewport.
        fh, fw = arr.shape[0] * scale, arr.shape[1] * scale
        r = self.render_supersampled(
            t, p, scale, ya * scale, xa * scale, max(0.25, float(supersample)),
            (float(fh), float(fw)),
        )
        r = r.squeeze(0).permute(1, 2, 0).cpu().numpy()

        oy, ox = round((y - ya) * scale), round((x - xa) * scale)
        oh, ow = max(1, round(bh * scale)), max(1, round(bw * scale))
        return r[oy: oy + oh, ox: ox + ow, :]

    def render_crop(
        self, arr: np.ndarray, p: dict, box: tuple[int, int, int, int],
        scale: float = 1.0, supersample: float = 2.0,
    ) -> np.ndarray:
        """1:1 render of ``box``, bit-identical to the same region of a full
        render. Thin wrapper kept for the invariant checks."""
        return self.render_view(arr, p, box, scale, supersample)

    # ------------------------------------------------------------------ #
    def render_image(
        self, arr: np.ndarray, p: dict, scale: float = 1.0,
        tile: int = 1024, supersample: float = 2.0, progress=None,
        should_cancel=None, checkpoint_id=None,
    ) -> np.ndarray:
        """Render a whole image, tiling when it is larger than ``tile``.

        ``arr`` is HxWx3 float32 in 0..1. Returns the same shape.

        ``should_cancel``, if given, is polled once per tile **and at every stage
        boundary inside `render`** (see `_poll_cancel`); returning true raises
        `RenderCancelled`. It matters because the caller cannot interrupt this any
        other way -- a Starlette threadpool worker runs to completion whatever the
        client does, so an abandoned preview would otherwise keep the render lock
        for its full duration and every request behind it would queue on work
        nobody is waiting for.

        Tile granularity alone used to be the whole of it, and it was the wrong
        unit for the case that matters: a proxy preview is a *single* tile, so it
        polled once and then ran to completion regardless. See `_poll_cancel`.
        """
        # Nothing switched on: hand the input straight back. Not merely an
        # optimisation -- see params.is_neutral for why rendering it would
        # *not* return the input.
        if P.is_neutral(p):
            return arr
        self._cancel = should_cancel
        self._ckpt_id = checkpoint_id
        try:
            return self._render_image(arr, p, scale, tile, supersample, progress)
        except RenderCancelled:
            # An abandoned render's tensors are dead the moment the exception
            # unwinds, but the allocator keeps their blocks -- which on a
            # superseded preview is the whole working set, reserved against a
            # frame nobody will ever see. The render that supersedes it is about
            # to ask for the same memory, so hand it back rather than making it
            # grow the pool.
            release_cache(self.device)
            raise
        finally:
            self._cancel = None
            self._ckpt_id = None

    def _render_image(
        self, arr: np.ndarray, p: dict, scale: float, tile: int,
        supersample: float, progress,
    ) -> np.ndarray:
        """`render_image`'s body, with the cancel hook already installed."""
        self._poll_cancel()
        ss = max(0.25, float(supersample))
        h, w, _ = arr.shape
        if max(h, w) <= tile:
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self.render_supersampled(
                t, p, scale, 0.0, 0.0, ss, (float(h), float(w))
            )
            if progress:
                progress(1.0)
            # **No `release_cache` here, deliberately.** A single-tile render has
            # no peak to bound -- it is one allocation cycle and the numbers are
            # the same whether the blocks are handed back or not -- so all the
            # call would buy is making the *next* render re-acquire them.
            # Measured, and it is not free: `Stock`'s proxy went 1.13s -> 1.45s
            # and `VintageDarkGrainy`'s 1.64s -> 1.94s with a release on this
            # path, for no memory saved. The tiled path below is where holding
            # the free list actually costs something.
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Overlap must cover every blur kernel in the pipeline plus the warp.
        pad = self.pad_for(p, scale)

        out = np.empty_like(arr)
        ny = math.ceil(h / tile)
        nx = math.ceil(w / tile)
        done = 0
        # Whether handing blocks back between tiles is worth the stall it costs.
        # Decided once from the worst tile rather than per tile, so every tile of
        # one render behaves the same way. See `_RELEASE_MIN_SHARE`.
        worst = (min(h, tile + 2 * pad) * ss) * (min(w, tile + 2 * pad) * ss)
        release = (
            worst * _WORKING_BYTES_PER_PX
            >= _RELEASE_MIN_SHARE * _tile_budget_bytes()
        )
        for ty in range(ny):
            for tx in range(nx):
                self._poll_cancel()
                y_a, y_b = ty * tile, min((ty + 1) * tile, h)
                x_a, x_b = tx * tile, min((tx + 1) * tile, w)
                # padded read window, clamped to the image
                py_a, py_b = max(0, y_a - pad), min(h, y_b + pad)
                px_a, px_b = max(0, x_a - pad), min(w, x_b + pad)

                chunk = arr[py_a:py_b, px_a:px_b, :]
                t = torch.from_numpy(np.ascontiguousarray(chunk))
                t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
                r = self.render_supersampled(
                    t, p, scale, float(py_a), float(px_a), ss, (float(h), float(w))
                )
                r = r.squeeze(0).permute(1, 2, 0).cpu().numpy()

                out[y_a:y_b, x_a:x_b, :] = r[
                    y_a - py_a: y_a - py_a + (y_b - y_a),
                    x_a - px_a: x_a - px_a + (x_b - x_a),
                    :,
                ]
                done += 1
                # This tile's device tensors are dead the moment `r` is on the
                # host. Handing their blocks back *between* tiles rather than
                # letting the allocator hoard them is what keeps
                # `_WORKING_BYTES_PER_PX` honest -- and it is faster, not merely
                # leaner, because on unified memory the free list is system RAM.
                #
                # Between, not after: releasing past the last tile would only
                # make the next render re-acquire what it is about to ask for
                # again, which is the cost the single-tile path above measures.
                if release and done < ny * nx:
                    del t, r
                    release_cache(self.device)
                if progress:
                    progress(done / float(ny * nx))
        return out
