from __future__ import annotations

import math

import torch

from ... import params as P
from ..colour import _linear_to_srgb
from ..constants.edge import (
    _AA_DIR_K, _AA_DIR_MIN, _AA_PASSES, _AA_TAPS, _SAND_MIN_GRAD,
)
from ..constants.grade import _STEP_HI, _STEP_LO
from ..noise.fields import _scatter_offsets
from ..noise.grain import _SCATTER_STENCILS
from ..noise.lattice import _cell_noise
from ..primitives import _blur, _isophote, _luma, _smoothstep, _warp

class EdgeMixin:
    """Edge destruction that is not a filter: anti-aliasing and scatter.

    Both displace rather than blur, so both read pixels up to their peak
    travel away and belong in ``pad_for``'s additive term.
    """

    def _antialias(
        self, lin: torch.Tensor, p: dict, scale: float,
    ) -> torch.Tensor:
        """Take stair-stepping off hard edges without softening them.

        A stair-step is a *pixel-scale wobble along* a contour, not a hard
        transition across one -- so the cure is to filter along the isophote
        tangent and never across it. Averaging across is what a blur does, and
        it would take the edge with it; that is the whole reason this is a
        directional filter rather than a soften.

        It overlaps `edge_sand` in mechanism and is deliberately not the same
        control, on three counts. **Position**: this runs at step 1c on the
        source, in the optical block, where the aliasing that came in with the
        file lives; sanding runs at 8b to polish roughness the *jitter stage
        just added*, and cannot reach back to fix the input. **Scale**: three
        taps at about a pixel against sanding's five to +/-2 sigma, because
        the two are removing different wavelengths -- measured, 92% of a
        jittered contour's roughness sits above 8px, while a stair-step is one
        pixel by definition. **Gate**: this one fires on the luma *step*, so
        it finds hard borders and leaves texture alone, where sanding follows
        wherever the grit is dialled.

        In linear light, with the block it sits in. It averages light, and
        averaging gamma-encoded values holds the encoded mean rather than the
        light's -- the same reason `pre_blur` does its transfer round trip.

        Above strength 1 the filter is **repeated** rather than widened, up to
        ``_AA_PASSES``. One three-tap pass is a gentle thing -- 35% of a
        stair-step -- and a single pass was reported as doing "little to none".
        Reaching further is the wrong lever for the reason the taps are short in
        the first place: a stair-step is one pixel wide by definition, so a
        longer filter starts averaging away the shape the contour has rather
        than the wobble on it. Repeating attacks only the wobble, and because
        each pass re-estimates the tangent from the frame it is handed, it
        re-aims along a curving edge where one wide pass cuts the corner. Same
        idiom, and same reasoning, as ``edge_sand``'s ``_SAND_PASSES``.
        """
        st = p["aa_strength"]
        radius = max(0.2, p["aa_radius"] * scale)
        edge_only = p["aa_edge_only"]

        # Whole passes plus a fractional last one, so the control stays
        # continuous and strength <= 1 is bit-for-bit the single pass it always
        # was. Capped: pad_for reserves for _AA_PASSES exactly.
        passes = min(_AA_PASSES, int(math.ceil(st - 1e-6)))

        for i in range(passes):
            # The last pass carries the remainder -- strength 2.5 is two full
            # passes and one at half. Earlier passes are full strength.
            amt = min(1.0, st - i)

            # Display-referred for the detector, encode-then-luma rather than
            # the other way round: the transfer curve does not commute with a
            # weighted sum, and the step thresholds below are shared with edge
            # softening, which measures the same quantity the same way.
            #
            # Re-measured every pass, on the current frame rather than the
            # original. That is the entire value of iterating: the tangent
            # follows the contour as the previous pass left it, so a curve gets
            # followed instead of chorded.
            lum_d = _luma(_linear_to_srgb(lin))
            tx, ty, mag = _isophote(lum_d, max(_AA_DIR_MIN, _AA_DIR_K * radius))
            # Fade out where the tangent is meaningless -- see _isophote.
            # Without it a flat region's direction swings on float noise and
            # tiled and untiled renders disagree by a scatter of single pixels.
            m = _smoothstep(0.0, _SAND_MIN_GRAD, mag)

            # The aliasing gate: how far the luma actually steps across this
            # neighbourhood. `_STEP_LO`/`_STEP_HI` already separate a real
            # transition from fine texture -- fine texture measures an order of
            # magnitude below a hard border -- so a jagged border is found and
            # fabric is left alone. Reused rather than re-derived: two constants
            # for one discrimination would be two things to keep in step.
            if edge_only > 0.001:
                # Measured exactly as edge softening measures it -- same
                # high-pass, same radius convention, no scale factor. The
                # thresholds are calibrated against that quantity, so a fudge
                # here would silently put this control on a different scale from
                # the constants it borrows. An earlier ×2 did precisely that and
                # left the gate firing on fabric.
                step = (lum_d - _blur(lum_d, radius)).abs()
                hard = _smoothstep(_STEP_LO, _STEP_HI, step)
                # Smoothed, or the mask is as ragged as the staircase it is
                # selecting and the filter switches on and off down the edge.
                hard = _blur(hard, radius * 0.6)
                # At 0 the filter runs everywhere, at 1 only on hard edges. A
                # mix rather than a switch, because a CG render aliases on
                # gentler steps than a photograph does.
                m = m * ((1.0 - edge_only) + edge_only * hard)

            out = None
            wsum = 0.0
            for offv, wgt in _AA_TAPS:
                tap = (
                    lin if offv == 0.0
                    else _warp(lin, tx * (offv * radius), ty * (offv * radius))
                )
                out = tap * wgt if out is None else out + tap * wgt
                wsum += wgt
            # Normalised from the weights actually used, not trusted to the
            # table.
            out = out / wsum
            lin = lin + (out - lin) * (amt * m)

        return lin

    # ------------------------------------------------------------------ #
    def _scatter(
        self, x: torch.Tensor, h: int, w: int, y0: float, x0: float,
        p: dict, scale: float,
    ) -> torch.Tensor:
        """Displace a share of the pixels onto their neighbours, without averaging.

        A blur and this stage model the same physics from opposite ends. Light
        diffusing through the emulsion is a stochastic process: a photon either
        goes straight or is deflected onto a neighbouring grain. Average over
        infinitely many photons and you get a convolution -- ``micro_blur``,
        which is smooth because it is an expectation. Resolve the deflections
        individually and you get this: detail lands somewhere it was not,
        every value survives intact, and the result is *disordered* rather
        than smoothed. That is the whole reason the stage exists. A digital
        frame softened with a blur reads as out of focus because the blur
        removes the micro-contrast along with the edge; scatter removes
        neither, and takes the exactness instead.

        Three properties follow from never averaging, and all three are why
        this is not just another kernel:

        * **No value is invented.** Every output pixel is a bit-exact copy of
          some input pixel, so the frame's histogram, its grit and its noise
          come through untouched. Sampling is nearest-neighbour on whole-pixel
          offsets specifically to keep that true -- bilinear at a fractional
          offset would quietly turn each sample into a 2x2 average.
        * **Amount is coverage, not opacity.** ``scatter`` moves the threshold
          on a uniform field, so it sets *how many* pixels travel. Cross-fading
          a displaced pixel with the one it left would be an average by
          another name, and at 0.5 it would read as exactly the blur this
          replaces.
        * **It masks itself.** Displacing a pixel whose neighbours already
          match it changes nothing, so smooth sky, skin and studio backdrops
          come out untouched with no mask anywhere in the code. The stage acts
          only where there is detail to disorder, which is the inverse of
          ``micro_blur``'s failure mode -- that one takes texture down first
          and edges second.

        There is deliberately no frequency split here, and I built one before
        working out why it was pointless -- see the note in CLAUDE.md. The
        stage is already frequency-selective by construction: a displacement
        can only change a pixel by as much as the picture varies over the
        distance travelled, so structure coarser than the reach survives for
        free and ``scatter_radius`` is the frequency control.
        """
        amt = p["scatter"]
        reach = max(0.5, p["scatter_radius"] * scale)
        # Cells finer than a working pixel cannot be resolved; below that the
        # nearest-neighbour read just aliases between them.
        cell = max(1.0, p["scatter_cell"] * scale)
        pattern = int(round(p["scatter_pattern"])) % len(_SCATTER_STENCILS)

        n = _cell_noise(h, w, y0, x0, cell, int(p["seed"]) + 3301, 3, self.device)
        sel, mag_n, gate = n[:, 0:1], n[:, 1:2], n[:, 2:3]

        # Direction and distance, on the stencil, in whole pixels -- whole so
        # the gather stays a copy rather than an interpolation. Reach Spread:
        # 0 puts every displaced pixel on the shape's edge (detail hollows
        # out), 1 fills it inward.
        dx, dy = _scatter_offsets(
            sel, mag_n, reach, p["scatter_spread"], pattern
        )
        # Coverage: a uniform field thresholded at the amount, so `amt` is
        # literally the fraction of the frame that moves. Applied after the
        # rounding so a pixel that is not travelling gets a displacement of
        # exactly zero and reads itself back.
        move = (gate < amt).to(x.dtype)
        dx, dy = dx * move, dy * move

        return _warp(x, dx, dy, mode="nearest")
