from __future__ import annotations

import math

import torch

from ..colour import _linear_to_srgb, _srgb_to_linear
from ..constants.core import EDGE_REF, _AMP_SCALE, _MIN_CELL
from ..constants.edge import (
    _JITTER_MAX, _SAND_DIR_K, _SAND_MIN_GRAD, _SAND_PASSES, _SAND_TAPS,
)
from ..constants.grade import _STEP_HI, _STEP_LO, _TEX_HI, _TEX_LO
from ..noise.fields import _fbm
from ..primitives import (
    _blur, _edge_magnitude, _isophote, _luma, _smootherstep, _smoothstep,
    _warp,
)

class RenderMixin:
    """The pipeline itself -- one tile, at one scale, in stage order."""

    def render(
        self, img: torch.Tensor, p: dict, scale: float = 1.0,
        y0: float = 0.0, x0: float = 0.0,
        full_hw: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Render one tile. ``img`` is [1,3,h,w] float in 0..1 on ``self.device``.

        ``scale`` is working-res / full-res; ``y0``/``x0`` are the tile's offset
        in working-resolution coordinates.
        """
        h, w = img.shape[-2:]

        # -1. Colour grading, above everything -- temperature, shadows,
        #     highlights, clarity, then the 3D LUT. See _grade.
        #
        #     Above `pre_blur` rather than anywhere else because this is the
        #     decision about what the photograph *is*; every stage below it is
        #     the emulsion's response to that photograph. Putting it after the
        #     film stages would mean grading grain, halation and dust along with
        #     the picture, and a LUT built to be fed a photograph would be being
        #     fed a rendered negative instead.
        # Checkpoint A restores the frame as it stood after Pre Sharpen, which
        # is everything Colour Grading, Pre Blur and Pre Sharpen do. Cheap on
        # the GPU and worth ~24% on the CPU, where highlight reconstruction
        # alone is 2.55s of a `SuperPortra` render -- and it hits on nearly
        # every edit, since almost every slider is below it.
        ck_a = self._ckpt_key("Grain Structure", p, scale, y0, x0, h, w)
        hit_a = None if ck_a is None else self.ckpt.get(ck_a)
        if hit_a is not None:
            img = hit_a
        else:
            # -2. Normalize, above Colour Grading and so above everything.
            #
            #     Grading says what the photograph should look like; this says
            #     what it *is* before anyone gets to decide that. An
            #     under-exposed or colour-cast frame handed to the section below
            #     is graded on top of an error, and every film stage under that
            #     is calibrated around a normally exposed picture -- the
            #     characteristic curve and Contrast both pivot on `_MID_GREY`,
            #     and the grain envelope peaks in the mid-tones.
            #
            #     Its own checkpoint, and the shallowest one. Nothing sits above
            #     it, so its signature is a single key and it hits on every edit
            #     anywhere else in the app -- see checkpoint.py. The stage itself
            #     is per-pixel arithmetic on six numbers `models/upload.py`
            #     measured once from the whole frame, so it reserves nothing in
            #     `pad_for` and the metering never re-runs.
            ck_n = self._ckpt_key("Colour Grading", p, scale, y0, x0, h, w)
            hit_n = None if ck_n is None else self.ckpt.get(ck_n)
            if hit_n is not None:
                img = hit_n
            else:
                img = self._normalize(img, p)
                if ck_n is not None:
                    self.ckpt.put(ck_n, img)

            self._poll_cancel()
            img = self._grade(img, p, scale)

            self._poll_cancel()
            # 0. Pre-blur, on the untouched input, in linear light.
            #
            #    The same gaussian as `micro_blur` and a different stage, because
            #    a blur's effect here is not only what it does to the pixels:
            #
            #    * It runs before `lum_ref` is taken, so the edge mask, the
            #      hard-edge step mask and the smooth-area guard all measure the
            #      *softened* frame. Micro-blur is deliberately excluded from that
            #      -- the masks read the untouched tile input so diffusing the
            #      frame cannot quietly talk the grain amount down. Here that
            #      coupling is the point: soften the source and the grain follows
            #      the softer edges and backs off where detail has gone.
            #    * It runs before the pre-sharpen below, so a broad radius here
            #      against a tight one there is a detail-killing pair the pipeline
            #      could not otherwise express. The other order would just throw
            #      the sharpening away.
            #
            #    In linear light for micro-blur's reason: a blur is light spreading
            #    sideways, and averaging gamma-encoded values instead darkens every
            #    edge it crosses. Gated so the transfer round trip costs nothing
            #    when the stage is off.
            pb = p["pre_blur"] * scale
            if pb >= 0.05:
                img = _linear_to_srgb(_blur(_srgb_to_linear(img), pb)).clamp(0.0, 1.0)

            self._poll_cancel()
            # 0b. Pre-sharpen, on the input.
            #
            #    Placed before every film stage so it sharpens the *photograph* and
            #    nothing else -- there is no grain yet to amplify. It is not
            #    cosmetic to put it here rather than at the end: every mask
            #    downstream is measured from this image, so sharpening now makes
            #    edges read as harder to the edge mask and pulls grain onto them.
            ps = p["pre_sharpen"]
            if ps > 0.01:
                img = (
                    img
                    + (img - _blur(img, max(0.3, p["pre_sharpen_radius"] * scale))) * ps
                ).clamp(0.0, 1.0)
            if ck_a is not None:
                self.ckpt.put(ck_a, img)

        # ================================================================ #
        # PIPELINE ORDER == PANEL ORDER  (reordered 2026-08-08, on request)
        #
        # The request was about *people*: a slider near the bottom of the panel
        # must not change anything above it. Before this the two disagreed --
        # Halation was panel section 8 and ran 5th, Tone Response was 9 and ran
        # 6th, and Grain Structure and Edge Destruction interleaved across each
        # other -- so reading the panel top to bottom told you nothing about
        # what depended on what.
        #
        # **It changed the look, deliberately.** Three things moved that cannot
        # move for free, and `docs/pipeline-order.md` carries the detail:
        # jitter and sanding now displace the grain rather than the clean
        # picture; halation blooms the grain rather than the latent image; and
        # the characteristic curve is applied to the finished frame. Every
        # shipped preset was re-checked against the result.
        # ================================================================ #

        # ---- 3. GRAIN STRUCTURE -------------------------------------------
        #
        # Display space from here down, and that placement is load-bearing. The
        # exposure block used to convert the whole frame to linear and back
        # around itself; now only the stages that genuinely average *light* do
        # their own round trip and they are far apart. Letting the transition
        # follow Tone Response down to section 9 would put grain in linear light
        # and change its shadow distribution completely -- `_AMP_SCALE` is
        # calibrated against grain applied here.
        base = img

        # 6b. Luminance response: how much grain each density carries. Grain is
        #     at full strength across the band [lum_low, lum_high] and eases out
        #     over a falloff width on each side. Band edges and transition
        #     widths are independent -- welding them together forces the ramp to
        #     start at pure black or run all the way to white, which is what
        #     makes the boundary visible.
        #
        #     **Measured here, off the developed density, rather than down at
        #     step 10 where it is used** (moved 2026-08-06, on request). What
        #     this mask asks is "how dense is the negative at this point", and
        #     the answer is settled the moment the characteristic curve and base
        #     fog have run: it is a property of development, so it belongs with
        #     the development stages and not among the destruction ones.
        #
        #     Read at its old position it was measured off a `base` that edge
        #     softening, edge jitter and sanding had already been through, which
        #     is wrong in the specific way step 7 and the smooth-area guard are
        #     both written to avoid: a blurred frame's luma is not the density
        #     the emulsion recorded, so softening the picture silently moved the
        #     grain around.
        #
        #     Measurable, and `verify.py` measures it. Put a hard black-to-white
        #     step on a frame and set the band to mid-tones only, so the mask
        #     reads zero on both sides. Softening the border invents a mid-tone
        #     ramp across it that was never in the photograph -- and read at the
        #     old position the mask believed it, laying a **0.095 sigma ribbon of
        #     grain** along a border whose two sides are both meant to be clean.
        #     Here it reads 0.00000: the density either side of a border is what
        #     it was, whatever was done to the border itself.
        #
        #     It also means the mask is no longer warped along with the image by
        #     edge jitter. That is the right way round: jitter displaces where
        #     the *picture* is, not how dense the silver is, and the mask is
        #     blurred over several pixels anyway -- an order more than jitter's
        #     peak travel.
        #
        #     The mask is driven by a spatially blurred luma so the transition
        #     is smooth across the *frame* as well as across the tone curve.
        #     Reading per-pixel luma lets image detail modulate the mask itself,
        #     which speckles the boundary region.
        #
        #     `lum_d` is the **density luma**, and it is kept as its own name
        #     rather than folded into `lum` because the two now mean different
        #     things. `lum` below is the luma of the picture *as it currently
        #     stands*, recomputed after every stage that moves a pixel, and the
        #     sanding filter needs exactly that -- it is steering along the
        #     contour it can see. `lum_d` is how much silver is here, and every
        #     control keyed on that reads this one: the band below, and Shadow
        #     Clumping over in `_grain_field`. Two density-keyed controls
        #     sampling at two different points in the pipeline would be a
        #     disagreement about what "the shadows" are, and it would show up as
        #     the clump size and the grain amount responding to a softened
        #     border differently.
        # **B1: the density luma, with the tone curve now six sections
        #     below.** `lum_d` answers "how dense will this develop", and the
        #     answer used to be sitting right there because the characteristic
        #     curve had just run. It has not, so the curve is evaluated *as a
        #     mask input only* -- the frame itself is untouched and Tone
        #     Response applies it for real at section 9.
        #
        #     This keeps the control's meaning exactly rather than
        #     approximating it: the same function, on the same photograph, at
        #     the same point in its life. It costs one extra evaluation,
        #     measured at 0.3% of a 24MP render. Reading `_luma(base)` raw
        #     instead would silently redefine both this band and Shadow
        #     Clumping from "how much silver" to "how bright the file is",
        #     which is the change `verify.py` pins against.
        dens = self._tone(base, p)
        lum_d = _luma(dens)
        lum_m = _blur(lum_d, max(1.0, 3.0 * scale))
        lo = p["lum_low"]
        hi = max(p["lum_high"], lo + 0.05)
        sf = max(p["shadow_falloff"], 1e-3)
        hf = max(p["highlight_falloff"], 1e-3)

        up_ramp = _smootherstep(max(0.0, lo - sf), lo, lum_m)
        dn_ramp = 1.0 - _smootherstep(hi, min(1.0, hi + hf), lum_m)
        m = (1.0 - p["shadow_drop"]) + p["shadow_drop"] * up_ramp
        m = m * ((1.0 - p["highlight_drop"]) + p["highlight_drop"] * dn_ramp)

        self._poll_cancel()
        # 7. Edge isolation (needed before jitter so we only warp real edges).
        #
        #    Measured from the *untouched tile input*, not from `base`. Every
        #    softening stage above -- micro-blur especially -- flattens exactly
        #    the micro-edges this mask keys on, so reading `base` meant that
        #    softening the picture also quietly turned the grain down: dial in
        #    some diffusion and you lost noise you never asked to lose. Keying
        #    off the original structure decouples the two, so softness and
        #    grain amount are independent controls. Tone curves ship neutral,
        #    so this is also very close to what `base` used to give.
        lum = _luma(base)
        # **B2: anti-aliasing runs at section 5 now, below this.** Its
        #    documented reason for sitting above the masks was that otherwise
        #    the grain keeps keying on the jaggies it is about to remove. So the
        #    mask input -- and only the mask input -- takes an anti-aliased copy
        #    here, while the visible filter stays where the panel puts it. One
        #    extra pass, on the frame rather than on a plane, and the property
        #    survives the move.
        lum_ref = _luma(
            self._antialias(img, p, scale)
            if p["aa_strength"] > 0.001 else img
        )
        hp_r = max(0.3, p["highpass_radius"] * scale)
        # Chroma-aware since 2026-08-09 -- see `_edge_magnitude`. At
        # `edge_chroma_sense` 0 this is bit for bit the luma high-pass it was.
        ch = p["edge_chroma_sense"]
        hp_mag = _edge_magnitude(img, hp_r, ch)
        # `edge_sensitivity` divides the reference the magnitude is normalised
        # against, so raising it makes gentler edges reach full strength. It was
        # the fixed `EDGE_REF`, which is the *other* reason edges were being
        # missed: 0.06 is a firm step, and everything softer than it only ever
        # reached a fraction of the mask however the sliders were set.
        ref = EDGE_REF / max(p["edge_sensitivity"], 0.05)
        edge = (hp_mag / ref).clamp(0.0, 1.0)
        edge = _blur(edge, hp_r * 0.8)

        self._poll_cancel()
        # 10. Grain field, weighted toward micro-edges and away from flat areas.
        #     `m` is the luminance-response mask, measured back at step 6b off
        #     the developed density rather than off this stage's input.
        #
        #     `lum_d` rather than `lum`, and for the same reason `m` is measured
        #     up there: the only thing `_grain_field` reads a luma for is Shadow
        #     Clumping, which asks how *dense* this area is -- shadows carry
        #     larger, less densely packed crystals -- and that is settled by
        #     development, not by what edge softening later did to the border.
        #     Passing the late `lum` here while `m` came from step 6b would have
        #     the two halves of the same physical question answered from two
        #     different frames.
        g = self._grain_field(h, w, y0, x0, lum_d, p, scale)
        eb = p["edge_bias"]
        weight = m * ((1.0 - eb) + eb * edge)

        # Smooth-area guard. The edge mask only sees micro-edges, so a smooth
        # gradient -- skin, a clear sky, a studio backdrop -- gets no protection
        # from it and takes the full flat-area floor. That is what makes skin
        # read as jagged. Measure local contrast over a medium radius instead:
        # a linear gradient has almost none (blurring a ramp returns the ramp),
        # while fabric, foliage and hair have plenty. Suppress grain where that
        # measure says the region is genuinely featureless.
        sg = p["smooth_guard"]
        if sg > 0.01:
            med_r = max(1.0, hp_r * 2.5)
            # From the reference luma for the same reason as the edge mask: a
            # softened region is not a featureless one, and blurring the frame
            # should not talk the guard into treating fabric as skin.
            tex = _blur((lum_ref - _blur(lum_ref, med_r)).abs(), med_r)
            textured = _smoothstep(_TEX_LO, _TEX_HI, tex)
            weight = weight * ((1.0 - sg) + sg * textured)

        amp = (p["intensity"] / 100.0) * _AMP_SCALE
        out = base + g * weight * amp

        self._poll_cancel()
        # ---- 4. EDGE DESTRUCTION ------------------------------------------
        #
        # After the grain now, not before it, so jitter and sanding move and
        # polish the grain along with the picture. This is the one change with
        # no workaround, and it is arguably the better model: these stages
        # describe the emulsion's own edge behaviour, and the grain is in the
        # emulsion. Previously it was laid on afterwards and stayed put.
        self._poll_cancel()
        # 7b. Edge softening. A global blur is the wrong tool for "make it
        #     softer": it takes the whole frame down, texture and all, and
        #     reads as out of focus rather than as film.
        #
        #     Note this cannot key on `edge` above. That mask asks "is there a
        #     micro-edge here", and fine texture is *made of* micro-edges -- so
        #     weighting by it softened fabric and hair almost as much as it
        #     softened a hard border. The discriminator has to be edge
        #     *amplitude*: a real transition steps a long way in luminance,
        #     where texture wobbles by a little. Measured over the softening
        #     radius, a hard border reads several times _STEP_HI while fine
        #     texture sits under _STEP_LO, so the threshold cleanly separates
        #     them where a high-pass alone cannot.
        es = p["edge_soften"]
        if es > 0.01:
            sr = max(0.3, p["edge_soften_radius"] * scale)
            step = _edge_magnitude(img, sr, ch)
            # `edge_soften_edges_only` scales the gate rather than replacing
            # it, so 1 is exactly the fixed thresholds this stage always used
            # and 0 opens it completely (`_smoothstep` with both edges at 0
            # returns 1 everywhere). Scaling both together keeps the ramp's
            # shape, so the control changes *which* edges qualify without
            # changing how abruptly they start to.
            sel = p["edge_soften_edges_only"]
            hard = _smoothstep(_STEP_LO * sel, _STEP_HI * sel, step)
            hard = _blur(hard, sr * 0.6)
            out = out + (_blur(out, sr) - out) * (hard * es)
            lum = _luma(out)

        # A smooth envelope traces an edge too precisely and reads as a digital
        # outline. Emulsion erodes an edge unevenly, so break the envelope up
        # with its own noise field (mean preserved at ~1.0).
        edge_clean = edge
        ragged = _fbm(
            h, w, y0, x0, max(_MIN_CELL, p["grain_size"] * scale * 2.0),
            int(p["seed"]) + 4241, 1, 2, 0.6, self.device,
        )
        edge = edge * (0.55 + 0.9 * ragged)

        self._poll_cancel()
        # 8. Sub-pixel edge jitter -- destroys hyper-sharp digital borders
        #    without wobbling flat areas. The noise cell is several times the
        #    clump size, so the displacement field is smooth along the edge and
        #    a border *wanders*: long, slow deviations.
        jit = p["edge_jitter"]
        if jit > 0.01:
            d = _fbm(h, w, y0, x0, max(_MIN_CELL, p["grain_size"] * scale * 3.0),
                     int(p["seed"]) + 911, 2, 1, 1.0, self.device) * 2.0 - 1.0
            dx, dy = d[:, 0:1], d[:, 1:2]

            # Directional bias. The raw field is isotropic -- measured, every
            # 45-degree sector takes 12-13% of displacements at the same mean
            # magnitude -- so simply *rotating* it would be a no-op: a rotated
            # isotropic field is the same field. What makes an angle mean
            # something is squeezing the displacement onto one axis first.
            #
            # Work in the rotated frame: u runs along the chosen axis, v across
            # it. Scaling v down concentrates the travel along u, so at
            # anisotropy 1 edges only ever move parallel to the angle. At 0
            # this is exactly the isotropic behaviour, whatever the angle says.
            aniso = p["jitter_aniso"]
            if aniso > 0.01:
                th = math.radians(p["jitter_angle"])
                ca, sa = math.cos(th), math.sin(th)
                u = dx * ca + dy * sa
                v = (dy * ca - dx * sa) * (1.0 - aniso)
                dx, dy = u * ca - v * sa, u * sa + v * ca

            amp = _JITTER_MAX * jit * max(scale, 0.25) * edge
            out = _warp(out, dx * amp, dy * amp)
            lum = _luma(out)

        self._poll_cancel()
        # 8b. Edge sanding -- takes the jaggedness back off, the way sandpaper
        #     does. Jitter roughens a border; left alone that reads as stair-
        #     stepped and harsh. This polishes it.
        #
        #     The operation is a blur *along* the edge, not across it. Smooth
        #     across a border and you have destroyed the border; smooth along
        #     it and the fine burrs average out while the transition stays as
        #     sharp as it was. So each pixel is averaged with its neighbours in
        #     the direction perpendicular to the local gradient -- the isophote
        #     tangent, i.e. the direction the edge actually runs.
        #
        #     The radius is what "grit" means here: a small radius reaches only
        #     the pixel-scale jaggies (a fine polish, shape untouched), a large
        #     one flattens broader undulations too.
        snd = p["edge_sand"]
        if snd > 0.01:
            total = max(0.5, p["edge_sand_grit"] * scale)
            # Applied as several short passes rather than one long one, with
            # the direction recomputed each time. The taps run in a straight
            # line, but the edge being sanded is precisely one that wanders --
            # so a single wide pass runs off the contour and cuts across it,
            # costing sharpness the filter exists to preserve. Short passes
            # re-aim, following the curve.
            #
            # The gain is real but modest: matched at 32% of the jaggedness
            # removed, iterating keeps 81% of the wander and 73% of the edge
            # sharpness against 79% and 71% for a single wide pass. It also
            # spreads the response more evenly over the grit range, which
            # matters more here -- this is a fine-tuning control.
            passes = int(min(_SAND_PASSES, max(1, round(total / 1.2))))
            sr = total / passes
            for _ in range(passes):
                # Direction from a blurred luma: taken per-pixel it would
                # follow the grain and jitter it is meant to remove, and sand
                # in circles.
                #
                # The blur has to scale with the sanding radius, not sit at a
                # fixed width. Where the gradient is weak the tangent is
                # numerically unstable -- it is a ratio of two near-zero
                # numbers -- and a filter reaching 13px along an arbitrary
                # direction samples somewhere entirely different for an
                # imperceptible change in input. That is not merely noisy: it
                # made tiled exports seam from 8px grit upward, because the
                # two tilings hand the gradient marginally different values.
                # Estimating direction over a window comparable to the reach
                # keeps it coherent and the result tile-independent.
                tx, ty, mag = _isophote(lum, max(0.6, _SAND_DIR_K * sr))
                # Where the gradient vanishes the tangent is a ratio of two
                # near-zero numbers and its direction is meaningless -- it
                # will swing on floating-point noise alone, and a filter
                # reaching a dozen pixels along it then samples somewhere
                # entirely different. Left ungated this showed up as a handful
                # of isolated pixels per frame disagreeing between a tiled and
                # a single-pass render. Fading the effect out with the
                # gradient fixes it and costs nothing: a region with no
                # gradient has no edge to sand.
                coherent = _smoothstep(0.0, _SAND_MIN_GRAD, mag)

                sanded = None
                wsum = 0.0
                for offv, wgt in _SAND_TAPS:
                    tap = (
                        out if offv == 0.0
                        else _warp(out, tx * (offv * sr), ty * (offv * sr))
                    )
                    sanded = tap * wgt if sanded is None else sanded + tap * wgt
                    wsum += wgt
                # Normalised here rather than trusting the table to sum to one
                # -- truncated gaussian weights do not, and the shortfall would
                # show up as every sanded edge being fractionally darker.
                sanded = sanded / wsum

                # Gated on the pre-ragged mask: the ragged envelope exists to
                # make erosion uneven, and sanding through it would polish in
                # patches.
                # **Clamped, and it was not** (fixed 2026-08-08). This is a
                # cross-fade toward `sanded`, so a factor above 1 is not "more
                # sanding" -- it is extrapolation past the filtered result,
                # amplifying the difference it was asked to remove. `edge_sand`
                # runs to 5 and `edge_clean` reaches ~1, so the factor reached
                # ~5 and the stage inverted: measured on a jittered border,
                # strength 1 took **46% of the jaggedness off** and strength 5
                # put **235% more on**, while the border's own wander grew to
                # 284% of what jitter gave it. The slider read as "more effect"
                # in both directions, which is why it survived.
                out = out + (sanded - out) * (
                    edge_clean * coherent * min(snd, 1.0)
                )
                lum = _luma(out)

        # Scatter then micro-blur, in that order -- swapping them makes the pair
        # come out *harder* on borders than the blur alone, measured in
        # `docs/edge-destruction.md`. Micro-blur averages light so it needs a
        # transfer round trip; scatter is a pure gather and needs none, but it
        # rides inside the same round trip rather than paying for its own.
        self._poll_cancel()
        mb = p["micro_blur"] * scale
        if p["scatter"] > 0.001 or mb >= 0.05:
            lin = _srgb_to_linear(out)
            if p["scatter"] > 0.001:
                lin = self._scatter(lin, h, w, y0, x0, p, scale)
            lin = _blur(lin, mb)
            out = _linear_to_srgb(lin).clamp(0.0, 1.0)

        # ---- 4c. EDGE DETAIL ----------------------------------------------
        #
        # **Erosion and acutance run at the *end* of Edge Destruction, not the
        # start, and that is a bug fix rather than a preference** (2026-08-08).
        # Both add fine, high-frequency structure; every other stage in this
        # section removes it. Run first -- which is where the panel listed them
        # and where the reorder first put them -- micro-blur, softening and
        # sanding averaged their entire contribution back out. Measured:
        # `edge_erosion` moved 0.01% of pixels by more than one 8-bit level and
        # `edge_chroma` 0.00%, against 2.67% and 0.22% before the reorder, even
        # though the term being added was *larger*.
        #
        # The panel section was reordered to match, so the rule still holds:
        # what you read last runs last.
        # 11. Structural erosion: modulate the image's own micro-detail by the
        #    grain field. Zero in flat areas, strongest on edges.
        er = p["edge_erosion"]
        if er > 0.01:
            detail = base - _blur(base, hp_r)
            # Per-channel modulation of a high-contrast edge gives each dye
            # layer its own erosion, producing coloured speckle along the edge.
            # ``edge_chroma`` blends between neutral erosion and full fringing.
            mono_g = g.mean(dim=1, keepdim=True)
            eg = mono_g + p["edge_chroma"] * (g - mono_g)
            out = out + eg * detail * weight * (1.6 * er)

        self._poll_cancel()
        # 12. Adjacency (Eberhard) effect. Developer exhausts faster on the
        #     dense side of an edge and diffuses across it, leaving a local
        #     contrast boost. Extracted from the pre-grain base so it sharpens
        #     the image rather than amplifying the grain we just added.
        acut = p["acutance"]
        if acut > 0.01:
            out = out + (base - _blur(base, hp_r * 1.5)) * (0.35 * acut)

    # Stored only on a miss: on a hit this frame *is* the cached one, and
    # re-putting it would be pure bookkeeping.

        # ---- 5. ANTI ALIASING ---------------------------------------------
        #
        # Below the masks now. Its documented reason for sitting above them was
        # that otherwise the grain keys on the jaggies it is about to remove --
        # which is why the mask input at section 3 takes an anti-aliased copy of
        # its own. The visible filter stays where the panel puts it.
        self._poll_cancel()
        if p["aa_strength"] > 0.001:
            out = _linear_to_srgb(
                self._antialias(_srgb_to_linear(out), p, scale)
            ).clamp(0.0, 1.0)

        # Checkpoint B restores the frame as it stood here -- everything above
        # is skipped on a hit. Safe because nothing but the image crosses this
        # boundary: halation and the tone curve take `out`, `_film_texture`
        # takes `out` plus parameters, `_source_masks` derives from `out`,
        # `_grain_delta` takes `out`, and sharpening is
        # `out + (out - blur(out))*sh`. See `checkpoint.py`.
        #
        # **Named for the section below it, which is Halation as of
        # 2026-08-09**: the boundary has not moved a statement, but Global Grain
        # and Sharpening moved out from under it to the bottom of the pipeline,
        # so the first section it protects is this one. Getting that name off by
        # one is a stale hit.
        ck_b = self._ckpt_key("Halation", p, scale, y0, x0, h, w)
        hit_b = None if ck_b is None else self.ckpt.get(ck_b)
        if hit_b is not None:
            out = hit_b
        elif ck_b is not None:
            self.ckpt.put(ck_b, out)

        # ---- 6. HALATION ---------------------------------------------------
        #
        # **It blooms the grain now**, which is the scan/print model rather than
        # the film-base one it used to be: the halo is a blur of an already
        # grained highlight instead of light added to the latent image and then
        # developed. That is the change, and it is what moving the section to
        # where the panel puts it means.
        #
        # `halation_recovery` survives intact, which is worth saying because it
        # nearly did not: it meters against `1 - lin`, the real linear headroom
        # *before* the characteristic curve compresses the highlights, and Tone
        # Response still runs below this. Move Tone Response above Halation and
        # that control stops meaning anything.
        #
        # In linear light, on its own round trip -- adding light in display
        # space is the usual reason simulated halation reads as a painted-on
        # glow. Gated so the transfer costs nothing when the stage is off.
        self._poll_cancel()
        if p["halation"] > 0.01:
            out = _linear_to_srgb(
                self._halation(_srgb_to_linear(out), p, scale)
            ).clamp(0.0, 1.0)

        # ---- 7. TONE RESPONSE ----------------------------------------------
        #
        # Applied to the finished frame. Section 3 already evaluated this same
        # function once as a mask input, to get the density luma the grain band
        # and Shadow Clumping key on -- same function, same photograph, so the
        # two cannot disagree about what "the shadows" are.
        #
        # It no longer develops the Global Grain layer, which goes on two
        # sections below as of 2026-08-09: that layer is not compressed by the
        # toe or the shoulder and not lifted by base fog. Consistent with this
        # section already sitting above Film Texture -- the characteristic curve
        # is what the *negative* does, and neither a speck of dust nor the grain
        # of the print stock was ever in the negative.
        self._poll_cancel()
        out = self._tone(out, p)

        # ---- 8. FILM TEXTURE ----------------------------------------------
        #
        # Masked by none of the image masks -- a scratch does not care what is
        # underneath it. **No longer last**: Global Grain and Sharpening run
        # below it as of 2026-08-09, on request, so the print grain lies over
        # the debris and the unsharp mask bites on it.
        self._poll_cancel()
        out = self._film_texture(out, h, w, y0, x0, p, scale, full_hw)

        # ---- 9. GLOBAL GRAIN ----------------------------------------------
        #
        # Below Film Texture, so the four source-masked layers read a frame with
        # the dust, the hair and the leaks in it and their envelopes follow the
        # debris -- a black hair pulls the lightness bell down along its length,
        # a leak drags the hue masks toward its own colour.
        self._poll_cancel()
        out = self._global_grain(out, h, w, y0, x0, p, scale)

        # ---- 10. SHARPENING ------------------------------------------------
        #
        # Still last, still for its own reason: the high-frequency content an
        # unsharp mask amplifies is the grain as much as the image. What changed
        # is that Film Texture is above it now, so the marks are part of that
        # content -- at the levels the shipped presets carry, this rings every
        # speck and every hair.
        self._poll_cancel()
        out = self._sharpen(out, p, scale)

        return out.clamp(0.0, 1.0)
