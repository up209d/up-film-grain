from __future__ import annotations

import math

import torch

from ... import params as P
from ..colour import _characteristic_curve, _linear_to_srgb, _srgb_to_linear
from ..constants.core import EDGE_REF, _AMP_SCALE, _GSRC_KEYS, _MIN_CELL
from ..constants.edge import (
    _JITTER_MAX, _SAND_DIR_K, _SAND_MIN_GRAD, _SAND_PASSES, _SAND_TAPS,
)
from ..constants.grade import _STEP_HI, _STEP_LO, _TEX_HI, _TEX_LO
from ..constants.tone import _WARM_AXIS, _WARM_GAIN, _WARM_HI_BAND, _WARM_LO_BAND
from ..masks import _grain_delta, _source_masks
from ..noise.fields import _fbm
from ..primitives import (
    _blur, _hsv_to_rgb, _isophote, _luma, _smootherstep, _smoothstep, _warp,
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
        hp_r = max(0.3, p["highpass_radius"] * scale)
        mb = p["micro_blur"] * scale

        # -1. Colour grading, above everything -- temperature, shadows,
        #     highlights, clarity, then the 3D LUT. See _grade.
        #
        #     Above `pre_blur` rather than anywhere else because this is the
        #     decision about what the photograph *is*; every stage below it is
        #     the emulsion's response to that photograph. Putting it after the
        #     film stages would mean grading grain, halation and dust along with
        #     the picture, and a LUT built to be fed a photograph would be being
        #     fed a rendered negative instead.
        img = self._grade(img, p, scale)

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

        # ---- EXPOSURE STAGE (linear light) --------------------------------
        # Diffusion and halation are things that happen to *light*, before the
        # emulsion records anything, so they are done in linear light. Doing
        # them in gamma-encoded space is the usual reason simulated halation
        # looks like a painted-on glow rather than light.
        lin = _srgb_to_linear(img)

        # 1. Diffusion resolved as discrete deflections rather than as an
        #    average -- see _scatter for why that is a different operation and
        #    not a slower blur. In linear light because it happens to the light,
        #    before the emulsion records anything.
        #
        #    **Ahead of micro-blur, and the order is deliberate** (changed
        #    2026-08-03, on request). Both model the same physical event from
        #    opposite ends, and which runs first changes the result a long way --
        #    measured on separate plates at scatter 0.85 / reach 3 / blur 1px,
        #    against the same stages alone:
        #
        #    | | fine texture | hard edge |
        #    |---|---|---|
        #    | scatter alone | 100% | 100% |
        #    | micro-blur alone | 28% | 34% |
        #    | blur then scatter (old) | 28% | **60%** |
        #    | scatter then blur (new) | 32% | **28%** |
        #
        #    The edge column is the whole story, and the old order's number is
        #    the surprising one: **scatter was undoing the blur.** Displacing a
        #    blurred gradient by whole pixels drops a hard step back into it, so
        #    the pair came out *harder* on borders than the blur alone -- 60%
        #    against 34% -- which is not a thing either stage claims to do.
        #
        #    This way round each stage does its own job. Scatter gets the
        #    source's own detail at full contrast and shreds the border into
        #    raggedness; the blur then averages that raggedness into a genuinely
        #    soft transition, ending *below* blur-alone at 28%. Fine texture
        #    barely notices the swap (28% -> 32%) because scatter does not touch
        #    texture sigma either way. It is also the physical order: light
        #    deflects off a grain and then goes on diffusing.
        #
        #    Note the masks below are measured from the *untouched* tile input,
        #    so scattering the frame does not talk the edge mask or the
        #    smooth-area guard into turning grain down -- the same independence
        #    micro-blur has, and for the same reason.
        if p["scatter"] > 0.001:
            lin = self._scatter(lin, h, w, y0, x0, p, scale)

        # 1b. Light diffusing sideways through the gel layers, as an average.
        #     Last in the light path -- see above.
        lin = _blur(lin, mb)

        # 1c. Anti-aliasing -- stair-stepping off the incoming file's hard
        #     edges, filtered along the contour so the edge itself stays put.
        #
        #     Here rather than at the top of the pipeline because an
        #     anti-alias filter is an *optical* element: on a camera it is a
        #     birefringent plate in front of the sensor, so it belongs in the
        #     light path beside the other two optical stages and ahead of
        #     anything the emulsion does. It also has to run before the masks
        #     are measured, or the grain would keep keying on the jaggies this
        #     just removed.
        if p["aa_strength"] > 0.001:
            lin = self._antialias(lin, p, scale)

        # 2. Halation: light reaching the film base reflects and re-exposes the
        #    emulsion from behind, blooming warm around bright highlights.
        hal = p["halation"]
        if hal > 0.01:
            thr = min(p["halation_threshold"], 0.98)
            thr_lin = ((thr + 0.055) / 1.055) ** 2.4
            lum0 = _luma(lin)
            hi = ((lum0 - thr_lin) / max(1.0 - thr_lin, 0.02)).clamp(0.0, 1.0)
            glow = _blur(hi, max(1.0, p["halation_radius"] * scale))

            # 2a. Blue compensation, applied to the image the wash is about to
            #     land on rather than to the result.
            #
            #     Halation adds warm light, and *adding light desaturates
            #     whatever it lands on* -- that is not a side effect to be
            #     tuned out, it is what addition does. A red-tinted bloom
            #     lifts a blue sky's red channel by the full glow and its blue
            #     channel by a tenth of it, so the sky loses colour and drifts
            #     toward grey and then toward purple.
            #
            #     Correcting afterwards was the obvious alternative and is
            #     worse for two measured reasons. It has no brake: the wash
            #     eats a fixed share of anything added *before* it, so
            #     compensating here self-limits -- everything from amount 1.0
            #     to 3.0 lands 3% past the untouched sky's own saturation --
            #     where the identical correction applied *after* is 9% past by
            #     0.5 and by 1.0 has driven a channel to black and pinned the
            #     sky at fully saturated. And it cannot tell blue
            #     that was unfairly washed from blue the bloom is *supposed*
            #     to be sitting on, so re-saturating there fights the glow you
            #     paid for -- it would need the glow field carried out of this
            #     block to know the difference. Here the question never
            #     arises: this changes what was recorded, and halation then
            #     does its job to it. That is also the physical order --
            #     a punchier blue layer or a polariser, not retouching.
            #
            #     Deliberately after `glow` is computed, so compensation
            #     cannot move the bloom: the two controls stay independent and
            #     `verify.py` pins it. Purely per-pixel, so `pad_for` is
            #     unaffected.
            blue = p["halation_blue"]
            bshift = p["halation_blue_shift"]
            if blue > 0.001 or abs(bshift) > 0.5:
                lin = self._blue_guard(
                    lin, blue, p["halation_blue_level"],
                    p["halation_blue_falloff"], bshift,
                )
            # Tint from a full hue wheel rather than the old red-to-amber
            # ramp, which spanned about 25 degrees and could not desaturate.
            # Real halation is red -- that is what the antihalation layer and
            # the red-sensitive layer conspire to produce, and 0-40 degrees is
            # the physically honest region -- but this is a look tool, so the
            # rest of the wheel is reachable.
            tint = torch.tensor(
                _hsv_to_rgb(p["halation_hue"], p["halation_sat"]),
                device=lin.device, dtype=lin.dtype,
            ).view(1, 3, 1, 1)
            add = glow * tint * (hal * 0.9)

            # 2b. Highlight recovery: add the bloom into the headroom that is
            #     actually there, instead of adding it flat and letting the
            #     total clip.
            #
            #     The stage exists because halation adds light in linear space
            #     with no ceiling until display space, so a highlight already
            #     near white gets pushed the rest of the way to a flat,
            #     textureless clip -- reported as halation burning highlights
            #     out. Holding the *glow* back was the first answer and it is
            #     the wrong one: it buys headroom by deleting the bloom, so the
            #     highlights stop burning because the effect stopped happening
            #     there. It also cannot restore anything, because two pixels
            #     that both clipped are still both at 1.0 afterwards.
            #
            #     What this does instead is meter the light against the room
            #     that is actually left. With ``H = 1 - lin`` the headroom each
            #     channel still has,
            #
            #         add' = add * (H + add * (1 - r)) / (H + add)
            #
            #     which at ``r = 1`` is ``add * H / (H + add)``. Three
            #     properties, and each is why it is this expression and not one
            #     of the others tried:
            #
            #     * **Free where there is room.** For ``add << H`` it is ``add``
            #       to first order, so an ordinary highlight with headroom to
            #       spare gets the whole bloom at full strength and the control
            #       costs nothing there. Only a pixel being asked to take more
            #       light than it can hold is metered at all.
            #     * **Cannot reach white at r = 1**, since ``add' < H`` strictly.
            #     * **Strictly increasing in ``lin``**: d(out)/d(lin) =
            #       ``1 - r * a^2 / (H + a)^2``, bounded below by ``1 - r`` and
            #       positive throughout. Nothing flattens, so nothing is lost.
            #
            #     An exponential soft-add -- ``lin + H(1 - exp(-add/H))``, the
            #     obvious tone-mapping answer -- was built and measured first
            #     and is *worse*, which is worth recording because it looks
            #     better on paper. It bends from the origin, so it compresses
            #     hard even where the bloom was modest: on a bright plate
            #     carrying real fine texture it held only 51% of that texture
            #     against this expression's 60%, at less bloom retained. Bending
            #     late beats bending smoothly when what you are protecting is
            #     local contrast rather than the peak value.
            #
            #     Measured on that plate (mean 0.93, fine texture, halation 0.9
            #     at threshold 0.6), against holding the glow back at the same
            #     setting -- highlight texture kept / bloom light kept:
            #
            #     | recovery | hold the glow back | meter against headroom |
            #     |---|---|---|
            #     | 0.5 | 56% / 82% | 53% / 91% |
            #     | 1.0 | 55% / 52% | **60% / 68%** |
            #
            #     At full strength it is better on both axes at once, which is
            #     the whole claim: more of the highlight's detail survives *and*
            #     more of the bloom does.
            #
            #     Keyed on real per-channel headroom rather than on `hi`, the
            #     threshold field the old version used. `hi` answers "is this
            #     pixel bright enough to bloom", which is not the question: a
            #     saturated highlight can sit far above the threshold in luma
            #     while one of its channels still has most of its range free,
            #     and only that channel's own headroom knows so.
            #
            #     What is left on the table, measured: the remaining loss is
            #     compression, not clipping, and it is forced -- red here is
            #     asked to absorb 0.63 of linear light into 0.15 of headroom, so
            #     no metering can be free. The way past it is not a better curve
            #     but a better *model*: real halation is light that *left* the
            #     highlight to re-expose its surroundings, so an
            #     energy-conserving bloom would darken the core as it lights the
            #     halo and the core's texture would survive intact. That is a
            #     change to what halation *is* rather than to this dial, and it
            #     would move every preset that uses the stage, so it is not done
            #     here.
            #
            #     Still purely per-pixel, so `pad_for` is unaffected -- same as
            #     blue compensation above.
            recover = p["halation_recovery"]
            if recover > 0.001:
                head = (1.0 - lin).clamp_min(1e-4)
                add = add * (head + add * (1.0 - recover)) / (head + add)
            lin = lin + add

        # ---- DEVELOPMENT STAGE (density / display space) ------------------
        base = _linear_to_srgb(lin)

        # 3. Brightness, then the characteristic curve: toe, straight line,
        #    shoulder.
        #
        #    Brightness is a multiply in *linear* light, which is what makes it
        #    behave like exposure rather than like a levels slider: doubling
        #    the light doubles it everywhere, and the sRGB encoding on the way
        #    back rolls the top off by itself. Multiplying the display-referred
        #    signal instead would stretch the highlights straight into a flat
        #    clip.
        #
        #    Before the curve, not after, so the shoulder catches the
        #    highlights brightness raises instead of being applied to the
        #    unbrightened image and then overrun.
        br = p["brightness"]
        if abs(br) > 0.001:
            base = _linear_to_srgb(_srgb_to_linear(base) * (2.0 ** br))
        base = _characteristic_curve(base, p["contrast"], p["toe"], p["shoulder"])

        # 4. Dye layers desaturate as they approach saturation, rather than
        #    clipping to a hue-shifted edge the way a sensor does.
        hd = p["highlight_desat"]
        if hd > 0.01:
            lum_h = _luma(base)
            wgt = _smoothstep(0.62, 1.0, lum_h) * hd
            base = base + wgt * (lum_h - base)

        # 4b. Vibrance: a saturation push weighted *against* how saturated a
        #     pixel already is, so muted colour comes up while colour that is
        #     already strong is left alone. That weighting is the whole
        #     difference from a flat saturation control, which drags everything
        #     up together and takes already-saturated regions straight out of
        #     gamut -- skin and skies being the usual casualties.
        #
        #     Saturation is measured as chroma over value, the HSV definition,
        #     which reads a deep red as fully saturated regardless of how dark
        #     it is. Distance from the luma axis would call the same red
        #     unsaturated and then boost it further.
        vib = p["vibrance"]
        if abs(vib) > 0.001:
            mx = base.amax(dim=1, keepdim=True)
            mn = base.amin(dim=1, keepdim=True)
            sat = (mx - mn) / mx.clamp_min(1e-4)
            lum_v = _luma(base)
            # Clamped at zero so a strong negative setting lands on neutral
            # grey rather than inverting the colour through it.
            gain = (1.0 + vib * (1.0 - sat)).clamp_min(0.0)
            base = lum_v + (base - lum_v) * gain

        # 5. Split tone: a cross-channel bias on each end of the range. Most of
        #    what reads as "a film palette" lives here, not in the grain.
        #
        #    **Both controls are signed** (rewritten 2026-08-06, on request).
        #    They were `warm_highlights` and `cool_shadows`, each 0..1 and each
        #    locked to one direction, so the panel could describe warm-over-cool
        #    and nothing else -- not tungsten stock's cool highlights, not a
        #    cross-process, not warm shadows under a cold sky. Now each end of
        #    the range runs cool at -1 through neutral at 0 to warm at +1, which
        #    is the same two stages with the sign let out.
        #
        #    They were also reported as invisible, and they were: see
        #    `_WARM_GAIN` for the arithmetic. Both the amplitude and the two
        #    weighting bands were widened along with the sign.
        #
        #    One axis for both, in opposite directions, rather than a separate
        #    "warm" and "cool" vector. Two hand-written vectors are two things
        #    that can drift apart; a signed push along one axis is warm and cool
        #    by construction, and it is what makes 0 exactly neutral rather than
        #    approximately so.
        hw_, sw_ = p["highlight_warmth"], p["shadow_warmth"]
        if abs(hw_) > 0.001 or abs(sw_) > 0.001:
            lum_s = _luma(base)
            axis = torch.tensor(_WARM_AXIS, device=base.device,
                                dtype=base.dtype).view(1, 3, 1, 1) * _WARM_GAIN
            w_hi = _smoothstep(*_WARM_HI_BAND, lum_s) * hw_
            w_lo = (1.0 - _smoothstep(*_WARM_LO_BAND, lum_s)) * sw_
            base = base + (w_hi + w_lo) * axis

        # 6. Base fog: the film base has a minimum density, so there is no true
        #    black. Lifts the floor without touching the white point.
        fog = p["base_fog"]
        if fog > 0.001:
            base = fog + (1.0 - fog) * base

        base = base.clamp(0.0, 1.0)

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
        lum_d = _luma(base)
        lum_m = _blur(lum_d, max(1.0, 3.0 * scale))
        lo = p["lum_low"]
        hi = max(p["lum_high"], lo + 0.05)
        sf = max(p["shadow_falloff"], 1e-3)
        hf = max(p["highlight_falloff"], 1e-3)

        up_ramp = _smootherstep(max(0.0, lo - sf), lo, lum_m)
        dn_ramp = 1.0 - _smootherstep(hi, min(1.0, hi + hf), lum_m)
        m = (1.0 - p["shadow_drop"]) + p["shadow_drop"] * up_ramp
        m = m * ((1.0 - p["highlight_drop"]) + p["highlight_drop"] * dn_ramp)

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
        lum_ref = _luma(img)
        hp = lum_ref - _blur(lum_ref, hp_r)
        edge = (hp.abs() / EDGE_REF).clamp(0.0, 1.0)
        edge = _blur(edge, hp_r * 0.8)

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
            step = (lum_ref - _blur(lum_ref, sr)).abs()
            hard = _smoothstep(_STEP_LO, _STEP_HI, step)
            hard = _blur(hard, sr * 0.6)
            base = base + (_blur(base, sr) - base) * (hard * es)
            lum = _luma(base)

        # A smooth envelope traces an edge too precisely and reads as a digital
        # outline. Emulsion erodes an edge unevenly, so break the envelope up
        # with its own noise field (mean preserved at ~1.0).
        edge_clean = edge
        ragged = _fbm(
            h, w, y0, x0, max(_MIN_CELL, p["grain_size"] * scale * 2.0),
            int(p["seed"]) + 4241, 1, 2, 0.6, self.device,
        )
        edge = edge * (0.55 + 0.9 * ragged)

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
            base = _warp(base, dx * amp, dy * amp)
            lum = _luma(base)

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
                        base if offv == 0.0
                        else _warp(base, tx * (offv * sr), ty * (offv * sr))
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
                base = base + (sanded - base) * (edge_clean * coherent * snd)
                lum = _luma(base)

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

        # 12. Adjacency (Eberhard) effect. Developer exhausts faster on the
        #     dense side of an edge and diffuses across it, leaving a local
        #     contrast boost. Extracted from the pre-grain base so it sharpens
        #     the image rather than amplifying the grain we just added.
        acut = p["acutance"]
        if acut > 0.01:
            out = out + (base - _blur(base, hp_r * 1.5)) * (0.35 * acut)

        # 13. Global grain -- five overlay layers, applied last. The first is
        #     masked by nothing; the other four by the picture itself.
        #
        #     Everything above is masked: by the luminance band, by the edge
        #     envelope, by the smooth-area guard. That is emulsion behaviour,
        #     and it is why smooth skies and skin stay clean. This layer is
        #     deliberately none of that. It sits on the finished frame at one
        #     amplitude everywhere, the way a scanned print carries grain from
        #     the print stock and the scan itself rather than from the
        #     negative -- so it reaches exactly the areas the masks protect.
        #
        #     On its own seed offset: sharing the main grain's seed would lay it
        #     directly on top of the same clumps and read as nothing more than a
        #     louder version of the same field. Monochrome unless
        #     `global_chroma` asks otherwise -- see below for why that is built
        #     as a separate mean-zero field rather than by the main grain's
        #     recipe.
        #
        #     Min and Max are the two ends of one grain-size distribution, and
        #     since 2026-08-05 they select nothing else: `_grain_points` draws
        #     every setting, Min == Max included. It used to be two
        #     constructions, value-noise fBm below Max and a cellular field
        #     above it, and the switch between them was a change in *kind* --
        #     the layer's whole character, and 43% of its loudness, turned on
        #     whether Max happened to exceed Min. Both were also reported as
        #     showing a visible grid, from different causes. See `_GRAIN_ROT`.
        #
        #     Since 2026-08-05 the section renders **five** such layers, not
        #     one. The other four are the same field on their own seeds, each
        #     multiplied by an envelope read off the picture -- see the masks
        #     below. The flat layer stays exactly what it was and stays first,
        #     so a shadow the masks turn down is never left perfectly clean.
        go = p["global_opacity"]
        # Amounts in layer order: the flat layer, then the four masked ones,
        # matching `_GLAYER_SEEDS` and `_source_masks`.
        gamt = (p["global_intensity"],) + tuple(p[k] for k in _GSRC_KEYS)
        gmode = int(round(p["global_blend"]))
        gcell = max(_MIN_CELL, p["global_size"] * scale)
        # Max can never pull the effective ceiling *below* Min: clamped up
        # to it rather than swapped with it -- the two are not a symmetric
        # pair the way the light-leak sizes are, because Min already has an
        # established meaning on its own and Max is purely "how much
        # further can it stretch".
        #
        # Derived out here rather than inside the branch because all five
        # layers need the identical pair: two derivations that could drift
        # apart would put them on different lattices while every slider
        # claimed otherwise.
        gcell_max = max(gcell, p["global_size_max"] * scale)

        if go > 0.001 and any(a > 0.01 for a in gamt):
            # The four envelopes, read off the frame **before** any of the five
            # layers goes on, so they describe the picture rather than the grain
            # already laid over it. Built once and only if something wants one.
            masks = None
            if any(a > 0.01 for a in gamt[1:]):
                masks = _source_masks(out.clamp(0.0, 1.0))

            # Composited in order, each onto the result of the one before, the
            # way a stack of layers in an image editor behaves -- which is what
            # `Blend Mode` has to mean for the menu to be worth having. Under
            # Add, the default, that is identical to summing them.
            #
            # **Masking, not seeding.** The obvious reading of "grain that
            # follows the picture" is to derive each grain's *seed* from the
            # source pixel, and it fails three ways at once: a flat region
            # hashes every pixel the same, rebuilding the axis-aligned 1px grid
            # `_GRAIN_ROT` exists to destroy; one grain per pixel centred on
            # that pixel makes every falloff 1, so the construction collapses to
            # a blur of white noise with no gaps and no grain edges; and a seed
            # drawn from the frame changes with every upstream slider, so grain
            # rerolls and swims while you grade. A mask has none of that. The
            # pattern comes from the seed as it always did and only the envelope
            # moves, which is also what keeps the fields cacheable -- they read
            # no image data, the mask does, and the mask is applied out here.
            #
            # The five amounts are likewise applied out here, outside the cache
            # boundary, so dragging any of them cannot miss it.
            for li, amt in enumerate(gamt):
                if amt <= 0.01:
                    continue
                g = self._global_grain_field(
                    h, w, y0, x0, p, gcell, gcell_max, li,
                )
                a = (amt / 100.0) * _AMP_SCALE * go
                d = _grain_delta(out, g, gmode)
                out = out + (d * a if li == 0 else d * (a * masks[li - 1]))

        # 14. Output sharpening -- deliberately the last thing in the pipeline.
        #
        #     An unsharp mask amplifies whatever high-frequency content it
        #     finds, and by this point that is the grain as much as the image.
        #     That is the entire reason it sits here rather than earlier: it
        #     cranks the noise already present instead of generating any, so
        #     grain gains bite and the picture gains acutance from the same
        #     operation. Run before the grain stages it would sharpen a clean
        #     image and leave the grain flat, which is the opposite of the
        #     intent.
        #
        #     Distinct from `acutance`, which is an edge-local development
        #     effect extracted from the *pre-grain* base specifically so it
        #     sharpens the image without amplifying grain. This one is the
        #     blunt instrument, and it is applied to the unclamped signal so
        #     overshoot keeps its headroom until the final clamp.
        sh = p["sharpen"]
        if sh > 0.01:
            out = out + (out - _blur(out, max(0.3, p["sharpen_radius"] * scale))) * sh

        # 15. Physical damage, after everything including sharpening -- a
        #     speck of dust sits on the film, it was never in the picture, so
        #     it must not be sharpened, grained or masked along with it.
        out = self._film_texture(out, h, w, y0, x0, p, scale, full_hw)

        return out.clamp(0.0, 1.0)
