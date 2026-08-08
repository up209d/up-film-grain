from __future__ import annotations

import math

import torch

from ... import params as P
from ..colour import _linear_to_srgb, _srgb_to_linear
from ..constants.marks import (
    _BLOB_CELLS_SCRATCH, _DUST_DARK_LUM, _DUST_ECCENT_HI, _DUST_ECCENT_LO, _DUST_EDGE_MAX, _DUST_EDGE_MIN, _DUST_EDGE_PX, _DUST_HARMONICS, _DUST_LITE_LUM, _DUST_SOFT_FADE, _HAIR_LUM, _HAIR_TAPER, _HAIR_TIP, _HAIR_WIDTH, _LEAK_GAIN, _LEAK_REACH_SAFETY, _LEAK_WARP, _MARK_MIN_PX, _TEX_LUM_FLOOR,
)
from ..marks import (
    _count_threshold, _dust_sites, _hair_sites, _leak_anchor, _leak_sites, _mark_window,
)
from ..noise.lattice import _value_noise
from ..primitives import _blur, _smoothstep, _spread

class FilmTextureMixin:
    """Dust, hair, scratches and light leaks as marks drawn from absolute
    coordinates.

    Every tile builds the identical site list from the count, the seed and
    the frame, then clips each mark to its own footprint -- which is what
    keeps a per-frame mark population tile-independent.
    """

    def _film_texture(
        self, out: torch.Tensor, h: int, w: int, y0: float, x0: float,
        p: dict, scale: float, full_hw: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Physical damage: dust, scratches, hair, light leaks.

        Everything above this point models what the *emulsion* does. This
        models what happened to the piece of film afterwards -- it got dusty,
        it got dragged through a gate, someone's hair landed on the scanner
        bed, the back came loose. That is why it sits last and is weighted by
        none of the image masks: a scratch does not care what is underneath it.

        All four are drawn by thresholding noise addressed in global
        coordinates rather than by scattering objects. Scattering would need a
        list of positions, and a list is a statistic of the region -- it would
        break tile independence the moment an export split a scratch across
        two tiles. Thresholded noise gives every pixel the same answer no
        matter which tile asks, and it also stops the marks looking stamped:
        their outlines are organic because the field is.
        """
        dev = self.device
        seed = int(p["texture_seed"])
        # Counts are per *frame*, so they need its size. Without it (a caller
        # that did not pass full_hw) the counted marks are skipped rather than
        # guessed at from the tile, which would put N marks on every tile.
        area = None if full_hw is None else max(full_hw[0] * full_hw[1], 1.0)

        # -- light leak ---------------------------------------------------
        # Light that got past a seal, so it is anchored to the frame rather
        # than floating in the image, and it is added in linear light because
        # it is light.
        #
        # Drawn as a handful of discrete *beams*, which is the whole shape of
        # this stage and the thing it got wrong before. The old version was a
        # falloff from the nearest border gated by a slow noise field along it:
        # every leak was therefore a soft inward wash with no direction, no
        # length and no edge, present on all four borders at once -- a chewed-up
        # vignette. Real leaks are streaks with a definite edge limiting their
        # reach; they come from one or two places on the frame, they lean
        # across it, and they stop somewhere.
        #
        # So each leak is a beam: a source on the perimeter, a depth it
        # penetrates (`leak_size_*`), a lean (`shear`), a width that fans out
        # as it travels, and one hard edge where the obstruction's shadow is.
        # Noise now *perturbs* that shape instead of being it.
        ll = p["light_leak"]
        if ll >= 1.0 and full_hw is not None:
            fh = max(float(full_hw[0]), 1.0)
            fw = max(float(full_hw[1]), 1.0)
            Ypx = (torch.arange(h, device=dev, dtype=torch.float32)
                   + float(y0)).view(1, 1, h, 1)
            Xpx = (torch.arange(w, device=dev, dtype=torch.float32)
                   + float(x0)).view(1, 1, 1, w)

            var = p["leak_variation"]
            # Swapped if given the wrong way round, so dragging either slider
            # past the other never makes the leaks vanish.
            s_lo = min(p["leak_size_min"], p["leak_size_max"]) * scale
            s_hi = max(p["leak_size_min"], p["leak_size_max"]) * scale
            # Cap at half the frame's short side over the warp's headroom:
            # that is the depth at which the falloff dies exactly in the
            # middle of the frame, and past it a leak leaves a floor over the
            # whole picture -- centre fog, which reads as a bad exposure
            # rather than as a leak. Geometric, not a taste constant.
            reach_cap = 0.5 * min(fh, fw) / _LEAK_REACH_SAFETY
            # The along-border edges want a softness as a 0..1, and the honest
            # 0..1 is the feather measured against the sizes asked for -- a
            # 50px feather is a rim on a 400px leak and a wash on an 80px one.
            # Derived from the parameters alone, never from the field, so it
            # is a constant per render and tiles cannot disagree about it.
            soft = min(1.0, p["leak_feather"] / max(
                0.5 * (p["leak_size_min"] + p["leak_size_max"]), 1.0))
            bw_soft = 0.12 + 0.75 * soft

            expo_lin = _srgb_to_linear(out)
            # Per-channel exposure, accumulated over the beams. Light adds, so
            # two leaks overlapping is brighter than either -- and it has to be
            # per channel rather than a scalar times one tint, because each
            # leak carries its own hue.
            expos = torch.zeros(1, 3, h, w, device=dev, dtype=torch.float32)

            for k, st in enumerate(_leak_sites(ll, seed, var)):
                border, s0 = _leak_anchor(st["pos"], fh, fw)
                # `u` is the perpendicular depth from this leak's own border
                # and `s` runs along it. Keeping the obliquity in a shear on
                # `s` rather than rotating the whole frame is what lets a leak
                # lean hard across the picture while `reach` stays exactly the
                # depth the slider promises.
                if border == 0:
                    u, s, blen = Ypx, Xpx, fw
                elif border == 1:
                    u, s, blen = fh - Ypx, Xpx, fw
                elif border == 2:
                    u, s, blen = Xpx, Ypx, fh
                else:
                    u, s, blen = fw - Xpx, Ypx, fh

                reach = min(s_lo + (s_hi - s_lo) * st["reach_t"], reach_cap)
                # How far the leak runs *along* its border. Measured against
                # the border, not against the reach -- and that is the second
                # thing the old shape got wrong. A seal fails along a seam, so
                # the leak is a band that runs a long way sideways and comes in
                # a modest depth; sizing its length off its depth instead makes
                # every leak roughly as long as it is deep, which is a blob.
                # Floored against the reach because light through a slot cannot
                # be much narrower than it is deep.
                hw0 = max(blen * st["width"], 0.55 * reach)

                # Two octaves of domain warp. The coarse one wanders the whole
                # beam, the fine one frays its edge; between them the outline is
                # organic while still being an outline -- which is the inversion
                # that matters here. Noise used to *be* the shape and the result
                # was fog; now it perturbs a shape that has a definite edge.
                # The depth amplitudes sum to exactly `_LEAK_WARP * reach`,
                # which is what the reach cap was sized against.
                wn = _value_noise(h, w, y0, x0, max(16.0, 0.80 * reach),
                                  seed + 9137 + k * 37, 3, dev)
                wf = _value_noise(h, w, y0, x0, max(6.0, 0.25 * reach),
                                  seed + 9701 + k * 37, 2, dev)
                warp = (wn[:, 0:1] - 0.5) * 1.5 + (wf[:, 0:1] - 0.5) * 0.5
                # Clamped at zero: the warp may pull the beam *inward*, and
                # the falloff below has to stay defined at the border.
                du = (u + warp * _LEAK_WARP * reach).clamp_min(0.0)
                lat = (wn[:, 1:2] - 0.5) * 1.5 + (wf[:, 1:2] - 0.5) * 0.5
                dv = (s - s0) - st["shear"] * du + lat * 0.18 * hw0

                # Along the beam: the same feather-to-exponent mapping the
                # pixel sizes have always used. Solving (1 - hl/reach)^e = 0.5
                # gives e = ln(0.5) / ln(1 - hl/reach), so the feather is a
                # visible distance -- short is a tight bright rim on the
                # border, half the reach is a straight ramp, most of the reach
                # is a broad wash. Scalars per leak now rather than fields,
                # since a beam has one of each.
                hl = (p["leak_feather"] * scale) * (
                    1.0 + var * 0.45 * (2.0 * st["halo"] - 1.0))
                hl = min(max(hl, 0.5), reach * 0.95)
                expo = math.log(0.5) / math.log1p(-min(hl / reach, 0.95))
                # Floored at *zero*, not at an epsilon: raising a 1e-4 floor
                # to a small exponent gives 0.12, not something small, and
                # that is a fog over the whole beam's footprint.
                along = (1.0 - (du / reach).clamp(0.0, 1.0)).clamp_min(0.0) ** expo

                # Across the beam: narrow at the source and fanning inward,
                # which is what a shaft through a gap does and is most of why
                # this reads as a beam rather than as a band.
                hwid = (hw0 * (0.75 + st["flare"] * du / reach)).clamp_min(1.0)
                q = dv.abs() / hwid
                # One edge is the obstruction's shadow and is much harder than
                # the other. Both soft is haze; both hard is a painted shape.
                bw_hard = max(0.03, bw_soft * (1.0 - 0.95 * st["hard"]))
                on_hard = (dv * st["hard_side"] >= 0.0).to(dv.dtype)
                bw = bw_soft + (bw_hard - bw_soft) * on_hard
                tt = ((1.0 + bw - q) / (2.0 * bw)).clamp(0.0, 1.0)
                across = tt * tt * (3.0 - 2.0 * tt)

                # A beam is not uniform inside itself either -- dust in the
                # chamber, an uneven gap. Mean 1.0, so it modulates without
                # changing the strength the leak was drawn with.
                dens = 0.72 + 0.56 * wn[:, 2:3]

                hue = min(max(p["leak_hue"] + st["hue"], 0.0), 1.0)
                tint = torch.tensor(
                    [1.0, 0.16 + 0.46 * hue, 0.04 + 0.18 * hue],
                    device=dev, dtype=torch.float32,
                ).view(1, 3, 1, 1)
                expos = expos + (along * across * dens * st["strength"]) * tint

            # Saturating response, per channel and per dye layer. A leak's core
            # is *white* with the colour only in its falloff, and no amount of
            # adding a fixed warm ratio can do that -- a fixed ratio stays the
            # same colour at every strength, which is exactly why the old wash
            # read as flat tan everywhere. Each layer saturating separately
            # gives the real progression: deep red where only the red-sensitive
            # layer caught enough light, through orange and yellow, to white
            # where all three are at the top. It also self-limits at 1.0 in
            # linear light, so a hot leak cannot drive a channel past white.
            added = -torch.expm1(-expos * (p["leak_strength"] * _LEAK_GAIN))
            out = _linear_to_srgb(expo_lin + added.to(out.dtype))

        # -- scratches ----------------------------------------------------
        # A gouge through the emulsion lets the light straight through, so on
        # a positive it prints bright. Drawn as noise whose cells are a couple
        # of pixels wide and hundreds tall: that anisotropy *is* the scratch.
        sc = p["scratches"]
        if sc >= 1.0 and area is not None:
            wpx = max(0.4, p["scratch_width"] * scale)
            n = _value_noise(
                h, w, y0, x0, wpx * 2.0, seed + 4409, 1, dev,
                cell_y=max(60.0, 900.0 * scale),
            )
            # A scratch occupies one cell of a very tall, very thin lattice,
            # so its "area" is that cell -- the count then works out the same
            # way as for dust despite the anisotropy.
            cell_x, cell_y = wpx * 2.0, max(60.0, 900.0 * scale)
            th_a = _count_threshold(sc * 2.5, cell_x * cell_y, area, _BLOB_CELLS_SCRATCH)
            th_b = _count_threshold(sc * 0.5, cell_x * cell_y, area, _BLOB_CELLS_SCRATCH)
            line = _smoothstep(th_a, max(th_b, th_a + 1e-4), n)
            # Break them along their length, or every scratch runs the full
            # height of the frame and reads as a printing artifact.
            brk = _value_noise(
                h, w, y0, x0, max(24.0, 300.0 * scale), seed + 4410, 1, dev,
                cell_y=max(8.0, 90.0 * scale),
            )
            line = line * _smoothstep(0.30, 0.72, brk)
            # Variation field shares the scratch's own anisotropy, so softness
            # and density are constant *along* a scratch and differ *between*
            # scratches -- the other way round would make one scratch fade in
            # and out down its length.
            vary = _value_noise(
                h, w, y0, x0, wpx * 6.0, seed + 4411, 2, dev,
                cell_y=max(90.0, 1300.0 * scale),
            )
            out = out + self._weather(
                line, vary, p["scratch_soften"],
                p["scratch_soften"] * 3.0 * max(wpx, 0.6),
                origin=(y0, x0),
            ) * 0.85

        # -- hair ---------------------------------------------------------
        # A hair on the scanner bed is opaque, so it prints as a dark filament,
        # and it is drawn one filament at a time from `_hair_sites` -- see there
        # for why "count 1 drew several hairs" was structural rather than a
        # tuning error, and why a list of objects is still tile-independent.
        # Truncated rather than rounded, so the mark-count dead zone means
        # what `docs/presets.md` says it means: anything under 1 renders nothing,
        # here as for scratches and leaks. Rounding would quietly make 0.6 draw
        # a hair and leave a hand-edited preset behaving differently from the
        # one control the check in `verify.py` was written for.
        hr = int(p["hair"])
        if hr >= 1 and full_hw is not None:
            fh = max(float(full_hw[0]), 1.0)
            fw = max(float(full_hw[1]), 1.0)
            l_nom = max(p["hair_length"], 1.0) * scale
            w_nom = max(_HAIR_WIDTH * scale, 0.35)
            h_soft = p["hair_soften"]

            for st in _hair_sites(hr, seed):
                half = max(0.5 * l_nom * st["len"], 1.0)
                halfw = max(0.5 * w_nom * st["width"], 0.02)
                w1, w2 = st["wob"]
                f1, f2 = st["freq"]
                ph1, ph2 = st["phase"]
                # Per-hair softness, spread about the slider so a frame carries
                # both a hair on the glass and one a layer away at any setting.
                soft = min(h_soft * (0.25 + 1.5 * st["soft"]), 1.0)
                # Edge width across the filament, relative to its own half-width.
                # Floored on the pixel grid: a sub-pixel hair with a hard edge
                # aliases into a dotted line.
                er = min(0.35 + 2.2 * soft, 0.9)
                er = max(er, _DUST_EDGE_PX / halfw)
                # Everything the filament can reach from its own centre. The
                # bend is a fraction of the half-length, so it scales with it.
                reach = (
                    half * (1.0 + abs(st["curve"]) + abs(w1) + abs(w2))
                    + halfw * (1.0 + er) + 2.0
                )
                cy, cx = st["y"] * fh, st["x"] * fw
                win = _mark_window(cy, cx, reach, h, w, y0, x0, dev)
                if win is None:
                    continue
                sl_y, sl_x, dy, dx = win

                ca, sa = math.cos(st["angle"]), math.sin(st["angle"])
                # Along the filament (s, normalised to +-1 at the tips) and
                # across it.
                s = (dx * ca + dy * sa) / half
                across = dy * ca - dx * sa

                # The curve itself, and its slope, both in working pixels. The
                # slope is what turns "vertical distance to the curve" into
                # "perpendicular distance to it" -- without it a bent hair reads
                # as fatter wherever it is steep, which is precisely where the
                # eye looks.
                tau1 = 2.0 * math.pi * f1
                tau2 = 2.0 * math.pi * f2
                sin1, cos1 = torch.sin(tau1 * s + ph1), torch.cos(tau1 * s + ph1)
                sin2, cos2 = torch.sin(tau2 * s + ph2), torch.cos(tau2 * s + ph2)
                curve = half * (st["curve"] * s * s + w1 * sin1 + w2 * sin2)
                slope = 2.0 * st["curve"] * s + w1 * tau1 * cos1 + w2 * tau2 * cos2
                d = (across - curve).abs() / torch.sqrt(1.0 + slope * slope)

                # Taper: a real fibre comes to a point, and a filament of
                # constant width with two blunt ends reads as a line segment.
                sabs = s.abs()
                taper = 1.0 - _smoothstep(_HAIR_TAPER, 1.0, sabs)
                hw_raw = halfw * (_HAIR_TIP + (1.0 - _HAIR_TIP) * taper)
                # Below the grid's floor the tip is drawn *at* the floor and
                # faded by what is missing, rather than drawn thinner -- see
                # `_MARK_MIN_PX`, and note this is the whole reason a tapered
                # hair does not come out dotted.
                #
                # **Twice `_MARK_MIN_PX` for a filament**, i.e. a full pixel of
                # width where a speck needs half a pixel of radius. The two are
                # not the same condition: a disc always has a pixel centre
                # within reach of its own soft edge, but a line can thread
                # between pixel centres for its whole length and hit none of
                # them -- which is exactly what the tip did, measured as a 488px
                # filament with a 4px and a 2px fragment strung out past its end.
                hw_min = 2.0 * _MARK_MIN_PX
                hw_t = hw_raw.clamp_min(hw_min)
                thin = (hw_raw / hw_min).clamp(0.0, 1.0)
                shape = 1.0 - _smoothstep(1.0 - er, 1.0 + er, d / hw_t)
                # And a hard stop at the ends, since the taper alone leaves a
                # thin thread running on past them.
                shape = shape * thin * (1.0 - _smoothstep(0.92, 1.0, sabs))

                lo_, hi_ = _HAIR_LUM
                col = lo_ + (hi_ - lo_) * st["lum"]
                alpha = shape * (
                    st["alpha"] * (1.0 - _DUST_SOFT_FADE * soft)
                )
                sub = out[:, :, sl_y, sl_x]
                out[:, :, sl_y, sl_x] = sub * (1.0 - alpha) + col * alpha

        # -- dust ---------------------------------------------------------
        # Two populations: opaque specks that block light and print dark, and
        # the pinholes and lint that print bright. `dust_balance` sets the
        # split, and both ends are wanted -- dust that is only ever dark reads
        # as sensor dirt rather than as film, which is what it was reported as.
        #
        # Each speck is drawn as its own shape rather than thresholded out of a
        # noise field; `_dust_sites` has the reasoning, and the constants above
        # it have the geometry.
        du = int(p["dust"])
        if du >= 1 and full_hw is not None:
            fh = max(float(full_hw[0]), 1.0)
            fw = max(float(full_hw[1]), 1.0)
            r_nom = 0.5 * max(p["dust_size"], 0.1) * scale
            o_var, l_var = p["dust_opacity_var"], p["dust_lum_var"]
            base_op = p["dust_opacity"]
            d_soft = p["dust_soften"]
            irr = max(p["dust_irregular"], 0.0)

            for st in _dust_sites(du, seed, p["dust_balance"]):
                r = max(r_nom * st["size"], 0.2)
                # Eccentricity ceiling rides `dust_irregular` along with the
                # harmonics, so the slider controls *how far from a circle* a
                # speck can get rather than only how dented its outline is. At 0
                # the population is 90% round; at 1 it reaches well past oval.
                e = st["eccent"] * (
                    _DUST_ECCENT_LO + (_DUST_ECCENT_HI - _DUST_ECCENT_LO) * irr
                )
                ra, rb = r * (1.0 + e), r * (1.0 - e)
                # This speck's harmonic amplitudes. At `dust_irregular` 0 they
                # are all zero, the radius is exactly 1 and the outline is
                # exactly the ellipse -- which is what the slider promises.
                amp = tuple(a * irr * st["rough"] for a in _DUST_HARMONICS)
                bump_max = 1.0 + sum(amp)
                # The per-speck clamp is `max(d_soft, 1.0)` rather than 1.0 so
                # that the mapping below 1 is untouched -- presets carry
                # `dust_soften` values from when 1 was the top of the slider and
                # must still render as they did.
                soft = min(d_soft * (0.25 + 1.5 * st["soft"]), max(d_soft, 1.0))
                # Edge width as a fraction of the speck's own radius, floored on
                # the pixel grid for the reason `_DUST_EDGE_PX` gives. The floor
                # is what the 0.9 clamps: a sub-pixel speck asks for an absurd
                # relative edge, whereas an edge the *softness* asked for is
                # wanted however wide it is.
                edge = _DUST_EDGE_MIN + (_DUST_EDGE_MAX - _DUST_EDGE_MIN) * soft
                edge = max(edge, min(_DUST_EDGE_PX / rb, 0.9))
                reach = ra * bump_max * (1.0 + edge) + 1.0
                cy, cx = st["y"] * fh, st["x"] * fw
                win = _mark_window(cy, cx, reach, h, w, y0, x0, dev)
                if win is None:
                    continue
                sl_y, sl_x, dy, dx = win

                ca, sa = math.cos(st["angle"]), math.sin(st["angle"])
                # In the speck's own frame, scaled by its two semi-axes, so the
                # unit circle *is* its outline before the harmonics dent it.
                u = (dx * ca + dy * sa) / ra
                v = (dy * ca - dx * sa) / rb
                q = torch.sqrt(u * u + v * v)
                if bump_max > 1.0:
                    phi = torch.atan2(v, u)
                    a3, a4, a5 = amp
                    p3, p4, p5 = st["phase"]
                    q = q / (
                        1.0
                        + a3 * torch.cos(3.0 * phi + p3)
                        + a4 * torch.cos(4.0 * phi + p4)
                        + a5 * torch.cos(5.0 * phi + p5)
                    )
                shape = 1.0 - _smoothstep(1.0 - edge, 1.0 + edge, q)

                # Composited rather than added, which is what separates opacity
                # from luminosity. Additively they are the same number: a
                # fainter speck and a lighter speck are indistinguishable. As a
                # composite, opacity is how much of the photograph the speck
                # hides and luminosity is what colour the speck itself is, so a
                # solid grey mote and a faint black veil are different things.
                # **Variation spreads *inward from* full strength, not outward
                # from the middle** (changed 2026-08-08, reported: "the light
                # dust even with Dust opacity at 1 is still not at the brightest
                # level"). It was centred -- `mid + (draw - 0.5) * span * var`
                # -- so at Luminosity Variation 0 every light speck sat at 0.86,
                # the midpoint of its range, and pure white was reachable only
                # by the luckiest draw at variation 1. That is the wrong default
                # twice over: a control named *variation* should do nothing at
                # 0, and what it does nothing to should be the full-strength
                # speck rather than a half-strength one.
                #
                # Now variation 0 puts every light speck at 1.0 and every dark
                # one at 0.0, and raising it walks the population back toward
                # the middle. Note this is still *colour*, not coverage --
                # `dust_opacity` below decides how much of the photograph the
                # speck hides, which is the other half of what "brightest" needs.
                lum_lo, lum_hi = _DUST_LITE_LUM if st["light"] else _DUST_DARK_LUM
                if st["light"]:
                    col = lum_hi - (lum_hi - lum_lo) * (1.0 - st["lum"]) * l_var
                else:
                    col = lum_lo + (lum_hi - lum_lo) * st["lum"] * l_var
                col = min(max(col, 0.0), 1.0)
                # A speck smaller than a pixel fades rather than thinning, for
                # `_MARK_MIN_PX`'s reason -- otherwise it registers only where
                # it happens to land on a pixel centre, so the *count* would
                # quietly depend on the render scale.
                thin = min(1.0, (ra * rb) / (_MARK_MIN_PX * _MARK_MIN_PX))
                # **`dust_opacity` is a reachable ceiling** (changed
                # 2026-08-09, reported as "the dust dot still looks grey at
                # opacity 1"). Three separate terms used to cut it and only one
                # of them was named opacity: at 1 with everything else at its
                # default the brightest pixel measured 0.907 and the median lit
                # speck 0.656.
                #
                # The soft-fade is the one that had no business being there.
                # Out-of-focus debris really is fainter as well as softer -- the
                # same light over a wider footprint -- but expressing that by
                # multiplying the opacity meant the *softness* slider was
                # quietly varying opacity, which is a second control doing the
                # first one's job. It rides under `dust_opacity_var` now, so a
                # variation of 0 means exactly what it says: every speck at
                # `dust_opacity`, whatever its softness.
                fade = 1.0 - o_var * (
                    (1.0 - st["opacity"])
                    + _DUST_SOFT_FADE * min(soft, 1.0) * st["opacity"]
                )
                alpha = shape * min(max(base_op * fade * thin, 0.0), 1.0)
                sub = out[:, :, sl_y, sl_x]
                out[:, :, sl_y, sl_x] = sub * (1.0 - alpha) + col * alpha

        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _weather(
        mark: torch.Tensor, vary: torch.Tensor, soften: float, radius: float,
        lum_floor: float = _TEX_LUM_FLOOR,
        origin: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Make a field of marks non-uniform in sharpness and in brightness.

        A thresholded noise field gives every mark the same crisp edge and the
        same opacity, which is the tell that they were generated: real debris
        is at different depths, so some of it is in focus and some is not, and
        none of it is equally dark.

        **Scratches only, as of 2026-08-06.** Dust and hair are drawn from lists
        now and carry their own per-mark softness, opacity and tone straight off
        the site record -- which is strictly better than this, because a drawn
        mark can vary its *edge width* where a thresholded one can only be
        blurred, and blurring a 2px speck by several times its own size erases
        it rather than softening it. This stays because a scratch is still a
        field, and a field has no per-mark anything to attach a draw to.

        ``vary`` carries two decorrelated fields addressed at mark scale, so a
        whole scratch shares its blur and its density rather than varying
        pixel-to-pixel down its own length. The first drives how far each mark
        blends toward a blurred copy; the second scales its strength.

        Blurring also thins a mark, which is left uncorrected on purpose --
        out-of-focus debris really is both softer and fainter.
        """
        # Spread, not raw. Value noise clusters so tightly around its median
        # (p10-p90 spans 0.41-0.71) that a floor-to-1.0 mapping delivered only
        # a +/-16% spread however wide the range it was given -- which is why
        # the marks still looked uniform. Same fix as the light leaks needed.
        v_soft, v_lum = _spread(vary[:, 0:1]), _spread(vary[:, 1:2])
        if soften > 0.01 and radius > 0.05:
            # `origin` is what lets this take `_blur`'s decimated path, and it
            # matters more here than anywhere else in the pipeline: the radius is
            # `soften * 3 * width`, so a preset's 0.9 on a 14.85px scratch is
            # sigma 40 at scale 1 and **sigma 80 at supersample 2**. Measured
            # 2026-08-08, this one call was 101s of a 154s CPU export.
            blurred = _blur(mark, radius, origin)
            # Centre the field so `soften` sets the *average* blur, with marks
            # either side of it, rather than a floor everything sits above.
            b = (soften * _smoothstep(0.15, 0.85, v_soft)).clamp(0.0, 1.0)
            mark = mark * (1.0 - b) + blurred * b
        # Never all the way to zero: a mark that fades out entirely just thins
        # the population rather than varying it.
        if lum_floor >= 1.0:
            return mark
        return mark * (lum_floor + (1.0 - lum_floor) * v_lum)
