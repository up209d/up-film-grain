from __future__ import annotations

import math

import torch

from ..constants.core import (
    _AMP_SCALE, _GLAYER_SEEDS, _GNORM, _GSRC_KEYS, _MIN_CELL,
)
from ..device import _grain_cache_bytes
from ..masks import _grain_delta, _source_masks
from ..noise.fields import _fbm, _smooth_noise
from ..noise.grain import _grain_points
from ..primitives import _smoothstep

class GlobalGrainMixin:
    """The flat overlay applied after everything else, and its cache.

    Deliberately weighted by no image mask -- it stands in for grain that
    arrives with the print stock or the scan, and is the only way to put
    grain into the smooth regions the emulsion masks exist to protect.
    """

    def _grain_field(
        self, h: int, w: int, y0: float, x0: float, lum: torch.Tensor,
        p: dict, scale: float,
    ) -> torch.Tensor:
        """Signed, roughly unit-scale grain field, shape [1,3,h,w]."""
        dev = self.device
        cell = max(_MIN_CELL, p["grain_size"] * scale)
        seed = int(p["seed"])
        octaves = int(round(p["octaves"]))
        rough = p["roughness"]

        n = _fbm(h, w, y0, x0, cell, seed, 3, octaves, rough, dev)

        # Shadows carry larger, less densely packed crystals.
        ss = p["shadow_size"]
        if ss > 0.02:
            big = _fbm(
                h, w, y0, x0, cell * (1.0 + 1.2 * ss), seed + 5077, 3,
                max(1, octaves - 1), rough, dev,
            )
            sw = ss * (1.0 - _smoothstep(0.0, 0.6, lum))
            n = n * (1.0 - sw) + big * sw

        s = n * 2.0 - 1.0

        # Monochrome component is the mean of the three dye layers, rescaled to
        # preserve variance; chroma_grain blends toward independent layers.
        mono = s.mean(dim=1, keepdim=True) * math.sqrt(3.0)
        g = mono + p["chroma_grain"] * (s - mono)

        # Clump curve: push the distribution toward discrete clumps.
        t = (g / _GNORM).clamp(-1.0, 1.0)
        gamma = 1.0 - 0.75 * p["clump"]
        if abs(gamma - 1.0) > 1e-3:
            t = torch.sign(t) * t.abs().clamp_min(1e-6) ** gamma
        return t

    # ------------------------------------------------------------------ #
    def _global_grain_field(
        self, h: int, w: int, y0: float, x0: float, p: dict,
        gcell: float, gcell_max: float, idx: int = 0,
    ) -> torch.Tensor:
        """One Global Grain texture layer, normalised and clamped. Cached.

        Shape is ``[1,1,h,w]`` at chroma 0 and ``[1,3,h,w]`` above it; both
        broadcast against the frame, which is why the channel count is allowed to
        depend on a parameter.

        ``idx`` picks the layer -- 0 is the flat one, 1-4 the source-masked set
        -- and it selects **nothing but a pair of seed offsets** out of
        `_GLAYER_SEEDS`. All five layers are otherwise the same field through the
        same code: same size range, same smoothing, same mottling, same chroma
        construction, same normalise-and-clamp. That is what puts the five amount
        sliders on one scale before their masks take a share, and what makes Size
        Min, Size Max, Smoothness, Mottling and Chroma Grain mean one thing
        across the section rather than five.

        Different offsets per layer is the whole reason they are separate calls
        rather than five brightness fields off one geometry: a red-masked grain
        and a blue-masked grain have to sit in genuinely *different places*, the
        way separate emulsion layers do. `global_seed` moves all five together
        and leaves those relative offsets alone, so reshuffling the section
        cannot accidentally collapse two layers onto each other. Sharing geometry is the deliberate
        choice one level down, inside `global_chroma`, where a single grain takes
        a colour without its edge moving from channel to channel -- both are
        wanted, which is why they are two mechanisms and not one slider.

        Layer 0 with ``idx`` at its default is byte for byte the field this
        method built before the set existed, because `_GLAYER_SEEDS[0]` is its
        historical ``7717/3391``. That is the property every shipped preset
        depends on, and folding the source layers in here rather than into a
        parallel copy of this function is what keeps it *provable* -- a second
        implementation is a second thing to drift.

        **Cached because it reads no image data at all.** Every input is either a
        parameter or the tile's own global coordinates, so nudging Halation or
        Sharpen currently pays to rebuild a texture that has not changed --
        measured at 1.29s of a 3.70s `Stock` proxy preview, 35%. The two sliders
        in this section anyone actually drags, `global_intensity` and
        `global_opacity`, are applied by the *caller* as a single scalar multiply
        and so sit outside this boundary entirely: they cannot miss the cache.

        The cache key has to cover every input, and the failure mode if it does
        not is the nasty kind -- a stale hit renders a perfectly plausible
        texture that is simply the previous one, so nothing looks broken. What is
        in it, and why:

        * ``y0, x0, h, w`` -- **absolute global coordinates, never a tile
          index.** The field is addressed globally (invariant 1), so keying on
          anything relative would hand one tile another tile's texture and seam
          every export while every preview looked fine.
        * ``gcell, gcell_max`` -- the *derived* working cells, not the raw
          sliders. Two different (size, scale) pairs that floor to the same
          working cell genuinely produce the same field and should share an
          entry, and this folds in both `scale` and the supersample level for
          free, since the caller has already multiplied them in.
        * ``idx`` -- five different fields live in the one dict.
        * ``seed + global_seed`` -- their **sum**, since that is all the field
          sees. Keying on the two separately would be equally correct and would
          miss on a pair that had merely swapped which slider carried the total.
        * ``global_smooth`` -- `_smooth_noise` is inside this boundary.
        * ``global_mottle`` -- the cluster depth, which changes both the field's
          texture and, through `_grain_gain`, its normalisation. It is the one
          key here that is cheap to *leave out* and expensive to get wrong: the
          layer looks entirely plausible at any depth, so a stale hit is
          invisible rather than obviously broken.
        * ``global_chroma`` -- decides whether the second field is built at all,
          and changes the returned channel count.
        * the device -- these are device-resident tensors.

        Deliberately *not* keyed on the image or upload: the field never reads
        either. Two photographs of the same dimensions already get an identical
        global-grain field today, because the lattice is addressed in absolute
        coordinates; caching does not change that, it just stops recomputing it.

        Caching the finished field rather than `_lattice_np` is the deliberate
        choice even though the lattice would be more general. The lattice is the
        one array in the pipeline you least want to hold: at `Stock` it is ~58
        points per output pixel (see `_lattice_np`). This is one plane per
        channel at working resolution.
        """
        # `global_seed` is an *offset* on the frame seed, not a seed of its own,
        # so it folds in here and the key needs only their sum. Two properties
        # come out of that shape rather than out of a convention anyone has to
        # remember: Seed still rerolls this section along with the whole frame,
        # and `global_seed` at 0 is bit-identical to the layer that existed
        # before the slider did -- for every preset, including the one that
        # ships a non-default Seed, which an absolute seed here would have
        # rerolled.
        base_seed = int(p["seed"]) + int(p["global_seed"])
        key = (
            idx, h, w, float(y0), float(x0), gcell, gcell_max,
            base_seed, p["global_smooth"], p["global_chroma"],
            p["global_mottle"], str(self.device),
        )
        # Everything after the tile's own coordinates is the *generation* -- the
        # parameter state this field belongs to. Entries from an older
        # generation can never be asked for again: any render that could want
        # one would have to put those parameters back, and then it would be the
        # current generation. Under plain LRU they sit there anyway until
        # pressure evicts them, which on an 8GB machine is most of the budget
        # held for values the user has already dragged past.
        #
        # Two generations are kept, not one. One would make dragging a slider
        # back and forth cost a full rebuild every time, and A/B against the
        # previous value is exactly what people do. It also covers the proxy and
        # the 1:1 render coexisting, since `scale` reaches the key through
        # `gcell` and so makes them different generations.
        gen = key[5:]
        if gen != self._gg_gen:
            self._gg_prev_gen, self._gg_gen = self._gg_gen, gen
            live = (self._gg_gen, self._gg_prev_gen)
            for k in [k for k in self._gg_cache if k[5:] not in live]:
                old = self._gg_cache.pop(k)
                self._gg_bytes -= old.element_size() * old.nelement()
                self.gg_evicted += 1

        hit = self._gg_cache.get(key)
        if hit is not None:
            self._gg_cache.move_to_end(key)
            # Counted apart so a test can tell *which* layer missed. The two
            # share the one LRU and the one byte budget, deliberately -- five
            # layers competing for one allowance rather than five allowances --
            # so a hit rate alone could not say that.
            if idx:
                self.gs_hits += 1
            else:
                self.gg_hits += 1
            return hit
        if idx:
            self.gs_misses += 1
        else:
            self.gg_misses += 1

        off_mono, off_chroma = _GLAYER_SEEDS[idx]

        mottle = p["global_mottle"]

        def field(seed_off: int, nfields: int) -> torch.Tensor:
            return _grain_points(
                h, w, y0, x0, gcell, gcell_max,
                base_seed + seed_off, self.device, nfields, mottle,
            )

        gg = field(off_mono, 1)
        # Before the normalise-and-clamp, not after: the clamp is what
        # gives the field its hard tails, and smoothing a clamped field
        # would leave the plateaus it created and merely round their
        # corners. Smoothed first, the clamp bites on a field that has
        # already lost its extremes, so the rails are reached less often.
        #
        # Referenced against Max, which is the field's own characteristic
        # scale -- the pitch of the lattice `_grain_points` scatters its
        # grains over, and therefore the largest a grain can be.
        gg = _smooth_noise(gg, gcell_max, p["global_smooth"],
                           gcell / gcell_max)
        gg = gg * 2.0 - 1.0

        # Chroma: decorrelate the three channels without touching the
        # monochrome field.
        #
        # The obvious construction is `_grain_field`'s -- draw three
        # independent fields, take their rescaled mean as the monochrome
        # component and blend outward. It is not used here for two reasons.
        # It would replace the single field this layer has always been built
        # from, rerolling every existing preset's global grain at chroma 0;
        # and that blend does not hold amplitude, because the mean and the
        # per-channel fields are correlated -- measured pre-clamp, it dips
        # to 88.8% of its own strength at chroma 0.5 and returns to 99.9% by
        # 1.0, so the slider quietly moves loudness as well as colour.
        #
        # Instead the mono field `m` is kept exactly as it was and a
        # *mean-zero* deviation `d` is added on top, from its own seed.
        # Because `d` sums to zero across channels its statistics are fixed
        # -- var 2/3 and covariance -1/3 of a single field -- and the two
        # coefficients can be solved rather than guessed:
        #
        #     g_c = A*m + B*d_c,   A = sqrt(1 - 2/3 c),  B = sqrt(c)
        #
        # gives unit variance and cross-channel correlation exactly `1 - c`
        # at every setting. Measured: correlation 1.000 / 0.501 / 0.001 at
        # chroma 0 / 0.5 / 1, pre-clamp amplitude flat to 0.6%, and chroma 0
        # bit-identical to the old layer (max channel spread 0.0).
        #
        # The one thing that does move is the clamp below. Mixing in `d`
        # gaussianises the field, so it reaches the rails less often --
        # clipping falls 25.4% -> 22.8% across the slider -- and since a
        # clipped sample sits at exactly +-1 rather than wherever it was
        # headed, less clipping means slightly less measured sigma. Rendered
        # amplitude therefore drifts 100% -> 96.8% from chroma 0 to 1. That
        # is the hard tails doing their job, not the blend, and it is a
        # third of the wobble the other construction has.
        gc = p["global_chroma"]
        if gc > 0.001:
            # Same construction as `gg` above, through the same closure --
            # the decorrelation below only needs `gs` to be a second field
            # of comparable amplitude, but sharing the geometry generator
            # means the chroma field reuses each grain's own position and
            # radius across channels and only randomises its per-channel
            # brightness, which is what gives a coloured grain its speckle
            # without moving its edge from channel to channel.
            gs = field(off_chroma, 3)
            gs = _smooth_noise(gs, gcell_max, p["global_smooth"],
                               gcell / gcell_max)
            gs = gs * 2.0 - 1.0
            gd = gs - gs.mean(dim=1, keepdim=True)
            gg = gg * math.sqrt(1.0 - (2.0 / 3.0) * gc) + gd * math.sqrt(gc)

        gg = (gg / _GNORM).clamp(-1.0, 1.0)

        # LRU insert. An entry larger than the whole budget is returned without
        # being cached rather than immediately evicting itself -- otherwise a
        # single-tile render of a large frame would thrash the cache empty on
        # every pass and pay the bookkeeping for nothing.
        nbytes = gg.element_size() * gg.nelement()
        # Read per insert rather than at import: it is derived from the device's
        # own budget now, and a process that changes `FILM_GRAIN_TILE_BUDGET_GB`
        # to reproduce a small machine has to see the smaller cache too.
        cap = _grain_cache_bytes()
        if nbytes <= cap:
            self._gg_cache[key] = gg
            self._gg_bytes += nbytes
            while self._gg_bytes > cap and len(self._gg_cache) > 1:
                _, old = self._gg_cache.popitem(last=False)
                self._gg_bytes -= old.element_size() * old.nelement()
        return gg

    def _global_grain(
        self, out: torch.Tensor, h: int, w: int, y0: float, x0: float,
        p: dict, scale: float,
    ) -> torch.Tensor:
        """The five overlay layers, applied to the finished frame.

        Extracted from `render()` on 2026-08-08 so the pipeline body reads as
        a sequence of sections. Bit-identical to the inline version.
        """
        # 9. Global grain -- five overlay layers, applied second to last, below
        #     Film Texture since 2026-08-09. The first is masked by nothing; the
        #     other four by the picture itself -- and "the picture" now includes
        #     the dust, the hair and the leaks, so the envelopes follow the
        #     debris. Right for a layer that models the print stock and the
        #     scan: that pass sees the marks.
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
                masks = _source_masks(
                    out.clamp(0.0, 1.0), p["global_src_l_pivot"],
                )

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
        return out
