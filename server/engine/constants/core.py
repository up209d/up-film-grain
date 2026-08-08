from __future__ import annotations


# Luma coefficients (Rec. 709).
_LUMA = (0.2126, 0.7152, 0.0722)

# Fixed reference for normalising high-pass edge magnitude into 0..1. Must be a
# constant rather than a per-image statistic, or tiles would normalise
# differently and seam.
EDGE_REF = 0.06

# Normalising divisor applied to the raw noise field before the clump curve.
# Measured field std is ~0.27, so 0.55 puts roughly 2 sigma at full scale and
# clips only ~3.6% of samples -- tight enough for the clump curve to bite,
# loose enough to leave the distribution's tails intact. Constant, not a
# per-image statistic, so tiles stay seamless.
_GNORM = 0.55

# The Global Grain texture cache's byte cap **is not here any more** (moved
# 2026-08-08). It is `device._grain_cache_bytes()`, derived from the same pool
# `tile_for` sizes tiles against.
#
# It was a flat 0.5GB, and it was wrong in a way a constant cannot fix: the
# comment sizing it described tile 1536 and *one* layer, both of which stopped
# being true when `tile_for` started computing the tile and the section grew to
# five layers. `SuperPortra` at a 2400px proxy needs 922MB, so the LRU held two
# entries of five and every render missed all of them -- measured 0 hits, 5
# misses, on re-renders with identical parameters.
#
# It cannot live here regardless, now that it is derived: rule 1 in
# `docs/architecture.md` is that `engine/constants/` imports nothing from the
# rest of the engine, and this needs `_render_budget_bytes`. The budget split
# between renderer and cache belongs with the other memory decisions anyway.

# Converts the 0..100 intensity slider into image-referred amplitude. Chosen so
# the default intensity of 32 lands near 3.5% luminance sigma in the midtones,
# which is about right for a 400-speed stock viewed at 100%.
#
# Was 0.5. Recalibrated to 0.38 when _fbm started preserving variance across
# octaves: the old normaliser let the field's variance collapse as octaves were
# added, so the default 3-octave field was running at 43% strength and 0.5 was
# compensating for it. Measured back to 99.7% of the previous look on the
# textured patch, with grain and erosion separated (they share the residual).
_AMP_SCALE = 0.38

# Grain finer than this many working pixels cannot be represented, so the
# lattice is clamped. Below Nyquist it would simply alias.
_MIN_CELL = 0.8

# The Global Grain section renders **five layers**, and they are built
# identically: same Size Min, Size Max, Smoothness, Chroma and Seed, through the
# same `_global_grain_field`. They differ in exactly two things -- the seed
# offsets that decide where their grains fall, and the mask that decides where
# they are visible. Index 0 is the flat layer (`global_intensity`, no mask);
# 1-4 are the source-masked set.
#
# The amount sliders for 1-4, in that index order. Their *names* refer to the
# mask and nothing else -- Source Red is a full-colour grain layer masked by how
# red the picture is, not a layer confined to the red channel. All five write
# into all three channels and all five take `global_chroma`.
_GSRC_KEYS = ("global_src_r", "global_src_g", "global_src_b", "global_src_l")

# Per layer: (mono seed offset, chroma seed offset).
#
# Layer 0's pair is the flat layer's historical 7717/3391 and **must not
# change** -- every shipped preset was dialled in against that exact field, and
# a different offset would reroll all of them while rendering something
# perfectly plausible.
#
# The other four are spaced so that no offset equals another's `+991`, which is
# the cluster draw inside `_grain_points`. A collision there would have two
# layers sharing the clump pattern that decides where grain bunches up -- not
# obviously wrong in a render, and the pair would quietly read as one layer.
_GLAYER_SEEDS = (
    (7717, 3391),
    (11003, 12007),
    (13009, 14011),
    (15013, 16033),
    (17011, 18013),
)

# Global-grain smoothing: a blur on the finished grain field, with the
# amplitude it costs put back analytically.
#
# It was built as a *repair*: the layer used to be value noise, whose extrema
# sit on an axis-aligned lattice, so past roughly 8px its cells read as
# rectangles -- measured on a cell-20 field, |gradient| binned by phase within
# a cell swung by 1.74x its own mean. `_grain_points` has no such quilt to
# remove (0.09 on the same metric before any smoothing), so this is now a shape
# control: it rounds grains off and softens where they meet.
#
# _SMOOTH_MAX  peak blur sigma as a fraction of the clump, at Smoothness 1.
#              Half a clump is where a grain's own edge is gone rather than
#              merely eased, which is as far as a shape control needs to reach.
# _SMOOTH_GAIN_K  restores the amplitude the blur costs, as
#              sqrt(1 + k(sigma/cell)^2). Analytic on purpose: reading the
#              tile's own std would restore a different amount per tile and
#              seam the export.
#
#              **Fit it against the field it is actually used on**, which this
#              constant has now had to learn three times. Calibrated on
#              single-octave value noise it came out 7.7; against the two-octave
#              fBm the layer used to be, 5.62, because a coarse octave survives
#              a blur far better and 7.7 over-restored enough to make full
#              Smoothness 10% *louder* than none. `_grain_points` goes the other
#              way: a field of discrete grains carries far more of its energy at
#              its own edges, so the same blur takes much more of it. Shipping
#              the old 5.62 against it under-restored by 21%, which `verify.py`
#              caught as Smoothness quietly turning the layer down.
#
#              And on this field `k` is **not one number**: it depends on the
#              Min/Max ratio, from 18.3 at a wide range down to 13.5 at a single
#              size. Wide ranges contain small grains, small grains are fine
#              structure, and fine structure is what a blur takes first --
#              measured, 43% of the field survives sigma/cell 0.5 at ratio 0.25
#              against 48% at ratio 1.0. One constant cannot hold better than 6%
#              across that; the quadratic below holds **2.1%**, and it is the
#              same device `_grain_gain` already uses for the same reason (a
#              closed form in a scale-free ratio, never a measurement).
#              Coefficients highest power first; the loss still depends on
#              sigma/cell alone at a fixed ratio, so nothing here varies with
#              the clump's absolute size.
_SMOOTH_MAX = 0.5
_SMOOTH_GAIN_K_FIT = (-8.9785, 4.6986, 17.5924)

__all__ = [
    '_LUMA',
    'EDGE_REF',
    '_GNORM',
    '_AMP_SCALE',
    '_MIN_CELL',
    '_GSRC_KEYS',
    '_GLAYER_SEEDS',
    '_SMOOTH_MAX',
    '_SMOOTH_GAIN_K_FIT',
]
