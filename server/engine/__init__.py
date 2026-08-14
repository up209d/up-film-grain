"""Approach A -- edge-destruction procedural grain pipeline.

Design notes that matter for correctness:

* **Scale invariance.** Every spatial quantity (clump size, high-pass radius,
  micro-blur) is specified in *full-resolution* pixels and multiplied by the
  working ``scale`` at render time. The noise lattice is indexed by working
  coordinates divided by the scaled cell size, which equals the full-res
  coordinate divided by the full-res cell size. A supersampled pass and a
  plain one therefore show the same grain structure, not the same pixel noise.

* **Tile independence.** Nothing in the pipeline depends on a statistic of the
  region being rendered -- no per-tile normalisation, no global mean. Edge
  strength is normalised against the fixed ``EDGE_REF`` constant and the noise
  lattice is addressed by absolute global coordinates. Two adjacent tiles
  sampling the same global position get bit-identical values, so tiles composite
  without seams given enough overlap to cover the blur kernels.

* **Not every softening stage is a filter.** ``scatter`` displaces a share of
  the pixels onto their neighbours and averages nothing at all, so it takes the
  image's exactness without taking its micro-contrast. It samples nearest-
  neighbour on whole-pixel offsets precisely so each output pixel stays a copy
  of a real one; measured against a blur of the same reach it keeps 100% of
  fine-texture sigma where the blur keeps 14%. Anything that turns it into an
  average -- bilinear resampling, cross-fading the moved pixel with the
  original -- destroys the only reason it exists.

* **Grain is structural.** Alongside the weighted additive term, the grain field
  multiplies the image's own micro-detail (``edge_erosion``). That term is zero
  in flat areas and grows on edges, so grain erodes existing edge structure
  rather than being stamped over it.

* **One deliberate exception.** The final stage, ``global_*``, is a flat grain
  overlay applied after everything else and weighted by no mask at all. It is
  not emulsion behaviour and is not meant to be -- it stands in for grain that
  arrives with the print stock or the scan, and it is the only way to put grain
  into the smooth regions the masks above exist to protect. It ships at zero.
"""

from __future__ import annotations

from .grain_engine import GrainEngine
from .colour import (
    _MID_GREY, _apply_lut, _characteristic_curve, _linear_to_srgb, _recon_estimate, _reconstruct_highlights, _shoulder, _soft_knee, _srgb_to_linear, _tone_roll,
)
from .device import (
    _GRAIN_CACHE_SHARE, _RENDER_BUDGET_FRACTION, _TILE_BUDGET_SHARE, _TILE_MAX, _TILE_MIN, _WORKING_BYTES_PER_PX, _grain_cache_bytes, _render_budget_bytes, _tile_budget_bytes, device_name, pick_device, release_cache,
)
from .exceptions import RenderCancelled
from .marks import (
    _count_threshold, _dust_sites, _hair_sites, _leak_anchor, _leak_sites, _mark_rng, _mark_spread, _mark_window, _threshold_for,
)
from .masks import _grain_delta, _source_masks
from .noise.fields import _fbm, _scatter_offsets, _smooth_noise
from .noise.grain import (
    _GRAIN_CLUSTER_CELLS, _GRAIN_CLUSTER_OCTAVES, _GRAIN_CLUSTER_REF, _GRAIN_CLUSTER_ROUGHNESS, _GRAIN_CLUSTER_VAR, _GRAIN_COS, _GRAIN_FILL, _GRAIN_RINGS, _GRAIN_ROT, _GRAIN_SHARE, _GRAIN_SIN, _GRAIN_SLOTS, _GRAIN_STD_FIT, _GRAIN_TARGET_STD, _SCATTER_NAMES, _SCATTER_STENCILS, _grain_cluster, _grain_gain, _grain_lattice_noise, _grain_points,
)
from .noise.lattice import (
    _HASH_KX, _HASH_KY, _HASH_M1, _HASH_M2, _cell_noise, _lat_span, _lattice_np, _lsr, _u64, _value_noise,
)
from .primitives import (
    _BLUR_DECIMATE_TO, _BLUR_EXACT_MAX_SIGMA, _blur, _hsv_to_rgb, _hue_sat, _isophote, _luma, _rotate_hue, _smootherstep, _smoothstep, _spread, _warp,
)
from .constants import *  # noqa: F401,F403

__all__ = [
    'EDGE_REF',
    'GrainEngine',
    'RenderCancelled',
    '_AA_DIR_K',
    '_AA_DIR_MIN',
    '_AA_PASSES',
    '_AA_TAPS',
    '_AMP_SCALE',
    '_BLOB_CELLS_SCRATCH',
    '_BLUE_HUE',
    '_BLUE_RANGE',
    '_BLUE_SAT_FLOOR',
    '_DUST_DARK_LUM',
    '_DUST_ECCENT_HI',
    '_DUST_ECCENT_LO',
    '_DUST_EDGE_MAX',
    '_DUST_EDGE_MIN',
    '_DUST_EDGE_PX',
    '_DUST_HARMONICS',
    '_DUST_LITE_LUM',
    '_DUST_SIZE_SPREAD',
    '_DUST_SOFT_FADE',
    '_GLAYER_SEEDS',
    '_GNORM',
    '_GRADE_CLARITY_GAIN',
    '_GRADE_CONTRAST_GAIN',
    '_GRADE_TEMP_GAIN',
    '_GRADE_TINT_GAIN',
    '_GRADE_TONE_KNEE',
    '_GRADE_TONE_MAX',
    '_GRAIN_CLUSTER_CELLS',
    '_GRAIN_CLUSTER_OCTAVES',
    '_GRAIN_CLUSTER_REF',
    '_GRAIN_CLUSTER_ROUGHNESS',
    '_GRAIN_CLUSTER_VAR',
    '_GRAIN_COS',
    '_GRAIN_FILL',
    '_GRAIN_RINGS',
    '_GRAIN_ROT',
    '_GRAIN_SHARE',
    '_GRAIN_SIN',
    '_GRAIN_SLOTS',
    '_GRAIN_STD_FIT',
    '_GRAIN_TARGET_STD',
    '_GSRC_KEYS',
    '_HAIR_ALPHA',
    '_HAIR_CURVE',
    '_HAIR_LEN_SPREAD',
    '_HAIR_LUM',
    '_HAIR_SLOPE',
    '_HAIR_TAPER',
    '_HAIR_TIP',
    '_HAIR_WIDTH',
    '_HAIR_WIDTH_SPREAD',
    '_HAIR_WOBBLE',
    '_HASH_KX',
    '_HASH_KY',
    '_HASH_M1',
    '_HASH_M2',
    '_JITTER_MAX',
    '_LEAK_CORNER_BIAS',
    '_LEAK_GAIN',
    '_LEAK_PHI',
    '_LEAK_REACH_SAFETY',
    '_LEAK_WARP',
    '_LUMA',
    '_MARK_JITTER',
    '_MARK_MIN_PX',
    '_MID_GREY',
    '_MIN_CELL',
    '_NOISE_ICDF',
    '_R2_A1',
    '_R2_A2',
    '_RECON_CEIL',
    '_RECON_HI',
    '_RECON_LO',
    '_RECON_MIN_EVIDENCE',
    '_RECON_ROLL_GATE_FRAC',
    '_RECON_ROLL_KNEE',
    '_RENDER_BUDGET_FRACTION',
    '_SAND_DIR_K',
    '_SAND_MIN_GRAD',
    '_SAND_PASSES',
    '_SAND_TAPS',
    '_SCATTER_NAMES',
    '_SCATTER_STENCILS',
    '_SMOOTH_GAIN_K_FIT',
    '_SMOOTH_MAX',
    '_STEP_HI',
    '_STEP_LO',
    '_TEX_HI',
    '_TEX_LO',
    '_TEX_LUM_FLOOR',
    '_TILE_MAX',
    '_TILE_MIN',
    '_WARM_AXIS',
    '_WARM_GAIN',
    '_WARM_HI_BAND',
    '_WARM_LO_BAND',
    '_WARM_NULL',
    '_WARM_RAW',
    '_WORKING_BYTES_PER_PX',
    '_apply_lut',
    '_BLUR_DECIMATE_TO',
    '_BLUR_EXACT_MAX_SIGMA',
    '_blur',
    '_cell_noise',
    '_characteristic_curve',
    '_count_threshold',
    '_dust_sites',
    '_fbm',
    '_grain_cluster',
    '_grain_delta',
    '_grain_gain',
    '_grain_lattice_noise',
    '_grain_points',
    '_hair_sites',
    '_hsv_to_rgb',
    '_hue_sat',
    '_isophote',
    '_lat_span',
    '_lattice_np',
    '_leak_anchor',
    '_leak_sites',
    '_linear_to_srgb',
    '_lsr',
    '_luma',
    '_mark_rng',
    '_mark_spread',
    '_mark_window',
    '_recon_estimate',
    '_reconstruct_highlights',
    '_GRAIN_CACHE_SHARE',
    '_TILE_BUDGET_SHARE',
    '_grain_cache_bytes',
    '_render_budget_bytes',
    '_tile_budget_bytes',
    'release_cache',
    '_rotate_hue',
    '_scatter_offsets',
    '_shoulder',
    '_smooth_noise',
    '_smootherstep',
    '_smoothstep',
    '_soft_knee',
    '_source_masks',
    '_spread',
    '_srgb_to_linear',
    '_threshold_for',
    '_tone_roll',
    '_u64',
    '_value_noise',
    '_warp',
    'device_name',
    'pick_device',
]
