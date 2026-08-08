"""Calibrated constants, grouped by the stage that consumes them.

Every number in here was measured rather than picked; the comment above
each one is the measurement. Nothing in this package imports from the rest
of the engine, which is what keeps the stage modules free of import cycles.
See docs/tuning-constants.md.
"""
from __future__ import annotations

from .core import *  # noqa: F401,F403
from .edge import *  # noqa: F401,F403
from .grade import *  # noqa: F401,F403
from .halation import *  # noqa: F401,F403
from .marks import *  # noqa: F401,F403
from .tone import *  # noqa: F401,F403

__all__ = [
    'EDGE_REF',
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
    '_DUST_ECCENT',
    '_DUST_EDGE_MAX',
    '_DUST_EDGE_MIN',
    '_DUST_EDGE_PX',
    '_DUST_HARMONICS',
    '_DUST_LITE_LUM',
    '_DUST_SIZE_SPREAD',
    '_DUST_SOFT_FADE',
    '_GG_CACHE_BYTES',
    '_GLAYER_SEEDS',
    '_GNORM',
    '_GRADE_CLARITY_GAIN',
    '_GRADE_CONTRAST_GAIN',
    '_GRADE_TEMP_GAIN',
    '_GRADE_TINT_GAIN',
    '_GRADE_TONE_KNEE',
    '_GRADE_TONE_MAX',
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
    '_JITTER_MAX',
    '_LEAK_CORNER_BIAS',
    '_LEAK_GAIN',
    '_LEAK_PHI',
    '_LEAK_REACH_SAFETY',
    '_LEAK_WARP',
    '_LUMA',
    '_MARK_JITTER',
    '_MARK_MIN_PX',
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
    '_SAND_DIR_K',
    '_SAND_MIN_GRAD',
    '_SAND_PASSES',
    '_SAND_TAPS',
    '_SMOOTH_GAIN_K_FIT',
    '_SMOOTH_MAX',
    '_STEP_HI',
    '_STEP_LO',
    '_TEX_HI',
    '_TEX_LO',
    '_TEX_LUM_FLOOR',
    '_WARM_AXIS',
    '_WARM_GAIN',
    '_WARM_HI_BAND',
    '_WARM_LO_BAND',
    '_WARM_NULL',
    '_WARM_RAW',
]
