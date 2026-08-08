from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ---------------------------------------------------------------- grain
    Param(
        "intensity", "Intensity", "Grain Structure",
        0.0, 100.0, 0.5, 32.0, "%",
        "Overall grain amplitude before luminance and edge weighting.",
    ),
    Param(
        "grain_size", "Clump Size", "Grain Structure",
        # Floored at 0.4, not 0.1 (2026-08-08). **0.1 to 0.4 was a dead zone
        # that cost 1.8x for a bit-identical grain field.** Both floor to
        # `_MIN_CELL`, so `_grain_field` returns the same field (verified
        # 0.00e+00); what escapes the floor is the *secondary* fields, whose
        # cells are `grain_size * scale * 2` and `* 3` -- the ragged edge
        # envelope and the jitter displacement. Those got finer and denser
        # without the grain itself getting finer at all. Measured on a 2400px
        # proxy: 0.53s at 0.1, 0.52s at 0.4, 0.47s at 0.8.
        #
        # Ten of the twelve shipped presets sat at 0.1 and were re-authored to
        # 0.4 in the same change. The look moves, slightly and only at edges:
        # mean 0.06 levels on `Stock` and 0.09 on `SuperPortra`, p99 1.3 and 1.9
        # levels, with 1.1-1.3% of pixels moving by more than one 8-bit level.
        0.4, 10.0, 0.05, 1.6, "px",
        "Silver-halide clump diameter, measured at full resolution -- the "
        "finest structure in the grain. Octaves stack coarser scales on top of "
        "it. Held in full-res units, so it means the same thing at any zoom. "
        "The bottom of the range is where the lattice hits its own floor: below "
        "about 0.4 the clump cannot get any finer, so asking for less only "
        "sharpens the edge envelope and the jitter around it.",
        spatial=True,
    ),
    Param(
        "shadow_size", "Shadow Clumping", "Grain Structure",
        0.0, 1.0, 0.01, 0.45, "",
        "Enlarges clumps in shadows, where halide crystals grow larger and "
        "less densely packed.",
    ),
    Param(
        "clump", "Clump Hardness", "Grain Structure",
        0.0, 1.0, 0.01, 0.5, "",
        "Sharpens the grain distribution toward discrete clumps instead of "
        "smooth gaussian noise.",
    ),
    Param(
        "octaves", "Octaves", "Grain Structure",
        1.0, 10.0, 1.0, 3.0, "",
        "Number of noise scales stacked to build the emulsion structure. Clump "
        "Size is the finest scale and each octave adds one twice as coarse, so "
        "this builds clumps-of-clumps and then broad mottling over the top. "
        "How far up the stack still matters is set by Roughness: at 0.5 the "
        "effect has largely landed by 3-4 octaves, at 1.0 all ten count.",
    ),
    Param(
        "roughness", "Roughness", "Grain Structure",
        0.0, 1.0, 0.01, 0.5, "",
        "Amplitude falloff between successive octaves, and so how much of the "
        "coarse structure survives. 0 = the base clump alone and Octaves does "
        "nothing; 1 = every octave weighs the same and the grain goes visibly "
        "clumpy and mottled. Total grain strength stays put either way -- this "
        "moves structure around, Intensity sets how much of it there is.",
    ),
    Param(
        "chroma_grain", "Chroma Grain", "Grain Structure",
        0.0, 1.0, 0.01, 0.35, "",
        "0 = monochrome grain shared across channels. 1 = independent dye "
        "cloud noise per layer.",
    ),
]
