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
        0.1, 10.0, 0.05, 1.6, "px",
        "Silver-halide clump diameter, measured at full resolution -- the "
        "finest structure in the grain. Octaves stack coarser scales on top of "
        "it. Held in full-res units, so it means the same thing at any zoom.",
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
