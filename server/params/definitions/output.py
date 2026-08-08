from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # --------------------------------------------------------------- output
    # The master blend, applied after literally everything -- and after the
    # supersample pool, which is the only place it can be bit-exact at 0.
    # Defaults to 1.0, so it is the one parameter whose neutral value is not
    # zero and the one that must stay out of NEUTRAL_ZERO.
    Param(
        "master_opacity", "Overall Opacity", "Output",
        0.0, 1.0, 0.01, 1.0, "",
        "How much of the finished result is laid over the untouched photo. "
        "1 = the full effect, 0 = the original returned bit for bit, and "
        "anything between is a straight cross-fade -- so it dials back "
        "everything at once: grain, halation, softening, marks, the lot. "
        "Reach for it when a preset is right in character but too strong, "
        "instead of walking a dozen sliders down together.\n"
        "\n"
        "Not to be confused with Global Opacity under Global Grain, which "
        "only mixes that one noise layer. This one is the whole pipeline.",
    ),
]
