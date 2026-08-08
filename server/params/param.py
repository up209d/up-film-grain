"""The ``Param`` record every control is described by, and the two literal
orderings that are load-bearing elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    group: str
    min: float
    max: float
    step: float
    default: float
    unit: str = ""
    help: str = ""
    #: True when the value is a *length* in full-resolution pixels, so it has
    #: to be rescaled when a preset authored on one image size is applied to
    #: another. See ``rescale``.
    spatial: bool = False
    #: Names for a *discrete* parameter, indexed by the value. Non-empty turns
    #: the control into a menu instead of a slider -- the value is still a
    #: number, so nothing else in the schema, the engine or a preset file has
    #: to know the difference. Only for genuine either/or choices: a stencil
    #: shape has no midpoint between "cross" and "diagonal", and a slider that
    #: pretends otherwise invites you to leave it at 2.5.
    choices: tuple[str, ...] = ()


# The Global Grain blend modes, indexed by ``global_blend``. Defined here rather
# than in the engine because this module is the single source of truth and the
# engine imports it -- two literal tuples that had to agree on an *index* would
# be a silent renderer bug the first time one of them was reordered.
#
# Order is therefore load-bearing in preset files: appending is safe, reordering
# is not. Index 0 is Add, which is what this section did before the menu
# existed, so an old preset with no `global_blend` key sanitizes to the historic
# behaviour rather than to something new.
GLOBAL_BLENDS: tuple[str, ...] = (
    "Add", "Overlay", "Soft Light", "Hard Light", "Multiply", "Screen",
)


# Groups are rendered in this order by the client.
#
# `Luminance Response` is **gone as a section** (2026-08-06, on request): its six
# parameters moved into `Grain Structure`, under the controls that build the
# field they mask. It was never a stage of its own -- it says which densities
# carry the grain the section above it makes -- and a heading of its own read as
# a second thing to set up rather than as the tail of the first. See
# `docs/panel-layout.md`, which also has why this list is not, and cannot be,
# pipeline-ordered as a whole.
GROUPS: list[str] = [
    "Colour Grading",
    "Pre Blur",
    "Pre Sharpen",
    "Grain Structure",
    "Edge Destruction",
    "Anti Aliasing",
    "Global Grain",
    "Sharpening",
    "Halation",
    "Tone Response",
    "Film Texture",
    "Output",
]
