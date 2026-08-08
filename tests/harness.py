"""Scaffolding shared by every invariant check.

Kept out of the check modules so the checks themselves read as checks.
`FAILURES` is module state on purpose -- `check()` appends to it and the runner
reads it once per module to decide the exit code.

Three things live here:

* `check()` and the two metrics several modules share.
* `suite()`, the registry `tests/checks/*.py` register themselves into. A module
  is a *unit of selection and of parallelism*, so the grouping is by what you
  would want to re-run after touching one area of the engine, not by tidiness.
* `Ctx`, the fixtures. Every one is a `cached_property`, so a module that never
  touches `big` never pays for it -- which is the whole reason
  `verify.py grading` is seconds rather than minutes. They are fixtures rather
  than per-module locals because global grain, edge destruction and sharpening
  all measure *against the same smooth-patch sigma*, and a second copy of that
  number computed slightly differently would quietly weaken all three.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, NamedTuple

import numpy as np

FAILURES: list[str] = []

# The scene the smooth-area guard and everything keyed to it are measured on.
# 1000x1400 rather than the 700x900 `img`: the patches have to be big enough to
# inset past their own borders and still leave a usable sample.
BIG = (1000, 1400)

# Inset well past the patch borders: the guard's medium-radius blur reaches
# across a hard patch edge and would inflate the reading.
INSET = 25

# Everything that would add its own texture, off -- so what is left on the plate
# is only the marks. Shared rather than copied because `film_texture` measures
# each mark's geometry against it and `film_tiling` tiles the same section: two
# copies that drifted apart would make those two modules disagree about what
# they had switched off.
TEX_OFF = {
    "intensity": 0, "global_intensity": 0, "micro_blur": 0, "acutance": 0,
    "edge_erosion": 0, "halation": 0, "edge_jitter": 0, "sharpen": 0,
}


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ registry --

class Suite(NamedTuple):
    name: str
    title: str
    fn: Callable[["Ctx"], None]


SUITES: dict[str, Suite] = {}


def suite(name: str, title: str):
    """Register a check module under `name`, which is what you type to select it."""
    def deco(fn):
        SUITES[name] = Suite(name, title, fn)
        return fn
    return deco


# ------------------------------------------------------------------ fixtures --

class Patches(NamedTuple):
    sy: int
    sx: int
    ph: int
    pw: int
    ty: int
    tx: int
    inset: int


class Ctx:
    """The engine and the test scenes, built once per process.

    Constructed lazily and cached, so running one module does not pay for the
    fixtures of the other fourteen. Under `-j` each worker process builds its
    own -- a `GrainEngine` is cheap; the renders behind `big_residual` are not,
    which is why the modules that share it are grouped to land together.
    """

    def __init__(self) -> None:
        from server import params as P
        from server.engine import GrainEngine, pick_device

        self.dev = pick_device()
        self.eng = GrainEngine(self.dev)
        self.p = P.sanitize(None)

    @cached_property
    def img(self) -> np.ndarray:
        from tests.scene import scene
        return scene(700, 900)

    @cached_property
    def big(self) -> np.ndarray:
        from tests.scene import scene
        return scene(*BIG)

    @cached_property
    def plain(self) -> np.ndarray:
        """A flat mid-grey plate -- the background film texture is measured against."""
        return np.full((900, 1400, 3), 0.5, np.float32)

    @cached_property
    def patches(self) -> Patches:
        from tests.scene import patch
        sy, sx, ph, pw = patch(*BIG, "smooth")
        ty, tx, _, _ = patch(*BIG, "textured")
        return Patches(sy, sx, ph, pw, ty, tx, INSET)

    @cached_property
    def big_residual(self) -> np.ndarray:
        """`big` rendered at the defaults, minus `big` -- what the pipeline added."""
        return self.eng.render_image(self.big, self.p, 1.0, supersample=2) - self.big

    @cached_property
    def smooth_sigma(self) -> float:
        """Grain sigma inside the smooth patch at the defaults.

        The number the smooth-area guard is judged by, and the baseline global
        grain has to beat to prove it reaches what the masks protect.
        """
        q = self.patches
        return float(self.big_residual[
            q.sy + q.inset:q.sy + q.ph - q.inset,
            q.sx + q.inset:q.sx + q.pw - q.inset,
        ].std())


# ------------------------------------------------------------------- metrics --

def area_downsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w, _ = a.shape
    h -= h % f
    w -= w % f
    return a[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))


def gridiness(lum: np.ndarray, cell: float) -> float:
    """|gradient| binned by phase within a cell -- how much a field's structure
    lines up with its own lattice, and therefore with the pixel grid.

    The metric the Global Grain quilt was diagnosed with and the one its
    replacement is held to. A lattice-addressed field swings a long way between
    phases, because its extrema sit *on* the lattice and the gradient vanishes
    there; a field that does not care where the cell boundaries fall does not
    swing. Value noise at a 20px cell scores 1.74; `_grain_points` scores 0.03.
    """
    gx = np.abs(np.diff(lum, axis=1))
    xs = (np.arange(lum.shape[1] - 1) + 0.5) / cell
    ph = np.floor((xs % 1.0) * 8).astype(int)
    # Only bins that actually contain samples. A whole-number cell puts every
    # pixel at one of exactly `cell` phases, so with a cell under 8 some bins
    # are empty and averaging them is a nan, not a zero -- which silently makes
    # the metric useless rather than making it fail.
    m = np.array([gx[:, ph == b].mean() for b in range(8) if (ph == b).any()])
    assert m.size >= 3, f"too few distinct phases at cell {cell}"
    return float((m.max() - m.min()) / m.mean())
