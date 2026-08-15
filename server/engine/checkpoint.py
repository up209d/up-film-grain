"""The pipeline checkpoint cache: the finished image at a section boundary.

Editing a slider near the *end* of the pipeline re-runs the whole thing to
produce a frame that differs only in its last few stages. This holds the
intermediate at a boundary so those edits restore it instead.

Measured on `SuperPortra` at a 2400px proxy, with the cost distributed the way
it is after the texture-cache fix: 89% of the GPU render and 68% of the CPU one
sits *above* Global Grain, and the sliders below that boundary -- Global Grain's
eleven, Sharpening's two, Film Texture's twenty-two -- are a large share of what
anyone actually tunes.

**Which boundaries are usable is a property of the pipeline, not a choice.** A
checkpoint has to store every value that crosses it, and the middle of `render()`
carries five or six planes at once: `lum_ref`, `hp`, `m`, `edge` and `wgt` are
derived early and consumed late. An AST liveness pass over the 73 top-level
statements found seven boundaries where the image is the *only* thing live, in
two clusters -- three near the top and four near the bottom -- and within each
cluster only the deepest is worth keeping, since the shallower ones protect
strictly less for the same bytes. That left the two deep ones here.

`Colour Grading` joined them on 2026-08-16 and is the exception that proves the
rule rather than a counter-example: `Normalize` arrived as a new section above
everything, so the boundary under it has *nothing* above it at all -- one live
plane by construction, and a signature of a single key. It protects the least
work of the three (one per-pixel curve; the metering it applies is measured once
per upload) and hits the most often, which is the opposite trade from the deep
one and is why both are worth keeping.

**A stale hit is the worst failure this codebase has.** The texture cache's
version renders a plausible but wrong *texture*; this one renders a plausible but
wrong *photograph*. So the key is derived rather than listed: it hashes the whole
sanitised parameter dict minus the keys belonging to sections below the boundary,
with those sections read from `GROUPS`. A hand-maintained list of upstream keys
would silently stop covering the next parameter anyone adds, and CLAUDE.md
promises that adding a control is one `Param` and one `p["key"]` read.
"""

from __future__ import annotations

import collections

import torch

from .. import params as P
from ..params.param import GROUPS


#: Sections that run entirely *below* each checkpoint, by name.
#:
#: **Derived from execution order, not from `GROUPS`.** The reason used to be
#: that the panel and the pipeline disagreed about where Halation ran; they have
#: agreed since the 2026-08-08 reorder, and the distinction is *still*
#: load-bearing for a second reason that the first one was hiding. A section's
#: parameters can be read above the boundary by a stage that is not that
#: section: `render()` evaluates the characteristic curve at section 3 as a mask
#: input, to get the density luma the grain band and Shadow Clumping key on, and
#: only applies it for real at section 7. So `Tone Response` is *below* the
#: boundary and its parameters are consumed *above* it, and taking a plain
#: `GROUPS` suffix drops them from the key -- measured, a `brightness` edit then
#: came back 2.3e-01 wrong against a cold render, which is a plausible and wrong
#: photograph.
#:
#: `verify.py` catches exactly that by re-rendering one parameter from every
#: section against a warm cache. Anything added here has to be a section whose
#: keys nothing above the boundary reads, which is a stronger condition than
#: "runs below the boundary" and cannot be read off the panel.
def _from(section: str) -> frozenset[str]:
    """Every section from ``section`` down, by name.

    Name-based rather than a `GROUPS[n:]` literal, and that is the fix for a
    trap rather than a tidy-up: the slices used to be written as indices, so
    inserting a section anywhere above one shifted it silently. `Normalize`
    joining the top in 2026-08-16 would have turned `GROUPS[3:]` from "Grain
    Structure down" into "Pre Sharpen down", putting Pre Sharpen below a
    checkpoint saved *after* it had run -- exactly the stale hit the comment on
    that boundary already records, reintroduced by an edit nowhere near it.
    """
    return frozenset(GROUPS[GROUPS.index(section):])


_BELOW: dict[str, frozenset[str]] = {
    # Saved after Normalize, so what is below it is everything from Colour
    # Grading down and the only thing above it is Normalize itself.
    #
    # The shallowest boundary, and the cheapest thing it could protect -- one
    # per-pixel curve, since the metering it applies is measured once per upload
    # and cached on the `Upload` rather than per render. It exists because
    # Normalize is the one stage whose *input* is the untouched source: holding
    # its output means every edit anywhere below replays from a normalised frame
    # without the stage running again, and the section above it is empty, so its
    # signature is one key and it hits on literally every other edit.
    "Colour Grading": _from("Colour Grading"),
    # Saved after Pre Sharpen, so what is below it is everything from Grain
    # Structure down. Protects little on the GPU (~17%) and ~24% on the CPU,
    # where highlight reconstruction alone is 2.55s of a 10.4s `SuperPortra`
    # render -- but it hits on nearly every edit, so it catches what the deeper
    # one lets through.
    #
    # **Named for the section it sits above, and getting that off by one is a
    # stale hit.** It was first written as `GROUPS[1:]`, which put `pre_blur`
    # itself below a checkpoint saved *after* pre-blur had run -- so dragging
    # Pre Blur returned the previous frame. `verify.py` caught it at 9.77e-01,
    # which is most of full scale.
    "Grain Structure": _from("Grain Structure"),
    # The valuable one: 89% of the GPU render and 68% of the CPU one sits above
    # it, and 35 sliders sit below.
    #
    # **Named `Halation` as of 2026-08-09, when Global Grain and Sharpening
    # moved below Film Texture** and Halation became the first section under the
    # boundary. The boundary itself did not move a statement.
    #
    # `Halation` came with the rename for free -- it always ran below the
    # checkpoint and was never listed, so its edits were re-rendering everything
    # above for nothing. `Tone Response` is the one section between the boundary
    # and the bottom of the pipeline that is deliberately **not** here, for the
    # double-evaluation reason above: it is applied below, and read above.
    "Halation": _from("Halation") - {"Tone Response"},
}

#: Boundaries in the order they execute, shallowest first.
CHECKPOINTS: tuple[str, ...] = ("Colour Grading", "Grain Structure", "Halation")


def _downstream(boundary: str) -> frozenset[str]:
    """Parameter keys the checkpoint at ``boundary`` is allowed to ignore.

    Recomputed from `PARAM_BY_KEY` every call rather than frozen into a literal:
    this is the set the key *skips*, so a key missing from the group map has to
    fall on the safe side by default. Adding a parameter to a section listed in
    `_BELOW` puts it below the boundary automatically; adding one anywhere else
    makes it part of the signature, which is the conservative answer.
    """
    below = _BELOW[boundary]
    return frozenset(
        k for k, prm in P.PARAM_BY_KEY.items() if prm.group in below
    )


def upstream_signature(p: dict, boundary: str) -> tuple:
    """Everything above ``boundary`` that the render depends on, as a key part.

    Sorted so the tuple is stable across dict orderings, and it carries the LUT
    by identity because a LUT is a *resource* rather than a number -- it rides
    beside the values in `p["lut"]` and is emphatically upstream, being the last
    thing Colour Grading does.
    """
    skip = _downstream(boundary)
    vals = tuple(
        (k, p[k]) for k in sorted(p)
        if k not in skip and isinstance(p[k], (int, float))
    )
    lut = p.get("lut")
    # `id()` would be wrong here -- two equal LUTs loaded twice are the same
    # grade, and a reused address is a different one. `lut.id` is the folder
    # name or the upload token, which is what the request named.
    return vals + (("lut", getattr(lut, "id", None)),)


class CheckpointCache:
    """A byte-capped LRU of section-boundary frames, keyed by everything above.

    Not thread-safe, and does not need to be: `runtime.RENDER_LOCK` serialises
    every render, the same assumption the texture cache already rests on.
    """

    def __init__(self, cap_bytes: int) -> None:
        self._d: collections.OrderedDict = collections.OrderedDict()
        self._bytes = 0
        self.cap = cap_bytes
        self.hits = 0
        self.misses = 0
        self.evicted = 0
        # The last two upstream signatures seen at each boundary. An older one
        # is unreachable for the same reason an old texture generation is: a
        # render that wanted it would have to put those parameters back, and
        # would then be current.
        #
        # Two rather than one because a frame is 184MB at a 2400px proxy and
        # this is the largest thing the app caches -- letting the byte cap alone
        # decide means holding 1.6GB of frames nothing can ask for, measured,
        # which is the waste this class would otherwise be introducing while the
        # texture cache next door was having it removed.
        self._gens: dict[str, list] = {}

    def get(self, key) -> torch.Tensor | None:
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key, t: torch.Tensor) -> None:
        # Key layout is (id, boundary, scale, y0, x0, h, w, device, signature),
        # so the boundary is [1] and the signature is [-1]. Tile coordinates are
        # *not* part of the generation: one render fills several tiles at the
        # same signature and they all have to survive each other.
        boundary, sig = key[1], key[-1]
        seen = self._gens.setdefault(boundary, [])
        if sig not in seen:
            seen.insert(0, sig)
            del seen[2:]
            for k in [k for k in self._d
                      if k[1] == boundary and k[-1] not in seen]:
                old = self._d.pop(k)
                self._bytes -= old.element_size() * old.nelement()
                self.evicted += 1

        n = t.element_size() * t.nelement()
        # An entry bigger than the whole budget is not stored rather than
        # immediately evicting itself -- otherwise a large single-tile render
        # would empty the cache on every pass and pay the bookkeeping for it.
        if n > self.cap:
            return
        # Subtract first if this key is already present. Re-putting is legal --
        # a caller that stored on a miss and stores again on a hit is doing
        # nothing wrong -- but counting the bytes twice is not, and the symptom
        # is a byte total that climbs while the entry count stays flat.
        prev = self._d.pop(key, None)
        if prev is not None:
            self._bytes -= prev.element_size() * prev.nelement()
        self._d[key] = t
        self._bytes += n
        while self._bytes > self.cap and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self._bytes -= old.element_size() * old.nelement()

    def clear(self) -> None:
        self._d.clear()
        self._bytes = 0
        self._gens.clear()

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._d)
