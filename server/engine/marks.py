from __future__ import annotations

import math

import numpy as np
import torch

from .constants.marks import (
    _DUST_ROUGH_SPREAD, _DUST_SIZE_SPREAD, _HAIR_ALPHA, _HAIR_CURVE, _HAIR_LEN_SPREAD, _HAIR_SLOPE, _HAIR_WIDTH_SPREAD, _HAIR_WOBBLE, _LEAK_CORNER_BIAS, _LEAK_PHI, _MARK_JITTER, _NOISE_ICDF, _R2_A1, _R2_A2,
)

def _threshold_for(fraction: float) -> float:
    """Field threshold that leaves roughly ``fraction`` of pixels above it."""
    f = min(max(fraction, 5e-7), 0.30)
    pts = _NOISE_ICDF
    for (f0, q0), (f1, q1) in zip(pts, pts[1:]):
        if f1 <= f <= f0:
            # log-linear between the bracketing samples
            w = (math.log(f) - math.log(f0)) / (math.log(f1) - math.log(f0))
            return q0 + w * (q1 - q0)
    return pts[0][1] if f >= pts[0][0] else pts[-1][1]


def _count_threshold(
    count: float, cell_area: float, frame_area: float, blob_cells: float,
) -> float:
    """Threshold that puts roughly ``count`` marks on a frame.

    A mark occupies about one lattice cell, so the frame holds
    ``frame_area / cell_area`` of them and the fraction wanted is the ratio.
    Approximate by construction -- blobs merge and clip at the frame edge --
    but it makes the control a *count* rather than an opaque 0-1, and a
    requested 20 lands within a few of 20 rather than an order of magnitude
    away.
    """
    cells = max(frame_area / max(cell_area, 1e-6), 1.0)
    return _threshold_for(max(count, 0.0) / (cells * blob_cells))


def _leak_sites(count: float, seed: int, var: float) -> list[dict]:
    """Per-leak parameters, in units that do not depend on the frame's size.

    This *is* a list of objects, which the rest of the film-texture section
    refuses to use -- and it is still tile-independent, because the list is a
    function of the count, the seed and nothing else. Every tile builds the
    identical list; so does the proxy, so does the export. What breaks tile
    independence is deriving a list from the *region being rendered* (N specks
    per tile, or positions drawn against the tile's own area), and nothing
    here reads either.

    Objects rather than thresholded noise because a leak is not a mark, it is a
    beam: it has a source, a direction and a length, and a field that only
    knows "how far am I from the nearest border" can express none of those.

    ``var`` is `leak_variation`: every draw except the reach is a blend from
    the middle of its range toward the drawn value, so 0 makes every leak
    identical in everything but where it sits and how far it comes in.
    """
    n = int(min(max(round(count), 0), 64))

    def mix(u: float, lo: float, hi: float) -> float:
        return 0.5 * (lo + hi) + var * (u - 0.5) * (hi - lo)

    sites = []
    for k in range(n):
        # Seeded per leak, not once per frame, for the same reason the
        # positions come off a low-discrepancy sequence: leak 3 must not
        # change when leak 9 is added.
        rng = np.random.default_rng(
            np.uint64((int(seed) & 0xFFFF) * 1000003 + k * 7919 + 17)
        )
        u = rng.random(10)
        sites.append({
            # Where on the perimeter, 0..1. The jitter is small on purpose --
            # the golden step already spreads them, and a large jitter just
            # lets two leaks land on top of each other.
            "pos": (0.37 + _LEAK_PHI * k + 0.10 * (u[0] - 0.5)) % 1.0,
            # Reach is the one draw `var` does not touch: the two size sliders
            # state its spread outright, and the help text promises variation
            # changes everything *except* size.
            "reach_t": u[1],
            # Half-length along the border, as a fraction of that border.
            # A fraction of the *border* rather than a multiple of the reach:
            # a failed seal runs along a seam, so a leak is long sideways and
            # shallow inward, and sizing it off its own depth makes blobs.
            "width": mix(u[2], 0.03, 0.30),
            # Lateral drift per unit depth. This is what makes a leak a streak
            # leaning across the frame instead of a symmetric wedge, and it is
            # kept as a shear rather than as a rotation so that "reach" stays
            # exactly the perpendicular depth the slider claims.
            "shear": mix(u[3], -1.7, 1.7),
            # How much the beam fans out as it travels in.
            "flare": mix(u[4], -0.55, 0.75),
            # Asymmetry of the two long edges. A leak is light spilling past an
            # obstruction, so one side is the obstruction's shadow and is much
            # harder than the other; a shape soft on both sides reads as haze.
            "hard": mix(u[5], 0.25, 1.0),
            "hard_side": 1.0 if u[6] >= 0.5 else -1.0,
            "strength": mix(u[7], 0.45, 1.35),
            # Halo: pushes this leak's half-strength distance around.
            "halo": u[8],
            # Hue jitter, added to `leak_hue`.
            "hue": mix(u[9], -0.20, 0.20),
        })
    return sites


def _mark_rng(seed: int, salt: int, k: int) -> np.random.Generator:
    """Per-mark generator, seeded on the mark's own index.

    Seeded per mark rather than once per frame, for the reason `_leak_sites`
    documents: mark 3 must not change when mark 9 is added. That is what makes
    a count slider *add* marks instead of rerolling the frame every time it
    moves, and it is why raising Dust Count from 20 to 21 leaves twenty specks
    exactly where they were.
    """
    return np.random.default_rng(
        np.uint64((int(seed) & 0xFFFF) * 1000003 + salt + k * 7919 + 17)
    )


def _mark_spread(
    salt: int, seed: int, u0: float, u1: float, k: int,
) -> tuple[float, float]:
    """Where mark ``k`` sits, as a fraction of the frame.

    A low-discrepancy step plus a small jitter, which is `_leak_sites`' trick in
    two dimensions and it is here for the same reason. **Independent uniform
    draws clump, and at small counts they clump visibly**: measured on the hair
    generator, four of the first five marks landed in the top fifth of the
    frame. That is not a bug in the hash -- over 400 marks the draws are uniform
    to 1% and uncorrelated to 0.02 -- it is just what five uniform points look
    like, and "I asked for five hairs and they are all in one corner" is a
    complaint whether or not the statistics are innocent.

    The R2 sequence steps by the reciprocal powers of the plastic number, which
    fills the unit square about as evenly as a sequence can without knowing how
    long it will be. That last part is what makes it usable here: **any prefix
    is well spread**, so mark 6 can be added without moving marks 1 to 5, and
    the count slider keeps the add-don't-reroll behaviour `_mark_rng` exists
    for.

    The jitter is fixed in frame units rather than scaled to the count, and that
    is deliberate in both directions. At high counts the R2 spacing is smaller
    than the jitter, so the placement is locally random and dust clumps the way
    dust does; at low counts the spacing is much larger than the jitter, so the
    sequence dominates and the marks spread out. Scaling the jitter to the count
    would move every existing mark whenever the count changed.
    """
    off = _mark_rng(seed, salt + 977, 0).random(2)
    y = (off[0] + (k + 1) * _R2_A2 + _MARK_JITTER * (u0 - 0.5)) % 1.0
    x = (off[1] + (k + 1) * _R2_A1 + _MARK_JITTER * (u1 - 0.5)) % 1.0
    return float(y), float(x)


def _dust_sites(count: int, seed: int, balance: float) -> list[dict]:
    """One record per speck, in units that do not depend on the frame's size.

    **Dust is a list of objects now** (rewritten 2026-08-06). It was a threshold
    on a value-noise field, which is the construction `docs/film-texture.md`
    still insists on for scratches, and two things it could not do were asked
    for outright:

    * **A count that is a count.** A threshold selects *area*, and the number of
      countable blobs that area breaks into was a fitted constant (14.0, good to
      about a factor of 1.5). Ask for 20 specks and you got somewhere between 13
      and 30. Here 20 is twenty.
    * **A shape.** The outline of a thresholded noise field is whatever the
      field happened to do -- lumpy, frequently merged with its neighbour, and
      occasionally a long tear that reads as a scratch. A speck is a small round
      thing; you cannot get one out of a level set of noise except by accident,
      which is exactly what the user reported seeing.

    **This does not break tile independence, and the reason is `_leak_sites`'.**
    The list is a function of the count, the seed and the *frame* -- never of
    the region being rendered. Every tile builds the identical list, positions
    resolve against `full_hw` rather than against the tile, and a speck
    straddling a tile boundary is drawn by both tiles from the same absolute
    geometry. What breaks the invariant is N specks *per tile* or positions
    drawn against the tile's own area, and neither happens here.

    ``balance`` is `dust_balance`: -1 all dark, +1 all bright, 0 an even split.
    The split is a prefix of the list rather than a per-speck coin flip, which
    is what makes it exact *and* makes moving the slider convert specks in place
    instead of reshuffling them -- position is drawn per index and never touched
    by the balance.
    """
    n = int(min(max(count, 0), 4000))
    n_light = int(round(n * (min(max(balance, -1.0), 1.0) + 1.0) * 0.5))

    sites = []
    for k in range(n):
        u = _mark_rng(seed, 5501, k).random(12)
        s_lo, s_hi = _DUST_SIZE_SPREAD
        # Position as a fraction of the frame, so it lands in the same place at
        # any working scale and in any tiling.
        py, px = _mark_spread(5501, seed, u[0], u[1], k)
        sites.append({
            "y": py,
            "x": px,
            "size": s_lo + (s_hi - s_lo) * u[2] * u[2],
            # Squared draw above: small debris outnumbers large debris, and a
            # flat draw puts as many 1.5x specks on the frame as 0.6x ones,
            # which reads as gravel rather than dust.
            # The raw 0..1 draw, **not** scaled by a fixed ceiling here any
            # more (2026-08-08). How elongated a speck may get is now a
            # function of `dust_irregular`, which the draw site cannot see --
            # so the site records the draw and `_film_texture` scales it. It
            # used to bake in a fixed 0..0.35, which meant even at
            # `dust_irregular` 0 a third of the population came out as clean,
            # obvious ellipses. Reported as exactly that.
            "eccent": u[3],
            "angle": u[4] * 2.0 * math.pi,
            "phase": tuple(v * 2.0 * math.pi for v in u[5:8]),
            # How lumpy this particular speck is, before `dust_irregular`
            # scales the whole population. See `_DUST_ROUGH_SPREAD`.
            "rough": _DUST_ROUGH_SPREAD[0] + (
                _DUST_ROUGH_SPREAD[1] - _DUST_ROUGH_SPREAD[0]
            ) * u[11],
            "soft": u[8],
            "opacity": u[9],
            "lum": u[10],
            # Which population. See the docstring: a prefix, not a coin.
            "light": k < n_light,
        })
    return sites


def _hair_sites(count: int, seed: int) -> list[dict]:
    """One record per hair. Same construction as `_dust_sites`, same reasons.

    The reported bug was "I can see more than one hair when I set the count to
    1", and it was not a tuning error -- it was structural. A hair used to be
    the level set ``|n - 0.5| < eps`` of a smooth field, gated by a second field
    thresholded to select roughly one blob's worth of area per hair asked for.
    A level set is not one curve: inside any given gate blob the field crosses
    0.5 along however many separate arcs it happens to, so one unit of "hair"
    drew one filament, or three, or none. The gate constant (`_BLOB_CELLS_HAIR`
    = 0.5) was a fitted apology for exactly that.

    Drawn from a list there is nothing to fit: one record is one filament, and
    the count is the length of the list.

    A hair still has to *wander* -- the level set's one real virtue was that it
    curved the way a hair lies rather than along a curve somebody chose. So the
    filament carries a quadratic sag plus two sinusoidal wobbles at incommensurate
    frequencies, all scaled by its own length, which gives a curve with no
    repeating period and no preferred direction. See `_HAIR_CURVE`.

    **The wobbles are bounded by their slope, not by their amplitude**, and that
    is the one non-obvious thing in here. The renderer measures how far a pixel
    is from the filament as the vertical gap divided by ``sqrt(1 + slope^2)``,
    which is the perpendicular distance only while the curve is locally close to
    a straight line. A large amplitude at a high frequency is not: the curve
    doubles back within a pixel or two, a point genuinely sitting on it is
    scored against the wrong part of it, and the filament comes out with gaps
    where it bends hardest. Measured, a fifth of the hairs broke into two or
    three pieces that way. So each wobble's amplitude is capped at
    ``_HAIR_SLOPE / (2 pi f)``, which holds its steepest slope to a constant
    however fast it ripples -- and it is the physical answer too, since a fibre
    does not zigzag tightly *and* widely at the same time.
    """
    n = int(min(max(count, 0), 400))

    sites = []
    for k in range(n):
        u = _mark_rng(seed, 6607, k).random(15)
        l_lo, l_hi = _HAIR_LEN_SPREAD
        w_lo, w_hi = _HAIR_WIDTH_SPREAD
        a_lo, a_hi = _HAIR_ALPHA
        py, px = _mark_spread(6607, seed, u[0], u[1], k)
        # Incommensurate frequencies: a whole number of cycles over the filament
        # would make both ends leave at the same angle, which reads as a drawn
        # arc. These do not divide each other either, so the pair never repeats
        # over one hair's length.
        freq = (0.6 + 0.9 * u[7], 1.7 + 1.6 * u[8])
        sites.append({
            "y": py,
            "x": px,
            "len": l_lo + (l_hi - l_lo) * u[2],
            "angle": u[3] * 2.0 * math.pi,
            # Signed, so hairs curl both ways.
            "curve": _HAIR_CURVE * (2.0 * u[4] - 1.0),
            "wob": tuple(
                min(a, cap / (2.0 * math.pi * f)) * (2.0 * v - 1.0)
                for a, cap, f, v in zip(_HAIR_WOBBLE, _HAIR_SLOPE, freq, u[5:7])
            ),
            "freq": freq,
            "phase": (u[9] * 2.0 * math.pi, u[10] * 2.0 * math.pi),
            "width": w_lo + (w_hi - w_lo) * u[11],
            "alpha": a_lo + (a_hi - a_lo) * u[12],
            "lum": u[13],
            "soft": u[14],
        })
    return sites


def _mark_window(
    cy: float, cx: float, reach: float, h: int, w: int, y0: float, x0: float,
    device: torch.device,
) -> tuple[slice, slice, torch.Tensor, torch.Tensor] | None:
    """Tile-local slice and centre-relative coordinate ramps for one mark.

    Returns ``None`` when the mark does not touch this tile at all, which is the
    usual answer and is what keeps a list of four hundred specks cheap: the cost
    is the marks' own total area, not the count times the frame.

    **The arithmetic here is what makes a drawn mark tile-independent**, so it is
    worth being explicit about. A pixel's offset from the mark is
    ``(i + y0) - cy``: ``i`` is its index within the tile and ``y0`` the tile's
    absolute origin, both whole numbers and both exact in float32 below 2^24, so
    their sum is the pixel's absolute coordinate *exactly* -- the same value
    whichever tile asks, whatever offset that tile happens to start at. ``cy``
    comes from the frame's size and the site record, neither of which knows a
    tile exists. So two tilings agree bit for bit rather than approximately, and
    a speck split down the middle by a tile boundary is drawn as one speck.
    """
    ys0 = max(0, int(math.floor(cy - reach - y0)))
    ys1 = min(h, int(math.ceil(cy + reach - y0)) + 1)
    xs0 = max(0, int(math.floor(cx - reach - x0)))
    xs1 = min(w, int(math.ceil(cx + reach - x0)) + 1)
    if ys1 <= ys0 or xs1 <= xs0:
        return None
    # Absolute coordinate first, *then* the centre subtracted -- not
    # `arange + (y0 - cy)`. Folding the origin and the centre together first
    # gives two tilings two different float roundings of the same offset, which
    # is a sub-pixel disagreement across a tile seam. This way the absolute
    # coordinate is an exact integer in both and only one rounding happens.
    dy = (
        (torch.arange(ys0, ys1, device=device, dtype=torch.float32) + float(y0))
        - cy
    ).view(1, 1, -1, 1)
    dx = (
        (torch.arange(xs0, xs1, device=device, dtype=torch.float32) + float(x0))
        - cx
    ).view(1, 1, 1, -1)
    return slice(ys0, ys1), slice(xs0, xs1), dy, dx


def _leak_anchor(pos: float, fh: float, fw: float) -> tuple[int, float]:
    """Map a perimeter position to (border, along-border coordinate in px).

    Borders are 0 top, 1 bottom, 2 left, 3 right; the coordinate is x on the
    horizontal borders and y on the vertical ones.
    """
    a = (pos % 1.0) * 2.0 * (fh + fw)
    for border, length in ((0, fw), (3, fh), (1, fw), (2, fh)):
        if a < length or length <= 0.0:
            t = (a / length) if length > 0.0 else 0.0
            # Pull toward both ends of the segment, i.e. toward the corners.
            t = min(max(t - _LEAK_CORNER_BIAS * math.sin(2.0 * math.pi * t),
                        0.0), 1.0)
            return border, t * length
        a -= length
    return 0, 0.0
