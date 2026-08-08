from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _u64(v: int) -> int:
    """Reinterpret a 64-bit constant as the signed int64 with the same bits."""
    return int(np.uint64(v).astype(np.int64))


# splitmix64's mixing constants, as signed int64. Multiplication and addition are
# bit-identical for signed and unsigned two's complement, so the same bit
# patterns give the same hash -- only the *shifts* need care. See `_lsr`.
_HASH_KY = _u64(0xC2B2AE3D27D4EB4F)
_HASH_KX = _u64(0x9E3779B97F4A7C15)
_HASH_M1 = _u64(0xBF58476D1CE4E5B9)
_HASH_M2 = _u64(0x94D049BB133111EB)


def _lsr(n: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (unsigned) right shift on an int64 tensor.

    torch has no uint64, and int64 `>>` is *arithmetic* -- it smears the sign bit
    down instead of shifting zeros in, so it is simply the wrong operator for a
    hash. Masking off the bits above the shift width restores the unsigned
    result: after shifting right by ``k`` only ``64 - k`` bits can be set.

    **The mask has to come from ``k``.** A first pass at this used one wide
    constant for every shift, which produced something that still *looked* like
    noise and was a different field -- exactly the kind of bug that survives an
    eyeball test, which is why ``verify.py`` asserts bit-equality against the
    numpy reference rather than rendering a frame and squinting at it.
    """
    return (n >> k) & ((1 << (64 - k)) - 1)


def _lattice_np(iy0: int, ix0: int, hl: int, wl: int, seed: int, nfields: int) -> np.ndarray:
    """Deterministic hash noise on an integer lattice window.

    Integer hashing, so it runs on the CPU rather than the GPU: 64-bit integer
    ops are poorly supported on MPS, and the exactness is the point -- two tiles
    asking about the same lattice point must agree bit for bit or exports seam.

    **It is not a small amount of work, whatever the shape of the lattice
    suggests.** An older version of this note claimed the lattice is "far smaller
    than the pixel grid, so this is cheap", and that is false at every setting
    the app actually ships. ``cell`` is floored at ``_MIN_CELL`` = 0.8 *working*
    pixels and every preset in ``presets/`` sets ``grain_size`` to 0.1-0.3, so
    the base lattice is *denser* than the pixel grid. Measured lattice points
    hashed per output pixel: defaults 2.5x, Dreamy 5.7x, Dramatic 38x, Subtle
    48x, ExtraGrain 54x, **Stock 58x** -- 291M hashes for one 2400px proxy
    preview. It was 23% of that render's wall time.

    So it runs in **torch on the CPU rather than numpy**, which is worth 2.5x for
    free: numpy's uint64 elementwise ops are single-threaded, torch's int64 ones
    use ``at::parallel_for`` across every core, and the arithmetic is otherwise
    identical. Measured 113ms -> 46ms on a 44M-point lattice, bit-exact. It stays
    on the CPU (rather than moving to the GPU in 32-bit, which would be faster
    still) because that would change every value and reroll every preset's grain.

    Returns numpy for its callers' ``torch.from_numpy(...)``; the conversion
    shares memory and costs nothing.
    """
    # Lattice indices go negative near the origin; the int64 bit pattern *is* the
    # unsigned one, so this needs no reinterpretation -- it is the same wrap the
    # old `.view(np.uint64)` produced.
    yy = torch.arange(iy0, iy0 + hl, dtype=torch.int64).unsqueeze(1)
    xx = torch.arange(ix0, ix0 + wl, dtype=torch.int64).unsqueeze(0)
    out = torch.empty((nfields, hl, wl), dtype=torch.float32)
    for f in range(nfields):
        # Fold the seed in Python ints so the wrap is explicit.
        s = _u64(((seed + f * 7919) * 0x165667B19E3779F9) % (1 << 64))
        n = xx * _HASH_KX + yy * _HASH_KY + s
        n = n ^ _lsr(n, 29)
        n = n * _HASH_M1
        n = n ^ _lsr(n, 32)
        n = n * _HASH_M2
        n = n ^ _lsr(n, 31)
        out[f] = _lsr(n, 40).to(torch.float32) / float(1 << 24)
    return out.numpy()


def _lat_span(
    n: int, origin: float, cell: float, pad_lo: int, pad_hi: int,
) -> tuple[int, int]:
    """First lattice index and count covering ``n`` working pixels from ``origin``.

    Exists to keep this arithmetic off the GPU. The three noise builders used to
    build their coordinate ramp on the device and then read scalars back off it
    (``int(math.floor(float(ys[0])))``), and every such read drains the MPS
    command queue -- counted 32 per ``render_supersampled`` at defaults and 108
    at ``Stock``, each a full pipeline stall for a number Python already had.

    **Computed in float32, and that is not pedantry.** The device ramp is
    ``torch.arange(n, dtype=float32) + origin``, and a Python float scalar takes
    the tensor's dtype, so the whole expression is float32. Doing it in float64
    here would occasionally land on the other side of an integer boundary and
    select a *different* lattice window -- which is a different noise field, not
    a rounding difference. ``verify.py`` pins this against the device path.
    """
    f = np.float32
    lo = f(f(f(0.0) + f(origin)) / f(cell))
    hi = f(f(f(n - 1) + f(origin)) / f(cell))
    i0 = int(math.floor(float(lo))) - pad_lo
    return i0, int(math.floor(float(hi))) + pad_hi - i0 + 1


def _value_noise(
    h: int, w: int, y0: float, x0: float, cell: float,
    seed: int, nfields: int, device: torch.device, cell_y: float | None = None,
) -> torch.Tensor:
    """Quintic-interpolated value noise addressed by global coordinates.

    Trick: ``grid_sample`` in bilinear mode interpolates by the fractional part
    of the sampling coordinate. Feeding it ``floor(t) + quintic(frac(t))``
    yields exact quintic value noise while keeping the sampler bilinear, which
    is the mode with the broadest backend support.

    ``cell_y`` defaults to ``cell``, giving the isotropic field the grain uses.
    Setting it far larger stretches the field along y, which is how the film
    texture draws scratches: a scratch is just noise whose cells are hundreds
    of pixels tall and a couple wide.
    """
    cy = cell if cell_y is None else cell_y
    iy0, hl = _lat_span(h, y0, cy, 1, 2)
    ix0, wl = _lat_span(w, x0, cell, 1, 2)

    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cy
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell

    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, nfields))
    lat = lat.to(device).unsqueeze(0)

    def remap(t: torch.Tensor) -> torch.Tensor:
        fl = torch.floor(t)
        f = t - fl
        return fl + f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

    vi = remap(ys) - iy0
    ui = remap(xs) - ix0
    gy = vi / max(hl - 1, 1) * 2.0 - 1.0
    gx = ui / max(wl - 1, 1) * 2.0 - 1.0
    Y, X = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)

    return F.grid_sample(
        lat, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )


def _cell_noise(
    h: int, w: int, y0: float, x0: float, cell: float, seed: int, nfields: int,
    device: torch.device,
) -> torch.Tensor:
    """One constant hash value per lattice cell -- blocky, and *uniform*.

    The counterpart to ``_value_noise``, and it exists for one reason:
    interpolation destroys the distribution. Quintic value noise is heavily
    centre-weighted -- p10-p90 spans only 0.41-0.71 -- which is fine for a
    field you are going to threshold or spread, and useless for one you are
    going to *quantise*. ``floor(n * 4)`` over an interpolated field returns 1
    or 2 almost every time, so a four-way stencil would fire two of its four
    directions and the scatter would come out with a diagonal bias nobody
    asked for. Reading the lattice without interpolating gives back the hash's
    own uniform distribution, so every direction is equally likely.

    Blockiness is the other half of the point. Every pixel inside one cell
    reads the same value, so a whole cell of image is displaced as a unit --
    that is what ``scatter_cell`` means, and it is why detail can survive the
    trip instead of dissolving. Addressed by global coordinates like every
    other field here, so two tiles asking about the same pixel agree.
    """
    iy0, hl = _lat_span(h, y0, cell, 0, 0)
    ix0, wl = _lat_span(w, x0, cell, 0, 0)

    ys = (torch.arange(h, device=device, dtype=torch.float32) + float(y0)) / cell
    xs = (torch.arange(w, device=device, dtype=torch.float32) + float(x0)) / cell

    lat = torch.from_numpy(_lattice_np(iy0, ix0, hl, wl, seed, nfields)).to(device)
    iy = (torch.floor(ys).long() - iy0).clamp(0, hl - 1)
    ix = (torch.floor(xs).long() - ix0).clamp(0, wl - 1)
    return lat[:, iy][:, :, ix].unsqueeze(0)


# --------------------------------------------------------------------------- #
# The Global Grain point field
# --------------------------------------------------------------------------- #
#
# Rewritten 2026-08-05. Reported as "renders repetitive pattern when zooming
# out, I can clearly see and feel the grid even when zooming in, sometimes it
# does a good job, sometimes it does not, even with the same config". All three
# complaints were real and each had its own cause; see the section in CLAUDE.md
# for the measurements. In short:
#
#   * the field was addressed on an *axis-aligned* lattice, so its structure
#     lined up with the pixel grid -- the grid you can see;
#   * exactly one point per cell, jittered only within the middle half of that
#     cell, is a near-lattice point process. Zoomed out, an evenly spaced mesh
#     is exactly what "repetitive pattern" looks like;
#   * a domain warp bolted on to hide the first two shredded the discs into
#     torn-paper shapes, and it only *partly* hid the pixel-grid resonance it
#     was added for -- which is the "sometimes good, sometimes not".
#
# What replaces it is one construction (`_grain_points`) used at every setting,
# built out of four decisions, each of which is load-bearing:
#
#   1. the cell lattice is *rotated* against the pixel grid by an irrational
#      slope, so the two grids are incommensurate at every cell size;
#   2. points jitter over their *whole* cell, which the 3x3 search below is
#      still exact for -- see `_GRAIN_RINGS`;
#   3. several points per cell, a fraction of them absent, so the local density
#      genuinely varies instead of being one point per cell everywhere;
#   4. grain brightness is modulated by a multi-octave cluster field, which is
#      what gives the layer structure at scales far above a single clump.

# Rotation of the grain lattice against the pixel grid, in radians -- 31.717
# degrees, the golden-ratio slope.
#
# **This is what replaced the domain warp, and it is a better answer to the
# same problem.** The warp existed because when the working cell size lands on
# (or near) a whole number of pixels the two grids phase-lock: every pixel sits
# at the same fractional offset inside its own cell, so no pixel is ever near a
# point and the field cannot reach its own amplitude. Measured on the old
# construction, cell 1.00 scored 0.123 std against 0.193 at 1.05 or 0.95 -- a
# 35% amplitude hole sitting exactly on the round numbers a slider lands on.
#
# A rotation removes the phase lock outright rather than papering over it: an
# irrational slope means the cell grid and the pixel grid are never
# commensurate at *any* cell size, so there is no setting left for the field to
# resonate at. It is also strictly cheaper -- the warp cost a whole
# `_value_noise` call of its own and forced the neighbour search out to 5x5 to
# pay for its travel, where this is four multiplies on the coordinate ramp and
# leaves the search at 3x3.
#
# And it is the only fix available for the *axis alignment*, which the warp
# never addressed at all. Rotating a value-noise field would merely rotate its
# quilt, because that quilt is made of plateaus at the lattice points; rotating
