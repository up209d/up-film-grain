from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants.core import _LUMA

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

# Sigma above which `_blur` drops to a reduced-resolution path, and the residual
# sigma it aims for on the small grid.
#
# The threshold *is* the safety argument. `k = floor(sigma / _BLUR_DECIMATE_TO)`
# is below 2 for every sigma under 32, so everything below that falls through to
# the exact path and is **bit-identical** -- which covers every kernel this
# pipeline uses except the scratch soften, and halation or highlight
# reconstruction when a preset pushes their radii past 16 at supersample 2.
#
# 16 rather than a smaller target because the error is set by how much of the
# gaussian the box-plus-resample approximates away. Measured against the exact
# blur on a scratch-like field, as a share of the result's own peak:
#
#   target  4 -> sigma 40: 1.7%   sigma 80: 5.3%
#   target  8 -> sigma 40: 0.8%   sigma 80: 1.5%
#   target 16 -> sigma 40: 0.3%   sigma 80: 0.7%
#
# and 16 is still 24x faster at sigma 40 and ~5000x at sigma 80 on the CPU.
_BLUR_EXACT_MAX_SIGMA = 32.0
_BLUR_DECIMATE_TO = 16.0


def _blur(
    x: torch.Tensor, sigma: float, origin: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Separable gaussian blur with reflect padding.

    Cost is O(N * 6 sigma), which is fine until a stage asks for a sigma in the
    tens. Scratch softening does: its radius is
    ``scratch_soften * 3 * scratch_width * scale``, so a preset's 0.9 on a 14.85px
    scratch is **sigma 40 at scale 1 and sigma 80 at supersample 2** -- a 483-tap
    separable convolution over the whole frame. Measured 2026-08-08, that single
    call was **101s of a 154s CPU export**, and it moves 0.04% of pixels by more
    than one 8-bit level.

    So above `_BLUR_EXACT_MAX_SIGMA` the blur is done on a decimated grid:
    average-pool by ``k``, blur at the residual sigma, resample back. A gaussian
    that wide has nothing above 1/sigma cycles per pixel left to lose, which is
    why this is an approximation only in arithmetic and not in what it renders.

    **``origin`` is what makes this legal, and without it the decimated path is
    off.** ``avg_pool2d`` lays its grid from the tensor's own top-left, so two
    tiles covering the same pixel would pool it into differently-phased cells and
    the export would seam -- invariant 1, and it is not theoretical: the first
    version of this seamed colour grading at 6.35e-03 against a 2e-03 bar. Given
    the tile's absolute offset in working pixels, the grid is shifted onto
    absolute multiples of ``k`` instead, so every tile pools the same pixels into
    the same cells. Callers that cannot supply it get the exact path, always --
    the default is the safe one.

    Three details that are not free to get wrong:

    * **The residual sigma accounts for the pool's own blur.** ``avg_pool2d`` by
      ``k`` is a box filter of variance ``(k^2 - 1) / 12``, so asking for
      ``sigma / k`` on the small grid over-blurs. Subtracting it in quadrature is
      what keeps the result the same width as the exact one.
    * **Bicubic on the way back, not bilinear.** Both are accurate here -- the
      field being resampled is smooth by construction -- but MPS's bilinear
      `interpolate` is pathologically slow at large upsample factors: measured
      0.9x against the exact blur at sigma 80, where bicubic is 34x.
    * The small grid's own border still reflects, exactly as the full-resolution
      kernel does, so `pad_for`'s existing 3x reach covers it unchanged.
    """
    if sigma < 0.05:
        return x
    if sigma > _BLUR_EXACT_MAX_SIGMA and origin is not None:
        k = int(min(64, math.floor(sigma / _BLUR_DECIMATE_TO)))
        if k >= 2:
            h, w = x.shape[-2:]
            # Shift the pool grid so that absolute coordinate 0 lands on a cell
            # boundary. `round` rather than `int`: the offset arrives as a float
            # that is a whole number of working pixels, and truncating a value
            # like 1023.9999997 would phase the grid one pixel out.
            py = int(-round(origin[0])) % k
            px = int(-round(origin[1])) % k
            xp = F.pad(x, (px, 0, py, 0), mode="replicate") if (py or px) else x
            ph, pw = xp.shape[-2:]
            # `ceil_mode` so the small grid covers the whole frame; a part-full
            # final cell only widens the effective support at the far edge, which
            # the tile overlap already covers.
            small = F.avg_pool2d(xp, k, ceil_mode=True)
            resid = math.sqrt(max(sigma * sigma - (k * k - 1) / 12.0, 0.0)) / k
            small = _blur(small, resid)
            big = F.interpolate(
                small, size=(ph, pw), mode="bicubic", align_corners=False,
            )
            return big[..., py:py + h, px:px + w]
    r = max(1, int(math.ceil(sigma * 3.0)))
    # reflect padding requires the pad to be smaller than the dimension
    r = min(r, min(x.shape[-1], x.shape[-2]) - 1)
    if r < 1:
        return x
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return x


def _luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = _LUMA
    return x[:, 0:1] * r + x[:, 1:2] * g + x[:, 2:3] * b


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    if e1 - e0 < 1e-5:
        return (x >= e1).to(x.dtype)
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    """Quintic easing -- second derivative is continuous at both ends.

    Used for the luminance band. Cubic smoothstep has a discontinuity in
    curvature where it meets the flat region, and on a wide tonal ramp that
    shows up as a faint edge where grain "switches on". Quintic does not.
    """
    if e1 - e0 < 1e-5:
        return (x >= e1).to(x.dtype)
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _spread(v: torch.Tensor) -> torch.Tensor:
    """Stretch a value-noise field so it actually uses 0..1.

    Quintic value noise is strongly centre-weighted: measured, p10-p90 spans
    only 0.41-0.71 with a standard deviation of 0.11. Used raw as a variation
    field it makes everything land near the middle however wide the range it
    is mapped onto -- which is why the light leaks stayed uniform even with an
    9x spread of reach available to them. The endpoints straddle the field's
    median (0.578), so this stretches without biasing the result up or down.
    """
    return _smoothstep(0.38, 0.78, v)


def _hsv_to_rgb(h_deg: float, sat: float, val: float = 1.0) -> tuple[float, float, float]:
    """HSV to RGB for a single colour, in plain Python.

    The halation tint is one constant per render, not a field, so there is no
    reason to build a tensor for it.
    """
    h = (h_deg % 360.0) / 60.0
    c = val * max(0.0, min(1.0, sat))
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = val - c
    r, g, b = (
        (c, x, 0.0), (x, c, 0.0), (0.0, c, x),
        (0.0, x, c), (x, 0.0, c), (c, 0.0, x),
    )[int(h) % 6]
    return (r + m, g + m, b + m)


def _hue_sat(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hue in degrees and HSV saturation, per pixel.

    Saturation is chroma over value, the HSV definition, matching what
    ``vibrance`` already uses: it reads a deep blue as fully saturated however
    dark it is, where distance from the luma axis would call the same blue
    unsaturated.
    """
    mx = x.amax(dim=1, keepdim=True)
    mn = x.amin(dim=1, keepdim=True)
    c = mx - mn
    sat = c / mx.clamp_min(1e-6)
    cc = c.clamp_min(1e-6)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    h = torch.where(
        mx == r, ((g - b) / cc) % 6.0,
        torch.where(mx == g, (b - r) / cc + 2.0, (r - g) / cc + 4.0),
    ) * 60.0
    # Hue is undefined on grey, and the ratio above is 0/0 there.
    return torch.where(c < 1e-6, torch.zeros_like(h), h), sat


def _rotate_hue(x: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """Rotate colours about the grey axis by a per-pixel angle.

    Rodrigues about (1,1,1)/sqrt(3), which is exactly a hue rotation in RGB:
    it leaves grey untouched by construction (grey lies *on* the axis) and
    preserves the channel sum, so it changes colour without changing how
    bright the pixel is. Cheaper and better behaved than a round trip through
    HSV, which has to divide by a chroma that goes to zero.
    """
    th = deg * (math.pi / 180.0)
    c, s = torch.cos(th), torch.sin(th)
    k = 1.0 / math.sqrt(3.0)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    # k . v, and the axis-aligned part that the rotation leaves alone.
    axis = (r + g + b) * k * k * (1.0 - c)
    return torch.cat([
        r * c + (b - g) * k * s + axis,
        g * c + (r - b) * k * s + axis,
        b * c + (g - r) * k * s + axis,
    ], dim=1)


def _warp(
    x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, mode: str = "bilinear",
) -> torch.Tensor:
    """Resample ``x`` displaced by ``(dx, dy)``, both given in working pixels.

    Shared by edge jitter, edge sanding and scatter -- they differ only in the
    spatial frequency of the field they hand in, not in how it is applied.

    ``mode="nearest"`` makes the result an exact *copy* of a source pixel
    rather than a blend of four, which is the whole point of the scatter
    stage: bilinear resampling at a fractional offset is a 2x2 average, and an
    average is precisely the thing that stage exists not to do. Callers using
    it hand in whole-pixel displacements, so the choice of nearest neighbour
    is unambiguous rather than resting on which side of a half-pixel the
    floating-point arithmetic lands.
    """
    h, w = x.shape[-2:]
    ys = torch.linspace(-1.0, 1.0, h, device=x.device)
    xs = torch.linspace(-1.0, 1.0, w, device=x.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    # Pixel displacements into grid_sample's normalised -1..1 coordinates.
    gx = X.unsqueeze(0).unsqueeze(0) + dx * (2.0 / max(w - 1, 1))
    gy = Y.unsqueeze(0).unsqueeze(0) + dy * (2.0 / max(h - 1, 1))
    grid = torch.stack([gx[:, 0], gy[:, 0]], dim=-1)
    return F.grid_sample(
        x, grid, mode=mode, align_corners=True, padding_mode="border"
    )


def _isophote(
    lum: torch.Tensor, dir_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit tangent along the contour, plus the gradient magnitude.

    Shared by edge sanding and anti-aliasing, which want the same vector for
    opposite-scale reasons -- one polishes roughness off a contour, the other
    takes stair-steps off one -- and differ only in how far they then reach
    along it and what they gate the result on.

    ``dir_sigma`` is not optional and must not be zero. Taken per-pixel the
    gradient follows whatever noise is present and the tangent sands in
    circles; worse, where the gradient is weak the direction is a ratio of two
    near-zero numbers, so it swings on floating-point alone and a filter
    reaching along it samples somewhere else entirely. That is not just noisy,
    it made tiled exports seam: two tilings hand the gradient marginally
    different values. Callers gate on the returned magnitude for the same
    reason.
    """
    gl = _blur(lum, dir_sigma)
    px_ = F.pad(gl, (1, 1, 0, 0), mode="replicate")
    gx_ = (px_[..., 2:] - px_[..., :-2]) * 0.5
    py_ = F.pad(gl, (0, 0, 1, 1), mode="replicate")
    gy_ = (py_[..., 2:, :] - py_[..., :-2, :]) * 0.5
    mag = (gx_ * gx_ + gy_ * gy_).sqrt().clamp_min(1e-6)
    # The tangent is the gradient turned 90 degrees.
    return -gy_ / mag, gx_ / mag, mag


def _edge_magnitude(
    img: torch.Tensor, radius: float, chroma: float,
) -> torch.Tensor:
    """High-pass magnitude of ``img`` at ``radius``, optionally seeing colour.

    **The edge masks were luma-only, and that is why they missed edges.** Every
    mask in the Edge Destruction family -- softening's step gate, the envelope
    jitter and sanding are weighted by, the erosion weight -- came from
    ``_luma(img)``, a Rec.709 weighted sum. A boundary between two colours of
    equal luminance is *flat* in that signal, so none of them fired on it.
    Measured on a red/green edge at identical luma: softening moved 0.29 8-bit
    levels against 5.20 for the same-size luma edge, and jitter and sanding were
    17-21x blind the same way. Photographs are full of such edges -- foliage
    against sky, skin against fabric -- which is exactly the "a lot of edges it
    ignores" that was reported.

    ``chroma`` 0 returns the luma high-pass **bit for bit**: blur is linear, so
    the luma of the blurred frame is the blur of the luma, and the weighted sum
    below is that same number. That is what lets the control ship at a value
    that changes nothing and be turned up deliberately.

    At 1 the magnitude is the largest high-pass any single channel shows. On a
    neutral edge every channel carries the same step, so the two agree exactly
    and greyscale content is unaffected at any setting; only coloured edges
    gain. Blended rather than switched so the slider is continuous, and clamped
    at zero so it can only ever *add* sensitivity.
    """
    if chroma <= 0.001:
        # Seeing colour costs a three-channel blur where luma costs one, so the
        # off setting takes the one-channel path rather than computing three and
        # discarding two. Identical either way: blur is linear, so the luma of
        # the blurred frame is the blur of the luma.
        lum = _luma(img)
        return (lum - _blur(lum, radius)).abs()
    hp = img - _blur(img, radius)
    r, g, b = _LUMA
    m_lum = (hp[:, 0:1] * r + hp[:, 1:2] * g + hp[:, 2:3] * b).abs()
    m_max = hp.abs().amax(dim=1, keepdim=True)
    return m_lum + chroma * (m_max - m_lum).clamp_min(0.0)
