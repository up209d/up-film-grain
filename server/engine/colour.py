from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants.core import _LUMA
from .constants.grade import (
    _GRADE_TONE_MAX, _RECON_CEIL, _RECON_HI, _RECON_LO, _RECON_MIN_EVIDENCE, _RECON_ROLL_GATE_FRAC, _RECON_ROLL_KNEE,
)
from .primitives import _blur, _smootherstep, _smoothstep

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _apply_lut(x: torch.Tensor, lut) -> torch.Tensor:
    """Trilinear 3D LUT lookup. ``x`` is [1,3,h,w] display-referred in 0..1.

    ``lut`` is a ``server.lut.Lut``; duck-typed rather than imported so the
    engine keeps no dependency on the file loader. It supplies the table as a
    ``[1, 3, D, H, W]`` volume and its input domain.

    One ``grid_sample`` call, which is trilinear in 3D and runs on the GPU -- so
    a 35-cube and a 65-cube cost the same and neither shows up against the
    stages below. The alternative, gathering eight corners by flat index and
    interpolating by hand, needs int64 index tensors that MPS handles badly and
    eight full-frame gathers of working memory.

    Two things are load-bearing:

    * **``align_corners=True``.** A LUT's first and last samples *are* input 0
      and input 1, not the centres of edge cells. With the default the whole
      table would be read at half a cell's offset -- a small, uniform, entirely
      wrong shift that would look like the LUT being slightly off rather than
      like a bug.
    * **The grid's last dimension is ``(x, y, z)`` mapping to ``(W, H, D)``,**
      and the table is stored ``[c][b][g][r]`` so that maps to ``(r, g, b)``.
      That is why the grid is simply the image's own channels in order. Get it
      backwards and any symmetric LUT still looks fine while every real one is
      channel-swapped, which is what ``verify.py`` uses an asymmetric table to
      pin.

    ``padding_mode="border"`` clamps rather than reflecting, so a value that has
    somehow left 0..1 reads the nearest real entry instead of folding back into
    the middle of the cube.
    """
    tab = lut.tensor(x.device)
    n = x
    # Almost every LUT in the wild declares the 0..1 domain, so the rescale is
    # skipped rather than paid for -- a per-channel multiply-add over the frame
    # for nothing.
    if lut.dmin != (0.0, 0.0, 0.0) or lut.dmax != (1.0, 1.0, 1.0):
        lo = torch.tensor(lut.dmin, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        hi = torch.tensor(lut.dmax, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        n = (n - lo) / (hi - lo).clamp_min(1e-6)
    grid = n.permute(0, 2, 3, 1).unsqueeze(1) * 2.0 - 1.0  # [1,1,h,w,3] as (r,g,b)
    out = F.grid_sample(
        tab, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return out.squeeze(2)


def _soft_knee(x: torch.Tensor, amount: float, span: float) -> torch.Tensor:
    """Roll values off asymptotically as they approach 1.0.

    Deliberately *not* normalised to land on 1.0. A shoulder is a region of
    falling slope; if it starts at slope 1 and the slope only decreases, the
    curve mathematically cannot reach 1.0 at the top. Forcing it to would make
    the "shoulder" a highlight *boost*, which is the opposite of film. Letting
    it asymptote below white is what gives film its creamy highlights -- and is
    why a film scan's brightest tone is rarely paper white.
    """
    if amount <= 0.001:
        return x
    knee = 1.0 - span * amount
    denom = max(1.0 - knee, 1e-4)
    t = ((x - knee) / denom).clamp_min(0.0)
    return torch.where(x > knee, knee + denom * torch.tanh(t), x)


def _shoulder(t: torch.Tensor) -> torch.Tensor:
    """``1 - exp(-t)``: the roll every recovery in this file blends toward.

    Slope 1 at the knee, so it joins the identity without a seam; asymptotes at
    1, so an unbounded input lands inside a bounded output; strictly increasing
    everywhere, so ordering -- and therefore detail -- is never lost. Shared by
    `_tone_roll` and by highlight reconstruction's own local roll so the two
    cannot drift apart.
    """
    return 1.0 - torch.exp(-t)


def _tone_roll(t: torch.Tensor, amount: float) -> torch.Tensor:
    """The one monotone curve behind both tone-recovery directions.

    ``t`` is distance from the knee toward a rail, in units of the distance to
    that rail: ``t = 0`` is the knee, ``t = 1`` is the rail itself, and
    ``t > 1`` is a value that has already left the cube -- which is the whole
    reason this exists. Returns the rolled distance, to be mapped back the same
    way it came in.

    ``amount > 0`` recovers: a convex blend of the identity and the exponential
    shoulder ``1 - exp(-t)``, so at 1.0 the rail becomes an *asymptote*. Two
    properties fall out of that and both are the point:

    * **Strictly monotone at every setting.** The slope is
      ``1 - amount * (1 - exp(-t))``, which for ``amount <= 1`` is bounded below
      by ``exp(-t) > 0``. Ordering is never lost, so neither is detail: two
      tones that differ before the curve still differ after it.
    * **Unbounded input, bounded output.** Anything from the knee to infinity
      lands inside the cube, monotonically. That is what makes over-range data
      -- from reconstruction, from exposure, from a bright source -- *visible*
      rather than clipped flat, and it is the difference between recovering a
      highlight and merely dimming it.

    ``amount < 0`` expands instead, and keeps the old share-of-headroom form:
    ``t + |amount| * _GRADE_TONE_MAX * quintic(t) * (1 - t)``. Also monotone
    (measured slope stays above ``1 - _GRADE_TONE_MAX``), and it cannot drive an
    in-gamut value out of the cube, which is the guarantee that half of each
    control has always made.

    The asymmetry is deliberate and is the same shape of decision Clarity's is:
    pushing a tone toward a rail and pulling one back off it are different
    operations, and one formula that did both would do neither well. What made
    the previous single formula fail was precisely that its strength was a
    function of the pixel's own level -- ``x + a * m(x) * (1 - x)`` with ``m``
    rising steeply through the band it was gating -- so in the recovering
    direction the ``m'`` term overwhelmed the ``1`` and the transfer *inverted*:
    measured slope **-0.21 over 16% of the range at 1.0**, which does not
    compress highlight detail, it destroys and flips it. Hence a curve whose
    monotonicity is a property of its own algebra rather than of how far the
    slider happens to be pushed.
    """
    if amount >= 0.0:
        # (1 - a) * t + a * shoulder(t), written so a = 0 returns t exactly.
        return t + amount * (_shoulder(t) - t)
    ramp = _smootherstep(0.0, 1.0, t)
    return t + (-amount) * _GRADE_TONE_MAX * ramp * (1.0 - t)


def _recon_estimate(
    img: torch.Tensor, amount: float, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The estimate half of highlight reconstruction: what was the clipped value?

    Returns ``(out, w)`` -- the image with clipped channels raised to their
    estimated true values, which are **above 1.0** by design, and the per-channel
    weight that raising was applied with. `_reconstruct_highlights` then rolls
    that back into the visible range; the two are split so the estimate's
    accuracy can be checked as an equality against a known unclipped scene
    without the roll's compression in the way, which is exactly what
    ``verify.py`` does.

    An 8-bit file clips per channel, not per pixel, and that asymmetry is the
    opening this works through: a warm highlight reaches the ceiling in red
    long before green and well before blue, so across a blown cloud the red
    channel is a flat plateau while green and blue are still recording the
    scene's own gradient. The detail is *in the file*; it is only missing from
    one channel at a time. Where every channel is at the ceiling there is
    genuinely nothing left, and this says so rather than inventing it.

    Per channel, in display-referred space:

    * ``clipped`` is the soft indicator over ``_RECON_LO.._RECON_HI``, and
      ``valid = 1 - clipped`` marks what is still a real measurement.
    * ``q`` is each channel's local level, blurred over ``radius`` but averaged
      **only over whole clean pixels** -- ones with nothing clipped in any
      channel -- so a plateau of ceiling values contributes nothing to the
      estimate of what its own colour should be, and every channel's mean is a
      mean over the *same* pixels. That second half is load-bearing; see the
      comment in the body for the 6% ratio error that using each channel's own
      mask produced, and for why that error made the whole stage a no-op.
      Normalising ``q`` by its own luma-weighted mean across channels gives
      ``k``, the local chromaticity: the colour of the light around here,
      measured where it could be measured.
    * ``guide`` is this pixel's own brightness read off whichever channels are
      still valid, divided back through ``k`` so a surviving channel that the
      local light happens to be poor in does not read as a darker pixel. With
      one channel valid it reduces exactly to ``x_c / k_c``.
    * ``recon = k * guide`` is then the pixel's brightness wearing the local
      chromaticity -- and it exceeds 1.0 exactly as far as the clipped channel
      really did.

    Only ever *raises* a channel, and only ever a clipped one, weighted by how
    clipped it is and by whether there was evidence to work from -- so an
    unblown photograph is untouched and this half cannot darken anything.

    Tile-independent for the ordinary reason: two fixed-radius blurs and
    per-pixel arithmetic, no statistic of the region anywhere.
    """
    lw = torch.tensor(_LUMA, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    clipped = _smoothstep(_RECON_LO, _RECON_HI, img)
    valid = 1.0 - clipped

    # Local chromaticity, averaged over **whole clean pixels** -- ones with
    # nothing clipped in any channel -- rather than over each channel's own
    # valid mask.
    #
    # That distinction is the difference between this working and this quietly
    # not working, and it took a measurement to see. Averaging each channel over
    # wherever *it* was valid compares means taken over different sets of
    # pixels: red's mask stops at the clip boundary while green's runs on into
    # the brighter region past it, so red's mean is drawn from a darker
    # neighbourhood than green's and the ratio between them comes out
    # compressed. Measured on a warm ramp clipping in red only, the estimate
    # landed at k_R = 1.136 against a true 1.205 -- a 6% underestimate, which is
    # enough to put the reconstruction *below* the ceiling it was recovering
    # from, so the ``clamp_min(0)`` swallowed it and the stage did precisely
    # nothing. One shared mask makes every channel's mean a mean over the same
    # pixels, and the ratio exact.
    #
    # Note the two masks answer different questions and both are needed:
    # `clean` is "is this *pixel* a trustworthy sample of the local colour",
    # `valid` below is "is this *channel* of this pixel a trustworthy reading of
    # its brightness".
    clean = valid.amin(dim=1, keepdim=True)
    den = _blur(clean, radius)
    q = _blur(img * clean, radius) / den.clamp_min(1e-3)
    ev = _smoothstep(0.0, _RECON_MIN_EVIDENCE, den)

    # Normalised so a neutral neighbourhood gives k = 1 in every channel.
    # `_LUMA` sums to 1, so this is a mean and not merely a sum.
    qg = (q * lw).sum(dim=1, keepdim=True)
    k = q / qg.clamp_min(1e-4)

    # This pixel's brightness from its surviving channels, read through the same
    # chromaticity so the two are on one scale. With a single channel valid it
    # reduces to x_c / k_c.
    wv = lw * valid
    guide = (wv * img).sum(dim=1, keepdim=True) / (wv * k).sum(
        dim=1, keepdim=True).clamp_min(1e-4)
    recon = (k * guide).clamp(0.0, _RECON_CEIL)

    # Two more things have to be true before a value is trusted. There has to
    # have been a clean pixel within reach to read the colour from (`ev`), and
    # this pixel has to have at least one channel of its own left to read its
    # brightness off -- one white in all three has nothing to be recovered from
    # and must come through untouched rather than through a division by an
    # epsilon.
    w = amount * clipped * ev
    return img + w * (recon - img).clamp_min(0.0), w


def _reconstruct_highlights(
    img: torch.Tensor, amount: float, radius: float,
) -> torch.Tensor:
    """`_recon_estimate`, then the roll that makes its result *visible*.

    Kept as one stage behind one slider because either half alone is useless: the
    estimate without the roll is invisible, and the roll without the estimate is
    just a highlight dimmer. ``pad_for`` carries all three of the kernels
    involved, in series.
    """
    out, w = _recon_estimate(img, amount, radius)

    # The roll that makes any of that *visible*, which the first version of this
    # stage left out and which made the whole control read as dead.
    #
    # Reconstruction's output is above 1.0 -- that is the entire point, it is
    # where the clipped channel really was -- and the section's final clamp then
    # took it straight back off, so the slider moved 0.0004 of mean level on a
    # real photograph. Reported as "I don't see any effect from those sliders at
    # all", and correct. Pairing it with Highlights did work (0.395 of max change
    # against 0.05) but a control that needs a second, differently-named control
    # to do anything is broken however clearly the help text says so.
    #
    # There is a hard constraint here worth stating, because it rules out the
    # tidier designs: **any curve that brings over-range data into view must move
    # in-gamut highlights too.** A gamut map with its knee exactly at 1.0 has to
    # jump -- it would send v = 1 to 1 - d -- so a smooth one needs its knee
    # below 1, and everything above that knee moves. "Visible on its own" and
    # "bit-exact no-op" are therefore in genuine conflict for a *global* curve.
    #
    # The way out is to make it **local**: gate the roll on reconstruction's own
    # weight field, blurred. Then the conflict dissolves rather than being
    # traded off --
    #
    # * where nothing was repaired the gate is 0, the roll is the identity, and
    #   an unblown photograph comes through **bit-exactly** untouched, which is
    #   the property that keeps this a repair tool and not a second highlight
    #   grade;
    # * where something was repaired the roll engages and the recovered detail
    #   appears;
    # * and the gate is smooth, so there is no contour at the boundary -- a
    #   per-pixel gate would outline every repaired region.
    #
    # Same `_shoulder` as `_tone_roll`, but applied **per channel** -- the
    # opposite of the tone stage's channel-max-and-uniform-scale, and the
    # difference is not stylistic.
    #
    # Uniform scaling holds hue *exactly*, which is a virtue in the tone stage
    # because its input is near the cube already. Here the input can be 2-4x over
    # white in **one** channel, and holding the ratio exact then means dragging
    # the other two down by the same factor. Measured on a real photograph before
    # this was changed: a bright warm highlight at (1.000, 0.871, 0.634) came out
    # (1.000, 0.305, 0.222) -- luma 0.882 -> 0.447, a **dark saturated red where a
    # bright highlight had been**, on about 6% of the frame. Exactly the artifact
    # this stage exists to remove, introduced by the stage itself.
    #
    # Per channel, each rolls against its own headroom: the reconstructed 2.0 in
    # red comes back to just under white while green barely moves and blue, below
    # the knee, is untouched. The highlight stays bright and loses a little
    # saturation -- which is what film does as a dye layer approaches saturation,
    # and is the same behaviour `highlight_desat` models further down the
    # pipeline. Fitting an out-of-gamut brightness into the cube costs either
    # saturation or luminance; for a highlight, saturation is the right one to
    # spend.
    # **Dilate before feathering, or the radius fights the repair.** A plain blur
    # of the weight field dilutes it: a blown region 120px across, gated through
    # a sigma-100 blur, comes out with a peak well under 1, so the roll weakens
    # and *less* of the recovered range becomes visible. Measured before this
    # was added, the recovered span ran 0.069 at a 16px radius down to 0.041 at
    # 200px -- i.e. reaching further to find the colour made the repair fainter,
    # which is not what the control says it does.
    #
    # Growing the mask first and feathering the grown version is the standard fix
    # and it decouples the two: the gate stays saturated across everything that
    # was repaired, whatever the radius, and only its outer ramp widens.
    # Separable, as two 1-D max pools, because a single 2-D pool at a 200px
    # radius is a 401x401 window.
    # Dilate **wider than the feather**, or the gate never saturates: a blur of a
    # mask dilated by exactly its own sigma pulls the peak back below 1 near the
    # mask's edge, so the roll runs at partial strength and leaves over-range
    # values for the hard clamp to flatten -- the original bug, in miniature.
    # Growing by 2x the feather leaves the interior at a clean 1.0.
    rg = max(1, int(round(radius * _RECON_ROLL_GATE_FRAC)))
    rd = 2 * rg
    gate = w.amax(dim=1, keepdim=True)
    gate = F.max_pool2d(gate, (1, 2 * rd + 1), stride=1, padding=(0, rd))
    gate = F.max_pool2d(gate, (2 * rd + 1, 1), stride=1, padding=(rd, 0))
    gate = _blur(gate, float(rg))
    d = 1.0 - _RECON_ROLL_KNEE
    t = ((out - _RECON_ROLL_KNEE) / d).clamp_min(0.0)
    return torch.where(
        out > _RECON_ROLL_KNEE,
        _RECON_ROLL_KNEE + d * (t + gate * (_shoulder(t) - t)),
        out,
    )


# Middle grey (0.18 linear) sits near here once sRGB-encoded; the straight-line
# section of the characteristic curve pivots about it.
_MID_GREY = 0.46


def _characteristic_curve(
    x: torch.Tensor, contrast: float, toe: float, shoulder: float,
) -> torch.Tensor:
    """Film's density-vs-log-exposure response.

    The classical three-part model, in the order film exhibits it: a toe where
    too little light was recorded to develop proportionally, a straight-line
    section whose slope is the gamma, and a shoulder where the halide is
    approaching saturation.
    """
    if contrast > 0.001:
        x = _MID_GREY + (x - _MID_GREY) * (1.0 + 1.1 * contrast)
    x = _soft_knee(x, shoulder, 0.55)
    if toe > 0.001:
        x = 1.0 - _soft_knee(1.0 - x, toe, 0.40)
    return x
