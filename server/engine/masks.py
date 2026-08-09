from __future__ import annotations

import torch

from .primitives import _luma

def _source_masks(m: torch.Tensor, pivot: float = 0.5) -> tuple[torch.Tensor, ...]:
    """The four visibility envelopes of the source-masked global layers.

    ``m`` is the frame **already clamped to 0..1** -- the caller's job, and not
    optional: `render` leaves `out` unclamped until the very end because the
    sharpening at step 10
    needs the headroom, and halation routinely drives a channel past 1.0. An
    unclamped envelope would run a layer louder than its own slider in
    highlights and *invert* it wherever a channel had gone negative.

    Returns ``(red, green, blue, lightness)``, each ``[1,1,h,w]`` in 0..1 and
    each broadcasting across all three channels: these select *where* a layer
    shows, never which channel it lands in.

    **The colour three are hue masks, not channel values.** ``R - max(G, B)``,
    which factors exactly into "how red in hue" x "how bright" -- so grain grows
    both as an area gets redder and as it gets lighter, which is what was asked
    for, and it needs no calibration constant to say it. The literal alternative,
    ``mask = R``, was rejected: white and grey have all three channels high, so
    all three layers would fire at full strength on neutral content and pile up
    into what is really just a brightness mask wearing three sliders.

    Two consequences worth knowing. The three are **mutually exclusive** -- only
    one channel can be the largest, so at most one is non-zero at any pixel and
    they can never stack on each other. And on a real photograph hue dominance
    rarely passes 0.3-0.5, so at equal slider settings these read quieter than
    Global Intensity; the mask is taking its share, which is the whole point.

    **Lightness is a mid-tone bell**, not a ramp: grain peaks at ``pivot`` and
    fades to nothing toward *both* white and black. The triangle is built from
    the distance to the pivot, normalised by the room on that side -- so the two
    halves are stretched independently and the bell still reaches exactly 0 at
    both ends wherever the peak is put. At ``pivot = 0.5`` that is literally
    ``1 - |2L - 1|``, the shape this had before the control existed, so the
    default is bit-identical to it. The smoothstep on top rounds off the kink at
    the peak and, more usefully, flattens the approach to both ends, so the
    layer leaves the highlights and the shadows gradually instead of at a
    constant rate. Zero at pure black and pure white, ~0.10 a tenth of the way
    in from either, 1.0 at the pivot.

    The pivot is clamped away from 0 and 1: at either extreme one side of the
    bell has no room left and the division would be by zero.

    Reads the frame as it stands *before* any of the five layers is added, so
    the envelopes come from the picture rather than from the grain already laid
    on it. They still carry the main grain's own noise, which is uncorrelated
    with these fields and zero-mean -- grain modulating grain, which is what
    print grain sitting on negative grain actually does.

    The one construction worth checking rather than assuming is ``clamp_min(0)``
    on a *neutral* area: the hue difference there is wandering either side of
    zero, so rectifying it leaves a small positive envelope where the answer
    should be nothing at all, and the three colour layers would bleed onto grey.
    Measured on a flat 0.5 plate with the main grain at 40 and the flat layer at
    20: Source Red at 100 renders sigma 0.000197 against the flat layer's
    0.038469, which is 0.5% of it, and a mean shift of +1e-6. Blurring the mask
    would remove even that, and would cost `pad_for` a kernel it does not
    otherwise need, for something already three orders of magnitude down.
    """
    r, g, b = m[:, 0:1], m[:, 1:2], m[:, 2:3]
    lum = _luma(m)
    pv = min(max(float(pivot), 0.02), 0.98)
    d = lum - pv
    # Each side normalised by its own room, so the bell is asymmetric but still
    # hits 0 at both ends. `torch.where` rather than a branch: the two sides are
    # both present in every tile.
    t = (1.0 - torch.where(d < 0.0, -d / pv, d / (1.0 - pv))).clamp(0.0, 1.0)
    return (
        (r - torch.maximum(g, b)).clamp_min(0.0),
        (g - torch.maximum(r, b)).clamp_min(0.0),
        (b - torch.maximum(r, g)).clamp_min(0.0),
        t * t * (3.0 - 2.0 * t),
    )


def _grain_delta(base: torch.Tensor, g: torch.Tensor, mode: int) -> torch.Tensor:
    """What one grain layer at **full strength** does to ``base``, as a delta.

    The five Global Grain layers composite the way layers in an image editor do:
    the grain is an image, ``L = 0.5 + g/2``, mid grey where there is no grain;
    the blend mode combines it with what is underneath; and the layer's amount
    and mask together act as its opacity. Returning the *difference* rather than
    the blended result is what makes that last part a plain lerp at the call
    site -- ``out + alpha * delta`` -- so every mode fades to nothing at 0 and
    the per-pixel mask needs no second code path.

    ``mode`` indexes `params.GLOBAL_BLENDS`.

    **Add returns ``g`` untouched, and that is deliberate rather than an
    optimisation.** Reconstructing it as ``(base + g) - base`` is not the same
    float, and Add is the default: every shipped preset has to render bit for
    bit what it rendered before this function existed.

    Every other mode is computed against ``base`` **clamped to 0..1**, because
    Overlay and friends are only defined there and `out` is deliberately
    unclamped through most of the pipeline. The delta is still added to the
    unclamped frame by the caller, so a blown highlight keeps the headroom
    sharpening relies on instead of being flattened to 1.0 on its way past.

    Since 2026-08-09 this layer runs below Film Texture, and Tone Response --
    which ends in a clamp -- is above that, so in practice the frame arriving
    here is already inside 0..1. The clamp stays: it is what makes the mode
    definitions correct rather than incidentally correct, and a stage moving
    back above the tone curve must not silently change what Overlay means.

    A note on the two that are not symmetric about mid grey: Multiply and Screen
    have no neutral value in 0..1 at all -- multiplying by a mid-grey layer
    halves the picture -- so their delta is dominated by a constant darkening or
    lightening that the grain then modulates. That is what those modes *are*,
    and the amount slider is the only thing holding them back. They are here
    because they were asked for; Overlay and Soft Light are the two that behave
    like a grain control.
    """
    if mode == 0:                                        # Add
        return g
    b = base.clamp(0.0, 1.0)
    lay = g * 0.5 + 0.5
    if mode == 1:                                        # Overlay
        o = torch.where(b <= 0.5, 2.0 * b * lay,
                        1.0 - 2.0 * (1.0 - b) * (1.0 - lay))
    elif mode == 2:                                      # Soft Light
        # The W3C / Photoshop curve, not the cheap `2*b*lay + b*b*(1-2*lay)`
        # approximation: that one has a discontinuous derivative where the
        # layer crosses mid grey, and a grain layer crosses mid grey at roughly
        # half of all pixels, so the kink would be everywhere at once.
        d = torch.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, torch.sqrt(b))
        o = torch.where(lay <= 0.5,
                        b - (1.0 - 2.0 * lay) * b * (1.0 - b),
                        b + (2.0 * lay - 1.0) * (d - b))
    elif mode == 3:                                      # Hard Light
        o = torch.where(lay <= 0.5, 2.0 * b * lay,
                        1.0 - 2.0 * (1.0 - b) * (1.0 - lay))
    elif mode == 4:                                      # Multiply
        o = b * lay
    else:                                                # Screen
        o = 1.0 - (1.0 - b) * (1.0 - lay)
    return o - b
