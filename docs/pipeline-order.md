# Pipeline order

## Pipeline order matters at both ends

Six stages are placed by *position*, not by what they compute, and moving them
breaks their whole purpose:

* `Colour Grading` (step -1) is above everything, `pre_blur` included. Every
  stage below it models an emulsion; this is the decision about what the
  photograph *is* before any of that runs. Put it after the film stages and it
  grades grain, halation and dust along with the picture, and a LUT built to be
  fed a photograph is fed a rendered negative instead. Within the block the LUT
  is last, after the eleven adjustments ahead of it, because the adjustments exist
  to hand the LUT the picture it was meant to read — and **highlight
  reconstruction is first**, above white balance, because it is the only stage
  that changes what the picture *is* rather than how it is rendered. White
  balance multiplying a channel sitting on the ceiling multiplies a wrong number;
  exposure raises a plateau as a plateau. Restore the channel first and all three
  are working on the scene instead of on the file's ceiling.
* `pre_blur` (step 0) is before `lum_ref` is taken, which is the only thing
  separating it from `micro_blur` — same kernel, same linear light. See the
  section below in this file.
* `scatter` (step 1) is before `micro_blur` (step 1b), and swapping the two makes
  the pair come out *harder* on borders than the blur alone — scatter drops a
  hard step back into a blurred gradient. See `docs/edge-destruction.md` for the table.
* `aa_strength` (step 1c) is in the optical block and, crucially, *before the
  masks are measured* — otherwise the grain keeps keying on the jaggies the
  stage just removed. See `docs/edge-destruction.md`.
* The **density luma** `lum_d` (step 6b) is taken directly after the
  characteristic curve and base fog and *above* edge softening, jitter and
  sanding. Both controls keyed on "how much silver is here" read it — the
  Luminance Response band and Shadow Clumping. See the section below in this
  file; it moved there on 2026-08-06 and the move is measurable.
* `global_*` (step 13) is after every mask, so it reaches the regions the masks
  protect. Fold it in earlier and it becomes just more masked grain.
* `sharpen` (step 14) is dead last, **after** the grain stages, because the
  high-frequency content it amplifies is meant to be the grain. Run it before
  them and it sharpens a clean image and leaves the grain flat — the opposite
  of the point. It is distinct from `acutance`, which is deliberately extracted
  from the *pre-grain* base so it sharpens the image *without* touching grain.
  Two sharpeners, opposite intents, and the difference is entirely where they
  sit. Measured: `sharpen 0.8` takes grain to 140% and image acutance to 133%;
  on a flat field it is a no-op to 6e-08, so it genuinely invents nothing.

`master_opacity` is later still and is not on that list, because it is not in
`render()` at all — it is applied in `render_supersampled` after the pool, and
it has to be. See its own section at the bottom of this file for why blending
one level down would make
"no effect" cost sharpness.

`ENGINE.render_view` / `render_crop` are no longer called by `main.py` but are
still exercised by `verify.py`, which is the only place the crop and zoom
invariants are checked at all now. Do not delete them.

## `pre_blur` is the same kernel as `micro_blur` and a different stage (added 2026-08-02)

Step 0, ahead of `pre_sharpen` and of everything else, in linear light. It is a
plain separable gaussian on the source, one radius in full-resolution pixels,
shipping at 0. Asking "why not just raise `micro_blur`?" is the right question,
and the answer is entirely positional — **`pre_blur` runs before `lum_ref`**:

* Every mask downstream is measured from `lum_ref`, so the edge mask, the
  hard-edge step mask and the smooth-area guard all read the *softened* frame.
  `micro_blur` is deliberately excluded from that (`verify.py` pins it at 100%
  of unblurred grain) so that dialling in diffusion cannot quietly cost noise
  you never asked to lose. Here the coupling is the feature: soften the source
  and the grain follows the softer edges and backs off where detail has gone.
  Measured on the half-border/half-texture plate, at 3px: `micro_blur` holds
  grain at 100%, `pre_blur` takes it to **20%**.
* It runs before `pre_sharpen`, so a broad radius here against a tight one
  there is a detail-killing pair the pipeline could not otherwise express —
  soften everything, then put the bite back at one chosen scale. The other
  order would just throw the sharpening away, which is what the ordering check
  tests: `pre_sharpen 8 @ 3px` on top of the pre-blur measures **482%** of the
  pre-blur's own edge slope, so the sharpen demonstrably sees the blurred
  image rather than being wiped by it.

**In linear light, and that is not a formality here.** Averaging gamma-encoded
values holds the *encoded* mean, which is a fraction of the light it stands for,
so every edge the kernel crosses comes out darker than the light that made it.
Measured on a black/white border with a 4px radius, mean linear luminance in the
transition band: **0.500 in linear, ~0.21 encoded.** `verify.py` asserts the
0.500. The transfer round trip is gated on the radius so it costs nothing when
the stage is off.

It is **not** `edge_soften` and does not pretend to be: `edge_soften` exists
because a global blur takes texture down with the edges (2% of fine texture
survives at 3px against 93% for softening). `pre_blur` keeps 3% of the texture,
by design — the whole point is destroying detail, and the user has asked for
detail destruction as an in-scope goal. Reach for `edge_soften` to take the snap
off borders and leave fabric; reach for this to take a digital-sharp source down
before the emulsion goes on.

`pad_for` adds `pre_blur + micro_blur`, **summed rather than the widest
winning**: micro-blur reads pixels the pre-blur has already spread, so the two
are kernels in series and their reaches genuinely add.

Cost on a 6MP render at 2×: **0.69s off, 0.75s at 2px, 0.80s at 8px**, with
`pad_for` 108 → 114 → 132px. A separable gaussian plus one transfer round trip
is cheap; the padding is what you actually pay for at 8px.


## Luminance Response is measured on density, not on the picture (moved 2026-08-06)

Audited on request. The mask that decides how much grain each tone carries used
to be computed at step 9, immediately before the grain field it multiplies. It
is now computed at **step 6b**, immediately after base fog — before edge
softening, edge jitter and sanding touch the frame.

**Only the measurement moved.** `m` is still applied at step 10, still
multiplying the grain field after `_grain_field` builds it, exactly as it always
was — worth stating outright because the distinction is easy to lose and the
user reasonably read the change as moving the suppression itself:

```python
g = self._grain_field(h, w, y0, x0, lum_d, p, scale)   # field built
weight = m * ((1.0 - eb) + eb * edge)                  # mask applied, step 10
out = base + g * weight * amp
```

**The argument.** What this mask asks is *how dense is the negative here*, and
the answer is settled the moment the characteristic curve and base fog have run.
It is a property of development. Everything between the old and the new position
is a property of *geometry* — softening a border, wandering an edge, sanding the
burrs off it — and none of those change how much silver was deposited.

Read at the old position the mask was measured off a `base` those three stages
had already been through, which is wrong in the specific way step 7 and the
smooth-area guard are both written to avoid: a blurred frame's luma is not the
density the emulsion recorded, so softening the picture silently moved the grain
around. It also meant edge jitter warped the mask along with the image, which is
the wrong way round — jitter displaces where the *picture* is, not how dense the
silver is.

**The measurement, and it is not subtle.** Put a hard black-to-white step on a
frame and set the band to mid-tones only, so both sides of the border are
suppressed and the frame should carry no grain at all. Now soften the border
hard. Softening invents a mid-tone ramp across it that was never in the
photograph — and the old order believed it, laying a **0.095 sigma ribbon of
grain** along a border whose two sides are both meant to be clean. At the new
position it reads **0.00000**. `verify.py` pins both numbers, and it pins the
control too (feeding the engine that same softened frame as its *input* really
does produce the 0.095, so the check cannot pass by measuring nothing).

At default parameters the difference is small, which is why this survived: the
mask is blurred over 3px anyway and the shipped tone curves are neutral. It
shows up exactly where a user would reach for it — a soft border, a narrow
grain band, or both. Measured across all eleven shipped presets on a test scene,
the move is worth a **mean of 0.00 8-bit levels** and a max of 1 to 33 levels,
and every one of those maxima sits on a softened border. The look does not move;
the artifact does.

### Shadow Clumping had to come with it

`_grain_field` reads a luma too, for one thing only: Shadow Clumping, which
enlarges the clumps where the negative is thin. That is the *same* physical
question the band asks — how dense is this area — so it now reads the same
`lum_d`. Moving one and not the other left two halves of one question answered
from two different frames, which is worse than either consistent choice, and it
was a defect in the first pass at this rather than a deliberate split.

`lum` still exists alongside `lum_d` and is still recomputed after every stage
that moves a pixel. It is the luma of the picture **as it currently stands**,
and the sanding filter genuinely needs that one — it steers along the contour it
can see, not along the one development recorded. Two names because they are two
different quantities.

## `master_opacity` lives outside `render()`, and it has to (added 2026-08-03)

A master cross-fade of the finished frame back over the untouched source.
`Output` group, bottom of the panel, default **1.0** — which makes it the one
parameter whose neutral value is not zero, and therefore the one that must stay
**out of `NEUTRAL_ZERO`**. Put it in and `is_neutral` could never be true,
because the list is checked for values at zero.

**It is applied in `render_supersampled`, after the `avg_pool`, and nowhere
else.** That is not tidiness, it is the only place it can be correct. Inside
`render()` at ss > 1 there is no untouched input to blend against — what that
method receives is already a bicubic upsample, so blending there and pooling
down makes opacity 0 return the up-then-down round trip, which `is_neutral`
already documents as **1.0e-01 softer** than the source on hard edges. "No
effect" would quietly cost sharpness. `verify.py` pins 0.00e+00 against the
source at supersample 1, 2 *and* 3 for exactly this.

Sitting at that seam also means every entry point inherits it — `render_image`
for the export, `render_view` for the preview — so the two cannot disagree
about what half strength looks like. And it is per-pixel against the tile's own
input, so `pad_for` is untouched.

**Blended display-referred, not in linear.** The reasoning that forces
`pre_blur` and halation into linear light does not carry: this is a
compositing control — "how much of the edit do I keep" — not a physical average
of light. The two are genuinely different, and not where you would guess.
Measured on a grained frame at half strength: mean deviation 5.4e-04 and
overall brightness within +0.05%, but a **worst case of 0.146** on individual
pixels, concentrated in the shadows where the transfer curve is steepest.

That concentration is the argument for encoded. Blending in the space the eye
reads makes the slider linear in *visible* deviation, so 0.5 is half the grain
everywhere. In linear the same 0.5 takes more than half out of the shadows and
less out of the highlights — an opacity control that changes the look's balance
as you dial it back, which is not what an opacity control is for.

Two smaller things:

* **Opacity 0 short-circuits before the render**, so dragging toward zero gets
  cheaper rather than paying for a full render to discard.
* **The section's mute button is correct even though it looks inverted.**
  Muting sets a group to its neutral value, which here is 1.0 — so muting
  Output removes the *dial-back*, restoring full strength. That reads oddly
  until you notice the section's contribution *is* the dial-back, and that
  every other section's mute is likewise a no-op when it is already at its
  no-op value.

Not to be confused with `global_opacity`, which mixes the Global Grain layer
alone. The names are close and the help text says so outright.

