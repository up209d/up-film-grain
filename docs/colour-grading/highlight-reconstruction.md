<!-- part of docs/colour-grading.md -->

## Highlight reconstruction: an 8-bit file clips per *channel* (added 2026-08-05)

`grade_recover` + `grade_recover_radius`, at the very top of Colour Grading,
above white balance. The only stage in the section that *adds* information rather
than rearranging it, and the answer to "restore the details from the original
source."

The opening it works through: **clipping is per channel, not per pixel.** A warm
highlight reaches the ceiling in red long before green and well before blue, so
across a blown cloud red is a flat plateau while green and blue are still
recording the scene's own gradient. The detail is *in the file*; it is only
missing from one channel at a time.

Per channel, display-referred: a soft `clipped` indicator over
`_RECON_LO.._RECON_HI` marks what hit the ceiling; the local chromaticity `k` is
read from the neighbourhood; a `guide` gives the pixel's own brightness from
whichever channels are still valid, divided back through `k`; and
`recon = k · guide` is the pixel's brightness wearing the local colour — above
1.0 exactly as far as the clipped channel really was. It only ever *raises* a
channel, and only a clipped one.

**Two masks, two different questions, and conflating them is what broke the first
version.** `clean = min over channels of valid` is "is this *pixel* a trustworthy
sample of the local colour"; `valid` per channel is "is this *channel* of this
pixel a trustworthy reading of its brightness". The first attempt averaged each
channel over its own valid mask — which compares means taken over **different
sets of pixels**: red's mask stops at the clip boundary while green's runs on into
the brighter region past it, so red's mean comes from a darker neighbourhood and
the ratio between them comes out compressed. Measured, `k_R` landed at 1.136
against a true 1.205 — a 6% underestimate, enough to put the reconstruction
*below* the ceiling it was recovering from, so the `clamp_min(0)` swallowed it and
**the stage did precisely nothing.** One shared mask makes the ratio exact.

It is verified as an **equality**, not a judgement. The test plate is a warm ramp
(1.00, 0.80, 0.62) running to 1.45, so the true scene is known and the 8-bit file
is `clip(truth)`: reconstruction must *equal* the truth, and it does — **0.0%
error** at 0/20/40/60/90px into the blown zone at a 64px radius.

Three properties hold by construction and are pinned: it never darkens anything,
it is a no-op on a frame with nothing blown, and a pixel white in **all three**
channels comes through untouched — there is genuinely nothing in the file there,
and admitting it beats inventing texture. Past the radius there is no surviving
sample of the channel's own colour in reach, so it fades out rather than
extrapolating; `verify.py` pins that *degradation* explicitly, so a later change
that started inventing values out there fails.

### It shipped invisible, and "documented as a trap" was the wrong call (fixed same day)

Reported within the hour: *"what is highlight reconstruction and reconstruction
radius for, I dont see any effect from those sliders at all?"* — and correct.
The stage produced values above white and the section's final clamp took them
straight back off, so on a real photograph the slider moved **0.0004 of mean
level**: nothing. It only came alive paired with `grade_highlights` negative
(0.395 of max change against 0.05), and I had shipped that as a documented
"usability trap" rather than a bug. **A control that does nothing on its own is
broken however clearly the help text explains why.** Do not make this trade
again.

The fix is a roll of its own, and the reason it took some care is a hard
constraint worth stating because it rules out the tidier designs: **any curve
that brings over-range data into view must move in-gamut highlights too.** A
gamut map with its knee exactly at 1.0 has to jump — it would send `v = 1` to
`1 − d` — so a smooth one needs its knee below 1, and everything above that knee
moves. "Visible on its own" and "bit-exact no-op on an unblown frame" are
therefore in direct conflict for any *global* curve.

The way out is to make the roll **local**: gate it on reconstruction's own
weight field, dilated and feathered. Then the conflict dissolves instead of being
traded off — where nothing was repaired the gate is 0 and the frame is
**bit-exact** (1.19e-07), where something was repaired the roll engages and the
detail appears, and the gate is smooth so there is no contour outlining every
repaired region. Measured: red's span inside the blown region goes 0.0000 →
**0.0735** across the slider, monotonically, with nothing else touched.

**On the effect's honest size:** the worst per-pixel change on a real photograph
is 0.074, and that is the roll's own bound rather than anything tunable — the
clipped channel comes down from 1.0 to about 0.93 to make room for what was
recovered above it. Frame-mean change is 0.0018, because only 2% of that
photograph's pixels have a clipped channel at all. So this is a *local repair*
whose reach is set by how blown the source is: a flat patch becomes textured, and
a photograph with nothing blown in it is untouched by design. Do not chase a
bigger number here — an earlier version of the roll produced 0.567 and that
figure was the artifact below, not the repair.

Three things had to be got right, and two of them were wrong first:

* **Dilate wider than you feather.** Blurring the gate *dilutes* it, so a wider
  radius made the repair *fainter* — measured, the recovered span ran 0.069 at a
  16px radius down to 0.041 at 200px, i.e. reaching further to find the colour
  weakened the result, which is not what the control claims. Growing the mask by
  2× the feather before feathering keeps the gate saturated across everything
  repaired and only widens its outer ramp. Span is now near-flat across the
  radius (0.0736 / 0.0736 / 0.0735 / 0.0657 at 8/16/32/64px), and no over-range
  value survives the roll for the hard clamp to flatten (0.0000%).
  `_RECON_ROLL_GATE_FRAC` is 0.25 rather than the 0.5 first tried, which is better
  on every axis at once — see its own comment for the contour sweep that sets the
  floor at 0.05.
* **Roll per channel, not on the channel maximum.** This is the opposite of the
  tone stage's choice and the difference is not stylistic. Uniform scaling holds
  hue *exactly*, which is right in the tone stage because its input is already
  near the cube; here the input can be 2–4× over white in **one** channel, and
  holding the ratio exact then drags the other two down with it. Measured before
  the change: a bright warm highlight at `(1.000, 0.871, 0.634)` came out
  `(1.000, 0.305, 0.222)` — luma 0.882 → 0.447, **a dark saturated red where a
  bright highlight had been**, on 6% of the frame. The stage was creating the
  exact artifact it exists to remove. Per channel, an unclipped channel is
  **bit-identical** and only the rebuilt one comes down; the highlight stays
  bright and loses a little saturation, which is what film does as a dye layer
  saturates and what `highlight_desat` already models. *Fitting an out-of-gamut
  brightness into the cube costs either saturation or luminance; for a highlight,
  saturation is the right one to spend.*
* **`_RECON_ROLL_KNEE` is 0.80, not `_GRADE_TONE_KNEE`'s 0.5.** The recovered data
  lands just above white, so only the top fifth of the range needs to give way.
  A knee at 0.5 would work and would be the wrong control — that is a broad
  highlight roll, i.e. what `grade_highlights` is for, and duplicating it here
  would mean reconstruction quietly graded the picture as well as repairing it.

`_recon_estimate` and `_reconstruct_highlights` are split for this: the estimate's
accuracy is checked as an **equality** against a known unclipped scene, which is
only possible with the roll's compression out of the way. Neither half is useful
alone — the estimate is invisible, the roll is just a highlight dimmer — so they
stay behind one slider.

`grade_highlights` still stacks on top for a broader, stronger roll; the two
compose rather than one needing the other.

`pad_for` grows by **twice** the radius — three kernels in series inside the one
stage: the chromaticity estimate reads `radius`, then its weight field is dilated
and feathered to gate the roll. And it is **summed with clarity's rather than
maxed**, because reconstruction runs above clarity, so clarity's band is measured
on pixels reconstruction has already changed from up to its own reach away.
Pinned: 33px off, 153px at a 40px clarity radius, 213px at a 40px reconstruction
radius, 333px with both.

### It is the most expensive stage in the section by a long way, and the radius is the dial

Measured on a 2400×1600 proxy at ss=2, `Stock` plus the one change, best of 3 in
fresh processes:

| | time | pad_for |
|---|---|---|
| `Stock` as shipped | 0.57s | 108px |
| + reconstruction at the 32px default | **1.56s** (2.7×) | 252px |
| + reconstruction at 80px | **3.56s** (6.3×) | 468px |
| + reconstruction at 200px (the top) | **14.0s** (25×) | 1008px |
| + both tone controls at full travel | 0.55s (1.0×) | 108px |
| + halation recovery | 0.49s (1.0×) | 108px |

Most of that is not the blurs, it is `pad_for`: a 200px radius reserves 1008px of
overlap, so a tiled render spends most of its time on overlap it throws away.

**The range was deliberately not clamped**, per this file's own standing
instruction not to clamp blur radii for performance alone — the radius *is* the
speed control, which is the honest place for it. What was done instead is to put
the measured numbers in the parameter's own help text, so the top of the slider
cannot surprise anyone. The other two fixes are free.

**The available optimisation, not taken.** The chromaticity field is smooth by
construction, so both blurs could be computed on a downsampled copy and
interpolated back up — roughly 16× cheaper at 1/4 scale with no visible
difference. Not done because it is not bit-exact and would need the
tile-independence proof redone against the downsampled grid, and because nothing
is depending on the current speed yet. It is the first thing to reach for if this
stage ever becomes the reason a preset is slow.
