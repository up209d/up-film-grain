<!-- part of docs/colour-grading.md -->

## Colour Grading grew six more sliders, all still before the LUT (added 2026-08-04)

Requested: Tint after Temperature, Exposure above Shadows, Contrast, Black
Point above Clarity, and Vibrance/Saturation after Clarity Radius — six new
`grade_*` parameters, all "quick" per-pixel adjustments with no kernel, all
still ahead of the LUT. Panel order and pipeline order both now read:
Temperature, Tint, Exposure, Shadows, Highlights, Contrast, Black Point,
Clarity, Clarity Radius, Vibrance, Saturation, LUT Mix — the same "panel order
matches pipeline order" rule the section has followed since it was added.
Every one ships at 0.

**None of the six duplicate an existing parameter, even where the name does.**
Tone Response already has a `contrast` and a `vibrance` — deferred, ships at 0,
must not be touched per this file's own standing instruction. The new
`grade_contrast` and `grade_vibrance` are separate keys with separate formulas,
because the two sections cannot share a slider: grading the picture and
grading the negative are different jobs done at different points in the
pipeline, and folding them together would mean the deferred section could never
be switched on later without re-touching a grade that was already finished.
Same reasoning as `grade_exposure` existing alongside Tone Response's
`brightness` — one physically identical formula (a stops multiply in linear
light), two independent parameters, because one section is deferred and the
other is not.

### Tint is Temperature's other axis, and needed its own gain constant

Temperature's gain (`_GRADE_TEMP_GAIN`, 0.40) applies to red and blue in
opposite directions with green untouched, then normalises the whole vector
against the luma weights so warming a frame does not also expose it. Tint is
the same construction on the other white-balance axis — green against
magenta — and the first version reused 0.40 outright on the reasoning that a
change of illuminant is one physical adjustment resolved along two axes, so
there was no reason for one axis to reach further than the other.

That reasoning does not survive contact with the luma weights. Green carries
0.7152 of them against red's 0.2126 and blue's 0.0722, so pushing green by the
same amount temperature pushes red/blue costs far more level: measured on the
same asymmetric plate the "temperature holds the level" check uses, 0.40 on
tint drifted luma **−2.3%** against temperature's own **−1.8%** at its 0.40 —
outside the 2% tolerance the existing check already holds temperature to.
Tried a doubled-green formula first (`gain = [1+g+t, 1-2t, 1-g+t]`, balancing
the raw pre-normalisation sum to zero) and it was worse, not better — doubling
green's coefficient means the luma-weighted normaliser has to divide by a much
smaller number, which amplifies red and blue far more on a non-grey pixel than
the single-coefficient form does. Measured at the same 0.40, the doubled
version drifted **−9.9%**. `_GRADE_TINT_GAIN` is **0.30** with the
single-coefficient formula (`gain = [1+g, 1-g, 1+g]`), which brings the worst
case to −1.4% — inside the same envelope Temperature's own check uses,
verified on the same plate rather than assumed from the arithmetic.

Applied in the *same* linear-light round trip as Temperature rather than a
second one: both axes are one physical operation (a change of illuminant), so
`abs(temp) > 0.001 or abs(tint) > 0.001` gates a single
`_srgb_to_linear`/`_linear_to_srgb` pair with both gain vectors summed before
the one normalisation, instead of paying the transfer cost twice for what the
white balance actually is.

### Exposure runs before every luma-keyed mask in the section

Same formula as Tone Response's `brightness` — `2.0 ** ev` multiplied in
linear light, so the sRGB encoding rolls the highlights off on the way back
instead of a display-referred stretch clipping them flat — but a separate
parameter and a separate stage, positioned directly ahead of Shadows. Every
mask after it in `_grade` (Shadows, Highlights, Clarity, Vibrance, Saturation)
measures `_luma(img)` on whatever `img` currently is, so raising exposure first
means all of them read the frame at the light level actually being graded.
Putting it anywhere later would mean, for instance, Shadows deciding what
counts as shadow from a frame that is about to get brighter or darker
underneath it. `verify.py` checks this one for an *exact* match against a
direct linear 2× multiply (2.4e-07) rather than a mean brightness change, since
a mean-only check cannot tell "did the right operation run" from "did some
operation that also brightens the image run".

### Contrast and Black Point are the clip-allowed pair Shadows/Highlights are not

Both new, both deliberately *not* clip-free — that is the division of labour
Shadows/Highlights already established: those two exist so a grade can move
tonally without any risk of crossing 0 or 1, and Contrast/Black Point exist for
when clipping is exactly what is wanted.

* **Contrast** pivots about the same `_MID_GREY` (0.46) the deferred film
  characteristic curve uses, but two-way and applied directly rather than
  through a toe and shoulder — `x' = MID_GREY + (x - MID_GREY) * gain`, gain
  `= 1 + _GRADE_CONTRAST_GAIN * contrast`, floored at 0. The floor is what
  stops a strongly negative setting from crossing zero and inverting the
  picture through grey; at −1 the gain is exactly 0.1, verified by pinning a
  flat mid-grey field at bit-exact invariance (2.98e-08) and a ramp's standard
  deviation at exactly 0.1× (measured 0.100×, not merely "close"). Deliberately
  smaller than the film curve's own 1.1 (`_GRADE_CONTRAST_GAIN` is 0.9,
  gain range 0.1–1.9): this control has no shoulder to catch what it steepens,
  so +1 already clips a ramp's extremes rather than rolling them off, and a
  gentler reach keeps that from happening immediately at the top of the
  slider.
* **Black Point** is the blunt Levels-style remap most photo tools mean by the
  name: `x' = clamp((x - bp) / (1 - bp), 0, 1)`. Every value at or below `bp`
  is driven to exactly 0 and 1 stays exactly at 1 — genuinely crushing shadow
  detail rather than easing it, which is the point of a black point control and
  the reason it is one-directional (range 0.0–0.3, not the two-way ±1 every
  other new slider here gets): there is nothing below 0 to lift from, and a
  floor lift is what Shadows or the deferred Base Fog are for. `verify.py`
  checks the clip is exact (max at or below the chosen level is 0.00e+00) and
  that white and monotonicity both hold above it.

### Vibrance and Saturation: the same weighting Tone Response has, and the blunt version it deliberately does not

`grade_vibrance` is bit-for-bit the same construction as Tone Response's own
`vibrance` — saturation measured as chroma-over-value (the HSV definition),
gain `= 1 + vib * (1 - sat)` clamped at zero — moved earlier in the pipeline
and given its own key for the reason above (the two sections must stay
independent). `verify.py` runs the identical saturation-ladder check against
the new key and gets the identical property: gain falls monotonically as
starting saturation rises (59% → 38% → 22% → 11% → 2% at vibrance 0.8 across
five saturation levels — the same ladder Tone Response's own check measures).

`grade_saturation` is new in a stronger sense: there was no flat, unweighted
saturation control anywhere in the engine before this (confirmed by grep — the
only chroma-about-luma-axis scale in the codebase was vibrance's own weighted
one). `gain = max(0, 1 + sat)`, applied as `lum + (img - lum) * gain` — every
pixel gains or loses the same proportion regardless of how saturated it
already is, which is the classic blunt control and will push a vivid area out
of gamut before a muted one catches up, in contrast to Vibrance immediately
above it. Checked as an *exact* claim rather than a ratio: because every
channel's offset from luma scales by precisely `gain`, chroma (`max − min`)
must scale by exactly 2× at `sat = 1` — measured 1.79e-07 off — and `sat = -1`
must be exactly monochrome — measured 0.00e+00. A saturation-ratio check
cannot make this claim as tightly, because the ratio's own denominator (the
max channel) shifts at the same time.

### Cost: still nothing in `pad_for`

All six are per-pixel with no kernel and no neighbourhood, same as every stage
in this section but Clarity — `pad_for`'s `grade_clarity_radius` term is
unchanged, and `verify.py` pins it explicitly with all six new sliders on at
once alongside Clarity. `_grade`'s own docstring is updated to stop citing
"four of the five stages" now that there are eleven.

(As of 2026-08-05 there are twelve stages and **two** kernels: highlight
reconstruction is the other one, and the two are summed in `pad_for` rather than
maxed because they run in series. Everything else in the section is still
per-pixel and still reserves nothing.)
