# Colour Grading

## Colour Grading: a LUT is a *resource*, not a parameter (added 2026-08-04)

Requested: a section at the top of the pipeline that applies a 3D LUT from
`luts/` or from a file, with temperature, shadow, highlight and two-way clarity
sliders **before** the LUT, and cheap enough not to stress the main pipeline.
Step -1 in `render()`, above `pre_blur`. Everything ships at 0.

**The structural decision, and the one that shapes everything else: the LUT does
not live in `params.py`.** Every other control the engine takes is a float with a
range, so it can be sanitised, clamped, rescaled for a different image size and
stored in a preset file as a value. A LUT is identified by *name* and its content
is a table. So it travels beside the parameters — `body["lut"]` next to
`reference_mp`, and a `lut` sibling key in a preset file — and `main._params_for`
attaches the resolved object as `p["lut"]` after `sanitize` and `rescale`, both of
which only touch keys that are in `PARAMS` and so leave it alone.

The obvious alternative needed no new plumbing at all: a `choices` menu indexed
into the folder listing. It is wrong for exactly the reason `_SCATTER_STENCILS`
documents, and worse here — that list is fixed in code, whereas `luts/` is
user-mutable *by design*, the same way `presets/` is. A preset stores the index,
so dropping one more `.cube` in the folder silently renumbers it and changes the
look of every preset that named one. Names it is.

**`lut_amount` *is* a parameter, and it is in `NEUTRAL_ZERO`.** That pair is what
keeps the Original button honest. `params.is_neutral` decides whether
`render_image` short-circuits, and it works from the numbers alone — it cannot see
the LUT. So:

* Zeroing the mix switches the LUT off as completely as unselecting it would,
  which is why the *name* stays out of `NEUTRAL_ZERO`: same reasoning that keeps
  sizes, radii and seeds out of it, so the section remembers what it had.
* **A mix above zero with no resolvable LUT would be a silent bug**, not a no-op:
  `is_neutral` would be false, the render would run, and at supersample 2 the
  bicubic-up/box-down round trip comes back a measured 1.0e-01 softer than the
  source. `_params_for` therefore forces `lut_amount = 0` whenever `lut.get`
  returns nothing, so the gate in the engine and `is_neutral` can never disagree.
  `verify.py` pins both halves.

An unresolvable name is deliberately **not** an error — a preset can name a
`.cube` that has since been renamed, or an upload from a previous run (those live
in process memory and do not survive a restart). The picker keeps the name as a
"— missing" entry with a hint rather than resetting itself to None, because
silently showing None makes it look like the preset never had a LUT.

### The four things that had to be right about the lookup

* **`align_corners=True`.** A LUT's first and last samples *are* input 0 and
  input 1, not the centres of edge cells. The default reads the whole table half
  a cell off — a small, uniform, entirely wrong shift that looks like the LUT
  being slightly wrong rather than like a bug.
* **The axis order.** `.cube` says red varies fastest, so a C-order reshape gives
  `table[b][g][r]`; permuted to `[c][b][g][r]` that puts red on `grid_sample`'s
  `W`, green on `H`, blue on `D`, which is why the sampling grid is just the
  image's own channels in order. Get this backwards and every *symmetric* LUT
  still looks fine while every real one is channel-swapped.
* **Both of the above are pinned by construction rather than by eyeball.**
  `verify.py` builds two exactly-linear 8-cubes — an identity and one that
  rotates the channels — and trilinear interpolation of a linear function is
  exact, so the check is an *equality* (2.4e-07) rather than a judgement. The
  rotation catches a transposed axis; the identity catches the alignment.
* **`F.grid_sample` in 3D works on MPS**, checked before building on it. One call,
  trilinear, so a 35-cube and a 64-cube cost the same and neither shows up against
  the stages below. The alternative — gathering eight corners by flat index —
  needs int64 index tensors MPS handles badly and eight full-frame gathers of
  working memory.

### Why each adjustment is where it is

* **Temperature in linear light.** A white balance is a change of *illuminant*, so
  it multiplies light, and gamma-encoded values are not light — done encoded, the
  same gain moves the shadows much further than the highlights, which is what
  makes a naive temperature slider read as a tint laid over the picture. Same
  argument as `pre_blur`'s, and gated the same way so the transfer round trip
  costs nothing at 0. The gain vector is normalised by its own luma, so the
  control is colour-only: measured, luminance holds to within 1% across the slider.
* **Shadows and highlights display-referred, clip-free, and — since 2026-08-05 —
  monotone.** Both halves of this were rewritten; the section below
  ("Shadows and Highlights were a brightness shift, not a recovery") is the
  authority and this bullet is only the pointer. The short version: the
  recovering directions are asymptotic rolls keyed on the channel maximum and
  applied as a uniform scale, so they are strictly monotone, gamut-safe by the
  curve's own bound, and hue-exact rather than hue-approximate; the expanding
  directions keep the original share-of-headroom form and `_GRADE_TONE_MAX`.
  `verify.py` pins the worst excursion outside 0..1 at 0.00e+00 *and* the worst
  transfer slope over twelve settings at +0.369. The two halves have disjoint
  supports about the knee, so they no longer need a shared luma to stay
  independent — the far end is bit-exactly untouched.
* **`_GRADE_TONE_MAX` is 0.35, not 1.0, and that is not a taste tweak.** At 1.0 a
  setting of +1 takes a black pixel to *pure white*, so the whole top of the
  slider is unusable and the useful range is squeezed into its first tenth.
  Measured on a real photograph (mean luma 0.21), Shadows at only **+0.5 took the
  frame's mean from 0.19 to 0.53** — that is a different exposure, not a shadow
  lift. Caught by rendering the actual photo through the actual API, not by
  reading the code. Same lesson as `_JITTER_MAX` from the other direction: the
  whole range has to be usable. Since the tone rewrite it governs the *expanding*
  half of each control only — a share-of-headroom cap is exactly what made the
  recovering half non-monotonic, so that half has no cap and its endpoint comes
  from the curve's own asymptote.
* **Clarity is asymmetric on purpose.** Positive gets `_GRADE_CLARITY_GAIN` (1.6);
  negative is pinned at exactly 1.0, because at gain 1 a setting of −1 subtracts
  precisely the band it measured — the local contrast is *gone*. Past that it does
  not keep flattening, it **inverts**: dark halos on the light side of every edge,
  an artifact rather than a look. `verify.py` measures the band's correlation with
  the source at −1 and fails on a negative number. Measured ladder: −1 → 5% of the
  band, −0.5 → 52%, +0.5 → 177%, +1 → 255%.
* **Clarity runs on luminance, which is both cheaper and better.** The signed
  detail goes to all three channels equally, so the channel *differences* — which
  is what hue is — come through untouched (pinned at 2.4e-07), a saturated area
  cannot be pushed out of gamut by a structure control, and it is one blur instead
  of three.

### Cost, and the one term in `pad_for`

Four of the five stages are per-pixel with no kernel and no neighbourhood, so
they reserve **nothing**. Clarity's high-pass is the only kernel in the section
and it is a real reach even though the stage runs first: a tile that cannot see
far enough measures a different band at its own edge, and that difference then
propagates through everything below it. `verify.py` pins both halves — that
`pad_for` grows by 3× the clarity radius, and that it is *unchanged* with
temperature, tone and a LUT all on.

(Highlight reconstruction, 2026-08-05, is the section's second kernel and the
only stage in it that was accepted as expensive on purpose — two three-channel
blurs. Both terms are summed in `pad_for`, not maxed, because reconstruction runs
above clarity and clarity's band is therefore measured on pixels reconstruction
has already changed from up to its own radius away. Measured: 33px off, 153px at
either kernel alone at a 40px radius, 273px with both.)

Measured on a 6MP render at 2×, best of 3 in fresh processes (MPS run-to-run
variance here is ±1s on larger frames, so single-shot numbers are worthless):

| | time | pad_for |
|---|---|---|
| section off | 0.67s | 108px |
| temperature / tone / a LUT at mix 1 | 0.67–0.73s, inside variance | 108px |
| clarity at the default 14px | 0.75s | 150px |
| clarity at 40px | 0.88s | 228px |
| all of it | 0.82s | 150px |

### Two things outside the section that had to change with it

* **`build.sh` copies `luts/`.** It already had this exact bug documented for
  `presets/` — a distribution without the folder has an empty LUT menu and a
  preset that names a `.cube` quietly grades nothing.
* **Editing a control in a muted section now switches that section on.** Found in
  a real browser, not by inspection: on a fresh load *every* section is muted (see
  the muted-on-boot section), so picking a LUT left the section's switch reading
  "off" while the LUT rendered — and a mute/un-mute round trip then reverted the
  mix to the snapshot `toggleGroup` took at mute time, measured going straight
  back to 0. `keptFor`/`liveFor` in `App.tsx` restore the section's kept values
  and lay the edit on top, which is exactly what clicking its own ● does. This is
  general, not LUT-specific: it was latent for every slider in the app the moment
  boot started muting everything, and the new section is simply where it is hit
  first. The pair is split into a pure half and a side-effecting half because a
  `setMuted` call inside a `setValues` updater would run twice under StrictMode.

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

## Shadows and Highlights were a brightness shift, not a recovery (fixed 2026-08-05)

Reported, and correct: "you just low down brightness in already lost highlight,
raise brightness on lost shadow, they are already lost so there is no recovery at
all, it is just pure brightness shift." Three separate defects were behind it,
and all three are now fixed. The stated priority for the fix was **highlight
detail above all else** — "film tends to have very detailed highlight, so I don't
want to lose highlight at all" — with performance explicitly accepted as a cost.

### Defect 1: the recovering directions were non-monotonic — they *inverted* tone

The construction was `x + a·m(x)·(1-x)` (or `·x` going down), with `m` a quintic
in the pixel's own luma. Read as a transfer function, the `m'` term competes with
the identity, and in the two *recovering* directions — **Shadows positive and
Highlights negative, exactly the two anyone reaches for** — it wins:

| setting | min transfer slope | non-monotone span |
|---|---|---|
| Highlights −1.0 | **−0.211** | 15.9% of the range |
| Highlights −0.7 | +0.152 | 0% |
| Shadows +1.0 | **−0.211** | 15.9% |
| Shadows +0.7 | +0.152 | 0% |

A negative slope is not compression, it is a **fold**: tonal order reverses, and
around the zero crossing the region goes genuinely flat. So the control turned a
highlight into the textureless patch it was supposed to rescue — precisely what
was reported. Even at −0.7, where it is technically monotone, slope 0.152 means
85% of the local contrast is gone.

The two *expanding* directions (Highlights positive, Shadows negative) were
monotone all along. That the broken pair is exactly the pair named in the report
is the confirmation that this was the bug and not a taste disagreement.

### Defect 2: everything above the tone stage clipped, so there was nothing left to recover

`_grade` clamped to 0..1 after **every** stage. White balance and Exposure sit
above Shadows/Highlights, so a stop of exposure was rounded off to white *before*
the recovery control ever saw it. "Recover the highlight" could only ever mean
"recover what is left of the highlight after we threw it away."

The section now clamps **once**, after the tone stage. Headroom from
reconstruction, white balance and exposure all survive into the curve that is
meant to roll it back in. With both tone controls at 0 this is bit-identical to
before — a monotone brightening followed by a clamp is the same picture whichever
end the clamp sits at — so nothing existing moved.

### Defect 3: an 8-bit file's clipped channels were never reconstructed

See the next section. "The source surely has those details" is *true* in a
specific and exploitable way, and nothing was exploiting it.

### The new construction: `_tone_roll`, on the channel maximum, uniformly scaled

Two decisions on top of the curve itself:

**The curve is monotone by algebra, not by luck.** Recovering, it is a convex
blend of the identity and the exponential shoulder `1 - exp(-t)`, so at full
travel the rail becomes an **asymptote**: the whole of `[knee, ∞)` folds into
`[knee, rail)` with ordering intact, and slope is bounded below by `exp(-t) > 0`.
Expanding, it keeps the old share-of-headroom form (which was already monotone
and clip-free). The asymmetry is the same shape of decision Clarity's is: pushing
a tone at a rail and pulling one off it are different operations.

**It keys on the channel maximum and scales all three channels together.** The
*value*, not the luma, is the right question for a control about clipping — a
saturated red at (1, 0, 0) has a channel hard against the ceiling while its luma
is 0.21, and the old luma key called that a shadow. And a uniform scale cannot
move a ratio, so hue and HSV saturation are held **exactly** (measured 2.7e-07)
rather than approximately, while gamut safety becomes structural: the curve's
output is bounded by the rail, so every channel, being at or below the maximum,
is too.

The two halves also stop needing a shared reference. Highlights only touches
`v > knee` and cannot push a value below it; Shadows only touches `v < knee` and
cannot push one above. Disjoint supports means independence by construction —
stronger than the "one luma measured before either runs" bookkeeping it replaces,
and impossible for a later edit to get wrong. `verify.py` pins the far end as
**bit-exact** now rather than merely close.

Measured on a real photograph (the repo's own 16×9 test frame, 2.1% of red at the
ceiling) — detail retained in the region each control is meant to rescue:

| setting | before | after |
|---|---|---|
| Highlights −0.5 | 54.4% | **78.6%** |
| Highlights −1.0 | 19.2% | **57.3%** |
| Exposure +1, Highlights −1 | 27.9% | **48.4%** |
| Shadows +0.5 | 66.4% | **97.9%** |
| Shadows +1.0 | 33.2% | **97.5%** |

One measurement in there is worth reading twice. Inside the frame's clipped-red
region, red's variation under the old Highlights −0.5 measured **6.1e-02** —
*fifty times the file's own* 1.2e-03. That was not detail; it was the fold
manufacturing structure that is not in the photograph. The new curve reads
8.2e-04 there (honest: the region is flat in the file) and 1.8e-02 with
reconstruction on (real, recovered structure).

**The controls are deliberately gentler in level than they were.** Shadows +1
used to take the frame's mean luma 0.209 → 0.441; it now takes it to 0.271. That
is the trade being made on purpose: the old reach *was* the wash-out. Exposure is
the control for level. A sweep of a steeper shoulder gain was run and declined —
gain 2.0 reaches mean 0.308 but drops the worst-case slope from 0.184 to 0.068,
i.e. it crushes the deepest blacks to buy reach. `_tone_roll` has no gain
constant for that reason; the ±1 endpoint is fixed by the curve's own shape
rather than chosen.

The check that matters is on the **slope of the transfer**, over 12 settings, not
on the mean level. A mean-only test passes happily on a curve that has folded
over — which is how this shipped in the first place.

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

