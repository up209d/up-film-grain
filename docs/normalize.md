# Normalize

Step -2, above Colour Grading and therefore above everything. One checkbox,
added 2026-08-16 on request: *"the lightness of input can vary, input can be
under exposure or over exposure, we need a mechanism to normalize the photo
before running into our pipeline, even auto white balance."*

Ships **off**, like every other section, so the pipeline is still a
pass-through with nothing selected.

## Why it is a section and not a control in Colour Grading

Both readings were put to the user and the answer was explicit: *"this checkbox
logic should be totally new logic on top of everything... Dont make it mess up
with any color grading logic."* So it is its own group, its own definitions
module, its own mixin, its own constants file and its own check module, and it
touches nothing in `stages/colour_grade.py`.

The distinction it draws is real rather than administrative. Grading decides
what the photograph should *look like*; this decides what it **is** before
anyone gets to make that decision. Everything below is calibrated around a
normally exposed frame — the characteristic curve and Contrast both pivot on
`_MID_GREY`, the Luminance Response band keys on developed density, the grain
envelope peaks in the mid-tones — so a frame two stops under does not merely
look dark, it puts the grain in the wrong tones and makes the preset read
wrong.

### Adding a section at the top broke both existing checkpoints, silently

`checkpoint.py` sliced `GROUPS[3:]` and `GROUPS[6:]` to decide which sections
sit below each boundary. Those are **positions**, so inserting `Normalize` at
index 0 shifted both by one: `GROUPS[3:]` stopped meaning "Grain Structure
down" and started meaning "Pre Sharpen down", which would have put Pre Sharpen
below a checkpoint saved *after* Pre Sharpen had run. That is character for
character the stale hit the comment on that boundary already records — *"it was
first written as `GROUPS[1:]`... `verify.py` caught it at 9.77e-01, which is
most of full scale"* — reintroduced by an edit nowhere near it.

Both are now `_from("Grain Structure")` and `_from("Halation")`, which take the
suffix **by name**. The lesson generalises past this change: a positional slice
of a list other people insert into is a tripwire, and the fix is not to be
careful, it is to make the position irrelevant.

## Its own checkpoint

`CHECKPOINTS` is `("Colour Grading", "Grain Structure", "Halation")` — the new
one named, like the others, for the section it sits *above*, so the frame it
holds is the one Normalize produced.

It is the shallowest boundary and it protects the least work: the stage is a
per-pixel curve, because the expensive part is measured once per upload and
cached on the `Upload`. What makes it worth a boundary is the other side of the
ledger — **nothing sits above it**, so its signature is a single key and it hits
on every edit anywhere else in the app. This was put to the user with that cost
spelled out (a frame per tier in an LRU shared with the valuable Halation
boundary) and they chose it.

## Two halves, and the split is invariant 1

`meter()` runs **once per upload**, on the whole frame, on the CPU, in numpy,
and returns six plain floats. `_normalize()` runs **per tile**, on the device,
and is pure per-pixel arithmetic on those six numbers.

A stage that measured the region it was handed would meter every tile against
its own crop, and a tiled export would come apart at the seams while every
preview looked perfect. `pad_for` can reserve a finite reach; a whole-image
statistic is an infinite one. Measuring the *frame* once and handing every tile
identical numbers is the same carve-out `_leak_sites` and the dust and hair
mark lists already use, and `verify.py` pins the seam at **0.00e+00** at tile
128 — exact equality rather than a kernel's 2e-3, because there is no kernel.

`verify.py` also pins the other half from the other direction: a *crop* must
meter differently. If it did not, the statistic would not be a property of the
frame at all and the per-tile version would have been safe — which it is not.

Two consequences fall out of caching on the `Upload` rather than recomputing:

* **Both tiers get the same numbers.** The proxy preview and the 1:1 export
  normalise identically, so "export what I am looking at" survives. Measured on
  a 3200x2000 source with a 0.75 proxy: both tiers land on mean **0.5304**.
* **The metering never re-runs on a slider drag.** Measured 232ms on a 1400x900
  frame, then 0.01ms from cache. It is also lazy — the control ships off, so a
  session that never ticks the box never pays for it at all.

The six floats ride in `p` beside `p["lut"]`, attached in `params_for` after
`sanitize`, and they are **plain floats deliberately**:
`checkpoint.upstream_signature` walks `sorted(p)` and keeps anything that is an
`int` or a `float`, so they land in the checkpoint key automatically. A tuple or
an array would be dropped by that filter in silence, and two photographs would
share a cached frame.

## What it does

Three corrections, in one linear-light round trip.

**Exposure** is the log average of the frame's luminance, solved onto
`_NORM_TARGET_LIN` (0.179 — `_MID_GREY` through the sRGB transfer, which is the
18% grey every meter is built around) and capped at ±2 stops. The log average
rather than the mean because the mean of a frame with a bright sky in it is the
sky. The cap because a night scene genuinely has a low log average, and a
normalizer without one would faithfully lift it to grey and destroy the
photograph.

**White balance** is a Minkowski p-norm at p=6 — grey-world at p=1, white-patch
at p=∞, and neither is usable alone. The gain vector is normalised against the
luma weights, so the correction is colour-only *by construction*: `verify.py`
asserts `|dot(LUMA, gain) - 1|` at **0.00e+00**, not within a tolerance. That is
the split tone's lesson applied ahead of time — a colour shift that is not
luma-neutral is two controls fighting.

It is damped toward identity when the frame's hues do not vary, which is the
guard against grey-world's famous failure: a sunset, blue hour or a close-up of
red fabric is legitimately one colour and is indistinguishable from a cast by
the channel means alone. On a synthetic sunset the damper backs the correction
off to **exactly** identity.

**Range compression** is what the user asked for after the first pass — *"I
wanna auto correction make it a bit of hdr, meaning still correct lightness but
should retain highlight and shadow information as much as possible, thinking
about the video LOG format."*

### Retaining the ends: a linear-light tone map, and the version that failed

**The first implementation destroyed highlights and shipped.** It applied the
exposure gain and then squashed whatever came out above a fixed knee at 0.82,
using `_tone_roll`. Measured on a real photograph needing +2 EV, the source band
0.70..1.00 -- 77 8-bit levels of highlight -- arrived as **3.2 levels**. Source
0.5 rendered at 230/255. The user reported it as "the highlight is totally blown
away", and that was exactly right.

Two things were wrong, and neither was the curve's shape:

* **A fixed knee cannot serve a variable lift.** At +2 EV every source value
  above 0.29 already lands above 0.82, so 70% of the tonal range had to be
  crammed into 18% of the output. The knee decides how much of the picture gets
  compressed, and the *lift* decides how much arrives above the knee -- pinning
  one while the other moves is the whole bug.
* **Sizing from the frame's true maximum let clipped pixels set the curve.**
  0.84% of that photograph was already blown -- flat white, no detail -- and
  those pixels forced near-maximal compression on the 99.16% that still had
  something in them. This was a deliberate choice, documented at the time as
  making the no-clip guarantee "unconditional". It optimised a provable property
  at the expense of the picture, which is the wrong trade every time.

The replacement is the **extended Reinhard tone map, in linear light**, with the
frame's own gained maximum `Lw` as its white point:

```
y = x * (1 + x / Lw**2) / (1 + x)
```

Three properties, none of them tuned:

* **`Lw` maps to exactly 1.0**, so the brightest pixel lands on white and
  nothing can exceed it. Applied to the channel maximum and scaled uniformly, so
  hue is held exactly -- a uniform scale cannot move a ratio.
* **At `Lw = 1` it is algebraically the identity** (`x(1+x)/(1+x)`). A frame
  that already fits is untouched, with no knee, no target and no special case.
  `verify.py` asserts this as an equality at 0.00e+00.
* It is strictly increasing, so ordering still cannot be lost.

It compresses gradually across the whole top end instead of gating at a knee,
and the mid-tones barely notice -- measured, the roll costs 7-12% at the log
average, which one gain correction absorbs.

Measured on the same photograph, old against new:

| source | old | new |
|---|---|---|
| 0.50 | 230.6 / 255 | 200.3 / 255 |
| 0.60 | 246.0 | 216.3 |
| 0.70 | 251.7 | 228.7 |
| 0.80 | 253.8 | 238.8 |
| 0.90 | 254.6 | 247.3 |

Highlight region: **153 of 256 levels kept, becomes 250 of 256.** Pixels at pure
white: **3.94% becomes 0.00%.** Across three real photographs the bright region
retains 98-100% of its distinct 8-bit levels.

### The check that passed while this was broken

The old module asserted that the transfer was **strictly increasing** over a
4096-step ramp, and measured 0 non-increasing steps. That was true, and
worthless: strictly increasing permits increasing by a millionth per step, which
is a flat white patch at 8-bit. It is the split tone's lesson one layer up --
a control whose failure mode is *being invisible* has to be measured in units a
human perceives.

The module now counts **distinct 8-bit levels surviving in the highlight band**,
on synthetic ramps and on real photographs, and keeps the monotonicity check
alongside it labelled as necessary-but-not-sufficient.

### The toe, and why the asymmetry flipped

The toe is unchanged in construction -- `_tone_roll` below `_NORM_TOE_KNEE`,
sized from the exposure correction, capped at `_NORM_TONE_MAX` -- but the
argument for the cap now points the other way.

It originally read: cap the toe because a lifted black point is the recognisable
half of the log look, and leave the shoulder uncapped because a knee means it can
only ever touch the top of the range. The premise was false for any large lift,
which is how the highlights were lost. The Reinhard roll needs no cap for a
different reason: it is gradual by construction, so there is no strength to run
away. The toe keeps its ceiling, and the user was explicit that highlights matter
more than shadows here, so the shadow end is the one that stays conservative.

## Highlight Priority (added 2026-08-16, on request)

`highlight_priority`, 0..1, default 0. The **only dialled control in the
section** -- everything else here is measured from the photograph -- and it
exists because what it settles is a genuine trade rather than a defect.

Lifting a dark frame's mid-tones leaves the bright end nowhere to go: it is
already near white in the file, and the ceiling does not move. So the tone map
has to compress it, and compressed highlights lose the fine separation that
reads as texture in a sky, a cloud or a lit face. **No curve escapes that.**
Something between the lifted mid-tones and the fixed ceiling has to give, so the
honest answer is not a cleverer curve but a control that lets the user say which
half they want.

The request was explicit: *"I want to retain highlight at all cost... if at 1 it
will need to restore all highlight detail from original source as much as
possible, at all cost (still retain the corrected rest areas of the photo)."*

It does exactly that, and literally: weight each pixel by how bright it was **in
the source**, and blend that far back toward the source's own value.

* At 0, the whole frame is corrected together and the highlights take whatever
  compression that costs.
* At 1, the bright areas come back at their original tonal spacing -- their
  values, their contrast, their hue -- while the mid-tones and shadows keep the
  correction.

Keyed on the *source* rather than on the corrected frame, because the question
is "was this a highlight in the photograph" and the corrected frame has already
moved.

Measured on a ramp driven through a real dark photograph's correction, the
0.70..1.00 band carries **21 of 77 8-bit levels at priority 0 and 79 at
priority 1**:

| source | priority 0 | priority 0.5 | priority 1 |
|---|---|---|---|
| 0.70 | 172 / 255 | 174 | 177 |
| 0.80 | 180 | 190 | 201 |
| 0.90 | 186 | 207 | 228 |
| 1.00 | 192 | 224 | 255 |

On the photographs themselves, priority 1 recovers the source's own highlight
local contrast almost exactly -- **0.3359 source, 0.3276 at priority 0, 0.3355
at priority 1** -- while the mid-tones keep **87%** of the lift.

### Why the blend band starts at 0.15 and not up in the highlights

The obvious choice is a narrow band high up: only touch the highlights. It is
the wrong one, and measurably so, because the blend has to fight the correction.
The corrected value is far *above* the original up there, so as the input rises
the weight shifts toward a lower number and the curve flattens instead of
climbing. Swept on a real photograph lifted two stops:

| band | minimum slope | mean slope 0.6-0.9 | mid-tone lift kept |
|---|---|---|---|
| tone map alone | +0.289 | 0.404 | 100% |
| 0.50-1.00 | **+0.029** | 0.286 | 100% |
| 0.30-1.00 | +0.197 | 0.514 | 99% |
| **0.15-1.00** | **+0.321** | 0.634 | 86% |
| 0.00-1.00 | +0.431 | 0.717 | 72% |

A 0.50 band produces a worse flat spot than the one the whole feature exists to
remove. 0.15 starts the weight moving before the correction has opened much of a
gap, so the two never pull against each other -- the minimum slope comes out
*better* than the tone map's own. Wider still keeps improving slope but gives
away the correction, which is the half the request said to protect.

### What it cannot do

Recover a highlight that was already blown in the file. Those pixels are a flat
plateau on arrival, blending toward the source returns that same plateau, and
they stay white. What it costs is *brightness* up there rather than detail:
highlights land nearer their original level than the corrected one, so a frame
with a lot of bright area reads as less lifted overall as the slider comes up.

## A known limit: the +/-2 EV cap under-corrects dark frames

Surfaced while measuring the fix. `_NORM_EV_MAX` is 2 stops, and genuinely dark
photographs want more: the sample frame's log-average key is 0.0209 against a
target of 0.179, which is **3.1 stops**, so the correction saturates at the cap
and the frame comes back darker than the target.

Left as it is deliberately, because the cap is doing the job it was written for
-- a night scene, a low-key portrait and a silhouette all have a genuinely low
key, and an uncapped normalizer would lift them to mid-grey and destroy the
photograph. Now that the roll makes large lifts safe at the top end, raising it
is a defensible *taste* decision rather than a correctness one, so it is the
user's call rather than a silent change.

## What it deliberately does not do

**It cannot recover detail that was never in the file.** A highlight blown out
at capture arrives as a flat patch and stays one. That is Highlight
Reconstruction's job.

**The two interact badly, and the help text says so.** Reconstruction finds
blown areas by looking for channels pinned between `_RECON_LO` and `_RECON_HI`;
Normalize runs first and, on an over-exposed frame, moves them off the ceiling,
so reconstruction stops firing. This was raised with the user before building —
the alternative was placing Normalize *inside* `_grade` below reconstruction —
and the separate-section architecture was chosen with the trade named. Both ship
off, so it only bites when someone turns on both.

**It does not fight a good photograph.** The correction is sized to what the
frame needs, so an already-neutral, correctly-exposed one meters at ev
**+0.000** with gains within **0.0000** of 1. Without that property there would
be a penalty for leaving the box ticked, and there is not.

## Two bugs worth keeping

Both are in `docs/lessons.md` and both were caught by measuring rather than by
reading:

* **The validity mask inverted the sign on exactly the input the control exists
  for.** Excluding clipped pixels is right for colour and wrong for level: a
  blown frame is mostly clipped, so the surviving samples are its *dark* ones,
  and a frame 1.4 stops over asked to be **+1.38 stops brighter**. Exposure now
  meters with only the dark end excluded.
* **The toe answered the wrong question.** Sized from the frame's own black
  level, a well-exposed picture with genuine deep shadows measured a toe of
  0.216 and had its blacks lifted to fix a problem it did not have. It is
  derived from the exposure correction instead — darkening by `ev` stops
  compresses the shadows by exactly `2**ev`, so the cost is a property of the
  correction, and brightening asks for none at all.
