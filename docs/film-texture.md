# Film texture

## Film texture sits last, and is masked by nothing (added 2026-07-31)

Step 15, dead last, after sharpening: dust, scratches, hair, light leaks.
Everything above it models what the *emulsion* does; this models what happened
to the strip of film afterwards. It is weighted by none of the image masks — a
scratch does not care what is underneath it — and every parameter ships at 0.

**What must never happen is a mark list derived from the region being
rendered.** N specks per tile, or positions drawn against the tile's own area,
and an export grows seams or draws different debris in every tile. That is the
invariant. It is *not* the same as "never build a list", which is what this file
used to say, and the distinction cost a rewrite to get right — see below.

How each shape is made, and the measured result:

| mark | how | count | geometry |
|---|---|---|---|
| dust | drawn per speck: an ellipse with three angular harmonics, from a frame-anchored list | **exact** | round, mean axis ratio 1.47 |
| scratches | noise with cells ~2px wide and ~900px tall — the anisotropy *is* the scratch | ±50% | 76:1, 1.2px wide |
| hair | drawn per filament: a tapered curve with a quadratic sag and two wobbles | **exact** | 2.9px wide |
| light leak | oriented beams anchored on the perimeter, added in **linear** light | exact | 1.3:1 along/deep |

Scratches still go through `_weather()`, which is what stops a *field* of marks
looking generated. A thresholded field gives every mark an identical crisp edge
and identical opacity; real debris sits at different depths, so some is in focus
and some is not, and none of it is equally dark. `_weather` blends each mark
toward a blurred copy and scales its density, both driven by fields addressed at
*mark* scale — a whole scratch shares its blur and its density rather than
fading in and out down its own length. Measured at full softening: mean edge
slope down 26%, while the crisp-to-soft ratio *widens* (13.8x to 18.2x) and
per-mark brightness spread runs 69% of the mean. Both halves are asserted: a
uniform blur would pass a mean test and be exactly the artificial result this
exists to avoid. Dust and hair carry their own per-mark draws now and no longer
need it.

## Dust and hair are drawn one mark at a time (rewritten 2026-08-06)

Three complaints, one cause. *"Too many dark dots, I want more light dots."*
*"Dots need to be in round form, some dot I found is not round."* *"I can see
hair count when I set to 1, I see more than 1 hairs."* All three were reported
against a construction that could not have satisfied any of them, because a
threshold on a noise field has neither a count nor a shape:

* **A threshold selects area, not marks.** How many countable blobs that area
  breaks into was a *fitted constant* — `_BLOB_CELLS_DUST = 14.0`,
  `_BLOB_CELLS_HAIR = 0.5` — good to about a factor of 1.5. Ask for 20 specks
  and you got somewhere between 13 and 30.
* **A level set is not one curve.** A hair was `|n − 0.5| < eps` of a smooth
  field, gated by a second field. Inside any one gate blob the field crosses 0.5
  along however many separate arcs it happens to, so one unit of "hair" drew
  one filament, or three, or none. `_BLOB_CELLS_HAIR = 0.5` was a fitted apology
  for exactly that, and it is why count 1 drew more than one hair.
* **The outline is whatever the field did.** Lumpy, frequently merged with a
  neighbour, occasionally a long tear that reads as a scratch. You cannot get a
  small round thing out of a level set of noise except by accident.

So dust and hair are now **frame-anchored lists of drawn marks**, and the
counts are exact. Measured on a 900x1400 plate, components counted:

| asked | 1 | 2 | 3 | 5 | 20 | 120 | 400 |
|---|---|---|---|---|---|---|---|
| dust drawn | 1 | 2 | 3 | 5 | 20 | 120 | 391 |
| hair drawn | 1 | 2 | 3 | 5 | — | — | — |

1–5 is exact across six seeds and `verify.py` asserts it. The 2% shortfall at
400 is marks genuinely overlapping — two specks on top of each other are one
blob and there is nothing to fix about that.

### This does not break tile independence, and here is exactly why

`_leak_sites` already set the precedent and this file already stated the rule:
what breaks the invariant is a list derived from *the region being rendered*.
A list derived from the count, the seed and the **frame** is a different thing —
every tile builds the identical list, and so does the proxy, and so does the
export.

Two mechanisms carry it:

* **Positions are fractions of `full_hw`**, which `render()` already receives
  for the light leaks and which `render_view` fills from the whole source rather
  than the read window. So a speck lands in the same place at any working scale
  and under any tiling.
* **`_mark_window` clips the mark's own footprint, not the tile's.** A speck
  straddling a tile boundary is drawn by both tiles from the same absolute
  geometry. The arithmetic is deliberate: the pixel offset is `(i + y0) − cy`
  and *not* `i + (y0 − cy)`, because folding the origin into the centre first
  hands two tilings two different float roundings of the same offset — a
  sub-pixel disagreement along a seam. Formed this way the absolute coordinate
  is an exact integer in both and only one rounding ever happens.

The consequence is that **`pad_for` reserves nothing at all for dust or hair**.
Both used to blur their mark fields and had to be counted there; neither has a
kernel any more. `verify.py` renders 300 specks and 20 hairs tiled at 128px
against a single pass and gets **0.00e+00**.

### The shape of a speck: round, and not a circle

Both halves were asked for — "a shape form of imperfect circle or imperfect
ellipse, which is correct". So a speck is an ellipse with a random eccentricity
(`_DUST_ECCENT`, up to 0.35) at a random angle, its radius perturbed by the
**3rd, 4th and 5th angular harmonics** with random phases.

Third and up, deliberately: the 2nd harmonic *is* an elongation, so it would
only fight the eccentricity draw. Their amplitudes sum to 0.22, and that sum is
the number that matters — the radius is `1 + Σ aₖ·cos(kφ + pₖ)`, so a sum at or
above 1 folds the outline through its own centre and draws a shape with a bite
out of it.

Measured with the isoperimetric quotient `4πA/P²`, against **a disc rendered
through the same rasteriser** rather than against 1.0 (a rasterised outline
over-counts its own perimeter by a factor that depends on the radius): specks
score 86% of a disc's, i.e. round. And the other half, from the second moments:
mean axis ratio **1.47** where the same disc renders 1.04 — so they are round
shapes and not circles.

### The dark/light balance

`dust_balance` runs −1 (every speck an opaque dark mote) through 0 (an even mix)
to +1 (every speck a bright pinhole). It replaces a hard-coded two-thirds dark,
which is what was reported as too dark.

The split is **a prefix of the list, not a per-speck coin flip**, and that buys
two things: it is exact (100 specks at balance 0 render 50 dark and 50 light,
asserted), and moving the slider converts specks *in place* — position is drawn
per index and the balance never touches it, so the frame does not reshuffle
under you while you hunt for the ratio you want.

### Two sampling traps, both of which drew visible artifacts

* **A filament narrower than a pixel renders as a dashed line.** It only
  registers where its centre passes near a pixel centre. A hair tapers to a
  point, so its tip did exactly this: measured, one hair came out as a 394-pixel
  filament plus a detached one-pixel speck strung out past its end — which is
  the "I see more than one hair" complaint reappearing in a new form. The fix is
  `_MARK_MIN_PX`: below the floor a mark is drawn *at* the floor and faded by
  what is missing, which is what area-averaging would have done anyway. A
  filament needs a full pixel of width where a speck needs half a pixel of
  radius — a disc always has a pixel centre within reach of its own soft edge, a
  line can thread between them for its whole length.
* **A wobble steep enough to double back breaks the distance formula.** The
  renderer measures a pixel's distance from the filament as the vertical gap
  over `sqrt(1 + slope²)`, which is the perpendicular distance only while the
  curve is locally straight. At high amplitude *and* high frequency it is not,
  a point genuinely on the curve gets scored against the wrong part of it, and
  the hair comes out in pieces — a fifth of them did. So each wobble's amplitude
  is capped by its **slope** rather than by its size: `_HAIR_SLOPE / 2πf`, which
  holds the steepest slope constant however fast it ripples. It is the physical
  answer too, since a fibre does not zigzag tightly and widely at once.

### Marks are placed on a low-discrepancy sequence

Independent uniform draws clump, and at small counts they clump *visibly*:
measured on the hair generator, four of the first five marks landed in the top
fifth of the frame. That is not a bug in the hash — over 400 marks the draws are
uniform to 1% and uncorrelated to 0.02 — it is just what five uniform points
look like. "I asked for five hairs and they are all in one corner" is a
complaint whether or not the statistics are innocent.

`_mark_spread` steps along the **R2 sequence** (the reciprocal powers of the
plastic number), which is `_leak_sites`' golden-ratio trick in two dimensions,
plus a small jitter. Any prefix of R2 is well spread, which is what keeps mark 6
from moving marks 1–5 — the same add-don't-reroll property `_mark_rng` exists
for. The jitter is fixed in frame units rather than scaled to the count: at 400
specks the R2 spacing (0.05) is under the jitter (0.06) so placement goes
locally random and dust clumps the way dust does, and at a count of three the
spacing dwarfs it and the sequence wins.

### A leak is a beam, not a border wash (rewritten 2026-08-02)

The user reported the leak shape as "very off" and they were right. Everything
above the shape — pixel sizes, the feather-to-exponent mapping, the centre-fog
cap — survived the rewrite unchanged; the *shape itself* was replaced.

**What it used to be.** `edge_d = min(distance to each border)`, raised to a
falloff exponent, multiplied by a slow noise field gated along the perimeter.
Read out loud that is: a soft inward wash, present on all four borders at once,
with a boundary made entirely of noise. Rendered, it is a chewed-up vignette.
It had none of the three things a light leak actually has:

* **No direction.** The only spatial variable was depth from the nearest
  border, which is isotropic along it. A leak could not lean, could not cross
  the frame, could not point anywhere.
* **No definite edge.** Every boundary was a `smoothstep` on value noise, so
  everything faded into everything. The reference photographs are unanimous
  that a leak has "a definite edge that is limiting its reach" — that is the
  shadow of whatever the light got past, and it is most of what separates a
  leak from haze.
* **No count.** The gate was a pair of noise quantiles, so asking for two
  leaks still washed most of the border; the control really only moved how
  ragged the wash was.

It also had a bug that shows in any render: `torch.where(near_horizontal, ...)`
picked between a horizontal and a vertical wash field on the frame's diagonals,
which draws a **hard 45° crease out of every corner**.

**What it is now.** `_leak_sites` returns one record per leak — position on the
perimeter, reach, along-border length, lean, fan, edge hardness, strength, hue
— and each is drawn as a beam:

```
u  = perpendicular depth from that leak's own border   (+ domain warp)
v  = (s - s0) - shear * u                              (+ domain warp)
along  = clamp(1 - u/reach, 0) ** expo                 <- the feather mapping
across = band(|v| / halfwidth(u)), one edge hard       <- the definite edge
```

Four things about that are deliberate:

* **The obliquity is a shear on `v`, not a rotation of the frame.** A rotated
  beam's "length" is measured along its own axis, and `leak_size` would stop
  being the depth it promises. Sheared, a leak can lean 60° across the picture
  while `reach` stays exactly the perpendicular penetration — so every existing
  size and feather measurement still means what it meant.
* **The along-border length is a fraction of the *border*, not a multiple of
  the reach** (floored at `0.55 * reach`, since light through a slot cannot be
  much narrower than it is deep). A seal fails along a seam: the leak runs a
  long way sideways and comes in a modest depth. Sizing the length off the
  depth instead — which is what I tried first — makes every leak roughly as
  long as it is deep, i.e. a blob. Measured, median length/depth 1.30.
* **One edge is much harder than the other**, picked per leak. Both soft is
  haze; both hard is a painted shape. Measured, median steepest-edge ratio 2.55
  between a leak's two sides, and `verify.py` pins it.
* **Noise perturbs the shape instead of being the shape.** Two octaves of
  domain warp on `u` and `v`. That inversion is the whole difference between an
  organic outline and fog.

**`_LEAK_CORNER_BIAS` must stay under 1/2π = 0.159.** The corner pull is
`t - bias * sin(2πt)` applied inside a border segment, and above that threshold
the map stops being monotonic and starts *folding*: at my first value of 0.24
its slope reaches −0.51, which sends a quarter of the way along a border to one
hundredth of the way along it. Every leak then piles into a corner — not a
bias, a collapse, and it looks exactly like the four-corner symmetry the
rewrite was meant to escape. It is 0.10.

**A leak's core is white and only its falloff is coloured, and no fixed tint
can do that.** Adding `leak * (1, 0.45, 0.19)` keeps the same chromaticity at
every strength, which is why the old wash read as flat tan wherever it was
visible. The response is now per channel and saturating —
`added = 1 − exp(−k_c · E)` — which is one dye layer clipping at a time, and it
gives the real progression: deep red where only the red-sensitive layer caught
enough light, through orange and yellow, to white where all three are at the
top. It also self-limits at 1.0 in linear light, so a hot leak cannot drive a
channel past white.

The consequence worth knowing: **`leak_feather` is the half-strength distance
of the light the leak *deposits*, not of the pixels.** The response compresses
the top, so on a blown leak the visible half-way point sits deeper — measured,
the same 150px feather reads as 227px at full strength and 149px where the
response is linear. `verify.py` measures the falloff, so it probes at
`leak_strength 0.1`, and it does it on **one** leak walked down its own centre
line: light adds, so a profile through a frame of twelve overlapping beams
keeps being propped up by the next leak along and read a 20px feather as 37px.

`leak_strength` is new, and there genuinely was no brightness control before —
the gain was hard-coded at 0.55 with only `leak_variation` moving it. Past
about 1.5 most leaks have a blown white core.

Cost is **+0.10s per leak** on a 24MP frame (1.05s with leaks off, 1.67s at 6,
2.25s at 12). `pad_for` is unchanged: the stage is per-pixel from global
coordinates with no kernel, so a tile needs no overlap for it.

### Leak sizes are pixels (changed 2026-08-01)

`leak_size` (a 0.05–10 fraction of a hidden maximum) is gone. Leaks draw their
reach from between **`leak_size_min` and `leak_size_max`, both lengths in
full-resolution pixels**, and `leak_feather` is a pixel distance too — *the
distance from the border at which the leak has fallen to half strength.* All
three are `spatial=True`, so a preset rescales them like every other length.

Feather-as-half-distance resolves to the same falloff exponent the old 0–1
softness drove, so none of that tuning was thrown away: solving
`(1 − hl/reach)^e = 0.5` gives `e = ln(0.5) / ln(1 − hl/reach)`. A short
half-distance is a large exponent and a tight bright rim; half the reach is
`e = 1`, a straight ramp; most of the reach is a broad wash. Because it is
absolute rather than a fraction, the same feather is a wash on a small leak and
a rim on a large one — which is what stops a frame of differently-sized leaks
looking like one shape at several scales. Measured on a 300px leak: asked
20/80/150/285px, delivered 20/78/149/270px.

Two things this fixed on the way:

* **The old edge distance was anisotropic.** It was the *normalised* distance,
  and X divides by the width where Y divides by the height — so on a 3:2 frame
  the same size reached 1.5× deeper from a side border than from the top, for
  no reason anybody asked for. In pixels the two agree by construction (240px
  now measures 220px from the side and 214px from the top).
* **`clamp_min(1e-4)` before the falloff power is a global fog.** Raising a
  1e-4 floor to a *small* exponent does not give a small number: at exponent
  0.23 it is **0.12**, a 12% lift over the whole frame wherever the leak
  reaches. It hid while the exponent bottomed out at 0.5 (a 1% floor) and reach
  was capped to a sixth of the frame — with both feather and size in pixels, a
  broad feather on a small leak reaches that exponent easily. Floor the base at
  zero instead; the exponent is always positive, so `0 ** e` is 0 and no guard
  is needed.

**A leak must not reach the middle of the frame.** The centre-fog guard is
**geometric rather than a taste constant**: reach is capped at half the frame's
short side, which is where `edge_d` tops out and therefore the reach at which a
leak just dies in the middle. Below it the pixel numbers are honoured exactly;
above it there is nothing left to reach. An earlier attempt let large sizes
lift a floor under the whole leak, which fogged the centre — measurably wrong,
and it reads as a bad exposure rather than as a leak. `verify.py` pins the
centre at **0.00e+00** for every size from 60 to 3000px and every feather from
2 to 1500px.

The cap has to be paid for **twice**, and that is easy to miss: the domain warp
can pull the falloff `_LEAK_WARP · reach` further in than the reach alone, so
the divisor is `_LEAK_REACH_SAFETY = 1.25` against a warp of 0.15 rather than
landing exactly on zero. A falloff exponent below 1 turns a float epsilon into
a visible lift, so the margin is not decoration.

`leak_variation` does not drive reach — the two size sliders state that
outright — and drives everything else: length, lean, fan, edge hardness, halo,
strength and hue. At 0 every leak is identical in all of those, which is
exactly what read as stamped.

**Measure the variation on few leaks, not many.** A run of lit border is only
one leak's peak if no other leak overlaps it, and light adds. Measured with the
same probe, the strength-spread ratio between variation 0 and 1 comes out 1.19
at eight leaks per frame and 2.07 at three — at eight it is mostly measuring
the brightest member of each merged pair. Measure at low `leak_strength` too,
or every leak drives its brightest channel to 1.0 and the spread reads as zero
whatever the setting.

**Calibrate pixel defaults against a photograph, not against the test plate.**
The first pixel defaults were 40–200px, which is about right on the 1500x1000
plate this section renders and **three to six times too small on a 6000px
frame** — leaks shipped as a few thin lines hugging the border and the user
reported it immediately. Every check passed, because every check ran on the
small plate. The defaults are 250/850 with a 180px feather. `verify.py` renders
a full-size frame at defaults for exactly this, and pins the proxy against it
(42.0% coverage at 1:1 and at half scale — a leak is a length, so the preview
owes the export the same scale invariance everything else does). Absolute
pixels do not adapt to frame size by design; that is what `reference_mp`
rescaling is for, and the bare defaults are tuned for a full-resolution photo.

Dust composites rather than adds, which is the only way opacity and luminosity
can be separate controls: added together they are the same number, since a
fainter speck and a lighter speck are indistinguishable. As a composite,
`dust_opacity` is how much of the photograph the speck hides and the luminosity
variation is what colour the speck itself is.

**Dust Softness is the speck's own edge width, and there is no blur any more.**
It used to widen a threshold band, because blurring a 2px speck by several times
its own size does not soften it — it erases it, since energy is conserved so the
peak collapses below anything visible and you end up with fewer specks rather
than softer ones. My first attempt read as "softness does nothing" for exactly
that reason, and the measurement was survivorship-biased on top: only the specks
that survived were left to measure. A *drawn* speck sidesteps all of it: the
edge width is a parameter of the shape, so softness costs nothing and removes
nothing. Measured 42% softer with the count untouched, and softness now varies
per speck straight off the site record instead of via a second noise field.

Traps from the thresholded era. The first three no longer apply to dust or hair
— they are here because scratches are still a field, and because the shape of
the mistake generalises:

* **Read thresholds off the field's real distribution.** Value noise is heavily
  centre-weighted, so a threshold of 0.88 — which sounds extreme — selects
  **10% of the frame**. Measured quantiles: 1% above 0.943, 0.1% above 0.988,
  0.01% above 0.998. First attempt put 9.7% dust on the frame, which reads as
  weather rather than film.
* **A gating field coarser than the image is a constant, not a mask.** The hair
  sparsity field had a 900px cell, so across a frame it spanned only 0.38–0.73
  and never crossed its 0.72 gate — hair rendered as *literally nothing*, and
  which nothing depended on where in the noise plane the frame happened to sit.
  Keep gating cells well under a frame.
* **Solve level-set widths, do not pick them.** A hair's width is ~`2·eps·cell`
  pixels. At `eps = 0.0016` with a 110px working cell that is 0.35px — sub-pixel
  before supersampling halved it again, so it drew nothing.
* **A fitted constant standing in for a count is a smell, not a calibration.**
  `_BLOB_CELLS_DUST` and `_BLOB_CELLS_HAIR` were honest about being accurate to
  a factor of 1.5, and that was treated as the price of the construction rather
  than as evidence the construction could not express what the slider claimed.
  When a control needs a fudge factor to mean what its label says, the label is
  describing a different implementation.

Light leaks need the *frame* size, which is why `render()` now takes `full_hw`
and `render_supersampled` scales it by `ss` alongside `y0`/`x0`. `render_view`
passes the whole source's size, not the read window, or the leak would slide
around as you panned. `verify.py` pins that: a crop of a leaked frame matches
the same region of the full render to 2e-05.

