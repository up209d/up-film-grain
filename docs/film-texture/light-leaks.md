<!-- part of docs/film-texture.md -->

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

### Where a leak lands has to answer to the seed (fixed 2026-08-16)

Reported as "when light leak count is 1 it is most likely on the right edge of
the photo, so not really randomizing at all", and it was worse than *most
likely* — it was **always**, at every seed and every aspect ratio.

Leak `k` sits at `(base + φ·k + 0.10·(u−0.5)) mod 1` of the perimeter, and
`base` was the constant `0.37`. The perimeter is walked top → right → bottom →
left, so 0.37 of it lands past the top border and short of the bottom-right
corner whether the frame is 16:9, 3:2, square or 2:3 — and a tenth of a
perimeter of jitter is nowhere near a border's width. Swept over the whole seed
space on three frame shapes, one leak reached **one** of the four borders. The
seed was moving the leak's reach, lean, hardness and hue, which is why it read
as randomised until you noticed it never moved.

`base` is now drawn from `texture_seed`, off the same generator
`_mark_spread` uses for dust and hair. Two things had to survive it, and both
are checked:

* **The φ step still does the spreading.** `base` is drawn **once for the whole
  list**, not per leak, so every leak is still `base + φ·k` and raising the
  count adds leaks instead of rerolling the frame. Drawing the position from
  each leak's own generator would look identical on any single render and quietly
  break that.
* **Tile independence.** The list is still a function of the count and the seed
  alone — never of the region being rendered — so every tile builds the same one.

The fix moved every existing preset's leaks, which is the point; nothing about
their size, shape or strength changed. It also broke four `verify.py` checks
that had been reading fixed windows of the frame and assuming a leak would be
sitting in one — see `docs/testing.md`.

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
