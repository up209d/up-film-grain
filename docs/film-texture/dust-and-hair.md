<!-- part of docs/film-texture.md -->

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
only fight the eccentricity draw. Their sum is the number that matters — the
radius is `1 + Σ aₖ·cos(kφ + pₖ)`, so a sum at or above 1 folds the outline
through its own centre and draws a shape with a bite out of it.

Measured with the isoperimetric quotient `4πA/P²`, against **a disc rendered
through the same rasteriser** rather than against 1.0 (a rasterised outline
over-counts its own perimeter by a factor that depends on the radius). And the
other half, from the second moments: mean axis ratio **1.47** where the same
disc renders 1.04 — so they are round shapes and not circles.

### `dust_irregular`: the harmonics became a slider (2026-08-08)

The perturbation above used to be **unconditional** — every speck carried it,
the amplitudes summed to a fixed 0.22, and there was no setting at which a speck
was a clean ellipse. The next report was the mirror image of the one that put it
there: *"dust particle now is not rounded"*. Shape was doing something the user
could see and could not reach.

So the whole perturbation is scaled by `dust_irregular`, and **it ships at 0**:

* **At 0 the outline is exactly the ellipse.** Not nearly it — the amplitudes
  are literally zero, the radius factor is literally 1, and `verify.py` asserts
  a `0.00e+00` delta against a build with `_DUST_HARMONICS` zeroed out. A
  roundness control whose zero still dents the outline is the original complaint
  wearing a slider, so the check is an equality rather than a tolerance.
* **At 1 the specks are chipped.** `_DUST_HARMONICS` now holds the amplitudes
  *at 1* rather than the amplitudes that ship, and they sum to 0.53 — 2.4× what
  was there before, because the top of the slider has to be a visibly different
  thing from the middle of it. Measured, the isoperimetric quotient drops 30%
  from 0 to 1.
* **The amount is drawn per speck** (`_DUST_ROUGH_SPREAD`, 0.45…1.5), for the
  reason `_DUST_SIZE_SPREAD` exists: one lumpiness stamped out N times reads as
  a pattern. The 1.5 at the top is what sets the real worst case — 0.53 × 1.5 =
  0.795, still under the 1 that would fold the outline.

Everything the ellipse does is *outside* this slider. Size spread, eccentricity
and angle are untouched at 0, so "round" still means a varied population of
ovals pointing in every direction rather than a field of identical dots.

### Dust Softness runs to 5 (2026-08-08)

Reported as too weak at its maximum, and it was — the slider stopped at 1 but so
did three things underneath it, which is why raising the range alone would have
done nothing:

* the per-speck draw was clamped to 1.0;
* the edge fraction was clamped to 0.9 of the speck's own radius;
* `_DUST_SOFT_FADE` was applied linearly, and extended past 1 it crosses zero at
  2.2 — the specks would have got softer for half the new travel and then
  deleted themselves.

All three are lifted, and **the mapping below 1 is bit-for-bit what it was**,
which is the constraint that shaped the fix: ten shipped presets carry
`dust_soften` values chosen when 1 was the top, and they have to render as they
did. So the per-speck clamp is `max(dust_soften, 1.0)` rather than a constant,
the 0.9 now clamps only the sub-pixel *floor* on the edge (the one case where an
absurd relative edge is an artifact rather than a request), and the fade applies
over the first unit of softness only.

Past an edge of 1 the inner smoothstep bound goes negative. That is the point
rather than a bug: the speck stops having a solid core and becomes a diffuse
smudge that dims its own peak as it widens, which is both what badly
out-of-focus debris looks like and why the explicit fade does not need to keep
going. At the top of the slider the edge reaches 3.85 radii.

`pad_for` still reserves nothing for any of this — a speck is clipped to its own
footprint in absolute coordinates, so a wider footprint is simply a wider window
that both tiles compute identically. The zero-overlap tiling check runs at
softness 5 and irregularity 1 for exactly that reason.

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
