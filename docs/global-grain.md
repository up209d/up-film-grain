# Global Grain

## Value noise is a quilt, and that is why Global Grain looked pixelated (2026-08-03)

> **Superseded 2026-08-05.** The global layer is no longer built from value
> noise at any setting — see "Global Grain is one tilted point field now"
> below, which is the authority. Everything here about *why* value noise
> quilts, and the phase-binned metric that measures it, is still correct and is
> still the argument against ever putting a lattice-interpolated field back in
> this layer. What has changed: `_SMOOTH_GAIN_K`'s 5.62 was refitted (the
> constant is now a quadratic in the Min/Max ratio), and Global Smoothness is a
> shape control rather than the cure for a defect.

Reported by the user as "global grain is so pixelate especially with high
global size", and it is a real structural defect rather than a bad setting.
**Value noise puts its extrema on an axis-aligned lattice** and interpolates
each cell separably, so the cells read as rectangles. Invisible at the default
1.6px clump; plainly blocky by 10px; at 20px it is a quilt.

Two things had to be ruled out first, and both were:

* **It is not clipping.** `gg` is clamped to ±1 after `/_GNORM`, which sounded
  like a plateau generator. Measured on flat grey at intensity 20, the rendered
  output never reaches a rail — range 0.424–0.576 at every size — so the clamp
  is not what is showing.
* **It is not the fBm stack.** Single-octave `_value_noise` and the two-octave
  field the global layer actually uses score the *same* gridiness. Adding
  octaves cannot help: every octave is built on the same kind of lattice.

The metric, because "looks blocky" needs a number: bin `|∂I/∂x|` by the
**phase within a cell** and take (max−min)/mean. Value noise's gradient
vanishes *at* its lattice points, so a gridded field swings hard between phases
and an organic one does not care where the boundaries are. A cell-20 field
scores **1.74**; through a full render the same field smoothed scores **1.72 -> 0.32**.

`global_size` max went 10 → 20px on request. `global_smooth` is the cure, and
it lives in the Anti Aliasing panel section rather than under Global Grain — same
job as the sliders next to it.

**The blur carries an analytic gain, and that is the whole design.** Blurring
by half a cell costs 40% of the amplitude, so without a normaliser "smoother"
and "less" would be the same slider. The gain cannot be `n.std()` — that is a
statistic of the region, invariant 1, and it would restore a different amount
in every tile of an export while every preview looked fine. It does not have to
be: measured, the attenuation depends on **sigma/cell alone** — 8, 16 and 32px
cells attenuate within 0.5% of each other at every ratio — so one closed form
covers every size. `1/sqrt(1 + k(σ/c)²)` fits it to 0.6% over the whole 0–0.5
range the stage can reach.

**Fit the constant against the field it is actually used on.** Calibrated on
single-octave value noise `k` came out 7.7; the global layer is a two-octave
fBm whose coarse half survives a blur far better, and 7.7 over-restored enough
to make full Smoothness **10% louder** than no smoothing — exactly the
amplitude coupling the gain exists to prevent. It is 5.62. `verify.py` pins
amplitude drift under 6% across the slider and under 8% at 4px and 12px clumps.

A domain warp was the other candidate and is worth knowing about: it takes
gridiness 1.74 → 0.39 at the same time as preserving variance *exactly*
(0.991–0.996×, since re-addressing a field cannot change its marginal
distribution) and needs no `pad_for` at all. It was not chosen because it does
not read as **smooth** — it trades squares for a crumpled-foil swirl, which is
a different look, not a softer one. Reach for it if the ask is ever "organic
but still harsh".

## Global Grain grew a size range, and it is a different noise entirely above Min (2026-08-04)

> **Superseded 2026-08-05.** There is one construction at every setting now, so
> Min and Max are the two ends of one size distribution and no longer switch
> anything — see "Global Grain is one tilted point field now" below. The two
> failure modes recorded here are worth keeping for the shape of the traps:
> the **resonance** is now fixed by rotating the lattice rather than warping the
> pixel coordinates (which is cheaper and works at every size, not just the ones
> a warp was tuned for), and the **tie-break** stopped existing when brightness
> readout became a weighted mean instead of an argmax. `_VARCELL_*` are all
> gone; their replacements are `_GRAIN_*`.

Reported as "too digital a job" — the layer looked too even, every clump the
same size, reading as manufactured rather than organic. `_fbm` cannot fix
this no matter how it is tuned: Octaves and Roughness stack coarser structure
*on top of* the base clump, but the base clump itself is always the same
diameter, because the whole field is one lattice at one pitch. Making clumps
genuinely differ from their neighbours means giving up the lattice.

**Three ways to read "randomise the size" were on the table, and they are not
the same feature.** One random size per render (the whole frame uniform, but
different between renders) does not touch the actual complaint, which is
variation *within* one image. Spatially blended patches — build the field at
Min and again at Max, blend between them with a slow selector field — is
cheap and safe but still gives every clump in a given patch the same size,
just a size that drifts region to region. The option that actually matches
"clumps genuinely differ from their neighbours" is a jittered point field
where **every clump independently draws its own radius** — closer to real
film, and a materially bigger rebuild, chosen deliberately over the cheaper
option once the tradeoff was explicit.

`global_size` is relabelled **Global Size Min** and a new **Global Size Max**
sits next to it, same range, defaulting equal to Min. The two are not a
symmetric pair the way the light-leak sizes are: Min already has an
established meaning on its own, so the effective ceiling is **`max(Min, Max)`,
never a swap**. A preset that raises Min alone, with Max still at its own
untouched default, must never cross into the new construction just because
the default happens to be smaller than the new Min — `verify.py` pins this
explicitly, separately from the plain "Max below Min" case.

### `_variable_cell_noise`: one independently-sized point per lattice cell

At or below Min, nothing changes — every existing preset (none of which sets
Max) renders the bit-identical field it always has. Above Min,
`_variable_cell_noise` replaces `_fbm` for that layer:

* One point per cell of a base lattice pitched at **Max** — the largest any
  point can be — jittered to the middle half of its own cell and given its
  own radius drawn uniformly from `[Min, Max]` and its own signed brightness,
  all from one hash per cell.
* A pixel takes its value from whichever candidate point has the strongest
  claim on it — the largest radial falloff, "closest relative to that point's
  *own* size" rather than closest in raw distance — and reads out that
  winner's own brightness. Selected, not summed: summing would read as fog,
  and selection is what keeps individual grains looking like discrete
  particles with their own edges.
* **Centred on 0.5, matching `_fbm`'s own convention, and this is load-bearing
  rather than cosmetic.** `_smooth_noise` re-centres explicitly
  (`0.5 + blur(n - 0.5, sigma) * gain`), so a field that means something
  different at 0.5 gets blurred around the wrong point. A gap — every
  candidate's falloff at zero, real film base between grains at a wide
  Min-Max range — has to land exactly on 0.5. The first version did not: it
  read out raw brightness with no gap-aware centring, so gaps landed near
  zero and the shared `*2-1` remap then drove every gap toward the fully
  negative rail — measured, a frame-wide **dark cast** the moment any real
  gap existed, the opposite of "nothing is here". Fixed by drawing brightness
  signed in `[-1, 1)` and returning `0.5 + 0.5 * falloff * brightness`: a gap
  is 0.5 regardless of any candidate's brightness, and a pixel at a point's
  centre reaches that point's own brightness at full amplitude, lighter or
  darker with equal odds — matching how real grain is not one-sided either.
  Swept Min 1.0 against Max 4/10/18: frame mean stays within 0.2% of neutral
  at every width, where the first version would have shown the bias grow with
  the gap.
* **Coverage gaps are the honest cost of a wide range, not a defect.** At Min
  1px / Max 20px the field measures ~70% coverage on a flat plane; narrower
  ranges cover more. The help text says so, because a control that quietly
  turns into "70% grain, 30% nothing" without warning reads as broken rather
  than as a dial.

Shape is `[1, nfields, h, w]`, matching `_fbm`, so it drops into the same
normalise-and-clamp pipeline. Chroma shares the exact same construction
through `_global_field` (a small closure picking `_fbm` or the variable field
per the same gate) — geometry (point positions and radii) is identical across
channels and only brightness is drawn independently per channel, the same
"shared shape, independent value" pattern `_lattice_np` already gives `_fbm`'s
multi-field calls. That is what lets a coloured variant share every grain's
position and size across channels while still giving each channel its own
intensity, rather than three unrelated point fields that would not even
agree on where a grain's edge is from channel to channel.

### Two failure modes found building it, both regression-tested directly

**The resonance.** When the effective cell size lands on (or very near) an
exact integer number of working pixels, the pixel grid and the point lattice
phase-lock: every pixel sits at *exactly* the same fractional offset within
its own cell, for every row and column, so no pixel is ever nearer than a
quarter-cell to any point, anywhere. Measured on a flat field: cell 1.00
scored 0.123 std against 0.193 at 1.05 or 0.95 — a **>35% amplitude hole**
sitting exactly where a user's slider is likeliest to land, since round
numbers are round numbers before and after the working-scale multiply.

Not a rare edge case to shrug off: it is the *default* well-behaved case for
anyone who reaches for whole-pixel sizes, which is most people. Fixed with a
small domain warp on the pixel side of the distance computation only — never
on which cell a pixel is nominally assigned to, which would need the
neighbour-search proof re-done against a wider slack. A first attempt warped
by 0.12 cells, the most a 3x3 (1-ring) search could safely absorb without
risking that proof, and it was not enough (0.123 → 0.128): breaking a phase
lock that pins *every* pixel at the same offset takes a warp comparable to
half a cell, not a tenth of one. Widening the search to 5x5 (2 rings) buys
room for exactly that — the far-corner geometry allows up to 1.25 cells of
warp before the proof would need widening further, and 0.7 cells is where a
sweep against cell 1.6 as a control lands within 0.3%. The warp itself is
sourced from `_value_noise` rather than a second point lattice, because
interpolated noise does not itself suffer this failure mode — its output
varies smoothly pixel to pixel regardless of whether its own cell aligns with
the pixel grid, so it cannot reintroduce the same problem at one remove.

**The tie-break.** Fixing the resonance surfaced tile independence failures
the isolated function did not show — comparing a whole-image render against a
tiled one turned up single-pixel deltas up to 0.07, tracing to a genuine
near-tie between two candidate points where the coordinate arithmetic behind
`dist` differs in its last bit or two depending on how a tile happened to be
padded and cropped to get there. Because the winner is chosen by a bare `>`,
a difference of a few counts in the last float32 bit can flip which point
wins — and therefore which brightness gets read out — which is a **discrete
jump**, not the gradual sub-pixel drift every other float-precision
discrepancy in this codebase settles for. Fixed with a fixed margin
(`_VARCELL_TIE_MARGIN`): a candidate has to beat the current winner by more
than the margin to take over, so whichever candidate the fixed iteration
order reaches first within a near-tie keeps the win consistently, regardless
of tile layout. Swept 1e-4 to 1e-2 against a scene that reproduced the
failure: 1e-4 left a 4.9e-3 gap open (too close to the measured ~2e-5 noise
floor to fully cover it), 3e-4 to 1e-3 both closed it to 1.4e-4, and 1e-2
reopened a much larger one (4.9e-2) — wide enough to start treating pixels
with a genuine, non-noise falloff difference as tied, a different failure
from the one the margin exists to fix. Set at 1e-3, in the middle of the
working range rather than at its edge. Re-verified across 48 combinations of
seed, size range, smoothing and tile size: worst delta 2.6e-4, comfortably
under the 2e-3 this codebase holds every other tile-independence check to.

### `pad_for` has two new terms, both gated on the variable construction being on

`_variable_cell_noise` reads lattice cells up to two rings away from a
pixel's own — a real read reach, not merely an addressing convenience,
because the render pipeline only ever hands the function a padded window
rather than the whole image. Under-reserve this and a pixel near a tile edge
silently substitutes a clamped boundary cell for the true neighbour, which
two different tile splits do differently — invisible in a single preview, a
seam in a tiled export, exactly the failure mode this codebase's whole
tile-independence discipline exists to catch. The gate matches `render()`'s
own decision **exactly**, including the floor on Min, so `pad_for` and the
renderer can never disagree about whether the field switched construction.
Global-grain smoothing's own kernel term was *already* wrong for this
feature before this section existed — it referenced Min alone, and above Min
the field's characteristic scale is Max, so a tile computed with only Min in
view would under-reserve and seam exactly where Max exceeds Min. Fixed
alongside the new reach term, not as an afterthought.

## Global Grain is one tilted point field now (rewritten 2026-08-05)

Reported: *"global grain really causes artifacts, makes it look fake, it renders
a repetitive pattern when zooming out, I can clearly see and feel the grid even
when zooming in, sometimes it does a good job, sometimes it does not, even with
the same config of min and max."* Every clause of that was a separate real
defect, and they had three different causes. The two constructions this layer
used to have — value-noise fBm at Max <= Min, a cellular field above it — are
both gone, replaced by one `_grain_points` used at every setting.

**Diagnose before rebuilding: the randomness was never the problem.** The first
thing checked was `_lattice_np` itself, since "repetitive" sounds like a bad
hash — measured over a 512x512 window its strongest off-origin autocorrelation
is 0.009, its uniformity chi2/dof is 0.89 and its fields are decorrelated to
0.001. The hash is fine. Everything below is *geometry*.

### The three defects, each with the measurement that found it

* **The grid, when you zoom in.** Value noise interpolates between lattice
  points with a curve whose derivative vanishes *at* those points, so every cell
  reads as a blob with flat corners and the blobs tile a visible quilt. On the
  phase-binned gridiness metric (bin `|∂I/∂x|` by phase within a cell; see the
  2026-08-03 section, which is still the authority on the metric) it scores
  **1.41 to 1.49** at 8, 12 and 20px clumps, and its autocorrelation peaks at
  exactly the lattice pitch — 0.24 at lag 5 for a 5px cell. Three shipped
  presets render that field at a 5px clump.
* **The repetitive pattern, when you zoom out.** The cellular path was a
  *stratified* point process: exactly one point per cell, jittered only within
  the middle half of that cell. The count in any region is then fixed by its
  area, so density cannot vary, and a point field whose density cannot vary
  averages to a featureless screen as you step back. A featureless screen at a
  distance is exactly what "repetitive pattern" describes. No amount of jitter
  fixes it — jitter moves points without changing how many there are.
* **"Sometimes good, sometimes not."** The pixel-grid resonance, already
  documented in the 2026-08-04 section: at cell sizes near a whole number of
  pixels the two grids phase-lock and the field loses up to 35% of its
  amplitude. The domain warp added to break it only partly worked, and it
  brought an artifact of its own — 0.7 cells of warp from a field pitched at
  0.37 cells shreds round grains into torn-paper shapes, which is most of the
  "artifacts, looks fake".

### The four decisions in `_grain_points`

1. **The cell lattice is rotated** against the pixel grid by the golden-ratio
   slope, 31.717 degrees. This is the single highest-value change: an irrational
   slope leaves the two grids incommensurate at *every* cell size, so the
   resonance has nowhere left to happen — measured std spread is **3.4%** across
   0.95, 1.0, 1.05, 1.6, 1.95, 2.0, 2.05, 3, 4 and 5px, against the old field's
   35% hole at the round numbers. It replaced the domain warp outright and is
   *cheaper* than it: four multiplies on the coordinate ramp instead of a whole
   `_value_noise` call, and it lets the neighbour search shrink from 5x5 to 3x3.
2. **Points jitter over the whole cell**, not the middle half. The 3x3 search is
   still exact, and the proof is short enough to state: a point in an excluded
   cell has a coordinate at least 2 cells from the pixel's own cell origin, so
   it is strictly more than 1 cell away; no radius can exceed 1 cell, because
   the lattice is pitched at Max and radii are drawn from `[Min, Max]`; and the
   falloff is exactly zero at the radius. `verify.py` runs the 3x3 against a
   reference 5x5 and they agree to 2.7e-07.
3. **Several points per cell, a fraction of them absent** — three slots at 0.62,
   so a cell holds Binomial(3, 0.62) grains: mean 1.86, and genuinely 0, 1, 2 or
   3. That is what gives the layer density variation instead of an even mesh.
   Slot counts 2 to 5 at matched mean density render almost identically, so it
   sits at the cheapest count that still varies: 3 slots over a 3x3 search is 27
   candidate evaluations, exactly what the old 5x5 single-slot search cost.
4. **A multi-octave cluster field** modulates each grain's brightness. This is
   the direct answer to the zoomed-out complaint and nothing else in the
   construction can supply it. Measured as the spread of *local contrast* at 16
   clumps to a block, it takes the layer from **0.026 to 0.152**, 5.9x.

Brightness readout also changed, from a hard argmax over candidates to a
weighted mean under `falloff ** _GRAIN_SHARE`. Amplitude still comes from the
single largest falloff, so a gap is still exactly a gap — this is *not* the
"sum the candidates" construction the old docstring warned reads as fog, since
the sum is normalised. It removed two things at once: the hard cusp where two
overlapping grains of different brightness changed places, and the tie-break
margin, because a weighted mean has no winner to flip. Tile independence went
**2.6e-04 -> 1.2e-06**.

### Three things that were built or tried and are worse

* **Rotating the value noise instead of replacing it.** The obvious cheap fix,
  and it does not work: the quilt is made of plateaus *at* the lattice points,
  so rotating the lattice produces a rotated quilt — and a worse-looking one,
  because rotated squares alias against the pixel grid. Rendered and looked at
  before it was discarded. The quilt is a property of the interpolant, so the
  only repair is a different kind of field.
* **Single-scale clustering.** Clustering at one pitch gives every
  clump-of-clumps the same diameter, so zoomed out the frame reads as regular
  blobs — a different repeating pattern rather than none. Three octaves is where
  the eye stops finding a characteristic size.
* **Measuring the clustering by block *means*.** The first version of that check
  read a flat 0.99x and was measuring nothing. Clustering scales each grain's
  brightness *magnitude*, and brightness is signed, so it leaves every local mean
  alone and moves only how grainy one region is against another. The metric has
  to be the spread of local *contrast*.

### `global_intensity` used to mean two loudnesses; now it means one

The two old constructions disagreed by 43% on a flat field (post-clamp sigma
0.684 for the value-noise path against 0.477 for the cellular one), so the
layer's loudness turned on whether Max happened to exceed Min. `_grain_gain`
normalises the field to a fixed target with a closed form in the Min/Max
*ratio* — closed form for invariant 1, since a measured `std()` would normalise
every tile of an export differently. It can be closed form because the point
pattern in cell units is scale-free, so the std depends on the ratio alone:
verified across an 8x range of absolute size (pitch 4, 8 and 16 working px agree
to 1.4%), and the cubic fits the ratio sweep to 0.12%. Rendered amplitude is now
flat to **6.8%** across the whole size range.

The target is the *cellular* path's old level, which keeps the default preset
where it is. Measured end to end on the repo's own test frame, global-layer
sigma per shipped preset:

| preset | Min / Max | old | new | |
|---|---|---|---|---|
| `Stock`, `Vintage`, `ClassicSoft` | 1 / 3 | 0.01352 | 0.01475 | **+9%** |
| `Dreamy` | 5 / — | 0.00856 | 0.00751 | −12% |
| `Subtle` | 5 / — | 0.00900 | 0.00776 | −14% |
| `Dramatic` | 5 / — | 0.01119 | 0.00906 | −19% |
| `ExtraGrain` | 2.05 / — | 0.01479 | 0.01166 | −21% |

Nothing was migrated, per this file's standing rule: the request was to fix the
field, and re-tuning presets to hide the change would defeat it. The four that
got quieter come back with about 1.2x on their `global_intensity`.

### Two knock-on changes outside the field

* **`_SMOOTH_GAIN_K` had to be refitted, and it is no longer one number.** It
  restores the amplitude Global Smoothness's blur costs, and it was 5.62 —
  fitted against the two-octave fBm. A field of discrete grains carries far more
  of its energy at its own edges, so the same blur takes much more of it, and
  5.62 under-restored by **21%**: Smoothness was quietly turning the layer down.
  Worse, the right `k` depends on the Min/Max ratio (18.3 at a wide range, 13.5
  at a single size), because wide ranges contain small grains and small grains
  are what a blur takes first. One constant cannot hold better than 6% across
  that; a quadratic in the ratio holds **2.1%**. Third time this constant has had
  to be refitted against a new field — the lesson from 2026-08-03 stands.
* **`pad_for` reserves nothing for the field.** The old cellular path carried a
  `_VARCELL_RINGS * cell` term. `_grain_points` derives its own lattice window
  from whichever window it is handed, with a ring of slack on every side, so a
  pixel always sees its true neighbouring cells however the frame was split —
  measured at **1.2e-06** between a whole-frame render and arbitrary sub-windows
  at zero padding, against the 2e-03 every other tile-independence check here is
  held to. `global_smooth` is the layer's only remaining kernel and is still
  reserved for. `verify.py` pins both halves, because with the term gone there is
  no slack left over to hide a reach nobody accounted for.

### Cost: this is the expensive option, taken deliberately

The user's framing was "this is an expensive operation, we should make it
worthy... we already cache this, so the next render is quick", so quality was
taken over speed throughout — the standing priority in this file anyway.

Per-call on MPS at 1536 square, three fields (the chroma case):

| Min-Max | old | new |
|---|---|---|
| 1-3 (`Stock`) | 67ms | 103ms |
| 4-8 | 62ms | 90ms |
| 0.8-0.8, the finest lattice | 95ms | 288ms |
| 5-5 via the old fBm path, one field | 2.3ms | 65ms |

That last row looks alarming and mostly is not, because **the cache absorbs
it**. End to end on a 2400px proxy at ss=2, best of 3 in a fresh process per
configuration:

| preset | cold (cache miss) | | cached | | `pad_for` | |
|---|---|---|---|---|---|---|
| | old | new | old | new | old | new |
| `Stock` | 2.27s | 2.50s | 1.34s | **1.32s** | 178px | **160px** |
| `Dramatic` | 1.31s | 1.63s | 1.17s | **1.17s** | 119px | 119px |
| `ExtraGrain` | 1.46s | 2.48s | 1.37s | **1.37s** | 133px | 133px |
| `Subtle` | 0.55s | 0.94s | 0.53s | **0.51s** | 73px | 73px |

So a repeat render is unchanged or a hair *faster* — `Stock` gains back more
from the dropped `pad_for` term (178 → 160px of overlap on every tile) than it
loses on the field. The whole cost lands on the cache miss, which is +0.2s to
+1.0s, and the two sliders anyone actually drags (`global_intensity`,
`global_opacity`) are applied outside the cache boundary and cannot miss it.
What does miss it is dragging Global Size Min or Max, and that is the one
interaction this rewrite made slower.

Two speed levers if it ever matters, neither taken: `_GRAIN_SLOTS` 3 -> 2 (18
evaluations instead of 27, and the rendered difference was hard to see), and the
per-slot lattice hash, which is now `3 + nfields` channels *per slot* against the
old `3 + nfields` total — that is the whole of the 0.8px row above, and it is the
reason the presence draw and the radius draw share one hash channel rather than
taking two.


## Global Grain grew a chroma slider (2026-08-03)

On request, at the same time as the Colour panel merge (see
`docs/panel-layout.md`). Same job as `chroma_grain` on the main grain layer —
decorrelate the three channels so this layer carries colour speckle instead of
pure luminance noise — and deliberately a different construction. The
main grain draws three independent fields and blends out from their rescaled
mean. Copying that here fails twice:

* **It would reroll every existing preset.** The mean of three fields is not the
  single field this layer has always been built from, so chroma 0 would render a
  different pattern than the one `Stock` was dialled in against. `verify.py`
  pins chroma 0 as bit-exactly monochrome (max channel spread **0.00e+00**).
* **That blend does not hold amplitude.** The mean and the per-channel fields
  are correlated, so measured pre-clamp it dips to **88.8%** of its own strength
  at chroma 0.5 and returns to 99.9% by 1.0 — the slider moves loudness as well
  as colour, which is exactly the coupling `_SMOOTH_GAIN_K` exists to prevent
  one slider along.

So the mono field is left alone and a **mean-zero** deviation `d` is added on
top, from its own seed. Because `d` sums to zero across channels its statistics
are fixed — var `2/3`, covariance `-1/3` of a single field — which makes the
coefficients solvable rather than a matter of taste:

```
g_c = A·m + B·d_c,    A = sqrt(1 - 2/3·c),  B = sqrt(c)
```

giving unit variance and cross-channel correlation **exactly `1 - c`** at every
setting. Measured 1.000 / 0.497 / -0.003 at chroma 0 / 0.5 / 1, with pre-clamp
amplitude flat to 0.6%.

**The one thing that does move is the clamp, and it is worth knowing why.**
Mixing in `d` gaussianises the field, so it reaches the `±1` rails less often —
clipping falls **25.4% → 22.8%** across the slider. A clipped sample sits at
exactly ±1 rather than wherever it was headed, so *less* clipping means slightly
less measured sigma: rendered amplitude drifts 100% → **97.0%**. That is the
hard tails doing their job rather than the blend leaking, and it is a third of
the wobble the other construction has. `verify.py` allows 5%.

Gated on `global_chroma > 0.001`, so the second `_fbm` is not paid for at the
default. No `pad_for` change — the field is addressed in global coordinates on
the same cell as the mono one and goes through the same smoothing kernel, which
is already covered.
