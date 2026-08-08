<!-- part of docs/global-grain.md -->

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
