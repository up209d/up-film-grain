# Performance

## Measured performance (Apple MPS, 24MP source, 2× supersample)

**The numbers below the 2026-08-04 audit section are historical.** Read that
section first — it supersedes the proxy-preview figures here.

* Proxy preview (2400px): **~1.25s** wall (0.7s render + encode + HTTP), 5MB
  — what a slider costs
* Full 1:1 preview (6000px): **~4.7s** (4.1s render + 0.6s encode), 32MB — the
  Render 1:1 button. Was ~9.5s before the `_fbm` cascade was turned around.
* Octaves is now nearly free: full-res 24MP is 4.14s at 3 octaves and 4.40s at
  10, against 32.6s at 10 before. Coarse lattices are a handful of points;
  the floored fine lattice it replaced was nearly per-pixel CPU hashing.
* Full export: tiled with progress. **JPEG 95 is the shipped default** (client
  `format` state and the `/api/export` fallback both) — measured 100.2% of
  grain sigma at 0.43MB against 9.82MB for png16, 4:4:4 so chroma grain
  survives. png16 stays for anything going on to further grading.
* Zoom and pan: **free** — pure browser transform, no request

### Performance audit, 2026-08-01

No regression at defaults — a 24MP full-res render measures **2.71s** against
the 4.14s recorded earlier, and the proxy **0.86s** against 1.25s. What got
slower is the *preset*: `Stock.json` turns on five stages that ship at 0
(`edge_soften`, `edge_sand`, `sharpen`, `global_intensity`, and jitter well
above default), so it costs **8.9s full / 1.6s proxy** — 3.3x defaults.

Ablated from Stock at full res, best of 3 in fresh processes:

| stage removed | saves |
|---|---|
| `edge_sand` (1.25 @ grit 5) | 1.7s — up to 3 passes x 4 warps each, the single dearest stage |
| `edge_erosion` | 0.8s |
| `edge_jitter` (1.5) | 0.6s |
| `global_intensity` | 0.3s |
| `sharpen` (6.49) | 0.1s — an unsharp is nearly free |

**Clump Size dominates everything else.** Holding Stock fixed and varying only
it: 0.1 -> 8.85s, 0.4 -> 7.76s, 0.8 -> 4.82s, 1.6 -> 4.25s. The noise lattice is
hashed on the CPU, so its cost is quadratic in density: a 1536px tile at cell
0.8 is 3.7M points against 0.23M at cell 3.2, measured **88ms vs 5ms per
3-field call, 16x**.

And `grain_size` 0.1 buys nothing over 0.4: both floor to `_MIN_CELL`, so
`_grain_field` returns a **bit-identical** field (verified 0.00e+00) while
costing 1.1s more. The extra cost is only the *secondary* fields — the ragged
edge envelope and the jitter displacement, whose cells are `grain_size * scale
* 2` and `* 3` and so escape the floor sooner. If a preset wants the finest
possible grain, 0.4 is the cheapest value that reaches it at full res.

Benchmark in a **fresh process per configuration**, and take a best-of-N: MPS
run-to-run variance here is +/-1s, enough that a single-shot ablation reported
*removing* `edge_soften` as making the render slower. Repeated large renders in
one long-lived process get progressively slower on MPS — a first pass at tile
sizing showed 3072 "beating" 1024 purely because of measurement order, and
reversed completely once each ran clean.

Tile size used to be hard-coded at 1536 for the preview and 1024 for the export:
per-tile overlap is fixed padding, so wider tiles amortise it (~5% better at the
default halation radius, ~12% at the widest). Both constants are gone — see
`tile_for` in the 2026-08-04 audit below.

The `/api/source` image is encoded once per upload and cached on the `Upload`
(18ms → 1.2ms on repeat). The untouched image never changes, so re-encoding it on
every parameter change was pure waste.

## Performance audit, 2026-08-04 — 3.70s → 1.31s, bit-identically

The `1.6s` proxy figure above was **2.3x stale**. Re-measured with one fresh
process per configuration (mandatory here — see the warning below), 2400x1600
proxy, ss=2:

| | before | after |
|---|---|---|
| parameter defaults | 0.419s | **0.309s** |
| `Stock`, first render of a parameter set | 3.701s | **2.223s** |
| `Stock`, any repeat render | 3.701s | **1.314s** |

**The engine's baseline was never the problem: `Stock` costs 8.8x defaults.** All
five changes below are **bit-identical** — asserted at 0.00e+00 in `verify.py`
against reference implementations of the code they replaced, not argued from a
tolerance. Nothing about the look moved.

Where the time actually went, profiled with `torch.mps.synchronize()` around each
primitive (exclusive time, `Stock` proxy):

| primitive | | |
|---|---|---|
| `_variable_cell_noise` (now `_grain_points`) | 1.130s | 24.1% |
| `_lattice_np` | 1.085s | 23.2% |
| `_value_noise` (excl. lattice) | 0.444s | 9.5% |
| `_smoothstep` | 0.334s | 7.1% |
| `_blur` / `_linear_to_srgb` / `_warp` | 0.536s | 11.4% |

1. **The Global Grain texture layer is cached** (`_global_grain_field`). It reads
   no image data, so it was being rebuilt for nothing on every render: 1.29s of
   3.70s, 35%. `global_intensity` and `global_opacity` are applied by the caller
   as one scalar multiply and so sit *outside* the cache — the two sliders anyone
   drags cannot miss it. This is the whole difference between the 2.22s and 1.31s
   rows above.
2. **`_lattice_np` runs in torch on the CPU, not numpy** — 2.2-2.6x for free,
   because numpy's uint64 elementwise ops are single-threaded and torch's int64
   ones are not. Worth ~17%.
3. **`_variable_cell_noise` carries the winning cell index** through its 25-cell
   search and gathers brightness once at the end, instead of rewriting `nfields`
   full-frame planes on all 25 iterations. 1.26x on that function at nfields=3,
   and a real cut in peak memory.
4. **Lattice bounds are computed in Python** (`_lat_span`) instead of by reading
   scalars back off device tensors. That was 32 MPS queue drains per
   `render_supersampled` at defaults and **108 at `Stock`**, each a full pipeline
   stall for a number Python already had.
5. **Tile size comes from a memory budget** (`tile_for`), replacing the two
   constants. See below.

### Two traps this audit walked into, both worth knowing

**`inference_mode` and kernel caching are both no-ops here, measured.** Wrapping
the render in `torch.inference_mode()` came out at 3.764s against 3.701s — no
change, because nothing sets `requires_grad` so no graph is ever built and there
is nothing to skip. Caching `_blur`'s gaussian kernels and `_warp`'s sampling
grids measured 0.90-1.02x: building them is noise next to the convolution. Both
look like obvious wins and neither is one.

**Never benchmark two configurations in one process.** The existing warning in
this file is not cautious enough about *why*. A long-lived MPS process gets
progressively slower, so a sequential sweep penalises whatever runs last — the
first pass at this audit reported `inference_mode` and `grain_size 0.4` as
*slower than baseline*, and reported the variable-cell path as costing 32% when a
clean measurement says 1.19s of 3.70s. One process per configuration, best of 3.

### `tile_for`: the tile is a memory decision, not a constant

Tiling is pure overhead — `pad_for` overlap is read, rendered and thrown away on
all four sides. Measured on the `Stock` proxy, fresh process each:

| tile | tiles | overdraw | time |
|---|---|---|---|
| 1024 (the old export default) | 6 | 1.59x | 4.460s |
| 1536 (the old preview default) | 4 | 1.32x | 3.701s |
| 2048 | 2 | 1.15x | 3.302s |
| single tile | 1 | 1.00x | 2.774s |

Interior *export* tiles are the worst case, since they pad on all four sides:
`1024 + 2*178 = 1380` square rendered for `1024` square kept, **1.82x**.

So why not always one tile? **Because peak memory went 6.0GB at tile 1536 to
8.0GB at 2048, and an 8GB machine is exactly where tiling matters most.** An
out-of-memory render is not slow, it is broken, so this is the one place the
"quality beats speed, take the lag" licence does not apply. `tile_for` derives
the tile from `_render_budget_bytes()` (half of the backend's recommended working
set, `FILM_GRAIN_TILE_BUDGET_GB` to override) and `_WORKING_BYTES_PER_PX` = 512,
which is *measured* and deliberately above the worst marginal figure — erring
that way picks a smaller tile, and the other way runs out of memory.

Two consequences worth expecting:

* **A wide-kernel preset now gets a smaller tile**, because it pads more and so
  has a larger working set for the same nominal tile. That coupling is the point;
  the old constants had none. `verify.py` pins it.
* **The renderer now sees tile sizes nobody hard-coded**, which is only safe
  because of invariant 1. `verify.py` checks tiled-vs-single-pass equality across
  several sizes rather than trusting the two the app used to use.

`main._render_tier` is new and is the *only* place either preview tier is
rendered. That is load-bearing rather than tidy: a preview-scale export must be
byte-for-byte the live preview, and once tile size became a *computed* value, two
call sites agreeing about it was no longer something a comment could guarantee.

### The lattice is bigger than the pixel grid, not smaller

`_lattice_np`'s docstring used to claim "the lattice is far smaller than the pixel
grid, so this is cheap." **False at every shipped setting**, and it is why 23% of
the render hid in plain sight for so long. `cell` is floored at `_MIN_CELL` = 0.8
*working* pixels and every preset sets `grain_size` to 0.1-0.3, so the base
lattice is *denser* than the pixel grid. Lattice points hashed per output pixel:

| defaults | Dreamy | Dramatic | Subtle | ExtraGrain | `Stock` |
|---|---|---|---|---|---|
| 2.5x | 5.7x | 38x | 48x | 54x | **58x** |

That is 291M hashes for one proxy preview. Moving it to the GPU in 32-bit would
be faster still and is deliberately not done: it would change every value and
reroll every preset's grain.

### Superseded renders now stop

`/api/preview` is a sync `def` holding `_RENDER_LOCK`, and Starlette cannot
interrupt a threadpool worker — so an aborted preview (the client aborts on
*every* new render) used to run to completion and keep the lock the whole time.
Latency then grew with how many edits happened rather than with how long one
render takes. `render_image` takes a `should_cancel` hook polled once per tile and
raises `RenderCancelled`; `main` hands out monotonic tickets and returns 499.

Tile granularity is the right checkpoint: no plumbing inside `render`, and the
wasted work is bounded at one tile.

### Two things left on the table, deliberately

Both were measured and then declined, so the numbers are not lost:

* **`Stock`'s `global_size_max 3.0` is what switches on `_variable_cell_noise`**,
  and that path is 1.19s of the preset's 3.70s. `grain_size 0.1` also buys
  nothing over 0.4 for the grain field (both floor to `_MIN_CELL`) yet costs 4%.
  Both are *look* decisions, so they stay.
* **`_VARCELL_RINGS` 2 → 1** would cut the 25-cell search to 9, but the 2-ring
  search is what buys the 0.7-cell domain warp that breaks the pixel-grid
  resonance. It would need a different resonance fix (a staggered lattice needs
  no warp at all) and would reroll the global-grain realisation.

  **This one was taken on 2026-08-05**, and by exactly the route predicted: the
  different resonance fix is a rotation of the whole lattice, which needs no
  warp, so the search is 3x3 and the ring budget stopped being the constraint.
  It did reroll the global-grain realisation, which was the point — see "Global
  Grain is one tilted point field now". The saving was spent on more points per
  cell rather than banked.


## Drawn dust and hair cost effectively nothing (measured 2026-08-06)

The rewrite from thresholded noise fields to per-mark drawing (see
`docs/film-texture.md`) looked like it might be a regression: the field version
was a handful of whole-frame tensor ops, and this is a Python loop over up to
400 marks with a dozen tensor ops each.

It is not, because **the cost is the marks' own total area rather than the count
times the frame**. `_mark_window` returns `None` for any mark that does not
touch the tile — the usual answer — and otherwise slices to the mark's own
footprint, so a 20px speck costs a 20px-square evaluation and not a 12MP one.
The old version evaluated two full-frame `_value_noise` calls plus a blur
whatever the count was.

Measured on a 12MP frame at supersample 1, against a 0.19s grain-only render:

| | added |
|---|---|
| dust 50 at 9px (a typical preset) | +0.00s |
| dust 400 at 40px (both sliders at their tops) | +0.02s |
| hair 10 at 200px | +0.00s |
| hair 40 at 600px (both at their tops) | +0.03s |

It also **removed work from `pad_for`**, which is a second-order saving that
outweighs any of the above on a tiled export: dust softening used to reserve
`dust_soften * 1.6 * dust_size` of overlap on every tile, so a 120px speck at
full softness widened every tile's read window by 200px in each direction and
the render then threw that overlap away.
