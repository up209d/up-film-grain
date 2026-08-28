# Performance

## Performance audit, 2026-08-08 — memory was costing time, and the cache was dead

Measured on an M4 Max (14 cores, 36GB), idle, one fresh process per
configuration. **Benchmarked on `SuperPortra` as well as `Stock`**, which is what
found the defect: a `Stock`-only measurement reports everything below as healthy.

| | before | after |
|---|---|---|
| `SuperPortra` proxy preview (24MP source) | 7.47s GPU / 21.5s CPU | **1.76s / 11.3s** |
| `VintageDarkGrainy` proxy | 1.64s / 9.36s | 1.64s / 9.47s |
| `Stock` proxy | 1.13s / 6.81s | 1.15s / 7.17s |
| 24MP export, `Stock` | 30.6s, **27.2GB** | **21.96s, 8.07GB** |
| 24MP export, CPU | 154.1s, 22.5GB | 145.3s, 22.0GB |

Three findings, and the first two are the kind that hide behind a healthy-looking
default preset.

**1. The Global Grain texture cache had a 0% hit rate on any preset using the
source-masked layers.** `_GG_CACHE_BYTES` was a flat 0.5GB, sized by a comment
reading "113MB per tile at tile 1536 / supersample 2" — written when the tile was
a hard-coded constant and the section had **one** layer. It has five now and
`tile_for` computes the tile, so `SuperPortra` at a 2400px proxy wants 922MB, the
LRU held two entries, and **every render missed all five**. Re-rendering identical
parameters measured 7.36s and 0 hits; with a budget that fits, 1.78s and 5 hits.
`Stock` never showed it because one layer always fitted — which is why the
2026-08-04 audit recorded this cache as working.

It is a share of a *disk* budget now (`engine/diskcache.py`, 2026-08-29);
between 2026-08-08 and then it was `device._grain_cache_bytes()`, a share of the
same device pool `tile_for`
draws on. The two genuinely compete and the split is explicit for the first time:
`tile_for` used to take the whole budget while the cache took 0.5GB on top, so
their sum was never a real ceiling.

**2. The MPS allocator was hoarding 17GB, and that cost time as well as memory.**
A 24MP export peaked at **27.2GB driver-allocated against 9.9GB of live
tensors**. On unified memory the allocator's free list is system RAM, so holding
it starves the machine and the render slows down. `release_cache` between tiles
brought the same render to 13.0GB *and* 1.5× faster. `_WORKING_BYTES_PER_PX` was
also under the true figure by 17% and is now 640.

Two things this got wrong on the way, both worth knowing:

* **Releasing after a single-tile render is a net loss.** It has no peak to
  bound, so all the call buys is making the next render re-acquire the blocks —
  `Stock`'s proxy went 1.13s → 1.45s and `VintageDarkGrainy`'s 1.64s → 1.94s for
  no memory saved. The release fires *between* tiles only.
* **Releasing on small tiles is a net loss too.** `verify.py` renders many small
  frames at tiles from 256 up and went **35.7s → 96.4s** with the call
  ungated. `_RELEASE_MIN_SHARE` gates it at half the tile budget; the suite is
  back to 36.1s and every tile large enough to matter still releases.

**3. A superseded preview could not be stopped.** `render_image` polled
`should_cancel` once per tile — but `tile_for` returns 2400 for a 2400px proxy,
so every live preview takes the single-pass branch, where the hook was checked
once *before* the pass and then not again. Measured on `SuperPortra`: **one poll
in 7.91s** on the GPU, 21.5s on the CPU, every second of it spent on a frame the
client had already abandoned while holding the render lock.

The ticket machinery in `runtime.py` was correct all along; it simply could not
act at the granularity that mattered. `_poll_cancel` now fires at all 24 stage
boundaries inside `render()` — 25 polls in an untiled pass, against 1 — and
`render_image` returns the abandoned render's memory rather than leaving it
reserved against a frame nobody will see. `verify.py` pins the poll count as
"many more than one" rather than exactly, so adding a stage does not fail it.

Worth noting what was *not* wrong: the client (`usePreview.ts`) debounces and
aborts the in-flight request on every new render, so requests never stacked, and
queued threadpool workers exit immediately on acquiring the lock. The waste was
one render — but on the flagship preset that is the whole interaction budget.

**4. One gaussian was 101s of the 154s CPU export.** Scratch softening blurs at
`scratch_soften * 3 * scratch_width * scale`, so the 0.9 / 14.85px that **10 of
12 presets ship** is sigma 40 at scale 1 and **sigma 80 at supersample 2** — a
483-tap separable convolution over the whole frame. It moves 0.04% of a frame's
pixels by more than one 8-bit level, and it crushes the scratch layer's own peak
from 1.00 to 0.14, which is why scratches read as barely there.

`_blur` now decimates above sigma 32: pool by `k = floor(sigma / 16)`, blur at
the residual sigma, resample back bicubic. Below the threshold `k < 2` and the
exact path runs, **bit-identical** — pinned from both sides in `verify.py`.

| | CPU | GPU |
|---|---|---|
| 24MP export | 145.3s → **58.9s** | 22.0s → 22.0s |
| `SuperPortra` proxy | 11.3s → **7.98s** | 1.78s |
| `Stock` proxy | 7.17s → **3.47s** | 1.13s |

**The first version of this seamed exports, and the reason is worth keeping.**
`avg_pool2d` lays its grid from the tensor's own top-left, so two tiles covering
the same pixel pooled it into differently-phased cells — invariant 1, measured at
6.35e-03 on colour grading against a 2e-03 bar. The fix is an explicit `origin`
argument: given the tile's absolute offset the grid is shifted onto absolute
multiples of `k`, and **a caller that cannot supply one gets the exact path**.
The default is the safe one, and the only opt-in so far is the scratch blur —
re-measured after gating, that alone accounts for the whole saving above.

**5. Three slider ranges were open past the point of usefulness**, and one was
open past the point of *possibility*. All three were narrowed; no shipped preset
sat near any of the old ceilings, so nothing in `presets/` changed as a result.

* `grade_recover_radius` 4…200 → **4…64**. At 200 a 12MP export is 24 tiles and
  9.14× overdraw, 79.7s against 3.12s at the default. **No tile size rescues
  it**: 2× overdraw would need `tile >= 4.83 * pad`, a 5072 tile whose working
  set is ~221GB. On the CPU, 100 takes 56.5s for a *proxy*.
* `grade_clarity_radius` 2…80 → **2…48**. Same shape, milder: 2.23× overdraw at
  24MP, 1.76s → 2.66s at 12MP.
* `grain_size` 0.1…10 → **0.4…10**, and the eleven presets below the new floor
  were re-authored. 0.1 to 0.4 was a dead zone: both floor to `_MIN_CELL`, so
  the grain field is bit-identical, and only the *secondary* fields (the edge
  envelope at ×2 and the jitter at ×3) got finer — 1.8× the cost of the coarse
  end for no additional grain detail. The look moves slightly and only at edges:
  mean 0.06 levels on `Stock`, p99 1.3, with 1.1% of pixels past one 8-bit level.

Worth recording what is **not** worth clamping, because it is the opposite of
what anyone expects: `global_size_max` is **cost-flat across its entire 0.1…20
range** (0.35–0.37s), because `_grain_points` runs a fixed 27 full-frame
iterations whatever the cell size. Lattice density is not the cost. `octaves`
1→10 is 1.45×, `edge_sand_grit` 0.3→20 is 1.27×, `scatter_radius` 0.5→24 is
1.07×.

**6. Supersampling is a user choice now** — 0.5× / 1× / 1.5× / **2×** / 3×,
default unchanged. Cost is roughly the square of the factor, so this is the
biggest single lever anyone has over render time, and the bar now says so:
past 5s on a GPU or 10s on CPU it flags the config as heavy and offers the next
factor down.

Two things this had to get right. `render_supersampled` **keeps `avg_pool2d` for
whole-number factors** — an antialiased `interpolate` at 2× is a 4-tap
triangular filter, not a 2×2 box, so swapping it in unconditionally would have
rerolled every shipped preset; `verify.py` pins 1×, 2× and 3× bit-exact
(0.00e+00) against the old path. And a fractional factor cannot give a whole
working grid on every tile, so the factor is rounded to whole pixels and
`scale`, `y0`, `x0` and `full_hw` are all derived from the grid *actually*
rendered — get that wrong and the noise lattice resolves to different global
coordinates than the geometry does. Tile independence is now checked at all five
factors, not just the two integers it used to cover.

**7. The texture cache held superseded parameter states until pressure evicted
them.** Everything in its key after the tile coordinates is a *generation* — the
parameter state a field belongs to — and an older generation can never be asked
for again, because any render that wanted one would have to put those parameters
back and would then be current. Under plain LRU those entries sat there to the
byte cap: on `SuperPortra` at a 2400px proxy, dragging Global Size a few times
filled toward the whole 4.5GB allowance with fields nothing could reach.

Two generations are now kept and the rest dropped on sight. Measured: resident
memory plateaus at **461MB / 10 entries** instead of growing to the cap, and
stepping a slider back one value still hits. Two rather than one because A/B
against the previous value is exactly what people do — and because `scale`
reaches the key through `gcell`, so the proxy and the 1:1 render are different
generations and both need to survive.

**SSD spill for this cache was then declined**, having been planned. The whole
point of spilling was that the working set did not fit; two generations at 461MB
fits comfortably on an 8GB machine, so there is nothing left to spill. The
checkpoint chain is the better candidate and is a bigger object read once per
tile.

**8. Pipeline checkpoints: the intermediate frame at a section boundary.**
Editing a slider near the end of the pipeline re-ran the whole thing to produce
a frame differing only in its last few stages. Two boundaries now hold that
intermediate — after Pre Sharpen, and immediately before Global Grain.

`SuperPortra` proxy, per slider drag:

| | GPU | CPU |
|---|---|---|
| Global Grain / Sharpening / Film Texture (35 sliders) | 1.76s → **0.20s** | 7.9s → **0.97s** |
| Edge Destruction, Grain Structure, Halation, Tone (CP-A hit) | 1.40s | 5.2s |
| Colour Grading (full render) | 1.76s | 8.8s |

**Which boundaries are usable is a property of the pipeline, not a choice.** An
AST liveness pass over `render()`'s 73 top-level statements found exactly seven
where the image is the only thing live, in two clusters; the middle carries five
to nine planes at once because `lum_ref`, `hp`, `m`, `edge` and `wgt` are derived
early and consumed late. Within each cluster only the deepest earns its place.

Three things this had to get right, and the third was got wrong first:

* **The key is derived, not listed** — the whole sanitised parameter dict minus
  the sections below the boundary, with the LUT carried by `lut.id`. A
  hand-maintained upstream list would stop covering the next parameter anyone
  adds, and CLAUDE.md promises adding a control is one `Param` and one
  `p["key"]` read.
* **Downstream comes from *execution* order, not `GROUPS`.** The panel and the
  pipeline do not agree yet: Halation is panel section 8 and runs 5th. Slicing
  `GROUPS` would call it "below Global Grain" and a Halation edit would hit a
  checkpoint taken before it ran.
* **Naming a boundary off by one is a stale hit.** CP-A is saved *after* Pre
  Sharpen; it was first written as `GROUPS[1:]`, which put `pre_blur` itself
  below it, so dragging Pre Blur returned the previous frame. `verify.py` caught
  it at **9.77e-01** — most of full scale — on the first run of the check.

That check is the deliverable as much as the cache is: it renders one parameter
from every section against a warm cache and requires the result to be bit-equal
to an engine that has never seen a checkpoint. A stale hit here renders a
plausible, wrong *photograph*, where the texture cache's version renders only a
wrong texture.

**Neither cache is in memory any more** (2026-08-29): both are indexes over
files on the SSD, and a checkpoint hit reads 184MB back in under 100ms to skip
1.2s of render. See `docs/disk-cache.md`. Everything below still holds — the
policy did not change, only the medium.

Memory is bounded the same way the texture cache is — two generations per
boundary, so 369–737MB at a 2400px proxy rather than growing to the cap.
Preview tier only: at export scale a frame is 1.15GB per tile and nobody is
dragging a slider.

**9. A single large tile is no longer fastest at export scale**, contradicting
the `tile_for` sweep below. That sweep was measured on a 2400px proxy. At 24MP,
tile 4096 measures 22.5s / 32.9GB against tile 2288's 20.1s / 13.0GB — memory
pressure dominates the overlap overdraw the sweep was reasoning about.

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
