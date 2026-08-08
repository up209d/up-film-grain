<!-- part of docs/global-grain.md -->

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
