<!-- part of docs/global-grain.md -->

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
