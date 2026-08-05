# Edge destruction: scatter and anti-aliasing

## Scatter: diffusion without the average (added 2026-08-01)

Step 1, in linear light and **ahead of `micro_blur`**, because it is the same
physical event. A blur is diffusion as an *expectation* — average over enough
photons and deflection becomes a convolution. `scatter` resolves the deflections
individually instead: a share of the pixels are displaced onto a neighbour and
nothing anywhere is averaged. That difference is the whole feature, and it is
what the user asked for — detail destroyed, harshness kept.

### The two run scatter-first, and the old order was undoing itself (2026-08-03)

Changed on request, along with moving `micro_blur` to the bottom of its panel
section (then "Optical", now merged into Edge Destruction — see
`docs/panel-layout.md`) so the
panel and the pipeline read the same way. It is **not** a cosmetic reorder —
measured on separate plates at scatter 0.85 / reach 3 / blur 1px:

| | fine texture | hard edge |
|---|---|---|
| scatter alone | 100% | 100% |
| micro-blur alone | 28% | 34% |
| blur then scatter (old) | 28% | **60%** |
| scatter then blur (new) | 32% | **28%** |

The edge column is the whole story, and the old order's number is the surprising
one: **scatter was undoing the blur.** Displacing a blurred gradient by whole
pixels drops a hard step back into it, so the pair came out *harder* on borders
than the blur alone — 60% against 34% — which is not something either stage
claims to do. Scatter-first, each does its own job: scatter shreds the border
into raggedness and the blur averages that into a genuinely soft transition,
landing *below* blur-alone at 28%. Fine texture barely notices the swap
(28% → 32%), because scatter does not touch texture sigma either way. It is also
the physical order — light deflects off a grain and then goes on diffusing.

**This changes the look of every preset with both stages on**, which is `Stock`,
`ExtraGrain` and `Dramatic` (all carry scatter 0.85 with micro-blur 1.0–1.25).
They are softer on borders than they were. Nothing was migrated: the request was
to change the order, and re-tuning three presets to hide it would defeat that.

`verify.py` pins the order rather than trusting a comment, because a swap back
would look perfectly reasonable in a diff. The check is that blur + scatter must
measure *below* blur alone on a hard edge — which is only true one way round.

`pad_for` is unchanged: blur-then-displace and displace-then-blur need the same
total reach, and the terms were already summed rather than maxed.

Measured against a blur of the same 3px reach, on a fine-texture plate:

| | texture sigma | local contrast |
|---|---|---|
| `micro_blur 3.0` | 9% | 2% |
| `scatter 1.0` | 100% | 96% |

**Three rules, and breaking any one of them turns it back into a blur:**

* **Nearest-neighbour sampling on whole-pixel offsets.** `_warp` grew a `mode`
  argument for this. Bilinear at a fractional offset *is* a 2×2 average — the
  stage would still look like it worked and would quietly be a filter again.
  Displacements are rounded before the gather so the nearest choice is
  unambiguous rather than resting on which side of a half-pixel the float
  arithmetic lands. Verified: every output pixel is a copy of a real neighbour
  to **1.2e-07**, where the same-reach blur deviates by 6.3e-02.
* **Amount is coverage, not opacity.** It moves a threshold on a uniform
  field, so it sets *how many* pixels travel. Cross-fading a displaced pixel
  with the one it left is an average by another name, and at 0.5 it would be
  precisely the blur this replaces.
* **No mask, ever.** It masks itself: displacing a pixel whose neighbours
  already match it cannot change it. A smooth ramp comes through at its own
  slope × the travel (0.003 at a 3px reach) while detail is the only thing
  that comes apart. That is the exact inverse of `micro_blur`'s failure mode,
  which takes texture down *first* and edges second.

`_cell_noise` exists only for this, and the reason is the `_spread` trap in a
new place. The choice field is **quantised**, not thresholded — `floor(n·4)`
picks one of four stencil directions — and quintic value noise spans only
0.41–0.71 at p10–p90, so quantising it would fire two of the four directions
and give the scatter a diagonal bias nobody asked for. Reading the lattice
*without interpolating* gives back the hash's own uniform distribution. Its
blockiness is the other half of the point: every pixel in a cell reads the same
value, so a whole cell travels intact. That is what `scatter_cell` means —
lag-1 correlation of the displacement field runs **0.00 at 1px to 0.87 at 8px**.

`scatter_cell`'s range is **0.1–5px** (was 1–16, changed 2026-08-03 on request).
Worth knowing about both ends. The top drops range that existed: any file
holding a value above 5 gets clamped by `sanitize`, and none of the shipped
presets do — they are all at 1.2. The bottom is subtler, because the engine
floors the working cell at 1.0px and **that floor does not move**: one choice per
pixel is already the finest this can be, so there is nothing below it to resolve.
That makes the sub-1 part of the slider reachable *only through supersampling*,
which is what makes a working pixel smaller than a real one — at the default
supersample 2 the effective floor is 0.5, and every setting below it renders
identically. Same shape of trap as `grain_size` 0.1 versus 0.4 both flooring to
`_MIN_CELL`; the help text says so outright rather than leaving a dead zone that
reads as a working control.

The nine stencils are `Any`, `Cross`, `Diagonal`, `Box`, `Diamond`, `Donut`,
`Star`, `Horizontal`, `Vertical`. A stencil is the *set of places a pixel may
land*, which takes more than a direction count to say: `Diamond` keeps every
angle but holds `|dx|+|dy|` constant, so it reaches 12.0px on the axes and
8.5px on the diagonals where a disc reaches 12 both ways; `Donut` holds a hole
open (nearest landing 7.2px of a 12px reach even at Reach Spread 1, where every
other stencil fills solid to 0); `Star` runs alternate spokes short, measured at
a 0.35 diagonal/axis ratio against `Box`'s 0.94 on the same eight directions.
Each is verified by enumerating `_scatter_offsets` over its two uniform inputs
— a *rendered* probe cannot see the shape, because the choice field gives each
cell one direction and a sparse stencil is then sampled a few points at a time.

Two things bit here. `_SCATTER_STENCILS` is indexed by the parameter's value
and a preset file stores that index, so **renumbering silently changes the look
of every preset that used one** — the order there and the `choices` tuple in
`params.py` are one list in two places, and `verify.py` now pins them together
name-for-name and looks every check's pattern up by name. It caught exactly
that drift immediately: a hard-coded `scatter_pattern: 4` in the shell/disc
check quietly became `Diamond` when the three new stencils were inserted.

And **peak travel is the reach plus one pixel, not the reach.** `dx` and `dy`
are rounded to whole pixels *independently*, so two half-pixel roundings the
same way lengthen the vector by up to √2/2 — measured 12.5px on a 12px reach.
`pad_for` carries the same +1. It would have fitted inside the trailing `+ 4`
either way, but a stage that silently depends on another term's slack is a seam
waiting for somebody to tighten it.

`Param.choices` is new and generic: non-empty turns the control into a menu in
the client. The value stays a plain number, so the schema, the engine, `rescale`
and preset files are all unchanged — `App.tsx` is the only place that knows.
It is only for genuine either/or choices; there is no midpoint between "cross"
and "diagonal", and a slider that pretends otherwise invites you to leave it
at 2.5. `rescale` must never touch one: it is an index, not a length.

Cost is 0.69s → **0.80s** on a 6MP render at 2× (+16%), and `pad_for` grows by
the reach alone (108 → 112px at reach 4).

One honest limitation: at supersample 2 the `avg_pool` back down averages the
sub-pixel scatter events, so texture retention on a per-pixel-white-noise plate
falls from 100% to 73%. That is mostly the supersample round trip itself, which
costs 21% on that plate before scatter does anything — the same bicubic-up /
box-down softening `is_neutral` documents. Raising `scatter_cell` does not
recover it, and it is anti-aliasing doing its job, so it is left alone.


## Anti-aliasing: filter along the contour, never across it (added 2026-08-03)

Step 1c, in the optical block beside micro-blur and scatter, in linear light.
Ships at 0. The UI section has moved twice since it was added — first between
Optical and Tone Response (Optical/Colour before that merge), now right after
Edge Destruction (2026-08-04, on request, alongside the panel reorg in
`docs/panel-layout.md`). The
pipeline position is its own decision either way, and an
anti-alias filter is an *optical* element — a birefringent plate in front of a
sensor — so the light path is where it belongs. It also has to run before the
masks are measured, or the grain keeps keying on the jaggies it just removed.

**A stair-step is a pixel-scale wobble *along* a contour, not a hard transition
across one.** That single sentence determines the whole design: the filter is
three 1-2-1 taps along the isophote tangent and never crosses the edge.
Measured on a deliberately-aliased shallow diagonal, at strength 1 and radius 1:
contour residual **0.289px → 0.189px** while the across-edge slope keeps
**86%**.

### One pass is gentle, so strength above 1 repeats it (2026-08-03)

Reported as "does little to none", and fair — one three-tap pass removes 34% of a
stair-step, which is a correction rather than an effect. `aa_strength` now runs
to **3.0**, where whole numbers are whole passes and the remainder fades the last
one in:

| strength | jaggedness removed | across-edge slope kept |
|---|---|---|
| 1 | 34% | 86% |
| 2 | 52% | 77% |
| 3 | 64% | 70% |

**Repeat, do not lengthen.** Raising `aa_radius` is the obvious lever and the
wrong one, for exactly the reason `_AA_TAPS` is short in the first place: a
stair-step is one pixel wide by definition, so a longer filter starts averaging
away the shape the contour *has* rather than the wobble *on* it. Repeating
attacks only the wobble, and because each pass re-estimates the tangent from the
frame it was handed, it re-aims along a curving edge where one wide pass cuts the
corner. Same idiom and same reasoning as `_SAND_PASSES`, which was built for the
same problem one stage along.

Two things that had to hold, both pinned:

* **Strength 1 is bit-identical to the old single pass** (`0.00e+00`). Raising a
  ceiling must not move the values underneath it, or every existing setting
  quietly means something new.
* **`pad_for` counts both terms three times over.** Each pass displaces by a
  radius *and* re-derives its tangent from a fresh blurred luma, so both
  accumulate — 50px → 99px at strength 3 / radius 4. Pinned at `_AA_PASSES`
  rather than derived from the strength, because `pad_for` runs at the
  un-supersampled scale and would otherwise disagree with the renderer about the
  count. This is the same trap `edge_sand` hit, where counting only the first
  pass was fine to 4px grit and seamed from 8px up.

The trade stays a trade at the top of the range, which is the thing worth
watching: 70% of the sharpness for 64% of the jaggedness is still better than
sanding's 73%-for-32%, so this has not quietly become a blur. `verify.py` asserts
the ladder is monotonic *and* that the slope stays above 60%.

**It overlaps `edge_sand` in mechanism and is deliberately not the same
control.** This repo has a standing lesson about building a second thing that
does the first thing's job (`scatter`'s frequency split, cut after it was
built), so the three differences are worth stating plainly:

* **Position.** This runs at 1c on the source, where the aliasing that arrived
  with the file lives. Sanding runs at 8b to polish roughness the *jitter stage
  just added* and cannot reach back to fix the input.
* **Scale.** Three taps at about a pixel against sanding's five to ±2σ. The two
  remove different wavelengths: 92% of a jittered contour's roughness sits
  above 8px, while a stair-step is one pixel by definition.
* **The trade is better at this scale, which is the justification.** Sanding
  keeps 73% of the sharpness for 32% of the jaggedness; this keeps **86% for
  34%**. Both numbers are pinned in `verify.py` so a regression in either half
  fails — a filter that removed the jaggies by softening the edge would pass a
  jaggedness-only test and be worthless.

`_isophote` was factored out of the sanding loop and is now shared. Same vector,
same reason it must be estimated over a window rather than per-pixel: where the
gradient is weak the tangent is a ratio of two near-zero numbers, it swings on
float noise, and a filter reaching along it samples somewhere else — which
seamed tiled exports. Both callers gate on the returned magnitude.

**`aa_edge_only` reuses `_STEP_LO`/`_STEP_HI` and must measure the step exactly
the way edge softening measures it.** My first version multiplied the high-pass
by 2 for no reason I could defend, which put the gate on a different scale from
the constants it borrows and left it firing on fabric — measured, Edge Only 1
and Edge Only 0 came out within 0.5% of each other, i.e. the control did
nothing. With the fudge removed: fabric-scale texture (mad 0.010) keeps
**100%** at Edge Only 1 against 88% at 0.

Worth knowing when testing this: **per-pixel white noise is not "fine
texture"**. `_TEX_LO`/`_TEX_HI` put real fabric at 0.002–0.015 mean absolute
deviation; a ±0.125 noise plate is a hard edge at every pixel, and a gate keyed
on step size is *right* to fire on it (it keeps 79%, not 100%). My first
texture-protection test used one and read as a broken gate.

Cost is nil at 6MP — inside MPS run-to-run variance either way. `pad_for` grows
108 → 113 at radius 1 and 130 at radius 4, and it has to count **both** terms:
the taps travel a radius (a displacement, added outside the ×3) and the tangent
and step gate come off blurred luma (a kernel, inside it).

