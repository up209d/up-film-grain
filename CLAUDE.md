# CLAUDE.md

Working notes for this repo. `TOPIC.md` holds the domain rules and pipeline
design; `README.md` holds setup and usage. This file is the catch-up: state,
constraints, gotchas, and the reasoning behind decisions that are not obvious
from the code.

## What this is

A web app that applies organic film grain to still photographs. Python/PyTorch
image service, React UI, single service (FastAPI serves the built client).
Working and verified end to end as of 2026-07-31.

## Priorities (from the user, and they override TOPIC.md's original draft)

**In scope — the point of the app:** detail destruction, edge softening and
edge noising, grain, halation, chromatic edge fringing.

**Deferred — the user has a separate project planned:** colour grading in
general. `vibrance` and `brightness` were added on request 2026-07-31 and
ship at 0 like the rest, so the pass-through still holds. The engine implements tone curves, contrast, toe/shoulder, split
toning, highlight desaturation and base fog, and exposes them as sliders, but
they **ship neutral (0)** so the pipeline is a colour pass-through. Do not tune
them or fold them into presets without asking.

**Quality beats speed.** The user has explicitly accepted lag and latency. Do
not clamp octaves, lattice density, blur radii, supersampling or preview
resolution for performance alone. If a quality/speed trade-off appears, take
quality by default and expose speed as opt-in.

`TOPIC.md` was originally Gemini-generated and the user has said it is not
authoritative — it has since been rewritten to match what was actually built,
including a section correcting three claims that did not survive implementation.

## Layout

```
server/params.py    parameter schema -- SINGLE SOURCE OF TRUTH for engine + UI
server/engine.py    the pipeline (module docstring states the invariants)
server/imageio.py   decode/encode, incl. a hand-written 16-bit RGB PNG encoder
server/main.py      FastAPI service
web/src/App.tsx     UI; slider panel generated from GET /api/params
web/src/api.ts      typed client
tests/verify.py     engine invariant checks -- run after touching engine.py
tests/scene.py      synthetic test scene
presets/            preset library -- files, not code; see below
run.sh / dev.sh     production from source / hot-reload dev
build.sh            compiles a distribution into build/
```

There are **no presets in the source any more.** `params.PRESETS` is gone;
`load_presets()` reads `presets/*.json` on every `schema()` call, so adding one
is dropping in a file. Named by filename, not by the `name` inside the file —
the files are the interface, and renaming one in Finder should rename the entry.
Values go through `sanitize()` like any other input, which is why the shipped
files still load with 40-44 of 52 keys: the rest fill from defaults.
`DEFAULT_PRESET` (env `FILM_GRAIN_DEFAULT_PRESET`, default `Stock`) names the
preset the client opens on; `schema()` reports it as `default_preset` only if a
file by that name actually exists, so a missing one degrades to the parameter
defaults instead of erroring. Boot and Reset share one `startingValues()` in
`App.tsx` deliberately — Reset meaning something other than "how it opened"
would be its own small bug. `build.sh`
copies the folder; a distribution without it would silently have no presets.

`APP_ENV` gates dev-only behaviour and **defaults to production** — CORS for
Vite's origin, `/docs` and `/openapi.json` are all dev-only, and a production
process with no `web/dist` raises at import rather than booting and serving
503s. `dev.sh` exports `APP_ENV=development`; `run.sh` and the distribution's
launcher export production. If you add a dev convenience, gate it on `IS_DEV`
or it ships.

Adding a parameter means adding one `Param` to `server/params.py` and reading
`p["key"]` in the engine. The UI picks it up automatically — never hand-add a
slider in `App.tsx`. Give it `choices=(...)` and it renders as a menu instead;
the value is still a number everywhere else, so nothing but the one branch in
`App.tsx` knows the difference.

## Two invariants that must not break

Both are silent killers: break either and previews still look fine while
exports are wrong.

1. **Tile independence.** No stage may depend on a statistic of the region being
   rendered — no per-tile normalisation, no global mean, no `arr.max()`. Edge
   strength normalises against the fixed `EDGE_REF`; the noise lattice is
   addressed by absolute global coordinates. Break it and exports grow seams
   that no preview will ever show.
2. **Scale invariance.** Every spatial quantity is specified in full-resolution
   pixels and multiplied by the working `scale`. The preview no longer relies
   on this (it renders at `scale = 1.0` like the export), but supersampling
   does — it renders at `scale * ss` — and so does every check in `verify.py`.
   Break it and 2× and 1× stop agreeing.

`pipenv run python tests/verify.py` checks both, plus zoom fidelity, colour
pass-through, luminance response, edge bias, the smooth-area guard, 16-bit PNG
validity, the global-grain overlay, edge softening, edge jitter and its
direction bias, edge sanding, scatter, output sharpening and the film-texture
section — 119 checks. It exits non-zero on failure.

The global-grain, edge-softening, edge-sanding, scatter and sharpening checks
exist because those stages ship at 0, so the default-parameter checks render
straight past them. Each re-runs tile independence with its stage switched on:
they all add work `pad_for` has to cover, and a kernel missing from `pad_for`
seams tiled exports along exactly its radius while every preview looks fine.
All three displacement stages — both warps and scatter — are in `pad_for` too;
they displace rather than blur, so they read pixels up to their peak travel
away and contribute to the additive term rather than the ×3 kernel sum.

One trap when adding a check here: `sanitize(None)` fills in *defaults*, not
zeros, so an override dict has to zero every other stage that could contribute
to the same measurement. The sanding check failed first time round because
`edge_jitter` defaults to 0.3 and was quietly adding its own wander to it.

## The preview is client-scaled, two-tier (changed 2026-07-31)

`/api/preview` renders **the whole frame**, never a crop. The request carries
`id`, `params`, `supersample` and `full` — no mode, no zoom, no viewport. The
browser does all the scaling.

`full` picks the fidelity, and it is the only difference between the two:

* `false` (default) — the working proxy, `PROXY_LONG_EDGE = 2400`. This is what
  every slider change triggers. **~1.25s on a 24MP source**, 5MB.
* `true` — the whole source at scale 1.0. The preview *is* the export at this
  point. **~9.5s on 24MP**, 32MB. Only ever fired by the explicit
  "Render 1:1" button; any parameter change drops back to the proxy.

Deliberately not automatic after an idle delay: it is ~8s of work, and spending
that every time a drag settles burns it on frames you are about to change.

What the client-side scaling bought:

* Zoom and pan are free. They never re-render and never hit the network. The
  wheel zooms, and two details in it are not optional: it is attached with
  `addEventListener(..., {passive: false})` rather than as React's `onWheel`,
  because **React registers wheel listeners as passive** and `preventDefault`
  is a no-op there — the React version zooms the photo and scrolls the page
  under it at once. And it is **anchored on the pointer**: the image point
  under the cursor has to still be under it afterwards, or zooming into a
  detail walks it off screen. The anchor is read from the frame's own
  `getBoundingClientRect` rather than recomputed from `center`, so it inherits
  `place()`'s clamping instead of duplicating it. Fit is a *mode*, not a
  number — it tracks the container so a resize keeps the frame visible — so
  the wheel snaps back to it within `FIT_SNAP`.
* The server-side crop grid-phase problem is gone. It used to be that below 1:1
  the read origin had to be **snapped** so `origin * scale` landed on a whole
  working pixel, because a crop starting mid-pixel resolves on a different grid
  phase than a whole-image downscale — invisible on smooth areas, a glaring
  half-pixel shift on hard edges. That is why `ZOOM_STEPS` used to be clean
  fractions. Nothing resamples on the server now, so arbitrary zoom is safe.

The honesty problem this creates, and how the UI handles it: enlarged past its
own resolution the proxy is soft, and a soft preview reads as a soft *result*.
The stage shows a `proxy` badge whenever `eff > proxy_width / source_width`, and
the panel says to Render 1:1 before judging grain. Do not remove those.

## Pipeline order matters at both ends

Two stages are placed by *position*, not by what they compute, and moving them
breaks their whole purpose:

* `global_*` (step 13) is after every mask, so it reaches the regions the masks
  protect. Fold it in earlier and it becomes just more masked grain.
* `sharpen` (step 14) is dead last, **after** the grain stages, because the
  high-frequency content it amplifies is meant to be the grain. Run it before
  them and it sharpens a clean image and leaves the grain flat — the opposite
  of the point. It is distinct from `acutance`, which is deliberately extracted
  from the *pre-grain* base so it sharpens the image *without* touching grain.
  Two sharpeners, opposite intents, and the difference is entirely where they
  sit. Measured: `sharpen 0.8` takes grain to 140% and image acutance to 133%;
  on a flat field it is a no-op to 6e-08, so it genuinely invents nothing.

`ENGINE.render_view` / `render_crop` are no longer called by `main.py` but are
still exercised by `verify.py`, which is the only place the crop and zoom
invariants are checked at all now. Do not delete them.

## Halation blue compensation, and why it runs *before* the wash (2026-08-01)

Halation adds warm light in linear light, and **adding light desaturates
whatever it lands on** — that is not a bug to tune out, it is what addition
does. A red-tinted bloom lifts a blue sky's red channel by the full glow and
its blue channel by a tenth of it, so the sky loses colour toward grey and
drifts toward purple. Reported by the user, and real: measured on an ordinary
sky, **16% of the saturation gone and a +5.8° hue swing**.

There are two regimes, and they want different answers:

* **Threshold above the sky's luma** (the default 0.72). The wash is a *local*
  rim: measured, sat 0.769 → 0.574 at 10px outside a highlight, 0.732 at 50px,
  and untouched past 90px. A uniform compensation over-corrects the 90% of the
  frame that was never damaged.
* **Threshold below the sky's luma** (`Stock`-era presets, and `Organic` at
  0.30 against a sky luma of 0.366). The sky is over the threshold, so **it
  blooms onto itself** and the loss is uniform across the frame. This is the
  case the control is for, and the one a shipped preset actually hits.

Worth saying out loud: in the second regime **raising `halation_threshold`
above the sky's luma fixes it outright** — 0.30 → 0.45 on `Organic` measures
sat 0.660 → 0.778 and hue 225.3 → 219.7, against an untouched 0.769/220.0.
Reach for the compensation when the low threshold is wanted for the look.

**Before the wash, not after** — I measured both with the identical mask:

| amount | before the wash | after the wash |
|---|---|---|
| 0.5 | +0.6% past target | +8.8% |
| 1.0 | +3.5% | **+30%, a channel driven to black** |
| 3.0 | +3.5% | pinned at fully saturated |

Compensating *before* is self-limiting because the wash eats the same share of
whatever is added, so the control has a natural brake and cannot be over-cooked.
Applied *after* there is no brake at all, and by amount 1.0 it has crushed the
minimum channel to zero — posterisation and hue break, not a correction.

Two more reasons, both structural:

* After the wash there is no way to tell blue that was *unfairly* greyed from
  blue the bloom is **supposed** to be sitting on. Re-saturating there fights
  the glow you paid for. It would need the glow field carried out of the
  halation block to know the difference. Before the wash the question never
  arises: this changes what was *recorded*, and halation then does its job to
  it — which is also the physical order, a punchier blue layer or a polariser
  rather than retouching.
* It is the cheaper place. Purely per-pixel, no kernel, so `pad_for` is
  unchanged (pinned at 58px in `verify.py`).

Two things that had to be right:

* **The glow is computed before the compensation runs**, so dialling blue
  cannot move the bloom. Pinned bit-exactly: probed on a grey field lit by a
  saturated blue source, the glow on the surrounding ring moves **0.00e+00**
  while the blue source itself moves 0.432. Without that ordering,
  `halation_blue` and `halation_threshold` would fight each other.
* **Saturation cannot fix hue.** Scaling chroma about the luma axis is a
  radial operation and by construction rotates nothing — measured, +7.1° of
  error left on the table by the amount slider alone, and `halation_blue_shift`
  at −8° takes it to +0.1°. The second slider is not decoration.

### The gate is brightness, not hue width (corrected 2026-08-01)

The first version exposed a **Blue Range** hue-width slider. That was the wrong
control, and the user found it by using it: cranking Blue Compensation made
deep blue go lurid. The mask knew *what colour* a pixel was and nothing about
whether the wash had ever reached it, so every bit of correction on an
undamaged pixel was pure overshoot.

Measured up a sky gradient away from the sun, saturation loss runs **23% at the
bright end and flat 0% below about half brightness** — halation only reaches
what is near the light. And at amount 2.0 the ungated mask took an untouched
deep sky from 0.872 saturation to **1.000**: a channel clamped to black. That is
a missing term in the mask, not a setting to avoid.

So the hue width is a constant (`_BLUE_RANGE`, 70°) and the slider is
`halation_blue_level` — how light a blue has to be before it is worth saving —
with `halation_blue_falloff` as a **separate** width, because deriving the ramp
from the knee would make moving one change the other and a sky is exactly the
broad gradient that shows up a hard switch-on. Same pattern as
`lum_low`/`shadow_falloff`, and quintic for the same reason.

**Read the brightness display-referred, and encode before taking the luma.**
`_linear_to_srgb(_luma(lin))` is cheaper and wrong — the transfer curve is
non-linear so it does not commute with a weighted sum, and it reads a deep sky
**23% brighter than it is**, putting this slider on a different scale from the
Luminance Response knees it is meant to match. Linear luma is worse still: it
crushes an ordinary sky to 0.05 and wastes the top nine tenths of the slider.

Known limit, measured: a fixed brightness gate is a *proxy* for "where the wash
reached", and in the high-threshold regime the two do not line up exactly —
0.487 display luma carries 1.8% damage and 0.519 carries 23.4%, only 0.03
apart. The exact answer is to weight the mask by the **glow field itself**,
which is already computed two lines above, is tile-independent (fixed
threshold, no statistics) and would make over-correction structurally
impossible in both regimes. Not done: it would silently change what every
existing `halation_blue` value means, and the brightness gate is what the user
asked for. One multiply if it is ever wanted.

`_BLUE_HUE` is **230°, measured in linear light** where the stage runs, not the
sRGB number. The transfer curve is per-channel and monotonic so it preserves
the hue *sector* but moves the angle inside it by 6–10°: an ordinary sky is
220° in sRGB and 230° in linear. Skies span 222° (pale) to 236° (zenith);
cyan water is 194° and purple shadow 249°, so the fixed 70° window separates
them. The mask is weighted by existing saturation on `vibrance`'s reasoning —
it must strengthen blue that is *there* and never invent it in grey, or every
neutral in the frame picks up a cast. Grey and red are left **bit-exact** at
maximum settings.

Gated on `halation > 0.01`. With no wash there is nothing to compensate and the
control would just be a blue grade, which is deferred — `verify.py` pins it at
0.00e+00.

## Scatter: diffusion without the average (added 2026-08-01)

Step 1b, in linear light beside `micro_blur`, because it is the same physical
event. A blur is diffusion as an *expectation* — average over enough photons
and deflection becomes a convolution. `scatter` resolves the deflections
individually instead: a share of the pixels are displaced onto a neighbour and
nothing anywhere is averaged. That difference is the whole feature, and it is
what the user asked for — detail destroyed, harshness kept.

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

## Film texture is drawn, never scattered (added 2026-07-31)

Step 15, dead last, after sharpening: dust, scratches, hair, light leaks.
Everything above it models what the *emulsion* does; this models what happened
to the strip of film afterwards. It is weighted by none of the image masks — a
scratch does not care what is underneath it — and every parameter ships at 0.

**Do not reimplement dust, scratches or hair by scattering objects.** A list of
speck positions is a statistic of the region: an export would split a scratch
across two tiles, or draw a different list per tile. Those three marks are each
a *threshold on a noise field addressed in global coordinates*, so every pixel
gets the same answer whichever tile asks. It also happens to look better — the
outlines are organic because the field is, where stamped sprites repeat.

**Light leaks are the exception, and the distinction is worth being precise
about.** `_leak_sites` does build a list of objects, and it is tile-independent
anyway, because the list is a function of the *count, the seed and nothing
else* — every tile builds the identical list, and so does the proxy, and so
does the export. What breaks tile independence is deriving a list from the
region being rendered: N specks per tile, or positions drawn against the tile's
own area. Neither happens here. Leaks earn the exception because a leak is not
a mark, it is a beam with a source, a direction and a length, and a field that
only knows "how far am I from the nearest border" can express none of those —
see the section below for what that cost.

How each shape is made, and the measured result at full strength:

| mark | how | coverage | geometry |
|---|---|---|---|
| dust | isotropic fine noise, two populations (dark motes, bright pinholes) | 0.62% | 1.0:1, compact |
| scratches | noise with cells ~2px wide and ~900px tall — the anisotropy *is* the scratch | 0.18% | 74:1, 1.1px wide |
| hair | level set of a smooth field: `|n − 0.5| < eps` is a curve that wanders | 0.37% | 2.0px wide |
| light leak | oriented beams anchored on the perimeter, added in **linear** light | 34% at 6 | 1.3:1 along/deep |

Every mark type is passed through `_weather()`, which is what stops the section
looking generated. A thresholded field gives every mark an identical crisp edge
and identical opacity; real debris sits at different depths, so some is in focus
and some is not, and none of it is equally dark. `_weather` blends each mark
toward a blurred copy and scales its density, both driven by fields addressed at
*mark* scale — a whole scratch shares its blur and its density rather than
fading in and out down its own length. Measured at full softening: mean edge
slope down 26-33%, while the crisp-to-soft ratio *widens* (scratches 13.8x to
18.2x) and per-mark brightness spread runs 66-87% of the mean. Both halves are
asserted: a uniform blur would pass a mean test and be exactly the artificial
result this exists to avoid.

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

**Dust Softness widens the speck's threshold band; the blur is secondary.**
Blurring a 2px speck by several times its own size does not soften it, it
erases it -- energy is conserved so the peak collapses below anything visible,
and you end up with fewer specks rather than softer ones. My first attempt read
as "softness does nothing" for exactly that reason, and the measurement was
survivorship-biased on top: only the specks that survived were left to measure.
Expanding the band symmetrically about its midpoint keeps the count and makes
each speck gradual. Measured 52% softer at 2px specks with coverage *rising*.

Three traps, all of which cost me a rebuild:

* **Read thresholds off the field's real distribution.** Value noise is heavily
  centre-weighted, so a threshold of 0.88 — which sounds extreme — selects
  **10% of the frame**. Measured quantiles: 1% above 0.943, 0.1% above 0.988,
  0.01% above 0.998. First attempt put 9.7% dust on the frame, which reads as
  weather rather than film.
* **A gating field coarser than the image is a constant, not a mask.** The hair
  sparsity field had a 900px cell, so across a frame it spanned only 0.38–0.73
  and never crossed its 0.72 gate — hair rendered as *literally nothing*, and
  which nothing depended on where in the noise plane the frame happened to sit.
  Keep gating cells well under a frame (280px now).
* **Solve level-set widths, do not pick them.** A hair's width is ~`2·eps·cell`
  pixels. At `eps = 0.0016` with a 110px working cell that is 0.35px — sub-pixel
  before supersampling halved it again, so it drew nothing.

Light leaks need the *frame* size, which is why `render()` now takes `full_hw`
and `render_supersampled` scales it by `ss` alongside `y0`/`x0`. `render_view`
passes the whole source's size, not the read window, or the leak would slide
around as you panned. `verify.py` pins that: a crop of a leaked frame matches
the same region of the full render to 2e-05.

## Two client traps worth knowing (2026-08-01)

* **`commit()` after `setValue()` applies the *previous* value.** `commit`
  reads `valuesRef`, and that ref is only refreshed during render, so calling
  the pair synchronously renders what was there before the change. Sliders
  never showed it because their `pointerup` arrives a render later and commits
  the right thing — but the `choices` menu has no second event, so picking an
  option did nothing until the control lost focus. `setValueNow` builds the
  next object and hands it to both setters. Any control that changes a value
  and expects a render in the same gesture needs it.
* **React registers wheel listeners as passive**, so `preventDefault` inside an
  `onWheel` prop is a no-op — the scroll-to-zoom handler is attached by hand
  with `{passive: false}` or the photo zooms and the page scrolls underneath it
  at the same time.

Compare and Wipe moved out of the panel onto the preview's `viewbar` alongside
the zoom controls, on the same reasoning that put zoom there: they change the
*view*, not the render, and driving a wipe across the photo from a panel on the
other side of the screen means looking away from the thing you are judging.

## The mark-count dead zone (found 2026-08-02)

`dust`, `scratches`, `hair` and `light_leak` are **counts**, and the engine
gates each on `>= 1.0` — you cannot render a third of a scratch. So any value
in **(0, 1) renders nothing at all** while reading, in the panel and in the
file, as though the section were slightly on.

Three shipped presets sat squarely in it. `Organic`, `Dreamy` and `Dreamy+1`
carried `dust 0.62`, `scratches 0.48`, `hair 0.10`, `light_leak 0.05` — 0–1
*amounts* from before these parameters became counts, never migrated. Their
entire Film Texture section had been silently inert, which is how the user
came to report light leaks as not rendering: nothing they did to the leak
sliders could matter while the count was 0.05.

Migrated to **0**, not to a count. Zero is what those presets have actually
been rendering ever since, so it is the faithful migration; rounding 0.62 up to
1 would change their look without being asked.

`verify.py` now refuses any shipped preset with a count in (0, 1). It is worth
a check rather than a comment because it is invisible from both ends — the code
looks right, the file looks deliberate, and the UI shows a number.

## Presets rescale across image sizes (added 2026-08-01)

A preset dialled in on one photo is locked to that photo's size: every spatial
parameter is a length in full-resolution pixels, so the same numbers on a
bigger frame give proportionally finer grain and tighter halation. Preset files
now carry `reference_mp`, the size they were authored at, and the client sends
it with every render; `_params_for` rescales before the engine sees anything.

**The ratio is linear, not area.** Eighteen parameters are marked
`spatial=True` and multiplied by `sqrt(current_mp / reference_mp)`. A 16MP
frame is 0.816x the *width* of a 24MP one, not 0.667x -- scaling lengths by the
megapixel ratio overshoots by the square root. `edge_jitter` is in that list
despite having no `px` unit: `_JITTER_MAX` makes it a length multiplier.

Not rescaled, on purpose: amounts and blend weights (dimensionless, per-pixel),
and mark counts (already resolved against frame area inside the engine, so
50 specks is 50 specks at any size). Leak sizes and the leak feather *are*
rescaled now that they are pixel lengths; they used to be exempt as fractions.

Measured on the same scene at 6MP and 15.4MP, both resampled to a common 900px
display width and grain isolated against a same-parameter grain-off render:
without scaling the larger frame carries **57%** of the authored grain sigma;
with it, **107%**. The residual 7% is the clump curve and the `_MIN_CELL` floor,
not a systematic error.

Two traps when measuring this:

* Resample both renders to the *same* display width. Downsampling by integer
  factors lands them at different sizes (1000px vs 1200px) and the comparison
  is meaningless.
* Isolate grain against a grain-off render using the **same rescaled**
  parameters. Using the unscaled ones puts the halation and blur rescaling into
  the residual too, which reported 148% instead of 107%.

Files without `reference_mp` -- everything authored before this -- scale by
1.0, so the behaviour is unchanged rather than guessed at. There is
deliberately **no built-in default size**: inventing one would silently change
the look of every legacy preset, and a wrong guess is worse than no scaling.
Two ways to populate it instead:

* Per preset: open a photo of the size it was dialled in on, press **Set from
  photo**, then **Save to file…**.
* All at once: `FILM_GRAIN_DEFAULT_REFERENCE_MP=24` makes every preset with no
  recorded size be treated as authored at 24MP.

## Tuning constants (all in engine.py, all calibrated by measurement)

| Constant | Value | Why |
|---|---|---|
| `EDGE_REF` | 0.06 | Fixed edge-magnitude reference. Must stay a constant, not a statistic — see invariant 1. |
| `_GNORM` | 0.55 | Noise normaliser. **The old note here claimed field std ~0.27 clipping ~3.6%; re-measured 2026-07-31 it is std ~0.45 clipping ~18%**, constant across octave counts now that `_fbm` preserves variance. The 18% is pre-existing — a single-octave field measures the same — so this row was simply wrong, not broken by a change. Lowering it flattens the distribution's tails further. |
| `_AMP_SCALE` | 0.38 | Maps the 0–100 intensity slider to amplitude; default 32 lands near 3.5% luminance sigma. Was 0.5 — recalibrated when `_fbm` started preserving variance, since the old value was silently compensating for a field running at 43% strength. |
| `_MIN_CELL` | 0.8 | Floor on lattice cell size in working pixels. Below Nyquist it is pure aliasing. |
| `_SAND_TAPS` | ±2σ, 5 taps | Tangential sanding filter. Reaches ±2σ, not ±1 — contour roughness sits at longer wavelengths than it appears to, and a ±1σ filter removed only 2% of it. Weights are normalised at use: the table sums to 0.991. |
| `_BLUE_HUE` | 230.0 | Centre of the blue-compensation window, **in linear light** where the stage runs — the sRGB number is 220. Skies span 222-236 there; cyan water 194, purple shadow 249. |
| `_BLUE_SAT_FLOOR` | 0.12 | Below this a pixel is grey and the compensation leaves it alone. Without it every neutral in the frame takes a cast — the failure `vibrance` is also written against. |
| `_SCATTER_STENCILS` | 9 entries | Scatter footprints: (name, first angle, count, locus, inner, alt). **Indexed by the parameter value, which is what a preset file stores** — renumbering silently changes every preset that used one, so append rather than insert. `verify.py` pins it against `choices` in params.py name-for-name. Every entry must keep peak travel ≤ reach, which is what `pad_for` reserves for; an L∞ "square" locus would reach 1.41× and would have to be paid for. |
| `_JITTER_MAX` | 3.0 | Peak edge displacement in full-res px at `edge_jitter` 1. Was an inline 0.6, whose *typical* displacement was 0.227px — invisible. |
| `_STEP_LO` / `_STEP_HI` | 0.030 / 0.110 | Luma-step bounds separating a real transition from fine texture, for the edge-softening mask. Fine texture measures an order of magnitude under a hard border, which is the gap that lets softening take the snap off a border and leave fabric alone. |
| `_TEX_LO` / `_TEX_HI` | 0.002 / 0.015 | Local mean-abs-deviation bounds separating "smooth" from "textured" for the smooth-area guard. Skin and clear sky sit at or below `_TEX_LO`; fabric, foliage and hair sit above `_TEX_HI`. |
| `_LEAK_PHI` | 0.618… | Golden step placing leaks around the perimeter. A low-discrepancy sequence, not a stratification, so leak *k* lands in the same place whatever the count is — raising the count must add a leak, not reshuffle the ones already on the frame. |
| `_LEAK_CORNER_BIAS` | 0.10 | Pull toward the ends of a border, as `t − bias·sin(2πt)`. **Must stay under 1/2π = 0.159** or the map folds and every leak collapses into a corner; measured slope −0.51 at 0.24. |
| `_LEAK_WARP` / `_LEAK_REACH_SAFETY` | 0.15 / 1.25 | Domain-warp peak as a fraction of a leak's reach, and the divisor on the reach cap that pays for it. The warp can carry the falloff inward, so the centre-fog cap has to cover both or the frame's centre stops being exactly zero. |
| `_LEAK_GAIN` | 2.0 | Exposure per unit of leak before `leak_strength`. Sized so a default-strength leak's core just saturates — that is what makes the core white and leaves the colour in the falloff. |
| `_MID_GREY` | 0.46 | Pivot for the (deferred) contrast section. |
| `MAX_UPLOAD_BYTES` | 30MB (imageio.py) | Input cap. JPEG/PNG only (`INPUT_FORMATS`, which also accepts `MPO` — a multi-frame JPEG; cameras emit it for burst and 3D and the file is still a .jpg). Not arbitrary — it is what keeps a full-resolution-per-keystroke preview affordable. |

`global_intensity` shares `_AMP_SCALE` with the main intensity slider but is
weighted by no mask, so the same number bites roughly 2.2× harder: measured on
flat 0.5 grey, 32 gives 7.7% luminance sigma where the masked slider gives 3.5%.
Response is exactly linear in intensity and in opacity, and the two multiply
(100 × 0.2 measures identically to 20 × 1.0). Do not "fix" the mismatch by
rescaling — the honest mapping is that this layer is unmasked; the help text
says so and points at 5–20 as the usable range.

## Things I got wrong, so you don't repeat them

* **A shoulder normalised to reach 1.0 is not a shoulder.** A region of falling
  slope mathematically cannot reach the top; forcing it turns the shoulder into
  a highlight *boost* and washes out skies. It must asymptote below white.
* **`halation_hue` changed meaning on 2026-07-31.** It was a 0-1 ramp that
  interpolated G and B against a fixed R, which spanned all of **25 degrees**
  of hue (5.3 to 30) and could never desaturate — so it read as a nudge, not a
  tint control. It is now a hue angle in degrees over the full wheel, with
  `halation_sat` alongside it so a neutral white bloom is reachable. The
  shipped presets were migrated in place: their stored 0.25 became 11deg/0.86,
  which renders the same colour to within 0.005 per channel. Any *older* file
  still holding a 0-1 value will now read as near-pure red — convert with
  `hue = rgb_to_hsv(1, 0.18 + 0.45h, 0.10 + 0.16h)`.
* **Halation must be computed in linear light.** Done in gamma-encoded space it
  reads as a painted-on glow rather than light.
* **Per-channel edge erosion causes coloured fringing.** I removed it as an
  artifact; the user likes it and calls it film vibe. It is now the
  `edge_chroma` slider (0 = neutral, 1 = full per-dye-layer speckle), not a
  hardcoded decision.
* **A smooth edge envelope traces edges too precisely** and reads as a digital
  outline. It is broken up by its own noise field so erosion is ragged.
* **Measuring "grain sigma" from the raw residual is wrong** once halation and
  acutance are on — their contribution sits in the same residual and masks the
  highlight falloff. `tests/verify.py` disables them for that measurement.
* **`edge_bias` alone does not protect smooth areas.** Its flat-area floor is
  `1 - edge_bias`, so the 0.55 default left skin and skies at 45% of full
  grain — the user reported this as skin looking jagged, and they were right.
  The edge mask only sees *micro-edges*, so a smooth gradient gets no
  protection from it at all. Fixed with `smooth_guard`, which measures local
  contrast over a medium radius: a linear gradient has almost none (blurring a
  ramp returns the ramp) while fabric and hair have plenty.
* **The octave cascade has to run coarse, not fine.** Conventional fBm
  subdivides downward, and this did too — but the base cell is already at the
  pixel grid, so every octave was instantly clamped to `_MIN_CELL` and differed
  from the previous only by seed. Measured effect of the Octaves slider on a
  real 24MP proxy: **0.02% mean pixel change, and byte-identical from 5 to
  10.** Roughness was 0.18%. Both sliders were inert, and the user reported
  exactly that. `_fbm` now takes `cell` as the *finest* scale and doubles
  upward, which is also the better physical model (clumps cluster, clusters
  mottle). It fixed the cost at the same time and far better than the octave
  cap it replaced: a coarse lattice is a handful of points, where a floored
  0.8px lattice was nearly per-pixel CPU hashing. Full-res 24MP at `octaves:
  10` went **32.6s → 4.4s**, now with the slider actually doing something.
* **`total / wsum` in an fBm holds the mean and loses the variance.** The
  octaves are decorrelated, so that normaliser leaves variance scaled by
  `sum(w²)/sum(w)²` — 43% at three octaves, 33% at ten. Every octave added
  structure and turned the grain down by the same stroke, which is most of why
  the slider felt like a no-op: you were trading amplitude for structure and
  netting roughly nothing. Rescaling the deviation by `sum(w)/sqrt(sum(w²))`
  preserves variance, so Octaves changes structure at constant strength and
  Intensity stays the only amplitude control.
* **A frequency split on top of `scatter` is redundant, and I built one before
  working that out.** The idea was a "Detail Only" control: split the image at
  the reach, scatter the fine half, put the broad half back untouched, so
  shapes hold their ground while texture comes apart. Because a
  nearest-neighbour gather is linear the maths is exact and elegant — and it
  delivers nothing, because **the stage is already frequency-selective by
  construction**. A displacement can only change a pixel by as much as the
  picture varies over the distance travelled, so structure coarser than the
  reach survives for free; that is the same self-masking that keeps skies
  clean. Worse, it does not do what the name promises even where it bites: a
  hard border is *high* frequency, so splitting at the reach puts the border in
  the scattered half and the control cannot protect it. Measured, it changed
  52% of the residual and only made the scatter weaker overall (0.0607 →
  0.0544), i.e. a second unpredictable strength knob dressed up as a
  structural one — while widening `pad_for` to 4× the reach. Cut. `scatter_radius`
  is the frequency control.
* **Noise must perturb a shape, not be one.** The light leak was a falloff
  field multiplied by a noise gate, and that is a recipe for fog: every
  boundary in it was a `smoothstep` on value noise, so nothing had an edge and
  nothing had a direction. The fix was not more noise or better noise, it was
  giving the leak an actual geometry — a source, a lean, a length, a hard side
  — and demoting the noise to a domain warp on top. The same test tells you
  which side of that line you are on: can the parameter set describe *one*
  mark, or only a texture? See the beam section for the full post-mortem.
* **A `sin` remap can fold.** `t − 0.24·sin(2πt)`, used to bias leaks toward
  the corners of their border, is not monotonic — slope −0.51 near the ends —
  so it does not bias, it collapses, and it sent a quarter of the way along a
  border to one hundredth of the way along it. Every such remap needs its
  coefficient checked against the derivative, not eyeballed.
* **"Softer" is not "blurrier", and a blur is the wrong tool for it.**
  `micro_blur` diffuses the whole frame, so it takes texture down with the
  edges: measured on a half-border/half-texture frame, `micro_blur 3.0` left
  12% of the border but only **2% of the fine texture**. That reads as out of
  focus, not as film, and it is what a user reported as "makes the photo
  blurry". `edge_soften` blends toward a blurred copy weighted by a *hard-edge*
  mask instead — 43% of the border, **93% of the texture**.
* **The softening mask cannot key on `edge`.** That mask asks "is there a
  micro-edge here", and fine texture is made of micro-edges, so weighting by it
  softened fabric almost as much as a border (first attempt: texture fell to
  42%). The discriminator has to be edge *amplitude* — a real transition steps
  a long way in luma where texture wobbles a little — hence `_STEP_LO/_STEP_HI`
  and a high-pass taken at the softening radius.
* **Softening must not be allowed to cost grain.** The edge mask and the
  smooth-area guard used to be measured from `base`, i.e. *after* micro-blur.
  Softening the picture therefore flattened the micro-edges the grain weighting
  keys on, and quietly turned the grain down with it — dial in diffusion, lose
  noise you never asked to lose. Both now measure from `lum_ref`, the untouched
  tile input, so softness and grain amount are independent. `verify.py` pins
  this at 100% for both controls.
* **A warp's amplitude is not its typical displacement.** `edge_jitter` capped
  at 0.6px, which sounds sub-pixel-by-design and reads as nothing at all: the
  noise field averages well under its own peak, so typical displacement
  measured **0.227px**, and the edge mask scales it down again from there. A
  quarter-pixel wobble survives neither the proxy render nor the browser
  downscale on top of it. `_JITTER_MAX` is 3.0 now — the bottom fifth of the
  slider still covers the old range, and 1.0 makes a straight border wander
  ±1.15px.
* **Rotating an isotropic field is a no-op.** `edge_jitter` builds its
  displacement from two independent noise channels read as `(dx, dy)`, and
  measured, that is isotropic: every 45° sector takes 12–13% of displacements
  at mean magnitudes within 2% of each other. So "rotate the jitter 45°"
  cannot do anything — you get a statistically identical field. An angle only
  becomes meaningful once the field is *squeezed* onto an axis first, which is
  what `jitter_aniso` does; `jitter_angle` is inert without it, and `verify.py`
  pins that at a bit-exact 0.00e+00.
* **The one real anisotropy is at the tails.** `(dx, dy)` pairs fill a square,
  not a disc, so peak travel is √2 further on the diagonals (1.40 vs 1.00)
  even though the mean is even. Rotating 45° would only move that bulge from
  the diagonals onto the axes.
* **A displacement parallel to an edge cannot move it.** Half an isotropic
  field's travel is therefore invisible, which is part of why the old 0.6px
  cap read as nothing. It also gives the sharpest possible test of the
  direction control: fully biased horizontal must leave a horizontal border at
  exactly 0.000px of wander.
* **Sanding is jitter's counterpart, not more of it.** Jitter roughens a
  border; sanding polishes that roughness back off, which means smoothing
  *along* the contour and never across it — each pixel averaged with its
  neighbours in the isophote tangent direction, so burrs average out while the
  transition stays as sharp as it was. I first built it as a fine-grit *warp*
  that frayed the edge further, which is the opposite of the ask.
* **Contour roughness lives at longer wavelengths than it looks.** Measured on
  a jittered border, only 8% of the contour's positional energy is below 8px
  and 92% is above — so a tangential filter reaching ±1σ barely touched it
  (2% of the jaggedness at fine grit). Taps run to ±2σ for that reason. It is
  also why selectivity is inherently limited: roughness and wander are close in
  frequency, so removing 34% of the jaggedness costs 20% of the wander.
* **A straight-line tangential filter runs off a wandering contour.** One wide
  pass cuts across the very edge it is following and throws away sharpness. The
  filter is applied as up to three short passes with the direction recomputed
  each time, which re-aims along the curve. The gain is modest — matched at 32%
  jaggedness removed, 81% of the wander and 73% of the sharpness kept against
  79% and 71% — but it also spreads the response evenly across the grit range,
  which matters more for a fine-tuning control.
* **Truncated gaussian tap weights do not sum to 1.** The five-tap table sums
  to 0.991, so an unnormalised filter darkened every sanded edge by ~1%.
  Normalise from the weights actually used, not from the table's intent.
* **Measure a warp by where the edge went, not by pixel delta.** Displacing an
  edge sub-pixel produces a big delta right at the border and almost none
  anywhere else, so a mean-delta test passes comfortably on a displacement far
  too small to see — which is how 0.227px shipped. `verify.py` measures
  sub-pixel border position by interpolating the 50% crossing instead.
* **Three client bugs made "open another photo" look broken**, none of them in
  the engine or the API — the server returns the right image per id, verified.
  (1) The image `<input type=file>` never cleared `value`, so re-picking the
  *same* file fired no change event and did nothing at all. (2) `onFile` swapped
  `meta` without clearing `previewUrl`/`sourceUrl`, and the stage sizes every
  layer from `meta` — so the previous photo was stretched to the new one's
  dimensions until the first render landed, and stayed there forever if that
  render failed. Clear them on success only: a rejected file must not wipe the
  session. (3) The source object URL shared the 4-entry preview LRU, so after
  four edits it was evicted and revoked while an `<img>` still referenced it,
  blanking before/after mid-session. It has its own lifecycle now.
* **`MPO` is a JPEG and must be accepted.** Pillow reports multi-frame JPEGs
  (burst, stereo) as format `MPO`, so a plain camera `.jpg` was being turned
  away with "MPO is not supported". It is in `INPUT_FORMATS` now and frame 0 --
  the photograph -- is read explicitly. The rejection message lists
  `INPUT_FORMAT_NAMES` instead, which deliberately omits MPO: nobody thinks of
  their file as "an MPO", and naming it would imply it needs converting.
* **The parameter defaults are not neutral, and supersampling is not the
  identity.** "Show me the untouched photo" needs both facts. Defaults ship
  intensity 32, halation 0.35 and micro-blur 0.45, so they alter the image;
  `params.NEUTRAL_ZERO` lists every parameter that makes a stage *run* (sizes,
  radii and seeds are excluded on purpose, so switching a section back on
  returns what you had dialled in). And even with all of those at zero, a
  render at 2x came back **1.0e-01** softer than its input, because a bicubic
  upsample followed by a box downsample rings and softens hard edges.
  `render_image` short-circuits on `is_neutral(p)` and hands the input straight
  back, so Original is bit-exact at any quality. `verify.py` asserts 0.00e+00
  at 1x, 2x and 3x -- almost-original is the whole failure here.
* **Do not try to protect skin with a luminance range.** Skin sits at 30–60%
  luma, exactly where grain is meant to peak, so excluding that band kills
  grain everywhere. Texture, not brightness, is the discriminator.
* **Band edges and transition widths must be separate controls.** Deriving the
  ramp width from the knee position (`smoothstep(0, lum_low, …)`) forces the
  fade to start at pure black, so moving the knee also changes the softness.
  That is what makes the transition feel artificial.
* Pillow **cannot write 16-bit RGB PNG** (only single-band I;16) and cannot read
  it either — it silently truncates to 8-bit, so a PIL roundtrip is not a valid
  test of the encoder. `tests/verify.py` decodes the chunks directly; `sips` is
  a good independent second opinion.

## Environment gotchas (this machine)

* **Python 3.13, not 3.14.** PyTorch has no 3.14 wheels. `pipenv` with
  `PIPENV_VENV_IN_PROJECT=1`; the venv is `.venv/`.
* **Homebrew and Laravel Herd sit ahead of nvm in `PATH`**, so `nvm use` alone
  does not switch `node` — it still resolves to Homebrew's v26. Prepend
  explicitly: `export PATH="$HOME/.nvm/versions/node/v24.15.0/bin:$PATH"`.
  `dev.sh` already does this.
* **npm blocks postinstall scripts by default here.** A fresh `npm install`
  needs `npm approve-scripts esbuild` or vite will not build.
* Torch runs on **Apple MPS**. 64-bit integer ops are poorly supported there,
  which is why the noise hash is computed in uint64 on the CPU over the (much
  smaller) lattice rather than per-pixel on device.
* Background a long-running server with the tool's own background mode; a
  trailing `&` inside a single bash call does not always survive, and a stale
  instance will hold port 8000 and make the next start fail to bind. **If you
  start a server for testing, stop it before handing back** — otherwise the
  user's own `./dev.sh` cannot bind. Both scripts now preflight the port and
  name the offending PID rather than dying on a bare `Errno 48`, and both
  accept `PORT=8001`.

## Measured performance (Apple MPS, 24MP source, 2× supersample)

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

Tile size for the preview is 1536, not the export's 1024: per-tile overlap is
fixed padding, so wider tiles amortise it (~5% better at the default halation
radius, ~12% at the widest). Past 2048 it turns around as tensors stop fitting.

The `/api/source` PNG is encoded once per upload and cached on the `Upload`
(18ms → 1.2ms on repeat). The untouched image never changes, so re-encoding a
full-resolution PNG on every parameter change was pure waste.

## State / not done

* **Input is JPEG/PNG only, 30MB max** (`imageio.INPUT_FORMATS`,
  `MAX_UPLOAD_BYTES`). TIFF was dropped and RAW was never implemented (it needs
  `rawpy`/LibRaw).
* **The neural pipeline (Approach B) is not started** and cannot be until a
  paired film-scan/digital dataset exists. The `/api/preview` and `/api/export`
  endpoints are shaped so a model could slot in behind them later.
* Uploads and export jobs are held **in process memory** (12 uploads max, LRU).
  Fine for local single-user use; would need real storage to deploy.
* No auth, no rate limiting — it binds to 127.0.0.1 only.
