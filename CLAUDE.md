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

**In scope as of 2026-08-04:** a `Colour Grading` section at the very top of the
pipeline — 3D LUTs plus temperature, shadows, highlights and a two-way clarity.
Requested outright, so the "colour grading is deferred" note below no longer
covers this one section. Everything in it still ships at 0, so the colour
pass-through holds with nothing selected. See its own section further down.

**Deferred — the user has a separate project planned:** colour grading
*elsewhere in the pipeline*. `vibrance` and `brightness` were added on request
2026-07-31 and
ship at 0 like the rest, so the pass-through still holds. The engine implements tone curves, contrast, toe/shoulder, split
toning, highlight desaturation and base fog, and exposes them as sliders, but
they **ship neutral (0)** so the pipeline is a colour pass-through. Do not tune
them or fold them into presets without asking. The `Tone Response` group is
still the deferred one; `Colour Grading` is not.

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
server/lut.py       .cube 3D LUT parsing + registry (folder and uploads)
server/main.py      FastAPI service
web/src/App.tsx     UI; slider panel generated from GET /api/params
web/src/api.ts      typed client
tests/verify.py     engine invariant checks -- run after touching engine.py
tests/scene.py      synthetic test scene
presets/            preset library -- files, not code; see below
luts/               3D LUTs -- files, not code; same idea as presets/
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
validity, the global-grain overlay with its smoothing, its chroma and its
independently-sized-clump construction, anti-aliasing, the pre-blur, edge
softening, edge jitter and its direction bias, edge sanding, scatter, output
sharpening, the master opacity cross-fade, the colour-grading section with its
3D LUT lookup and `.cube` parsing, and the film-texture section — 205
checks. It exits non-zero on failure.

Seventeen of those are the 2026-08-04 performance work, and they are a different
*kind* of check worth knowing about: rather than measuring a property, they assert
**bit-equality against a reference implementation of the code that was replaced**
(`lattice_ref`, `span_ref`, `varcell_ref`, all inline in `verify.py`). A faster
rewrite of a noise generator is only correct if it changes nothing, and "the
render still looks like grain" cannot tell you that. Keep the references when
touching those functions — deleting them turns the checks into tautologies.

The Global Grain cache checks are the same idea from the other direction: a stale
cache hit renders a perfectly plausible texture, so they test *which parameters
miss* via a hit counter (`GrainEngine.gg_hits` / `gg_misses`) rather than only
that the output is right.

The global-grain, colour-grading, anti-aliasing, pre-blur, edge-softening,
edge-sanding,
scatter and sharpening checks exist because those stages ship at 0, so the
default-parameter checks render straight past them. Each re-runs tile independence with its stage switched on:
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
  every slider change triggers. At *parameter defaults* it is a few hundred ms;
  what it actually costs depends almost entirely on the preset — see the
  performance section, where `Stock` measures 8.8× defaults.
* `true` — the whole source at scale 1.0. The preview *is* the export at this
  point. Only ever fired by the explicit "Render 1:1" button; any parameter
  change drops back to the proxy.

Payload is a **JPEG q95 4:4:4** (`imageio.encode_preview`), not the PNG it used
to be: grain defeats PNG's predictor, so a level-1 PNG of a 2400px proxy measured
10.4MB and 108ms of zlib against 3.4MB and 24ms for the JPEG. 4:4:4 is not
optional — the default 4:2:0 would average away the chroma grain.

Deliberately not automatic after an idle delay: it is ~8s of work, and spending
that every time a drag settles burns it on frames you are about to change.

### Export can write either tier (added 2026-08-02)

`/api/export` takes `scale`: `"full"` (default, unchanged — the source at 1.0,
tile 1024) or `"preview"`, which renders `up.proxy` at `up.proxy_scale` with
`tile=1536`, i.e. the *identical call* `/api/preview` makes. Verified end to
end: a preview render and a preview-scale png8 export of the same parameters
are **bit-identical, max abs diff 0**. Keep it that way — if the two calls ever
drift apart, "export what I am looking at" quietly stops being true and there
is nothing on screen to show it.

The user asked for this because the 1:1 render is *unpredictable from the
preview*, and that is not a bug to fix — it is invariant 2 working. Every
spatial parameter is a length in full-resolution pixels times the working
`scale`, so scale invariance promises the two tiers agree about *the picture*
and promises nothing about grain per output pixel. Two places they diverge, and
both bite hardest exactly where the user is looking:

* **The proxy cannot resolve its own finest structure.** The default 1.6px
  clump is 0.64px on a 24MP frame's 0.4x proxy — sub-pixel, so it renders as
  something smoother than the 1:1 version of the same numbers.
* **`_MIN_CELL` floors what is left.** At 0.4x that same clump asks for a
  0.64px lattice and gets 0.8px, so the proxy's finest grain is 25% coarser
  *relative to the picture* than the export's. Finer `grain_size` settings
  diverge further; the floor does not move.

So this is a **look**, not a resolution. Downscaling a full export to 2400px
does not reproduce it — that grain was drawn on the source's grid and then
averaged away, which is a different operation from drawing it on the proxy's.

Two details in the implementation:

* Preview-scale filenames carry the long edge (`plate_grain_2400px.jpg`), and
  only when `proxy_scale < 0.999`. Two files from one photo differing only in
  resolution are indistinguishable in a folder, and the small one is the
  surprising one. On a sub-2400px source `up.proxy is up.arr`, so both options
  render the same pixels and the tag would be a lie.
* `_params_for` still rescales against the **full** image's megapixels, not the
  proxy's. A preset's `reference_mp` is about the photograph, not about which
  tier is being written, and doing it the other way would make a preview-scale
  export disagree with the preview it is meant to reproduce.

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
  the wheel locks to it within `FIT_SNAP` of the fit value, on either side, and
  is no longer floored at Fit going the other way -- see the wheel-zoom section
  below for a real bug this uncovered.
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

Six stages are placed by *position*, not by what they compute, and moving them
breaks their whole purpose:

* `Colour Grading` (step -1) is above everything, `pre_blur` included. Every
  stage below it models an emulsion; this is the decision about what the
  photograph *is* before any of that runs. Put it after the film stages and it
  grades grain, halation and dust along with the picture, and a LUT built to be
  fed a photograph is fed a rendered negative instead. Within the block the LUT
  is last, after the four adjustments, because the adjustments exist to hand the
  LUT the picture it was meant to read.
* `pre_blur` (step 0) is before `lum_ref` is taken, which is the only thing
  separating it from `micro_blur` — same kernel, same linear light. See the
  section below.
* `scatter` (step 1) is before `micro_blur` (step 1b), and swapping the two makes
  the pair come out *harder* on borders than the blur alone — scatter drops a
  hard step back into a blurred gradient. See the scatter section for the table.
* `aa_strength` (step 1c) is in the optical block and, crucially, *before the
  masks are measured* — otherwise the grain keeps keying on the jaggies the
  stage just removed. See the anti-aliasing section.
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

`master_opacity` is later still and is not on that list, because it is not in
`render()` at all — it is applied in `render_supersampled` after the pool, and
it has to be. See its own section for why blending one level down would make
"no effect" cost sharpness.

`ENGINE.render_view` / `render_crop` are no longer called by `main.py` but are
still exercised by `verify.py`, which is the only place the crop and zoom
invariants are checked at all now. Do not delete them.

## `pre_blur` is the same kernel as `micro_blur` and a different stage (added 2026-08-02)

Step 0, ahead of `pre_sharpen` and of everything else, in linear light. It is a
plain separable gaussian on the source, one radius in full-resolution pixels,
shipping at 0. Asking "why not just raise `micro_blur`?" is the right question,
and the answer is entirely positional — **`pre_blur` runs before `lum_ref`**:

* Every mask downstream is measured from `lum_ref`, so the edge mask, the
  hard-edge step mask and the smooth-area guard all read the *softened* frame.
  `micro_blur` is deliberately excluded from that (`verify.py` pins it at 100%
  of unblurred grain) so that dialling in diffusion cannot quietly cost noise
  you never asked to lose. Here the coupling is the feature: soften the source
  and the grain follows the softer edges and backs off where detail has gone.
  Measured on the half-border/half-texture plate, at 3px: `micro_blur` holds
  grain at 100%, `pre_blur` takes it to **20%**.
* It runs before `pre_sharpen`, so a broad radius here against a tight one
  there is a detail-killing pair the pipeline could not otherwise express —
  soften everything, then put the bite back at one chosen scale. The other
  order would just throw the sharpening away, which is what the ordering check
  tests: `pre_sharpen 8 @ 3px` on top of the pre-blur measures **482%** of the
  pre-blur's own edge slope, so the sharpen demonstrably sees the blurred
  image rather than being wiped by it.

**In linear light, and that is not a formality here.** Averaging gamma-encoded
values holds the *encoded* mean, which is a fraction of the light it stands for,
so every edge the kernel crosses comes out darker than the light that made it.
Measured on a black/white border with a 4px radius, mean linear luminance in the
transition band: **0.500 in linear, ~0.21 encoded.** `verify.py` asserts the
0.500. The transfer round trip is gated on the radius so it costs nothing when
the stage is off.

It is **not** `edge_soften` and does not pretend to be: `edge_soften` exists
because a global blur takes texture down with the edges (2% of fine texture
survives at 3px against 93% for softening). `pre_blur` keeps 3% of the texture,
by design — the whole point is destroying detail, and the user has asked for
detail destruction as an in-scope goal. Reach for `edge_soften` to take the snap
off borders and leave fabric; reach for this to take a digital-sharp source down
before the emulsion goes on.

`pad_for` adds `pre_blur + micro_blur`, **summed rather than the widest
winning**: micro-blur reads pixels the pre-blur has already spread, so the two
are kernels in series and their reaches genuinely add.

Cost on a 6MP render at 2×: **0.69s off, 0.75s at 2px, 0.80s at 8px**, with
`pad_for` 108 → 114 → 132px. A separable gaussian plus one transfer round trip
is cheap; the padding is what you actually pay for at 8px.

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

Step 1, in linear light and **ahead of `micro_blur`**, because it is the same
physical event. A blur is diffusion as an *expectation* — average over enough
photons and deflection becomes a convolution. `scatter` resolves the deflections
individually instead: a share of the pixels are displaced onto a neighbour and
nothing anywhere is averaged. That difference is the whole feature, and it is
what the user asked for — detail destroyed, harshness kept.

### The two run scatter-first, and the old order was undoing itself (2026-08-03)

Changed on request, along with moving `micro_blur` to the bottom of its panel
section (then "Optical", now merged into Edge Destruction — see below) so the
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

## `master_opacity` lives outside `render()`, and it has to (added 2026-08-03)

A master cross-fade of the finished frame back over the untouched source.
`Output` group, bottom of the panel, default **1.0** — which makes it the one
parameter whose neutral value is not zero, and therefore the one that must stay
**out of `NEUTRAL_ZERO`**. Put it in and `is_neutral` could never be true,
because the list is checked for values at zero.

**It is applied in `render_supersampled`, after the `avg_pool`, and nowhere
else.** That is not tidiness, it is the only place it can be correct. Inside
`render()` at ss > 1 there is no untouched input to blend against — what that
method receives is already a bicubic upsample, so blending there and pooling
down makes opacity 0 return the up-then-down round trip, which `is_neutral`
already documents as **1.0e-01 softer** than the source on hard edges. "No
effect" would quietly cost sharpness. `verify.py` pins 0.00e+00 against the
source at supersample 1, 2 *and* 3 for exactly this.

Sitting at that seam also means every entry point inherits it — `render_image`
for the export, `render_view` for the preview — so the two cannot disagree
about what half strength looks like. And it is per-pixel against the tile's own
input, so `pad_for` is untouched.

**Blended display-referred, not in linear.** The reasoning that forces
`pre_blur` and halation into linear light does not carry: this is a
compositing control — "how much of the edit do I keep" — not a physical average
of light. The two are genuinely different, and not where you would guess.
Measured on a grained frame at half strength: mean deviation 5.4e-04 and
overall brightness within +0.05%, but a **worst case of 0.146** on individual
pixels, concentrated in the shadows where the transfer curve is steepest.

That concentration is the argument for encoded. Blending in the space the eye
reads makes the slider linear in *visible* deviation, so 0.5 is half the grain
everywhere. In linear the same 0.5 takes more than half out of the shadows and
less out of the highlights — an opacity control that changes the look's balance
as you dial it back, which is not what an opacity control is for.

Two smaller things:

* **Opacity 0 short-circuits before the render**, so dragging toward zero gets
  cheaper rather than paying for a full render to discard.
* **The section's mute button is correct even though it looks inverted.**
  Muting sets a group to its neutral value, which here is 1.0 — so muting
  Output removes the *dial-back*, restoring full strength. That reads oddly
  until you notice the section's contribution *is* the dial-back, and that
  every other section's mute is likewise a no-op when it is already at its
  no-op value.

Not to be confused with `global_opacity`, which mixes the Global Grain layer
alone. The names are close and the help text says so outright.

## Value noise is a quilt, and that is why Global Grain looked pixelated (2026-08-03)

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
it lives in the new Anti Aliasing section rather than under Global Grain — same
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

## The Colour section is gone, and Global Grain grew a chroma slider (2026-08-03)

Both on request. The section merge is pure UI — group names live only in
`params.py` and the client generates the panel from them, so it is five `group`
strings and one line off `GROUPS`. Each parameter went to the section that owns
its *mechanism* rather than all five landing in one place:

| was | now | why |
|---|---|---|
| `chroma_grain`, `seed` | Grain Structure | properties of the grain field itself |
| `edge_chroma` | Edge Destruction | it modulates `edge_erosion`, and does nothing without it |
| `warm_highlights`, `cool_shadows` | Tone Response | colour grading, deferred, ships at 0 — the same as everything already in there |

That is a judgement call on top of what was asked: "Colour" was a grab-bag of
three unrelated jobs, and putting the fringing slider directly under the erosion
slider it modifies is more discoverable than the merge alone would be. One
`group` string each to move if it reads wrong.

## The panel order, again (2026-08-04)

A second reorg on top of the Colour merge above, this time touching `GROUPS`
only — no parameter changed section. `Optical` is gone the same way `Color`
went: its six params (`scatter*`, `micro_blur`) took on group `"Edge
Destruction"`, so the mechanism they share with jitter and sanding — tearing
detail apart before grain goes on — now lives under one heading instead of two.
Nothing else about them changed; the pipeline still runs scatter and micro-blur
where step 1/1b says, and `verify.py`'s per-stage checks key on parameter values,
not on group names, so none of them needed touching.

`GROUPS` itself, before and after:

```
before                          after
------                           -----
Pre Blur                        Pre Blur
Pre Sharpen                     Pre Sharpen
Grain Structure                 Grain Structure
Luminance Response              Edge Destruction   (+ former Optical)
Edge Destruction                Anti Aliasing
Halation                        Global Grain
Optical                         Sharpening
Anti Aliasing                   Luminance Response
Tone Response                   Halation
Global Grain                    Tone Response
Sharpening                      Film Texture
Film Texture                    Output
Output
```

Read as one story rather than twelve independent moves: everything that tears
the image apart at the pixel and edge level -- scatter, micro-blur, jitter,
sanding, then the anti-alias pass that cleans up stair-stepping, then the two
overlay layers that ride on top of the result (Global Grain, Sharpening) -- now
runs as one uninterrupted block. Luminance Response and Halation, both about how
light behaves rather than how detail is destroyed, sit together right after it,
directly ahead of Tone Response. Grouping by request rather than by re-deriving
a rationale from scratch — the four moves were independent asks and this is
simply where they land in combination.

Same job — decorrelate the three channels so the layer carries colour speckle
instead of pure luminance noise — and deliberately a different construction. The
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

## Anti-aliasing: filter along the contour, never across it (added 2026-08-03)

Step 1c, in the optical block beside micro-blur and scatter, in linear light.
Ships at 0. The UI section has moved twice since it was added — first between
Optical and Tone Response (Optical/Colour before that merge), now right after
Edge Destruction (2026-08-04, on request, alongside the panel reorg below). The
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

## Colour Grading: a LUT is a *resource*, not a parameter (added 2026-08-04)

Requested: a section at the top of the pipeline that applies a 3D LUT from
`luts/` or from a file, with temperature, shadow, highlight and two-way clarity
sliders **before** the LUT, and cheap enough not to stress the main pipeline.
Step -1 in `render()`, above `pre_blur`. Everything ships at 0.

**The structural decision, and the one that shapes everything else: the LUT does
not live in `params.py`.** Every other control the engine takes is a float with a
range, so it can be sanitised, clamped, rescaled for a different image size and
stored in a preset file as a value. A LUT is identified by *name* and its content
is a table. So it travels beside the parameters — `body["lut"]` next to
`reference_mp`, and a `lut` sibling key in a preset file — and `main._params_for`
attaches the resolved object as `p["lut"]` after `sanitize` and `rescale`, both of
which only touch keys that are in `PARAMS` and so leave it alone.

The obvious alternative needed no new plumbing at all: a `choices` menu indexed
into the folder listing. It is wrong for exactly the reason `_SCATTER_STENCILS`
documents, and worse here — that list is fixed in code, whereas `luts/` is
user-mutable *by design*, the same way `presets/` is. A preset stores the index,
so dropping one more `.cube` in the folder silently renumbers it and changes the
look of every preset that named one. Names it is.

**`lut_amount` *is* a parameter, and it is in `NEUTRAL_ZERO`.** That pair is what
keeps the Original button honest. `params.is_neutral` decides whether
`render_image` short-circuits, and it works from the numbers alone — it cannot see
the LUT. So:

* Zeroing the mix switches the LUT off as completely as unselecting it would,
  which is why the *name* stays out of `NEUTRAL_ZERO`: same reasoning that keeps
  sizes, radii and seeds out of it, so the section remembers what it had.
* **A mix above zero with no resolvable LUT would be a silent bug**, not a no-op:
  `is_neutral` would be false, the render would run, and at supersample 2 the
  bicubic-up/box-down round trip comes back a measured 1.0e-01 softer than the
  source. `_params_for` therefore forces `lut_amount = 0` whenever `lut.get`
  returns nothing, so the gate in the engine and `is_neutral` can never disagree.
  `verify.py` pins both halves.

An unresolvable name is deliberately **not** an error — a preset can name a
`.cube` that has since been renamed, or an upload from a previous run (those live
in process memory and do not survive a restart). The picker keeps the name as a
"— missing" entry with a hint rather than resetting itself to None, because
silently showing None makes it look like the preset never had a LUT.

### The four things that had to be right about the lookup

* **`align_corners=True`.** A LUT's first and last samples *are* input 0 and
  input 1, not the centres of edge cells. The default reads the whole table half
  a cell off — a small, uniform, entirely wrong shift that looks like the LUT
  being slightly wrong rather than like a bug.
* **The axis order.** `.cube` says red varies fastest, so a C-order reshape gives
  `table[b][g][r]`; permuted to `[c][b][g][r]` that puts red on `grid_sample`'s
  `W`, green on `H`, blue on `D`, which is why the sampling grid is just the
  image's own channels in order. Get this backwards and every *symmetric* LUT
  still looks fine while every real one is channel-swapped.
* **Both of the above are pinned by construction rather than by eyeball.**
  `verify.py` builds two exactly-linear 8-cubes — an identity and one that
  rotates the channels — and trilinear interpolation of a linear function is
  exact, so the check is an *equality* (2.4e-07) rather than a judgement. The
  rotation catches a transposed axis; the identity catches the alignment.
* **`F.grid_sample` in 3D works on MPS**, checked before building on it. One call,
  trilinear, so a 35-cube and a 64-cube cost the same and neither shows up against
  the stages below. The alternative — gathering eight corners by flat index —
  needs int64 index tensors MPS handles badly and eight full-frame gathers of
  working memory.

### Why each adjustment is where it is

* **Temperature in linear light.** A white balance is a change of *illuminant*, so
  it multiplies light, and gamma-encoded values are not light — done encoded, the
  same gain moves the shadows much further than the highlights, which is what
  makes a naive temperature slider read as a tint laid over the picture. Same
  argument as `pre_blur`'s, and gated the same way so the transfer round trip
  costs nothing at 0. The gain vector is normalised by its own luma, so the
  control is colour-only: measured, luminance holds to within 1% across the slider.
* **Shadows and highlights display-referred, and clip-free by construction.** The
  lift is a share of the headroom that is *actually there* — toward white going
  up, toward zero coming down — so for any setting the map is affine with positive
  slope and both endpoints inside 0..1. No channel can leave the cube and none can
  cross another, which is what would break a hue. `verify.py` sweeps eight
  settings and pins the worst excursion at 0.00e+00. One luma, taken before either
  runs, feeds both masks: recomputing between them would make lifting the shadows
  change what the highlight control thinks a highlight is, and the two sliders
  would pull on each other — the same independence `lum_ref` buys the grain masks.
* **`_GRADE_TONE_MAX` is 0.35, not 1.0, and that is not a taste tweak.** At 1.0 a
  setting of +1 takes a black pixel to *pure white*, so the whole top of the
  slider is unusable and the useful range is squeezed into its first tenth.
  Measured on a real photograph (mean luma 0.21), Shadows at only **+0.5 took the
  frame's mean from 0.19 to 0.53** — that is a different exposure, not a shadow
  lift. Caught by rendering the actual photo through the actual API, not by
  reading the code. Same lesson as `_JITTER_MAX` from the other direction: the
  whole range has to be usable.
* **Clarity is asymmetric on purpose.** Positive gets `_GRADE_CLARITY_GAIN` (1.6);
  negative is pinned at exactly 1.0, because at gain 1 a setting of −1 subtracts
  precisely the band it measured — the local contrast is *gone*. Past that it does
  not keep flattening, it **inverts**: dark halos on the light side of every edge,
  an artifact rather than a look. `verify.py` measures the band's correlation with
  the source at −1 and fails on a negative number. Measured ladder: −1 → 5% of the
  band, −0.5 → 52%, +0.5 → 177%, +1 → 255%.
* **Clarity runs on luminance, which is both cheaper and better.** The signed
  detail goes to all three channels equally, so the channel *differences* — which
  is what hue is — come through untouched (pinned at 2.4e-07), a saturated area
  cannot be pushed out of gamut by a structure control, and it is one blur instead
  of three.

### Cost, and the one term in `pad_for`

Four of the five stages are per-pixel with no kernel and no neighbourhood, so
they reserve **nothing**. Clarity's high-pass is the only kernel in the section
and it is a real reach even though the stage runs first: a tile that cannot see
far enough measures a different band at its own edge, and that difference then
propagates through everything below it. `verify.py` pins both halves — that
`pad_for` grows by 3× the clarity radius, and that it is *unchanged* with
temperature, tone and a LUT all on.

Measured on a 6MP render at 2×, best of 3 in fresh processes (MPS run-to-run
variance here is ±1s on larger frames, so single-shot numbers are worthless):

| | time | pad_for |
|---|---|---|
| section off | 0.67s | 108px |
| temperature / tone / a LUT at mix 1 | 0.67–0.73s, inside variance | 108px |
| clarity at the default 14px | 0.75s | 150px |
| clarity at 40px | 0.88s | 228px |
| all of it | 0.82s | 150px |

### Two things outside the section that had to change with it

* **`build.sh` copies `luts/`.** It already had this exact bug documented for
  `presets/` — a distribution without the folder has an empty LUT menu and a
  preset that names a `.cube` quietly grades nothing.
* **Editing a control in a muted section now switches that section on.** Found in
  a real browser, not by inspection: on a fresh load *every* section is muted (see
  the muted-on-boot section), so picking a LUT left the section's switch reading
  "off" while the LUT rendered — and a mute/un-mute round trip then reverted the
  mix to the snapshot `toggleGroup` took at mute time, measured going straight
  back to 0. `keptFor`/`liveFor` in `App.tsx` restore the section's kept values
  and lay the edit on top, which is exactly what clicking its own ● does. This is
  general, not LUT-specific: it was latent for every slider in the app the moment
  boot started muting everything, and the new section is simply where it is hit
  first. The pair is split into a pure half and a side-effecting half because a
  `setMuted` call inside a `setValues` updater would run twice under StrictMode.

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

### Wheel zoom-out is no longer floored at Fit, and Fit itself keeps a margin (2026-08-04)

Two independent requests landed on the same code at once. **Fit reserves
`FIT_PADDING` (30px, screen pixels) on every side now**, mount or no mount --
before it sized the image to the exact pane, so the photo butted against the
panel edge with nothing to judge it against. It folds into the same `inset`
the mount's own room reservation uses, so a framed *and* fit image gets both
allowances at once rather than either one clobbering the other.

**The wheel used to bottom out at Fit and hand off to the - button for
anything smaller** -- deliberate at the time (see the `FIT_SNAP` doc comment's
history), but reported back as wrong: a continuous gesture should not need a
different control partway through it. `lo` is `ZOOM_STEPS[0]` now, matching the
button's own floor, and zooming in from there climbs back past Fit and on up
without a floor either.

**That surfaced a real bug in the Fit-snap band, not just a missing feature.**
`FIT_SNAP` decides how close to Fit counts as "close enough to lock to Fit
mode", and the check used to be one-sided (`next <= fit * (1 + snap)`) because
scrolling out could never go far enough below fit for the distinction to
matter -- the floor caught it first. Remove the floor and that one-sided check
means *every* zoomed-out value satisfies "at or below fit", so the wheel would
lock to Fit on the first tick past it and never come back out. Fixed by
checking both sides: `abs(next - fit) <= fit * FIT_SNAP`.

That fix alone was not enough, and finding out why is the more interesting
part. `zoom` displays as `null` (Fit) for the whole time a continuous scroll
sits inside the band, and the next wheel tick used to compute its step from
`eff` -- which, displaying Fit, reports exactly `fitZoom` regardless of *how
far* into the band the gesture actually was. A slow scroll advancing by less
than the band's own width per tick therefore recomputed from `fitZoom` every
single time, landed back inside the band every single time, and never
escaped: **the gesture was stuck exactly at Fit.** Measured with synthetic
wheel events sized to a plausible trackpad tick (2% zoom change per event
against a 2% `FIT_SNAP` band): every tick relocked, and 25 ticks in a row
moved the display 0.00 percentage points.

The fix is `wheelContRef`, a ref that remembers the true, unsnapped position
through a Fit-locked stretch, independent of what is on screen. Each tick reads
its starting point from that ref rather than from `eff` whenever the display is
currently `null`, so the *next* computation continues from where the gesture
really is rather than from the band's centre; whenever the ref holds nothing
(nothing in flight) or the display is a concrete number (no ambiguity to
begin with), `eff` is trusted directly. The ref is deliberately cleared at both
`setZoom(null)` call sites that are *not* this handler's own snap decision --
the Fit button and the new-image reset -- so a fresh "go to Fit" never
inherits a stale excursion left over from a previous scroll. Re-measured after
the fix: the same 2%-per-tick gesture takes exactly one tick to lock to Fit and
the very next tick to leave it, in both directions, and a moderate zoom-out
followed by zooming back in climbs straight through Fit and on past it rather
than sticking.

Verified end to end with a real running instance (headless Chrome driven over
CDP, synthetic wheel events dispatched on the actual `.pane` element) rather
than by inspection alone -- this is exactly the kind of interaction bug that
looks fine in a diff and only shows up when something actually scrolls.

### The mount is a view control, and its width is in *screen* pixels (2026-08-03)

`Frame` on the `viewbar`: an off-white board around the photo plus a drop
shadow, with a width slider that appears only when it is on. Requested, and it
earns its place on the same reasoning as the wipe — a photograph butted straight
against a dark panel reads darker and flatter than it is, and the edge of the
frame stops being visible at all where the picture goes to black. It is **not** a
parameter: nothing about it reaches the engine, the schema or an export.

Four things in it are not free choices:

* **Screen pixels, not source pixels.** Every spatial parameter the engine takes
  is a full-resolution length precisely so it means the same thing at any zoom;
  this is the opposite kind of quantity. It is furniture around the viewport, so
  it has to hold its apparent thickness rather than grow to fill the pane at
  800%.
* **Fit has to reserve room for it.** `.pane` is `overflow: hidden` and Fit puts
  the image exactly against the pane edge, so a mount drawn outside the image
  would be entirely invisible in the one view you would most want it in. The
  border width plus a shadow allowance comes out of `fitZoom` before the zoom is
  computed. `place()`'s clamping is left alone — it works in image coordinates,
  and the mount hangs outside the image box without moving it. The allowance is
  the blur radius **plus** the vertical offset, not whichever is larger: at a
  wide frame the mount nearly fills the pane and there is no background left to
  darken, so an allowance that is merely close reads fine at 18px and has no
  visible shadow at all by 96px. Caught by screenshotting the widths side by
  side, which is worth doing again to anything on this bar.
* **Its own element, drawn as two spread `box-shadow`s.** Not a border on the
  `<img>`: overlay mode draws the wipe by clipping the result image with
  `clipPath`, and **clipPath clips a box-shadow with it**, so a ring on that
  image would lose whichever side was wiped away. And a CSS border would grow
  the box past the `dw × dh` every coordinate here derives from, putting the
  pointer-anchored zoom half a border out. A spread shadow occupies no layout at
  all.
* **The drop shadow carries the same spread as the border.** Laid down from the
  image's edge instead, it sits *underneath* the opaque mount rather than around
  it, and the effect is half missing without ever looking broken.

The board is `#e8e6e0`, not `#fff`: a pure-white surround is brighter than any
highlight in the picture and drags the eye's white point with it, so highlights
read duller than they are — which is the one thing you cannot afford to misjudge
while dialling in halation. The element is also *filled*, not merely ringed,
because `place()` produces fractional `left`/`top` at most zooms and a ring
alone leaves a subpixel seam of dark background at the image edge.

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

## The app opens with every section muted (added 2026-08-04)

Requested: the photo should show untouched on boot -- and after Reset -- even
though the starting point is still `Stock` (or whatever `DEFAULT_PRESET`
resolves to), and picking a preset from the dropdown or loading a file should
be the one thing that switches every section on at once.

This is built entirely out of the mute mechanism that already existed for one
section at a time (`toggleGroup`): muting a group keeps its real values in
`muted[group]` while the *displayed* values for that group go to
`schema.neutral`, so un-muting restores exactly what was there. `muteAll(s,
src)` does that for every group in one pass, seeding each group's kept values
from `src` -- the starting preset's authored values, not whatever the sliders
currently show.

Boot and `resetAll` both now call `setValues(schema.neutral)` /
`setApplied(schema.neutral)` plus `setMuted(muteAll(schema, start.values))`,
instead of applying `start.values` directly. That is a small but deliberate
extension of the existing "boot and Reset share `startingValues` so they
cannot drift" rule: Reset means "how it opened", and now that opening means
*muted*, Reset has to produce the muted state too, or pressing it would
un-mute sections that boot left off -- reintroducing exactly the drift the
shared helper was written to prevent.

`applyPreset` and `loadPreset` both call `setMuted({})` after applying their
values -- picking a whole look, whether from the menu or from a file, is the
one thing that is not a partial "try this and see" action, so every section
goes live rather than staying staged behind its own toggle.

Two things that fall out of this rather than needing separate handling: the
`Original` button's `disabled={isOriginal}` correctly reads *true* right after
boot, because `values === schema.neutral` at that point -- the app opens
already agreeing with its own "show the untouched photo" state, not merely
looking like it does. And `master_opacity` (excluded from `NEUTRAL_ZERO`
because 1.0, not 0, is its neutral) is unaffected by any of this: muting the
Output group still means "no dial-back", exactly as it does when toggled by
hand.

Verified against a real running instance rather than by inspection: every
section reads muted on a fresh load, picking a preset flips all twelve to
enabled in one render, and loading a preset file from disk does the same.

## Tuning constants (all in engine.py, all calibrated by measurement)

| Constant | Value | Why |
|---|---|---|
| `EDGE_REF` | 0.06 | Fixed edge-magnitude reference. Must stay a constant, not a statistic — see invariant 1. |
| `_GRADE_TEMP_GAIN` | 0.40 | Peak channel gain for Temperature at ±1: red and blue move this far in opposite directions, green is left alone, and the vector is then normalised against the luma weights so the control cannot also expose the frame. |
| `_GRADE_TONE_KNEE` | 0.5 | Where the Shadows and Highlights ramps meet. Both are quintic over half the range, so every pixel is in exactly one of them and a gradient shows no seam. Not exposed — a knee and a falloff per end would be four more sliders in a section that is meant to stay cheap. |
| `_GRADE_TONE_MAX` | 0.35 | How far a tone lift travels at ±1, as a share of its headroom. **Not 1.0**: that takes a black pixel to pure white at +1 and squeezes the useful range into the slider's first tenth. Measured on a real photo, Shadows +0.5 took the mean from 0.19 to 0.53 at 1.0. |
| `_GRADE_CLARITY_GAIN` | 1.6 | Gain on the *positive* side of Clarity only. The negative side is pinned at exactly 1.0 and must stay there: at gain 1, −1 removes precisely 100% of the band, and past that it inverts local contrast rather than flattening further. |
| `_GNORM` | 0.55 | Noise normaliser. **The old note here claimed field std ~0.27 clipping ~3.6%; re-measured 2026-07-31 it is std ~0.45 clipping ~18%**, constant across octave counts now that `_fbm` preserves variance. The 18% is pre-existing — a single-octave field measures the same — so this row was simply wrong, not broken by a change. Lowering it flattens the distribution's tails further. |
| `_AMP_SCALE` | 0.38 | Maps the 0–100 intensity slider to amplitude; default 32 lands near 3.5% luminance sigma. Was 0.5 — recalibrated when `_fbm` started preserving variance, since the old value was silently compensating for a field running at 43% strength. |
| `_MIN_CELL` | 0.8 | Floor on lattice cell size in working pixels. Below Nyquist it is pure aliasing. |
| `_SAND_TAPS` | ±2σ, 5 taps | Tangential sanding filter. Reaches ±2σ, not ±1 — contour roughness sits at longer wavelengths than it appears to, and a ±1σ filter removed only 2% of it. Weights are normalised at use: the table sums to 0.991. |
| `_BLUE_HUE` | 230.0 | Centre of the blue-compensation window, **in linear light** where the stage runs — the sRGB number is 220. Skies span 222-236 there; cyan water 194, purple shadow 249. |
| `_BLUE_SAT_FLOOR` | 0.12 | Below this a pixel is grey and the compensation leaves it alone. Without it every neutral in the frame takes a cast — the failure `vibrance` is also written against. |
| `_SCATTER_STENCILS` | 9 entries | Scatter footprints: (name, first angle, count, locus, inner, alt). **Indexed by the parameter value, which is what a preset file stores** — renumbering silently changes every preset that used one, so append rather than insert. `verify.py` pins it against `choices` in params.py name-for-name. Every entry must keep peak travel ≤ reach, which is what `pad_for` reserves for; an L∞ "square" locus would reach 1.41× and would have to be paid for. |
| `_AA_TAPS` | ±1, 3 taps | Anti-aliasing filter along the isophote. Short *on purpose*, and the opposite choice from `_SAND_TAPS` for the same underlying reason: a stair-step is a pixel-scale wobble, so reaching further only averages away the shape the contour has. To make the stage bite harder, raise the pass count — not this. |
| `_AA_PASSES` | 3 | Maximum anti-aliasing passes, and therefore the top of `aa_strength`. One pass removes 34% of a stair-step, three removes 64% while still keeping 70% of the edge. `pad_for` multiplies **both** AA terms by this and pins it rather than deriving it from the strength, for `_SAND_PASSES`' reason: `pad_for` runs at the un-supersampled scale and must not disagree with the renderer about the count. |
| `_AA_DIR_K` / `_AA_DIR_MIN` | 0.5 / 0.7 | Tangent-estimate blur for AA, as a fraction of its radius, and a floor. Smaller than `_SAND_DIR_K` against a smaller radius — this filter follows a contour at the pixel scale, and a wide estimate window cuts the corners off small features. The floor is what stops the tangent swinging on single-pixel noise. |
| `_SMOOTH_MAX` | 0.5 | Peak global-grain smoothing sigma as a fraction of the clump. Half a cell takes measured gridiness 1.74 → 0.27, so the lattice is gone rather than softened, while clump-scale structure survives. |
| `_SMOOTH_GAIN_K` | 5.62 | Restores the amplitude that smoothing blur costs, as `sqrt(1 + k(σ/cell)²)`. Analytic because a measured `std()` would be a statistic of the region and would seam exports. **Fit against the two-octave field it is used on** — calibrated on single-octave noise it comes out 7.7 and makes full Smoothness 10% louder than none. |
| `_VARCELL_JITTER_LO` / `_SPAN` | 0.25 / 0.5 | Where in its own cell a variable-size grain's point can jitter to — the middle half, `[0.25, 0.75]`. Bounding it away from the cell edge is what makes the fixed-ring neighbour search below exact rather than heuristic. |
| `_VARCELL_RINGS` | 2 | Neighbour-cell rings the variable-size search checks each way (5x5). 1 ring only budgets 0.25 cells of warp margin, measurably not enough to fix the resonance below; 2 rings budgets 1.25. |
| `_VARCELL_WARP_CELL_FRAC` / `_AMOUNT` | 0.37 / 0.7 | Domain warp breaking the pixel-grid/cell-grid resonance at exact-integer cell sizes (measured: cell 1.00 scored 0.123 std against 0.193 at neighbouring sizes). Swept 0.5–1.2 cells of warp against cell 1.6 as a control; 0.7 is where they agree to 0.3%. |
| `_VARCELL_TIE_MARGIN` | 1e-3 | How much better a candidate point's falloff must be before it displaces the current winner. Below the ~2e-5 float noise floor between tile layouts a near-tie can flip winners (a discrete jump, not drift); above ~1e-2 it starts treating genuinely different falloffs as tied. Swept 1e-4 to 1e-2; set in the middle. |
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
  mask instead — 43% of the border, **93% of the texture**. `pre_blur`
  (2026-08-02) is the *wanted* version of that behaviour and does not
  contradict this: it was asked for as a global blur, it ships at 0, and its
  help text says outright that it takes texture with it. The lesson stands —
  do not reach for a blur when the ask is "softer edges".
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
| `_variable_cell_noise` | 1.130s | 24.1% |
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
