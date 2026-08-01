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
slider in `App.tsx`.

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
direction bias, edge sanding, output sharpening and the film-texture
section — 58 checks. It exits non-zero on failure.

The global-grain, edge-softening, edge-sanding and sharpening checks exist
because those stages ship at 0, so the default-parameter checks render straight
past them. Each re-runs tile independence with its stage switched on: they all
add work `pad_for` has to cover, and a kernel missing from `pad_for` seams
tiled exports along exactly its radius while every preview looks fine. Both
warps are in `pad_for` too — they displace rather than blur, so they read
pixels up to their peak travel away.

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

* Zoom and pan are free. They never re-render and never hit the network.
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

## Film texture is drawn, never scattered (added 2026-07-31)

Step 15, dead last, after sharpening: dust, scratches, hair, light leaks.
Everything above it models what the *emulsion* does; this models what happened
to the strip of film afterwards. It is weighted by none of the image masks — a
scratch does not care what is underneath it — and every parameter ships at 0.

**Do not reimplement this by scattering objects.** A list of speck positions is
a statistic of the region: an export would split a scratch across two tiles, or
draw a different list per tile. Every mark here is a *threshold on a noise
field addressed in global coordinates*, so every pixel gets the same answer
whichever tile asks. It also happens to look better — the outlines are organic
because the field is, where stamped sprites repeat.

How each shape is made, and the measured result at full strength:

| mark | how | coverage | geometry |
|---|---|---|---|
| dust | isotropic fine noise, two populations (dark motes, bright pinholes) | 0.62% | 1.0:1, compact |
| scratches | noise with cells ~2px wide and ~900px tall — the anisotropy *is* the scratch | 0.18% | 74:1, 1.1px wide |
| hair | level set of a smooth field: `|n − 0.5| < eps` is a curve that wanders | 0.37% | 2.0px wide |
| light leak | frame-edge falloff × slow noise, added in **linear** light | 22% | broad |

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

Light leaks vary per leak too, through `leak_variation`: reach into the frame
(the size), falloff exponent (the halo, broad glow versus tight rim), edge
hardness and strength. At 0 every leak is identical, which is exactly what read
as stamped. Two things had to be fixed together to make it work, and either
alone does nothing:

* The wash cell was 700px — **wider than a frame**, so there was only ever one
  leak and nothing to be non-uniform *against*. Same trap as the hair gate.
  240px gives three or four along a border.
* Value noise is far too centre-weighted to use raw as a variation field:
  measured p10-p90 spans only 0.41-0.71, std 0.11. A 9x range of available
  reach still produced near-identical leaks. `_spread()` stretches it about the
  field's median (0.578) before use.

Measured on a 2000x1400 plate, per-leak strength spread goes 3% at variation 0
to 20% at 1. Note the plate size: the leak field's cells are fixed in pixels,
so on a small frame only a handful of leaks fit and the corners truncate them —
the statistic then ranks the two settings *backwards*. The check uses its own
larger plate for that reason.

A light leak has **two** visible edges and both need softening or it reads as
a painted shape: the radial falloff coming in from the border, and the
transition along the border where one leak stops. `leak_feather` drives both --
the falloff exponent (4.0 tight rim down to 0.5 broad wash) and the gate band
width. The exponent matters more than the reach: at a high exponent a leak with
plenty of reach still lands as a thin bright rim, because it drops away
immediately past the border.

**The gate must vary only *along* the border, never with depth.** It used to be
one isotropic field, which also varied going inward -- so it shut the leak off a
little way in however much reach it had, and `leak_size` was fighting the gate
rather than setting the size. There are now two washes, each stretched to be
constant inward (`cell_y` of 40000px for the horizontal one), selected by which
border a pixel is nearest. Measured after the fix: size 0.1 reaches 69px into a
1000px frame, size 1.0 reaches 193px, coverage 4.1% to 20.7%.

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

**A leak must not reach the middle of the frame.** Inward reach is capped at
`_LEAK_REACH_MAX = 0.17` of the half-frame and saturates by size 1; past that,
Leak Size opens more of the *border* (by widening the gate) rather than going
deeper. An earlier attempt let large sizes lift a floor under the whole leak,
which fogged the centre -- measurably wrong, and it reads as a bad exposure
rather than as a leak. Measured now: centre lift 0.0000 at every size from 0.55
to 10, while coverage still grows 10% to 33%.

Corners bloom `_LEAK_CORNER = 1.6` times further in than edge midpoints, keyed
on the diagonal distance rather than the nearest-edge distance. That is where
the cassette mouth and the film gate actually let light past, and it is most of
what makes a leak read as a leak: measured 0.315 at the corners against 0.024
at an edge midpoint.

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

## Presets rescale across image sizes (added 2026-08-01)

A preset dialled in on one photo is locked to that photo's size: every spatial
parameter is a length in full-resolution pixels, so the same numbers on a
bigger frame give proportionally finer grain and tighter halation. Preset files
now carry `reference_mp`, the size they were authored at, and the client sends
it with every render; `_params_for` rescales before the engine sees anything.

**The ratio is linear, not area.** Thirteen parameters are marked
`spatial=True` and multiplied by `sqrt(current_mp / reference_mp)`. A 16MP
frame is 0.816x the *width* of a 24MP one, not 0.667x -- scaling lengths by the
megapixel ratio overshoots by the square root. `edge_jitter` is in that list
despite having no `px` unit: `_JITTER_MAX` makes it a length multiplier.

Not rescaled, on purpose: amounts and blend weights (dimensionless, per-pixel),
mark counts (already resolved against frame area inside the engine, so 50
specks is 50 specks at any size), and `leak_size` (a fraction of the frame).

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
1.0, so the behaviour is unchanged rather than guessed at.

## Tuning constants (all in engine.py, all calibrated by measurement)

| Constant | Value | Why |
|---|---|---|
| `EDGE_REF` | 0.06 | Fixed edge-magnitude reference. Must stay a constant, not a statistic — see invariant 1. |
| `_GNORM` | 0.55 | Noise normaliser. **The old note here claimed field std ~0.27 clipping ~3.6%; re-measured 2026-07-31 it is std ~0.45 clipping ~18%**, constant across octave counts now that `_fbm` preserves variance. The 18% is pre-existing — a single-octave field measures the same — so this row was simply wrong, not broken by a change. Lowering it flattens the distribution's tails further. |
| `_AMP_SCALE` | 0.38 | Maps the 0–100 intensity slider to amplitude; default 32 lands near 3.5% luminance sigma. Was 0.5 — recalibrated when `_fbm` started preserving variance, since the old value was silently compensating for a field running at 43% strength. |
| `_MIN_CELL` | 0.8 | Floor on lattice cell size in working pixels. Below Nyquist it is pure aliasing. |
| `_SAND_TAPS` | ±2σ, 5 taps | Tangential sanding filter. Reaches ±2σ, not ±1 — contour roughness sits at longer wavelengths than it appears to, and a ±1σ filter removed only 2% of it. Weights are normalised at use: the table sums to 0.991. |
| `_JITTER_MAX` | 3.0 | Peak edge displacement in full-res px at `edge_jitter` 1. Was an inline 0.6, whose *typical* displacement was 0.227px — invisible. |
| `_STEP_LO` / `_STEP_HI` | 0.030 / 0.110 | Luma-step bounds separating a real transition from fine texture, for the edge-softening mask. Fine texture measures an order of magnitude under a hard border, which is the gap that lets softening take the snap off a border and leave fabric alone. |
| `_TEX_LO` / `_TEX_HI` | 0.002 / 0.015 | Local mean-abs-deviation bounds separating "smooth" from "textured" for the smooth-area guard. Skin and clear sky sit at or below `_TEX_LO`; fabric, foliage and hair sit above `_TEX_HI`. |
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
