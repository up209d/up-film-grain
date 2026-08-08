# CLAUDE.md

Working notes for this repo. `TOPIC.md` holds the domain rules and pipeline
design; `README.md` holds setup and usage. This file is the catch-up: state,
constraints, gotchas, and the reasoning behind decisions that are not obvious
from the code.

**Read this file in full — it is the part that is always relevant.** The
per-area reasoning it used to carry inline now lives in `docs/`, indexed at the
bottom. Read the ones covering whatever you are about to touch.

This repo is indexed by CodeGraph (`.codegraph/`), so reach for
`codegraph explore "<symbols or question>"` before grep when locating code.

## What this is

A web app that applies organic film grain to still photographs. Python/PyTorch
image service, React UI, single service (FastAPI serves the built client).
Working and verified end to end as of 2026-07-31.

## Priorities (from the user, and they override TOPIC.md's original draft)

**In scope — the point of the app:** detail destruction, edge softening and
edge noising, grain, halation, chromatic edge fringing.

**In scope as of 2026-08-04:** a `Colour Grading` section at the very top of the
pipeline — 3D LUTs plus temperature, tint, exposure, shadows, highlights,
contrast, black point, a two-way clarity, vibrance and saturation. Requested
outright, so the "colour grading is deferred" note below no longer covers this
one section. Everything in it still ships at 0, so the colour pass-through
holds with nothing selected. See `docs/colour-grading.md` (the six added
after the first pass are in their own dated subsection there).

**Deferred — the user has a separate project planned:** colour grading
*elsewhere in the pipeline*. `vibrance` and `brightness` were added on request
2026-07-31 and
ship at 0 like the rest, so the pass-through still holds. The engine implements tone curves, contrast, toe/shoulder, split
toning, highlight desaturation and base fog, and exposes them as sliders, but
they **ship neutral (0)** so the pipeline is a colour pass-through. Do not tune
them or fold them into presets without asking. The `Tone Response` group is
still the deferred one; `Colour Grading` is not.

Deferred does not mean frozen: the split tone in `Tone Response` was rewritten
2026-08-06 on request. `warm_highlights` / `cool_shadows` (0…1, one direction
each) are now `highlight_warmth` / `shadow_warmth` (**−1 cool … +1 warm**), the
amplitude is nearly 3× what it was, and the axis is projected onto the luma-null
plane so a warmth push cannot also change brightness. Both still ship at 0. See
`docs/colour-grading.md`.

**Quality beats speed.** The user has explicitly accepted lag and latency. Do
not clamp octaves, lattice density, blur radii, supersampling or preview
resolution for performance alone. If a quality/speed trade-off appears, take
quality by default and expose speed as opt-in.

`TOPIC.md` was originally Gemini-generated and the user has said it is not
authoritative — it has since been rewritten to match what was actually built,
including a section correcting three claims that did not survive implementation.

## Layout

Split into packages on 2026-08-08 — `engine.py`, `params.py`, `main.py` and
`App.tsx` were single files of 5000, 1650, 500 and 1900 lines. Nothing moved
*between* layers and no behaviour changed; `docs/architecture.md` has the map
and the two import rules that keep it acyclic.

```
server/params/        parameter schema -- SINGLE SOURCE OF TRUTH for engine + UI
  param.py              the Param record, GROUPS, GLOBAL_BLENDS
  definitions/          the controls, one module per panel section
  registry.py           PARAM_BY_KEY, DEFAULTS, NEUTRAL_ZERO, rescale
  sanitize.py           the only door params enter the engine through
  presets.py / schema.py
server/engine/        the pipeline (package docstring states the invariants)
  constants/            every calibrated number, grouped by consumer
  primitives.py         blur, luma, warp, isophote
  colour.py             transfer curves, LUT lookup, highlight reconstruction
  noise/                hashed lattice, grain point field, smooth fields
  marks.py              dust/hair/leak site lists
  stages/               one mixin per panel section; render.py is the order
  tiling.py             supersampling, pad_for, tile_for, render entries
  checkpoint.py         section-boundary frame cache (the two usable boundaries)
  grain_engine.py       GrainEngine -- composes the stage mixins
server/models/        Upload, export jobs (domain, no HTTP)
server/services/      render_tier -- the one path both preview tiers take
server/controllers/   FastAPI routers, one per area
server/runtime.py     IS_DEV, DEVICE, ENGINE, the render lock and ticket
server/main.py        app assembly only
server/imageio.py     decode/encode, incl. a hand-written 16-bit RGB PNG encoder
server/lut.py         .cube 3D LUT parsing + registry (folder and uploads)
web/src/models/       Values/Compare, view constants, pure value-set rules
web/src/services/     typed API client
web/src/controllers/  hooks: schema, values, history, preview, export, upload, luts
web/src/views/        App composes panels/, stage/, controls/
web/src/styles/       CSS partials, imported by styles.css in cascade order
tests/verify.py       the CLI -- run after touching engine/
tests/runner.py       selection, scheduling, reporting
tests/harness.py      check(), the registry, Ctx (the fixtures), shared metrics
tests/checks/         one module per area -- where the checks live
tests/refs.py         slow reference implementations the rewrites are held to
tests/scene.py        synthetic test scene
presets/              preset library -- files, not code; see below
luts/                 3D LUTs -- files, not code; same idea as presets/
run.sh / dev.sh       production from source / hot-reload dev
build.sh              compiles a distribution into build/
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
`web/src/models/paramState.ts` deliberately — Reset meaning something other than "how it opened"
would be its own small bug. `build.sh`
copies the folder; a distribution without it would silently have no presets.

`APP_ENV` gates dev-only behaviour and **defaults to production** — CORS for
Vite's origin, `/docs` and `/openapi.json` are all dev-only, and a production
process with no `web/dist` raises at import rather than booting and serving
503s. `dev.sh` exports `APP_ENV=development`; `run.sh` and the distribution's
launcher export production. If you add a dev convenience, gate it on `IS_DEV`
or it ships.

Adding a parameter means adding one `Param` to the right module under
`server/params/definitions/` and reading `p["key"]` in the engine. The UI picks it up automatically — never hand-add a
slider in a view. Give it `choices=(...)` and it renders as a menu instead;
the value is still a number everywhere else, so nothing but the one branch in
`views/controls/ParamControl.tsx` knows the difference.

## Two invariants that must not break

Both are silent killers: break either and previews still look fine while
exports are wrong.

1. **Tile independence.** No stage may depend on a statistic of the region being
   rendered — no per-tile normalisation, no global mean, no `arr.max()`. Edge
   strength normalises against the fixed `EDGE_REF`; the noise lattice is
   addressed by absolute global coordinates. Break it and exports grow seams
   that no preview will ever show.

   Note what this does *not* forbid, because the distinction has now been got
   wrong twice in the same file: a list of objects is fine as long as it is a
   function of the count, the seed and the **frame**. Light leaks have always
   worked that way, and dust and hair joined them on 2026-08-06 — every tile
   builds the identical list and clips each mark to *its own* footprint in
   absolute coordinates. What breaks the invariant is a list derived from the
   region being rendered: N specks per tile, or positions drawn against the
   tile's own area. See `docs/film-texture.md`.
2. **Scale invariance.** Every spatial quantity is specified in full-resolution
   pixels and multiplied by the working `scale`. The preview no longer relies
   on this (it renders at `scale = 1.0` like the export), but supersampling
   does — it renders at `scale * ss` — and so does every check in `verify.py`.
   Break it and 2× and 1× stop agreeing.

`pipenv run python tests/verify.py` checks both, plus zoom fidelity, colour
pass-through, the luminance-response band and that it is keyed on developed
density rather than on the softened frame, edge bias, the smooth-area guard, 16-bit PNG
validity, the global-grain point field — its freedom from the pixel grid, its
structure above the clump, its flat amplitude, its smoothing and its chroma —
anti-aliasing, the pre-blur, edge
softening, edge jitter and its direction bias, edge sanding, scatter, output
sharpening, the master opacity cross-fade, the colour-grading section with its
3D LUT lookup, `.cube` parsing, monotone tone recovery and highlight
reconstruction, the bidirectional split tone, the four source-masked global
layers with their hue masks and
mid-tone bell, the six Global Grain blend modes, `global_seed` as an offset, and
the film-texture section including its exact mark counts and the speck's shape
and softness controls — 362 checks. It exits
non-zero on failure.

Those 362 live in `tests/checks/`, one module per area, since 2026-08-08 — it
was a single 3900-line function taking 4m24s, and it is 17 modules taking 39s.
**Name the modules covering what you touched and only those run:**
`verify.py global` is four modules and ~20s. `verify.py -l` lists them, `-j 1`
runs in one process when you need a traceback in place. Nothing about what is
checked changed in that split, and the bar it was held to was a byte-identical
log — same names, same order, same measured values. See `docs/testing.md`.

Three of those are a *third* kind of assertion and the newest: they measure a
control in **8-bit levels** rather than in float deltas. The split tone shipped
for weeks doing something real and invisible — a peak shift of 0.055 reached
only at pure white, so an ordinary highlight moved by under two levels. Any
"did it change the picture" test passes on that. If a control's failure mode is
*being too subtle to see*, the check has to name the threshold at which a human
sees it.

One of those is a different *kind* of assertion again and is worth copying when
adding a tonal control: the tone-curve check measures the **slope of the
transfer function** over twelve settings rather than the mean level it produces.
A mean-only test cannot tell "this moved the tones" from "this folded the tones
over", which is exactly how a non-monotonic Shadows/Highlights shipped and
survived a full check suite — see `docs/colour-grading.md`.

Several of those are a different *kind* of check worth knowing about: rather than
measuring a property, they assert **agreement with a deliberately slow reference
implementation written out beside them** (`lattice_ref`, `span_ref`,
`grain_ref`, all in `tests/refs.py`). A faster rewrite of a noise generator is
only correct if it changes nothing, and "the render still looks like grain"
cannot tell you that. `grain_ref` carries a second job: it searches a wider 5x5
neighbourhood than the engine's 3x3, so it is also the *proof* behind
`_GRAIN_RINGS` written out as a measurement. Keep the references when touching
those functions — deleting them turns the checks into tautologies.

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

**Dust and hair reserve nothing there any more** (2026-08-06) and that is the
one direction this argument runs the other way, so it has its own check. They
used to blur their mark fields; drawn one mark at a time from absolute
coordinates with an analytic soft edge, neither has a kernel at all. A stage
*removed* from `pad_for` is exactly as dangerous as one missing from it, so
`verify.py` tiles 300 specks and 20 hairs at 128px against a single pass and
requires 0.00e+00.

One trap when adding a check here: `sanitize(None)` fills in *defaults*, not
zeros, so an override dict has to zero every other stage that could contribute
to the same measurement. The sanding check failed first time round because
`edge_jitter` defaults to 0.3 and was quietly adding its own wander to it.

## The rest is in `docs/`

This file is the catch-up. Everything below is the *why* behind one area, split
out so this one stays readable — each is self-contained and none of it is
optional reading before touching the area it covers.

| file | what is in it |
|---|---|
| [docs/using-the-controls.md](docs/using-the-controls.md) | What each control does, for a user rather than a maintainer — moved out of `README.md` 2026-08-08 |
| [docs/architecture.md](docs/architecture.md) | Where everything lives after the 2026-08-08 package split, the two import rules that keep it acyclic, and why `Stage.tsx` was left whole |
| [docs/pipeline-order.md](docs/pipeline-order.md) | Which stages are placed by *position* and what breaks if they move; `pre_blur` vs `micro_blur`; why `master_opacity` lives outside `render()` |
| [docs/preview-and-export.md](docs/preview-and-export.md) | The client-scaled two-tier preview, and why every export is full size with the supersample as the only choice |
| [docs/colour-grading.md](docs/colour-grading.md) | Step −1: LUTs as *resources*, the twelve adjustments, the Shadows/Highlights rewrite, highlight reconstruction — and, filed with them, Tone Response's bidirectional split tone |
| [docs/halation.md](docs/halation.md) | Blue compensation and why it runs before the wash; highlight recovery metered against real headroom |
| [docs/edge-destruction.md](docs/edge-destruction.md) | Scatter (diffusion without the average) and anti-aliasing (filter along the contour) |
| [docs/global-grain.md](docs/global-grain.md) | Why value noise quilts, the two superseded constructions, the tilted point field, the chroma slider, the five layers (hue mask vs channel value, mask vs seed) and the blend modes |
| [docs/film-texture.md](docs/film-texture.md) | Dust and hair as drawn marks with exact counts, scratches as a field, light leaks as beams |
| [docs/presets.md](docs/presets.md) | The mark-count dead zone, and `reference_mp` rescaling across image sizes |
| [docs/panel-layout.md](docs/panel-layout.md) | `GROUPS` reorgs and where each parameter's section went, including why Luminance Response stopped being a section |
| [docs/client-ui.md](docs/client-ui.md) | Wheel zoom, the mount, muted-on-boot, and two React traps |
| [docs/tuning-constants.md](docs/tuning-constants.md) | Every calibrated constant in `engine/constants/`, with the measurement behind it |
| [docs/testing.md](docs/testing.md) | The check modules, how to run only the ones you need, and why the suite went from 4m24s to 36s without checking any less |
| [docs/lessons.md](docs/lessons.md) | Things I got wrong, so you don't repeat them |
| [docs/performance.md](docs/performance.md) | Measured timings and the two performance audits |
| [docs/environment.md](docs/environment.md) | Machine gotchas (Python 3.13, MPS, PATH, ports) and what is not done |

Start with `docs/lessons.md` before changing anything in the engine, and with
`docs/tuning-constants.md` before changing a number in it.
