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

**Licensed AGPL-3.0-or-later as of 2026-08-16** (it had no licence at all before
then, which meant all-rights-reserved on a public repo — nobody could legally
use it). AGPL rather than GPL or MIT specifically *because it is a web app*: the
user's requirement was that nobody fork it, rebrand it and pass it off as their
own, and §13 is the only clause that reaches someone who hosts a modified copy
without ever distributing a binary. Three consequences that constrain code:

* The `Source` link in `TopBar.tsx` is the §13 offer, not decoration. Its
  `SOURCE_URL` must be a lone constant so a fork can repoint it.
* `build.sh` copies `LICENSE` and `NOTICE` and **fails** without them —
  distributing a build without the text is the *builder's* breach.
* `luts/gmic/` is **not** AGPL. It is Pat David's film emulation set under
  CC BY-SA 4.0 (attribution + share-alike, **no** NonCommercial clause, so
  users selling their photos is safe). `luts/gmic/LICENSE` must travel with the
  data — `build.sh` copies it in a second walk because the LUT walk filters on
  `*.cube` and would drop the one file that makes the redistribution lawful.

The name, logo and `UP-` LUT names are reserved under GPL-3.0 §7(e); copyright
keeps a fork open, the name reservation is what stops it being marketed as this
product. Duc Duong is sole author across the whole history, so dual-licensing
(AGPL for the community, paid commercial licence for closed use) stays
available — accepting outside PRs without a CLA or DCO is what would erode it.

## Priorities (from the user, and they override TOPIC.md's original draft)

**In scope — the point of the app:** detail destruction, edge softening and
edge noising, grain, halation, chromatic edge fringing.

**In scope as of 2026-08-16:** a `Normalize` section **above** Colour Grading —
one checkbox, shipping off, that corrects an input photograph's lightness and
white balance and compresses its dynamic range so neither end clips. Requested
outright, and requested as *its own* logic: "totally new logic on top of
everything... Dont make it mess up with any color grading logic." So it is a
separate group, module, mixin, constants file and check module, and it touches
nothing in `stages/colour_grade.py`. It is the only stage whose settings are
measured from the image rather than dialled in. See `docs/normalize.md`.

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
                          (`choices` renders a menu, `toggle` a checkbox; the
                           value is a number either way)
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
  checkpoint.py         section-boundary frame cache (three boundaries; `_BELOW`
                          takes its suffixes **by name**, never by index -- a
                          positional slice breaks silently the moment a section
                          is inserted above one, and did)
  grain_engine.py       GrainEngine -- composes the stage mixins
server/models/        Upload, export jobs (domain, no HTTP)
server/services/      render_tier -- the one path both preview tiers take
server/controllers/   FastAPI routers, one per area
server/runtime.py     IS_DEV, DEVICE, ENGINE, the render lock and ticket
server/main.py        app assembly only
server/imageio.py     decode/encode, incl. a hand-written 16-bit RGB PNG encoder
server/lut.py         .cube 3D LUT parsing + registry (the tree and uploads)
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
luts/                 3D LUTs -- files, not code; same idea as presets/, but a
                        *tree* since 2026-08-09: an id is the path relative to
                        luts/ without the extension, and subfolders become
                        collapsed groups in the picker
launch.py             the server entrypoint -- uvicorn programmatically, port
                        binding, the parent watchdog and `--selftest`. The only
                        way to start the server without a shell.
electron/             the desktop shell: main.js owns the Python process
tools/bundle.py       assembles build/bundle/ -- payload + a relocatable CPython
tools/freeze.py       Pipfile.lock -> requirements/ (restores dropped markers)
run.sh / dev.sh       production from source / hot-reload dev
run-prod.sh           rebuild the bundle and run it (uses its own interpreter)
build.sh              client -> payload -> Electron app in dist/
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
slider in a view. Give it `choices=(...)` and it renders as a menu instead, or
`toggle=True` and it renders as a checkbox; the value is still a number
everywhere else, so nothing but the branches in
`views/controls/ParamControl.tsx` knows the difference.

One asymmetry to know when adding a field to `Param`: the server ships it to the
client for free (`schema.py` uses `asdict`), but the client's own `Param` in
`web/src/services/api.ts` is a **hand-written mirror**, so a field left out
there is simply invisible — `spatial` has been shipping unread for exactly that
reason.

## It is a desktop app now (2026-08-19)

`build.sh` produces `dist/mac-arm64/Film Grain.app`: Electron plus a **relocatable
CPython 3.13 with torch inside it**, so the machine it lands on needs nothing
installed. The old distribution was self-contained apart from Python, which is the
part that mattered — `build/run.sh` hunted for an interpreter and, failing, told
the recipient to pip-install ~700MB of torch.

Four things worth knowing before touching any of it:

* **Electron loads `http://127.0.0.1:<port>`, never `file://`.** Every call in
  `services/api.ts` is a same-origin relative path, so serving the client off disk
  would break all of them and need `base: "./"` in the Vite config. There is one
  client build, not two, and **no Electron-aware code in `web/src` at all** — the
  downloads, the navigation guards and the quit guard all live in
  `electron/main.js`.
* **A portable interpreter, not a frozen binary.** `client.py:18` resolves the
  client from `__file__` and *raises at import* without it, and nothing here knows
  about `sys._MEIPASS`. PyInstaller `--onefile` would die on that line. Keeping a
  real directory tree is why all three data roots resolve unchanged and no
  path-handling code moved.
* **The compute backend is a run-time decision, not a build-time one.**
  `pick_device()` already did this and `/api/health` already reported it; one
  artifact adapts to the hardware. That is why a future Windows build ships CUDA
  wheels and still works on machines with no NVIDIA GPU. AMD/Intel GPUs are CPU
  only, permanently: ROCm is Linux-only and the Windows DirectML route is an
  unmaintained build pinned to an old torch. Intel Macs cannot be supported —
  torch 2.2.2 was the last macOS x86_64 wheel.
* **`tools/bundle.py` fails the build if a shipped file contains the build
  machine's path.** Keep that check. It caught two real leaks that nothing else
  would have: pip's console-script wrappers (absolute shebangs, broken on arrival)
  and stdlib `.pyc` written by the import machinery during `compileall`, which
  ignores `-s`.

`build/` still holds the *previous* build system's output and is deliberately left
alone; everything new is under `build/bundle/` and `dist/`.

**The window has no title bar** (`titleBarStyle: "hidden"`), so the app's own
`.bar` *is* the title bar and there is no strip of system grey above it. Three
numbers are coupled by that and they are in two different files, so changing one
alone is a visible bug rather than a subtle one:

1. `SEAMLESS_CSS` in `electron/main.js` pads `.bar` to clear the traffic lights.
2. `trafficLightPosition` in the same file centres the lights **in the bar's
   measured height** — which changes the moment that padding does.
3. `.bar-stand-in` in `electron/splash.html` reproduces the bar at the same
   height, so the top strip does not change shade when the app replaces the
   splash.

The heights are *measured in the running window*, not derived from the CSS: the
controls set the line box, so `.bar` is 45.84px with the injected padding where
reading `bar.css` suggests less. Two of those numbers were wrong first time round
for exactly that reason. `SEAMLESS_CSS` also needs `!important` — `bar.css` sets
the `padding` shorthand at the same specificity, and an injected stylesheet does
not reliably win that tie; measured, the drag region applied while the padding was
silently overridden back to 14px.

It is injected from the main process rather than written into `web/src` on
purpose: 90px of padding and a drag region are *wrong* in a browser, and one
client build serving both is the property that keeps the browser and desktop
versions honest. It is also scoped to `http(s)` URLs, because the splash is a
second document that had used the same `.bar` class name and the injection
stretched its progress bar to 250x17px.

**`backgroundColor` can never match the page on a P3 display, so the window is
not shown until it has painted.** The window background is filled in by AppKit
and the page by Chromium, and the two do not render the same hex the same way:
`--bg` #0d0e10 comes out **#111215 from AppKit and #151617 from the renderer**.
Any frame where the web contents does not yet cover the window therefore shows
bands of slightly-wrong dark at the top and bottom -- which is exactly what a
screenshot showed, while `capturePage()` of the same splash was provably uniform,
because the page was never the problem. `show: false` plus
`win.once("ready-to-show", …)` removes the exposure instead of trying to
colour-match two different painters. Do not "simplify" that back to `show: true`.

The corollary is worth remembering for any future window work: an on-screen
colour problem that does not reproduce in `capturePage()` is not in the page.

## Pipeline order (a section added at the top 2026-08-16)

`Normalize` is step **-2**, above Colour Grading and so above everything. It is
the first section in `GROUPS` and the first in the panel, and it brought a third
checkpoint with it (`"Colour Grading"`, named for the section it sits above).

Adding it exposed a live tripwire worth knowing before you insert any section
anywhere: `checkpoint.py` sliced `GROUPS[3:]` and `GROUPS[6:]` **positionally**,
so a new group at index 0 shifted both and quietly put `Pre Sharpen` below a
checkpoint saved after it ran. That is the same stale hit the comment on that
boundary already recorded from 2026-08-09, reintroduced from a different file.
Both suffixes are taken by name now. See `docs/normalize.md`.

## Pipeline order (changed 2026-08-09)

`Global Grain` and `Sharpening` moved **below** `Film Texture`, on request, and
the panel moved with them so `pipeline order == panel order` still holds. The
order under the checkpoint is now Halation, Tone Response, Film Texture, Global
Grain, Sharpening.

It is a change of *look*, not a refactor, and a large one: sharpening now bites
on the marks, and ten of the twelve shipped presets carry `sharpen 12` against
the ~1.2 where that stage's own help says halos start, so every speck and hair
comes out ringed. Global Grain's four source-masked layers key on a frame that
has the debris in it, and that layer is no longer bloomed by halation or
developed by the characteristic curve. `docs/film-texture/placement.md` has the
argument and the measurements; nothing in `pad_for` or the caches needed to
change.

One thing the move surfaced and `checkpoint.py` now states outright: a section
running below a checkpoint is **not** enough to keep it out of the key. `render()`
evaluates the characteristic curve at section 3 as a mask input and applies it
for real at section 7, so `Tone Response` is read above the boundary it sits
below — taking a plain `GROUPS` suffix there made a `brightness` edit come back
2.3e-01 wrong. `verify.py` catches it by re-rendering one parameter per section
against a warm cache.

## Two reported bugs, both "the seed does nothing" (fixed 2026-08-16)

Filed together and unrelated in the code, but they rhyme: in both, something
that looks randomised is not, and on any single render it is indistinguishable
from something that is.

* **A lone light leak was always on the right-hand edge.** Leak `k` sits at
  `base + φ·k` of the perimeter and `base` was the constant `0.37`, which lands
  on the right border at *every* aspect ratio. Swept over the whole seed space,
  `light_leak 1` reached one of four borders. `base` is drawn from
  `texture_seed` now — once for the whole list, so the φ step still means
  raising the count *adds* leaks instead of rerolling them. This moves the leaks
  in every existing preset, deliberately.
  See `docs/film-texture/light-leaks.md`.
* **`With random seed` was doing nothing at all.** Not a drag-and-drop bug —
  both entry points call the same `onFile`. The reroll was written into a muted
  section's kept snapshot *instead of* the live values, and every session boots
  with every section muted; then `applyPreset` dropped the snapshots and laid the
  preset's own fixed `seed` over the top. Measured on the shipped build, every
  photo rendered with `seed 1234`. See `docs/client-ui.md` — and note the
  schema grew a `neutral_zero` field so the client can tell an *amount* from a
  seed, which is the distinction all three parts of that fix turn on.

Both are pinned by `verify.py`, and the leak fix broke four checks that had been
measuring the old arrangement rather than the geometry — `docs/testing.md` has
what they were doing wrong, because the pattern (a probe reading a fixed window
of the frame and assuming a mark is sitting in it) will recur.

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
structure above the clump, its flat amplitude, its smoothing, its chroma and its
mottling —
anti-aliasing, the pre-blur, edge
softening, edge jitter and its direction bias, edge sanding, scatter, output
sharpening, the master opacity cross-fade, the colour-grading section with its
3D LUT lookup, `.cube` parsing, that every LUT in the `luts/` **tree** loads and
that no id can escape the folder, monotone tone recovery and highlight
reconstruction, the bidirectional split tone, the four source-masked global
layers with their hue masks and
mid-tone bell, the six Global Grain blend modes, `global_seed` as an offset, and
the film-texture section including its exact mark counts, the speck's shape
and softness controls and that a lone light leak can land on any of the four
borders rather than always the right-hand one, and Normalize — that its metering corrects in the right
*direction* on a known-wrong frame, that its white balance is luma-neutral to
0.00e+00 and backs off to exactly identity on a scene that is legitimately one
colour, and that its highlight roll keeps 250 of 256 8-bit levels in a real
photograph's bright region where the version it replaced kept 153, and that Highlight Priority hands that band back at 21 levels against 79, and that a shipped preset's author credit reaches the client instead of being dropped at the door — 420 checks. It exits
non-zero on failure.

Those 420 live in `tests/checks/`, one module per area, since 2026-08-08 — it
was a single 3900-line function taking 4m24s, and it is 18 modules taking ~72s
(39s until `luts/` grew to 303 files, every one of which the `grading` module
parses on purpose).
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
| [docs/normalize.md](docs/normalize.md) | Step −2: the one stage whose settings are *measured* rather than dialled, why the metering is per-upload and not per-tile, and why the shoulder is uncapped where the toe is not |
| [docs/using-the-controls.md](docs/using-the-controls.md) | What each control does, for a user rather than a maintainer — moved out of `README.md` 2026-08-08 |
| [docs/architecture.md](docs/architecture.md) | Where everything lives after the 2026-08-08 package split, the two import rules that keep it acyclic, and why `Stage.tsx` was left whole |
| [docs/pipeline-order.md](docs/pipeline-order.md) | Which stages are placed by *position* and what breaks if they move; `pre_blur` vs `micro_blur`; why `master_opacity` lives outside `render()` |
| [docs/preview-and-export.md](docs/preview-and-export.md) | The client-scaled two-tier preview, and why every export is the preview tier enlarged to full size with the supersample as the only choice |
| [docs/colour-grading.md](docs/colour-grading.md) | Step −1: LUTs as *resources*, the twelve adjustments, the Shadows/Highlights rewrite, highlight reconstruction — and, filed with them, Tone Response's bidirectional split tone |
| [docs/halation.md](docs/halation.md) | Blue compensation and why it runs before the wash; highlight recovery metered against real headroom |
| [docs/edge-destruction.md](docs/edge-destruction.md) | Scatter (diffusion without the average) and anti-aliasing (filter along the contour) |
| [docs/global-grain.md](docs/global-grain.md) | Why value noise quilts, the two superseded constructions, the tilted point field, the chroma slider, the five layers (hue mask vs channel value, mask vs seed) and the blend modes |
| [docs/film-texture.md](docs/film-texture.md) | Dust and hair as drawn marks with exact counts, scratches as a field, light leaks as beams |
| [docs/presets.md](docs/presets.md) | The mark-count dead zone, `reference_mp` rescaling across image sizes, and the author credit a re-save used to strip |
| [docs/panel-layout.md](docs/panel-layout.md) | `GROUPS` reorgs and where each parameter's section went, including why Luminance Response stopped being a section |
| [docs/client-ui.md](docs/client-ui.md) | Wheel zoom, the mount, muted-on-boot, and two React traps |
| [docs/tuning-constants.md](docs/tuning-constants.md) | Every calibrated constant in `engine/constants/`, with the measurement behind it |
| [docs/testing.md](docs/testing.md) | The check modules, how to run only the ones you need, and why the suite went from 4m24s to 36s without checking any less |
| [docs/lessons.md](docs/lessons.md) | Things I got wrong, so you don't repeat them |
| [docs/performance.md](docs/performance.md) | Measured timings and the two performance audits |
| [docs/environment.md](docs/environment.md) | Machine gotchas (Python 3.13, MPS, PATH, ports) and what is not done |

Start with `docs/lessons.md` before changing anything in the engine, and with
`docs/tuning-constants.md` before changing a number in it.
