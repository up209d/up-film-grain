# Prescaling Source

*Added 2026-08-29, on request. Step −3, above `Normalize` and therefore above
everything.*

Resamples the photograph to a fixed megapixel count before the pipeline is
handed it, so the whole engine sees the same resolution whatever came out of the
camera. Three controls: a switch (**on** by default), a free target in MP
(**24** by default), and a menu deciding whether the exported *file* is written
at that resolution or resampled back to the photograph's own.

## Why it exists: the other half of `reference_mp`

Every spatial quantity in the engine is a length in full-resolution pixels times
the working `scale` — invariant 2. That is what makes the proxy and the export
agree about *the picture*, and it is also what makes the same numbers render a
different look on a 12MP frame than on a 50MP one: a 1.6px clump is a finer
thing on a bigger photograph. The look follows the file's resolution rather than
the photograph.

`reference_mp` rescaling (see `docs/presets.md`) answers that by moving **the
parameters** to fit the photograph. This answers it by moving **the photograph**
to fit the parameters. They are the same correction from opposite ends, and that
has three consequences worth stating outright:

* **You want one of them doing the work, not both.** Applying both is not twice
  as good, it is wrong. `params_for` therefore measures the rescale factor
  against the *frame* rather than the file — see "the double-scaling guard"
  below.
* **Every shipped preset records `reference_mp: 24`**, so with the target at 24
  the rescale factor is 1.00× and the preset's numbers reach the engine exactly
  as their author dialled them. That is the payoff, and it is why the section is
  rendered directly above Size Scaling rather than where `GROUPS` puts it.
* **Prescaling avoids a loss the rescale cannot.** `rescale` clamps every value
  back into `[min, max]` afterwards, so a large factor silently pins parameters
  at their sliders' ceilings — a real, irreversible loss on a big upscale.
  Moving the photograph has no such ceiling.

It also makes one long-standing divergence constant instead of per-photograph.
`docs/preview-and-export.md` records that the proxy cannot resolve its own
finest structure and that `_MIN_CELL` floors what is left, so the proxy's grain
is coarser *relative to the picture* than the export's. How much coarser depended
entirely on the file's size, because `proxy_scale` is `2400 / max(h, w)`. With
prescaling on, `proxy_scale` is a function of the target alone — 0.4 on a 3:2
24MP frame, on every photograph — so what the preview under-resolves is the same
amount every time.

## Not a stage, and that is the whole design

It appears nowhere in `engine/stages/render.py`, has no mixin, no constants
module, and reserves **nothing** in `pad_for`. The resample lives in
`server/models/upload.py`, above the engine.

That placement is what keeps both invariants untouched by the feature. No stage
ever learns that the frame it is rendering is not the file, so tile independence
and scale invariance are exactly the properties they were — there is no new
statistic of a region, and no new length to scale. `tests/checks/prescale.py`
tiles a prescaled frame at 128px anyway, not because it could plausibly seam but
because *that claim* is the thing worth pinning: a future version that resampled
inside a stage would break invariant 1, and that check is what would say so.

It is the second stage-shaped thing to live outside the engine, after
Normalize's metering, and it is outside for a different reason. Normalize is
outside because invariant 1 forbids a whole-image statistic inside a tile
(`docs/architecture.md` rule 1c). This is outside because it is not an operation
on pixels at all — it changes which array the engine is called with.

## `Frame`: the resample happens once per photograph

The user's requirement was explicit: *"make it a checkpoint so we don't have to
rescale every time we move any slider."* Prescaling is a property of the
photograph, not of the parameters being dragged, so paying for it per render
would put a full-frame interpolation inside the drag loop for no change in its
result.

`Upload.at(target)` returns a `Frame` — the photograph at one working resolution,
with its own `h`, `w`, `proxy`, `proxy_scale` and `id` — and caches it in a
single slot on the upload. Four details are load-bearing:

* **`at(None)` returns the `Upload` itself.** Not an identity `Frame`. That is
  what makes "Prescaling off behaves exactly as it did before this existed"
  structural rather than hopeful: there is no second code path to keep in step,
  because there is no second object. `render_tier` and `params_for` read the
  same handful of attributes off either and cannot tell which they were handed.
* **So does a photograph already at the target.** An identity `Frame` would
  carry a `@24mp` checkpoint id and split one photograph's engine cache in two
  for no gain.
* **`arr` is lazy and `proxy` is built in one pass from the source.** Only the
  full tier and the 1:1 export ever want `arr`; at 24MP it is ~288MB. And the
  proxy deliberately does *not* go via `arr` — upscaling a 6MP photograph to
  24MP only to throw 23.4MP of it away again is 288MB and a second
  interpolation for pixels within rounding of one pass.
* **A single slot, not a dict.** At most one `Frame` is ever needed, since both
  identity cases return the upload. Changing the target is a deliberate act
  taken rarely next to moving a slider, so rebuilding on a change is the right
  side of that trade — and a dict would grow one full-resolution array per
  target the user tried.

`f"{up.id}@{target:g}mp"` is the frame's id and it reaches `render_tier`'s
`checkpoint_id`. The engine's `_ckpt_key` also carries `h`, `w` and `scale`, so
two working resolutions of one photograph are separated three times over. That
is the right amount of redundancy for a cache whose stale hit renders a
plausible but wrong *photograph*.

### The metering deliberately does not move with it

`Frame.norm_stats()` delegates to the upload's. Normalize answers a *tonal*
question, and a colour correction that shifted when you changed working
resolution would be a surprise nobody asked for. It also keeps `norm_white` — a
channel maximum, the one metered value a resample genuinely does move — reading
the real photograph, and keeps a 24MP array from being materialised purely to
meter it.

## The double-scaling guard

`params_for` runs in this order and it is the part that must not be got wrong:

1. `sanitize`, so a junk or out-of-range target is already clamped to the
   slider's range and `at()` cannot be asked for a 4000MP frame.
2. `up.at(prescale_target(p))`.
3. `scale_factor(reference_mp, fr.w * fr.h / 1e6)` — **the frame, not the
   upload.** Measuring against the file here would resample the photograph to
   24MP *and* rescale every length for the 50MP frame it no longer is.

One genuine property falls out of it: pixel dimensions are integers, so a frame
prescaled to 1MP is 816×1225 = 0.9996MP and the factor lands *within a rounding*
of 1.0 rather than on it. `rescale`'s own `abs(k - 1) < 1e-6` early-out does not
catch that. It is not worth a special case — 0.02% of a 1.6px clump is 0.0003px,
and it shrinks as the frame grows, with a 3:2 frame at 24MP landing on exactly
24.0MP. Using the *nominal* target instead of the frame's real pixel count would
make the factor exactly 1.0 and introduce a second answer to "how big is this
frame", which is the kind of thing that drifts.

## The exported file, and why `prescale_output` exists

Requested as a menu choice rather than a fixed rule. `prescale_output` picks:

* **Prescaled size** (default, index 0) writes the frame that was actually
  rendered. Every pixel in it was computed rather than interpolated. A 50MP
  photograph exports at 24MP and a 6MP one exports at 24MP too — literally "as
  if you imported a 24MP photo", including the file.
* **Photo's own size** resamples the finished render back to the file's
  dimensions, for when something downstream expects them. It is a resample of
  *finished grain*, so going up it is soft and going down it averages grain
  away — the texture judged on screen is not quite the texture in the file.

Two implementation notes:

* **`imageio.resize_to`, not `upscale`.** `upscale` omits `antialias` because
  there is nothing to alias against when adding samples, which was true while
  the only direction was up. Prescaling can enlarge the input, so writing back
  at the photograph's own size can be a *reduction* of a grainy frame — and
  grain is nothing but content at the Nyquist limit, so an unfiltered reduction
  of it folds into visible crawl. `resize_to` turns `antialias` on as soon as
  either axis shrinks. `downscale` and `upscale` are unchanged and still
  byte-identical; all three now share one `_interp`.
* **The filename carries `_pre{mp}mp` whenever prescaling resampled the frame**,
  and unlike the supersample tag this one is not optional. With `prescale_output`
  writing the photograph's own size, a prescaled export and a plain one are the
  *same dimensions* and a different picture — exactly the case a folder listing
  cannot show, which is what every tag in that block exists for.

Index 0 must stay "Prescaled size" forever. `choices` order is load-bearing in
every saved preset file: appending is safe, reordering silently rewrites them.

## It is part of the look, and it boots muted

Asked for outright: the values are real schema `Param`s, they travel with a
preset file, they are undone by Undo, and every file in `presets/` is stamped
`prescale: 1, prescale_mp: 24`.

`prescale` is in `NEUTRAL_ZERO`; `prescale_mp` and `prescale_output` are not.
The rule from `docs/panel-layout.md` is that a section's mute button must not be
a lie — prescaling changes which pixels the pipeline is handed, so Original and
the mute button have to switch it off. The target is a *size*, like a radius or
a seed, so switching the section back on returns what you had dialled in rather
than 1MP.

The consequence, which is worth knowing rather than discovering: **the app boots
with every section muted**, so on a cold start Prescaling Source reads *off*
with 24MP staged behind its `○`. That is exactly how `Normalize` and every other
section behaves, and it is why the photograph opens untouched. "Default 24MP"
means the parameter default is 24 and on, every preset says 24 and on, and
picking any preset or unmuting the section prescales.

## The one panel exception

`GROUPS[0]`, but `App` renders this one section **above Size Scaling** rather
than at the head of the parameter panel. Size Scaling is a hand-written sidebar
block that sits above the preset picker, and the two sections answer the same
question from opposite ends; reading them a panel apart made neither make sense.

It is still generated from the schema by the same `SliderPanel`, fed one section
— a second instance of it, not a hand-built panel — so `docs/architecture.md`'s
rule that no view defines a control still holds, and mute, reset, the jump menu,
collapse-all, preset save/load and undo all need no special case. Both instances
take one shared `panelProps` object, because a second copy of that wiring is how
the two would quietly stop behaving the same way.

## What the client had to learn

Prescaling makes `meta.width / height / megapixels` facts about
the *file* rather than about the frame, and almost everything in the app that
quoted a size was quoting the wrong one. `models/prescale.ts` is the one place
the difference is worked out — `prescaleGeom` for the frame, `exportDims` for
the file — and `Stage`, `ScalePanel`, `ExportPanel`, `TopBar` and the proxy hint
all read it.

Two details:

* It is a **deliberate mirror** of `prescale_dims` and `Frame`, with the server
  as the authority — the same bargain `coerceValues` already makes with
  `sanitize()`. The rounding is written `floor(x + 0.5)` on both sides because
  Python's `round` is banker's rounding and `Math.round` is not, and they
  disagree on exactly the half-pixel case that a ratio-preserving resize is
  built around.
* `/api/upload` reports the proxy long edge, not a measured proxy size. The
  client cannot derive the ceiling from a measurement: on a photograph smaller
  than it, a measured `proxy_width` is the photograph's own width and says
  nothing about where the ceiling is. Since 2026-08-29 that is
  `proxy_edge_default` plus the bounds a request may move it over, because the
  edge became a property of the request and there is no longer one proxy to
  measure at all — `proxyOf` takes the edge the session is asking for. See
  `docs/preview-and-export.md`.

Nothing new goes on the wire. The three values ride in `params` like every other
control, which is the main dividend of making them parameters rather than
session preferences.

**The aspect ratio is preserved to within half a pixel per axis, and that is a
hard requirement rather than a nicety.** `/api/source` still serves the untouched
photograph at its original resolution, and the client lines the two up under the
before/after wipe by scaling both into one box. A ratio that drifted would show
as the two layers parting company at one edge — not as an error anywhere.

## Cost, honestly

* **Reducing a large photograph throws away real detail, and enlarging a small
  one adds none.** An upscaled 6MP frame is a 6MP photograph on a 24MP grid, and
  the grain drawn on top of it is the only thing actually at 24MP.
* **Memory.** A prescaled `Frame` holds its own proxy (~46MB at 2400px) and, once
  a 1:1 render or full export has asked for it, its own full array — 288MB at
  24MP, 1.44GB at the slider's 120MP ceiling — *on top of* the upload's. The
  proxy is what the drag loop uses and it is small; the full array is only built
  by deliberate one-off actions. `reap()` drops both with the upload, and
  changing the target releases the previous frame. The 120MP ceiling is the same
  one `imageio.MAX_PIXELS` already puts on an upload, so a prescaled frame can
  never be larger than a photograph the app would have accepted directly — but
  it can now be larger than the *file* you opened, which is new.
* **Render time** scales with the target. Prescaling a 6MP photograph to 24MP is
  four times the pipeline work for a picture with 6MP of detail in it.
