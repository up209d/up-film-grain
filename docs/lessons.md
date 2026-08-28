# Things I got wrong, so you don't repeat them

* **Guarding the callers is not guarding the resource, and on MPS the penalty
  is the process, not a wrong answer** (2026-08-29). The idle flush that hands
  the allocator's free list back fires on a timer thread. First version took
  `runtime.RENDER_LOCK` before calling `release_cache`, which looks like the
  right lock -- it is the one every render takes. The whole check suite died:

      -[IOGPUMetalCommandBuffer validate]: failed assertion
        `commit an already committed command buffer'

  `torch.mps.empty_cache()` commits and synchronises the command queue, so
  running it while *any* thread has device work in flight aborts the process
  outright. And plenty of device work does not go through `render_image`: every
  check in `tests/checks/` that calls `render()` directly, and every upload,
  which resamples through `imageio._interp` on a request thread holding nothing
  at all. Moving the lock onto the engine did not help either -- it was the same
  mistake one object over.

  The fix is that the thing being protected is **the device**, so the guard has
  to live there: `device.device_work()` is a counter every GPU entry point
  increments, and `device.try_release_cache` does nothing unless it reads zero.
  A counter rather than a mutex because the exclusion is asymmetric -- many
  threads may use the device at once, which was already true and already fine,
  and only the release must be alone. A lock held for a whole render would have
  made an upload during a 24MP export wait for the export.

  The general shape: when a call is unsafe *concurrently with a resource being
  in use*, enumerate the resource, not the callers. Enumerating callers means
  being right about every path today and every path anyone adds, and the failure
  is a crash in a thread nobody was looking at.

* **`atexit` does not run on SIGTERM, and SIGTERM is how the desktop app
  quits** (2026-08-29). The disk cache registered its cleanup with `atexit` and
  swept dead `run-<pid>` directories on the next start, which covered a clean
  exit and a `SIGKILL` and looked complete. It missed the *normal* path:
  `electron/main.js` sends SIGTERM, Python's default handler terminates without
  running `atexit`, so every ordinary quit left a full run's spill -- up to the
  disk budget plus the open photograph's frames -- lying in `~/Library/Caches`
  until the app was next opened.

  Two things made this worth writing down. First, it was invisible in every test
  I ran, because I kept restarting the app and the sweep tidied up behind me;
  the bug only exists for the user who quits and does not come back. Second, the
  fix depends on a uvicorn detail that looks like it should not work: uvicorn
  installs its own signal handlers for the duration of `Server.run` and, on the
  way out, **restores the previous ones and re-raises the captured signal**. So a
  handler installed *before* `run()` is not shadowed, it is deferred. Installing
  one after `run()` returns is too late; replacing uvicorn's breaks the graceful
  shutdown. Check what your server framework does with signals before assuming
  either.

* **A check module's memory footprint is multiplied by the pool, and mine
  crashed the machine** (2026-08-29, reported by the user). `tests/checks/prescale.py`
  was written with realistic scenes -- a 22MP source, a 24MP one, a 6MP one --
  and prescaled frames of them, holding roughly **1.1GB live** for the whole
  function because every one stayed referenced to the end. On its own that is
  fine, and it passed in 5.4s at `-j 1`. The runner defaults to `cores - 2`
  workers, so it landed beside `edges`, `global_layers` and `film_tiling` each
  holding fixtures of their own, and the machine ran out of memory.

  What makes it a lesson rather than a slip is that **none of those arrays were
  earning anything.** Every branch the module tests -- the target arithmetic,
  the two identity paths, the cache, `proxy_scale`, the checkpoint id -- is
  scale-free, so a 0.24MP source exercises exactly the code a 50MP one would.
  The rewrite is sub-megapixel throughout, builds one 4MP frame for the single
  assertion that genuinely needs a frame wider than `PROXY_LONG_EDGE`, drops it
  again, and runs in 1.5s. A module is a unit of parallelism, so a big array in
  one is a big array in fourteen: reaching for a realistic size in a check has
  to be justified by a property that actually depends on it.

* **"Strictly increasing" is not "detail survives", and a check that confuses
  them will pass while the stage destroys the picture** (reported by the user
  2026-08-16). Normalize's highlight roll was verified by asserting the transfer
  was monotone over a 4096-step ramp — 0 non-increasing steps, exact, and
  completely worthless. Monotone permits increasing by a millionth per step,
  which is a flat white patch at 8-bit. Measured properly on a real photograph,
  the source band 0.70..1.00 — 77 levels of highlight — was arriving as **3.2
  levels**, and everything above source 0.5 rendered as white.

  This is the split tone's lesson one layer up and I still walked into it: if a
  control's failure mode is *being invisible*, the assertion has to be in units
  a human perceives. The module counts **distinct 8-bit levels surviving in the
  highlight band** now, on real photographs as well as synthetic ramps, and
  keeps the monotonicity check beside it explicitly labelled
  necessary-but-not-sufficient. A mathematically exact check on the wrong
  quantity gives more false confidence than no check at all.

* **A fixed knee cannot serve a variable gain.** The same bug's other half. The
  roll compressed everything above 0.82 — fine for a small lift, catastrophic
  for a large one, because at +2 EV every source value above 0.29 already lands
  above 0.82 and 70% of the tonal range had to be crammed into 18% of the
  output. The knee decides how much of the picture gets compressed while the
  gain decides how much arrives above it; pinning one while the other moves is
  the whole defect. The fix was not a better knee but no knee: an extended
  Reinhard tone map in linear light, whose only parameter is the frame's own
  measured maximum. Same photograph, same lift, **35.3 levels instead of 3.2**.

* **Optimising a provable property at the expense of the picture.** The same
  roll was sized from the frame's *true maximum* rather than a percentile, and I
  documented the reason as making the no-clip guarantee "unconditional rather
  than true-for-most-pixels". It read like rigour. What it actually did was let
  the 0.84% of pixels that were **already blown** — flat white, carrying no
  detail whatsoever — dictate the compression for the 99.16% that still had
  something in them. When a guarantee's cost is paid by the data you were trying
  to protect, the guarantee is the wrong one; ask what the pixels being sacrificed
  were worth before defending the bound.

* **A validity mask that is right for one estimator can invert the sign of
  another** (caught by measurement 2026-08-16, building Normalize). Auto
  exposure and auto white balance were metered over one shared "trustworthy
  pixel" mask, excluding anything clipped — which is correct for colour, since a
  clipped pixel's ratios are set by the ceiling rather than by the light. It is
  catastrophic for *level*. An over-exposed frame is mostly clipped, so the
  surviving samples are its **dark** pixels, and the log average over those says
  the photograph is dark. Measured on a frame 1.4 stops over, the metering asked
  for **+1.38 stops brighter**: the right magnitude with the sign inverted, on
  precisely the input the control exists to fix.

  Two things generalise. A mask encodes *a question* — "is this a good sample of
  the local colour" is not "is this a good sample of the local brightness" — so
  two estimators sharing one mask need the questions to actually match. And an
  auto control's failure mode is not being slightly off, it is being confidently
  wrong; every check on one has to assert the *direction* against a known-wrong
  input, because "the frame changed" passes just as happily on a correction
  applied backwards.

* **Sizing a correction from the picture instead of from the correction.** The
  toe that keeps shadow separation when Normalize darkens a frame was first
  sized from the frame's own black level. That is a different question, and it
  showed: a well-exposed photograph with genuine deep shadows measured a toe of
  0.216 and had its blacks lifted to fix a problem it did not have — the deep
  shadows were the photographer's. Darkening by `ev` stops compresses everything
  below the knee by exactly `2**ev`, so what the correction *cost* is a property
  of the correction alone, and brightening cannot crush a shadow at all. Ask what
  your stage did, not what the input looked like.

* **A positional slice of a list other people insert into is a tripwire**
  (2026-08-16). `checkpoint.py` decided which sections sit below each boundary
  with `GROUPS[3:]` and `GROUPS[6:]`. Adding `Normalize` at index 0 shifted both
  by one, so `GROUPS[3:]` silently stopped meaning "Grain Structure down" and
  started meaning "Pre Sharpen down" — putting Pre Sharpen below a checkpoint
  saved *after* it ran. That is character for character the stale hit the
  comment on that very boundary already records from 2026-08-09, reintroduced by
  an edit in a different file that never mentions checkpoints. The fix is not to
  be careful next time: both are `_from("<section>")` now, which takes the suffix
  by name, so the position cannot matter. A comment warning about an off-by-one
  is evidence the construction allows one.

* **"Runs below the checkpoint" does not mean "can be left out of its key"**
  (caught by `verify.py` on 2026-08-09). Moving Global Grain and Sharpening
  below Film Texture made the panel and the pipeline agree so exactly that
  `checkpoint._BELOW` looked like it could finally be a plain `GROUPS` suffix —
  every section under the boundary, sliced straight off the list. It cannot.
  `render()` evaluates the characteristic curve **twice**: at section 3 as a
  mask input, to get the density luma the grain band and Shadow Clumping key on,
  and at section 7 for real. So `Tone Response` sits below the boundary and is
  read above it, and dropping its keys made a `brightness` edit come back
  **2.3e-01** wrong against a cold render — a plausible, wrong photograph, which
  is the exact failure that cache is built against.

  The condition is "no stage above the boundary reads this section's keys", and
  it is strictly stronger than a position in the panel. It survived review
  because the slice *looked* like a tidy-up of a comment that had gone stale for
  an unrelated reason. The check that caught it re-renders one parameter from
  every section against a warm cache; keep it exhaustive over sections.

* **`_grain_points` is arithmetic-bound, so restructuring its loop buys
  nothing** (measured and reverted 2026-08-08). It is the single most expensive
  thing in the pipeline — 4.75s of a 7.95s `SuperPortra` proxy on the GPU, 60% —
  and the shape of it invites optimisation: `_GRAIN_SLOTS` x a 3x3 ring is 27
  full-frame iterations, `ny`/`nx` depend only on the ring offset yet are rebuilt
  in all 27, and each iteration does four separate advanced-index gathers.
  Hoisting the ring loop outside the slot loop (27 index builds -> 9), stacking
  position and radius so they gather in one op instead of three, and inlining
  `_smoothstep` is a clean rewrite that stays bit-exact.

  It measures **7.898s against 7.900s on the CPU and 1.742s against 1.745s on
  MPS**. Nothing. The loop is ~28 element-ops per iteration over 15.4M working
  pixels — roughly 13 G element-ops per call — so it is bound by streaming those
  tensors, not by kernel launches or index arithmetic. Fewer, larger kernels move
  the same bytes.

  What *would* move it is less work or narrower types, and both are refused
  elsewhere for good reasons: fewer slots or rings rerolls every preset's grain,
  and fp16 would quantise a falloff whose whole range is 0..1. If you come back
  to this, measure before restructuring — this is the second time an obvious-
  looking win here measured at 1.00x, after `inference_mode` and kernel caching
  in the 2026-08-04 audit.


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

* **A colour name in a parameter can mean the mask or the channel, and picking
  the wrong one builds a different feature that looks fine.** "Source Red,
  masked with the red channel" was first built as a grain layer confined to the
  red channel and masked by `R`. Both halves were wrong: what was wanted was a
  *full-colour* layer masked by how red the picture is, `R - max(G, B)`. Neither
  version fails a "does grain appear" check, and the wrong one renders something
  perfectly plausible. Two tells that generalise — a channel-value mask fires at
  full strength on **white and grey**, where every channel is high, so three
  colour masks collapse into one brightness mask; and if a mask is a hue, it has
  to be *zero* on neutrals, which is an exact assertion rather than a tolerance.
  When a request names a colour, establish whether it selects a region or a
  channel before writing anything.
* **A mid-tone bell and a brightness ramp pass the same one-ended test.** Both
  are quiet in the shadows. What separates them is the *highlight* end, where a
  ramp is loudest and a bell is silent, so a mask test that only measures the
  dark end confirms nothing. Measure both ends of anything shaped like a band.
* **Do not measure a tonal response on a ramp.** Slicing a gradient to compare
  "dark end" against "mid" samples a *different patch of the noise field* at
  each end, and the field's own local sigma varies enough between patches
  (measured 0.029 against 0.042 on neighbouring 24-column slices) to swamp the
  effect. Three flat plates at the same coordinates read the identical field and
  differ only in what is underneath it. This cost a real debugging detour: the
  "Add is flat across the tonal range" check failed at 79% and looked like a
  pipeline non-linearity for a while.
* **Test plates clip.** Layers stack, so a patch at 0.85 carrying a flat layer
  at ±0.076 and a masked layer at ±0.098 hits the final `clamp(0, 1)` — and a
  clamp does not fail a test loudly, it quietly eats one tail so every sigma
  measured afterwards is measuring the clamp. Add up every layer's peak swing
  when choosing a level, not just the one under test.
* **A fitted constant standing in for a count is not a calibration, it is
  evidence.** `_BLOB_CELLS_DUST = 14.0` and `_BLOB_CELLS_HAIR = 0.5` converted
  "area above a threshold" into "number of countable marks", and both were
  honestly documented as good to a factor of 1.5. I read that as the price of
  the construction. It was not — it was the construction telling me it could not
  express what the slider's label claimed, and the user eventually reported the
  consequence ("I see more than 1 hairs when I set to 1"). **When a control needs
  a fudge factor to mean what its label says, the label is describing a
  different implementation.** Both constants are gone and the counts are exact.
* **"Never build a list of objects" was the wrong rule.** The real invariant is
  narrower: never derive a list from *the region being rendered*. A list built
  from the count, the seed and the frame is as tile-independent as any noise
  field — `_leak_sites` had been proving that for a week while `film-texture.md`
  told the next reader not to do it. An over-broad rule is worse than none,
  because it forbids the correct solution in the same breath as the wrong one.
* **A line thinner than a pixel renders as a dashed line.** It only registers
  where its centre passes near a pixel centre. A tapering hair does this at its
  tip and the detached fragments read as extra marks — which is the exact bug
  the rewrite was meant to fix, reappearing in a new form one layer down. Draw
  at a floor of one pixel and fade by what is missing, which is what
  area-averaging would have done. A *disc* is safe at half that, because it
  always has a pixel centre within reach of its own soft edge.
* **The vertical-gap-over-`sqrt(1+slope²)` distance to a curve is a
  small-slope approximation and it fails loudly.** Where a wobble is steep
  enough to double back within a pixel or two, a point genuinely on the curve is
  scored against the wrong part of it and the mark comes out in pieces — a fifth
  of the hairs did. Cap the *slope* rather than the amplitude: it keeps the
  approximation valid however fast the curve ripples, and it happens to be the
  physical answer too.
* **Five uniform random points look clumped, and the statistics being innocent
  is no defence.** Four of the first five hairs landed in the top fifth of the
  frame; over 400 marks the same draws are uniform to 1% and uncorrelated to
  0.02. Small counts need a low-discrepancy sequence, and it must be one whose
  *prefixes* are well spread (R2, or the golden step leaks already use) so that
  raising the count adds a mark instead of rerolling the frame.
* **"Does it change the picture?" is the wrong question for a subtle control.**
  `warm_highlights` and `cool_shadows` shipped for weeks doing something real
  and invisible: a peak shift of 0.055 in one channel, at full weight only at
  pure white, so an ordinary highlight moved by under two 8-bit levels. Every
  check passed. If a control's plausible failure mode is *being too faint to
  see*, assert a floor in units a human perceives — `verify.py` now measures the
  split tone in 8-bit levels.
* **A colour shift that is not luma-neutral is two controls fighting.** Pushing
  along a warm axis as written also brightens, because the axis has a luma of
  its own. Project it onto the plane where the luma weights sum to zero and the
  shift is pure colour by construction, in both directions and at every setting
  — which is what lets Highlight Warmth and Shoulder be set independently
  instead of chasing each other.
* **A mask must be measured where its meaning is settled, not where it is
  used.** The Luminance Response mask asks how dense the negative is, which the
  characteristic curve decides — but it was computed three stages later, after
  edge softening, jitter and sanding. Softening a border invents a mid-tone ramp
  that was never in the photograph, and the mask believed it: a **0.095 sigma
  ribbon of grain** along a border whose two sides were both meant to be clean.
  The same reasoning already had the edge mask and the smooth-area guard reading
  the untouched input; this one was simply missed.
* **"Where does this run" is two questions, and a panel answers the wrong one if
  you are not careful.** I moved `Luminance Response` above `Grain Structure` in
  the panel because the engine *measures* its mask before it builds the field.
  The user pushed back — "its main purpose is suppressing grain from grain
  structure" — and they were right: where the mask is **applied** never moved,
  and application is what a user reasons about. Ordering a panel by the internal
  read order put the suppression above the thing it suppresses. It ended as a
  merge rather than a reorder, which was the better answer to the original
  question anyway: Luminance Response was never a stage, so a heading of its own
  had always claimed too much.
* **Moving one half of a physical question is worse than moving neither.** When
  the luminance band moved to read the developed density at step 6b, Shadow
  Clumping was left reading the late luma at step 10 — and both ask exactly the
  same thing, *how dense is this area*. Two controls keyed on one quantity,
  sampling it at two points in the pipeline, is a bug that no single-control
  test can see. Find the other readers of a value before you move where it is
  read.
* **A popover's drop direction is measured, not declared.** The LUT menu shipped
  with a hard-coded `drop="down"` and a `60vh` list, which is correct exactly
  when the trigger is near the top of the window. It is not, most of the time:
  the LUT row is well down a long parameter panel, so with the trigger 30px off
  the bottom of an 800px window the panel ran **130px past the window edge and
  22 of its 33 rows were unreachable**. The seven root LUTs are the first rows,
  so they stayed pickable and every LUT in a folder did not — reported as "I
  can't select any luts from gmic", which sounds like a data or id bug and was a
  layout one. `Popover` now measures the space on each side of the trigger,
  keeps the stated direction only if the other side does not have more room, and
  caps the panel's height to what is actually there.
* **A plain `<div>` in the middle of a flex chain silently breaks it, and the
  symptom is clipping rather than an error.** Capping the panel's height only
  works if the cap reaches the scroll box. `MenuBody` returned its contents
  wrapped in an unstyled `<div>` carrying the keyboard handler, and an unstyled
  flex child has `flex: 0 1 auto` and `min-height: auto` — so it kept its
  natural height, the list never shrank, and the panel clipped its overflow
  instead of scrolling it. The give-away was a browser check reporting
  `scrollHeight === clientHeight` on a 76-row list. Every element between a
  height cap and the thing that scrolls needs `min-height: 0` and to flex.
* **Dispatching `el.click()` to dodge a test framework's auto-scroll throws away
  the thing you were testing.** Playwright scrolls before clicking, which moved
  the panel and wrecked my measurements, so I switched to
  `el.evaluate(e => e.click())`. That fires on elements that are clipped and
  invisible — so the clipped-list bug above passed a check written to catch it.
  If the assertion is "a user can reach this", the click has to be the one that
  fails when they cannot.
* **A glob in a packaging script is a silent filter, and it fails the day the
  folder gains a subfolder.** `build.sh` bundled LUTs with `cp luts/*.cube`,
  which was right while `luts/` was flat and wrong the moment it was not: the
  distribution shipped **7 of 303**, the LUT menu showed the seven at the root
  and no folders at all, and nothing in the build output said so — it cheerfully
  reported "Bundled 7 LUT(s)". It surfaced as "I can't select any luts from
  gmic" against a *running distribution*, while the repo checkout served all 303
  correctly, which is the worst version of this: the bug is invisible from the
  place you develop. `verify.py` cannot catch it either — it reads `luts/`, not
  `build/luts/`. Two rules fall out of it: a packaging step must walk what it
  copies, and its report must count what it *wrote*, not what it matched.
* **`focus()` scrolls every ancestor, and `scrollIntoView` does too.** The LUT
  menu focuses its search box on open. That box sits at the top of the panel,
  and an upward-opening panel puts it *above* the sidebar's visible area — so
  the browser dutifully scrolled the sidebar to reveal it, throwing the reader's
  place away by up to 694px on every open. `focus({ preventScroll: true })` is
  the fix, and the keyboard cursor's `scrollIntoView({ block: "nearest" })` had
  the identical disease: it walks *all* scrollable ancestors, one of which is
  the sidebar. Nudging the list's own `scrollTop` by hand is the version that
  cannot reach outside itself. Rule: inside a floating panel, neither call is
  safe — both talk to the whole ancestor chain, not to the thing you are
  looking at.
* **A key handler on the panel misses everything when focus never enters it.**
  `SelectMenu`'s arrow-key handling lived on the panel body. That works for the
  searchable LUT menu, whose search box takes focus — and silently does nothing
  for the five menus with no search box, where focus stays on the trigger
  *outside* the panel. Arrows did not move the cursor, and worse, reached the
  browser and scrolled the sidebar instead. Two fixes, because they answer two
  different questions: `Popover` swallows the scroll keys at the document level
  while open (they belong to the open panel wherever focus is, though never
  while someone is typing), and the list takes `tabIndex={-1}` so it can hold
  focus when there is no search box.
* **Separators are punctuation, and a search box should not care about them.**
  Typing `kodak portra` matched nothing, because on disk it is
  `kodak_portra_400`. A literal substring test makes the user guess the
  filename's punctuation. Normalising both sides — lower-case, `_`/`-`/`/`/`.`
  to spaces — and requiring every word to appear somewhere in `group + name`
  makes `kodak portra`, `bw agfa` and `PORTRA` all land where you would expect.
