# Things I got wrong, so you don't repeat them


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

