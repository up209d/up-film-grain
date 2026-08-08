<!-- part of docs/global-grain.md -->

## The section became five layers and a blend mode (2026-08-05)

Asked for as *"based on the Global Grain config (min, max, chroma grain) →
render 5 layers of noise, 1 for nothing, 1 for red, 1 for green, 1 for blue …
then slap them on top of each other"*, with a blend menu over the stack.

So: five layers, **built identically** from one set of shape controls — Size
Min, Size Max, Smoothness, Chroma Grain — differing in exactly two things.

| layer | amount | seed offsets | mask |
|---|---|---|---|
| flat | Global Intensity | 7717 / 3391 | none |
| red | Source Red | 11003 / 12007 | `clamp(R − max(G, B))` |
| green | Source Green | 13009 / 14011 | `clamp(G − max(R, B))` |
| blue | Source Blue | 15013 / 16033 | `clamp(B − max(R, G))` |
| lightness | Source Lightness | 17011 / 18013 | `t²(3−2t)`, `t = 1 − |2L − 1|` |

One `_global_grain_field(..., idx)` builds all five; `idx` selects nothing but
the seed pair. Layer 0's pair is the historical one, so it is byte for byte the
field the method built before the set existed — verified against the committed
engine across all seven shipped presets at 0.000e+00.

### Two things this got wrong on the first pass, and they are worth recording

The first implementation of this feature shipped four layers that were
**single-channel** — Source Red wrote into the red channel only — and masked
by the raw channel value, with Lightness a plain brightness ramp. All three
were wrong, and the correction is the whole design:

* **The colour names are the *mask*, never the output channel.** Each layer is
  a full-colour field written into all three channels, and each takes
  `global_chroma` like every other layer here. "Red" says only *where it shows*.
  The single-channel version renders as something perfectly plausible and is a
  different feature; `verify.py` now pins "Source Red writes to all three
  channels" and, at chroma 0, that the three deltas are the same number.
* **The mask is hue dominance, not channel value.** `mask = R` puts all three
  colour layers at full strength on white and on grey — where every channel is
  high — so they pile up into a brightness mask wearing three sliders, and the
  colour names mean nothing. `R − max(G, B)` factors exactly into *how red in
  hue* × *how bright*, which is what was actually asked for ("more visible if
  the area is more red, less when it has less red") and needs no calibration
  constant to say it. Two consequences: the three are **mutually exclusive**,
  since only one channel can be the largest, so they can never stack on each
  other; and hue dominance rarely passes 0.3–0.5 in a real photograph, so these
  sliders read quieter than Global Intensity at the same number.
* **Lightness is a mid-tone bell, not a ramp.** Grain peaks at mid grey and
  fades to nothing toward *both* white and black, which is where film grain
  actually lives — a blown highlight has no silver left to be grainy and a solid
  black has none developed. A ramp passes a dark-end test just as convincingly
  and is then *loudest* at white, so the check measures both ends. The
  smoothstep on top of the triangle is not cosmetic either: it flattens the
  approach to both extremes, so the layer leaves the highlights gradually rather
  than at a constant rate.

The rectified `clamp_min(0)` on a neutral area is the one place the hue mask
could misbehave — the difference there wanders either side of zero, so
rectifying it leaves a small positive envelope where the answer should be
nothing. Measured on a flat 0.5 plate with the main grain at 40 and the flat
layer at 20: Source Red at 100 renders sigma 0.000197 against the flat layer's
0.038469, 0.5% of it, with a mean shift of +1e-6. Blurring the mask would remove
even that and would cost `pad_for` a kernel it does not otherwise need.

### Masking is safe where seeding would not have been

Deriving each grain's *seed* from the source pixel — the obvious reading of
"grain that follows the picture" — fails three ways at once, and the shape of
the trap generalises:

* **It rebuilds the grid.** A flat region of one source value hashes every pixel
  to the same draw, so every grain in it is the same size, brightness and
  colour, centred on an integer pixel with an axis-aligned pitch of 1. That is
  the defect `_GRAIN_ROT` exists to prevent, reconstructed exactly — and a 1px
  pitch is maximally commensurate with the pixel grid.
* **It degenerates.** One grain per pixel, centred on that pixel, means every
  pixel's own falloff is 1, so `peak` is 1 everywhere: no gaps, no grain edges,
  and the construction collapses to a spatially-varying blur of white noise.
* **It cannot be cached, and it swims.** A seed drawn from the frame changes
  with every upstream slider, so grain re-rolls while you grade.

A **mask** has none of that. The pattern comes from the seed as it always did,
so only the envelope moves: nothing re-rolls, no grain shifts under a grade, and
the fields stay fully cacheable because the field still reads no image data —
the mask is applied by the caller, outside the cache boundary. That split is the
whole design, and it is why this landed as a per-pixel multiply.

### Different seeds, and why that is the opposite of `global_chroma`

Five `_grain_points` geometries rather than one, so a red-masked grain and a
blue-masked grain sit in genuinely different places. That is deliberately the
opposite choice from `global_chroma` one level down, which shares one geometry
across channels precisely so a single grain can take a colour without its edge
moving from channel to channel.

Both are wanted and they are not substitutes: chroma gives *coloured grains*,
this gives *differently placed grains picked out by colour*. `global_chroma` is
therefore not superseded — it governs each layer's own colour, and it reaches
all five.

The seed offsets are spaced so that none equals another's `+991`, which is the
cluster draw inside `_grain_points`. A collision there would have two layers
sharing the clump pattern that decides where grain bunches — not obviously wrong
in a render, and the pair would quietly read as one layer.

### `global_seed` is an offset, not a seed

Added on request so the section can be reshuffled without disturbing main grain
that is already dialled in — the same job `texture_seed` does for Film Texture.
It is deliberately **not** built the way that one is. `texture_seed` is an
absolute value independent of `seed`; `global_seed` is added to `seed`, which
buys two properties that an absolute seed here would have cost:

* **Seed still rerolls the whole frame**, including this section, which is what
  its own help text promises and what makes it the one control for "give me a
  different roll of everything".
* **0 is bit-identical to the layer that existed before the slider did** — for
  every preset, not just the ones at the default seed. That distinction is not
  hypothetical: `ClassicSoft` ships `seed: 4421`, so an absolute `global_seed`
  defaulting to 1234 would have silently rerolled its global grain.

The five per-layer offsets are applied on top of the sum, so moving it takes all
five together and leaves their relationship to each other alone — reshuffling
the section cannot collapse two layers onto the same geometry.

`verify.py` pins the three separable failures: it rerolls (correlation −0.02),
it is inert at 0 (0.00e+00), and it leaves the main grain untouched with the
global section off (0.00e+00). Unlike the five amounts it *must* miss the
texture cache, and that is asserted as the counterpart to the amounts hitting
it.

### The blend modes

Composited the way layers in an image editor are: the grain is an image
`L = 0.5 + g/2`, mid grey where there is no grain; the mode combines it with
what is underneath; and each layer's amount × mask acts as its opacity. Layers
go on in order, each onto the result of the one before.

`_grain_delta` returns the **difference** rather than the blended result, which
is what makes the opacity a plain lerp at the call site and lets the per-pixel
mask reuse one code path. Two things fall out of that and both matter:

* **Add returns `g` untouched.** Not an optimisation — reconstructing it as
  `(base + g) − base` is not the same float, and Add is the default that every
  shipped preset renders through.
* **The blend is computed against `base` clamped to 0..1, but the delta is added
  to the *unclamped* frame.** Overlay and friends are only defined on 0..1 and
  `out` is deliberately unclamped here, so a blown highlight keeps the headroom
  step 14 relies on instead of being flattened on its way past.

Measured response, sigma on flat plates at three levels (amount 25, identical
field at all three, so this is the mode and nothing else):

| mode | dark 0.12 | mid 0.50 | light 0.88 | mean shift |
|---|---|---|---|---|
| Add | 0.04809 | 0.04809 | 0.04809 | 0.0000 |
| Overlay | 0.00577 | 0.02404 | 0.00577 | −0.0000 |
| Soft Light | 0.00786 | 0.01101 | 0.00400 | −0.0008 |
| Hard Light | 0.02637 | 0.02404 | 0.02642 | −0.0000 |
| Multiply | 0.00289 | 0.01202 | 0.02116 | **−0.0238** |
| Screen | 0.02116 | 0.01202 | 0.00289 | **+0.0237** |

Add is the only even-handed one, which is why it is the one that can lift a
black. Overlay and Soft Light taper toward both ends on their own — that is the
reason to reach for them. **Hard Light does not taper**, which is worth knowing
because it is the one people assume does: it is driven by the grain rather than
by the image, so each half of the field tapers at a different end and the two
average out flat. Multiply and Screen have no neutral value in 0..1 at all, so
they carry a constant darkening or lightening that the grain then modulates;
they are here because they were asked for, and their help text says so.

`verify.py` pins each mode by the property that distinguishes it rather than by
"it changed something", and pins the four neutral modes as **exactly** neutral
with a grain-free layer.

### `pad_for`'s gate had to widen, and this is the one that would have shipped broken

The global-smoothing term was gated on `global_intensity`, but these layers run
through the same `_smooth_noise`. With the flat layer at 0 and a source layer
up, the blur still happens and nothing was reserved for it — a seam along
exactly the smoothing radius, invisible in every preview. Measured after the
fix: 51px reserved against 33px with the section off.

### Cost: all of it lands on the cache miss

Per `render()` on MPS at 1536², best of three, GPU synchronised:

| | cold | cached |
|---|---|---|
| section off | 42ms | 40ms |
| flat layer only | 106ms | 40ms |
| flat + Source Lightness | 172ms | 45ms |
| flat + all four | 400ms | 46ms |
| all four, flat layer off | 317ms | 47ms |
| flat + all four, chroma 1 | 888ms | 49ms |

So one extra field is ~66ms and four are ~294ms — linear, which is what
independent geometry costs and the reason it is not free the way sharing
`nfields` would have been. Chroma doubles it again, because each layer then
builds a second three-field geometry. **On a repeat render the whole set costs
about 6ms**: the five amounts, the masks and the blend are all applied outside
the cache, so dragging any of them cannot miss it. What misses is Size Min/Max,
Smoothness, Chroma and Seed.

The blend modes cost 4–23ms of that cached figure — Add 48ms, Multiply and
Screen 52–54ms, Overlay and Hard Light ~60ms, Soft Light 71ms, which is the
`sqrt` and the two `where`s in the W3C curve. All four amounts at 0 is not
merely bit-identical but free: each field is built only when its own slider is
up.
