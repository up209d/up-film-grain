<!-- part of docs/film-texture.md -->

## Film texture is masked by nothing (added 2026-07-31)

Dust, scratches, hair, light leaks. Everything above it models what the
*emulsion* does; this models what happened to the strip of film afterwards. It
is weighted by none of the image masks — a scratch does not care what is
underneath it — and every parameter ships at 0.

### It is no longer last (moved 2026-08-09, on request)

It was step 15, below sharpening, and the argument for that was one sentence: a
speck of dust sits on the film, it was never in the picture, so it must not be
sharpened, grained or graded along with it. **Global Grain and Sharpening now
run below it** — sections 9 and 10 — so two thirds of that sentence no longer
holds. The panel moved with them, so `pipeline order == panel order` still does.

The half that survives is the half that was doing the work. *Grading* is still
above: Colour Grading is step −1 and Tone Response is step 7, so a LUT and a
characteristic curve still see a photograph rather than a dusty one, and the
marks are still weighted by none of the image masks. What changed is the two
stages that model the **print and the scan** rather than the negative, and those
genuinely do come after the film got dusty: a scanner photographs the debris
along with the frame, so the print stock's grain lies over it and any sharpening
in the scan pipeline bites on it.

What it costs, and it is not small:

* **Sharpening rings the marks.** `sharpen`'s own help says unsharp halos show
  on hard borders past about 1.2, and a speck is a hard border. Ten of the
  twelve shipped presets carry `sharpen 12`. Every one of them now puts a
  visible ring around every speck and every hair.
* **Global Grain's four source-masked layers key on the debris.** `_source_masks`
  reads the frame immediately below Film Texture now, so a black hair pulls the
  lightness bell down along its length and a leak drags the hue masks toward its
  own colour.
* **Halation no longer blooms the Global Grain layer**, and **Tone Response no
  longer develops it** — that layer is not compressed by the toe or the shoulder
  and not lifted by base fog.

Nothing in the backend had to change for it. `pad_for` is one order-agnostic sum
of every kernel and displacement in the pipeline and already reserved the
sharpen radius; a mark is drawn over the *padded* tile window, clipped to its own
footprint in absolute coordinates, so a kernel running below it reads correct
mark pixels inside the overlap. `_source_masks` is per-pixel and adds no reach.
Measured on a plate at 30 specks, 4 hairs, `sharpen 6` and `global_intensity 12`:
**6.1e-05** between a 96px tiling and a single pass, against the 2e-03 every
other tile-independence check here is held to.

One thing that is now load-bearing and was not stated before. Dust and hair
composite **in place**, and the tensor they are handed can be the one stored at
the checkpoint below Anti Aliasing — every stage between that boundary and this
one returns its input unchanged when switched off. `_tone` always ends in a
`clamp` and so always allocates, which is the only reason the specks stay out of
the cache. Tone Response sits directly above Film Texture, so it holds; move it
and a warm render starts drawing a second population of dust on top of the first.

**What must never happen is a mark list derived from the region being
rendered.** N specks per tile, or positions drawn against the tile's own area,
and an export grows seams or draws different debris in every tile. That is the
invariant. It is *not* the same as "never build a list", which is what this file
used to say, and the distinction cost a rewrite to get right — see below.

How each shape is made, and the measured result:

| mark | how | count | geometry |
|---|---|---|---|
| dust | drawn per speck: an ellipse with three angular harmonics, from a frame-anchored list | **exact** | round, mean axis ratio 1.47 |
| scratches | noise with cells ~2px wide and ~900px tall — the anisotropy *is* the scratch | ±50% | 76:1, 1.2px wide |
| hair | drawn per filament: a tapered curve with a quadratic sag and two wobbles | **exact** | 2.9px wide |
| light leak | oriented beams anchored on the perimeter, added in **linear** light | exact | 1.3:1 along/deep |

Scratches still go through `_weather()`, which is what stops a *field* of marks
looking generated. A thresholded field gives every mark an identical crisp edge
and identical opacity; real debris sits at different depths, so some is in focus
and some is not, and none of it is equally dark. `_weather` blends each mark
toward a blurred copy and scales its density, both driven by fields addressed at
*mark* scale — a whole scratch shares its blur and its density rather than
fading in and out down its own length. Measured at full softening: mean edge
slope down 26%, while the crisp-to-soft ratio *widens* (13.8x to 18.2x) and
per-mark brightness spread runs 69% of the mean. Both halves are asserted: a
uniform blur would pass a mean test and be exactly the artificial result this
exists to avoid. Dust and hair carry their own per-mark draws now and no longer
need it.
