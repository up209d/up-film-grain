<!-- part of docs/film-texture.md -->

## Film texture sits last, and is masked by nothing (added 2026-07-31)

Step 15, dead last, after sharpening: dust, scratches, hair, light leaks.
Everything above it models what the *emulsion* does; this models what happened
to the strip of film afterwards. It is weighted by none of the image masks — a
scratch does not care what is underneath it — and every parameter ships at 0.

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
