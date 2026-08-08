<!-- part of docs/colour-grading.md -->

## Shadows and Highlights were a brightness shift, not a recovery (fixed 2026-08-05)

Reported, and correct: "you just low down brightness in already lost highlight,
raise brightness on lost shadow, they are already lost so there is no recovery at
all, it is just pure brightness shift." Three separate defects were behind it,
and all three are now fixed. The stated priority for the fix was **highlight
detail above all else** — "film tends to have very detailed highlight, so I don't
want to lose highlight at all" — with performance explicitly accepted as a cost.

### Defect 1: the recovering directions were non-monotonic — they *inverted* tone

The construction was `x + a·m(x)·(1-x)` (or `·x` going down), with `m` a quintic
in the pixel's own luma. Read as a transfer function, the `m'` term competes with
the identity, and in the two *recovering* directions — **Shadows positive and
Highlights negative, exactly the two anyone reaches for** — it wins:

| setting | min transfer slope | non-monotone span |
|---|---|---|
| Highlights −1.0 | **−0.211** | 15.9% of the range |
| Highlights −0.7 | +0.152 | 0% |
| Shadows +1.0 | **−0.211** | 15.9% |
| Shadows +0.7 | +0.152 | 0% |

A negative slope is not compression, it is a **fold**: tonal order reverses, and
around the zero crossing the region goes genuinely flat. So the control turned a
highlight into the textureless patch it was supposed to rescue — precisely what
was reported. Even at −0.7, where it is technically monotone, slope 0.152 means
85% of the local contrast is gone.

The two *expanding* directions (Highlights positive, Shadows negative) were
monotone all along. That the broken pair is exactly the pair named in the report
is the confirmation that this was the bug and not a taste disagreement.

### Defect 2: everything above the tone stage clipped, so there was nothing left to recover

`_grade` clamped to 0..1 after **every** stage. White balance and Exposure sit
above Shadows/Highlights, so a stop of exposure was rounded off to white *before*
the recovery control ever saw it. "Recover the highlight" could only ever mean
"recover what is left of the highlight after we threw it away."

The section now clamps **once**, after the tone stage. Headroom from
reconstruction, white balance and exposure all survive into the curve that is
meant to roll it back in. With both tone controls at 0 this is bit-identical to
before — a monotone brightening followed by a clamp is the same picture whichever
end the clamp sits at — so nothing existing moved.

### Defect 3: an 8-bit file's clipped channels were never reconstructed

See the next section. "The source surely has those details" is *true* in a
specific and exploitable way, and nothing was exploiting it.

### The new construction: `_tone_roll`, on the channel maximum, uniformly scaled

Two decisions on top of the curve itself:

**The curve is monotone by algebra, not by luck.** Recovering, it is a convex
blend of the identity and the exponential shoulder `1 - exp(-t)`, so at full
travel the rail becomes an **asymptote**: the whole of `[knee, ∞)` folds into
`[knee, rail)` with ordering intact, and slope is bounded below by `exp(-t) > 0`.
Expanding, it keeps the old share-of-headroom form (which was already monotone
and clip-free). The asymmetry is the same shape of decision Clarity's is: pushing
a tone at a rail and pulling one off it are different operations.

**It keys on the channel maximum and scales all three channels together.** The
*value*, not the luma, is the right question for a control about clipping — a
saturated red at (1, 0, 0) has a channel hard against the ceiling while its luma
is 0.21, and the old luma key called that a shadow. And a uniform scale cannot
move a ratio, so hue and HSV saturation are held **exactly** (measured 2.7e-07)
rather than approximately, while gamut safety becomes structural: the curve's
output is bounded by the rail, so every channel, being at or below the maximum,
is too.

The two halves also stop needing a shared reference. Highlights only touches
`v > knee` and cannot push a value below it; Shadows only touches `v < knee` and
cannot push one above. Disjoint supports means independence by construction —
stronger than the "one luma measured before either runs" bookkeeping it replaces,
and impossible for a later edit to get wrong. `verify.py` pins the far end as
**bit-exact** now rather than merely close.

Measured on a real photograph (the repo's own 16×9 test frame, 2.1% of red at the
ceiling) — detail retained in the region each control is meant to rescue:

| setting | before | after |
|---|---|---|
| Highlights −0.5 | 54.4% | **78.6%** |
| Highlights −1.0 | 19.2% | **57.3%** |
| Exposure +1, Highlights −1 | 27.9% | **48.4%** |
| Shadows +0.5 | 66.4% | **97.9%** |
| Shadows +1.0 | 33.2% | **97.5%** |

One measurement in there is worth reading twice. Inside the frame's clipped-red
region, red's variation under the old Highlights −0.5 measured **6.1e-02** —
*fifty times the file's own* 1.2e-03. That was not detail; it was the fold
manufacturing structure that is not in the photograph. The new curve reads
8.2e-04 there (honest: the region is flat in the file) and 1.8e-02 with
reconstruction on (real, recovered structure).

**The controls are deliberately gentler in level than they were.** Shadows +1
used to take the frame's mean luma 0.209 → 0.441; it now takes it to 0.271. That
is the trade being made on purpose: the old reach *was* the wash-out. Exposure is
the control for level. A sweep of a steeper shoulder gain was run and declined —
gain 2.0 reaches mean 0.308 but drops the worst-case slope from 0.184 to 0.068,
i.e. it crushes the deepest blacks to buy reach. `_tone_roll` has no gain
constant for that reason; the ±1 endpoint is fixed by the curve's own shape
rather than chosen.

The check that matters is on the **slope of the transfer**, over 12 settings, not
on the mean level. A mean-only test passes happily on a curve that has folded
over — which is how this shipped in the first place.
