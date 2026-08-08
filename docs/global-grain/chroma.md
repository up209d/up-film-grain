<!-- part of docs/global-grain.md -->

## Global Grain grew a chroma slider (2026-08-03)

On request, at the same time as the Colour panel merge (see
`docs/panel-layout.md`). Same job as `chroma_grain` on the main grain layer —
decorrelate the three channels so this layer carries colour speckle instead of
pure luminance noise — and deliberately a different construction. The
main grain draws three independent fields and blends out from their rescaled
mean. Copying that here fails twice:

* **It would reroll every existing preset.** The mean of three fields is not the
  single field this layer has always been built from, so chroma 0 would render a
  different pattern than the one `Stock` was dialled in against. `verify.py`
  pins chroma 0 as bit-exactly monochrome (max channel spread **0.00e+00**).
* **That blend does not hold amplitude.** The mean and the per-channel fields
  are correlated, so measured pre-clamp it dips to **88.8%** of its own strength
  at chroma 0.5 and returns to 99.9% by 1.0 — the slider moves loudness as well
  as colour, which is exactly the coupling `_SMOOTH_GAIN_K` exists to prevent
  one slider along.

So the mono field is left alone and a **mean-zero** deviation `d` is added on
top, from its own seed. Because `d` sums to zero across channels its statistics
are fixed — var `2/3`, covariance `-1/3` of a single field — which makes the
coefficients solvable rather than a matter of taste:

```
g_c = A·m + B·d_c,    A = sqrt(1 - 2/3·c),  B = sqrt(c)
```

giving unit variance and cross-channel correlation **exactly `1 - c`** at every
setting. Measured 1.000 / 0.497 / -0.003 at chroma 0 / 0.5 / 1, with pre-clamp
amplitude flat to 0.6%.

**The one thing that does move is the clamp, and it is worth knowing why.**
Mixing in `d` gaussianises the field, so it reaches the `±1` rails less often —
clipping falls **25.4% → 22.8%** across the slider. A clipped sample sits at
exactly ±1 rather than wherever it was headed, so *less* clipping means slightly
less measured sigma: rendered amplitude drifts 100% → **97.0%**. That is the
hard tails doing their job rather than the blend leaking, and it is a third of
the wobble the other construction has. `verify.py` allows 5%.

Gated on `global_chroma > 0.001`, so the second `_fbm` is not paid for at the
default. No `pad_for` change — the field is addressed in global coordinates on
the same cell as the mono one and goes through the same smoothing kernel, which
is already covered.
