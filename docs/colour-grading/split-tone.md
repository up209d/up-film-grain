<!-- part of docs/colour-grading.md -->

## Tone Response's split tone runs both ways (rewritten 2026-08-06)

Filed here rather than in a file of its own because it is the same subject —
colour — even though the two sliders live in the **Tone Response** panel section
and run at step 5, deep inside the film pipeline, rather than in Colour Grading
at step −1.

`warm_highlights` and `cool_shadows` are gone. In their place:

| was | now |
|---|---|
| `warm_highlights`, 0…1, warm only | `highlight_warmth`, **−1 cool … +1 warm** |
| `cool_shadows`, 0…1, cool only | `shadow_warmth`, **−1 cool … +1 warm** |

Both ship at 0, so the pipeline is still a colour pass-through and the deferred
status of the section is unchanged. Every shipped preset carried them at 0, so
the rename was a key rename in eleven files and nothing else.

**Why signed.** One direction each could describe warm-over-cool and nothing
else — not tungsten stock's cool highlights, not a cross-process, not warm
shadows under a cold sky. Each end of the range now runs cool through neutral to
warm, which is the same two stages with the sign let out.

**One axis, pushed both ways, rather than a "warm" vector and a "cool" one.**
Two hand-written vectors are two things that can drift apart; a signed push
along a single axis is warm and cool by construction, and it is what makes 0
*exactly* neutral rather than approximately so. `verify.py` measures the
symmetry at 0.00e+00 for highlights and 1.5e-08 for shadows.

**The axis is projected onto the luma-null plane.** The raw direction is a warm
shift — red up, a little green, blue down — and pushing along it as written also
*brightens*, because its own luma is 0.248 rather than 0. Warming the highlights
would then lift them too, fighting Shoulder and Brightness for the same range so
that neither could be set independently. Subtracting the axis' luma from every
channel lands it exactly on the plane where the luma weights sum to zero, and
by linearity the shift is a pure change of colour at every setting and in both
directions. Measured luma drift at full strength: **1.6e-08**.

**And they were invisible, which is what was actually reported.** The old pair
added a fixed `[0.055, 0.012, −0.040]` weighted by `smoothstep(0.45, 1.0, luma)`,
so the peak shift was 0.055 in one channel and full weight arrived only at pure
white. An ordinary highlight at luma 0.7 got 0.019 — under two 8-bit levels,
which is a rounding error and not a look. Two things changed together:

* **Amplitude**, `_WARM_GAIN` 0.14 against an effective 0.055: a full-strength
  push moves blue by 36 levels at the top of the range. Visible without being a
  colour filter, and the pair at opposite signs is a split tone you can see at a
  glance.
* **The weighting bands**, widened to (0.30, 0.85) and (0.15, 0.70) from
  (0.45, 1.0) and (0.0, 0.5), so a real photograph's highlights and shadows
  actually take the setting instead of a fraction of it. They **overlap through
  the mid-tones on purpose**: disjoint bands leave an untinted stripe across the
  middle of the range, so setting both the same way would tint the top and the
  bottom of a gradient and miss its centre.

`verify.py` asserts a floor in **8-bit levels** rather than "it moved", because
"it moved" would have passed on the old code too. Measured 32 levels for
highlights and 34 for shadows, and each stays at its own end of the range —
0.0 levels of cross-talk at the other end, since the two bands reach exactly
zero there.
