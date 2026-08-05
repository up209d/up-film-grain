# Panel layout

## The Colour section is gone (2026-08-03)

On request, alongside the Global Grain chroma slider (see
`docs/global-grain.md`). The section merge is pure UI — group names live only in
`params.py` and the client generates the panel from them, so it is five `group`
strings and one line off `GROUPS`. Each parameter went to the section that owns
its *mechanism* rather than all five landing in one place:

| was | now | why |
|---|---|---|
| `chroma_grain`, `seed` | Grain Structure | properties of the grain field itself |
| `edge_chroma` | Edge Destruction | it modulates `edge_erosion`, and does nothing without it |
| `warm_highlights`, `cool_shadows` | Tone Response | colour grading, deferred, ships at 0 — the same as everything already in there |

That is a judgement call on top of what was asked: "Colour" was a grab-bag of
three unrelated jobs, and putting the fringing slider directly under the erosion
slider it modifies is more discoverable than the merge alone would be. One
`group` string each to move if it reads wrong.

## The panel order, again (2026-08-04)

A second reorg on top of the Colour merge above, this time touching `GROUPS`
only — no parameter changed section. `Optical` is gone the same way `Color`
went: its six params (`scatter*`, `micro_blur`) took on group `"Edge
Destruction"`, so the mechanism they share with jitter and sanding — tearing
detail apart before grain goes on — now lives under one heading instead of two.
Nothing else about them changed; the pipeline still runs scatter and micro-blur
where step 1/1b says, and `verify.py`'s per-stage checks key on parameter values,
not on group names, so none of them needed touching.

`GROUPS` itself, before and after:

```
before                          after
------                           -----
Pre Blur                        Pre Blur
Pre Sharpen                     Pre Sharpen
Grain Structure                 Grain Structure
Luminance Response              Edge Destruction   (+ former Optical)
Edge Destruction                Anti Aliasing
Halation                        Global Grain
Optical                         Sharpening
Anti Aliasing                   Luminance Response
Tone Response                   Halation
Global Grain                    Tone Response
Sharpening                      Film Texture
Film Texture                    Output
Output
```

Read as one story rather than twelve independent moves: everything that tears
the image apart at the pixel and edge level -- scatter, micro-blur, jitter,
sanding, then the anti-alias pass that cleans up stair-stepping, then the two
overlay layers that ride on top of the result (Global Grain, Sharpening) -- now
runs as one uninterrupted block. Luminance Response and Halation, both about how
light behaves rather than how detail is destroyed, sit together right after it,
directly ahead of Tone Response. Grouping by request rather than by re-deriving
a rationale from scratch — the four moves were independent asks and this is
simply where they land in combination.

## The Global Grain section grew four sliders and a menu (2026-08-05)

Source Red / Green / Blue / Lightness, added inside the existing **Global
Grain** group rather than in a new one — no `GROUPS` change. They belong beside
Global Intensity because they share every shape control it has (Size Min, Size
Max, Smoothness, Chroma) and are on the same 0–100 amplitude scale, so splitting
them out would have implied a second, independent layer stack when they are five
layers of one section. Global Opacity governs all five, and its help text says
so. See `docs/global-grain.md` for what they do and why masking, not seeding.

`Global Seed` goes **last in the group**, after Global Opacity, matching where
`Texture Seed` sits in Film Texture — a seed is the thing you reach for least
often and it belongs under the controls it rerolls, not among them.

`Blend Mode` goes **first in the group**, above Global Intensity, which is the
one departure from this panel's usual pipeline order. It governs all five
sliders under it, and a control that changes what everything below it means
reads wrong sitting underneath them. It is the section's second `choices`
parameter after `scatter_pattern`, so the client renders it as a menu with no
change to `App.tsx` — the value is still a number everywhere else, and the
names live in `params.GLOBAL_BLENDS` where the engine imports them rather than
in two literal tuples that would have to agree on an index.
