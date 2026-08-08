# Panel layout

## The edge mask moved to Edge Destruction (2026-08-09)

`highpass_radius`, `edge_sensitivity` and `edge_chroma_sense` were under `Grain
Structure`, because that is the stage that *builds* the mask -- the grain's own
`edge_bias` needs it there, before any Edge Destruction stage runs. They are
under `Edge Destruction` now, on request, and the reasoning is worth keeping
because it cuts against the rule the rest of the panel follows.

**These three are the one place the panel deliberately reads ahead of
execution.** Everything else in the app is arranged so a section runs where it
is listed; these are consumed in section 3 and listed in section 4. The trade is
made on purpose: they define *what counts as an edge* -- the scale it is
measured at, how hard it has to step, whether a colour boundary counts -- and to
anyone using the app they are edge controls. Leaving them under a heading that
does not mention edges was the more confusing of the two wrongs.

`edge_bias` stayed behind, and the split is not arbitrary. Those three ask *what
an edge is*; `edge_bias` asks *how much the grain should care*, which is a grain
question. Measured, that is also exactly where the coupling lives: with every
Edge Destruction control off, raising `edge_sensitivity` from 1 to 4 moves 13.4%
of a frame's pixels at `edge_bias` 1.0, 10.5% at 0.5, and **0.0% at 0**. Turn
`edge_bias` off and the three have no effect on grain whatsoever -- which is what
its help text now says.

Ordered mask-first within the section, since every control below them is
weighted by what they produce.

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

(Those two were renamed to `highlight_warmth` and `shadow_warmth` on 2026-08-06
when they became bidirectional; the section they landed in is unchanged. See
`docs/colour-grading.md`.)

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

## The Luminance Response section is gone, merged into Grain Structure (2026-08-06)

Asked for in two steps, and the second step corrected the first — worth
recording because the correction is the useful part.

**What I did first, and why it was half-right.** The ask was to audit Luminance
Response's position and move it to where the pipeline actually runs it. I read
that as a `GROUPS` reorder and lifted the section *above* Grain Structure, on
the reasoning that the engine measures the mask before it builds the field. The
user pushed back:

> I don't see why you move it here, it is at least after grain structure
> section, in the sense it main purpose is supressing grain from grain
> structure, am I wrong?

They were not wrong, and the objection exposed a genuine ambiguity in "where
does this run". **Where the mask is *applied* never moved** — it has always
multiplied the grain field at step 10, after `_grain_field` builds it. What
moved in the engine is only which luma the mask *reads*. Ordering the panel by
the read rather than by the application put the suppression before the thing it
suppresses, which is backwards from every user's point of view: you set a grain
amount, then you say where it lands.

**What it is now.** The section does not exist. Its six parameters —
Shadow/Highlight Knee, the two Falloffs, Highlight and Black Suppression — took
`group = "Grain Structure"` and sit under the controls that build the field they
mask:

```
Intensity            Octaves               Shadow Knee
Clump Size           Roughness             Highlight Knee
Shadow Clumping      Chroma Grain          Shadow Falloff
Clump Hardness                             Highlight Falloff
                                           Highlight Suppression
                                           Black Suppression
                                           Seed
```

`GROUPS` loses an entry rather than reordering one:

```
before                          after
------                          -----
Colour Grading                  Colour Grading
Pre Blur                        Pre Blur
Pre Sharpen                     Pre Sharpen
Grain Structure                 Grain Structure   (+ the six, at the end)
Edge Destruction                Edge Destruction
Anti Aliasing                   Anti Aliasing
Global Grain                    Global Grain
Sharpening                      Sharpening
Luminance Response              Halation
Halation                        Tone Response
Tone Response                   Film Texture
Film Texture                    Output
Output
```

Merging beats reordering here for a reason that only shows up once you try the
reorder: **Luminance Response was never a stage, so a heading of its own was
always claiming too much.** It says which densities carry the grain the controls
above it make. As a section it read as a second thing to set up; as the tail of
Grain Structure it reads as what it is.

Two things fall out of the merge, both improvements:

* **The section's mute button stops being a lie.** `toggleGroup` sets a group's
  parameters to their neutral values, and none of these six are in
  `NEUTRAL_ZERO` — they are knees and widths, not amounts — so muting
  Luminance Response set them to their *defaults* and switched nothing off.
  Muting Grain Structure takes `intensity` to 0, which really does switch the
  whole thing off, these six included.
* **Per-section Reset now covers the pair together**, which is what you want:
  the band is meaningless apart from the field it applies to.

**`seed` stays last in the group**, under everything it rerolls, matching where
`Texture Seed` and `Global Seed` sit in theirs — so the six went in front of it
rather than literally at the bottom of the list. It is the control reached for
least often and burying it under six knees would cost more than the literal
reading is worth. One line to flip if that reads wrong.

## Three more moved into Grain Structure (2026-08-06)

Same day, same direction, asked for outright: **Edge Bias**, **Smooth-Area
Guard** and **High-Pass Radius** took `group = "Grain Structure"` and sit
directly under `seed`. UI only — three `group` strings, no `GROUPS` change, no
engine change, and `verify.py` keys on parameter values rather than on group
names so nothing there moved either.

It is the same argument as the Luminance Response merge above and lands the same
way. Edge Bias and Smooth-Area Guard are not edge destruction: neither of them
touches an edge. Both build a *weight* that multiplies the grain field at step
10 — `weight = m * ((1 - eb) + eb * edge)`, then the guard's textured mask —
sitting in the same expression as the luminance band that moved here for
precisely this reason. Under Edge Destruction they were filed by the mask they
read rather than by what they do with it.

`highpass_radius` is the genuinely arguable one and is worth naming as such,
because it has a foot in both camps: the high-pass mask it sizes also feeds Edge
Erosion and Acutance, which really are edge destruction. It follows its two main
consumers here — the ask named it, and a radius belongs beside the sliders whose
effect you are watching while you drag it — but the help text now says out loud
that widening it coarsens the other two as well, since the section heading no
longer hints at that. The alternative reading, leaving it behind under Edge
Destruction, splits Edge Bias from the radius that determines what "edge" means
to it, which is worse. One `group` string to flip if that trade reads wrong.

`seed` therefore stops being last in its group, which is a deliberate departure
from the rule stated just above and from where `Texture Seed` and `Global Seed`
sit. The ask was specific about the position ("after the Seed slider"), and the
three are a coherent block of *where the grain goes* that reads worse split
around a seed.

**The list as a whole is not pipeline-ordered and cannot be**, which is worth
stating plainly since this is now the third reorg asked for in those terms.
Edge Destruction alone spans steps 1, 1b, 7b, 8b and 11; Tone Response spans 3
to 6. A section is a group of *controls that share a mechanism*, and several
mechanisms are threaded through the pipeline at more than one point. What the
order can do — and what all three reorgs have done — is put each section next to
the one it actually interacts with, and this time the answer was that two of
them were one section.
