# Using the controls

What each control does and how to reach for it, moved out of `README.md` on
2026-08-08 to keep that file to setup and running. This is the *user-facing*
half; the engine's reasoning — why a stage sits where it does, what was measured
to place it — lives in the other files indexed from
[CLAUDE.md](../CLAUDE.md).

Every slider is generated from the server's schema, so the panel always matches
what the renderer accepts.

## Colour Grading, and 3D LUTs

The first section in the panel, and the first thing in the pipeline — above
Pre Blur and above every film stage. Everything below it models an emulsion;
this is the decision about what the photograph *is* before any of that runs.
It all ships at 0, so nothing here changes an existing look until you ask.

The section reads in the order it runs:

* **Temperature** — a warm/cool shift, done as channel gains in *linear* light,
  which is where a white balance physically happens. The gains are normalised
  against the luma weights, so warming a frame does not also expose it (measured:
  overall luminance holds to within 1% across the whole slider).
* **Shadows** / **Highlights** — a lift or a crush on each half of the range,
  over a broad quintic ramp so there is no line across a gradient. Both are a
  share of the headroom that is actually there, so **neither can clip and neither
  can break a hue** at any setting — that is by construction, not by a clamp
  afterwards. Their masks are read from the frame as it arrived, so the two do
  not pull on each other. Negative Highlights is the useful direction most of the
  time: it pulls blown highlights back *before* the LUT and the film pipeline get
  hold of them, which is the one thing no amount of Shoulder can recover later.
* **Clarity**, two-way — positive adds local contrast, negative takes it away.
  Above 0 it is mid-frequency punch without the halos of a small-radius sharpen;
  below 0 it flattens the same band into a soft, hazy, lifted look. **The two
  directions are deliberately not the same strength**: negative stops exactly
  where the band is *gone* (−1 removes 100% of it) and cannot go further into
  inverted contrast, which would put dark halos on the light side of every edge.
  Positive has no such ceiling and goes to 255% of the band. It runs on luminance
  only, so it holds hue **exactly** and costs one single-channel blur rather than
  three. **Clarity Radius** picks which band, as a length at full resolution.
* **LUT** — pick a `.cube` from the `luts/` folder, or **Load .cube…** to use one
  from disk for this session. **LUT Mix** cross-fades it in; part-way is the
  normal way to use a film LUT that is stronger than the photograph wants.
  Picking a LUT raises Mix to 1 if it was at 0, because a picker that appears to
  do nothing is worse than one that commits.

The LUT is applied display-referred, on the source, before every film stage —
which is what a `.cube` expects, and it means the grain, halation and texture
below all land on the graded picture instead of being graded themselves.

**Adding a LUT is dropping a `.cube` in `luts/`**, exactly like `presets/`. They
are referenced **by filename**, not by position in the folder: an index would be
silently renumbered by the next file you added, changing the look of every preset
that named one. A preset file carries its LUT as a `lut` key beside
`reference_mp`, and a name that no longer resolves renders with no LUT and says
so in the picker rather than pretending the grade is still there.

Both 3D LUT sizes in the wild are handled (35- and 64-cube here); `DOMAIN_MIN` /
`DOMAIN_MAX`, comments and vendor keywords are all read, 1D LUTs are refused with
a reason. Uploads are capped at 24MB and live in process memory, so they do not
survive a restart — a folder LUT does.

**It is cheap, which was the point.** Four of the five stages are pure per-pixel
arithmetic: no kernel, no neighbourhood, nothing added to the tile overlap.
Clarity is the only one that costs anything, and only when it is on. Measured on
a 6MP render at 2×, best of 3 in fresh processes:

| | time | tile overlap |
|---|---|---|
| section off | 0.67s | 108px |
| Temperature, Shadows, Highlights, or a LUT at Mix 1 | 0.67–0.73s (inside MPS variance) | 108px |
| Clarity at the default 14px radius | 0.75s | 150px |
| Clarity at 40px | 0.88s | 228px |
| everything at once | 0.82s | 150px |

## Light leaks

A leak is a *beam*, not a glow around the border. Each one has a source
somewhere on the frame's perimeter, a depth it pushes in, a length it runs
along that border, a lean across the picture, and one hard edge — the shadow of
whatever the light got past. That hard edge is most of what separates a leak
from haze, and a real one almost always has it.

**Leak Size Min** and **Leak Size Max** are how far the shallowest and deepest
leaks reach in from the frame edge, in full-resolution pixels. Every leak draws
its own reach from between them, so the spread is stated outright rather than
being a percentage of a hidden maximum — set them equal and every leak comes in
exactly as far as the next. Given the wrong way round they simply swap. How far
a leak runs *along* its border is separate and much larger, because a seal
fails along a seam: it is drawn against the border's own length, so leaks stay
proportioned the same way on any frame.

**Leak Feather** is a pixel distance too: *how far in the leak has faded to half
strength*. Small against the size is a tight bright rim hugging the edge; about
half the size is a straight ramp; most of the way is a broad wash. Measured on a
300px leak, asking for 20/80/150/285px delivers 20/78/149/270px. Because it is
absolute, the same feather is a wash on a small leak and a rim on a large one,
which is what stops a frame of differently-sized leaks reading as one shape at
several scales. It also softens the leak's long edges.

**Leak Strength** is not just an opacity. The light saturates one dye layer at
a time, so a faint leak is deep red — only the red-sensitive layer caught
enough light — and turning it up takes the core through orange and yellow to
white while the colour stays in the falloff. That is why the feather measures
slightly deeper on a blown leak than the number says: the core has clipped, so
the visible half-way point sits further in than the exposure's does. Past about
1.5 most leaks have a white core, which is the "sun got in the back" look.

Sizes and the feather are lengths, so a preset rescales them onto a
different-sized photo like every other length — and the defaults (250/850px,
180px feather) are tuned for a full-resolution photo. On a much smaller image
they will be proportionally bigger, which is what asking for pixels means.
**Leak Variation** covers everything *except* size: length, lean, fan, edge
hardness, halo, strength and hue.

**Leak Count is a count, and anything below 1 renders nothing** — you cannot
have a third of a leak. It is now an exact count: one beam is placed per leak,
so asking for two gives you two rather than washing most of the border. Raising
it adds leaks without moving the ones already there. Three shipped presets used
to carry `0.05` here (and
similar for dust, scratches and hair), left over from when these were 0–1
amounts, so their whole Film Texture section was silently doing nothing. They
now read 0; set a real count to turn them on.

A leak still cannot fog the middle of the frame — reach is capped at half the
short side, which is exactly the distance at which a leak dies in the centre.
Below that the pixel numbers are honoured exactly; the centre measures a flat
0.00e+00 at every size from 60 to 3000px and every feather from 2 to 1500px.

## When halation greys the sky

Halation adds warm light, and **adding light desaturates whatever it lands on**
— that is what addition does, not a flaw to tune out. A red bloom lifts a blue
sky's red channel by the full glow and its blue channel by a tenth of it, so
the sky drains toward grey and drifts toward purple. Measured on an ordinary
sky: **16% of the saturation gone and a +5.8° hue swing**.

**First, check `Halation Threshold` against the sky itself.** An ordinary blue
sky has a luma around 0.37. If the threshold is under that — `Organic` ships at
0.30 — the sky is over the threshold and **blooms onto itself**, so the wash is
global instead of a rim around bright things. Raising it above the sky's luma
fixes the symptom outright: 0.30 → 0.45 measures sat 0.660 → 0.778 and hue
225.3 → 219.7, against an untouched 0.769 / 220.0.

**Then, if you want the low threshold for the look, compensate for it.** Four
controls in **Halation**, applied to the image *before* the bloom lands:

* **Blue Compensation** strengthens blue so it survives the wash rather than
  being greyed by it — putting the colour into the exposure, which is what a
  punchier blue-sensitive stock or a polariser does, instead of repainting the
  result. It is **self-limiting**: the wash eats the same share of whatever you
  add, so 0.5 lands within 1% of the untouched sky and everything from 1.0 to
  3.0 sits at 3% past it.
* **Blue Level** — how light a blue has to be before it is worth saving, on the
  picture's own brightness scale. Halation only reaches what is near the light:
  measured up a sky gradient, the loss is **23% at the bright end and flat 0%
  below about half brightness**. Compensating a deep blue anyway is all
  overshoot, and it is what makes one go lurid — ungated at amount 2.0 an
  untouched deep sky went from 0.872 saturation to 1.000, a channel clamped to
  black. Raise this until only the blues that actually got washed are being
  saved.
* **Blue Level Falloff** — the width of the fade below it, so the boundary is a
  ramp rather than a line across the sky. Separate from the level on purpose:
  deriving the width from the knee would mean moving one changed the other.
* **Blue Hue Shift** — because saturation alone *cannot* fix hue: scaling
  chroma about the luma axis rotates nothing. Measured, the amount slider
  leaves +7.1° of error and −8° of shift takes it to +0.1°.

The mask is also weighted by existing saturation, so it strengthens blue that
is there and never invents it in grey — grey and red are left **bit-exact** at
maximum settings.

Doing this before the wash rather than after is the whole design. The same
correction applied afterwards has no brake — it is 9% past target by 0.5, and
by 1.0 it has driven a channel to black and pinned the sky at fully saturated.
It also cannot tell blue that was unfairly greyed from blue the bloom is
*supposed* to sit on, so it fights the glow you paid for. Dialling blue never
moves the bloom (0.00e+00) and costs no tile overlap.

## Softening without blurring

Two controls in **Edge Destruction** do the same physical job from opposite
ends, and which one you want depends entirely on whether you mind losing
texture.

> Both run *after* the grain since the 2026-08-08 pipeline reorder, so both now
> soften the grain they pass over as well as the picture. Micro-Blur leaves
> **29%** of it, Scatter **78%** — the gap below is the same gap, measured on
> the picture instead.

* **Micro-Blur** averages each pixel with its neighbours. That is diffusion as
  an expectation, and it is smooth — which is the problem. Measured on a fine
  texture plate, a 3px micro-blur leaves **9%** of the texture's sigma and
  **2%** of its local contrast. The picture goes soft because everything went
  soft, and it reads as out of focus.
* **Scatter** displaces a share of the pixels onto their neighbours and
  averages nothing at all. Same reach, same physics resolved photon by photon
  instead of in bulk: **100%** of the texture sigma and **96%** of the local
  contrast survive. The frame loses its digital exactness and keeps its bite.

Every displaced pixel is a bit-exact copy of a real pixel nearby — verified to
1.2e-07, where a blur of the same reach deviates by 6.3e-02 — so no in-between
values are invented and the grit, noise and contrast come through whole.

It masks itself, with no mask in the code: shuffling pixels that already match
their neighbours cannot change them, so a smooth ramp comes through at its own
slope times the travel (0.003 at a 3px reach) while detail is the only thing
that comes apart.

* **Scatter** is *coverage*, not opacity — the fraction of the frame that
  travels. Cross-fading a moved pixel with the one it left would be an average
  by another name, and at 0.5 it would be exactly the blur this replaces.
* **Scatter Reach** is how far, and so also *what* moves: displacing a pixel
  only changes anything where the picture varies over the distance travelled.
* **Scatter Pattern** is the stencil — the set of places a pixel may land.
  Restricting it is what makes the result read as structure rather than as
  noise. `Any` is isotropic; `Cross`, `Diagonal` and `Box` are the 4-, 45- and
  8-neighbour stencils; `Horizontal` and `Vertical` are a one-axis slip that
  leaves an edge running along that axis untouched to the float floor. Three
  are shapes rather than direction sets:
  * `Diamond` keeps every angle but holds `|dx| + |dy|` constant, so it reaches
    the full 12.0px on the axes and 8.5px on the diagonals where a disc reaches
    12 both ways — detail spreads as a rhombus.
  * `Donut` holds a hole open in the middle: nothing lands near where it
    started, so detail is thrown outward and hollowed out. Measured, the
    nearest landing is 7.2px of a 12px reach even at Reach Spread 1, where
    every other stencil fills solid to 0.
  * `Star` runs alternate spokes short — a 0.35 diagonal/axis reach ratio
    against `Box`'s 0.94 on the same eight directions, which is the shape a
    cross filter flares into.
* **Reach Spread** — 0 is a shell (everything lands on the shape's edge,
  measured 6.0 ± 0.00px), 1 fills it inward (3.3 ± 1.60px). `Donut` ignores it
  to the extent of keeping its hole.
* **Scatter Clump** is how big a piece moves as one, from per-pixel dissolve to
  whole tiles travelling intact — lag-1 correlation of the displacement field
  runs 0.00 at 1px to 0.87 at 8px. Past ~4px the tiles start reading as tiles.

Costs about 16% on a render (0.69s → 0.80s on 6MP at 2×) and widens the tile
overlap by its reach.

## Overall Opacity

The last slider in the panel, in its own **Output** section, and the only one
that ships at 1 rather than 0. It cross-fades the finished frame back over the
untouched photo, so it dials back *everything at once* — grain, halation,
softening, marks, the lot. Reach for it when a preset is right in character but
too strong, instead of walking a dozen sliders down together.

0 returns the original bit for bit at any quality setting, 1 is the full
pipeline unchanged, and the middle is a straight cross-fade — all three
verified exactly. Dragging toward 0 also gets *faster*, since there is no point
rendering a frame that will be thrown away.

Not the same control as **Global Opacity** under Global Grain, which only mixes
that one noise layer.

## Anti Aliasing

A third way to treat an edge, and the only one that does not cost sharpness.
Runs at section 5, below the grain — the masks that decide where grain lands
take their own anti-aliased copy of the frame so they still cannot key on
jaggies this is about to remove.
Both controls above work *across* an edge; these work *along* it.

* **AA Strength** takes stair-stepping off hard edges in the source — the
  ragged diagonal from an upscaled JPEG, a screenshot or a CG render. It
  averages each pixel with its neighbours **along** the contour, never across
  it, so the jaggies cancel while the edge stays where it was. Measured on a
  deliberately aliased diagonal: the contour's residual falls from 0.289px to
  0.189px while 86% of the across-edge slope survives. (For scale, Edge Sanding
  under Edge Destruction keeps 73% for a comparable 32% — it is aimed at much
  longer-wavelength roughness, so it is the wrong tool for a one-pixel step.)
* **AA Radius** is how far along the edge, in full-resolution pixels. A
  stair-step is one pixel by definition, so ~1 is the honest setting and the
  default. Go larger only if the source was upscaled and its steps are several
  pixels wide; it starts rounding off genuine corners.
* **Edge Only** is the texture guard. At 1 it touches only borders that step a
  long way in brightness — fabric-scale texture measures **100%** intact — and
  at 0 it runs everywhere, which suits a CG render that aliases on gentle steps
  and will visibly soften a photograph (88%).

Effectively free, and it widens the tile overlap by a few pixels.

## Global Grain is drawn as scattered grains (rewritten)

The layer used to be built on an axis-aligned lattice, and it showed. Past
about 8px of **Global Size Min** it stopped looking like grain and started
looking pixelated — a visible quilt of rectangles — and at any size it read as
an evenly spaced mesh once you stepped back from it. Neither was a bad setting;
both were the noise itself.

It is now drawn as discrete grains scattered over a lattice **tilted off the
pixel grid**, with several per cell and a fraction of them missing, and with
every grain's strength modulated by a multi-scale clumping field. In practice:

* **No grid, at any size.** Measured on the metric that diagnosed the old
  quilt, the layer scores 0.03–0.05 where the old field scored 1.4–1.7.
* **No repeating mesh when you zoom out.** The clumping field gives the layer
  real variation at scales far above one grain — some regions grainier than
  others, with no characteristic patch size.
* **Consistent.** The old field lost up to 35% of its strength whenever the
  clump size landed near a whole number of pixels — which is where a slider
  lands — so the same settings could look good or flat for no visible reason.
  The tilt removes that outright; strength now varies under 4% across sizes.

**Global Size Min** and **Global Size Max** are the two ends of one grain-size
distribution, and nothing else. Min is the smallest a grain can be, Max the
largest; leave them equal for one uniform size, or open them up so each grain
picks its own diameter between the two. It is a range, not a switch — widening
it changes how much the sizes vary and nothing else. A very wide gap leaves
visible clear patches between grains; real film has them too, but narrow the
gap if it reads as sparse. Both run to 20px.

**Global Smoothness** was the cure for the blockiness and no longer has
anything to repair. It is now a shape control: it blurs the layer by up to half
a clump, rounding grains off and softening where they meet. It still holds the
strength constant as you raise it, so it changes the *shape* of the grain and
not how much there is; Global Intensity remains the only amplitude control.

**One thing to expect from the rewrite:** `global_intensity` used to mean two
different loudnesses depending on whether Max exceeded Min — a 43% gap. It now
means one. Presets that left Max at its default (Dramatic, Dreamy, Subtle,
ExtraGrain) get a quieter global layer than before and can be brought back by
raising their Global Intensity; the rest are within about 10% of where they
were.

Sliders only render on release, not during the drag — a fit preview is seconds
of work, so rendering every intermediate position just queues frames that are
stale on arrival. The number beside each slider still tracks the thumb live.

## Why Edge Destruction skips edges you can see

**Start with the radius, not the thresholds.** Every measurement here is a
high-pass, which only responds to structure *finer* than its radius — so a soft
edge is invisible however hard it steps. A portrait's skin-against-background
boundary is a 0.26 luma step, far above every gate, and ramps over 30–100px at
24MP. There are two radii and they serve different stages: **High-Pass Radius**
(0.5–24) drives Jitter, Sanding, Erosion and the grain's Edge Bias; **Softening
Radius** (0.3–64) is Edge Softening's own, and softening reads nothing else.
Raise the relevant one until the edge appears.

Two internal thresholds also decided what counted as an edge, and both are
controls now (2026-08-09).

**Edge Colour Sensitivity.** Every mask in the section was built from luminance
alone, so a boundary between two colours of *equal brightness* was flat to it —
foliage against sky, skin against fabric. Measured on a red-to-green edge at
matched luma, softening moved 0.29 8-bit levels against 5.20 for the same-size
luminance edge; jitter and sanding were 17–21× blind the same way. At 1 (the
default) an edge is as strong as its strongest single channel. At 0 you get the
old luminance-only mask, bit for bit. **Greyscale content is identical at every
setting** — there is no colour difference to find — so this only ever adds
edges.

**Edge Sensitivity.** How hard a transition has to step before it counts as a
full-strength edge. This was a fixed number, and everything gentler than it only
ever reached a fraction of the mask however the sliders were set. 1 is that
number; 4 takes the share of the frame reading as a strong edge from 5.9% to
10.4%. Reach for it on soft-lit or low-contrast frames, which can sit almost
entirely under the default.

Both live at the top of **Edge Destruction**, beside High-Pass Radius: the three
of them define the mask and everything below is weighted by it.

They still reach one control outside the section — **Edge Bias**, under Grain
Structure, which is how much the grain follows the same mask. At Edge Bias 0
they have no effect on grain at all. **Softening has its own gate** (Softening
Selectivity) because it is asking a different question: "border or texture", not
"how much of an edge".


## Edge Erosion, Colour Fringing and Acutance

The last three controls in **Edge Destruction**, and they are last in the panel
because they are last in the pipeline. All three add fine, high-frequency
structure, and every other control in the section removes it — run first, their
entire contribution is averaged straight back out by Micro-Blur, softening and
sanding. Measured when they were: `edge_erosion` moved **0.01%** of a frame's
pixels by more than one 8-bit level. Moved to the end: **2.63%**.

* **Edge Erosion** modulates the image's own micro-detail by the grain field —
  zero in flat areas, strongest on edges.
* **Edge Colour Fringing** blends that erosion from neutral to per-dye-layer, so
  each channel erodes independently and edges pick up coloured speckle.
* **Acutance** is the adjacency effect: developer exhausts faster on the dense
  side of an edge and diffuses across it, leaving a local contrast boost. Taken
  from the pre-grain frame, so it sharpens the picture without amplifying grain
  — which is what separates it from Sharpening.

## Edge Sanding

**Sanding Grit is the strength control, not Edge Sanding.** The amount is a
cross-fade toward the sanded frame, so 1 is the whole of it; the grit decides
how far along the contour the polish reaches, and that is what changes how much
comes off. At full strength: 5px grit removes ~46% of a jittered border's
roughness, 10px ~67%, and 20px *less* again (53%) because the filter starts
reaching across the contour rather than along it, costing the edge its
sharpness for nothing.

The amount used to run to 5, and above about 3 it inverted — it extrapolated
past the filtered result and put roughness back on, 235% of it at the top. Fixed
2026-08-08; the range is 0–1 now, which is what it always was.

## Export

**Every export is full size.** The menu picks the supersample —
`Full size W×H / SS 0.5× … 3×`, default **2×** — and that is a quality choice,
not a size one. Below 1 the frame renders smaller than its output and is scaled
back up; above 1 it renders finer than the output grid and is integrated down,
which is what gives each grain clump genuine partial-pixel coverage instead of
a hard, aliased footprint. Cost is roughly the square of the factor.

Files carry the factor in the name (`photo_grain_ss1_5.jpg`) unless it is the
default, because now that every export is the same dimensions a folder listing
can no longer tell two apart.

The menu used to offer three *scales*, and it moved resolution and look together
— "As previewed" wrote a smaller file *and* a coarser grain. See
[preview-and-export.md](preview-and-export.md), including what that change gave
up.


## Presets

The **Preset…** dropdown is the contents of the `presets/` folder — one `.json`
per preset, read fresh on every page load. Drop a file in and it appears; no
restart, nothing to edit in the code. A preset is named by its **filename**, so
renaming the file renames the entry. `build.sh` copies the folder into the
distribution.

**`Stock.json` is the starting point.** The app opens on it and **Reset**
returns to it, so "reset" always means "how it opened". Delete the file and both
fall back to the raw parameter defaults — that is a supported way to start
neutral, not a broken install. `FILM_GRAIN_DEFAULT_PRESET=Dreamy` picks a
different one.

**Save to file…** writes the current settings to a `.json` you name; **Load
file…** reads one back without installing it. Move a saved file into `presets/`
to make it a permanent entry.

Set `FILM_GRAIN_PRESETS=/some/dir` to read them from somewhere else.

## Preset size scaling

Every spatial setting — clump size, radii, jitter, speck and scratch size — is
a length in full-resolution pixels, so a preset dialled in on a 24MP photo
gives proportionally finer grain on a 45MP one. Presets can record the size
they were authored at, and the server then rescales those lengths by the
**linear** ratio `sqrt(current / reference)` — a 4x-the-pixels photo is only
2x as wide. Amounts and mark counts are not scaled; they already mean the same
thing at any size.

The panel shows `preset 24.0MP → photo 45.0MP = 1.369×` and lets you switch it
off. A preset with no recorded size scales by 1.0 — nothing is guessed. To
populate it, either press **Set from photo** with the right photo open and
re-save, or start the server with `FILM_GRAIN_DEFAULT_REFERENCE_MP=24` to
treat every unstamped preset as 24MP.

```json
{
  "format": "film-grain-preset",
  "version": 1,
  "name": "my-look",
  "values": { "intensity": 41, "grain_size": 2.4, ... }
}
```

The file is meant to be hand-editable, and the same leniency applies whether it
is loaded through the button or read from `presets/`: unknown keys are dropped,
values are clamped into range, and anything absent falls back to its default. A
file saved before a slider's range changed still loads, and a bare
`{"intensity": 40}` typed by hand works too. A file that will not parse is
skipped with a line on stderr rather than taking the whole list down.

Every slider is generated from the server's schema, so the panel always matches
what the renderer accepts.
