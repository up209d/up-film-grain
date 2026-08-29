# Preview and export

**"The source" means the working frame, not the file (2026-08-29).** Prescaling
Source resamples the photograph to a fixed megapixel count before either tier
is derived, so every "the source's own dimensions" below is the *prescaled
frame's* dimensions whenever it is on -- both tiers come from the same frame, so
they still agree with each other exactly as described here. Its own
`prescale_output` control is the one opt-out, resampling the finished render
back to the file's dimensions. The `_MIN_CELL` divergence documented under "The
preview is client-scaled" becomes constant across photographs with it on,
because `proxy_scale` stops being a function of the file's size. See
`docs/prescale.md`.

## Every export is the preview's look, at full size (2026-08-09, on request)

The menu below is unchanged -- same five entries, same labels, same output
dimensions. What each one *renders* changed: instead of the source at scale 1.0
with the factor as its supersample, every entry renders **the preview tier** at
that supersample and then enlarges the result to the source's own pixel
dimensions.

Concretely, `run_export` is now `render_tier(up, p, ss, full=False)` followed by
`imageio.upscale(out, up.h, up.w)`. That is the same call `/api/preview` makes,
which is the point rather than a convenience: the export is byte-for-byte the
live preview before the enlargement, so "export what I am looking at" is
literally true at every setting. Verified across all five factors on a source
with a 0.5 proxy -- every one writes the source's dimensions and matches
`upscale(render_tier(...))` at max abs diff **0**.

Why this and not the 1:1 render: a full-resolution render of the same numbers is
not a sharper version of the preview, it is a *different picture*. Every spatial
length scales with the frame, so it resolves finer, denser grain -- the two
reasons are spelled out under "The export scale now defaults to..." below, and
they have not changed. Making the file disagree with the screen it was judged on
is the failure this removes.

What it costs, and it is the same trade the old `preview_full` made: **the
enlargement adds no detail.** Zoomed to 100% the file carries the proxy's
texture, just bigger. The supersample still buys real quality *within* that
render -- partial-pixel coverage on every clump -- it just no longer decides
which tier is rendered. On a source no bigger than `PROXY_LONG_EDGE` the
question is moot: `up.proxy is up.arr`, `upscale` is a pass-through, and this is
exactly what the old behaviour did anyway.

### A sixth entry: the real 1:1 render (2026-08-09, same day, on request)

The paragraph above used to end "there is no longer a way to get the frame's own
finest grain; if that is wanted back it needs its own control". It got one, the
same day: `Full size W×H / 1:1 SS 1×`, which renders `up.arr` at scale 1.0 with
`supersample: 1` and has nothing to enlarge. On the wire it is `full: true` on
`/api/export`; in the UI it is the sixth entry in the same menu.

**It is not the default and that was explicit.** `ss2` -- the previewed frame at
2× -- still is. This entry is the one file the preview cannot show you, so
making it the default would reintroduce exactly the disagreement between screen
and file that the section above removes. `Render 1:1` is how you look at it
first, and the two are the same pixels because the export goes through
`render_tier(up, p, ss, True)`, the identical call the 1:1 preview makes.

Two consequences worth knowing:

* **The menu's value is a key, not a factor.** `ss1` and `full` are both 1× and
  differ only in tier, so `EXPORT_OPTIONS` carries `{key, ss, full}` and
  `useExport` holds `exportKey`. A number cannot identify an entry any more.
* **The filename needs its own word.** `_grain_ss1` and the 1:1 render at 1× are
  the same dimensions and the same factor and differ in the one thing you would
  keep both files for, so the full tier tags `_grain_full`
  (`_grain_full_ss2` if the API is asked for a factor the menu does not offer --
  a full-tier render at any supersample is a coherent request, it is just not
  one the UI makes).

Verified alongside the five: it writes the source's dimensions, matches
`render_tier(..., full=True)` at max abs diff **0**, and differs from the
previewed 1× entry by **153/65535** at peak -- which is the point of having it.

## Every export is full size (2026-08-08, on request)

*Superseded in part by the section above -- the menu and the output size are as
described here, the tier it renders is not.*

The export menu used to offer three *scales* -- `full`, `preview` and
`preview_full` -- and it was asking the wrong question. Its entries differed in
two things at once, resolution **and** look, and only one of those was ever the
choice being made: "As previewed" wrote a smaller file *and* a coarser grain,
because every length scales with the frame, so the proxy resolves fewer, larger
clumps per frame than a 1:1 render of the same numbers.

The menu now picks the **supersample** -- 0.5, 1, 1.5, 2 (default) or 3 -- and
the output is always the source's own dimensions. That separates the two:

* below 1 the frame is rendered smaller than its output and resampled back up,
  so the file is full size and genuinely soft;
* above 1 it is rendered finer than the output grid and integrated down, which
  is what gives each grain clump partial-pixel coverage instead of a hard,
  aliased footprint.

Cost is roughly the square of the factor, so 3x is 2.25x the work of 2x and 1x
is a quarter of it.

**What this gives up, and it is worth naming.** There is no longer a way to
export the proxy's own look. `preview_full` existed for "the file I am looking
at, enlarged", and 0.5x is *not* the same thing: it renders the full frame at
half resolution, where the proxy renders a smaller frame with every length
scaled to it. If that look is wanted again it needs its own control, not a
supersample setting. *(This is what 2026-08-09 reversed: the proxy's look came
back as the behaviour of all five entries, and it is the 1:1 render that has no
control now.)*

Filenames carry the factor (`photo_grain_ss1_5.jpg`) unless it is the default,
for the reason the old `_grain_2400px` tag existed: two files from one photo
that a folder listing cannot tell apart are worth naming apart, and now that
every export is the same dimensions, size cannot do it.

## The preview is client-scaled, two-tier (changed 2026-07-31)

`/api/preview` renders **the whole frame**, never a crop. The request carries
`id`, `params`, `supersample`, `proxy_edge` and `full` — no mode, no zoom, no
viewport. The browser does all the scaling.

`full` picks the fidelity, and it is the only difference between the two:

* `false` (default) — the working proxy, whose long edge is `proxy_edge`,
  defaulting to `PROXY_LONG_EDGE = 2400`. This is what every slider change
  triggers. At *parameter defaults* it is a few hundred ms; what it actually
  costs depends almost entirely on the preset — see the performance section,
  where `Stock` measures 8.8× defaults.
* `true` — the whole source at scale 1.0. The preview *is* the export at this
  point. Only ever fired by the explicit "Render 1:1" button; any parameter
  change drops back to the proxy. `proxy_edge` is ignored here — there is no
  proxy.

### The proxy edge is adjustable (2026-08-29, on request)

`proxy_edge` moves that long edge over **100–4800 in steps of 100**, defaulting
to 2400 so a client that never sends it behaves exactly as every client did
before. `_clamp_edge` in `models/upload.py` clamps and snaps it, the same
bargain `_clamp_ss` makes with the supersample: a value outside the range is a
client bug, and answering at the nearest legitimate one beats refusing.

Cost is roughly the **square** of the edge, which makes it the largest single
lever over render time — larger than the supersample, because it moves the pixel
count of the frame rather than of one tile's working grid. Measured on a
synthetic 24MP source, `Stock` at 1× on the M4 Max, after a warm-up pass:

| edge | tier | cold | warm checkpoint |
|---|---|---|---|
| 400 | 400×267 | 0.19s | 0.05s |
| 800 | 800×533 | 0.23s | 0.08s |
| 1200 | 1200×800 | 0.46s | 0.10s |
| 2400 | 2400×1600 | **1.08s** | 0.38s |
| 4800 | 4800×3200 | 3.96s | 1.63s |

It runs in **both** directions and that is deliberate: below the default for a
slider that keeps up on a large photograph, above it for a proxy that resolves
more than the default does. A 4800px tier on a 24MP source is 15MP of working
frame, which is most of the way to the 1:1 render at a third of its cost.

**It is shared with the export, not preview-only**, and that is the whole
design rather than an oversight. Five of the six export entries render the proxy
tier and enlarge it, so this number already set export texture before it was
adjustable; pinning the export at 2400 while the preview moved would break the
one property this file exists to defend — that the file is the picture you
judged. The costs of sharing it are paid where they are visible:

* the export menu label carries the edge whenever it is not the default
  (`Full size 5657×4243 / SS 2× / 800px proxy`), and every preview-tier entry's
  help says what that means for the file;
* the written filename carries `_px<edge>`, for the reason every other tag in
  `controllers/export.py` exists — it changes the texture of the file without
  changing a single one of its dimensions, so nothing in a folder listing could
  otherwise tell two exports apart.

**Why the edge is not in the checkpoint id.** It does not need to be. The
engine's own key is `(id, boundary, scale, y0, x0, h, w, device, signature)` and
a different edge moves `scale`, `h` and `w` together, so two edges cannot
collide; entries for an edge the user has moved off become unreachable and age
out under the byte cap — a miss, never a wrong picture. The one case where two
edges key identically is when both sit at or past the frame's long side, and
there they are genuinely the same render. `tests/checks/prescale.py` pins this by
rendering four alternating edges against cold engines and requiring 0.00e+00.

**Where it is cached.** `proxy_at(edge)` on both `Upload` and `Frame` holds a
*single slot*, rebuilt when the edge changes and the outgoing file unlinked —
the same argument `Upload.at()` makes about the prescale target, and for the same
reason: a photograph is edited by dragging sliders, and neither of these moves
while that is happening. A dict would grow one array per edge the user tried. An
edge at or past the long side is not a resample at all and hands back `arr`
itself, which is what removed the old aliased `_proxy is _arr` file and with it
the hazard of a proxy slot that must not be released.

### It travels with the preset, and the render waits for the release (2026-08-29, later the same day)

Two changes on request, and they are the same realisation from two ends: the
edge is *part of the look*, and it was being treated as neither a look nor a
gesture.

**It is a preset key now, and it is still not a `Param`.** `proxy_edge` sits
beside `reference_mp` and `lut` at the top level of a preset file — a sibling of
the values, never one of them, because the engine never reads it. It decides
which *frame* the engine is handed, exactly as `reference_mp` decides which
lengths it is handed. Every shipped preset is stamped `"proxy_edge": 2400`, and
`usePresetFile` writes it into anything saved from the app, so a look now records
the tier it was judged on and an export reproduces that tier without being told.
A file written before this existed carries no key and renders at 2400, which is
what it always did.

The previous paragraph here said the opposite — session state, not part of the
look — and it was wrong for one specific reason worth keeping: five of the six
export entries render this tier and enlarge it, so the edge is in the *file*.
Anything that changes what comes out of the export cannot be a preference about
this session.

**Three sources, in strict order, and `controllers/export.py` is the one place
that resolves them:** the request body if it names an edge, else the named
`preset`'s own, else `PROXY_LONG_EDGE`. So `./export.sh photo.jpg -p Stock`
renders Stock's edge and `-e 800` overrides it. The CLI's `-e` has **no
argparse default** for that to work — `None` is what makes "not declared"
distinguishable from "declared as 2400", and a default there would have made
every CLI render silently override the preset. The web client always sends the
edge it is showing, so its precedence is the first rule and the preset reaches it
by having seeded the slider.

`load_presets` deliberately does **not** clamp the value: `_clamp_edge` lives in
`models/upload.py`, which imports `params`, so reaching for it there would close
the import graph on itself. Every consumer clamps instead — the export
controller server-side, the slider's own bounds client-side — which is the same
bargain `reference_mp` already makes with `sanitize`.

**The slider commits on release, like every other slider.** It was a plain
`useState` in `App.tsx` feeding `usePreview` directly, so *every step* of a drag
across it started a render: a proxy resample plus a full pipeline pass, at a size
nobody had stopped on, for each of up to 47 steps. The state moved into
`useValues` beside `referenceMp` and `lut` — where it belonged once it became a
preset fact anyway — and split in two the way the values already are:
`proxyEdge` is what the slider shows, `appliedProxyEdge` is what the renderer
sees, and the window `pointerup` listener that has always committed slider drags
now commits both. `onKeyUp`/`onBlur` cover the keyboard path, exactly as
`ParamControl` does.

The split is visible in who reads which: the preview and every size readout read
`appliedProxyEdge` (what is on screen), the slider and the export read
`proxyEdge` (what you asked for). An export is a click, so a release has already
committed by the time it fires and the two agree.

It stays out of the undo history for the reason the supersample and the mount
are out of it: undo is about the look, and a fidelity gesture would fill the
stack with steps that change no pixel of the picture's *content*.

Payload is a **JPEG q95 4:4:4** (`imageio.encode_preview`), not the PNG it used
to be: grain defeats PNG's predictor, so a level-1 PNG of a 2400px proxy measured
10.4MB and 108ms of zlib against 3.4MB and 24ms for the JPEG. 4:4:4 is not
optional — the default 4:2:0 would average away the chroma grain.

Deliberately not automatic after an idle delay: it is ~8s of work, and spending
that every time a drag settles burns it on frames you are about to change.

### Export can write either tier (added 2026-08-02)

`/api/export` takes `scale`: `"full"` (default, unchanged — the source at 1.0,
tile 1024) or `"preview"`, which renders `up.proxy` at `up.proxy_scale` with
`tile=1536`, i.e. the *identical call* `/api/preview` makes. Verified end to
end: a preview render and a preview-scale png8 export of the same parameters
are **bit-identical, max abs diff 0**. Keep it that way — if the two calls ever
drift apart, "export what I am looking at" quietly stops being true and there
is nothing on screen to show it.

The user asked for this because the 1:1 render is *unpredictable from the
preview*, and that is not a bug to fix — it is invariant 2 working. Every
spatial parameter is a length in full-resolution pixels times the working
`scale`, so scale invariance promises the two tiers agree about *the picture*
and promises nothing about grain per output pixel. Two places they diverge, and
both bite hardest exactly where the user is looking:

* **The proxy cannot resolve its own finest structure.** The default 1.6px
  clump is 0.64px on a 24MP frame's 0.4x proxy — sub-pixel, so it renders as
  something smoother than the 1:1 version of the same numbers.
* **`_MIN_CELL` floors what is left.** At 0.4x that same clump asks for a
  0.64px lattice and gets 0.8px, so the proxy's finest grain is 25% coarser
  *relative to the picture* than the export's. Finer `grain_size` settings
  diverge further; the floor does not move.

So this is a **look**, not a resolution. Downscaling a full export to 2400px
does not reproduce it — that grain was drawn on the source's grid and then
averaged away, which is a different operation from drawing it on the proxy's.

Two details in the implementation:

* Preview-scale filenames carry the long edge (`plate_grain_2400px.jpg`), and
  only when `proxy_scale < 0.999`. Two files from one photo differing only in
  resolution are indistinguishable in a folder, and the small one is the
  surprising one. On a sub-2400px source `up.proxy is up.arr`, so both options
  render the same pixels and the tag would be a lie.
* `_params_for` still rescales against the **full** image's megapixels, not the
  proxy's. A preset's `reference_mp` is about the photograph, not about which
  tier is being written, and doing it the other way would make a preview-scale
  export disagree with the preview it is meant to reproduce.

What the client-side scaling bought:

* Zoom and pan are free. They never re-render and never hit the network. The
  wheel zooms, and two details in it are not optional: it is attached with
  `addEventListener(..., {passive: false})` rather than as React's `onWheel`,
  because **React registers wheel listeners as passive** and `preventDefault`
  is a no-op there — the React version zooms the photo and scrolls the page
  under it at once. And it is **anchored on the pointer**: the image point
  under the cursor has to still be under it afterwards, or zooming into a
  detail walks it off screen. The anchor is read from the frame's own
  `getBoundingClientRect` rather than recomputed from `center`, so it inherits
  `place()`'s clamping instead of duplicating it. Fit is a *mode*, not a
  number — it tracks the container so a resize keeps the frame visible — so
  the wheel locks to it within `FIT_SNAP` of the fit value, on either side, and
  is no longer floored at Fit going the other way -- see the wheel-zoom section
  below for a real bug this uncovered.
* The server-side crop grid-phase problem is gone. It used to be that below 1:1
  the read origin had to be **snapped** so `origin * scale` landed on a whole
  working pixel, because a crop starting mid-pixel resolves on a different grid
  phase than a whole-image downscale — invisible on smooth areas, a glaring
  half-pixel shift on hard edges. That is why `ZOOM_STEPS` used to be clean
  fractions. Nothing resamples on the server now, so arbitrary zoom is safe.

The honesty problem this creates, and how the UI handles it: enlarged past its
own resolution the proxy is soft, and a soft preview reads as a soft *result*.
The stage shows a `proxy` badge whenever `eff > proxy_width / source_width` —
where `proxy_width` is now derived from the edge this session is rendering at,
via `prescaleGeom`, so lowering the edge makes the badge appear sooner exactly as
it should. The panel says to Render 1:1 before judging grain. Do not remove
those.

### A third export tier: the preview's look, at the source's own size (2026-08-05)

Requested: an export that guarantees a pixel match to the on-screen preview
(just enlarged), for people who want *that specific look* -- the proxy's
softer, coarser grain -- as a full-size file, rather than the finer grain a
fresh full-resolution render would resolve at the same settings. Before
building it, the two readings of "export what I'm looking at, at full
resolution" were put to the user explicitly, because they are genuinely
different features with different costs:

* **Re-render at full resolution with the current settings.** This is what
  `"full"` already is and always was -- it always uses whatever is currently
  dialled in, it just does not pixel-match the proxy, because grain is
  resolved on a finer grid at 1.0 scale. Free: no new code.
* **Upscale the exact proxy image.** Guarantees the pixel match, at the cost
  of adding no real detail -- zoomed in, the file shows the same soft texture
  as the proxy, just enlarged.

The user chose the second, so `scale: "preview_full"` is a new third value
alongside `"full"`/`"preview"` on `/api/export`. It renders through the
*identical* `_render_tier(up, p, ss, False, ...)` call `"preview"` already
uses -- the same one-call-site discipline that keeps `"preview"` byte-for-byte
the live preview -- and then a new `imageio.upscale(out, up.h, up.w, DEVICE)`
blows the result up to the source's own dimensions with a plain bicubic
resize.

**`upscale` is deliberately not `downscale` run backwards.** `downscale` passes
`antialias=True` because it is throwing samples away and aliasing is a real
risk; `upscale` omits it, because there is nothing to alias against when
*adding* samples -- the same plain bicubic `render_supersampled` already uses
for its own upsample step, not `downscale`'s antialiased one. It returns the
input array itself, not a copy, when the requested size already matches (the
common case once a source is no bigger than the proxy long edge, where
`"preview"` and `"preview_full"` render bit-identical pixels already).

Two things needed to stay honest once a third tier existed:

* **The filename has to say which kind of "same size as full" this is.**
  `"preview_full"` writes the source's own pixel dimensions, so *size* alone
  can no longer tell it apart from `"full"` in a folder -- unlike `"preview"`,
  where the resolution tag already did that job. So `"preview_full"` tags the
  filename `_grain_previewlook` instead of a pixel count, and both the pixel
  tag and the look tag are skipped together whenever the source was never
  bigger than the proxy in the first place (`proxy_scale >= 0.999`), where
  every mode renders the same file and tagging one as different would be a
  lie -- the same reasoning `"preview"`'s tag already followed, now shared by
  a `downscaled` flag rather than checked inline twice.
* **The UI has to say this file has no more detail than the preview.** The
  hint text under the new option says so outright ("adds no detail... not a
  fresh full-resolution render"), the same way the existing `"preview"` hint
  already warns that full-size settings resolve finer grain than the proxy
  shows -- this is that same honesty problem from the other direction.

Verified end to end against a running server rather than by inspection alone:
uploaded a source above the proxy threshold, requested `"preview_full"`, and
confirmed the downloaded file's pixel dimensions equal the source's exactly
and the filename carries `_grain_previewlook`; then repeated all three scales
against a source *below* the threshold and confirmed all three collapse to
the plain `_grain` tag and an identical byte size, matching the existing
"both options render the same pixels" case extended to three.


### The export scale now defaults to "As previewed, full size" (2026-08-08)

`exportScale` opens on `preview_full` rather than `full`, on request. The
reasoning is the section above read from the user's end: what you dialled in is
what the preview showed you, so the file that matches it is the one that starts
from those pixels and enlarges them. A full-size render of the same numbers is
not a sharper version of that picture, it is a *different* one -- finer, denser
grain, for the two reasons listed above -- so having it be the default made the
exported file quietly disagree with the screen it was judged on. `full` is still
there and still the right answer when you want the frame's own finest grain;
`Render 1:1` is how you look at it before committing.
