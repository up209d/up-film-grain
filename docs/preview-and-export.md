# Preview and export

## Every export is full size (2026-08-08, on request)

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
supersample setting.

Filenames carry the factor (`photo_grain_ss1_5.jpg`) unless it is the default,
for the reason the old `_grain_2400px` tag existed: two files from one photo
that a folder listing cannot tell apart are worth naming apart, and now that
every export is the same dimensions, size cannot do it.

## The preview is client-scaled, two-tier (changed 2026-07-31)

`/api/preview` renders **the whole frame**, never a crop. The request carries
`id`, `params`, `supersample` and `full` — no mode, no zoom, no viewport. The
browser does all the scaling.

`full` picks the fidelity, and it is the only difference between the two:

* `false` (default) — the working proxy, `PROXY_LONG_EDGE = 2400`. This is what
  every slider change triggers. At *parameter defaults* it is a few hundred ms;
  what it actually costs depends almost entirely on the preset — see the
  performance section, where `Stock` measures 8.8× defaults.
* `true` — the whole source at scale 1.0. The preview *is* the export at this
  point. Only ever fired by the explicit "Render 1:1" button; any parameter
  change drops back to the proxy.

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
The stage shows a `proxy` badge whenever `eff > proxy_width / source_width`, and
the panel says to Render 1:1 before judging grain. Do not remove those.

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
