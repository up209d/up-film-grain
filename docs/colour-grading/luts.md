<!-- part of docs/colour-grading.md -->

## Colour Grading: a LUT is a *resource*, not a parameter (added 2026-08-04)

Requested: a section at the top of the pipeline that applies a 3D LUT from
`luts/` or from a file, with temperature, shadow, highlight and two-way clarity
sliders **before** the LUT, and cheap enough not to stress the main pipeline.
Step -1 in `render()`, above `pre_blur`. Everything ships at 0.

**The structural decision, and the one that shapes everything else: the LUT does
not live in `params.py`.** Every other control the engine takes is a float with a
range, so it can be sanitised, clamped, rescaled for a different image size and
stored in a preset file as a value. A LUT is identified by *name* and its content
is a table. So it travels beside the parameters — `body["lut"]` next to
`reference_mp`, and a `lut` sibling key in a preset file — and `main._params_for`
attaches the resolved object as `p["lut"]` after `sanitize` and `rescale`, both of
which only touch keys that are in `PARAMS` and so leave it alone.

The obvious alternative needed no new plumbing at all: a `choices` menu indexed
into the folder listing. It is wrong for exactly the reason `_SCATTER_STENCILS`
documents, and worse here — that list is fixed in code, whereas `luts/` is
user-mutable *by design*, the same way `presets/` is. A preset stores the index,
so dropping one more `.cube` in the folder silently renumbers it and changes the
look of every preset that named one. Names it is.

**`lut_amount` *is* a parameter, and it is in `NEUTRAL_ZERO`.** That pair is what
keeps the Original button honest. `params.is_neutral` decides whether
`render_image` short-circuits, and it works from the numbers alone — it cannot see
the LUT. So:

* Zeroing the mix switches the LUT off as completely as unselecting it would,
  which is why the *name* stays out of `NEUTRAL_ZERO`: same reasoning that keeps
  sizes, radii and seeds out of it, so the section remembers what it had.
* **A mix above zero with no resolvable LUT would be a silent bug**, not a no-op:
  `is_neutral` would be false, the render would run, and at supersample 2 the
  bicubic-up/box-down round trip comes back a measured 1.0e-01 softer than the
  source. `_params_for` therefore forces `lut_amount = 0` whenever `lut.get`
  returns nothing, so the gate in the engine and `is_neutral` can never disagree.
  `verify.py` pins both halves.

An unresolvable name is deliberately **not** an error — a preset can name a
`.cube` that has since been renamed, or an upload from a previous run (those live
in process memory and do not survive a restart). The picker keeps the name as a
"— missing" entry with a hint rather than resetting itself to None, because
silently showing None makes it look like the preset never had a LUT.

### The four things that had to be right about the lookup

* **`align_corners=True`.** A LUT's first and last samples *are* input 0 and
  input 1, not the centres of edge cells. The default reads the whole table half
  a cell off — a small, uniform, entirely wrong shift that looks like the LUT
  being slightly wrong rather than like a bug.
* **The axis order.** `.cube` says red varies fastest, so a C-order reshape gives
  `table[b][g][r]`; permuted to `[c][b][g][r]` that puts red on `grid_sample`'s
  `W`, green on `H`, blue on `D`, which is why the sampling grid is just the
  image's own channels in order. Get this backwards and every *symmetric* LUT
  still looks fine while every real one is channel-swapped.
* **Both of the above are pinned by construction rather than by eyeball.**
  `verify.py` builds two exactly-linear 8-cubes — an identity and one that
  rotates the channels — and trilinear interpolation of a linear function is
  exact, so the check is an *equality* (2.4e-07) rather than a judgement. The
  rotation catches a transposed axis; the identity catches the alignment.
* **`F.grid_sample` in 3D works on MPS**, checked before building on it. One call,
  trilinear, so a 35-cube and a 64-cube cost the same and neither shows up against
  the stages below. The alternative — gathering eight corners by flat index —
  needs int64 index tensors MPS handles badly and eight full-frame gathers of
  working memory.

### Why each adjustment is where it is

* **Temperature in linear light.** A white balance is a change of *illuminant*, so
  it multiplies light, and gamma-encoded values are not light — done encoded, the
  same gain moves the shadows much further than the highlights, which is what
  makes a naive temperature slider read as a tint laid over the picture. Same
  argument as `pre_blur`'s, and gated the same way so the transfer round trip
  costs nothing at 0. The gain vector is normalised by its own luma, so the
  control is colour-only: measured, luminance holds to within 1% across the slider.
* **Shadows and highlights display-referred, clip-free, and — since 2026-08-05 —
  monotone.** Both halves of this were rewritten; the section below
  ("Shadows and Highlights were a brightness shift, not a recovery") is the
  authority and this bullet is only the pointer. The short version: the
  recovering directions are asymptotic rolls keyed on the channel maximum and
  applied as a uniform scale, so they are strictly monotone, gamut-safe by the
  curve's own bound, and hue-exact rather than hue-approximate; the expanding
  directions keep the original share-of-headroom form and `_GRADE_TONE_MAX`.
  `verify.py` pins the worst excursion outside 0..1 at 0.00e+00 *and* the worst
  transfer slope over twelve settings at +0.369. The two halves have disjoint
  supports about the knee, so they no longer need a shared luma to stay
  independent — the far end is bit-exactly untouched.
* **`_GRADE_TONE_MAX` is 0.35, not 1.0, and that is not a taste tweak.** At 1.0 a
  setting of +1 takes a black pixel to *pure white*, so the whole top of the
  slider is unusable and the useful range is squeezed into its first tenth.
  Measured on a real photograph (mean luma 0.21), Shadows at only **+0.5 took the
  frame's mean from 0.19 to 0.53** — that is a different exposure, not a shadow
  lift. Caught by rendering the actual photo through the actual API, not by
  reading the code. Same lesson as `_JITTER_MAX` from the other direction: the
  whole range has to be usable. Since the tone rewrite it governs the *expanding*
  half of each control only — a share-of-headroom cap is exactly what made the
  recovering half non-monotonic, so that half has no cap and its endpoint comes
  from the curve's own asymptote.
* **Clarity is asymmetric on purpose.** Positive gets `_GRADE_CLARITY_GAIN` (1.6);
  negative is pinned at exactly 1.0, because at gain 1 a setting of −1 subtracts
  precisely the band it measured — the local contrast is *gone*. Past that it does
  not keep flattening, it **inverts**: dark halos on the light side of every edge,
  an artifact rather than a look. `verify.py` measures the band's correlation with
  the source at −1 and fails on a negative number. Measured ladder: −1 → 5% of the
  band, −0.5 → 52%, +0.5 → 177%, +1 → 255%.
* **Clarity runs on luminance, which is both cheaper and better.** The signed
  detail goes to all three channels equally, so the channel *differences* — which
  is what hue is — come through untouched (pinned at 2.4e-07), a saturated area
  cannot be pushed out of gamut by a structure control, and it is one blur instead
  of three.

### Cost, and the one term in `pad_for`

Four of the five stages are per-pixel with no kernel and no neighbourhood, so
they reserve **nothing**. Clarity's high-pass is the only kernel in the section
and it is a real reach even though the stage runs first: a tile that cannot see
far enough measures a different band at its own edge, and that difference then
propagates through everything below it. `verify.py` pins both halves — that
`pad_for` grows by 3× the clarity radius, and that it is *unchanged* with
temperature, tone and a LUT all on.

(Highlight reconstruction, 2026-08-05, is the section's second kernel and the
only stage in it that was accepted as expensive on purpose — two three-channel
blurs. Both terms are summed in `pad_for`, not maxed, because reconstruction runs
above clarity and clarity's band is therefore measured on pixels reconstruction
has already changed from up to its own radius away. Measured: 33px off, 153px at
either kernel alone at a 40px radius, 273px with both.)

Measured on a 6MP render at 2×, best of 3 in fresh processes (MPS run-to-run
variance here is ±1s on larger frames, so single-shot numbers are worthless):

| | time | pad_for |
|---|---|---|
| section off | 0.67s | 108px |
| temperature / tone / a LUT at mix 1 | 0.67–0.73s, inside variance | 108px |
| clarity at the default 14px | 0.75s | 150px |
| clarity at 40px | 0.88s | 228px |
| all of it | 0.82s | 150px |

### Two things outside the section that had to change with it

* **`build.sh` copies `luts/`.** It already had this exact bug documented for
  `presets/` — a distribution without the folder has an empty LUT menu and a
  preset that names a `.cube` quietly grades nothing. It came back in a second
  form on 2026-08-09: the copy was `cp luts/*.cube`, which drops every
  subfolder, so the distribution shipped 7 of 303 and reported "Bundled 7
  LUT(s)" without complaint. It walks the tree now and counts what it wrote.
  A LUT's id *is* its path relative to `luts/`, so the layout has to survive the
  copy or every nested id in a preset stops resolving.
* **Editing a control in a muted section now switches that section on.** Found in
  a real browser, not by inspection: on a fresh load *every* section is muted (see
  the muted-on-boot section), so picking a LUT left the section's switch reading
  "off" while the LUT rendered — and a mute/un-mute round trip then reverted the
  mix to the snapshot `toggleGroup` took at mute time, measured going straight
  back to 0. `keptFor`/`liveFor` in `App.tsx` restore the section's kept values
  and lay the edit on top, which is exactly what clicking its own ● does. This is
  general, not LUT-specific: it was latent for every slider in the app the moment
  boot started muting everything, and the new section is simply where it is hit
  first. The pair is split into a pure half and a side-effecting half because a
  `setMuted` call inside a `setValues` updater would run twice under StrictMode.

## The folder became a tree (2026-08-09)

A library of 296 film-emulation LUTs arrived as `luts/gmic/`, organised into
nine folders — [Pat David's film emulation presets][pd] for [G'MIC][gmic],
converted to `.cube`. `luts/` had been flat and held seven files, and neither
end of the app survived the change untouched.

[pd]: https://patdavid.net/2013/09/film-emulation-presets-in-gmic-gimp/
[gmic]: https://gmic.eu/

**A LUT's id is now its path relative to `luts/`, extension dropped.**
`UP-SuperPortra` at the root, `gmic/colorslide/fuji_fp_100c` in a folder. Two
things follow, and the first is the reason for the second:

* **No preset needed migrating.** A root-level file's relative path *is* its
  bare stem, so the nine shipped presets naming `UP-SuperPortra`,
  `ClassicNegative` and `UP-Vintage` kept resolving without being touched. That
  is not luck — it is why a *path* was the right id rather than, say, a
  `folder:name` pair or a hash.
* **A bare stem could not stay sufficient.** `gmic/negative_new` and
  `gmic/negative_old` both ship a `kodak_portra_400`. Collapsing the tree into
  one namespace would make which of them a preset gets depend on the order the
  directory walk happened to return them in — the same class of bug as indexing
  a preset into a folder listing, which the top of this file rejects for the
  same reason. `verify.py` pins it from both sides: the nested path resolves and
  the bare name does *not*.

**The traversal guard had to be rebuilt, not relaxed.** `get()` used to reject
any id containing a separator. That is a perfectly good rule for a flat folder
and it was the *entire* path-traversal defence, so the moment separators became
meaningful the guard was gone. `lut.resolve_path` is the replacement and it is
strictly stronger than what it replaced:

1. no backslashes, no leading `/` — a Windows-style `gmic\bw\x` must not become
   a valid path on a POSIX box either;
2. no `.`, `..` or empty segment anywhere, so nothing climbs out textually;
3. and then, having built the path, `resolve()` it and require the result to
   still be under `LUT_DIR`.

Step 3 is the one a textual check cannot do: a symlink *inside* `luts/` pointing
at `/etc` passes 1 and 2 and is caught only by resolving. `verify.py` fires
eight probes at it — `..`, `../presets/Stock`, `gmic/../../presets/Stock`,
`/etc/passwd`, `gmic\bw\agfa_apx_100`, `gmic//bw/agfa_apx_100`, `.` and `""` —
and requires every one to resolve to no LUT. An end-to-end render with
`gmic/../../presets/Stock` as the LUT comes back bit-identical to a render with
none, which is the behaviour `params_for` guarantees by zeroing the mix.

**Listing still parses nothing.** `list_luts` walks with `rglob` and reports
`{id, name, size: null, source, group}` per file; 303 of them cost 9ms. `rglob`
does not follow symlinks, which matters more here than it did with `glob` — a
link pointing at `/` would otherwise walk the disk. `group` is the parent folder
relative to `luts/` (`""` at the root), reported rather than split back out of
the id in the client, because the id is a path and the server owns paths.

**The picker grew folders and a search box**, which is really a UI note but it
is downstream of the number: 303 entries in a flat menu is not a menu. Folders
are collapsed headings, the seven root LUTs sit above them and are always
visible, and the filter matches folder names as well as LUT names — typing
`instant` should get you the folder rather than nothing. Collapsed-by-default is
also what keeps it cheap: nine headings and seven rows are all that is in the
DOM until something is expanded. See `docs/client-ui.md`.

**The check that loads every LUT now loads 303 of them** and the `grading`
module went from ~4s to 7.9s (the suite, 42.7s). That is the check that catches
a malformed `.cube` shipping, so it stays; per `CLAUDE.md`, quality beats speed.
Its detail string prints a count and the *failures*, not a roll call — 303 names
in the log would bury every other line in the module.
