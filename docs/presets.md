# Presets

## Preset files carry every parameter now (2026-08-09)

They used to carry whatever the author happened to have moved -- `Subtle` had 41
of 112 keys, `Sandy` 54, `SuperPortra` 108 -- and the rest filled from defaults
through `sanitize`. That is still how they *load*, and a hand-written
`{"intensity": 40}` still works; what changed is what we write.

Every shipped preset now lists all 112, **in panel order**, at the value it was
already rendering. Verified bit-identical across all twelve (0.00e+00) -- of
course it is, since the values written are exactly what `sanitize` was filling
in, but a preset library is the wrong place to assume that rather than measure
it.

Two reasons it is worth the extra lines:

* **A file that says what it does.** Reading a preset used to require knowing
  which of 112 defaults it was inheriting, and the answer changed whenever a
  default did. Now the file is the whole answer.
* **New controls arrive visible.** `edge_sensitivity` and `edge_chroma_sense`
  were added on 2026-08-09 and both ship at 1. Left implicit, every preset would
  silently inherit whatever those defaults became; written out, changing a
  default no longer reaches back into looks somebody already dialled in.

Panel order rather than insertion order for the same reason the pipeline runs in
it: the file reads top to bottom the way the app does.

## The mark-count dead zone (found 2026-08-02)

`dust`, `scratches`, `hair` and `light_leak` are **counts**, and the engine
gates each on `>= 1.0` — you cannot render a third of a scratch. So any value
in **(0, 1) renders nothing at all** while reading, in the panel and in the
file, as though the section were slightly on.

Three shipped presets sat squarely in it. `Organic`, `Dreamy` and `Dreamy+1`
carried `dust 0.62`, `scratches 0.48`, `hair 0.10`, `light_leak 0.05` — 0–1
*amounts* from before these parameters became counts, never migrated. Their
entire Film Texture section had been silently inert, which is how the user
came to report light leaks as not rendering: nothing they did to the leak
sliders could matter while the count was 0.05.

Migrated to **0**, not to a count. Zero is what those presets have actually
been rendering ever since, so it is the faithful migration; rounding 0.62 up to
1 would change their look without being asked.

`verify.py` now refuses any shipped preset with a count in (0, 1). It is worth
a check rather than a comment because it is invisible from both ends — the code
looks right, the file looks deliberate, and the UI shows a number.

## Presets rescale across image sizes (added 2026-08-01)

A preset dialled in on one photo is locked to that photo's size: every spatial
parameter is a length in full-resolution pixels, so the same numbers on a
bigger frame give proportionally finer grain and tighter halation. Preset files
now carry `reference_mp`, the size they were authored at, and the client sends
it with every render; `_params_for` rescales before the engine sees anything.

**The ratio is linear, not area.** Eighteen parameters are marked
`spatial=True` and multiplied by `sqrt(current_mp / reference_mp)`. A 16MP
frame is 0.816x the *width* of a 24MP one, not 0.667x -- scaling lengths by the
megapixel ratio overshoots by the square root. `edge_jitter` is in that list
despite having no `px` unit: `_JITTER_MAX` makes it a length multiplier.

Not rescaled, on purpose: amounts and blend weights (dimensionless, per-pixel),
and mark counts (already resolved against frame area inside the engine, so
50 specks is 50 specks at any size). Leak sizes and the leak feather *are*
rescaled now that they are pixel lengths; they used to be exempt as fractions.

Measured on the same scene at 6MP and 15.4MP, both resampled to a common 900px
display width and grain isolated against a same-parameter grain-off render:
without scaling the larger frame carries **57%** of the authored grain sigma;
with it, **107%**. The residual 7% is the clump curve and the `_MIN_CELL` floor,
not a systematic error.

Two traps when measuring this:

* Resample both renders to the *same* display width. Downsampling by integer
  factors lands them at different sizes (1000px vs 1200px) and the comparison
  is meaningless.
* Isolate grain against a grain-off render using the **same rescaled**
  parameters. Using the unscaled ones puts the halation and blur rescaling into
  the residual too, which reported 148% instead of 107%.

Files without `reference_mp` -- everything authored before this -- scale by
1.0, so the behaviour is unchanged rather than guessed at. There is
deliberately **no built-in default size**: inventing one would silently change
the look of every legacy preset, and a wrong guess is worse than no scaling.
Two ways to populate it instead:

* Per preset: open a photo of the size it was dialled in on, press **Set from
  photo**, then **Save to file…**.
* All at once: `FILM_GRAIN_DEFAULT_REFERENCE_MP=24` makes every preset with no
  recorded size be treated as authored at 24MP.


## Prescaling is the same correction from the other end (2026-08-29)

Everything above moves **the parameters** to fit the photograph. `Prescaling
Source` moves **the photograph** to fit the parameters, and since every file in
`presets/` records `reference_mp: 24` and is now stamped `prescale_mp: 24`, the
factor this section computes is 1.00x whenever prescaling is on at its default.

That is deliberate, and it means the two must not both do the work:
`params_for` measures `scale_factor` against the *prescaled frame*, not the
file. It also removes a real loss this mechanism has and cannot avoid -- the
clamp back into `[min, max]` below, which silently pins parameters at their
sliders' ceilings on a large upscale. See `docs/prescale.md`.

Note the reasoning at the top of this section -- "there is deliberately **no
built-in default size**: inventing one would silently change the look of every
legacy preset" -- is knowingly *not* followed by `prescale`'s default of on at
24MP. That was requested outright, so a legacy preset file with no `prescale`
key does now prescale. The difference from the case above is that this one was
asked for with its consequence stated, rather than guessed at.

## The scaling factor can be set by hand (added 2026-08-08)

The Size Scaling section offered the computed factor or nothing at all, and
requested: a manual override. `Auto` / `Manual` sits under the on/off switch,
and Manual starts at whatever automatic had -- switching to it moves nothing, so
you adjust from the computed answer rather than being dropped at 1.00x and
having to find your way back.

Two things about how it is plumbed:

* **The API never learns about it.** The server scales lengths by
  `sqrt(photo_mp / reference_mp)`, so a hand-set linear factor `f` is exactly the
  reference size that solves that equation: `reference_mp = photo_mp / f²`. App
  computes that and hands it to the render and export hooks, while
  `useValues.referenceMp` goes on holding what the *preset* says. So saving a
  file still stamps the truth rather than a fudged size derived from a
  preference.
* **It is a view choice, not part of the look.** It lives in `App` beside
  `supersample` and the mount, is not written into a preset file, and does not
  travel with one. A preset records what size its numbers were authored at,
  which is a fact about the file; the override is a preference about this
  session.

It also makes the on/off switch live for a preset that records no size at all --
a hand-set factor needs no reference to scale against, so the "n/a" state now
only applies when both are absent.

## Attribution survives a re-save (2026-08-19)

Every file in `presets/` has carried `author` and `author_link` since the
library was written, and until now nothing read them. `load_presets()` builds
its own dict rather than passing the parsed file through, so the two keys were
dropped at the door; "Save to file…" then wrote a fresh object from the client's
state, which had never been told about them. The result is that the credit on a
shipped preset survived exactly until someone nudged one slider and saved — the
new file was the same look with nobody's name on it.

They are carried now, along the same path `lut` takes and for the same reason:
a credit is not a quantity, so it cannot be a value in the schema.
`load_presets` → `Preset` (the hand-written mirror in `api.ts`, where a field
left out is simply invisible) → `useValues.author` → `usePresetFile.savePreset`.

Three details worth knowing:

* **It is one `Author` object, not two loose strings.** A link with nobody's
  name on it is not attribution, so `presetAuthor()` drops a lone `author_link`
  and the whole credit is null or complete. Loading a file is the other
  direction of the same rule: a name with no link is still a name.
* **A look with no author writes no author keys**, rather than `"author": null`
  — the file shape a preset saved from scratch produces is the one the format
  had before this existed. It is spread into the object rather than assigned,
  and it sits directly after `name`, so a re-saved shipped preset diffs against
  the original on the values alone.
* **It is in the undo `Snapshot`.** `applyPreset` sets it *including its
  absence*, the way it already did for the LUT, so picking an unattributed
  preset clears the previous one's name instead of leaving it to be written out
  over someone else's look. Undo has to put that back or a step backwards lands
  on a state the user never had.

`tests/checks/presets.py` asserts the server half — every file that names an
author reports one, and the link comes with it. The client's write-back has no
automated check; it is a hand test (pick a preset, Save to file…, look at the
first four keys).

## The preset records which proxy tier it was judged on (2026-08-29)

`proxy_edge` joined `reference_mp`, `lut` and the two author keys at the top
level of a preset file, on request, and every shipped preset is stamped
`"proxy_edge": 2400`. It is a sibling of the values and never one of them, for
exactly the reason the other four are: the engine never reads it. It decides
which *frame* the engine is handed, the way `reference_mp` decides which lengths
it is handed.

It is in the file rather than in the session because it is in the **output**.
Five of the six export entries render the proxy tier and enlarge it
(`docs/preview-and-export.md`), so two exports of the same values at two edges
are two different pictures. A number that changes what comes out of the export
cannot be a preference about this session — which is what it had been until
today, and the note in that file that said so has been rewritten rather than
deleted, because the reasoning that got it wrong is the useful part.

**Precedence, resolved in `controllers/export.py` and nowhere else:** the
request body's `proxy_edge` if it names one, else the named `preset`'s, else
`PROXY_LONG_EDGE`. The CLI's `-e` therefore has no argparse default — `None` is
what keeps "not declared" distinguishable from "declared as 2400", and a default
would have made every `./export.sh` run silently override the preset it was
given. `./export.sh photo.jpg -p Stock` renders Stock's edge; `-e 800` wins over
it; a preset written before the key existed falls through to 2400 and renders
what it always did.

Client side it seeds the slider: `startingValues` returns it for boot and Reset,
`applyPreset` sets it *including its absence* (a look that names no edge is a
look dialled in at the default, so inheriting the last preset's edge would render
it at a texture its author never saw), and `usePresetFile` writes it back out
directly after `reference_mp` so a re-saved shipped preset still diffs on the
values alone. Loading a hand-written file is the one asymmetric case: the key is
read only when present, because `{"intensity": 40}` is a set of numbers rather
than a whole look and dropping the session back to 2400 would re-render at a tier
nobody asked for.

`tests/checks/prescale.py` pins all four paths at the filename tag — the only
place the resolved edge is observable from outside — with `load_presets` stubbed
rather than a file dropped into `presets/`, plus one check that every shipped
preset carries the key.
