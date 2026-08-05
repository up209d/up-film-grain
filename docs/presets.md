# Presets

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

