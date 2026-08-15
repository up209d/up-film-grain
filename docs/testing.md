# The check suite

`verify.py` was one 3900-line `main()` running 345 checks in a single process.
It took **4m24s** and measured **83% of one core** on a 14-core machine —
thirteen cores idle while a chain of GPU renders went past one at a time. Split
into modules on 2026-08-08 it runs in **36s** (**42.7s** since the LUT folder
grew to 303 files — see the last section), and the module covering whatever you
just touched runs in seconds.

Nothing about *what* is checked changed. The split was mechanical — line ranges
lifted verbatim, no reindentation, because the body of the old `main()` and the
body of the new `run(cx)` sit at the same indent — and the proof is that the
output is byte-identical to the pre-split log: 345 checks, same names, same
order, same measured values down to the last digit. If you ever restructure
this again, hold it to that bar. "The suite still passes" is not the same claim.

## Layout

```
tests/verify.py      the CLI, and the only thing you run
tests/runner.py      selection, scheduling, reporting
tests/harness.py     check(), the registry, Ctx (the fixtures), shared metrics
tests/checks/        one module per area -- where the checks actually live
tests/refs.py        slow reference implementations the rewrites are held to
tests/scene.py       the synthetic test scene
```

## Running it

```sh
pipenv run python tests/verify.py                 # everything, in parallel
pipenv run python tests/verify.py edges scatter   # only those modules
pipenv run python tests/verify.py global          # every module matching
pipenv run python tests/verify.py -l              # list the modules
pipenv run python tests/verify.py -j 1            # one process
pipenv run python tests/verify.py --times         # per-module seconds
```

Run the modules covering what you touched while you work; run the lot before you
call it done. Reach for `-j 1` when a check *fails* — a traceback from a worker
process arrives without the surrounding output, and in one process you get the
whole log in order with the failure in place.

## The modules

| module | covers |
|---|---|
| `tiling` | tile independence, crop fidelity, zoom fidelity |
| `colour` | colour pass-through, the Original button, vibrance, split tone |
| `presets` | `reference_mp` rescaling, the mark-count dead zone |
| `normalize` | auto exposure, auto white balance, the toe and shoulder, the metering's direction |
| `grading` | the Colour Grading section, highlight reconstruction, `.cube` parsing |
| `response` | the luminance-response band, edge bias, the smooth-area guard |
| `global_grain` | global grain unmasked and tile-independent, and its chroma |
| `global_layers` | the source-masked layers and the six blend modes |
| `global_field` | the grain point field — freedom from the pixel grid, structure, amplitude |
| `global_mix` | master opacity, global smoothness |
| `edges` | anti-aliasing, edge softening, pre-blur, jitter, sanding |
| `scatter` | scatter — diffusion without the average |
| `halation` | blue compensation, highlight recovery |
| `sharpen` | output sharpening |
| `film_texture` | dust, hair, scratches, light leaks, speck shape, mark counts |
| `film_tiling` | that drawn marks reserve no tile overlap |
| `internals` | bit-exact performance rewrites, the texture cache, tile size, cancellation |
| `imageio` | the hand-written 16-bit PNG writer, upscale |

The grouping is by **what you would re-run after touching one thing**, not by
tidiness. Touch `engine/stages/global_*` and the honest re-run is
`verify.py global` — four modules, ~20s — not the whole suite.

## Why it is faster

**Selection** is most of it in practice. The suite you actually run while
working is one or two modules.

**Processes**, not threads: the metrics between the renders are numpy and torch
calls that hold the GIL for long stretches, so threads would serialise on
exactly the part that was already serial. `spawn`, not `fork` — a forked child
inherits a half-initialised MPS context and deadlocks the first time it touches
the GPU.

Nothing was made cheaper by rendering less. No scene shrank, no supersample
dropped, no tolerance loosened; the constraint from `CLAUDE.md` that quality
beats speed applies to the checks too, and a check that is fast because it looks
at less is not the same check.

The floor is now the **critical path**, not the total: wall-clock is the longest
single module, because everything else finishes underneath it. That is why
`edges` and `film_texture` were each cut in two — at 41s and 34s they *were* the
wall-clock. Both halves now sit near 30s, and 36s wall against a 32s longest
module means there is nothing left to win by adding workers. The next real gain
would be cutting `edges`, `global_layers` and `film_tiling` again.

`COST` in `runner.py` is measured wall-clock per module, used only to start the
long ones first so the pool does not finish with one worker still grinding while
the rest sit idle. Stale numbers cost a little scheduling, never correctness —
refresh them from `--times` when they drift.

## The fixtures

`Ctx` in `harness.py` holds the engine and the scenes. Every fixture is a
`cached_property`, so a module that never touches `big` never pays to build it —
which is what makes single-module runs seconds rather than minutes.

They are fixtures rather than per-module locals for a reason that outlives the
performance argument: `global_grain`, `edges` and `sharpen` are all measured
against **the same smooth-patch sigma**, and `film_texture` and `film_tiling`
against **the same "everything else off" dict**. Those were shared locals in the
old single function, so the sharing was accidental but real. Copying them per
module would have let them drift apart, and three modules quietly disagreeing
about what they had switched off is a worse bug than a slow suite.

The cost of that independence: under `-j`, two modules in different workers each
build `big_residual` themselves. One extra render, in parallel, in exchange for
modules that mean the same thing whether you run them alone or together.

## Adding a check

Put it in the module covering the area. If it needs a stage that ships at 0,
switch that stage on and **re-run tile independence with it on** — every one of
those stages adds work `pad_for` has to cover, and a kernel missing from
`pad_for` seams tiled exports along exactly its radius while every preview looks
fine.

The trap that has caught this suite before: `sanitize(None)` fills in
*defaults*, not zeros, so an override dict has to zero every other stage that
could contribute to the same measurement. The sanding check failed first time
round because `edge_jitter` defaults to 0.3 and was quietly adding its own
wander to it.

If a new module is genuinely its own area, add it to `tests/checks/`, list it in
`tests/checks/__init__.py` — `ORDER` is both the print order and the import list
— and give it a `COST` entry. A module with no `COST` entry still runs; it is
just scheduled as if it were average.

## The `grading` module got slower on purpose (2026-08-09)

`luts/` went from 7 `.cube` files to 303 when a library arrived as
`luts/gmic/`, and `"every LUT in luts/ loads"` parses every one of them. The
module went from ~4s to **7.9s** and the suite from 36s to **42.7s**.

It is not the critical path — `edges` still is — so the whole cost is real but
none of it is on the wall clock that matters. And it is the check that catches
a malformed `.cube` shipping in a folder nobody opens by hand, which is exactly
the failure that would otherwise surface as one preset quietly grading nothing.
Per `CLAUDE.md`: quality beats speed.

Two things had to change with it, both about the *log* rather than the timing:

* the detail string prints `303/303 parsed` and then the failures, not a roll
  call — 303 names would bury every other line in the module;
* a new `"every .cube on disk is listed"` counts the tree independently and
  compares. The walk is what silently regresses: `glob` instead of `rglob` still
  lists seven LUTs and passes every other check in the file.
