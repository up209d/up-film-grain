# Film Grain Engine

Organic film grain, edge destruction and halation for still photographs.
Python/PyTorch image service behind a React UI. See `TOPIC.md` for the domain
rules and pipeline design.

## Requirements

* Python 3.13 (via `pipenv`) — PyTorch has no 3.14 wheels yet
* Node 24 (via `nvm`, pinned in `.nvmrc`)

## Setup

```bash
export PIPENV_VENV_IN_PROJECT=1
pipenv --python "$(brew --prefix)/bin/python3.13" install

nvm use            # reads .nvmrc
cd web && npm install && npm run build && cd ..
```

> On this machine Homebrew and Laravel Herd sit ahead of nvm in `PATH`, so
> `nvm use` alone may not switch `node`. The scripts below prepend nvm's bin
> explicitly.

## Run

```bash
./dev.sh          # http://localhost:5173  -- Vite HMR + uvicorn --reload
./run.sh          # http://127.0.0.1:8000  -- production, from the source tree
```

`run.sh` passes extra arguments through to uvicorn, e.g. `./run.sh --port 9000`.
Both accept `PORT=8001` and refuse to start with a clear message if the port is
already held.

## Production build

```bash
./build.sh                # compile a distribution into build/
./build.sh --venv         # ...and install its dependencies (slow -- torch)
./build.sh --clean        # wipe build/ first
cd build && ./run.sh      # run the distribution
```

`build/` is self-contained apart from the Python environment: the compiled
client, the server package (no tests, no dev scripts, no client sources), a
`requirements.txt` frozen from `Pipfile.lock` so the target installs exactly
what was tested, a `VERSION` stamp, and its own launcher. Copy it anywhere.

The launcher finds an interpreter in this order: `$FILM_GRAIN_PYTHON`, then
`build/.venv`, then `python3` on `PATH`. If none of them has the dependencies
it says so and prints the two commands that fix it. To reuse this project's
environment without installing a second copy of torch:

```bash
cd build && FILM_GRAIN_PYTHON=../.venv/bin/python ./run.sh
```

`HOST=0.0.0.0 ./run.sh` will expose it on the network. There is no auth and no
rate limiting, so only do that on a network you trust.

### What "production" changes

`APP_ENV` selects the mode and **production is the default** — dev is the mode
that needs holes in it, so it has to ask. `dev.sh` sets `APP_ENV=development`.

| | development | production |
|---|---|---|
| CORS for `localhost:5173` | on (Vite is a separate origin) | off (same-origin) |
| `/docs`, `/openapi.json` | served | 404 |
| missing client build | 503 with a hint | refuses to start |

That last row is deliberate: in dev the client normally comes from Vite, so a
missing build is expected. In production nothing else serves the UI, so a
process without one is broken rather than degraded and should not boot.

## Using it

* **Open image** or drop a file on the canvas. JPEG or PNG, up to 30MB.
  Multi-frame camera JPEGs (MPO — burst and 3D captures) are read as their
  primary frame.
* **Zoom lives on the preview itself** — the `Fit / − / % / +` bar in the top
  right, and drag to pan once the image is larger than its pane. None of it
  re-renders, so navigating is instant.
* **Editing renders a 2400px proxy** (~1.25s on a 24MP source) so sliders stay
  responsive. **Render 1:1** re-renders the whole frame at full resolution
  (~9.5s on 24MP); at that point the preview *is* the export, same pixels, same
  coordinates. Any adjustment drops back to the proxy.
* Judge grain at 100% zoom **after** Render 1:1. Enlarged past its own
  resolution the proxy is soft — a `proxy` badge appears on the preview to say
  so, because a soft preview otherwise reads as a soft result.
* **Compare** has two modes. *Overlay* stacks them under a wipe (hold **B** to
  peek at the original outright); *Side by side* puts the source and the result
  in two panes that zoom and pan together — one drag moves both.
* **Quality** selects supersampling. 2× is the default; 1× is faster but gives
  grain a hard, aliased pixel footprint.
* **Export full size** runs a tiled background render with progress. **JPEG 95
  is the default**: measured on a grained frame it keeps 100.2% of the grain
  sigma at 0.43MB against 9.82MB for 16-bit PNG — 23x smaller for no loss of
  grain *amount*. It is encoded 4:4:4, because chroma subsampling would smear
  the chroma grain away. What it does cost is up to ~1.7% per-pixel deviation,
  almost all of it ringing at high-contrast edges.
* Choose **PNG 16-bit** when the file is going on for further grading — 8-bit
  visibly posterises grain in smooth areas, and JPEG's edge ringing compounds
  through another round of processing.

Sliders only render on release, not during the drag — a fit preview is seconds
of work, so rendering every intermediate position just queues frames that are
stale on arrival. The number beside each slider still tracks the thumb live.

### Presets

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

## Layout

```
server/
  params.py    parameter schema -- single source of truth for engine and UI
  engine.py    the grain pipeline (see module docstring for the invariants)
  imageio.py   decode/encode, including a 16-bit RGB PNG writer
  main.py      FastAPI service
web/src/
  App.tsx      UI, schema-driven slider panel
  api.ts       typed client
```

## Verify

```bash
pipenv run python tests/verify.py
```

Checks tile independence, crop fidelity, colour pass-through, luminance
response, edge bias and 16-bit PNG validity. Run it after touching
`server/engine.py`; it exits non-zero on failure.

## Invariants worth not breaking

* **Tile independence** — no stage may depend on a statistic of the region being
  rendered. Break this and exports grow seams that previews never show.
* **Scale invariance** — spatial quantities are in full-resolution pixels,
  multiplied by the working scale. Break this and the preview stops predicting
  the export.

## Not implemented

* RAW input (needs `rawpy`/LibRaw)
* The neural pipeline — needs a paired film-scan dataset that does not exist yet
