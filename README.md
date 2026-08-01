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
* **Zoom lives on the preview itself** — **scroll over the photo** to zoom, or
  the `Fit / − / % / +` bar in the top right; drag to pan once the image is
  larger than its pane. Scroll zooming is anchored on the pointer, so the
  detail under the cursor stays under the cursor instead of walking off screen.
  It is continuous rather than stepping through the bar's stops, and it snaps
  back to `Fit` within 2% of it so a window resize still keeps the whole frame
  visible. A trackpad pinch (ctrl+wheel) works too. None of it re-renders, so
  navigating is instant.
* **Editing renders a 2400px proxy** (~1.25s on a 24MP source) so sliders stay
  responsive. **Render 1:1** re-renders the whole frame at full resolution
  (~9.5s on 24MP); at that point the preview *is* the export, same pixels, same
  coordinates. Any adjustment drops back to the proxy.
* Judge grain at 100% zoom **after** Render 1:1. Enlarged past its own
  resolution the proxy is soft — a `proxy` badge appears on the preview to say
  so, because a soft preview otherwise reads as a soft result.
* **Compare** lives on the preview's own bar, beside the zoom controls — it
  changes the view, not the render, so it belongs with the rest of the view.
  *Overlay* stacks the two under a wipe (hold **B** to peek at the original
  outright); *Side* puts the source and the result in two panes that zoom and
  pan together — one drag moves both.
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

### Light leaks

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

### When halation greys the sky

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

### Softening without blurring

Two controls in **Optical** do the same physical job from opposite ends, and
which one you want depends entirely on whether you mind losing texture.

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

### Preset size scaling

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
response, edge bias, scatter, blue compensation and 16-bit PNG validity — 116
checks. Run it after
touching `server/engine.py`; it exits non-zero on failure.

## Invariants worth not breaking

* **Tile independence** — no stage may depend on a statistic of the region being
  rendered. Break this and exports grow seams that previews never show.
* **Scale invariance** — spatial quantities are in full-resolution pixels,
  multiplied by the working scale. Break this and the preview stops predicting
  the export.

## Not implemented

* RAW input (needs `rawpy`/LibRaw)
* The neural pipeline — needs a paired film-scan dataset that does not exist yet
