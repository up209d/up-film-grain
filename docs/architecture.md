# Architecture

Where things live, and the rules that keep the layout from rotting. Split out
of four single files on 2026-08-08 — `engine.py` (4958 lines), `params.py`
(1651), `App.tsx` (1913) and `main.py` (504). No behaviour changed: the split
was verified by 345/345 invariant checks, a byte-identical `/api/params`
payload, an identical CSS bundle hash, and a browser run of the real UI.

## The two rules

**1. Constants never import upward.** `engine/constants/` imports nothing from
the rest of the engine. Everything else may import it. This is what makes the
package acyclic — the first attempt had `constants/core.py` importing the grain
field because a *comment* mentioned it, and the import graph closed on itself.

**1b. `render()` is a sequence of section calls, not a sequence of stages.**
Added 2026-08-08, when the body went from 790 lines to 463. Halation, Tone
Response, Global Grain and Output Sharpening each moved into their own mixin
(`stages/halation.py`, `stages/tone.py`, `stages/global_grain.py`,
`stages/sharpen.py`) and appear in `render()` as one line each. The extraction
was bit-identical and asserted so by the whole suite.

It is not tidiness. Pipeline *order* is a design decision the panel is supposed
to reflect (see `docs/pipeline-order.md`), and reordering a call is reviewable
where reordering ninety inline lines is not. The stages still carrying their body
inline are the ones whose intermediates are shared across sections -- the edge
and grain span, where `lum_ref`, `hp`, `edge` and `m` are all live at once.

**2. Nothing above `params` may define a parameter.** `server/params/` is still
the single source of truth. Adding a control is one `Param` in the right
`definitions/` module and one `p["key"]` read in a stage. The UI picks it up
from `GET /api/params`; never hand-add a slider in a view.

## Server

`params/` is the model. `definitions/` holds the controls one module per panel
section, in the order they render; `registry.py`, `sanitize.py` and `schema.py`
are all derived from that list. `presets.py` walks up one directory more than
the old flat module did — it sits a level deeper, and getting that wrong points
`PRESET_DIR` at `server/presets` and boots with no presets at all.

`engine/` is the pipeline. `GrainEngine` is a composition root: it holds only
the device and the two texture caches, and every stage is a mixin under
`stages/`. Mixins rather than separate collaborators because every stage reads
the same engine state and the test suite calls them as engine methods — the
alternative was threading `self` through seven constructors to no benefit.

Stage *order* is not expressed by the file layout and cannot be: it lives in
`RenderMixin.render`, and `docs/pipeline-order.md` says what moving one breaks.

`controllers/` are thin FastAPI routers. Anything surviving a request is in
`models/`; anything a second caller needs is in `services/`. `runtime.py` holds
the process singletons so controllers never import each other. `is_superseded()`
is a function rather than an exported counter on purpose — `from .runtime import
_preview_gen` would snapshot the value at import and every render would believe
it was the newest one forever.

## Client

Model / controller / view, in `models/`, `controllers/` and `views/`.

`models/paramState.ts` is the part worth reasoning about on its own: what a
value set starts as, what "muted" holds, what a preset file is allowed to
contain. No React in it. The hooks own the state; this owns the arithmetic.

`controllers/` is one hook per concern. Order matters at the call site:
`useSchema` boots, `useValues` seeds itself from the schema the moment it
lands, and everything else reads from those two.

`useHistory` is the one hook nothing calls into: it *watches* the committed
value state and records what it sees, so undo covers a way of editing that did
not exist when it was written. Composed inside `useValues` rather than in
`App`, at the bottom of the file, so no mutator above it knows it is there.
See `docs/client-ui.md`.

`views/App.tsx` is composition and nothing else. Panels are in `panels/`, the
preview in `stage/`, reusable controls in `controls/`.

Two deliberate departures from the obvious layout:

* **`controls/ParamControl.tsx`, not `Slider` + `Dropdown`.** A discrete
  parameter renders as a menu and a continuous one as a slider, but the value is
  a number either way and nothing else in the app ever chooses between them. Two
  components would never vary independently. The *menu* it renders is shared —
  `controls/SelectMenu.tsx`, on `controls/Popover.tsx` — because six unrelated
  places needed the same list-of-names widget once the native `<select>` went
  (2026-08-09; see `docs/client-ui.md`). That is the opposite call from this
  bullet's and for the opposite reason: those six do vary independently, and the
  thing they share is a behaviour rather than a decision.
* **`stage/Stage.tsx` is one 485-line component.** The zoom transform, the pane
  measurement and the two compare modes share about ten refs and a wheel
  handler that has to remember an in-flight excursion. Splitting a `Canvas` out
  means threading those, which trades a real invariant for a smaller file.

## Tests

`verify.py` was split the same day, into `tests/checks/` — one module per area,
plus `runner.py` for selection and scheduling. It had been held back on the
argument that its sections share a scope; measured, that turned out to be six
names across four sections, all of them fixtures the sections were *supposed*
to agree on. They are `Ctx` properties now, which makes the sharing explicit
instead of accidental. `refs.py` (the slow reference implementations) stays as
it was — keep those references; deleting them turns the bit-exactness checks
into tautologies.

The full story, including why the wall-clock floor is now the longest single
module and not the total, is in [testing.md](testing.md).

Two monkeypatch sites move with the split: a constant has to be patched on the
module that *reads* it, not on `server.engine`, which only re-exports it. See
the dust and clustering checks.
