# Client / UI

## Two client traps worth knowing (2026-08-01)

* **`commit()` after `setValue()` applies the *previous* value.** `commit`
  reads `valuesRef`, and that ref is only refreshed during render, so calling
  the pair synchronously renders what was there before the change. Sliders
  never showed it because their `pointerup` arrives a render later and commits
  the right thing — but the `choices` menu has no second event, so picking an
  option did nothing until the control lost focus. `setValueNow` builds the
  next object and hands it to both setters. Any control that changes a value
  and expects a render in the same gesture needs it.
* **React registers wheel listeners as passive**, so `preventDefault` inside an
  `onWheel` prop is a no-op — the scroll-to-zoom handler is attached by hand
  with `{passive: false}` or the photo zooms and the page scrolls underneath it
  at the same time.

Compare and Wipe moved out of the panel onto the preview's `viewbar` alongside
the zoom controls, on the same reasoning that put zoom there: they change the
*view*, not the render, and driving a wipe across the photo from a panel on the
other side of the screen means looking away from the thing you are judging.

### Wheel zoom-out is no longer floored at Fit, and Fit itself keeps a margin (2026-08-04)

Two independent requests landed on the same code at once. **Fit reserves
`FIT_PADDING` (30px, screen pixels) on every side now**, mount or no mount --
before it sized the image to the exact pane, so the photo butted against the
panel edge with nothing to judge it against. It folds into the same `inset`
the mount's own room reservation uses, so a framed *and* fit image gets both
allowances at once rather than either one clobbering the other.

**The wheel used to bottom out at Fit and hand off to the - button for
anything smaller** -- deliberate at the time (see the `FIT_SNAP` doc comment's
history), but reported back as wrong: a continuous gesture should not need a
different control partway through it. `lo` is `ZOOM_STEPS[0]` now, matching the
button's own floor, and zooming in from there climbs back past Fit and on up
without a floor either.

**That surfaced a real bug in the Fit-snap band, not just a missing feature.**
`FIT_SNAP` decides how close to Fit counts as "close enough to lock to Fit
mode", and the check used to be one-sided (`next <= fit * (1 + snap)`) because
scrolling out could never go far enough below fit for the distinction to
matter -- the floor caught it first. Remove the floor and that one-sided check
means *every* zoomed-out value satisfies "at or below fit", so the wheel would
lock to Fit on the first tick past it and never come back out. Fixed by
checking both sides: `abs(next - fit) <= fit * FIT_SNAP`.

That fix alone was not enough, and finding out why is the more interesting
part. `zoom` displays as `null` (Fit) for the whole time a continuous scroll
sits inside the band, and the next wheel tick used to compute its step from
`eff` -- which, displaying Fit, reports exactly `fitZoom` regardless of *how
far* into the band the gesture actually was. A slow scroll advancing by less
than the band's own width per tick therefore recomputed from `fitZoom` every
single time, landed back inside the band every single time, and never
escaped: **the gesture was stuck exactly at Fit.** Measured with synthetic
wheel events sized to a plausible trackpad tick (2% zoom change per event
against a 2% `FIT_SNAP` band): every tick relocked, and 25 ticks in a row
moved the display 0.00 percentage points.

The fix is `wheelContRef`, a ref that remembers the true, unsnapped position
through a Fit-locked stretch, independent of what is on screen. Each tick reads
its starting point from that ref rather than from `eff` whenever the display is
currently `null`, so the *next* computation continues from where the gesture
really is rather than from the band's centre; whenever the ref holds nothing
(nothing in flight) or the display is a concrete number (no ambiguity to
begin with), `eff` is trusted directly. The ref is deliberately cleared at both
`setZoom(null)` call sites that are *not* this handler's own snap decision --
the Fit button and the new-image reset -- so a fresh "go to Fit" never
inherits a stale excursion left over from a previous scroll. Re-measured after
the fix: the same 2%-per-tick gesture takes exactly one tick to lock to Fit and
the very next tick to leave it, in both directions, and a moderate zoom-out
followed by zooming back in climbs straight through Fit and on past it rather
than sticking.

Verified end to end with a real running instance (headless Chrome driven over
CDP, synthetic wheel events dispatched on the actual `.pane` element) rather
than by inspection alone -- this is exactly the kind of interaction bug that
looks fine in a diff and only shows up when something actually scrolls.

### The mount is a view control, and its width is in *screen* pixels (2026-08-03)

`Frame` on the `viewbar`: an off-white board around the photo plus a drop
shadow, with a width slider that appears only when it is on. Requested, and it
earns its place on the same reasoning as the wipe — a photograph butted straight
against a dark panel reads darker and flatter than it is, and the edge of the
frame stops being visible at all where the picture goes to black. It is **not** a
parameter: nothing about it reaches the engine, the schema or an export.

Four things in it are not free choices:

* **Screen pixels, not source pixels.** Every spatial parameter the engine takes
  is a full-resolution length precisely so it means the same thing at any zoom;
  this is the opposite kind of quantity. It is furniture around the viewport, so
  it has to hold its apparent thickness rather than grow to fill the pane at
  800%.
* **Fit has to reserve room for it.** `.pane` is `overflow: hidden` and Fit puts
  the image exactly against the pane edge, so a mount drawn outside the image
  would be entirely invisible in the one view you would most want it in. The
  border width plus a shadow allowance comes out of `fitZoom` before the zoom is
  computed. `place()`'s clamping is left alone — it works in image coordinates,
  and the mount hangs outside the image box without moving it. The allowance is
  the blur radius **plus** the vertical offset, not whichever is larger: at a
  wide frame the mount nearly fills the pane and there is no background left to
  darken, so an allowance that is merely close reads fine at 18px and has no
  visible shadow at all by 96px. Caught by screenshotting the widths side by
  side, which is worth doing again to anything on this bar.
* **Its own element, drawn as two spread `box-shadow`s.** Not a border on the
  `<img>`: overlay mode draws the wipe by clipping the result image with
  `clipPath`, and **clipPath clips a box-shadow with it**, so a ring on that
  image would lose whichever side was wiped away. And a CSS border would grow
  the box past the `dw × dh` every coordinate here derives from, putting the
  pointer-anchored zoom half a border out. A spread shadow occupies no layout at
  all.
* **The drop shadow carries the same spread as the border.** Laid down from the
  image's edge instead, it sits *underneath* the opaque mount rather than around
  it, and the effect is half missing without ever looking broken.

The board is `#e8e6e0`, not `#fff`: a pure-white surround is brighter than any
highlight in the picture and drags the eye's white point with it, so highlights
read duller than they are — which is the one thing you cannot afford to misjudge
while dialling in halation. The element is also *filled*, not merely ringed,
because `place()` produces fractional `left`/`top` at most zooms and a ring
alone leaves a subpixel seam of dark background at the image edge.


## The app opens with every section muted (added 2026-08-04)

Requested: the photo should show untouched on boot -- and after Reset -- even
though the starting point is still `Stock` (or whatever `DEFAULT_PRESET`
resolves to), and picking a preset from the dropdown or loading a file should
be the one thing that switches every section on at once.

This is built entirely out of the mute mechanism that already existed for one
section at a time (`toggleGroup`): muting a group keeps its real values in
`muted[group]` while the *displayed* values for that group go to
`schema.neutral`, so un-muting restores exactly what was there. `muteAll(s,
src)` does that for every group in one pass, seeding each group's kept values
from `src` -- the starting preset's authored values, not whatever the sliders
currently show.

Boot and `resetAll` both now call `setValues(schema.neutral)` /
`setApplied(schema.neutral)` plus `setMuted(muteAll(schema, start.values))`,
instead of applying `start.values` directly. That is a small but deliberate
extension of the existing "boot and Reset share `startingValues` so they
cannot drift" rule: Reset means "how it opened", and now that opening means
*muted*, Reset has to produce the muted state too, or pressing it would
un-mute sections that boot left off -- reintroducing exactly the drift the
shared helper was written to prevent.

`applyPreset` and `loadPreset` both call `setMuted({})` after applying their
values -- picking a whole look, whether from the menu or from a file, is the
one thing that is not a partial "try this and see" action, so every section
goes live rather than staying staged behind its own toggle.

Two things that fall out of this rather than needing separate handling: the
`Original` button's `disabled={isOriginal}` correctly reads *true* right after
boot, because `values === schema.neutral` at that point -- the app opens
already agreeing with its own "show the untouched photo" state, not merely
looking like it does. And `master_opacity` (excluded from `NEUTRAL_ZERO`
because 1.0, not 0, is its neutral) is unaffected by any of this: muting the
Output group still means "no dial-back", exactly as it does when toggled by
hand.

Verified against a real running instance rather than by inspection: every
section reads muted on a fresh load, picking a preset flips all twelve to
enabled in one render, and loading a preset file from disk does the same.


## Undo/redo, and four smaller panel moves (added 2026-08-08)

### Undo/redo is *observed*, not called (`useHistory`)

Requested with two buttons on the preview's overlay bar and no shortcut. The
part worth reading for is that **nothing calls into the history**. `useValues`
has ten ways to change a value -- slider, number field, menu, preset, file,
mute toggle, per-section reset, seed reroll, LUT pick, master reset -- and a
history each of them had to remember to `push()` to would be one new mutator
away from a hole in it permanently. `useHistory` instead watches the committed
state and records the transitions it sees, so an edit added tomorrow is
undoable the day it is written.

Three things that are not free choices:

* **It watches `applied`, not `values`.** `applied` only moves when a gesture is
  finished, so a slider drag is one step rather than one per pixel of travel --
  for free, out of a distinction the renderer already needed. The known
  coarseness at the other end: a typed number and an arrow-key nudge each commit
  per keystroke, so each is its own step. That is the granularity the renderer
  sees too, and collapsing it would mean guessing where one edit ends.
* **`muted` is recorded but does not *trigger* a step.** Mid-drag, touching a
  control in a muted section un-mutes it a render *before* the release commits.
  Treating that as its own transition files a state that was never on screen
  (section live, values not yet applied), and undo lands you on it. It is read
  at push time instead, so the step that does get filed carries the right one.
* **A snapshot is all four of `values`, `muted`, `referenceMp`, `lut`.**
  Restoring three of the four leaves a state the user never had -- undoing a
  preset back to Stock with the previous look's LUT still riding under it.

Equality is by object *identity*, not deep comparison: every path in
`useValues` builds a fresh object for a real change, so a shared reference means
nothing happened, and it makes the check cheap enough to run on every commit.

The stacks are refs with a counter to force re-renders, not `useState` arrays.
`undo` has to read and write them and then call `restore` in one go, and a state
updater doing that runs its side effect twice under StrictMode.

**Opening a photo clears the history**, on request -- a step describing a render
that no longer exists is not somewhere you can go back to. It is keyed on
`meta.id` and handled *inside the same effect that records steps*, rather than
as a `reset()` the uploader calls, and that is the whole subtlety: opening a
photo swaps the image and rerolls the seeds in one commit, so a `reset()` called
from the handler runs *before* the effect that records the reroll and leaves it
behind as the one undoable step on a freshly opened image. Checking the key
first inside the effect makes the post-reroll state the new baseline instead.

### The rest of the batch

* **`Random Seed` in the top bar**, beside the filename. The same reroll the
  `With random seed` checkbox does on open, on demand: a seed is not a quantity
  you dial in, it is one you keep drawing until the grain falls somewhere you
  like. Shown only with a photo loaded, for the same reason the filename is.
* **Preview background lightness**, in the panel under Preview fidelity. The
  chequerboard's two tones are now `--board-a`/`--board-b` custom properties fed
  from `boardTones()`; the gradient itself stays in the stylesheet. Lightness
  only -- hue and saturation stay put, because a coloured surround pulls the
  photo's apparent white balance with it, which is the same misjudgement the
  mount is off-white to avoid. It exists because the eye sets its black point
  from the whole field of view, so the same photo reads contrastier against
  near-black than against grey, and one fixed backdrop is one fixed opinion.
* **The export controls moved to a `cornerbar`**, overlaid bottom right, on the
  reasoning that moved Compare and Wipe to the top-right one: exporting is the
  last thing you do to a picture you are looking at, and it was below a scroll
  in the panel. The three paragraphs that ran under the scale menu are behind a
  help badge now -- reasonable to scroll past in a side panel, unreasonable laid
  over the picture, worth keeping either way. `Stage` takes the overlay as an
  opaque `corner: ReactNode` so it does not have to know what an export job is.
* **Collapse/expand all pipeline sections**, a single button in that same corner
  overlay. Which way it goes is never in doubt because it reads the panel in
  front of you: "Collapse all" while anything is open, "Expand all" once
  everything is shut.
