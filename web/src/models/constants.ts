/** Tuning constants for the client. Every one of these is a *view* quantity
 *  -- screen pixels, gesture rates, debounce -- deliberately unlike the
 *  engine's parameters, which are lengths in full-resolution pixels and live
 *  in `params.py`. See docs/client-ui.md. */

/** Renders are not started from the raw value stream -- see `applied` below --
 *  but pans, zooms and typed numbers still arrive in bursts, so requests wait
 *  this long to settle. Short, because a stale preview is worse than a late
 *  one. */
export const DEBOUNCE_MS = 140;

/** Zoom stops for the +/- buttons. 1 is a real 1:1 pixel view.
 *
 *  These used to be constrained to clean fractions because the server cropped
 *  and resampled at the requested zoom, and an awkward factor put the read
 *  origin on a half pixel. Zooming is a pure browser transform now, so the
 *  list is free -- it is only about how the steps feel. The wheel does not
 *  step through them; it only borrows the two ends as its limits. */
export const ZOOM_STEPS = [0.05, 0.1, 0.17, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8];

/** Wheel zoom rate, as an exponent on the scroll delta. A mouse notch is 100
 *  units in most browsers, so 0.0025 is about 2.8 notches per doubling --
 *  fast enough to cross the range without hunting, slow enough to land on a
 *  value. A trackpad pinch arrives as ctrl+wheel with a far smaller delta,
 *  hence its own rate. */
export const WHEEL_RATE = 0.0025;
export const PINCH_RATE = 0.01;

/** How close to Fit a wheel step has to land before it locks to Fit mode.
 *
 *  Fit is a *mode*, not a number: it follows the container, so a window resize
 *  keeps the whole frame visible. Landing on 0.1997 when fit is 0.2 would look
 *  identical and quietly lose that, so anything within this fraction of Fit
 *  becomes Fit outright rather than a very-close zoom value -- a continuous
 *  control almost never *lands* on a snap point, it crosses it, so a snap that
 *  only fires on a near-miss is a snap that fires at random.
 *
 *  The band is checked on **both** sides of Fit (changed 2026-08-04, on
 *  request): the wheel used to bottom out at Fit and hand off to the - button
 *  for anything smaller, which needed only a one-sided `next <= fit` check.
 *  Scrolling out is not capped there any more -- see `ZOOM_STEPS[0]` below --
 *  so a one-sided check would now catch *every* zoomed-out value, not just the
 *  ones near Fit, and the wheel would never leave Fit mode once it reached it. */
export const FIT_SNAP = 0.02;

/** Mount border around the previewed photo, in *screen* pixels.
 *
 *  Screen pixels rather than source pixels on purpose. Every spatial parameter
 *  the engine takes is a length in full-resolution pixels precisely so it means
 *  the same thing at any zoom -- this is the opposite kind of quantity. It is
 *  furniture around the viewport, not part of the picture, so it must hold its
 *  apparent thickness as you zoom instead of growing to fill the pane at 800%.
 *
 *  The shadow allowance is added to the border when reserving room for Fit.
 *  Without it the mount lands exactly on the pane's edge and `overflow: hidden`
 *  eats the shadow, which is the half of the effect that separates the photo
 *  from the background. It has to cover the blur radius plus the vertical
 *  offset, not just one of them -- at a wide frame the mount nearly fills the
 *  pane and there is no background left to darken, so an allowance that is
 *  merely close leaves the shadow visible at 18px and gone at 96px. */
export const FRAME_MAX = 96;
export const FRAME_DEFAULT = 30;
export const FRAME_SHADOW_BLUR = 24;
export const FRAME_SHADOW_DROP = 8;
export const FRAME_SHADOW_ROOM = FRAME_SHADOW_BLUR + FRAME_SHADOW_DROP;

/** Breathing room Fit leaves on every side, in screen pixels, whether or not
 *  the mount is on. Fit used to size the image to the exact pane, so it butted
 *  straight against the panel edge with no margin to judge it against -- this
 *  reserves the same kind of room the mount does, just always on rather than
 *  only with Frame enabled. */
export const FIT_PADDING = 30;

/** The chequerboard behind the previewed photo.
 *
 *  Adjustable because the surround is not neutral ground: the eye sets its
 *  black and white points from the whole field of view, so the same photo reads
 *  contrastier against near-black and flatter against grey, and judging
 *  shadow density or halation against one fixed backdrop is judging it against
 *  an opinion. The default is where it has always been -- the two tones the
 *  chequer was hard-coded to.
 *
 *  Lightness only. Hue and saturation stay put: a coloured surround pulls the
 *  photo's white balance with it, which is exactly the misjudgement the mount
 *  is off-white to avoid. The two squares stay a fixed number of lightness
 *  points apart so the pattern is legible at every setting rather than
 *  vanishing at the ends. */
export const BOARD_HUE = 228;
export const BOARD_SAT = 10;
export const BOARD_LIGHT_DEFAULT = 9;
export const BOARD_LIGHT_MAX = 100;
export const BOARD_CHECK_DELTA = 2;

/** The two chequer tones for a lightness, as CSS custom properties. Clamped at
 *  the dark end so the darker square never wraps past black and inverts the
 *  pattern. Returns a plain string map rather than a `CSSProperties` so this
 *  file stays free of React types, like the rest of `models/`. */
export function boardTones(light: number): Record<string, string> {
  const a = Math.min(BOARD_LIGHT_MAX, Math.max(0, light));
  const b = Math.max(0, a - BOARD_CHECK_DELTA);
  return {
    "--board-a": `hsl(${BOARD_HUE} ${BOARD_SAT}% ${a}%)`,
    "--board-b": `hsl(${BOARD_HUE} ${BOARD_SAT}% ${b}%)`,
  };
}

/** Bounds for a hand-set size-scaling factor.
 *
 *  A *linear* factor, like the automatic one it replaces -- see
 *  `docs/presets.md` for why the megapixel ratio is the wrong number. The range
 *  is deliberately wider than any real preset-to-photo ratio: overriding it by
 *  hand is what you do when the automatic answer is right about the arithmetic
 *  and wrong about the picture, and clamping it to plausible values would be
 *  clamping exactly the case it exists for. */
export const SCALE_MANUAL_MIN = 0.1;
export const SCALE_MANUAL_MAX = 4;
export const SCALE_MANUAL_STEP = 0.01;

/** Fallback bounds for the proxy-size slider, used before a photograph is open
 *  and as the starting edge for a preset that names none.
 *
 *  The real bounds come from `/api/upload` so they cannot drift out of step
 *  with `_clamp_edge` -- these exist because the control is rendered, and the
 *  value is held, before any meta exists. `def` mirrors `PROXY_LONG_EDGE`. */
export const PROXY_EDGE_FALLBACK = { min: 100, max: 4800, step: 100, def: 2400 };

/** Marker written into saved preset files. Only used to make a hand-inspected
 *  file self-describing -- loading deliberately does not require it, so a bare
 *  `{"intensity": 40}` typed by hand still works. */
export const PRESET_FORMAT = "film-grain-preset";

/** Section the LUT picker is rendered into, immediately above its own Mix
 *  slider.
 *
 *  The panel is generated from the schema and hand-adding a *slider* here would
 *  be a bug — the schema is the single source of truth for parameters. A LUT is
 *  not a parameter: it is a named resource, like a preset file, so it cannot be
 *  a number in the schema and there is nothing for the generator to pick up. It
 *  is placed by key rather than at the top or bottom of the section so the panel
 *  reads in pipeline order: the four adjustments, then the LUT they feed. */
export const LUT_ANCHOR_KEY = "lut_amount";

/** DOM id for a pipeline section, so the jump menu can find it in the panel.
 *
 *  Derived from the group name rather than an index: the id has to survive
 *  `GROUPS` being reordered, which it has been more than once, and a name is
 *  what the menu item carries anyway. Non-word characters collapse to `-`
 *  because group names have spaces in them ("Global Grain", "Tone Response"). */
export function sectionDomId(group: string): string {
  return `sec-${group.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
}
