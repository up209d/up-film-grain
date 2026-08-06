# Halation

## Halation blue compensation, and why it runs *before* the wash (2026-08-01)

Halation adds warm light in linear light, and **adding light desaturates
whatever it lands on** — that is not a bug to tune out, it is what addition
does. A red-tinted bloom lifts a blue sky's red channel by the full glow and
its blue channel by a tenth of it, so the sky loses colour toward grey and
drifts toward purple. Reported by the user, and real: measured on an ordinary
sky, **16% of the saturation gone and a +5.8° hue swing**.

There are two regimes, and they want different answers:

* **Threshold above the sky's luma** (the default 0.72). The wash is a *local*
  rim: measured, sat 0.769 → 0.574 at 10px outside a highlight, 0.732 at 50px,
  and untouched past 90px. A uniform compensation over-corrects the 90% of the
  frame that was never damaged.
* **Threshold below the sky's luma** (`Stock`-era presets, and `Organic` at
  0.30 against a sky luma of 0.366). The sky is over the threshold, so **it
  blooms onto itself** and the loss is uniform across the frame. This is the
  case the control is for, and the one a shipped preset actually hits.

Worth saying out loud: in the second regime **raising `halation_threshold`
above the sky's luma fixes it outright** — 0.30 → 0.45 on `Organic` measures
sat 0.660 → 0.778 and hue 225.3 → 219.7, against an untouched 0.769/220.0.
Reach for the compensation when the low threshold is wanted for the look.

**Before the wash, not after** — I measured both with the identical mask:

| amount | before the wash | after the wash |
|---|---|---|
| 0.5 | +0.6% past target | +8.8% |
| 1.0 | +3.5% | **+30%, a channel driven to black** |
| 3.0 | +3.5% | pinned at fully saturated |

Compensating *before* is self-limiting because the wash eats the same share of
whatever is added, so the control has a natural brake and cannot be over-cooked.
Applied *after* there is no brake at all, and by amount 1.0 it has crushed the
minimum channel to zero — posterisation and hue break, not a correction.

Two more reasons, both structural:

* After the wash there is no way to tell blue that was *unfairly* greyed from
  blue the bloom is **supposed** to be sitting on. Re-saturating there fights
  the glow you paid for. It would need the glow field carried out of the
  halation block to know the difference. Before the wash the question never
  arises: this changes what was *recorded*, and halation then does its job to
  it — which is also the physical order, a punchier blue layer or a polariser
  rather than retouching.
* It is the cheaper place. Purely per-pixel, no kernel, so `pad_for` is
  unchanged (pinned at 58px in `verify.py`).

Two things that had to be right:

* **The glow is computed before the compensation runs**, so dialling blue
  cannot move the bloom. Pinned bit-exactly: probed on a grey field lit by a
  saturated blue source, the glow on the surrounding ring moves **0.00e+00**
  while the blue source itself moves 0.432. Without that ordering,
  `halation_blue` and `halation_threshold` would fight each other.
* **Saturation cannot fix hue.** Scaling chroma about the luma axis is a
  radial operation and by construction rotates nothing — measured, +7.1° of
  error left on the table by the amount slider alone, and `halation_blue_shift`
  at −8° takes it to +0.1°. The second slider is not decoration.

### The gate is brightness, not hue width (corrected 2026-08-01)

The first version exposed a **Blue Range** hue-width slider. That was the wrong
control, and the user found it by using it: cranking Blue Compensation made
deep blue go lurid. The mask knew *what colour* a pixel was and nothing about
whether the wash had ever reached it, so every bit of correction on an
undamaged pixel was pure overshoot.

Measured up a sky gradient away from the sun, saturation loss runs **23% at the
bright end and flat 0% below about half brightness** — halation only reaches
what is near the light. And at amount 2.0 the ungated mask took an untouched
deep sky from 0.872 saturation to **1.000**: a channel clamped to black. That is
a missing term in the mask, not a setting to avoid.

So the hue width is a constant (`_BLUE_RANGE`, 70°) and the slider is
`halation_blue_level` — how light a blue has to be before it is worth saving —
with `halation_blue_falloff` as a **separate** width, because deriving the ramp
from the knee would make moving one change the other and a sky is exactly the
broad gradient that shows up a hard switch-on. Same pattern as
`lum_low`/`shadow_falloff`, and quintic for the same reason.

**Read the brightness display-referred, and encode before taking the luma.**
`_linear_to_srgb(_luma(lin))` is cheaper and wrong — the transfer curve is
non-linear so it does not commute with a weighted sum, and it reads a deep sky
**23% brighter than it is**, putting this slider on a different scale from the
Shadow/Highlight Knees it is meant to match (under Grain Structure since
the Luminance Response section was merged into it). Linear luma is worse still: it
crushes an ordinary sky to 0.05 and wastes the top nine tenths of the slider.

Known limit, measured: a fixed brightness gate is a *proxy* for "where the wash
reached", and in the high-threshold regime the two do not line up exactly —
0.487 display luma carries 1.8% damage and 0.519 carries 23.4%, only 0.03
apart. The exact answer is to weight the mask by the **glow field itself**,
which is already computed two lines above, is tile-independent (fixed
threshold, no statistics) and would make over-correction structurally
impossible in both regimes. Not done: it would silently change what every
existing `halation_blue` value means, and the brightness gate is what the user
asked for. One multiply if it is ever wanted.

`_BLUE_HUE` is **230°, measured in linear light** where the stage runs, not the
sRGB number. The transfer curve is per-channel and monotonic so it preserves
the hue *sector* but moves the angle inside it by 6–10°: an ordinary sky is
220° in sRGB and 230° in linear. Skies span 222° (pale) to 236° (zenith);
cyan water is 194° and purple shadow 249°, so the fixed 70° window separates
them. The mask is weighted by existing saturation on `vibrance`'s reasoning —
it must strengthen blue that is *there* and never invent it in grey, or every
neutral in the frame picks up a cast. Grey and red are left **bit-exact** at
maximum settings.

Gated on `halation > 0.01`. With no wash there is nothing to compensate and the
control would just be a blue grade, which is deferred — `verify.py` pins it at
0.00e+00.

## Halation highlight recovery: metering the bloom against real headroom (added 2026-08-05)

Reported: halation "burns" highlights a lot. Real, and structural rather than a
tuning problem — the glow is added in *linear* light with no upper clamp until
display space (`_linear_to_srgb` at engine.py only does `clamp_min(0.0)`, no
ceiling), and the first actual `.clamp(0, 1)` comes well after, in display
space, after brightness, the characteristic curve, highlight desaturation,
vibrance and the warm/cool cast have all had a chance to run on the
still-unclamped value. So a highlight already close to white gets pushed the
rest of the way to a flat, textureless clip rather than rolling off — the
bloom does not know the receiving pixel had almost no headroom left.

### The first version held the glow back, and that is not a recovery (rewritten 2026-08-05)

Worth reading before touching this, because the first answer was the obvious
one and it was wrong in a way that is easy to re-introduce. It was
`glow = glow * (1.0 - recover * hi)`: attenuate the bloom in proportion to `hi`,
the field the glow's own shape is built from. That buys headroom **by deleting
the effect** — the highlights stop burning because the halation stopped
happening there — and it cannot restore anything, because two pixels that both
clipped are still both at 1.0 afterwards. The user's objection to the tone
controls one section down applies word for word: dimming something already flat
is not recovery.

It also keyed on the wrong field. `hi` answers "is this pixel bright enough to
bloom", which is not the same question as "how much more light can this pixel
take": a saturated highlight can sit far above `halation_threshold` in luma
while one of its channels still has most of its range free, and only that
channel's own headroom knows so.

What it does now is **meter the added light against the room each channel
actually has**, with `H = 1 - lin` per channel in linear light:

```python
recover = p["halation_recovery"]
if recover > 0.001:
    head = (1.0 - lin).clamp_min(1e-4)
    add = add * (head + add * (1.0 - recover)) / (head + add)
lin = lin + add
```

Three properties, and each is why it is this expression:

* **Free where there is room.** For `add << H` it is `add` to first order, so an
  ordinary highlight with headroom to spare gets the whole bloom at full
  strength and the control costs nothing there. Only a pixel being asked to take
  more light than it can hold is metered at all.
* **Cannot reach white at `recover = 1`**, since `add' < H` strictly.
* **Strictly increasing in `lin`**: the derivative is `1 - r·a²/(H+a)²`, bounded
  below by `1 - r` and positive throughout. Nothing flattens, so nothing is
  lost.

**An exponential soft-add was built and measured first, and is worse** — worth
recording because it looks better on paper. `lin + H(1 - exp(-add/H))` is the
textbook tone-mapping answer, asymptotes at white, and is monotone; but it bends
from the origin, so it compresses hard even where the bloom was modest. On a
bright plate carrying real fine texture it held **51%** of that texture against
the hyperbolic form's **60%**, at less bloom retained. Bending *late* beats
bending smoothly when what you are protecting is local contrast rather than the
peak value.

Measured on that plate (mean 0.93, fine texture, `halation 0.9` at threshold
0.6) — highlight texture kept / bloom light kept:

| recovery | hold the glow back (old) | meter against headroom (now) |
|---|---|---|
| 0.5 | 56% / 82% | 53% / 91% |
| 1.0 | 55% / 52% | **60% / 68%** |

At full strength it is better on **both** axes at once, which is the whole
claim: more of the highlight's detail survives *and* more of the bloom does.
`verify.py` pins both halves together, because a texture-only test would pass
happily on a stage that simply turned the effect off.

The closed form is pinned as an **equality** rather than as "less clipping":
solved from the recovery-off render's own added light so the check does not
reimplement the bloom, measured 1.19e-07 against the analytic value over three
settings × three channels. And `recovery 0` is still bit-exactly the old
behaviour (0.00e+00).

**What is left on the table, measured.** The remaining loss at full recovery is
compression, not clipping, and it is *forced*: red on that plate is asked to
absorb 0.63 of linear light into 0.15 of headroom, so no metering curve can be
free — the frontier at 100% texture retention is about 14% bloom. The way past
it is not a better curve but a better **model**: real halation is light that
*left* the highlight to re-expose its surroundings, so an energy-conserving
bloom would darken the core as it lights the halo, and the core's texture would
survive intact. That is a change to what halation *is* rather than to this dial,
and it would move every preset that uses the stage, so it is not done.

No new geometry, so nothing new to reserve: still purely per-pixel, same
reasoning as blue compensation just above it (`pad_for` pinned unchanged), and
`halation_recovery` is in `NEUTRAL_ZERO` alongside `halation`/`halation_blue`
for the same reason both of those are — it is a modifier of a stage that is
itself gated off at 0, so "Original" has to be able to zero it along with the
rest.

**`presets/Vintage.json` is the only shipped preset that changes**, because it is
the only one that sets `halation_recovery` (0.75). Measured on the repo's own
test frame: mean 0.2701 → 0.2716, mean pixel delta 0.0018, worst pixel 0.0513 —
it keeps more of the bloom's light than it used to at that setting. Every other
shipped preset renders **bit-identically** (0.00e+00 across `Stock`, `Dramatic`,
`Subtle`, `ExtraGrain`, `Dreamy`), because the two tone controls and
reconstruction all ship at 0 and no preset sets them. `Vintage` was not migrated:
the request was to fix the mechanism, and re-tuning the preset to hide the change
would defeat it.

One trap worth recording for anyone measuring a preset this way: **the preset
files nest their parameters under a `values` key.** Handing the whole file to
`sanitize()` silently fills every parameter from defaults and reports a confident
0.00e+00 for every preset — which is exactly what the first version of that
measurement did.

