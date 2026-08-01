# Organic Film Grain Simulation Engine

## Project Overview
An organic film grain generator that avoids the synthetic "digital overlay" look of
traditional noise sliders. It targets physical emulsion behaviour: structural grain,
edge destruction, non-uniform luminance distribution, and the optical artefacts
(halation, dye-layer fringing) that read as "film" to the eye.

Delivered as a web app: a browser UI over a Python/PyTorch image service.

---

## Scope & Priorities

**Primary focus — the point of the project:**
1. **Detail destruction** — eroding image micro-structure rather than overlaying noise
2. **Edge softening and edge noising** — sub-pixel jitter, micro-blur, ragged erosion
3. **Grain** — structural, clumped, luminance- and edge-weighted
4. **Halation** — warm bloom around highlights
5. **Chromatic edge fringing** — per-dye-layer erosion producing coloured speckle on edges

**Deferred to a later project:** colour grading in general — tone curves, contrast,
toe/shoulder, split toning, highlight desaturation, base fog. The engine implements
these and exposes them as parameters, but they **ship neutral (0) by default** so the
pipeline is a colour pass-through and grain work can be judged without interference.

---

## Core Domain Truths & Architectural Rules

### 1. Physical Film Chemistry Principles
* **Grain is Structural, Not Additive:** Grain IS the image micro-structure. Do not
  stamp a 2D uniform noise grid over sharp digital edges. Implemented as an additive
  term *plus* a term that multiplies the image's own high-pass detail, so grain erodes
  existing structure instead of sitting on top of it.
* **Luminance Distribution:** Grain peaks in **midtones and shadows (15%–65% luminance)**
  due to silver halide clumping, and drops sharply in dense highlights and deep blacks.
  *Verified in the implementation:* measured grain sigma peaks across 15–65% and falls
  to ~16% of peak in the 95–100% band.
* **The "Edge-Destruction" Rule:** Noise defines and slightly erodes high-contrast
  micro-edges rather than invading flat, smooth areas (skies, flat backdrops).
  This rule needs *two* mechanisms, not one. An edge mask only sees micro-edges,
  so a smooth gradient — skin, clear sky, a studio backdrop — gets no protection
  from it and takes the full flat-area floor. That is what makes skin read as
  jagged. The second mechanism measures local contrast over a medium radius and
  suppresses grain where the region is genuinely featureless. Note that a
  *luminance* range cannot do this job: skin sits at 30–60% luma, exactly where
  grain is supposed to peak.
* **Shadow clumping:** Crystals in shadow are larger and less densely packed, so clump
  *size* varies with luminance, not only amplitude.
* **Halation:** Light passing through the emulsion reflects off the film base and
  re-exposes from behind, blooming warm around highlights. Must be computed in **linear
  light** — doing it in gamma-encoded space is the usual reason simulated halation looks
  like a painted-on glow.
* **Adjacency (Eberhard) effect:** Developer exhausts differently on either side of an
  edge, leaving a local contrast boost. It is why film reads as sharp despite resolving
  less detail than a sensor.

### 2. Engineering & Technical Pipeline Constraints
* **Supersampled rendering.** Grain is a sub-pixel phenomenon. Rendering it directly on
  the output grid gives each clump a hard, aliased pixel footprint — precisely the
  synthetic look this project exists to avoid. Render at 2–3× and integrate down.
  This is the single biggest realism win in the pipeline.
* **Patch-Based Processing:** Full-resolution photos (24MP–45MP+) are tiled
  (1024×1024 with computed overlap) for export.
* **Tile independence is a hard rule.** No stage may depend on a statistic of the region
  being rendered — no per-tile normalisation, no global mean. Edge strength normalises
  against a fixed constant; the noise lattice is addressed by absolute global
  coordinates. Verified: tiled and single-pass renders agree to ~1e-4, and tile
  boundaries in a real export show *below-median* column difference.
* **Scale invariance is a hard rule.** All spatial quantities are specified in
  full-resolution pixels and multiplied by the working scale. The proxy preview and the
  full export therefore show the same grain *structure*.
* **Micro-Blur / Acutance Simulation:** A small micro-blur (0.3–0.6px) on the base
  simulates light diffusion through the 3D gel layers and destroys hyper-sharp pixel
  borders. Applied in linear light, before grain, so grain stays sharp against a
  softened base.
* **Quality over speed.** Latency is an acceptable cost. Do not clamp octaves, lattice
  density, blur radii, or preview resolution for performance alone.

### 3. Corrections to earlier drafts of this document
* The original draft prescribed a MSE-vs-GAN/LPIPS/Gram training recipe. That guidance
  is sound *for the neural approach*, but it is not applicable to the shipped procedural
  engine and no model is trained here.
* The original prescribed "PyTorch / CUDA / OpenCV" as the code style. The engine is
  PyTorch and device-agnostic (CUDA / Apple MPS / CPU via one device flag). OpenCV is
  not used — every operation is a tensor op.
* A "shoulder" that is normalised to still reach 1.0 is not a shoulder. A region of
  falling slope mathematically cannot reach the top; forcing it turns the shoulder into
  a highlight *boost*. Letting it asymptote below white is what gives film its creamy
  highlights.

---

## Implemented Pipeline

Order matters and follows the physics: things that happen to light, then things that
happen to the emulsion.

**Exposure stage (linear light)**
1. sRGB → linear
2. Micro-blur — lateral diffusion through the gel layers
3. Halation — threshold highlights, wide blur, warm tint, added back

**Development stage (density / display space)**
4. Characteristic curve — toe, straight line, shoulder *(deferred group, neutral by default)*
5. Highlight desaturation, split toning, base fog *(deferred group, neutral by default)*
6. Edge isolation — high-pass at 1.5–3.0px, normalised against a fixed reference,
   then broken up by its own noise field so the erosion envelope is ragged rather than
   tracing the edge like a digital outline
7. Sub-pixel edge jitter — up to 0.6px displacement, weighted by the edge mask
8. Luminance mask — full strength across the 15–65% band, easing out over
   independently-controlled falloff widths on each side. Quintic easing on a
   spatially blurred luma, so the transition is smooth both across the tone
   scale and across the frame
8b. Smooth-area guard — medium-radius local contrast suppresses grain in
   featureless regions
9. Grain field — multi-octave value noise on a globally-addressed lattice, dual-scale
   for shadow clumping, clump-hardness curve, mono/chroma dye-layer blend
10. Composite — additive term weighted by luminance × edge mask
11. Structural erosion — grain multiplies the image's own micro-detail; `edge_chroma`
    blends between neutral erosion and full per-dye-layer colour fringing
12. Adjacency effect — local contrast boost extracted from the pre-grain base

---

## Architecture

* **Backend:** Python 3.13 / FastAPI / PyTorch. Device-agnostic: CUDA, Apple MPS, or CPU.
* **Frontend:** React + TypeScript + Vite. The slider panel is generated from the
  server's parameter schema (`GET /api/params`), so the UI cannot drift out of sync
  with what the renderer accepts.
* **Deployment:** single service — FastAPI serves the built client.
* **Preview:** proxy-resolution for the whole frame, plus a true 1:1 crop view rendered
  with the same global coordinates as the export, so 100% view is an exact preview of
  the output rather than an approximation.
* **Export:** background job with progress, tiled, 16-bit PNG by default. 8-bit
  quantisation visibly posterises grain in smooth areas.
* **Focus:** still photography (RAW/TIFF workflows) rather than video frames.

### Verification
`tests/verify.py` asserts the invariants above: tile independence (tiled vs
single-pass agree to ~1e-4), crop fidelity (the 1:1 view is bit-identical to the
same region of the export), colour pass-through with the deferred grading group
neutral, the 15–65% luminance response with highlight suppression, edge bias,
and that the 16-bit PNG export is genuinely 16-bit.

### Not yet implemented
* **RAW input.** Requires `rawpy`/LibRaw; currently JPEG/PNG/TIFF only.
* **Approach B (neural).** A trained UNet/cGAN would slot in behind the same endpoint,
  but it needs the paired film-scan/digital dataset that does not exist yet. The
  procedural engine is the only path validatable without that data.
