from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ----------------------------------------------------------------- edge
    Param(
        "edge_soften", "Edge Softening", "Edge Destruction",
        0.0, 1.0, 0.01, 0.0, "",
        "Takes the digital snap off hard transitions without touching flat "
        "areas or fine texture. This is the one to reach for when the image "
        "wants to be softer -- Micro-Blur diffuses the whole frame, texture "
        "and all, which reads as out of focus rather than as film.\n"
        "\n"
        "It runs after the grain, so it does soften the grain it passes over "
        "-- about 7% of it at full strength, against Micro-Blur's 71%. Which "
        "edges it is allowed to touch is Softening Selectivity; by default only "
        "hard borders qualify. 0 = off.",
    ),
    Param(
        "edge_soften_radius", "Softening Radius", "Edge Destruction",
        # Topped at 64, not 8 (2026-08-09). Reported as softening ignoring the
        # boundary between skin and a light background, which is a *big* edge --
        # 0.26 of luma step, far above every threshold -- and was being missed
        # because it is a **soft** one. This radius is both the gate's
        # measurement scale and the blur's reach, so an edge that ramps over
        # more pixels than the radius barely registers. Measured on a skin
        # boundary, mean 8-bit change over the transition:
        #
        #   ramp   sr=2   sr=8   sr=16   sr=32   sr=64
        #   12px   0.04   2.26    8.79   14.69   18.15
        #   30px   0.01   0.01    0.96    6.53   13.07
        #   60px   0.00   0.00    0.01    0.96    6.53
        #  100px   0.00   0.00    0.00    0.04    1.94
        #
        # The old 8px ceiling covered a 12px ramp and nothing softer, while a
        # portrait boundary at 24MP ramps 30-100px. Cheap to raise: this term
        # enters `pad_for` once rather than through the high-pass's 3.3x, so 64
        # costs 1.82x overdraw at 24MP against 1.24x at the default.
        0.3, 64.0, 0.05, 1.5, "px",
        "How far a softened edge spreads, at full resolution -- **and how wide "
        "a transition has to be before this stage can see it at all.** The two "
        "are the same number because they are the same measurement: an edge "
        "that ramps over more pixels than the radius reads as flat ground.\n"
        "\n"
        "That is the setting to reach for when softening ignores an edge you "
        "can plainly see. A hard border needs 1-3px; a portrait's skin-against-"
        "background boundary is soft and often needs 16-64. Measured on a skin "
        "edge ramped over 30px, the effect is 0.01 levels at radius 8 and 6.5 "
        "at 32.",
        spatial=True,
    ),
    Param(
        "edge_soften_edges_only", "Softening Selectivity", "Edge Destruction",
        0.0, 1.0, 0.01, 1.0, "",
        "Which edges softening is allowed to touch, by how hard they step. At "
        "1 -- the default, and what this stage has always done -- only real "
        "borders are softened and fine texture is left alone: fabric, hair and "
        "foliage measure an order of magnitude below a hard edge, which is the "
        "gap the gate keys on. Turning it down opens the gate to progressively "
        "gentler edges, and at 0 there is no gate at all and the softening runs "
        "everywhere, texture included.\n"
        "\n"
        "Reach for it when softening appears to be doing nothing: a low-contrast "
        "subject can sit entirely below the gate. Measured on clean steps at the "
        "default, a luma step of 0.02 or 0.05 is untouched, 0.10 barely moves, "
        "and only 0.20 and above soften properly -- so a soft-lit or hazy frame "
        "can look immune to the control until this comes down.",
    ),
    Param(
        "edge_jitter", "Edge Jitter", "Edge Destruction",
        0.0, 5.0, 0.01, 0.3, "",
        "Warps edges along a noise field so a border wanders instead of "
        "running dead straight, which is most of what stops a rendered edge "
        "reading as vector art. Displacement is in full-resolution pixels and "
        "peaks at 3px; the default 0.3 makes a straight border wander about "
        "±0.4px. Flat areas are untouched — it is weighted by the edge mask.",
        spatial=True,
    ),
    Param(
        "jitter_aniso", "Jitter Direction", "Edge Destruction",
        0.0, 1.0, 0.01, 0.0, "",
        "Concentrates Edge Jitter onto one axis instead of displacing edges "
        "every way at once. 0 = isotropic, the default, and the angle below "
        "then does nothing -- rotating a field that is already the same in "
        "every direction changes nothing. 1 = edges only ever move parallel "
        "to that angle, which reads as a directional slip rather than a "
        "wobble.",
    ),
    Param(
        "jitter_angle", "Jitter Angle", "Edge Destruction",
        0.0, 180.0, 1.0, 0.0, "deg",
        "Axis the jitter is biased along, once Jitter Direction is above 0. "
        "0 = horizontal, 90 = vertical, 45 = diagonal. Only 0-180 is needed: "
        "the displacement is symmetric, so 200 degrees is 20 degrees.",
    ),
    Param(
        "edge_sand", "Edge Sanding", "Edge Destruction",
        # Topped at 1, not 5 (2026-08-08). It is a cross-fade toward the sanded
        # frame, so 1 *is* the whole effect and the 1..5 travel was never more
        # of it -- unclamped it extrapolated past the filtered result and put
        # jaggedness back on: measured, strength 1 removed 46% of a jittered
        # border's roughness and strength 5 added 235% of it. Now clamped, so
        # above 1 was simply inert; the presets carrying 1.65 and 2.0 render
        # bit-identically to 1.0 (0.00e+00), which is why re-authoring them cost
        # nothing. Reach is `edge_sand_grit`, and that is the strength lever.
        0.0, 1.0, 0.01, 0.0, "",
        "Polishes the jaggedness back off a roughened border, the way "
        "sandpaper does -- the counterpart to Edge Jitter rather than more of "
        "it. It averages each pixel with its neighbours *along* the edge, "
        "never across it, so the burrs and stair-stepping smooth out while "
        "the transition stays exactly as sharp. Raise it when jitter or "
        "erosion has left an edge looking harsh. 0 = off, 1 = the full "
        "cross-fade.\n"
        "\n"
        "**How much it takes off is Sanding Grit, not this.** At full strength "
        "a 5px grit removes about 46% of a jittered border's roughness and a "
        "10px grit about 67%; past that the filter starts reaching across the "
        "contour instead of along it and gets *less* effective (53% at 20px) "
        "while costing the edge its sharpness.",
    ),
    Param(
        "edge_sand_grit", "Sanding Grit", "Edge Destruction",
        0.3, 20.0, 0.05, 0.8, "px",
        "How far along the edge the polish reaches, at full resolution. Small "
        "is a fine grit: it takes off pixel-scale jaggies and leaves the "
        "border's shape alone. Large flattens broader undulations too, so the "
        "wander Edge Jitter added starts going with them.",
        spatial=True,
    ),
]
