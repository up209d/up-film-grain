"""preset rescaling and the mark-count dead zone

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import json

from server import params as P
from tests.harness import Ctx, check, suite


@suite("presets", "preset rescaling and the mark-count dead zone")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    # -- 3c. preset rescaling across image sizes -------------------------------
    # A preset dialled in on one size has to hold its look on another. The
    # scale is the ratio of *linear* dimensions, not of pixel counts -- a 24MP
    # frame is 1.29x the width of a 16MP one, not 1.5x -- and only lengths move.
    print("\npreset rescaling (same look on a different-sized photo)")
    check(
        "linear, not area", abs(P.scale_factor(24.0, 96.0) - 2.0) < 1e-6,
        f"24MP -> 96MP is 4x the pixels, {P.scale_factor(24.0, 96.0):.2f}x the width",
    )
    src = P.sanitize({"grain_size": 2.0, "halation_radius": 20.0,
                      "intensity": 45.0, "dust": 50.0, "leak_variation": 0.5})
    got = P.rescale(src, 1.6)
    check(
        "lengths scale", abs(got["grain_size"] - 3.2) < 1e-4
        and abs(got["halation_radius"] - 32.0) < 1e-4,
        f"grain_size {src['grain_size']} -> {got['grain_size']:.2f}, "
        f"halation_radius {src['halation_radius']} -> {got['halation_radius']:.1f}",
    )
    check(
        "amounts and counts do not",
        got["intensity"] == src["intensity"] and got["dust"] == src["dust"]
        and got["leak_variation"] == src["leak_variation"],
        "intensity, dust count and leak_variation all unchanged",
    )
    check(
        "every spatial param is marked",
        {x.key for x in P.PARAMS if x.spatial} == {
            "aa_radius",
            "dust_size", "edge_jitter", "edge_sand_grit", "edge_soften_radius",
            "global_size", "global_size_max", "grade_clarity_radius",
            "grade_recover_radius",
            "grain_size", "hair_length",
            "halation_radius", "highpass_radius", "leak_feather",
            "leak_size_max", "leak_size_min",
            "micro_blur", "pre_blur", "pre_sharpen_radius",
            "scatter_cell", "scatter_radius",
            "scratch_width", "sharpen_radius"},
        f"{sum(1 for x in P.PARAMS if x.spatial)} marked spatial",
    )
    check(
        "no reference means no change", P.rescale(src, 1.0) == src,
        "scale_factor(None, x) = " f"{P.scale_factor(None, 40.0):.1f}",
    )

    # -- 3d. no shipped preset may sit in a mark count's dead zone -----------
    # dust/scratches/hair/light_leak are *counts*, and the engine gates each on
    # `>= 1.0` -- you cannot render a third of a scratch. So a value in (0, 1)
    # renders nothing at all while reading, in the panel and in the file, as
    # though the section were slightly on. Three shipped presets sat there:
    # they carried 0-1 amounts from before these became counts and were never
    # migrated, so their entire Film Texture section had been silently inert.
    # It is invisible from the code and invisible from the UI, which is exactly
    # the kind of thing that only a check catches.
    print("\npreset sanity (mark counts must not sit in the dead zone)")
    dead = [
        (q["name"], k, q["values"][k])
        for q in P.load_presets()
        for k in ("dust", "scratches", "hair", "light_leak")
        if 0.0 < q["values"][k] < 1.0
    ]
    check(
        "no count between 0 and 1", not dead,
        "all counts are 0 or a real number of marks" if not dead
        else ", ".join(f"{n}.{k}={x}" for n, k, x in dead),
    )

    # -- 3e. attribution survives the trip to the client --------------------
    # Every shipped preset file names an author, and the client writes those
    # names back out when a look is saved to a file again. It can only write
    # what it was sent, and `load_presets` builds its own dict rather than
    # passing the parsed file through -- so a key it forgets is a credit that
    # disappears the moment someone nudges a slider and re-saves. That is
    # invisible from the render, which is why it is asserted here.
    print("\npreset attribution (the credit has to reach the client)")
    on_disk = 0
    for f in P.PRESET_DIR.glob("*.json"):
        try:
            on_disk += bool(json.loads(f.read_text()).get("author"))
        except (OSError, ValueError):
            pass
    loaded = P.load_presets()
    named = [q for q in loaded if q["author"]]
    check(
        "author reaches the schema", len(named) == on_disk and on_disk > 0,
        f"{len(named)} of {len(loaded)} presets report an author, "
        f"{on_disk} name one on disk",
    )
    check(
        "the link comes with it",
        all(q["author_link"] for q in named),
        f"{sum(1 for q in named if q['author_link'])} of {len(named)} "
        "attributed presets carry author_link",
    )
