"""Presets as files on disk, read fresh on every schema() call."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .param import Param
from .sanitize import sanitize

# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #

# Presets are files on disk, not a table in this module. Anything you save from
# the UI can be dropped in here and it is a preset; nothing has to be edited or
# redeployed to add one. The directory sits beside the server package, so the
# same relative layout works in the source tree and in a built distribution.
PRESET_DIR = Path(
    os.environ.get("FILM_GRAIN_PRESETS")
    or Path(__file__).resolve().parent.parent.parent / "presets"
)

# Preset applied when the app opens, and the one Reset returns to -- by name,
# which is to say by filename. It is only a *preferred* name: if no such file
# exists the client falls back to the parameter defaults, so deleting it is a
# supported way to start from neutral rather than a way to break startup.
DEFAULT_PRESET = os.environ.get("FILM_GRAIN_DEFAULT_PRESET", "Stock")

# Fallback size for presets that do not record one. Unset by default: a preset
# with no `reference_mp` scales by 1.0, which is exactly how it behaved before
# rescaling existed. Guessing a size here would silently change the look of
# every legacy preset, and a wrong guess is worse than no scaling.
#
# Set it if you know your existing presets were all dialled in on the same
# camera -- FILM_GRAIN_DEFAULT_REFERENCE_MP=24 retrofits the lot in one go.
try:
    DEFAULT_REFERENCE_MP: float | None = (
        float(os.environ.get("FILM_GRAIN_DEFAULT_REFERENCE_MP", "") or 0) or None
    )
except ValueError:
    DEFAULT_REFERENCE_MP = None


def load_presets() -> list[dict]:
    """Read every ``*.json`` in ``PRESET_DIR``, sorted by name.

    Read on each call rather than cached, so dropping a file into the folder
    shows up on the next page load without restarting the server. It is a
    handful of small files.

    A preset is named by its **filename**, not by whatever ``name`` the file
    carries inside. The files are the interface here -- renaming one in Finder
    should rename it in the dropdown, and a file saved under one name and
    renamed later should not keep announcing the old one.

    Values go through ``sanitize`` like any other input: unknown keys dropped,
    everything clamped into range, anything missing filled from defaults. So a
    hand-edited file, or one written before a slider's range changed, still
    loads instead of poisoning the engine.
    """
    out: list[dict] = []
    try:
        files = sorted(PRESET_DIR.glob("*.json"), key=lambda f: f.name.lower())
    except OSError:
        return out

    for f in files:
        try:
            # Explicit encoding rather than the locale's. A preset whose name
            # or author credit carries an accent would otherwise raise
            # UnicodeDecodeError under a cp1252 default -- and that is a
            # ValueError, so the handler below catches it and the preset
            # quietly ceases to exist on one platform and not another.
            # `lut.add_upload` already pins utf-8; this is the same decision.
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            # Say so rather than silently omitting it -- a typo in one file
            # should not make a preset quietly cease to exist.
            print(f"[presets] skipping {f.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(raw, dict):
            print(f"[presets] skipping {f.name}: not a JSON object", file=sys.stderr)
            continue
        # Accept our own wrapper or a bare {key: value} map, matching what the
        # client's file loader accepts.
        values = raw.get("values")
        if not isinstance(values, dict):
            values = raw
        ref = raw.get("reference_mp") or DEFAULT_REFERENCE_MP
        edge = raw.get("proxy_edge")
        lut = raw.get("lut")
        author = raw.get("author")
        author_link = raw.get("author_link")
        out.append({
            "name": f.stem,
            "values": sanitize(values),
            # Which 3D LUT the look wants, by name. A sibling key rather than a
            # value, for the reason server/lut.py sets out -- it is a resource,
            # not a quantity. Unresolvable names (a renamed file, an upload from
            # a previous run) degrade to no LUT rather than erroring.
            "lut": lut if isinstance(lut, str) and lut else None,
            # Size the preset was dialled in on, so it can be rescaled onto a
            # different photo. Absent in older files -> no scaling, which is
            # the pre-existing behaviour rather than a guess.
            "reference_mp": float(ref) if isinstance(ref, (int, float)) and ref else None,
            # Long edge of the proxy tier the look was judged on (2026-08-29,
            # on request). A sibling of the values for the same reason
            # `reference_mp` is: it is not a quantity the engine reads, it is a
            # fact about how finely the frame this look was dialled in on had
            # been resolved -- and five of the six export entries render that
            # tier, so it decides the texture of the file as much as any slider
            # does. Absent -> the caller's own default, which is
            # `PROXY_LONG_EDGE` everywhere it is consumed.
            #
            # Deliberately *not* clamped here. `_clamp_edge` lives in
            # `models/upload.py`, which imports this package, so reaching for it
            # would close the import graph on itself; every consumer already
            # clamps (`controllers/export.py` server-side, the slider's own
            # bounds client-side), which is the same bargain `reference_mp`
            # makes with `sanitize`.
            "proxy_edge": int(edge) if isinstance(edge, (int, float)) and edge else None,
            # Who made the look, and where to find them. Carried through rather
            # than dropped because the client writes them back out when the
            # preset is saved to a file again -- every shipped preset has
            # carried these two keys since the library was written, and a
            # round trip through "Save to file..." used to strip them, so the
            # attribution survived exactly until someone tweaked one slider.
            # Not sanitised beyond the type check: they are free text, and the
            # only thing the app does with them is put them back.
            "author": author if isinstance(author, str) and author else None,
            "author_link": (
                author_link if isinstance(author_link, str) and author_link else None
            ),
        })
    return out
