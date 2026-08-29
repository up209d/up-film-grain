"""Render one photograph from the command line.

    python -m server.cli photo.jpg -p Portra --mode ss2 -o out.png

Four questions and no more: **input, preset, output, output mode**. Everything
else -- the parameter values, the LUT, the size the file is written at, the
name it gets -- is already decided by the preset and by the same code the app
uses, and asking again here would be a second place for it to be decided.

It is deliberately a thin shell over the HTTP layer rather than a second way to
render. `POST /api/export` is reproduced by *calling* it: the same
`params_for`, the same `run_export`, the same tier, the same filename tag, the
same `prescale_output` decision. A CLI that reimplemented any of that would be
a copy that drifts, and the drift would show as files that do not match what the
app produces from the same preset -- which is the one thing this has to
guarantee.

No server, no port, no `web/dist`: this imports the domain directly, so it works
in a checkout, in the bundle, and on a machine that never opens the app.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import time
import uuid
from pathlib import Path

from . import imageio as iio
from .controllers.export import export as start_export
from .models import upload as up_model
from .models.export_job import JOBS
from .models.upload import (PROXY_EDGE_MAX, PROXY_EDGE_MIN,
                            PROXY_LONG_EDGE, SUPERSAMPLES)
from . import params as P
from .params import PRESET_DIR, load_presets

# The export menu, server-side: five preview-tier entries plus the one that
# renders the source itself at 1.0. Mirrors `EXPORT_OPTIONS` in the client
# (`web/src/services/api.ts`) because it is the same menu -- the CLI offers what
# the app offers, and neither invents a mode the other cannot produce.
#: Set by `--all`; read by `_describe_preset`. A module global rather than an
#: argument because it is a display flag for one command, and threading it
#: through would put it in a signature that is otherwise "which preset".
_ALL = False

_MODES: dict[str, tuple[float, bool]] = {
    **{f"ss{s:g}": (s, False) for s in SUPERSAMPLES},
    "full": (1.0, True),
}


def _pick_preset(name: str | None) -> dict | None:
    """The named preset, matched case-insensitively by filename.

    `None` means the parameter defaults, which is a supported answer rather than
    an omission -- it is what the app shows with no preset chosen.
    """
    presets = load_presets()
    if name is None:
        return None
    for pre in presets:
        if pre["name"].lower() == name.lower():
            return pre
    have = ", ".join(p["name"] for p in presets) or "(none)"
    raise SystemExit(f"Unknown preset {name!r}. Available: {have}")


def _format_for(out: Path | None, explicit: str | None) -> str:
    """The encoder to use: what was asked for, or what the output name implies.

    Inferring from the extension rather than defaulting to one format is the
    only behaviour that makes `-o out.png` mean what it says. 16-bit is the
    default depth for PNG because it is the reason to choose PNG at all --
    grain lives in the low bits.
    """
    if explicit:
        if explicit not in iio.FORMATS:
            raise SystemExit(
                f"Unknown format {explicit!r}. "
                f"One of: {', '.join(iio.FORMATS)}"
            )
        return explicit
    ext = (out.suffix.lower().lstrip(".") if out else "")
    return {"png": "png16", "jpg": "jpeg", "jpeg": "jpeg"}.get(ext, "jpeg")


def _colour(tty: bool):
    """`bold`, `dim` and `plain`, or three no-ops when this is not a terminal.

    Decided once from `isatty` rather than per call: `--list-presets | grep` and
    `> file` are both ordinary uses, and escape codes in a pipe are noise the
    caller has to strip. The same reason `NO_COLOR` is honoured -- it is the
    one convention every CLI that emits colour is expected to know.
    """
    if not tty or os.environ.get("NO_COLOR"):
        ident = lambda t: t  # noqa: E731
        return ident, ident
    return (lambda t: f"\033[1m{t}\033[0m", lambda t: f"\033[2m{t}\033[0m")


def _list_presets() -> int:
    """The preset library, with what each one needs to be used and credited.

    The heading says `-p` outright because the **filename** is the name to pass,
    not the `name` inside the file -- `ClassicSoft.json` calls itself "Classic"
    and `-p Classic` would fail. That mismatch is exactly the sort of thing a
    bare list of names invites someone to get wrong.

    Author and link are printed rather than merely carried because they are the
    reason `load_presets` keeps them at all: a look someone dialled in and gave
    away should say who made it wherever it is used, and the CLI is the one
    surface with no About panel to put that in.
    """
    presets = load_presets()
    bold, dim = _colour(sys.stdout.isatty())
    if not presets:
        print(f"No presets in {PRESET_DIR}")
        return 0

    print(f"{len(presets)} presets in {dim(str(PRESET_DIR))}")
    print(dim("pass one with -p, e.g.  ./export.sh photo.jpg -p "
              + presets[0]["name"]))
    print()
    for pre in presets:
        # The right-hand facts are what makes two similar-sounding presets
        # tellable apart: which LUT the look is built on, and the size it was
        # dialled in at -- the number `reference_mp` rescales every length
        # against when the photograph is a different size.
        facts = []
        if pre["lut"]:
            facts.append(f"LUT {pre['lut']}")
        if pre["reference_mp"]:
            facts.append(f"{pre['reference_mp']:g} MP")
        # Only when it is not the built-in default: the column is what makes
        # two presets tellable apart, and a number every entry repeats tells
        # nobody anything. A look judged on a smaller or larger proxy than the
        # rest of the library is exactly the thing worth seeing here.
        if pre["proxy_edge"] and pre["proxy_edge"] != PROXY_LONG_EDGE:
            facts.append(f"{pre['proxy_edge']:d}px proxy")
        line = "  " + bold(pre["name"])
        if facts:
            # Padded on the *undecorated* name -- the escape codes are zero
            # width on screen and four characters to `ljust`, which is what
            # makes a coloured column silently ragged.
            line += " " * max(1, 24 - len(pre["name"])) + dim(" · ".join(facts))
        print(line)
        credit = " · ".join(
            x for x in (pre["author"], pre["author_link"]) if x
        )
        if credit:
            print("    " + dim(credit))
    return 0


def _value_text(par, v: float) -> str:
    """One parameter's value as the panel would show it.

    A menu shows its label and a checkbox shows on/off, because the *number* is
    an implementation detail everywhere but in the engine -- `global_blend 3`
    tells you nothing and "Overlay" tells you the look. Same reason `unit` is
    printed: the schema knows these are pixels or degrees and a bare 0.8 does
    not.
    """
    if par is None:
        return f"{v:g}"
    if par.toggle:
        return "on" if v >= 0.5 else "off"
    if par.choices:
        i = int(round(v))
        if 0 <= i < len(par.choices):
            return par.choices[i]
        return f"{v:g}"
    return f"{v:g}{par.unit}"


def _describe_preset(name: str) -> int:
    """One preset in full: its credit, and every value it sets.

    Grouped by `GROUPS` and printed **in pipeline order**, which is the panel's
    order too -- reading a preset in the order the engine applies it is the only
    arrangement in which "this softens the edges and then grains it" is visible
    at all.

    Only the values that differ from the parameter defaults are printed. A
    preset carries all 50-odd keys after `sanitize` fills it in, and a wall of
    mostly-zeroes hides the handful of numbers that are the look. `--all` prints
    the rest.
    """
    pre = _pick_preset(name)
    assert pre is not None  # `_pick_preset` raises rather than returning None
    bold, dim = _colour(sys.stdout.isatty())
    values = pre["values"]

    print(bold(pre["name"]))
    credit = " · ".join(x for x in (pre["author"], pre["author_link"]) if x)
    if credit:
        print("  " + dim(credit))
    meta = [f"LUT {pre['lut']}"] if pre["lut"] else []
    if pre["reference_mp"]:
        meta.append(f"dialled in at {pre['reference_mp']:g} MP")
    # Unconditional here where the list prints it only when it is unusual: this
    # is the one command that answers "what will `-p this` actually render",
    # and the edge is a render decision the preset makes on your behalf unless
    # `-e` says otherwise.
    meta.append(f"{pre['proxy_edge'] or PROXY_LONG_EDGE:d}px proxy"
                + ("" if pre["proxy_edge"] else " (default)"))
    meta.append(f"{PRESET_DIR / (pre['name'] + '.json')}")
    print("  " + dim(" · ".join(meta)))

    shown = 0
    for group in P.GROUPS:
        rows = []
        for par in P.PARAMS:
            if par.group != group or par.key not in values:
                continue
            v = float(values[par.key])
            if _ALL or abs(v - par.default) > 1e-9:
                rows.append((par, v))
        if not rows:
            continue
        print()
        print("  " + bold(group))
        for par, v in rows:
            shown += 1
            pad = " " * max(1, 26 - len(par.label))
            line = f"    {par.label}{pad}{_value_text(par, v)}"
            if not _ALL:
                # What it is being compared against, since the list is a diff.
                line += dim(f"   (default {_value_text(par, par.default)})")
            print(line)
    print()
    print(dim(f"  {shown} "
              + ("values" if _ALL else "values differ from the defaults")
              + " · render it with:  ./export.sh photo.jpg -p " + pre["name"]))
    return 0


def _epilog() -> str:
    """The examples, and the preset names read off disk.

    Listed in the help rather than only behind `--list-presets` because a
    preset name is not guessable and `-p` is the one argument with no sensible
    default to fall back on. Read live for the same reason `load_presets` is:
    dropping a file into `presets/` is how a preset is added, so a hardcoded
    list here would be wrong the moment anyone did that.
    """
    names = [pre["name"] for pre in load_presets()]
    wrapped = textwrap.fill(", ".join(names) or "(none found)",
                            width=76, initial_indent="  ",
                            subsequent_indent="  ")
    return f"""output modes:
  ss0.5 ss1 ss1.5 ss2 ss3   the previewed frame, rendered that finely and
                            enlarged to full size. ss2 is the default and the
                            look every preset was dialled in against.
  full                      a real 1:1 render at 1x -- finer, denser grain
                            than the app's preview shows, so it is a different
                            picture rather than a sharper one.

presets:
{wrapped}

examples:
  ./export.sh -i photo.jpg -p KodakPortra
  ./export.sh photo.jpg -p Stock -m full -o out.png
  ./export.sh photo.jpg -o out.jpg -q 100
  ./export.sh -l
  ./export.sh -d KodakPortra
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="film-grain",
        description="Apply film grain to a photograph. Renders exactly what "
                    "the app would from the same preset.",
        epilog=_epilog(),
        # The epilog is laid out by hand -- columns and example commands, both
        # of which argparse's default formatter reflows into one paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The input is accepted both ways on purpose: `-i` so it reads like every
    # other option, and bare positional so `./export.sh photo.jpg` -- the form
    # every earlier example used -- keeps working. One dest, two spellings;
    # giving both is an error rather than one of them silently winning.
    ap.add_argument("-i", "--input", dest="input",
                    help="JPEG or PNG to render (may also be positional)")
    ap.add_argument("input_pos", nargs="?", metavar="input",
                    help=argparse.SUPPRESS)
    ap.add_argument("-p", "--preset", help="preset name (see -l)")
    ap.add_argument("-o", "--output",
                    help="output file; default is the app's own name, "
                         "beside the input")
    ap.add_argument("-m", "--mode", default="ss2", choices=list(_MODES),
                    help="output mode: how finely the frame is rendered. "
                         "ss2 is the default and what every preset was dialled "
                         "in against; full is a real 1:1 render (default: ss2)")
    # No `default`, so "not given" stays tellable from "given as 2400". The
    # preset is the fallback, and a preset that names an edge has to be able to
    # win over the built-in default while still losing to an explicit -e.
    ap.add_argument("-e", "--proxy-edge", type=int, default=None,
                    metavar="PX",
                    help=f"long edge of the proxy the frame is rendered at, "
                         f"{PROXY_EDGE_MIN}-{PROXY_EDGE_MAX} in steps of 100. "
                         f"Cost goes roughly as its square, and the file "
                         f"carries that tier's texture enlarged, so this is the "
                         f"largest lever over both render time and how much "
                         f"detail the export resolves. Ignored by -m full, "
                         f"which renders the frame itself at 1:1. Omitted, the "
                         f"preset's own edge is used (see -d), and a preset "
                         f"that names none renders at {PROXY_LONG_EDGE}")
    ap.add_argument("-f", "--format", choices=list(iio.FORMATS),
                    help="override the encoder implied by the output name")
    ap.add_argument("-q", "--quality", type=int, default=95,
                    help="JPEG quality, 60-100 (default: 95)")
    ap.add_argument("-d", "--describe-preset", metavar="NAME",
                    help="print one preset's credit and every value it sets, "
                         "grouped in pipeline order, and exit")
    ap.add_argument("-a", "--all", action="store_true",
                    help="with --describe-preset, print every parameter "
                         "rather than only the ones that differ from the "
                         "defaults")
    ap.add_argument("-l", "--list-presets", action="store_true",
                    help="print the preset library -- names, authors and the "
                         "LUT each look uses -- and exit")
    args = ap.parse_args(argv)
    if args.input and args.input_pos:
        ap.error("give the input once -- either -i/--input or positionally")
    args.input = args.input or args.input_pos

    if args.list_presets:
        return _list_presets()
    if args.describe_preset:
        global _ALL
        _ALL = args.all
        return _describe_preset(args.describe_preset)
    if not args.input:
        # Help rather than `ap.error`. Bare `./export.sh` is someone asking
        # what this does, not someone who forgot an argument -- a usage line
        # and exit 2 answers a question they were not asking.
        ap.print_help()
        return 0

    src = Path(args.input).expanduser()
    try:
        arr = iio.load_image(src.read_bytes())
    except OSError as e:
        raise SystemExit(f"Cannot read {src}: {e}")
    except Exception as e:
        raise SystemExit(f"Cannot decode {src}: {type(e).__name__}: {e}")

    uid = uuid.uuid4().hex[:12]
    up_model.UPLOADS[uid] = up_model.Upload(uid, src.name, arr)

    preset = _pick_preset(args.preset)
    out = Path(args.output).expanduser() if args.output else None
    ss, full = _MODES[args.mode]
    body = {
        "id": uid,
        # `params` goes through `sanitize` in `params_for` exactly as a request
        # body does, so a preset missing keys fills from defaults here for the
        # same reason it does in the app.
        "params": preset["values"] if preset else {},
        "lut": preset["lut"] if preset else None,
        "reference_mp": preset["reference_mp"] if preset else None,
        "format": _format_for(out, args.format),
        "supersample": ss,
        "full": full,
        # Clamped and snapped server-side by `_clamp_edge`, exactly as a request
        # body is, so the CLI cannot ask for anything the app could not. `None`
        # when `-e` was not given, which is not the same as absent-and-defaulted:
        # the controller reads it as "no opinion" and falls back to the preset's
        # own edge, then to `PROXY_LONG_EDGE`. Passing the default here instead
        # would make every CLI render silently override the preset.
        "proxy_edge": args.proxy_edge,
        "quality": args.quality,
        # Recorded in the written file's metadata, and only here -- the name,
        # not the values. Absent when `-p` was not given, so a default render
        # writes a file that claims nothing.
        "preset": preset["name"] if preset else None,
    }

    job_id = start_export(body)["job"]
    job = JOBS[job_id]
    # The render runs on its own thread inside the controller; this is the
    # progress bar the app draws, printed instead.
    last = ""
    while job["status"] in ("queued", "rendering", "encoding"):
        line = f"\r{job['status']:>9}  {job['progress'] * 100:5.1f}%"
        if line != last:
            sys.stderr.write(line)
            sys.stderr.flush()
            last = line
        time.sleep(0.1)
    sys.stderr.write("\r" + " " * len(last) + "\r")

    if job["status"] != "done":
        raise SystemExit(job.get("error", "Export failed."))

    # Named by the app's own rule when the user did not say -- the tag carries
    # the supersample and the prescale target, which is what makes two exports
    # of one photograph tellable apart in a folder listing.
    dst = out or src.with_name(job["filename"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob = job["blob"]
    if blob.path is not None:
        shutil.copyfile(blob.path, dst)
    else:  # no cache directory available; the bytes are held instead
        dst.write_bytes(blob.data)
    print(f"{dst}  {job['width']}x{job['height']}  {job['size'] / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
