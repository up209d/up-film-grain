"""3D LUT loading -- ``.cube`` files from the ``luts/`` folder or from an upload.

A LUT is the one thing in this app that is *not* a number, which is why it does
not live in ``params.py``. Every other control the engine takes is a float with
a range, so it can be sanitised, clamped, rescaled for a different image size
and stored in a preset file as a value. A LUT is a *resource*: it is identified
by name and its content is a table. So it travels beside the parameters -- in
the request body next to ``reference_mp``, and in a preset file as a sibling key
-- rather than inside them.

The obvious alternative was a ``choices`` menu indexed into the folder listing,
which would have needed no new plumbing at all. It is wrong for exactly the
reason ``_SCATTER_STENCILS`` documents: a preset stores the *index*, so dropping
one more ``.cube`` into the folder would silently renumber it and change the
look of every preset that named one. The folder is user-mutable by design -- it
is the whole interface, the same way ``presets/`` is -- so names it is.

Two sources, one namespace:

* **Disk.** ``luts/**/*.cube``, identified by the file's path *relative to*
  ``luts/`` with the extension dropped -- ``UP-SuperPortra`` at the root,
  ``gmic/colorslide/fuji_fp_100c`` in a folder. Named by the path rather than by
  the ``TITLE`` inside the file, matching how presets are named by filename. An
  id survives a restart, so a preset can reference one.

  The folder was flat until 2026-08-09, when a library of 300 arrived organised
  into subfolders. A bare stem would not do as an id from that point on: two
  folders may hold the same filename, and collapsing them would make which LUT a
  preset gets depend on directory iteration order. A root-level file's relative
  path *is* its bare stem, so no preset needed migrating.
* **Upload.** Held in process memory under an ``upload:`` id, LRU-capped. These
  deliberately do *not* survive a restart; a preset naming one degrades to no
  LUT rather than erroring, and the client can see the name is missing because
  it has the list.

Parsed tables are cached -- by path and mtime for disk, by id for uploads --
because the largest file here is a 64x64x64 table over 275k lines and reparsing
it on every slider drag would be the most expensive thing in the pipeline by an
order of magnitude.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

LUT_DIR = Path(
    os.environ.get("FILM_GRAIN_LUTS")
    or Path(__file__).resolve().parent.parent / "luts"
)

#: Upload cap. A 64-cube is ~6.8MB of text and a 128-cube would be ~55MB; the
#: cap sits above the former and below the latter on purpose, because a 128-cube
#: is 2M table entries and nothing in a film-grain preview needs that precision.
MAX_LUT_BYTES = 24 * 1024 * 1024

#: Grid points per axis. 65 is the largest anyone ships in practice.
MAX_LUT_SIZE = 96
MIN_LUT_SIZE = 2

#: How many uploaded LUTs are kept. They are a few MB of float32 each.
_MAX_UPLOADS = 8

CUBE_SUFFIX = ".cube"


class LutError(ValueError):
    """A file that is not a usable 3D LUT. The message is shown to the user."""


# eq=False: the generated __eq__ would compare `table` with ==, which on a numpy
# array returns an array and then raises on the truth test. Nothing compares two
# LUTs today, and this makes sure nothing can trip over it later.
@dataclass(eq=False)
class Lut:
    """A parsed 3D LUT, plus a lazily-built per-device tensor.

    ``table`` is ``[size, size, size, 3]`` indexed ``[b][g][r]``: the ``.cube``
    format states that the red channel varies fastest, so a C-order reshape of
    the flat data puts blue on the slowest axis. Getting that backwards is
    invisible on any LUT that happens to be symmetric and wrecks every one that
    is not, so ``verify.py`` pins it with a deliberately asymmetric table.
    """

    id: str
    name: str
    size: int
    table: np.ndarray
    dmin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dmax: tuple[float, float, float] = (1.0, 1.0, 1.0)
    source: str = "folder"
    #: Folder holding it, relative to ``LUT_DIR`` and POSIX-separated; ``""``
    #: for a root-level file and for every upload. The client groups the picker
    #: by it, so it is reported rather than re-derived there -- the id is a path
    #: and splitting paths is the server's business.
    group: str = ""
    #: Cached ``[1, 3, D, H, W]`` volumes, one per device string.
    _tensors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def tensor(self, device: torch.device) -> torch.Tensor:
        """The table as ``grid_sample``'s volume layout, cached per device.

        ``grid_sample`` reads a 5D input as ``[N, C, D, H, W]`` and a grid whose
        last dimension is ``(x, y, z)`` mapping to ``(W, H, D)``. Permuting the
        table to ``[c][b][g][r]`` therefore puts red on ``W``, green on ``H``
        and blue on ``D``, which is why the sampling grid downstream is just the
        image's own three channels in their natural order.
        """
        key = str(device)
        t = self._tensors.get(key)
        if t is None:
            t = (
                torch.from_numpy(np.ascontiguousarray(self.table.transpose(3, 0, 1, 2)))
                .unsqueeze(0)
                .to(device)
            )
            self._tensors[key] = t
        return t

    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "source": self.source,
            "group": self.group,
        }


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_KEYWORDS = {
    "TITLE", "LUT_3D_SIZE", "LUT_1D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX",
    "LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE",
}

_NUM = re.compile(r"^[-+.\d]")


def parse_cube(
    text: str, name: str, lut_id: str, source: str, group: str = ""
) -> Lut:
    """Parse an Adobe ``.cube`` 3D LUT.

    The header is scanned line by line -- there are only a handful of lines
    before the data -- and the data is then handed to ``np.fromstring`` in one
    call. Walking 275k data lines in Python would take longer than the render
    they are for; the filtering pass for interleaved comments only runs if there
    is actually a ``#`` past the header.
    """
    size = 0
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]

    lines = text.splitlines()
    start = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        head = s.split(None, 1)[0].upper()
        if head not in _KEYWORDS:
            if _NUM.match(s):
                start = i
                break
            # Unknown keyword: skip it rather than refusing the file. Vendors
            # emit their own metadata and it is never load-bearing.
            continue
        parts = s.split()
        try:
            if head == "LUT_3D_SIZE":
                size = int(float(parts[1]))
            elif head == "LUT_1D_SIZE":
                raise LutError(
                    f"{name} is a 1D LUT. Only 3D LUTs (LUT_3D_SIZE) are supported."
                )
            elif head == "DOMAIN_MIN":
                dmin = [float(v) for v in parts[1:4]]
            elif head == "DOMAIN_MAX":
                dmax = [float(v) for v in parts[1:4]]
            elif head.endswith("INPUT_RANGE"):
                # The older spelling of DOMAIN_MIN/MAX, as two numbers applied
                # to all three channels.
                lo, hi = float(parts[1]), float(parts[2])
                dmin, dmax = [lo] * 3, [hi] * 3
        except (IndexError, ValueError) as e:
            raise LutError(f"{name}: bad {head} line ({e}).") from e

    if not size:
        raise LutError(f"{name} has no LUT_3D_SIZE, so it is not a 3D LUT.")
    if not (MIN_LUT_SIZE <= size <= MAX_LUT_SIZE):
        raise LutError(
            f"{name} declares LUT_3D_SIZE {size}; supported range is "
            f"{MIN_LUT_SIZE}-{MAX_LUT_SIZE}."
        )

    body = lines[start:]
    if any("#" in ln for ln in body):
        body = [ln for ln in body if ln.strip() and not ln.lstrip().startswith("#")]
    data = np.fromstring("\n".join(body), dtype=np.float32, sep=" ")

    want = size ** 3 * 3
    if data.size < want:
        raise LutError(
            f"{name} is truncated: LUT_3D_SIZE {size} needs {want} numbers, "
            f"found {data.size}."
        )
    # Trailing junk (a stray footer) is dropped rather than refused.
    table = data[:want].reshape(size, size, size, 3)

    span = [hi - lo for lo, hi in zip(dmin, dmax)]
    if any(s <= 1e-6 for s in span):
        raise LutError(f"{name} has an empty input domain.")

    return Lut(
        id=lut_id, name=name, size=size,
        table=np.ascontiguousarray(table),
        dmin=(dmin[0], dmin[1], dmin[2]),
        dmax=(dmax[0], dmax[1], dmax[2]),
        source=source,
        group=group,
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

_LOCK = threading.Lock()
#: id (a relative path, extension dropped) -> (mtime, size_on_disk, Lut)
_DISK: dict[str, tuple[float, int, Lut]] = {}
#: id -> Lut, oldest first
_UPLOADED: "OrderedDict[str, Lut]" = OrderedDict()


def _group_of(f: Path) -> str:
    """The folder holding ``f``, relative to ``LUT_DIR``, ``""`` at the root.

    ``Path(".").as_posix()`` is ``"."`` for a root-level file; the client wants
    "in no folder" spelled as an empty string, so that one case is translated
    here rather than in every consumer.
    """
    group = f.parent.relative_to(LUT_DIR).as_posix()
    return "" if group == "." else group


def resolve_path(lut_id: str) -> Path | None:
    """``lut_id`` as a file inside ``LUT_DIR``, or ``None`` if it escapes it.

    This string arrives in a request body and is about to be joined onto a
    path, so it is the security boundary for the whole LUT folder. It used to
    be a one-liner -- reject anything with a separator in it -- which worked
    only while the folder was flat. Separators are now the *point*, so the rule
    has to be the stronger one it was standing in for:

    * no absolute paths, no drive letters, no backslashes (a Windows-style
      ``gmic\\bw\\x`` must not become a valid path on a POSIX box either);
    * no ``.`` or ``..`` segment anywhere, so nothing can climb out textually;
    * and then, having built the path, ``resolve()`` it and require the result
      to still be under ``LUT_DIR``. That last one is what a purely textual
      check cannot do: a symlink *inside* ``luts/`` pointing at ``/etc`` passes
      every rule above and is caught only by resolving it.
    """
    if not lut_id or "\\" in lut_id or lut_id.startswith("/"):
        return None
    parts = lut_id.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    path = LUT_DIR / f"{lut_id}{CUBE_SUFFIX}"
    try:
        real = path.resolve()
    except OSError:
        return None
    if not real.is_relative_to(LUT_DIR.resolve()):
        return None
    return path


def list_luts() -> list[dict]:
    """Every LUT the client can pick, folder first then uploads.

    The tree is re-walked on each call, like ``load_presets`` -- dropping a
    ``.cube`` in should show up on the next page load without a restart. Only
    the paths are listed here; nothing is parsed until it is used, so 300 LUTs
    cost a directory walk to browse and nothing else.

    ``rglob`` rather than ``glob``: a library of LUTs arrives organised into
    folders and flattening it would both lose that organisation and collide
    same-named files from different folders. It does not follow symlinks, which
    is what we want -- a link pointing at ``/`` would otherwise walk the disk.

    Root-level entries sort first and folders follow, so the picker's
    always-visible ungrouped section is the handful of LUTs that live at the top
    rather than an arbitrary slice of the library.
    """
    def sort_key(f: Path) -> tuple[int, str, str]:
        group = _group_of(f)
        # Root first, spelled out rather than leaning on "" sorting ahead of
        # every folder name -- that happens to be true and reads like an
        # accident.
        return (0 if not group else 1, group.lower(), f.name.lower())

    out: list[dict] = []
    try:
        files = sorted(
            (f for f in LUT_DIR.rglob("*") if f.suffix.lower() == CUBE_SUFFIX),
            key=sort_key,
        )
    except (OSError, ValueError):
        files = []
    for f in files:
        rel = f.relative_to(LUT_DIR)
        out.append({
            # The path, extension dropped -- see the module docstring for why a
            # bare stem stopped being enough.
            "id": rel.with_suffix("").as_posix(),
            "name": f.stem,
            "size": None,
            "source": "folder",
            "group": _group_of(f),
        })
    with _LOCK:
        for lut in reversed(_UPLOADED.values()):
            out.append(lut.info())
    return out


def get(lut_id: str | None) -> Lut | None:
    """Resolve a LUT by id, or ``None`` if there is no such thing.

    Deliberately not an error. A preset can name a LUT that has been renamed,
    deleted or was an upload from a previous run, and a render request is the
    wrong place to fail over it -- the picker already shows the name as missing
    because the client has the list.
    """
    if not lut_id:
        return None
    if lut_id.startswith("upload:"):
        with _LOCK:
            lut = _UPLOADED.get(lut_id)
            if lut is not None:
                _UPLOADED.move_to_end(lut_id)
            return lut

    # A path from the folder, relative to it. `resolve_path` is the guard; see
    # its docstring for why "no separators" stopped being the rule.
    path = resolve_path(lut_id)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None

    with _LOCK:
        hit = _DISK.get(lut_id)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
    if st.st_size > MAX_LUT_BYTES:
        return None
    try:
        # utf-8 explicitly, matching `add_upload`: the same .cube must decode
        # identically whether it was dropped in `luts/` or uploaded through the
        # API, and the locale default (cp1252 on Windows) would mangle a
        # non-ASCII TITLE in one path but not the other. `errors="replace"`
        # stays -- a cube body is numeric, so a stray byte should cost a
        # character rather than the whole LUT. Text mode also keeps universal
        # newlines, which `luts/ClassicNegative.cube` (CRLF) relies on: do not
        # "tidy" this into read_bytes().decode().
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        # `name` is the stem and `id` is the path: the two agree at the root and
        # part company in a folder, which is exactly the split `list_luts`
        # reports, so the picker and a directly-resolved LUT label the same way.
        lut = parse_cube(text, path.stem, lut_id, "folder", _group_of(path))
    except LutError as e:
        print(f"[luts] skipping {lut_id}: {e}")
        return None
    with _LOCK:
        _DISK[lut_id] = (st.st_mtime, st.st_size, lut)
    return lut


def add_upload(filename: str, data: bytes) -> Lut:
    """Parse an uploaded ``.cube`` and keep it under a fresh id."""
    if len(data) > MAX_LUT_BYTES:
        raise LutError(
            f"That LUT is {len(data) / 1e6:.1f}MB; the limit is "
            f"{MAX_LUT_BYTES // (1024 * 1024)}MB."
        )
    stem = Path(filename or "lut").stem or "lut"
    if Path(filename or "").suffix.lower() != CUBE_SUFFIX:
        raise LutError("Only .cube LUT files are supported.")
    lut_id = f"upload:{uuid.uuid4().hex[:10]}"
    lut = parse_cube(data.decode("utf-8", "replace"), stem, lut_id, "upload")
    with _LOCK:
        _UPLOADED[lut_id] = lut
        while len(_UPLOADED) > _MAX_UPLOADS:
            _UPLOADED.popitem(last=False)
    return lut
