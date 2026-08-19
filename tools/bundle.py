#!/usr/bin/env python3
"""Assemble a self-contained Film Grain bundle: app payload + a Python runtime.

The old distribution was self-contained *apart from Python*, which is the part
that matters: `build/run.sh` searched for an interpreter, checked
`import fastapi, torch, uvicorn`, and when that failed printed instructions
telling the recipient to make a venv and pip-install ~700MB of torch. This script
puts the interpreter and every dependency **inside** the bundle, so the machine
it lands on needs nothing at all.

Why a relocatable interpreter and not a frozen binary: `server/controllers/client.py`
resolves the compiled client relative to `__file__` and **raises at import** if it
is not there, and nothing in the repo knows about `sys.frozen`/`sys._MEIPASS`. A
PyInstaller `--onefile` build relocates the tree under a temp prefix and would die
on that line. Keeping a real directory layout means all three data roots --
the client, `presets/` and `luts/` -- resolve exactly as they do from source, and
no path-handling code had to change.

Written in Python rather than added to `build.sh` because the same steps have to
run on Windows eventually, and two shell scripts drifting apart is how this
project already shipped a distribution containing 7 of its 303 LUTs.

  python3 tools/bundle.py --target mac
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "bundle"
CACHE = ROOT / ".cache" / "runtimes"
REQS = ROOT / "requirements"

# ---------------------------------------------------------------- the runtime --
# astral-sh/python-build-standalone. The `install_only` archives are
# prefix-relocatable: they run from any directory, which is the one property that
# makes this whole approach work. Pinned to an exact release and verified by
# hash -- an unpinned "latest" would silently change the interpreter under a
# reproducible build, and an unverified download is arbitrary code running as the
# user who unpacks it.
#
# 3.13 rather than 3.14 to match `Pipfile.lock`. (The repo's docs say torch has no
# 3.14 wheels; that is now out of date -- 2.13.0 publishes cp314 -- but the lock
# is calibrated on 3.13 and moving it is not this change's business.)
_PBS_RELEASE = "20260814"
_PBS_PYTHON = "3.13.15"
_PBS_BASE = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{_PBS_RELEASE}/"
)
# From the release's own SHA256SUMS manifest.
_PBS_SHA256 = {
    "aarch64-apple-darwin": "7d50bb42813a5644db7c40d3ad79361d0b724bb29d25a91fab1048c2c5c6a8c5",
    "x86_64-pc-windows-msvc": "4ca61e4b09c2240cc50cc6910c90664051e93ab7caa2f48b3c6b3c070670c0bd",
    "x86_64-unknown-linux-gnu": "45816a2653b47a6cc48d8ada4ea1185758a4c2db389d012b31e0205e5ccb548b",
}


@dataclass(frozen=True)
class Target:
    triple: str
    torch: str          # which requirements/torch-*.txt to install
    python: str         # interpreter path inside the extracted runtime
    supported: bool
    note: str = ""


# One artifact per platform, and the *compute backend is chosen at run time*, not
# here. `pick_device()` already takes CUDA if it is there, else MPS, else CPU, and
# reports the result through `/api/health` and the `X-Render-Device` header, so
# the top bar says which one is in play. That is why windows and linux install the
# CUDA build of torch: a CUDA wheel runs perfectly well on a machine with no
# NVIDIA GPU -- `torch.cuda.is_available()` is simply False and the engine falls
# through to CPU -- so one download serves both, at the cost of carrying the CUDA
# runtime for people who never use it.
TARGETS: dict[str, Target] = {
    "mac": Target(
        triple="aarch64-apple-darwin", torch="mac", python="bin/python3",
        supported=True,
    ),
    "windows": Target(
        triple="x86_64-pc-windows-msvc", torch="cuda", python="python.exe",
        supported=False,
        note="the payload and runtime steps are ready, but native wheels cannot "
             "be cross-built and an Electron Windows app cannot be produced from "
             "macOS -- this target has to be built on Windows",
    ),
    "linux": Target(
        triple="x86_64-unknown-linux-gnu", torch="cuda", python="bin/python3",
        supported=False,
        note="same as windows: the wheels are Linux-native, so it has to be "
             "built on Linux",
    ),
}

# Pruned after install. Every entry is either test material, build-time-only, or
# regenerable -- nothing here is imported at run time. Deliberately conservative:
# torch is easy to break by deleting something that looks inert, and a bundle that
# is 80MB smaller and cannot render is not a saving.
_PRUNE_DIRS = (
    "torch/include",        # C++ headers, for building extensions
    "torch/test",           # test binaries
    "torch/_inductor/codegen/cpp_prefix.h",
    "numpy/tests",
)
_PRUNE_STDLIB = ("test", "idlelib", "tkinter", "turtledemo", "ensurepip")
_PRUNE_GLOBS = ("**/*.a", "**/*.pdb", "**/*.lib")


def say(msg: str) -> None:
    print(f"\033[1m==>\033[0m {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"\033[33mwarning:\033[0m {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> "None":
    raise SystemExit(f"\033[31merror:\033[0m {msg}")


def du(path: Path) -> str:
    if not path.exists():
        return "0B"
    total = (path.stat().st_size if path.is_file()
             else sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024
    return f"{total:.1f}GB"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


# ------------------------------------------------------------------- payload --
def assemble_payload(dest: Path) -> dict[str, int]:
    """Copy everything the server needs to run into ``dest``.

    Layout matches the source tree exactly -- `launch.py` beside `server/`, with
    `web/dist`, `presets/` and `luts/` as siblings -- because all three data roots
    are resolved from `__file__` and any other shape would need code changes.
    """
    counts: dict[str, int] = {}
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # The server package, minus bytecode (recompiled later, for this runtime).
    shutil.copytree(
        ROOT / "server", dest / "server",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(ROOT / "launch.py", dest / "launch.py")

    client = ROOT / "web" / "dist"
    if not (client / "index.html").is_file():
        die(f"no compiled client at {client}. Build it first: cd web && npm run build")
    (dest / "web").mkdir()
    shutil.copytree(client, dest / "web" / "dist")

    # Presets. The folder ships even when empty: the server tolerates a missing
    # one, but shipping it is what makes it obvious where presets go.
    (dest / "presets").mkdir()
    presets = sorted((ROOT / "presets").glob("*.json"))
    for f in presets:
        shutil.copy2(f, dest / "presets" / f.name)
    counts["presets"] = len(presets)

    # LUTs: **walked, preserving the tree**. `luts/` is a directory tree and a
    # LUT's id *is* its path relative to `luts/`, so a flattened copy breaks every
    # nested id a preset references. A glob here once shipped 7 of 303 and said
    # nothing about it.
    lut_src = ROOT / "luts"
    cubes = sorted(lut_src.rglob("*.cube")) if lut_src.is_dir() else []
    for f in cubes:
        rel = f.relative_to(lut_src)
        out = dest / "luts" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    counts["luts"] = len(cubes)

    # The LUT tree carries third-party data: luts/gmic/ is Pat David's film
    # emulation set under CC BY-SA 4.0, which requires the credit and licence to
    # travel WITH the data. The walk above filters on *.cube and so skips the one
    # file that makes the redistribution lawful -- hence a second walk rather than
    # a wider filter, which would also corrupt the LUT count.
    extra = 0
    for name in ("LICENSE", "NOTICE", "README.md"):
        for f in lut_src.rglob(name) if lut_src.is_dir() else []:
            rel = f.relative_to(lut_src)
            out = dest / "luts" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            extra += 1
    counts["lut_notices"] = extra

    say(f"payload: {counts['presets']} preset(s), {counts['luts']} LUT(s), "
        f"{extra} LUT licence file(s), {du(dest)}")
    if extra == 0 and cubes:
        warn("no LICENSE/NOTICE found under luts/ -- CC BY-SA requires the "
             "licence to travel with the data")
    return counts


def assemble_licenses(dest: Path) -> None:
    """Our licence text plus the third-party notices the bundle now carries.

    `build.sh` already refused to build without LICENSE and NOTICE, on the
    grounds that whoever runs the build is the one distributing and section 4
    obliges them to convey the text. That still holds, and the surface has grown:
    the bundle now also carries CPython, torch, numpy, pillow and (via
    electron-builder) Chromium and Node.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for name in ("LICENSE", "NOTICE"):
        src = ROOT / name
        if not src.is_file():
            die(f"{name} is missing -- refusing to build an unlicensed bundle")
        shutil.copy2(src, dest / name)

    gmic = ROOT / "luts" / "gmic" / "LICENSE"
    if gmic.is_file():
        shutil.copy2(gmic, dest / "LICENSE.luts-gmic")

    (dest / "THIRD-PARTY.md").write_text(
        "# Third-party components\n\n"
        "This application bundles the following, each under its own licence.\n"
        "The full text of the ones we redistribute as source or data is beside\n"
        "this file; the rest are named here with their licence.\n\n"
        f"* **CPython {_PBS_PYTHON}** -- Python Software Foundation License 2.0.\n"
        "  Built by astral-sh/python-build-standalone; `runtime/` in this bundle.\n"
        "* **PyTorch** -- BSD 3-Clause.\n"
        "* **NumPy** -- BSD 3-Clause.\n"
        "* **Pillow** -- MIT-CMU (HPND).\n"
        "* **FastAPI / Starlette / Uvicorn / Pydantic** -- MIT / BSD 3-Clause.\n"
        "* **React** -- MIT.\n"
        "* **Electron**, and **Chromium** and **Node.js** within it -- MIT, with\n"
        "  Chromium's own notices in the `LICENSES.chromium.html` that\n"
        "  electron-builder places in the application directory.\n"
        "* **Film emulation LUTs** under `luts/gmic/` -- Pat David's set,\n"
        "  CC BY-SA 4.0. See `LICENSE.luts-gmic`. Note this is share-alike but\n"
        "  **not** non-commercial: selling photographs made with them is fine.\n\n"
        "Film Grain itself is AGPL-3.0-or-later; see `LICENSE` and `NOTICE`.\n",
        encoding="utf-8",
    )
    say(f"licences: {len(list(dest.iterdir()))} file(s)")


# ------------------------------------------------------------------- runtime --
def fetch_runtime(triple: str) -> Path:
    """Download and hash-verify the interpreter archive, caching it."""
    name = f"cpython-{_PBS_PYTHON}+{_PBS_RELEASE}-{triple}-install_only.tar.gz"
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / name

    expected = _PBS_SHA256.get(triple)
    if not expected:
        die(f"no pinned checksum for {triple}")

    def digest(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if archive.is_file():
        if digest(archive) == expected:
            say(f"runtime: cached {name}")
            return archive
        warn(f"cached {name} failed its checksum; re-downloading")
        archive.unlink()

    url = _PBS_BASE + name.replace("+", "%2B")
    say(f"runtime: downloading CPython {_PBS_PYTHON} for {triple}")
    tmp = archive.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "film-grain-build"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    got = digest(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        die(f"checksum mismatch for {name}\n  expected {expected}\n  got      {got}")
    tmp.replace(archive)
    say(f"runtime: verified {du(archive)}")
    return archive


def extract_runtime(archive: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        # The archives contain a single top-level `python/` directory.
        members = tf.getmembers()
        top = {m.name.split("/", 1)[0] for m in members}
        if len(top) != 1:
            die(f"unexpected archive layout: {sorted(top)[:4]}")
        # `filter="data"` refuses absolute paths and traversal; it is the default
        # from 3.14 and explicit here so the behaviour does not depend on which
        # Python runs the build.
        tf.extractall(dest.parent / ".rt-tmp", filter="data")
    (dest.parent / ".rt-tmp" / top.pop()).replace(dest)
    shutil.rmtree(dest.parent / ".rt-tmp", ignore_errors=True)
    say(f"runtime: extracted to {dest.relative_to(ROOT)} ({du(dest)})")


def install_requirements(py: Path, target: Target) -> None:
    """Install the pinned dependency set into the runtime's own site-packages.

    No venv: the runtime *is* the environment. A venv would write absolute paths
    into `pyvenv.cfg` and the console scripts, which is precisely what has to not
    happen in something that gets moved to another machine.

    `--no-deps` throughout, and it is load-bearing rather than an optimisation.
    `base.txt` is the fully resolved set from `Pipfile.lock`, so there is nothing
    left to resolve; and torch comes from a different index for the CUDA targets,
    where letting pip resolve dependencies again could pull a second copy of
    something already pinned.
    """
    base = REQS / "base.txt"
    torch_req = REQS / f"torch-{target.torch}.txt"
    for f in (base, torch_req):
        if not f.is_file():
            die(f"{f.relative_to(ROOT)} is missing. Run: python3 tools/freeze.py")

    # `--check` here rather than trusting the committed files: a bundle built from
    # requirements that no longer match the lock is pinned to nothing in
    # particular.
    if subprocess.run([sys.executable, str(ROOT / "tools" / "freeze.py"), "--check"]
                      ).returncode != 0:
        die("requirements/ is out of date with Pipfile.lock")

    if subprocess.run([str(py), "-m", "pip", "--version"],
                      capture_output=True).returncode != 0:
        say("runtime: bootstrapping pip")
        run([str(py), "-m", "ensurepip", "--upgrade"], capture_output=True)

    common = ["-m", "pip", "install", "--no-deps", "--disable-pip-version-check",
              "--no-warn-script-location", "-q"]
    say(f"runtime: installing base requirements ({sum(1 for l in base.read_text().splitlines() if l and not l.startswith(('#', '-')))} packages)")
    run([str(py), *common, "-r", str(base)])
    say(f"runtime: installing torch ({target.torch})")
    run([str(py), *common, "-r", str(torch_req)])


def prune(runtime: Path) -> None:
    before = du(runtime)
    removed = 0

    site = next(runtime.glob("lib/python3.*/site-packages"), None) \
        or next(runtime.glob("Lib/site-packages"), None)
    stdlib = next(runtime.glob("lib/python3.*"), None) \
        or (runtime / "Lib" if (runtime / "Lib").is_dir() else None)

    if site:
        for rel in _PRUNE_DIRS:
            p = site / rel
            if p.is_dir():
                shutil.rmtree(p); removed += 1
            elif p.is_file():
                p.unlink(); removed += 1
    if stdlib:
        for name in _PRUNE_STDLIB:
            p = stdlib / name
            if p.is_dir():
                shutil.rmtree(p); removed += 1
    for pattern in _PRUNE_GLOBS:
        for p in runtime.glob(pattern):
            if p.is_file():
                p.unlink(); removed += 1

    # Console-script wrappers. pip writes these with an **absolute** shebang
    # naming the interpreter that installed them, so every one of `uvicorn`,
    # `torchrun`, `f2py`, `pip` and friends hardcodes the build machine's path and
    # is already broken the moment the bundle is copied anywhere -- including
    # inside the .app. Nothing here uses them: the shell invokes
    # `runtime/bin/python3 launch.py` directly, and pip is reached as
    # `python3 -m pip`, which needs no wrapper.
    #
    # Detected by the shebang rather than by a name list, so a dependency that
    # starts shipping a new script cannot quietly reintroduce the leak. The
    # interpreter itself is a Mach-O binary and has no shebang, so it survives.
    # `*-config` are build-time helpers that embed the install prefix in their
    # bodies rather than their shebang, so they are named explicitly.
    bindir = next((d for d in (runtime / "bin", runtime / "Scripts")
                   if d.is_dir()), None)
    if bindir:
        major_minor = ".".join(_PBS_PYTHON.split(".")[:2])
        keep = {"python", "python3", f"python{major_minor}",
                f"python{major_minor}.exe", "python.exe", "pythonw.exe"}
        for f in sorted(bindir.iterdir()):
            if f.is_dir() or f.name in keep:
                continue
            try:
                is_script = f.open("rb").read(2) == b"#!"
            except OSError:
                continue
            if is_script or f.name.endswith("-config"):
                f.unlink(); removed += 1
    # Pre-compile is next, so clearing stale bytecode first keeps the count honest.
    for p in runtime.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    say(f"runtime: pruned {removed} item(s), {before} -> {du(runtime)}")


def precompile(py: Path, *dirs: Path) -> None:
    """Ship the .pyc so nothing has to be written at run time.

    The current distribution demonstrates why: `build.sh` strips
    `server/__pycache__`, and then the running app writes it straight back into
    its own install directory. Inside a `.app` bundle that write fails, CPython
    ignores the failure silently, and the user pays full bytecode compilation on
    *every* launch. Shipping the bytecode also makes the bundle read-only-clean,
    which is what lets it live in /Applications.

    Failures are reported, not fatal: .pyc files are a startup optimisation, and
    a stray uncompilable file in a vendored package should not stop a build.
    """
    t0 = time.perf_counter()
    for d in dirs:
        if not d.is_dir():
            continue
        before = du(d)
        proc = subprocess.run(
            # `-s d` strips the staging prefix from the path recorded inside each
            # .pyc. Without it every file carries this machine's absolute build
            # directory: harmless to execution (CPython never resolves
            # `co_filename`) but it ships the builder's folder layout to every
            # user and makes tracebacks point at a path that does not exist on
            # their machine. Recorded relative instead, which is the only honest
            # answer -- where the app ends up is the user's choice.
            [str(py), "-m", "compileall", "-q", "-j", "0", "-s", str(d), str(d)],
            capture_output=True, text=True,
            # PYTHONDONTWRITEBYTECODE stops the *import machinery* from writing
            # bytecode for the modules compileall itself imports on the way up --
            # `threading`, `copyreg`, `contextlib`, `compileall` and friends. Those
            # writes go through the normal import path, which records an absolute
            # `co_filename` and so ignores `-s` entirely; the result was a handful
            # of stdlib .pyc carrying the build machine's path while every file
            # compileall was actually asked to compile was clean. `py_compile`,
            # which compileall uses, writes directly and is unaffected by the flag.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        note = "" if proc.returncode == 0 else "  (some files did not compile)"
        say(f"precompile: {d.name} {before} -> {du(d)}{note}")
        if proc.returncode != 0:
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
            for line in tail:
                warn(line)
    say(f"precompile: {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------------- icon --
def build_icon(dest: Path) -> Path | None:
    """Turn film-grain-1x1.jpg into an .icns using only what macOS ships.

    `sips` and `iconutil` are both in /usr/bin on every macOS install, so this
    adds no build dependency. electron-builder will not accept a .jpg, and the
    source image is 1024x1024, which is exactly the master size -- every slot
    below is a downscale, so nothing is upscaled.
    """
    src = ROOT / "film-grain-1x1.jpg"
    if not src.is_file():
        warn(f"{src.name} not found; the app will use Electron's default icon")
        return None
    if not shutil.which("sips") or not shutil.which("iconutil"):
        warn("sips/iconutil unavailable; skipping icon")
        return None

    iconset = dest / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)

    master = dest / "icon-master.png"
    run(["sips", "-s", "format", "png", str(src), "--out", str(master)],
        capture_output=True)

    for base in (16, 32, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = base * scale
            out = iconset / f"icon_{base}x{base}{suffix}.png"
            run(["sips", "-z", str(px), str(px), str(master), "--out", str(out)],
                capture_output=True)

    icns = dest / "icon.icns"
    run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
        capture_output=True)
    shutil.rmtree(iconset, ignore_errors=True)
    master.unlink(missing_ok=True)
    say(f"icon: {icns.name} from {src.name} ({du(icns)})")
    return icns


# -------------------------------------------------------------------- verify --
def write_version(dest: Path) -> str:
    pkg = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    version = pkg.get("version", "0.0.0")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    (dest / "VERSION").write_text(
        f"film-grain {version}\nbuilt {stamp}\n", encoding="utf-8")
    return version


def verify(py: Path, payload: Path, counts: dict[str, int]) -> None:
    """Prove the bundle works before calling the build a success.

    Three separate questions, because they fail in three different ways:

    1. Does it render? `--selftest` imports torch, builds the schema, parses every
       LUT and renders one frame through the real engine.
    2. Did everything get *copied*? The selftest can only compare the bundle
       against itself, so the counts are checked against the source tree here.
       This is the check that would have caught 7-of-303.
    3. Is it actually relocatable? Anything containing this machine's absolute
       build path will break on another machine.
    """
    # --- 2: counts, against the source tree ---
    src_presets = len(list((ROOT / "presets").glob("*.json")))
    src_luts = len(list((ROOT / "luts").rglob("*.cube")))
    if counts["presets"] != src_presets:
        die(f"payload has {counts['presets']} presets, source has {src_presets}")
    if counts["luts"] != src_luts:
        die(f"payload has {counts['luts']} LUTs, source has {src_luts}")
    say(f"verify: counts match source ({src_presets} presets, {src_luts} LUTs)")

    # --- 1: does it render ---
    # Run from a directory that is not the payload, to prove nothing depends on
    # the cwd; and with a scrubbed environment, so no FILM_GRAIN_* left over in
    # this shell can make a broken bundle look fine.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FILM_GRAIN_", "PYTHON")) and k != "APP_ENV"}
    # The shipped app lives somewhere it may not be able to write to, so verify it
    # the same way -- and this also stops the selftest itself from creating
    # absolute-path bytecode that the relocatability scan would then flag.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    say("verify: running selftest inside the bundle")
    proc = subprocess.run(
        [str(py), str(payload / "launch.py"), "--selftest"],
        cwd=str(ROOT.parent), env=env, capture_output=True, text=True,
    )
    for line in (proc.stdout or "").strip().splitlines():
        print(f"    {line}")
    if proc.returncode != 0:
        for line in (proc.stderr or "").strip().splitlines()[-15:]:
            print(f"    {line}", file=sys.stderr)
        die("the bundled runtime failed its selftest")

    # --- 3: no build-machine paths ---
    # Only the trees that actually get packaged. Scanning the whole staging
    # parent picked up the previous build system's leftovers and called them
    # leaks; a check that cries wolf is worse than no check.
    root = payload.parent
    shipped = [d for d in (payload, root / "runtime", root / "licenses")
               if d.is_dir()]
    needle = str(ROOT).encode()
    leaks: list[Path] = []
    scanned = 0
    for tree in shipped:
        for p in tree.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            scanned += 1
            try:
                if needle in p.read_bytes():
                    leaks.append(p)
            except OSError:
                continue
    if leaks:
        warn(f"{len(leaks)} of {scanned} shipped file(s) contain this machine's "
             f"build path ({ROOT}):")
        for p in leaks[:10]:
            warn(f"  {p.relative_to(root)}")
        die("the bundle is not relocatable")
    say(f"verify: no build-machine paths in {scanned} shipped files")


# ---------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS),
                    help="which platform to build for")
    ap.add_argument("--out", default=str(BUILD),
                    help="staging directory (default: build/bundle/)")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="reuse an already-installed runtime/ (fast iteration)")
    args = ap.parse_args(argv)

    target = TARGETS[args.target]
    if not target.supported:
        print(f"\033[1m{args.target}\033[0m is not supported yet.\n\n"
              f"  {target.note}.\n\n"
              "Supported targets: "
              + ", ".join(t for t, v in TARGETS.items() if v.supported)
              + "\n", file=sys.stderr)
        return 2

    if args.target == "mac" and sys.platform != "darwin":
        die("--target mac has to be built on macOS: the wheels and the .icns "
            "are both native")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = out / "payload"
    runtime = out / "runtime"

    t0 = time.perf_counter()
    counts = assemble_payload(payload)
    assemble_licenses(out / "licenses")
    version = write_version(out)

    if args.skip_runtime and runtime.is_dir():
        say(f"runtime: reusing {runtime.relative_to(ROOT)} (--skip-runtime)")
    else:
        extract_runtime(fetch_runtime(target.triple), runtime)
        install_requirements(runtime / target.python, target)
        prune(runtime)

    # Outside the branch on purpose: the payload is reassembled on *every* run, so
    # compiling it only when the runtime is rebuilt would ship a payload with no
    # bytecode whenever --skip-runtime was used. Recompiling an already-compiled
    # runtime is a no-op that costs seconds.
    precompile(runtime / target.python, runtime, payload)

    if args.target == "mac":
        build_icon(out)

    verify(runtime / target.python, payload, counts)

    say(f"bundle {version} ready in {out.relative_to(ROOT)} "
        f"({du(out)}, {time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
