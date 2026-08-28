#!/usr/bin/env python3
"""Start the Film Grain server, and be startable by something that is not bash.

This is the entrypoint the desktop app spawns, and the only one that works on a
machine with no shell scripting. Before it existed there was **no** way to start
this server programmatically: `server/` contains no `uvicorn.run` call and no
`if __name__ == "__main__"` anywhere, so every launcher was a bash script calling
the uvicorn CLI, using `lsof`/`ps`/`kill` to check the port and `cd` to put the
project on `sys.path`.

What it replaces, and why each one mattered:

* **`cd "$(dirname "$0")"`.** Every previous launcher depended on the working
  directory to make `import server.main` resolve. A double-clicked app or a
  Windows shortcut does not promise you any particular cwd, so the path is set
  from `__file__` here instead.
* **`lsof -nP -iTCP:$PORT`.** Not present on Windows, and a subprocess either
  way. The port is claimed by binding a socket -- and the socket is then handed
  to uvicorn, which closes the race that checking-then-binding always has.
* **`HOST`/`PORT`.** These existed only in the shell. Nothing in Python read
  them, and `run.sh` hardcoded `127.0.0.1` while documenting `HOST`.

Run it with no arguments to serve on the first free port from 8000 and open a
browser. `--selftest` renders one frame and exits, which is how a build proves a
bundle works.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Before anything imports `server`: this file sits beside the `server/` package
# in the source tree and beside it again inside the bundle, so its own directory
# is the import root in both. Nothing here may depend on the cwd.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

# How many consecutive ports to try before giving up. Only reached when a port
# was asked for explicitly and is taken; `--port 0` lets the OS choose and never
# needs this.
_PORT_TRIES = 20

# Printed on stdout, once, as soon as the socket is bound and before the slow
# import of torch. The desktop shell reads this rather than assuming the port it
# asked for was the port it got -- which is the whole point of binding here.
_URL_SENTINEL = "FILM_GRAIN_URL"


# --------------------------------------------------------------------- port --
def bind_socket(host: str, port: int) -> socket.socket:
    """Claim a listening address and return the bound socket.

    Bound but **not** listening: uvicorn is handed this socket and asyncio's
    `create_server` calls `listen()` itself, matching what uvicorn's own
    `config.bind_socket()` produces.

    Binding rather than probing is deliberate. Every previous launcher asked
    `lsof` whether the port was free and then told uvicorn to bind it, which is a
    race with any other process doing the same thing -- and, more practically, a
    race with the *previous* copy of this app still shutting down. Holding the
    socket from here means the port cannot be taken between the check and the
    bind, because there is no check.
    """
    candidates = [0] if port == 0 else range(port, port + _PORT_TRIES)
    last: OSError | None = None
    for candidate in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError as e:
            sock.close()
            last = e
            continue
        if port and candidate != port:
            print(f"[launch] port {port} is in use; using {candidate} instead",
                  file=sys.stderr, flush=True)
        return sock
    raise SystemExit(
        f"[launch] no free port in {port}..{port + _PORT_TRIES - 1} on {host}"
        f" ({last})"
    )


# ------------------------------------------------------------------ watchdog --
def _parent_alive(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            WAIT_TIMEOUT = 0x00000102
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                return False
            try:
                # Signalled means the process object is done, i.e. it exited.
                # Still-running shows up as a timeout on a zero-length wait.
                return (ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
                        == WAIT_TIMEOUT)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            # Cannot tell -- assume alive. Guessing "dead" here would kill a
            # perfectly good server.
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else. Alive as far as we care.
        return True
    except OSError:
        return True
    return True


def watch_parent(pid: int, interval: float = 2.0) -> None:
    """Exit when the process that started us goes away.

    Without this, killing or crashing the desktop shell orphans a Python process
    holding torch, the model weights and a multi-gigabyte allocator pool, with no
    window and no obvious way for the user to find it. The shell kills its child
    on a clean quit; this is what covers the unclean ones.

    `os._exit` rather than `sys.exit`: there is no parent left to report to, and a
    graceful teardown here would mean waiting on uvicorn's shutdown and torch's
    allocator while the thing we are shutting down for is already gone.
    """
    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                alive = _parent_alive(pid)
            except Exception:
                # Could not tell. Try again rather than killing a healthy server
                # on the strength of one failed probe.
                continue
            if alive:
                continue
            # **Nothing between here and `os._exit` may be allowed to raise.**
            # Reporting the shutdown used to come first and unguarded, and that
            # was a real orphan bug rather than a theoretical one: the parent
            # holds the other end of this stderr pipe, so the moment it dies this
            # write raises BrokenPipeError, which propagated out of the thread and
            # killed the watchdog one line before it could exit -- leaving exactly
            # the orphaned multi-gigabyte process it exists to prevent. Measured:
            # with stderr on a pipe the child survived indefinitely; with stderr
            # on a file it exited in 2s. The message is best-effort; the exit is not.
            try:
                print(f"[launch] parent {pid} exited; shutting down",
                      file=sys.stderr, flush=True)
            except Exception:
                pass
            # `os._exit` runs no `atexit` handler, deliberately (see above), so
            # the disk cache has to be removed by hand here or an unclean shell
            # exit leaves a full run's spill behind. Guarded like the print
            # above it, for the same reason: nothing between here and the exit
            # may be allowed to raise.
            try:
                cache_cleanup()
            except Exception:
                pass
            os._exit(0)

    threading.Thread(target=loop, name="parent-watchdog", daemon=True).start()


def cache_cleanup() -> None:
    """Delete this run's disk cache, if there is one. Never raises."""
    try:
        from server.engine import diskcache

        diskcache.cleanup()
    except Exception:
        # Import can fail if we are shutting down before `server` was reachable,
        # and a failed tidy-up must never turn a clean exit into a traceback.
        pass


def install_cache_cleanup() -> None:
    """Delete the disk cache on the way out, including on a signal.

    **The signal path is the desktop app's normal quit**, not an edge case:
    `electron/main.js` sends SIGTERM, and Python does not run `atexit` handlers
    for a signal, so without this every ordinary quit left a full run's spill --
    up to the disk budget plus the open photograph's frames -- lying in
    `~/Library/Caches` until the next start swept it up.

    It works because of a specific uvicorn behaviour worth writing down, since
    it looks like it should not: uvicorn installs its own SIGINT/SIGTERM
    handlers for the duration of `Server.run`, and on the way out it **restores
    whatever was there before and then re-raises the captured signal**
    (`Server.capture_signals`). So a handler installed *before* `run()` is not
    shadowed -- it is deferred, and fires after the graceful shutdown has
    finished. Installing one *after* `run()` returns would be too late, and
    replacing uvicorn's would break the graceful shutdown outright.

    `signal.SIG_DFL` is restored before exiting so a second SIGTERM during
    cleanup kills the process outright rather than re-entering this.
    """
    def handler(signum, _frame):
        cache_cleanup()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not the main thread, or a platform without it. The next start's
            # sweep is the backstop.
            pass


# ------------------------------------------------------------------ selftest --
def selftest() -> int:
    """Prove this bundle can actually render, and report what it found.

    Deliberately reports **counts**, not just success. The two worst failures
    this project has shipped were both silent shortfalls rather than errors: a
    distribution with 7 of 303 LUTs because the build flattened a directory tree,
    and presets vanishing one platform at a time because a decode error was
    caught and logged to a stream nobody reads. Both pass any did-it-boot check.

    The counts are compared against **what is on disk in this bundle**, not
    against numbers written down here. A hardcoded 303 would start failing the
    day someone adds a LUT. "Everything present parsed" is the invariant that
    stays true; whether the right number of files was *copied in* is a question
    only the build can answer, and `tools/bundle.py` answers it there.
    """
    import numpy as np

    from server import lut as lutlib
    from server import params as P
    from server.engine import device_name
    from server.runtime import DEVICE, ENGINE

    ok = True

    print(f"device        : {device_name(DEVICE)}")

    # --- schema and presets ---
    sch = P.schema()
    presets = sch["presets"]
    on_disk = sorted(P.PRESET_DIR.glob("*.json")) if P.PRESET_DIR.is_dir() else []
    print(f"params        : {len(sch['params'])}")
    print(f"presets       : {len(presets)} loaded / {len(on_disk)} on disk"
          f"  (default: {sch['default_preset']})")
    if len(presets) != len(on_disk):
        loaded = {p["name"] for p in presets}
        missing = [f.stem for f in on_disk if f.stem not in loaded]
        print(f"FAIL: {len(missing)} preset(s) on disk did not load: "
              f"{', '.join(missing)}", file=sys.stderr)
        ok = False
    if not presets:
        print("FAIL: no presets at all", file=sys.stderr)
        ok = False

    # --- LUTs ---
    luts = lutlib.list_luts()
    cubes = (sorted(lutlib.LUT_DIR.rglob("*.cube"))
             if lutlib.LUT_DIR.is_dir() else [])
    print(f"luts          : {len(luts)} listed / {len(cubes)} .cube on disk")
    if len(luts) != len(cubes):
        print(f"FAIL: {len(cubes) - len(luts)} .cube file(s) on disk are not "
              "listed", file=sys.stderr)
        ok = False
    # Listing only walks the tree; parsing is what actually reads them, and a
    # LUT that lists but will not parse grades nothing at render time. All 303
    # measure 0.7s together because `get` caches, so there is no reason to sample.
    unparsed = [e["id"] for e in luts if lutlib.get(e["id"]) is None]
    if unparsed:
        print(f"FAIL: {len(unparsed)} LUT(s) failed to parse: "
              f"{', '.join(unparsed[:5])}", file=sys.stderr)
        ok = False
    elif luts:
        print(f"luts parsed   : {len(luts)}/{len(luts)}")

    # --- one real render ---
    # A *preset*, not the defaults: `render_image` hands the input straight back
    # when `is_neutral(p)` is true, and everything in the schema ships neutral.
    # Rendering the defaults would exercise nothing and pass.
    name = sch["default_preset"] or (presets[0]["name"] if presets else None)
    chosen = next((p for p in presets if p["name"] == name), None)
    if chosen is None:
        print("FAIL: no preset to render with", file=sys.stderr)
        return 1

    p = P.sanitize(chosen.get("values") or {})
    if P.is_neutral(p):
        print(f"FAIL: preset {name!r} is neutral; the render would be a no-op",
              file=sys.stderr)
        ok = False

    # A gradient rather than flat grey: several stages key off luminance and
    # edges, so a constant frame would leave them untested.
    h, w = 96, 128
    ramp = np.linspace(0.05, 0.95, w, dtype=np.float32)
    arr = np.repeat(ramp[None, :], h, axis=0)[:, :, None].repeat(3, axis=2)
    arr[h // 3:2 * h // 3, w // 3:2 * w // 3] = 0.9  # a hard edge to bite on

    t0 = time.perf_counter()
    out = ENGINE.render_image(arr, p, 1.0, tile=1024, supersample=2.0)
    dt = time.perf_counter() - t0

    if out.shape != arr.shape:
        print(f"FAIL: render changed shape {arr.shape} -> {out.shape}",
              file=sys.stderr)
        ok = False
    elif not np.isfinite(out).all():
        print("FAIL: render produced non-finite values", file=sys.stderr)
        ok = False
    elif out.min() < -1e-3 or out.max() > 1 + 1e-3:
        # The engine clamps to 0..1 before the encoders see it, so out-of-range
        # here means an op misbehaved -- which is exactly the shape a wrong wheel
        # or a broken backend takes, rather than an exception.
        print(f"FAIL: render left 0..1: [{out.min():.3f}, {out.max():.3f}]",
              file=sys.stderr)
        ok = False
    elif float(np.abs(out - arr).max()) == 0.0:
        print(f"FAIL: render with preset {name!r} changed nothing",
              file=sys.stderr)
        ok = False
    else:
        print(f"render        : {name} on {w}x{h} in {dt:.2f}s, "
              f"max delta {float(np.abs(out - arr).max()):.3f}")

    print("selftest      : " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="launch.py", description="Serve the Film Grain app.",
    )
    ap.add_argument("--host", default=os.environ.get("HOST") or DEFAULT_HOST)
    ap.add_argument(
        "--port", type=int,
        default=int(os.environ.get("PORT") or DEFAULT_PORT),
        help="0 lets the OS pick. A taken port steps to the next free one.",
    )
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="exit if this process disappears")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser once serving")
    ap.add_argument("--selftest", action="store_true",
                    help="render one frame, report what loaded, and exit")
    args = ap.parse_args(argv)

    # Production unless something has deliberately said otherwise. `IS_DEV`
    # gates CORS, /docs and the missing-client behaviour, so defaulting the
    # other way would ship all three.
    os.environ.setdefault("APP_ENV", "production")

    if args.parent_pid:
        watch_parent(args.parent_pid)

    if args.selftest:
        return selftest()

    # Bound before importing the app: the import pulls in torch and builds the
    # engine, which takes seconds, and the shell wants the URL as early as it can
    # get it so it can start polling.
    sock = bind_socket(args.host, args.port)
    host, port = sock.getsockname()[:2]
    url = f"http://{host}:{port}"
    print(f"{_URL_SENTINEL} {url}", flush=True)

    import uvicorn

    from server.engine import device_name
    from server.runtime import DEVICE

    print(f"[launch] device: {device_name(DEVICE)}", flush=True)

    if not args.no_browser:
        import webbrowser

        # After the server is actually accepting, not before -- the socket is
        # bound already, so a connection will queue rather than be refused, but
        # a page that loads before the app is importable shows an error instead
        # of the app.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    install_cache_cleanup()

    server = uvicorn.Server(uvicorn.Config(
        "server.main:app", log_level="info",
        # The socket carries the address; passing host/port as well would be a
        # second source of truth for the same thing.
    ))
    server.run(sockets=[sock])
    # Reached only on a shutdown that was *not* signalled -- a signalled one is
    # re-raised inside `run()` and lands on the handler installed above. Both
    # roads have to clean up, and `cleanup` is idempotent so saying so twice
    # costs nothing.
    cache_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
