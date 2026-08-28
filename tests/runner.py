"""Selects, runs and reports the check modules. `verify.py` is the CLI onto it.

Two things make this faster than the single function it replaced.

**Selection.** Naming modules runs only those. That is the whole point of the
split: after touching one stage you re-run the two modules that cover it in
seconds instead of the suite in minutes.

**Processes.** The suite is a chain of GPU renders with numpy metrics between
them, and it measured 83% of a single core over 4m24s -- thirteen of fourteen
cores idle. Each worker builds its own `Ctx` and runs whole modules, so nothing
is shared and nothing needs locking. Threads would not do: the metrics are numpy
and torch calls that hold the GIL for long stretches.

`spawn`, not `fork`: a forked child inherits a half-initialised MPS context and
deadlocks the first time it touches the GPU.

Output is buffered per module and printed in `ORDER`, so a parallel log is
byte-comparable with a sequential one and both are comparable with the log from
before the split.
"""

from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import harness  # noqa: E402
from tests.checks import ORDER  # noqa: E402
from tests.harness import SUITES  # noqa: E402

# Wall-clock seconds per module, measured on an M4 Max / MPS. Used only to
# schedule the long ones first, so the pool does not finish with one worker
# still grinding through `edges` while the rest sit idle. Being stale costs a
# little scheduling, never correctness -- refresh from the `--times` output.
COST = {
    "edges": 32.4, "global_layers": 30.3, "film_tiling": 28.9,
    "global_field": 24.2, "global_grain": 22.0, "scatter": 20.5,
    "internals": 19.6, "global_mix": 19.5, "film_texture": 18.5,
    "grading": 15.4, "tiling": 14.0, "sharpen": 9.9, "halation": 4.9,
    "normalize": 3.6, "prescale": 1.5,
    "response": 2.5, "colour": 0.3, "presets": 0.2, "imageio": 0.0,
}

_CTX: harness.Ctx | None = None


def _ctx() -> harness.Ctx:
    global _CTX
    if _CTX is None:
        _CTX = harness.Ctx()
    return _CTX


def run_one(name: str) -> tuple[str, str, list[str], float]:
    """Run one module in this process. Top-level so `spawn` can pickle it."""
    harness.FAILURES.clear()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        SUITES[name].fn(_ctx())
    return name, buf.getvalue(), list(harness.FAILURES), time.perf_counter() - t0


def _worker_init(threads: int) -> None:
    import torch
    torch.set_num_threads(max(1, threads))


def select(names: list[str], grep: str | None) -> list[str]:
    if not names:
        chosen = list(ORDER)
    else:
        chosen, unknown = [], []
        for n in names:
            if n in SUITES:
                chosen.append(n)
            else:
                hits = [k for k in ORDER if n in k]
                chosen.extend(hits) if hits else unknown.append(n)
        if unknown:
            raise SystemExit(
                f"unknown module(s): {', '.join(unknown)}\n"
                f"available: {', '.join(ORDER)}"
            )
    if grep:
        chosen = [n for n in chosen if grep in n or grep in SUITES[n].title]
    seen: set[str] = set()
    return [n for n in chosen if not (n in seen or seen.add(n))]


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="verify.py",
        description="Engine invariant checks. With no arguments, runs all of them.",
    )
    ap.add_argument("modules", nargs="*", help="modules to run (substrings are matched)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes; 0 picks one per module up to the core "
                         "count, 1 runs in this process")
    ap.add_argument("-k", type=str, default=None, metavar="TEXT",
                    help="only modules whose name or title contains TEXT")
    ap.add_argument("-l", "--list", action="store_true", help="list modules and exit")
    ap.add_argument("--times", action="store_true", help="print per-module seconds")
    a = ap.parse_args(argv)

    if a.list:
        for n in ORDER:
            print(f"  {n:<14} {SUITES[n].title}")
        return 0

    chosen = select(a.modules, a.k)
    if not chosen:
        print("nothing selected")
        return 1

    cores = os.cpu_count() or 4
    jobs = a.jobs if a.jobs > 0 else min(len(chosen), max(1, cores - 2))

    from server.engine import device_name, pick_device
    print(f"device: {device_name(pick_device())}")

    t0 = time.perf_counter()
    results: dict[str, tuple[str, list[str], float]] = {}

    if jobs == 1:
        print()
        for n in chosen:
            name, out, fails, secs = run_one(n)
            print(out, end="")
            results[name] = (out, fails, secs)
    else:
        # Longest first: the pool's wall-clock is its critical path, and
        # starting `edges` last would set that path all on its own.
        order = sorted(chosen, key=lambda n: -COST.get(n, 10.0))
        print(f"{len(chosen)} modules across {jobs} processes\n", flush=True)
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=get_context("spawn"),
            initializer=_worker_init, initargs=(max(1, cores // jobs),),
        ) as ex:
            for name, out, fails, secs in ex.map(run_one, order):
                results[name] = (out, fails, secs)
        for n in chosen:
            print(results[n][0], end="")

    wall = time.perf_counter() - t0
    failures = [f for n in chosen for f in results[n][1]]
    total = sum(x.count("[PASS]") + x.count("[FAIL]") for x, _, _ in results.values())

    if a.times:
        print("\nper-module seconds")
        for n in sorted(chosen, key=lambda k: -results[k][2]):
            print(f"  {n:<14} {results[n][2]:6.1f}s")

    n = len(chosen)
    print(f"\n{total} checks in {n} module{'s' if n != 1 else ''}, {wall:.1f}s wall")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all invariants hold")
    return 0
