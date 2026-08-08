"""Process-wide singletons: the environment flag, the device, the engine, and
the two locks every render goes through.

Its own module so the controllers can import it without importing each other,
and so ``IS_DEV`` is decided in exactly one place. Production is the default --
dev mode is the special case, because it is the one that needs CORS holes and an
interactive schema browser, so it has to be asked for rather than assumed or a
distribution ships with both.
"""

from __future__ import annotations

import os
import threading

from .engine import GrainEngine, pick_device

IS_DEV = os.environ.get("APP_ENV", "production").lower() in ("dev", "development")

DEVICE = pick_device()
ENGINE = GrainEngine(DEVICE)

# Renders are serialised: concurrent tensor work on one GPU just thrashes, and
# the UI only ever needs the newest preview anyway.
RENDER_LOCK = threading.Lock()

# Monotonic counter identifying the newest requested preview.
#
# The lock above serialises renders, but it does not make a *stale* one stop.
# Starlette cannot interrupt a threadpool worker, so when the client aborts --
# which `App.tsx` does on every new render -- the abandoned render used to run to
# completion and keep holding the lock the whole time, and the preview the user
# actually wants would queue behind every superseded one. Latency then grows with
# how many edits happened rather than with how long one render takes, which is
# the difference between usable and not on a slow machine.
#
# So each preview takes a ticket on entry and gives up as soon as a newer ticket
# exists. Guarded by its own lock rather than relying on the GIL, because the
# read-modify-write has to be atomic against other threadpool workers.
_GEN_LOCK = threading.Lock()
_preview_gen = 0


def next_preview_gen() -> int:
    """Take a ticket for a preview about to start."""
    global _preview_gen
    with _GEN_LOCK:
        _preview_gen += 1
        return _preview_gen


def is_superseded(ticket: int) -> bool:
    """True once a newer preview has taken a ticket.

    A function rather than an exported counter on purpose: `from .runtime import
    _preview_gen` would snapshot the value at import and every render would think
    it was the newest one forever.
    """
    return _preview_gen != ticket
