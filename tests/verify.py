"""Engine invariant checks.

    pipenv run python tests/verify.py                 everything, in parallel
    pipenv run python tests/verify.py edges halation  only those modules
    pipenv run python tests/verify.py global          every module matching
    pipenv run python tests/verify.py -l              list the modules
    pipenv run python tests/verify.py -j 1            one process, for a traceback
    pipenv run python tests/verify.py --times         per-module seconds

These are the properties that, if broken, produce bugs you will not see in a
preview -- seams that only appear in the export, a preview that stops predicting
the output, colour drift in a build that is meant to be a colour pass-through.
Run this after touching anything under `server/engine/`.

The checks themselves live in `tests/checks/*.py`, one module per area, and
`tests/runner.py` selects and schedules them. Run the modules covering what you
touched while you work; run the lot before you call it done.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
