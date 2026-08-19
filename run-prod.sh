#!/usr/bin/env bash
# Build the server bundle and run it -- the actual artifact, run the way a copy
# handed to someone else would be, rather than the source tree.
#
# Unlike ./run.sh (which serves web/dist out of the source tree via pipenv), this
# rebuilds first, so what runs is guaranteed to match the client source as it
# stands right now.
#
# It no longer has to find an interpreter. The old build/run.sh searched
# $FILM_GRAIN_PYTHON, then build/.venv, then python3 on PATH, and this script
# existed largely to point it at the project's .venv so a second copy of torch did
# not have to be downloaded. The bundle carries its own Python, so all of that is
# gone -- which is the whole point of the change.
#
#   ./run-prod.sh                rebuild and serve on :8000
#   PORT=9000 ./run-prod.sh      another port
#   HOST=0.0.0.0 ./run-prod.sh   expose on the network (no auth -- see README)
#   ./run-prod.sh --no-browser   do not open a browser
#
# For the desktop app rather than the bare server, use ./build.sh and open
# dist/mac-arm64/Film Grain.app.
set -euo pipefail
cd "$(dirname "$0")"

# The runtime is ~730MB and changes only when Pipfile.lock does, so reuse it if it
# is already there. The client and the payload are rebuilt every time, which is
# the reason to run this script at all.
ARGS=(--no-electron)
[ -d build/bundle/runtime ] && ARGS+=(--skip-runtime)
./build.sh "${ARGS[@]}"

PY="build/bundle/runtime/bin/python3"
[ -x "$PY" ] || { echo "no bundled interpreter at $PY" >&2; exit 1; }

exec "$PY" build/bundle/payload/launch.py "$@"
