#!/usr/bin/env bash
# Build a fresh distribution and run it -- the actual build/ artifact, run the
# same way a copy handed to someone else would be, not the source tree.
#
# Unlike ./run.sh (which serves web/dist straight out of the source tree via
# pipenv), this always rebuilds first, so what ends up running is guaranteed
# to match the client source as it stands right now.
#
#   ./run-prod.sh              rebuild build/ and serve on :8000
#   PORT=9000 ./run-prod.sh    another port
#   HOST=0.0.0.0 ./run-prod.sh expose on the network (no auth -- see README)
set -euo pipefail
cd "$(dirname "$0")"

./build.sh

# build/run.sh resolves an interpreter in this order: $FILM_GRAIN_PYTHON, then
# build/.venv, then python3 on PATH. Plain ./build.sh (no --venv) leaves
# build/.venv missing, and building one from scratch just to run it is a large,
# unnecessary torch download -- so point it at this project's own environment
# instead, unless the caller already set one or a real build/.venv exists.
if [ -z "${FILM_GRAIN_PYTHON:-}" ] && [ ! -x "build/.venv/bin/python" ] && [ -x ".venv/bin/python" ]; then
  export FILM_GRAIN_PYTHON="$(pwd)/.venv/bin/python"
fi

exec ./build/run.sh "$@"
