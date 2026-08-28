#!/usr/bin/env bash
# Render one photograph from the command line -- no server, no browser.
#
#   ./export.sh photo.jpg -p KodakPortra
#   ./export.sh photo.jpg -p Stock -m full -o out.png
#   ./export.sh --list-presets
#
# Everything after the script name goes straight to `server/cli.py`; see it, or
# `./export.sh -h`, for the four things it asks: input, preset, output, mode.
set -e
cd "$(dirname "$0")"

# No port check and no web/dist check, unlike run.sh -- the CLI imports the
# domain directly and never starts a server or serves the client.
export PIPENV_VENV_IN_PROJECT=1
export APP_ENV=production
exec pipenv run python -m server.cli "$@"
