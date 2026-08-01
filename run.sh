#!/usr/bin/env bash
# Run production mode from the source tree: FastAPI serves the built client.
#
# For a distributable copy instead, use ./build.sh and run build/run.sh.
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

if holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null) && [ -n "$holder" ]; then
  echo "Port $PORT is already in use by PID(s): $(echo "$holder" | tr '\n' ' ')" >&2
  ps -o pid=,command= -p $holder 2>/dev/null >&2 || true
  echo >&2
  echo "Free it with:  kill $(echo "$holder" | tr '\n' ' ')" >&2
  echo "Or use another port:  PORT=8001 ./run.sh" >&2
  exit 1
fi

if [ ! -f web/dist/index.html ]; then
  echo "No compiled client at web/dist/. Build it first:" >&2
  echo "  ./build.sh          (or: cd web && npm run build)" >&2
  exit 1
fi

export PIPENV_VENV_IN_PROJECT=1
export APP_ENV=production
exec pipenv run uvicorn server.main:app --host 127.0.0.1 --port "$PORT" "$@"
