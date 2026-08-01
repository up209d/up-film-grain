#!/usr/bin/env bash
# Dev mode: uvicorn with reload on :8000, Vite dev server on :5173.
# Vite proxies /api to the backend, so open http://localhost:5173
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# Check the port before starting anything. Otherwise uvicorn dies on bind but
# Vite comes up regardless, leaving a UI whose every request fails.
if holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null) && [ -n "$holder" ]; then
  echo "Port $PORT is already in use by PID(s): $(echo "$holder" | tr '\n' ' ')" >&2
  ps -o pid=,command= -p $holder 2>/dev/null >&2 || true
  echo >&2
  echo "Free it with:  kill $(echo "$holder" | tr '\n' ' ')" >&2
  echo "Or use another port:  PORT=8001 ./dev.sh" >&2
  exit 1
fi

export PIPENV_VENV_IN_PROJECT=1
# Production is the default in server/main.py, so dev has to ask for itself.
# This is what opens CORS for Vite's origin and enables /docs.
export APP_ENV=development
# Homebrew and Laravel Herd sit ahead of nvm in PATH on this machine, so
# `nvm use` alone does not switch node. Prepend the pinned version explicitly.
NVM_BIN="$HOME/.nvm/versions/node/v24.15.0/bin"
[ -d "$NVM_BIN" ] && export PATH="$NVM_BIN:$PATH"

pipenv run uvicorn server.main:app --host 127.0.0.1 --port "$PORT" --reload &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT

# Fail loudly if the backend dies during startup instead of leaving Vite
# running on its own.
sleep 3
if ! kill -0 $BACK 2>/dev/null; then
  echo "Backend failed to start -- see the uvicorn output above." >&2
  exit 1
fi

cd web && npm run dev
