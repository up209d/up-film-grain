#!/usr/bin/env bash
# Compile a production distribution into build/.
#
# The output is self-contained apart from the Python environment: server code,
# the compiled client, a frozen requirements.txt and a launcher. Copy build/
# anywhere and run its own ./run.sh.
#
#   ./build.sh            assemble the distribution
#   ./build.sh --venv     also create build/.venv and install into it (slow --
#                         torch is a large download)
#   ./build.sh --clean    remove build/ first
set -euo pipefail
cd "$(dirname "$0")"

OUT="build"
WITH_VENV=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --venv)  WITH_VENV=1 ;;
    --clean) CLEAN=1 ;;
    --out=*) OUT="${arg#--out=}" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

[ "$CLEAN" = 1 ] && { say "Removing $OUT/"; rm -rf "${OUT:?}"; }

# ---------------------------------------------------------------- client --
# Homebrew and Laravel Herd sit ahead of nvm in PATH on this machine, so
# `nvm use` alone does not switch node. Prepend the pinned version explicitly.
NVM_BIN="$HOME/.nvm/versions/node/v24.15.0/bin"
[ -d "$NVM_BIN" ] && export PATH="$NVM_BIN:$PATH"

command -v node >/dev/null || { echo "node not found on PATH" >&2; exit 1; }
say "Building client with node $(node --version)"

cd web
[ -d node_modules ] || {
  say "Installing client dependencies"
  # npm blocks postinstall scripts by default here, and vite will not build
  # without esbuild's.
  npm install
  npm approve-scripts esbuild 2>/dev/null || true
}
npm run build
cd ..

[ -f web/dist/index.html ] || { echo "Client build produced no index.html" >&2; exit 1; }

# ---------------------------------------------------------------- assemble --
say "Assembling $OUT/"
rm -rf "${OUT:?}/server" "${OUT:?}/web" "${OUT:?}/presets"
mkdir -p "$OUT/web"

# Server package only -- no tests, no dev scripts, no client sources.
cp -R server "$OUT/server"
find "$OUT/server" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
cp -R web/dist "$OUT/web/dist"

# Presets are data the server reads at runtime, so they have to travel with it.
# The directory is created even when empty: the server tolerates a missing one,
# but shipping the folder is what makes it obvious where presets go.
mkdir -p "$OUT/presets"
if compgen -G "presets/*.json" > /dev/null; then
  cp presets/*.json "$OUT/presets/"
  say "Bundled $(ls -1 presets/*.json | wc -l | tr -d ' ') preset(s)"
else
  say "No presets to bundle (presets/ is empty)"
fi

# Freeze dependencies from the lock file so the distribution pins exactly what
# was tested, rather than re-resolving on the target.
say "Freezing dependencies"
export PIPENV_VENV_IN_PROJECT=1
if pipenv requirements > "$OUT/requirements.txt" 2>/dev/null; then
  :
else
  # Older pipenv spells it differently; fall back rather than shipping nothing.
  pipenv lock -r > "$OUT/requirements.txt"
fi

python3 - "$OUT" <<'PY'
import json, pathlib, sys, datetime
out = pathlib.Path(sys.argv[1])
pkg = json.loads(pathlib.Path("web/package.json").read_text())
(out / "VERSION").write_text(
    f"film-grain {pkg.get('version', '0.0.0')}\n"
    f"built {datetime.datetime.now().isoformat(timespec='seconds')}\n"
)
PY

cat > "$OUT/run.sh" <<'LAUNCH'
#!/usr/bin/env bash
# Production launcher for a built distribution.
#
#   ./run.sh              serve on 127.0.0.1:8000
#   PORT=9000 ./run.sh    another port
#   HOST=0.0.0.0 ./run.sh expose on the network (no auth -- see README)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null) && [ -n "$holder" ]; then
  echo "Port $PORT is already in use by PID(s): $(echo "$holder" | tr '\n' ' ')" >&2
  ps -o pid=,command= -p $holder 2>/dev/null >&2 || true
  echo "Free it with:  kill $(echo "$holder" | tr '\n' ' ')" >&2
  echo "Or use another port:  PORT=8001 ./run.sh" >&2
  exit 1
fi

# Interpreter: an explicit one, else a venv beside this script, else whatever
# python3 is on PATH -- which only works if the deps are installed there.
if [ -n "${FILM_GRAIN_PYTHON:-}" ]; then
  PY="$FILM_GRAIN_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

[ -n "$PY" ] || { echo "No python3 found." >&2; exit 1; }

if ! "$PY" -c "import fastapi, torch, uvicorn" 2>/dev/null; then
  cat >&2 <<MSG
Dependencies are missing from: $PY

Create an environment for this distribution:
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt

Or point at an existing one:
  FILM_GRAIN_PYTHON=/path/to/python ./run.sh
MSG
  exit 1
fi

export APP_ENV=production
echo "film-grain -> http://${HOST}:${PORT}   ($(cat VERSION | head -1))"
exec "$PY" -m uvicorn server.main:app --host "$HOST" --port "$PORT" "$@"
LAUNCH
chmod +x "$OUT/run.sh"

# ------------------------------------------------------------------- venv --
if [ "$WITH_VENV" = 1 ]; then
  say "Creating $OUT/.venv (torch is a large download)"
  python3 -m venv "$OUT/.venv"
  "$OUT/.venv/bin/pip" install --upgrade pip >/dev/null
  "$OUT/.venv/bin/pip" install -r "$OUT/requirements.txt"
fi

say "Done: $OUT/"
du -sh "$OUT" 2>/dev/null | sed 's/^/    /'
echo
echo "  Run it:   cd $OUT && ./run.sh"
[ "$WITH_VENV" = 1 ] || cat <<'NEXT'
  First run needs an environment:
      cd build && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  or reuse this project's:
      cd build && FILM_GRAIN_PYTHON=../.venv/bin/python ./run.sh
NEXT
