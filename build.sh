#!/usr/bin/env bash
# Build the Film Grain desktop app: a single self-contained artifact.
#
# Three stages, and each is a separate script so the middle one can be run and
# verified on its own:
#
#   1. the client        web/  ->  web/dist         (npm run build)
#   2. the payload       everything the server needs, plus its own Python
#                        runtime with torch inside it   (tools/bundle.py)
#   3. the shell         an Electron app wrapping stage 2  (electron-builder)
#
# The result needs nothing installed on the machine it lands on -- no Python, no
# pipenv, no node.
#
#   ./build.sh                        build for this platform (mac)
#   ./build.sh --target mac           the same, said explicitly
#   ./build.sh --no-electron          stages 1-2 only: the server bundle
#   ./build.sh --skip-client          reuse the existing web/dist
#   ./build.sh --skip-runtime         reuse build/bundle/runtime (fast iteration)
#
# Windows and Linux are not supported yet and say so; tools/bundle.py records
# what they will need.
set -euo pipefail
cd "$(dirname "$0")"

TARGET="mac"
WITH_ELECTRON=1
SKIP_CLIENT=0
SKIP_RUNTIME=0

for arg in "$@"; do
  case "$arg" in
    --target=*)    TARGET="${arg#--target=}" ;;
    --target)      echo "use --target=NAME" >&2; exit 2 ;;
    mac|windows|linux) TARGET="$arg" ;;
    --no-electron) WITH_ELECTRON=0 ;;
    --skip-client) SKIP_CLIENT=1 ;;
    --skip-runtime) SKIP_RUNTIME=1 ;;
    -h|--help)     sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# One build at a time. Two concurrent runs share build/bundle/ and dist/, and the
# failure is not obvious when it happens: one process removes the output directory
# while the other is copying into it, and electron-builder reports it as
# `ENOENT ... copyfile` naming a *source* file that is present and fine. mkdir is
# atomic, which is why it is the lock rather than a -f test.
LOCK=".build.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  die "another build is already running (delete $LOCK if it is stale)"
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# Gate unsupported targets before doing any work. The message lives in exactly one
# place -- tools/bundle.py -- so both entry points say the same thing; it exits 2.
case "$TARGET" in
  mac) ;;
  windows|linux) python3 tools/bundle.py --target "$TARGET" || exit $? ;;
  *) die "unknown target: $TARGET (mac, windows, linux)" ;;
esac

# ---------------------------------------------------------------- 1. client --
# node from nvm, chosen by .nvmrc rather than a hardcoded patch version. The old
# script pinned v24.15.0 by absolute path, which tied the build to one machine and
# silently fell through to whatever node was on PATH anywhere else -- here that is
# v26, a different major than the project pins.
setup_node() {
  local want best
  want="$(tr -d '[:space:]' < .nvmrc 2>/dev/null || true)"
  if [ -n "$want" ] && [ -d "$HOME/.nvm/versions/node" ]; then
    # Exact version first, then the newest matching that major/minor prefix.
    best="$(ls -1d "$HOME/.nvm/versions/node/v$want" 2>/dev/null | tail -1 || true)"
    [ -n "$best" ] || best="$(ls -1d "$HOME/.nvm/versions/node/v$want."* 2>/dev/null | sort -V | tail -1 || true)"
    [ -n "$best" ] && export PATH="$best/bin:$PATH"
  fi
  command -v node >/dev/null || die "node not found on PATH (and none in ~/.nvm matching .nvmrc)"
  local have
  have="$(node --version)"
  if [ -n "$want" ] && [ "${have#v}" != "${want}" ] && [ "${have%%.*}" != "v$want" ]; then
    printf '\033[33mwarning:\033[0m .nvmrc asks for node %s, using %s\n' "$want" "$have" >&2
  fi
  say "node $have"
}

if [ "$SKIP_CLIENT" = 1 ]; then
  [ -f web/dist/index.html ] || die "--skip-client but web/dist/index.html is missing"
  say "client: reusing web/dist"
else
  setup_node
  cd web
  [ -d node_modules ] || {
    say "client: installing dependencies"
    npm install
    # npm blocks postinstall scripts here and vite will not build without
    # esbuild's.
    npm approve-scripts esbuild 2>/dev/null || true
  }
  say "client: building"
  npm run build
  cd ..
  [ -f web/dist/index.html ] || die "the client build produced no index.html"
fi

# --------------------------------------------------------------- 2. payload --
say "payload: assembling (this downloads a Python runtime the first time)"
BUNDLE_ARGS=(--target "$TARGET")
[ "$SKIP_RUNTIME" = 1 ] && BUNDLE_ARGS+=(--skip-runtime)
python3 tools/bundle.py "${BUNDLE_ARGS[@]}"

if [ "$WITH_ELECTRON" = 0 ]; then
  say "done: build/bundle (--no-electron)"
  echo
  echo "  Run the server directly:"
  echo "    build/bundle/runtime/bin/python3 build/bundle/payload/launch.py"
  exit 0
fi

# ----------------------------------------------------------------- 3. shell --
[ -d electron/node_modules ] || {
  setup_node
  say "shell: installing Electron"
  (cd electron && npm install --no-audit --no-fund)
}
setup_node
say "shell: packaging"
(cd electron && npx --no-install electron-builder \
    --config electron-builder.yml --publish never)

say "done"
ls -1 dist/*.tar.gz 2>/dev/null | sed 's/^/    /' || true
du -sh dist/*.app dist/mac*/*.app 2>/dev/null | sed 's/^/    /' || true

# Note: build/ still holds the previous build system's output (build/server,
# build/run.sh and friends). It is left alone deliberately -- it is what the
# current production process serves -- and can be deleted once this app is
# confirmed working. The new artifacts are entirely under build/bundle/ and dist/.
