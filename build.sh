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
# CLEANUP_OLD_BUILD=1 ./build.sh    delete the previous build/bundle and dist/
#                                   first, so nothing from an earlier run can be
#                                   carried into this one. --skip-runtime still
#                                   wins for build/bundle/runtime: the cleanup
#                                   would otherwise re-download it and the two
#                                   flags together would mean nothing.
#
# ELECTRON_BUILDER_COMPRESSION_LEVEL=9 ./build.sh
#                                   compress the .tar.gz as hard as 7-Zip will.
#                                   Defaults to 5 here; see the note at stage 3
#                                   for why 9 costs minutes and buys ~1%.
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
    -h|--help)     sed -n '2,34p' "$0" | sed 's/^# \?//'; exit 0 ;;
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

# Optional clean slate, off by default -- a normal build is incremental on
# purpose (the Python runtime is a ~700MB download). Everything removed here is
# build output; nothing in the source tree is touched.
if [ -n "${CLEANUP_OLD_BUILD:-}" ] && [ "${CLEANUP_OLD_BUILD}" != "0" ]; then
  say "cleanup: removing previous build output"
  rm -rf dist build/bundle/payload
  if [ "$SKIP_RUNTIME" = 1 ]; then
    say "cleanup: keeping build/bundle/runtime (--skip-runtime)"
  else
    rm -rf build/bundle/runtime
  fi
fi

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

# The gzip level for the .tar.gz, and the reason the last step used to take
# minutes. electron-builder writes a plain .tar and then hands it to 7-Zip's
# Deflate encoder, which defaults to `-mx=9` -- and 7-Zip's -mx=9 is not gzip -9,
# it is a far more exhaustive match search that this payload cannot pay off.
# Measured end to end, `./build.sh --skip-client --skip-runtime` on the 1.1GB
# .app -- the whole difference is this one step:
#
#     -mx=9   6m29s   327,334,315 bytes
#     -mx=5   1m46s   330,841,139 bytes    (+1.07%)
#
# Seven times the compression time for one percent of the size, and all of it on
# one core -- 7-Zip's gzip encoder is not threaded, so the other thirteen are idle
# whichever level is picked. Nothing else in the build is close to this: stages 1
# and 2 together are ~15s, and the plain .tar that feeds this is ~19s.
#
# `-mx=1` was measured too (2.5s against 98.2s on the 353MB of torch dylibs that
# dominate the payload) but costs 6% of the size, which is 20MB people download.
#
# Set the variable yourself to override; `9` restores the old behaviour.
export ELECTRON_BUILDER_COMPRESSION_LEVEL="${ELECTRON_BUILDER_COMPRESSION_LEVEL:-5}"

say "shell: packaging (gzip -mx=$ELECTRON_BUILDER_COMPRESSION_LEVEL)"
(cd electron && npx --no-install electron-builder \
    --config electron-builder.yml --publish never)

say "done"
ls -1 dist/*.tar.gz 2>/dev/null | sed 's/^/    /' || true
du -sh dist/*.app dist/mac*/*.app 2>/dev/null | sed 's/^/    /' || true

# Note: build/ still holds the previous build system's output (build/server,
# build/run.sh and friends). It is left alone deliberately -- it is what the
# current production process serves -- and can be deleted once this app is
# confirmed working. The new artifacts are entirely under build/bundle/ and dist/.
