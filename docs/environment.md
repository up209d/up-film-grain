# Environment and current state

## Environment gotchas (this machine)

* **Python 3.13.** `pipenv` with `PIPENV_VENV_IN_PROJECT=1`; the venv is
  `.venv/`. The reason used to be "PyTorch has no 3.14 wheels", and that is
  **no longer true** — torch 2.13.0 publishes `cp314`, verified against PyPI
  2026-08-19. 3.13 is now just what `Pipfile.lock` is resolved against and what
  the shipped runtime uses, and moving it is a deliberate change rather than a
  constraint.
* **Homebrew and Laravel Herd sit ahead of nvm in `PATH`**, so `nvm use` alone
  does not switch `node` — it still resolves to Homebrew's v26. `dev.sh` and
  `build.sh` both prepend nvm's bin themselves, resolving the version **from
  `.nvmrc`** rather than a hardcoded patch. They used to hardcode
  `v24.15.0`, which tied both scripts to this machine and, anywhere else, fell
  through silently to a different node major.
* **npm blocks postinstall scripts by default here.** A fresh `npm install`
  needs `npm approve-scripts esbuild` or vite will not build.
* Torch runs on **Apple MPS**. 64-bit integer ops are poorly supported there,
  which is why the noise hash is computed in uint64 on the CPU over the (much
  smaller) lattice rather than per-pixel on device.
* Background a long-running server with the tool's own background mode; a
  trailing `&` inside a single bash call does not always survive, and a stale
  instance will hold port 8000 and make the next start fail to bind. **If you
  start a server for testing, stop it before handing back** — otherwise the
  user's own `./dev.sh` cannot bind. Both scripts now preflight the port and
  name the offending PID rather than dying on a bare `Errno 48`, and both
  accept `PORT=8001`.


## State / not done

* **Input is JPEG/PNG only, 30MB max** (`imageio.INPUT_FORMATS`,
  `MAX_UPLOAD_BYTES`). TIFF was dropped and RAW was never implemented (it needs
  `rawpy`/LibRaw).
* **The neural pipeline (Approach B) is not started** and cannot be until a
  paired film-scan/digital dataset exists. The `/api/preview` and `/api/export`
  endpoints are shaped so a model could slot in behind them later.
* Uploads and export jobs are held **in process memory** (12 uploads max, LRU).
  Fine for local single-user use; would need real storage to deploy.
* No auth, no rate limiting — it binds to 127.0.0.1 only. The desktop app also
  takes an OS-assigned port rather than a fixed 8000, so two copies cannot
  collide, but anything else on the machine can still reach the API.


## The bundled runtime (2026-08-19)

The app ships its own interpreter, so none of the above applies to a *built* app —
only to developing one.

* **CPython 3.13.15 from `astral-sh/python-build-standalone`**, release
  `20260814`, pinned by SHA-256 in `tools/bundle.py` and cached under
  `.cache/runtimes/`. The `install_only` archives are prefix-relocatable, which is
  the single property the whole approach rests on.
* **No venv.** Dependencies install into the runtime's own `site-packages` and the
  shell invokes `runtime/bin/python3` directly. A venv would write absolute paths
  into `pyvenv.cfg` and every console script — exactly what must not happen in
  something that gets copied to another machine.
* **pip's console-script wrappers are deleted** during the prune. pip writes them
  with an absolute shebang naming the interpreter that installed them, so
  `uvicorn`, `torchrun`, `f2py`, `pip` and the rest were both broken on arrival and
  a leak of the build path. Nothing uses them: pip is reached as
  `python3 -m pip`, and the server as `python3 launch.py`.
* **Bytecode is precompiled and shipped**, with `compileall -s` so the recorded
  source paths are relative. Measured on this machine: importing `server.main`
  takes **1.0s** with the bytecode present against **2.4s** without it when the
  install directory is unwritable — which is the case inside a `.app`. Costs about
  155MB.
* `tools/bundle.py` fails the build if any shipped file contains the build
  machine's absolute path. That check has already earned itself twice: once on the
  pip wrappers above, and once on stdlib `.pyc` written by the *import* machinery
  while `compileall` was running, which ignores `-s` (hence
  `PYTHONDONTWRITEBYTECODE=1` around that call).
* **Near-black flat fills band on this display.** `--bg` was #0d0e10 and showed
  horizontal bands across a flat area of the loading screen, measured at #111215
  and #151617 within one uniform-in-markup fill. `capturePage()` of the same page
  was uniform, so it is not the markup and not the window background (proved by
  painting the window red and the page lime: all lime). `--bg` is #111215 now,
  and `electron/main.js`'s `APP_BG` and the splash's body have to match it.
  A saturated colour does not show the artifact at all, so test with the real value.
* **`screencapture` and `osascript` are permission-blocked here**, but Electron's
  `desktopCapturer.getSources({types:["screen"]})` captures the composited screen
  at full resolution and needs nothing granted. That is the way to see window
  chrome; `webContents.capturePage()` only ever shows the page.
* **Not code-signed.** See the README for the three ways past Gatekeeper's
  quarantine. Signing needs a Developer ID, hardened-runtime entitlements for an
  embedded interpreter, and `signIgnore` scoping so electron-builder does not try
  to sign ~600MB of torch dylibs.
