# Environment and current state

## Environment gotchas (this machine)

* **Python 3.13, not 3.14.** PyTorch has no 3.14 wheels. `pipenv` with
  `PIPENV_VENV_IN_PROJECT=1`; the venv is `.venv/`.
* **Homebrew and Laravel Herd sit ahead of nvm in `PATH`**, so `nvm use` alone
  does not switch `node` — it still resolves to Homebrew's v26. Prepend
  explicitly: `export PATH="$HOME/.nvm/versions/node/v24.15.0/bin:$PATH"`.
  `dev.sh` already does this.
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
* No auth, no rate limiting — it binds to 127.0.0.1 only.
