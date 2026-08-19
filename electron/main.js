"use strict";
/**
 * The desktop shell: one window, and the Python render server it owns.
 *
 * The two processes ship and die together, which is the whole point -- the user
 * launches one thing. This file is everything that makes that true, and it is
 * deliberately *all* of it: the renderer is the unmodified web app, served over
 * http by the same FastAPI process as always, so there is no preload script, no
 * IPC and not one line of Electron-aware code in `web/src`.
 *
 * Why http and not file://. Every API call in `web/src/services/api.ts` is a
 * same-origin relative path, so loading `index.html` off disk would break all of
 * them and would need `base: "./"` in the Vite config besides. Pointing the
 * window at 127.0.0.1 keeps the browser build and the desktop build the same
 * build.
 */

const { app, BrowserWindow, Menu, dialog, session, shell } = require("electron");
const { spawn, execFile } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const REPO = path.resolve(__dirname, "..");
const MAC = process.platform === "darwin";

// Taken from web/src/styles/base.css rather than eyeballed: --bg is the body
// background and --panel is the top bar's. The window has to be painted the same
// colour as the page or there is a flash of the wrong shade on every launch, and
// with the title bar hidden that flash is the full height of the window.
// The window background, and it must equal `--bg` in web/src/styles/base.css and
// the splash's body. All three are the same value on purpose.
//
// An earlier version of this comment claimed the window background needed a
// *different* value to compensate for AppKit and Chromium rendering the same hex
// differently on a P3 display. That was wrong, and the test that killed it is
// worth keeping: painting the window red and the page lime showed **all lime**,
// so the page covers the window completely and its background is never visible.
const APP_BG = "#111215";           // = --bg
const BAR_BG = "#16181c";     // --panel

/**
 * Make the title bar disappear into the app.
 *
 * With `titleBarStyle: "hidden"` the page owns the full window height and macOS
 * just floats the traffic lights on top, so the app's own bar becomes the title
 * bar and there is no seam of system grey above it.
 *
 * Injected from here rather than written into `web/src/styles/bar.css` on purpose.
 * These rules are wrong in a browser -- 84px of empty space where no traffic
 * lights exist, and a drag region that does nothing -- and the whole design of
 * this shell is that the browser build and the desktop build are the same build.
 * insertCSS applies them only where they are true.
 *
 * The `no-drag` list matters: `-webkit-app-region: drag` is inherited, so without
 * it every control in the bar would move the window instead of doing its job --
 * including the Source link, which is the AGPL section 13 offer.
 */
const SEAMLESS_CSS = `
  .bar {
    /* !important is load-bearing, not laziness. bar.css sets the shorthand
       \`padding: 10px 14px\`, which has the same specificity as this rule, and an
       injected stylesheet does not reliably win that tie -- measured in the live
       window, the drag region below applied while a plain \`padding-left: 84px\`
       was overridden back to 14px and the traffic lights sat on top of the brand.
       84px clears the lights; 90px is that plus the breathing room the author
       asked for after seeing it. padding-top needs the same treatment for the
       same reason -- it is the other half of that shorthand. */
    padding-left: 90px !important;
    padding-top: 8px !important;
    -webkit-app-region: drag;
  }
  .bar button, .bar a, .bar input, .bar select, .bar label,
  .bar [role="button"], .bar [tabindex] {
    -webkit-app-region: no-drag;
  }

  /* Pad the right panel */
  .panel {
    margin-right: 6px !important;
    padding-right: 12px !important;
  }
  .panel::after {
    content: '';
    display: block;
    position: absolute;
    width: 6px;
    height: 100%;
    top: 0;
    right: 0;
    background: var(--panel);
    pointer-events: none;
    z-index: -1;
  }

  /* 1. Target the entire page or specific scrollable containers */
  ::-webkit-scrollbar {
    width: 6px;  /* Very thin vertical bar */
    height: 6px; /* Very thin horizontal bar */
  }

  /* 2. Make the track completely transparent so it's seamless */
  ::-webkit-scrollbar-track {
    background: transparent;
  }

  /* 3. Style the handle with a subtle color that matches dark themes */
  ::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15); /* Semi-transparent white */
    border-radius: 10px;
    transition: background 0.2s ease;
  }

  /* 4. Make it slightly more visible when hovered */
  ::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.35);
  }
`;

// Importing torch and building the engine takes a second or two warm and longer
// on a cold filesystem cache, so the window has to appear before the app does.
const URL_TIMEOUT_MS = 45_000;   // waiting for the server to name its port
const HEALTH_TIMEOUT_MS = 120_000; // waiting for it to answer /api/health
const HEALTH_INTERVAL_MS = 200;
const EXPORT_POLL_MS = 2_000;

let child = null;        // the Python server
let win = null;
let appUrl = null;
let logFile = null;
// Polled rather than fetched on demand: `before-quit` has to call
// preventDefault() synchronously, so it cannot await an HTTP round trip.
let activeExports = 0;
let confirmedQuit = false;
// Set the moment we start tearing down. Without it, the child's exit handler below
// treats the SIGTERM *we* sent as a crash and pops an error dialog on every
// ordinary quit -- the signal exit reports code null, which is not 0.
let shuttingDown = false;

// --------------------------------------------------------------- locating it --
function pythonIn(root) {
  const rels = [["bin", "python3"], ["bin", "python"],
                ["python.exe"], ["Scripts", "python.exe"]];
  for (const rel of rels) {
    const p = path.join(root, ...rel);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * Where the server lives, packaged or not.
 *
 * Packaged, it is the bundled runtime in Resources. Unpackaged, it prefers a
 * staged bundle if one has been built and otherwise falls back to the repo's own
 * `.venv` and source tree -- which is what makes `npm start` a usable dev loop
 * without waiting on a 90-second bundle build.
 */
function resolveInstall() {
  const candidates = app.isPackaged
    ? [{ label: "bundled runtime",
         payload: path.join(process.resourcesPath, "payload"),
         runtime: path.join(process.resourcesPath, "runtime") }]
    : [{ label: "staged bundle",
         payload: path.join(REPO, "build", "bundle", "payload"),
         runtime: path.join(REPO, "build", "bundle", "runtime") },
       { label: "source tree + .venv",
         payload: REPO,
         runtime: path.join(REPO, ".venv") }];

  for (const c of candidates) {
    const python = pythonIn(c.runtime);
    if (python && fs.existsSync(path.join(c.payload, "launch.py"))) {
      return { ...c, python };
    }
  }
  return null;
}

function licensesDir() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "licenses")
    : path.join(REPO, "build", "bundle", "licenses");
}

// ----------------------------------------------------------------- logging --
function openLog() {
  const dir = app.getPath("logs");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, "server.log");
  // Keep exactly one previous run. A user reporting "it worked yesterday" is
  // usually holding the log that matters, and unbounded growth in a log nobody
  // rotates is its own bug.
  try { if (fs.existsSync(file)) fs.renameSync(file, `${file}.1`); } catch { /* ignore */ }
  const stream = fs.createWriteStream(file, { flags: "a" });
  stream.write(`--- ${new Date().toISOString()} film-grain ${app.getVersion()}\n`);
  return { file, stream };
}

// ------------------------------------------------------------------- server --
function startServer(install, log) {
  return new Promise((resolve, reject) => {
    const args = [
      path.join(install.payload, "launch.py"),
      // Port 0: the server binds whatever is free and tells us which, so two
      // copies cannot collide and a half-dead previous run cannot block startup.
      "--port", "0",
      "--no-browser",
      // Belt and braces with the kill on quit below: if this process dies
      // without running its handlers, the child notices and exits itself rather
      // than lingering with a multi-gigabyte allocator pool and no window.
      "--parent-pid", String(process.pid),
    ];
    const env = { ...process.env, APP_ENV: "production", PYTHONUNBUFFERED: "1" };
    // The bundle ships precompiled bytecode and may sit somewhere unwritable, so
    // there is nothing to gain by trying. In dev, leave caching on.
    if (app.isPackaged) env.PYTHONDONTWRITEBYTECODE = "1";

    const proc = spawn(install.python, args, {
      cwd: install.payload, env, stdio: ["ignore", "pipe", "pipe"],
    });

    let settled = false;
    let seen = "";
    const done = (fn, arg) => { if (!settled) { settled = true; fn(arg); } };

    const onOutput = (chunk) => {
      const text = chunk.toString();
      log.stream.write(text);
      if (settled) return;
      seen += text;
      // The server prints this once its socket is bound, before the slow imports.
      const m = seen.match(/FILM_GRAIN_URL (\S+)/);
      if (m) done(resolve, { proc, url: m[1] });
    };
    proc.stdout.on("data", onOutput);
    proc.stderr.on("data", onOutput);

    proc.on("error", (e) =>
      done(reject, new Error(`could not start ${install.python}: ${e.message}`)));
    proc.on("exit", (code, signal) =>
      done(reject, new Error(
        `the render server exited (code ${code}, signal ${signal}) before it `
        + `reported a URL`)));
    setTimeout(() => done(reject, new Error(
      `the render server did not report a URL within ${URL_TIMEOUT_MS / 1000}s`)),
      URL_TIMEOUT_MS);
  });
}

function stopServer() {
  shuttingDown = true;
  const proc = child;
  child = null;
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;
  if (process.platform === "win32") {
    // child.kill() on Windows does not reliably take the whole tree, and Python
    // plus torch is a tree.
    try { execFile("taskkill", ["/pid", String(proc.pid), "/T", "/F"]); } catch { /* ignore */ }
    return;
  }
  try { proc.kill("SIGTERM"); } catch { /* ignore */ }
  // If it has not gone by the time the app is actually tearing down, the
  // watchdog inside launch.py finishes the job.
  setTimeout(() => { try { proc.kill("SIGKILL"); } catch { /* ignore */ } }, 3_000);
}

// --------------------------------------------------------------------- http --
function get(url, timeoutMs = 3_000) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = "";
      res.on("data", (c) => { body += c; });
      res.on("end", () => resolve({ status: res.statusCode, body }));
    });
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error("timeout")));
  });
}

async function waitHealthy(url) {
  // `/api/health` already exists and already reports the chosen device, so the
  // readiness probe and the "which backend am I on" answer are the same call.
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  let last = "no response yet";
  while (Date.now() < deadline) {
    if (!child) throw new Error("the render server stopped while starting up");
    try {
      const res = await get(`${url}/api/health`);
      if (res.status === 200) return JSON.parse(res.body);
      last = `HTTP ${res.status}`;
    } catch (e) { last = e.message; }
    await new Promise((r) => setTimeout(r, HEALTH_INTERVAL_MS));
  }
  throw new Error(`the render server never became healthy (${last})`);
}

function pollExports(url) {
  setInterval(async () => {
    try {
      const res = await get(`${url}/api/exports`, 1_500);
      if (res.status === 200) activeExports = JSON.parse(res.body).active | 0;
    } catch { /* a missed poll is not worth reporting */ }
  }, EXPORT_POLL_MS).unref?.();
}

// ------------------------------------------------------------------- window --
function isInternal(target) {
  if (!appUrl) return false;
  try { return new URL(target).origin === new URL(appUrl).origin; }
  catch { return false; }
}

function guardNavigation(contents) {
  // Two separate traps, both of which look like the app "just breaking".
  contents.on("will-navigate", (event, target) => {
    if (isInternal(target)) return;
    event.preventDefault();
    // A dropped file navigates the window to file:// and takes the app with it,
    // and this app is drag-and-drop-first. Swallow those outright; send real
    // links to the real browser -- which matters most for the `Source` link in
    // TopBar.tsx, since that is the AGPL section 13 offer and replacing the app
    // with a GitHub page would be a poor way to honour it.
    if (/^https?:/i.test(target)) shell.openExternal(target);
  });
  contents.setWindowOpenHandler(({ url }) => {
    if (isInternal(url)) return { action: "allow" };
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440, height: 900, minWidth: 960, minHeight: 640,
    title: "Film Grain",
    backgroundColor: APP_BG,
    // **Not shown until it has painted.** `backgroundColor` is filled in by
    // AppKit while the web contents is not covering the window yet, and on a P3
    // display AppKit and Chromium do not render the same hex the same way: this
    // app's --bg #0d0e10 comes out #111215 from AppKit and #151617 from the
    // renderer. So any frame where the page has not painted shows as a band of
    // slightly-wrong dark at the top and bottom of the window -- measured off a
    // screenshot, and provably not in the page itself, whose own bitmap is
    // uniform. Waiting for `ready-to-show` removes the window-background frame
    // entirely rather than trying to colour-match two different painters.
    show: false,
    // The app is dark-only (`color-scheme: dark` in base.css), so tell macOS
    // that too -- otherwise the traffic lights and any system sheet are drawn
    // for a light window sitting on a near-black page.
    ...(MAC ? {
      titleBarStyle: "hidden",
      // y centres the 12px light cluster in the bar; x aligns it with the bar's
      // own 14px side padding, so the lights sit on the bar's axis rather than
      // near it. Both numbers are measured in the live window rather than derived
      // from the CSS -- the controls set the line box, so the bar is taller than
      // the padding suggests. It measures 45.84px with the padding-top above, so
      // this is (45.84 - 12) / 2. **Change the padding and this moves**, and the
      // splash's band in splash.html has to follow too.
      trafficLightPosition: { x: 14, y: 17 },
    } : {}),
    webPreferences: {
      // Nothing in the renderer needs node, and everything privileged lives in
      // this file, so the window is just a browser tab pointed at localhost.
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  // Re-applied per load, not once: insertCSS is scoped to the current document,
  // so the splash, the app and any reload each need it.
  win.webContents.on("did-finish-load", () => {
    // Only the served app. These rules are about the app's own top bar, and the
    // splash is a different document that happens to have used the same class
    // name -- injecting into it stretched its progress bar to 250x17px.
    if (MAC && /^https?:/.test(win.webContents.getURL())) {
      win.webContents.insertCSS(SEAMLESS_CSS).catch(() => { /* non-fatal */ });
    }
  });
  if (process.env.DEBUG === "true") win.webContents.openDevTools({ mode: "detach" });
  win.once("ready-to-show", () => win.show());
  win.loadFile(path.join(__dirname, "splash.html"));
  guardNavigation(win.webContents);
  win.on("close", (event) => {
    if (confirmedQuit || activeExports === 0) return;
    event.preventDefault();
    confirmDiscardExports().then((ok) => {
      if (ok) { confirmedQuit = true; win.close(); }
    });
  });
  win.on("closed", () => { win = null; });
  return win;
}

function buildMenu() {
  const isMac = process.platform === "darwin";
  // The Edit menu is not decoration: without it, Cmd+C/Cmd+V do nothing at all
  // in an Electron window on macOS, because the accelerators come from the menu.
  const template = [
    ...(isMac ? [{ role: "appMenu" }] : []),
    { role: "fileMenu" },
    { role: "editMenu" },
    { role: "viewMenu" },
    { role: "windowMenu" },
    {
      role: "help",
      submenu: [
        {
          label: "Licences",
          click: () => {
            const dir = licensesDir();
            if (fs.existsSync(dir)) shell.openPath(dir);
            else dialog.showErrorBox("Licences", `Not found: ${dir}`);
          },
        },
        {
          label: "Show Server Log",
          click: () => { if (logFile) shell.showItemInFolder(logFile); },
        },
      ],
      // Deliberately no "Source Code" item. The AGPL section 13 offer is the
      // `Source` link in TopBar.tsx, whose URL is required to be a lone constant
      // so a fork can repoint it in one place. A copy of that URL here would be a
      // second place to forget, on the one link that must not be wrong -- and the
      // in-app link already opens in a real browser via guardNavigation.
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function wireDownloads() {
  // The reason this exists: both the export and the preset-save flows go through
  // `<a download>` plus a blob/http URL (`useExport.ts`, `usePresetFile.ts`). In
  // a browser that is complete; in Electron a download with no handler gives the
  // user no dialog and no feedback, so exports would appear to do nothing. This
  // covers both, because will-download fires for http and blob alike.
  session.defaultSession.on("will-download", (event, item) => {
    item.setSaveDialogOptions({
      title: "Save image",
      defaultPath: path.join(app.getPath("downloads"), item.getFilename()),
    });
    item.once("done", (_e, state) => {
      if (state === "completed" || state === "cancelled") return;
      dialog.showErrorBox(
        "Save failed",
        `${item.getFilename()} could not be saved (${state}).`);
    });
  });
}

/**
 * Ask before discarding a running export. Shared by the window's close button and
 * by Quit, and that matters: guarding only the quit path let someone close the
 * window, keep rendering, and end up with an app running with no window and no way
 * to see the progress they chose to protect.
 */
function confirmDiscardExports() {
  return dialog.showMessageBox(win, {
    type: "warning",
    buttons: ["Discard and Close", "Keep Rendering"],
    defaultId: 1,
    cancelId: 1,
    message: activeExports === 1
      ? "An export is still rendering."
      : `${activeExports} exports are still rendering.`,
    detail: "The export worker is stopped with the app, so no file will be saved.",
  }).then(({ response }) => response === 0);
}

function fatal(message, detail) {
  const extra = logFile ? `\n\nServer log:\n${logFile}` : "";
  dialog.showErrorBox(message, `${detail || ""}${extra}`);
  // app.exit() emits neither before-quit nor will-quit, so the child would be
  // left behind. The watchdog inside launch.py would eventually notice, but
  // taking it down here means the failure does not also cost the user a stray
  // multi-gigabyte process for a couple of seconds.
  stopServer();
  app.exit(1);
}

// --------------------------------------------------------------------- boot --
async function boot() {
  buildMenu();
  wireDownloads();
  createWindow();

  const install = resolveInstall();
  if (!install) {
    return fatal("Film Grain could not start",
      app.isPackaged
        ? "The bundled Python runtime is missing from this application."
        : "No runtime found. Build one with:\n"
          + "  python3 tools/bundle.py --target mac\n"
          + "or create the dev environment with pipenv.");
  }

  const log = openLog();
  logFile = log.file;
  log.stream.write(`using ${install.label}: ${install.python}\n`);

  let started;
  try {
    started = await startServer(install, log);
  } catch (e) {
    return fatal("The render engine did not start", e.message);
  }
  child = started.proc;
  appUrl = started.url;

  // If the server dies later, say so rather than leaving a dead window.
  child.on("exit", (code, signal) => {
    if (shuttingDown || confirmedQuit || code === 0) return;
    child = null;
    fatal("The render engine stopped",
      signal ? `It was terminated by ${signal}.` : `It exited with code ${code}.`);
  });

  let health;
  try {
    health = await waitHealthy(appUrl);
  } catch (e) {
    return fatal("The render engine did not become ready", e.message);
  }

  log.stream.write(`ready on ${appUrl} using ${health.device}\n`);
  pollExports(appUrl);
  if (win) {
    win.loadURL(appUrl);
    win.setTitle(`Film Grain — ${health.device}`);
  }
}

// A second copy would start a second engine, each holding gigabytes. Focus the
// one that is already open instead.
if (!app.requestSingleInstanceLock()) {
  app.exit(0);
} else {
  app.on("second-instance", () => {
    if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
  });

  app.whenReady().then(boot);

  app.on("window-all-closed", () => { app.quit(); });

  app.on("before-quit", (event) => {
    if (confirmedQuit || activeExports === 0) return;
    // The export worker is a daemon thread, so process exit kills it with no
    // drain and no error -- the user just never gets the file. Worth one dialog.
    event.preventDefault();
    confirmDiscardExports().then((ok) => {
      if (ok) { confirmedQuit = true; app.quit(); }
    });
  });

  app.on("will-quit", stopServer);
  process.on("exit", stopServer);
}
