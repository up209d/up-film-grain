# The SSD is the cache (2026-08-29)

Requested outright: *"App seem to eat a lot of ram, I am not sure if it is
because of the cache we have, can we dump all cache to ssd, make ssd to be the
channel for cache, I dont want retain cache in ram, flush all ram cache after
pipeline done rendering. Reload cache from ssd."* — and, a few minutes later,
*"dont forget to clear all cache when open new image."*

The uncertainty in the first sentence turned out to be the interesting part. It
was partly the cache, and two of the five things holding memory were not caches
at all.

## What was actually holding it

Measured on this machine, a 24MP photograph, before any of this:

| holder | size | bounded by |
|---|---|---|
| `UPLOADS` — source, proxy, prescaled frame, prescaled proxy | ~670MB **per photograph** | a cap of 12, so ~8GB |
| `CheckpointCache` | 184MB per boundary per tile | 15% of the device pool, ~2GB here |
| `_gg_cache` (Global Grain textures) | 184MB per layer, five layers | 15% of the device pool, ~2GB here |
| `JOBS[…]["bytes"]` — finished exports | ~140MB per 24MP 16-bit PNG | **nothing** |
| `lut._DISK` — parsed `.cube` tables | 3.1MB table + 3.1MB device tensor each | **nothing**, against a library of 303 |

So roughly twelve gigabytes was reachable before anything had been rendered
twice, and the two unbounded ones were the surprise. `JOBS` was an outright
leak: a finished export's encoded bytes sat on the job dict for the life of the
process, against no future hit, because nothing ever removed them. `lut._DISK`
was fine while `luts/` held a handful and stopped being fine on 2026-08-09 when
a library of 300 arrived — browsing the picker parsed them.

The biggest single item, `UPLOADS`, was never described as a cache anywhere and
is the reason "is it the cache?" could not be answered by looking at the caches.

## The arrangement

`server/engine/diskcache.py` holds all of it. Three shapes, because three
different questions are being asked:

* **`DiskStore`** — a byte-capped LRU whose values are files. `put` writes and
  keeps a path; `get` reads that file and returns a fresh tensor. A miss is
  legal: the caller recomputes. This is the checkpoint cache and the texture
  cache.
* **`Spill`** — one array whose only copy is a file, mapped back copy-on-write
  when something reads it. A miss is *not* legal — the decoded photograph
  cannot be recomputed from anything the process still has. This is
  `Upload.arr`, the one proxy `Upload.proxy_at` holds, and both of `Frame`'s.
* **`Blob`** — encoded bytes on disk, handed to `FileResponse`. This is a
  finished export, which is the user's output rather than a cache at all; it
  moved for the same reason but is kept under different rules.

There is **no in-memory tier to spill from**. That is the part that is unlike
the usual arrangement and it is deliberate: within a single render there are no
repeat `get`s to serve — each tile asks for its own coordinates, and each
boundary is read once — so a hot tier would add a copy of every 184MB frame to
protect a hit that does not occur.

## Why this is affordable here and would not be in general

Every one of these caches exists to skip work measured in **seconds**:

* a checkpoint hit skips 68–89% of a render
* a texture hit skips 1.29s of a 3.70s `Stock` proxy preview
* an upload's array skips a decode

A 184MB read from an NVMe SSD is 50–90ms. Trading 90ms for 1.2s is not close.
Measured end to end on a 24MP source, three previews with one slider moving
between them:

```
                          RSS      peak     on disk
baseline (torch loaded)   2131MB   2132MB      0MB
24MP uploaded             2475MB   2475MB    334MB
preview 1 (1.47s)         3860MB   3860MB    948MB
preview 2 (0.43s)         3861MB   3861MB    948MB
preview 3 (0.42s)         3861MB   3861MB    948MB
3s idle (flush fired)      471MB   3861MB    948MB
```

1.47s → 0.43s is the checkpoint hit *from disk*, so the cache is not merely
surviving the move, it is still paying for itself at 3.4×. And 3861MB → 471MB
is the idle flush.

The ratio is what makes this a good trade rather than a resigned one. **Do not
add anything small to a store here**: a payload cheaper to recompute than to
read back belongs in neither tier, which is why `lut._DISK` was given an LRU cap
instead of a file (a 3MB table is not worth either).

## The cost, named

A slider drag re-renders continuously and every render writes its checkpoints —
~180MB per boundary crossed at a 2400px proxy, so sustained dragging writes on
the order of 150MB/s to the drive. Over a long session that is real SSD wear.
`FILM_GRAIN_DISK_CACHE_GB=0` switches the stores off outright rather than merely
shrinking them, which is both the honest way to measure what they are worth and
the escape hatch for anyone who would rather spend RAM than write cycles.

## The flush is on an idle timer, not on the last render

`arm_flush` resets a two-second timer after every render; the flush unmaps every
`Spill` and hands the allocator's free list back.

The naive version — flush the moment a render returns — is wrong for the case
that matters, and measurably so. A drag is a stream of single-tile proxy
renders, and `release_cache` on that path costs `Stock` 1.13s → 1.45s and
`VintageDarkGrainy` 1.64s → 1.94s, to hand back blocks the very next render
immediately asks for again. During a drag the free list is not idle memory; it
is the next frame's working set arriving early. What is worth reclaiming is the
list held while the user is *not* rendering. `FILM_GRAIN_FLUSH_IDLE` tunes the
delay, and 0 restores the literal reading for anyone who wants it.

## The guard that has to be on the device, not the callers

`release_cache` on MPS commits and synchronises the command queue. Calling it
while another thread has device work in flight does not race — **it aborts the
process**:

```
-[IOGPUMetalCommandBuffer validate]: failed assertion
  `commit an already committed command buffer'
```

That is not theoretical. The idle flush was first written to fire on its timer
thread under `runtime.RENDER_LOCK`, and the whole check suite died on exactly
that assertion, because a check calling `render()` directly holds no such lock —
and neither does an upload resampling on a request thread.

Guarding the *callers* means enumerating every path that touches the GPU and
being right about all of them. Guarding the *device* is one counter:
`device.device_work()` is taken by `render_image`, `render_supersampled`,
`render_view`, `render()` and `imageio._interp`, and `device.try_release_cache`
does nothing unless it reads zero. A counter rather than a mutex because the
exclusion is asymmetric — any number of threads may use the device at once, and
only the release must be alone. A plain lock held for the length of a render
would have made an upload during a 24MP export wait for the export.

**Between tiles is the exception and must stay one.** `_render_image` calls
`release_cache` directly, from inside `device_work`; routing it through
`try_release_cache` would make it always answer False and silently stop the
per-tile release that keeps `_WORKING_BYTES_PER_PX` honest.

## Opening a photograph is a hard reset

`models.upload.reset` drops every other upload's files and clears both stores.
It is the right call rather than merely what was asked for: **every cache in the
app is keyed to a photograph and none of it can ever be hit again once you open
a different one.** Checkpoint keys carry the upload id by construction; the
source and proxy arrays *are* that photograph's pixels.

So the twelve-upload cap is no longer what bounds the disk — it is the fallback
for whatever this misses, which is why it stays. Two things survive
deliberately:

* **Finished exports**, which are output rather than cache. Dropping one because
  the user opened the next photograph is silent data loss.
* Nothing else. The Global Grain textures are cleared for tidiness rather than
  necessity — they read no image data at all, so they are *technically* still
  valid for the new photograph. That is a judgement, not a correctness
  requirement, and it is the line to reconsider if a workflow ever turns out to
  open images with the look already dialled in.

## Lifetime, and the quit path that skips `atexit`

The root is `~/Library/Caches/UpFilmGrain/run-<pid>-<rand>` (the platform cache
directory elsewhere), removed on exit and swept on the next start.

**Never `/tmp`.** On most Linux distributions that is a tmpfs — RAM with a
filesystem in front of it — so spilling there would move twelve gigabytes from
one part of memory to another and report success.

Nothing survives the process, and that is a correctness property rather than
tidiness: a key can never be answered by a previous run's bytes.

Three exits, three mechanisms, and the middle one is the app's *normal* quit:

| exit | what cleans up |
|---|---|
| clean return | `atexit` |
| **SIGTERM — what `electron/main.js` sends** | a handler installed in `launch.py` |
| SIGKILL, crash, power cut | `_sweep` on the next start |

The signal path works because of a uvicorn behaviour worth writing down, since
it looks like it should not: uvicorn installs its own SIGINT/SIGTERM handlers
for the duration of `Server.run`, and on the way out **restores whatever was
there before and then re-raises the captured signal** (`Server.capture_signals`).
So a handler installed *before* `run()` is not shadowed — it is deferred, and
fires after the graceful shutdown finishes. Installing one after `run()` returns
is too late (the re-raise kills the process inside it); replacing uvicorn's
breaks the graceful shutdown outright. Measured before this was wired up: every
ordinary desktop quit left a full run's spill behind.

`launch.py`'s parent watchdog needs its own call for a different reason — it
exits with `os._exit`, which runs no `atexit` handler, deliberately.

## Reporting

`GET /api/cache` returns what every part is holding plus the process's current
and peak RSS; `POST /api/cache/clear` drops what can be rebuilt. The client polls
it from `useCacheStats` and shows it in the bottom-right overlay beside Export.

**Both memory numbers, and the current one is the headline.** `ru_maxrss` is a
high-water mark that never comes down, so on its own it cannot show memory being
*given back* — which is the entire subject of this change, and would make the
readout unable to display the thing it exists to display. macOS has no current
figure in `resource` at all, hence `libproc`; Linux reads `/proc/self/statm`;
Windows uses `GetProcessMemoryInfo`. No dependency: `psutil` is not in the
Pipfile and this is not worth adding one for.

The readout exists because "the app eats RAM" was a question nobody in the
session could answer. A fix you cannot see is indistinguishable from no fix.

## What did not change

* **Both invariants.** No stage learns that its frame came from a file, so tile
  independence and scale invariance are untouched. The caches are keyed exactly
  as they were.
* **The renderer's `_TILE_BUDGET_SHARE`**, still 0.7. The two 15% cache shares
  are gone from `device.py` rather than set to zero, so nothing competes for the
  device pool any more — and raising 0.7 toward 1.0 is now free of the old
  objection and deliberately *not* done. Larger tiles mean a larger peak working
  set, and this change was made to lower the app's memory use, not to spend the
  saving somewhere else.
* **Generation eviction**, in both caches. It was never about which medium the
  waste sat in; it is about entries no render can ever ask for again. A version
  that started tolerating unbounded growth because growth is cheaper now would
  be measuring the budget instead of the policy.

## Traps

* **Two engines in one process must not share a store directory.** Keys carry
  the image id, the geometry and the device but no engine identity, so two
  engines can produce byte-identical keys. Two in-memory dicts made that safe
  for free; two directories with the same name would not. Hence a uuid in the
  directory name, and `weakref.finalize` so a discarded engine takes its files
  with it. `tests/checks/` is the one place that builds more than one.
* **`Spill` maps with `mmap_mode="c"`, not `"r"`.** `torch.from_numpy` refuses a
  read-only array and `imageio._interp` — which every resample in the app goes
  through — calls exactly that. Copy-on-write gives a writable view whose
  unwritten pages stay shared with the file.
* **`get` returns a fresh tensor every call.** The in-memory version handed the
  same object to every caller, one careless in-place op away from a corrupted
  entry. Reading from a file makes independence structural; `tests/checks/
  diskcache.py` pins that it stayed that way.
* **`np.save` writes a 128-byte header.** The store accounts the *array's*
  bytes, not the file's, so the UI figure is comparable to what used to be
  resident. The checks spell the difference out rather than allowing slop.
