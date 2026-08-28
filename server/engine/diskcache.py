"""The SSD is the cache. Nothing large is held in RAM between renders.

Everything in this app that is worth caching is a *large array*: a 24MP source
is 288MB of float32, a checkpoint frame at a 2400px proxy is 184MB, a Global
Grain texture layer is another 184MB, and a 16-bit PNG export is ~140MB of
encoded bytes. Held in process memory, the five caches that wanted them summed
to roughly twelve gigabytes on this machine before anything was even rendered
twice -- 8GB of uploads, 2GB of checkpoints, 2GB of textures -- and two of them
had no ceiling at all.

So the storage moved to the SSD and the RAM copy stopped existing. That is a
different arrangement from the usual "spill when full": there is no in-memory
tier to spill *from*. A `put` writes a file and returns; a `get` reads that file
and materialises a tensor which the caller owns and which dies with the render
that asked for it. Between renders the process holds an index of paths and byte
counts and nothing else.

**Why that is affordable here, when it usually is not.** Every one of these
caches exists to skip work measured in *seconds*, not milliseconds:

* a checkpoint hit skips 68-89% of a render -- 1.2s at a proxy, far more at 24MP
* a texture hit skips 1.29s of a 3.70s `Stock` proxy preview
* an upload's array skips a decode

A 184MB read from an NVMe SSD is 50-90ms. Losing a millisecond-scale cache to
disk latency would be absurd; trading 90ms for 1.2s is not close. The ratio is
what makes this a good trade rather than a resigned one, and it is worth
re-checking before anything *small* is added to a store here -- a cache whose
payload is cheaper to recompute than to read back belongs in neither tier.

**Write amplification is the real cost, and it is bounded deliberately.** A
slider drag re-renders continuously, and every render writes its checkpoints.
At a 2400px proxy that is ~180MB per boundary crossed, so sustained dragging
writes on the order of 150MB/s to the drive. Over a long editing session that
is real SSD wear, which is why `FILM_GRAIN_DISK_CACHE_GB=0` switches the stores
off outright rather than merely shrinking them -- it is the honest way to
measure what they are worth, and the escape hatch for anyone who would rather
spend RAM than write cycles.

**Never `/tmp`.** On most Linux distributions `/tmp` is a tmpfs, i.e. RAM with a
filesystem in front of it, so spilling there would move twelve gigabytes from
one part of memory to another and report success. The root is the platform's
*cache* directory, which is always real storage.

**Stale hits are the failure mode that matters**, and it is the same one
`checkpoint.py` warns about at length: a wrong frame is a plausible photograph.
Nothing here survives the process. The root is a per-run directory named for the
PID, removed on exit and swept on the next start, so a key can never be answered
by a previous run's bytes -- the keys embed upload ids and device names that are
themselves per-process, and a cache whose correctness depends on that had better
not outlive it.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import os
import re
import shutil
import sys
import threading
import uuid
import weakref
from pathlib import Path

import numpy as np
import torch

#: Total disk budget shared by the byte-capped stores, in bytes.
#:
#: Deliberately unrelated to `_render_budget_bytes`: that one divides *device
#: memory*, which is the scarce thing a render competes for, and the whole point
#: of this module is that these caches no longer take any of it. Disk is the
#: cheap resource, so the default is generous enough that the caches actually
#: hit -- the old device-memory shares were 15% apiece and on an 8GB machine
#: that meant 300MB against a texture set wanting 922MB, i.e. a partial hit rate
#: by construction.
#:
#: `FILM_GRAIN_DISK_CACHE_GB` overrides it. **0 switches every store off**, which
#: is how you measure what they are worth and how someone who would rather not
#: spend the write cycles opts out.
_DEFAULT_DISK_CACHE_GB = 8.0

#: Never take more than this share of what is actually free on the volume. The
#: budget above is a wish; a laptop with 6GB free must not have 8GB of grain
#: textures written into it, because the thing that breaks then is the user's
#: disk rather than this app.
_FREE_SPACE_SHARE = 0.25

#: How the budget above divides between the two byte-capped stores.
#:
#: Even, unlike the device-memory split in `device.py`, and for a reason that is
#: the whole point of this module: on the GPU the renderer had to be paid first
#: and the caches took 0.15 each of what was left, because a cache that shrinks
#: a tile makes the *uncached* path slower. Disk competes with nothing -- there
#: is no renderer taking a share of it -- so both stores simply get half and
#: both are large enough that their working sets fit, which is what turns a
#: measured 0% hit rate into a hit.
_CHECKPOINT_DISK_SHARE = 0.5
_GRAIN_DISK_SHARE = 0.5

_APP_DIR = "UpFilmGrain"

_LOCK = threading.Lock()


def _platform_cache_dir() -> Path:
    """The OS's cache directory -- real storage on every platform.

    Not `tempfile.gettempdir()`, and that is the one decision in this function
    worth defending: `/tmp` is a tmpfs on most Linux distributions, so a cache
    written there is held in RAM. This module exists to get large arrays *out*
    of RAM, and a root that silently put them back would leave every measurement
    here technically true and completely wrong.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / _APP_DIR
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
        return Path(base or Path.home()) / _APP_DIR / "Cache"
    base = os.environ.get("XDG_CACHE_HOME")
    return Path(base) / _APP_DIR if base else Path.home() / ".cache" / _APP_DIR


_RUN_RE = re.compile(r"^run-(\d+)-[0-9a-f]{8}$")


def _pid_alive(pid: int) -> bool:
    """Whether a process by that id exists. Used only to sweep dead runs."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        # EPERM means it exists and belongs to somebody else, which for our
        # purposes is "alive" -- we would not be able to remove its files
        # anyway, and guessing otherwise deletes a live run's cache.
        return e.errno == errno.EPERM
    except Exception:
        return True
    return True


def _sweep(base: Path, keep: Path) -> None:
    """Remove `run-*` directories left by processes that are no longer running.

    A crash or a `SIGKILL` skips `atexit`, so without this the cache directory
    accumulates a full run's spill every time the app dies badly -- and for an
    app that writes gigabytes per session, "accumulates" is the wrong verb for
    what that would do to a disk over a few months.

    A dead PID can be reused by an unrelated process, which would make this skip
    a directory it could have removed. That is the harmless direction: the next
    sweep gets it.
    """
    try:
        entries = list(base.iterdir())
    except OSError:
        return
    for d in entries:
        if d == keep or not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if m and not _pid_alive(int(m.group(1))):
            shutil.rmtree(d, ignore_errors=True)


#: This run's cache directory, or `None` when the disk cannot host one. Assigned
#: below, once `_make_root` has run; declared here because `cleanup` closes over
#: the module global rather than over a local, and is registered with `atexit`
#: from inside `_make_root` itself.
ROOT: Path | None = None


def cleanup() -> None:
    """Remove this run's cache directory. Idempotent, and never raises.

    Registered with `atexit`, and **called explicitly by `launch.py` on both
    paths that bypass it** -- a signal, and the parent watchdog's `os._exit`.
    Both matter here rather than being belt and braces: the desktop shell quits
    by sending SIGTERM, so the signal path is the *normal* way this app closes,
    and `atexit` does not run on it. Measured before that was wired up: every
    ordinary quit left a full run's spill behind, up to the disk budget plus a
    photograph's frames, and it sat there until the next start swept it.

    `_sweep` on the next start is still the backstop and still earns its place:
    nothing can run on `SIGKILL`, on a power cut, or on a crash inside the
    interpreter.

    Safe to call twice -- `ignore_errors` covers the second call finding nothing
    -- which it has to be, since the signal path and `atexit` can both fire.
    """
    if ROOT is not None:
        shutil.rmtree(ROOT, ignore_errors=True)


def _make_root() -> Path | None:
    """This run's cache directory, or `None` if the disk cannot host one.

    Returning `None` rather than raising is the point: a read-only home
    directory, a full disk or a sandbox that forbids the path are all reasons to
    render *without* caching, not reasons to refuse to start. Every store checks
    for it and degrades to a permanent miss, which is slower and correct.
    """
    env = os.environ.get("FILM_GRAIN_CACHE_DIR")
    base = Path(env).expanduser() if env else _platform_cache_dir()
    root = base / f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        # Prove it is writable now rather than discovering it on the first
        # 184MB write, halfway through a render.
        probe = root / ".probe"
        probe.write_bytes(b"1")
        probe.unlink()
    except OSError as e:
        print(f"[cache] cannot use {root}: {e}; caching to disk is off.",
              file=sys.stderr)
        return None
    _sweep(base, root)
    atexit.register(cleanup)
    return root


ROOT = _make_root()


def _budget_bytes() -> int:
    """The shared disk budget, clamped to what the volume can actually spare."""
    env = os.environ.get("FILM_GRAIN_DISK_CACHE_GB")
    gb = _DEFAULT_DISK_CACHE_GB
    if env:
        try:
            gb = float(env)
        except ValueError:
            pass
    want = int(gb * (1 << 30))
    if want <= 0 or ROOT is None:
        return 0
    try:
        free = shutil.disk_usage(ROOT).free
    except OSError:
        return want
    return max(0, min(want, int(free * _FREE_SPACE_SHARE)))


#: Every live store, for `stats()`. Weak, so a discarded engine's stores are not
#: kept alive by the reporting -- which would defeat the `weakref.finalize` that
#: deletes their files. Production has two and they live for the process; the
#: check suite makes dozens.
_STORES: "list[weakref.ref[DiskStore]]" = []


def _live_stores() -> "list[DiskStore]":
    out = []
    for r in list(_STORES):
        s = r()
        if s is None:
            _STORES.remove(r)
        else:
            out.append(s)
    return out


def _dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class DiskStore:
    """A byte-capped LRU of arrays whose only copy lives on the SSD.

    The API is the one `collections.OrderedDict`-as-an-LRU gave the caches this
    replaced -- `get`, `put`, `clear`, `nbytes`, `len` -- so the call sites read
    the same. What changed underneath is that a value is a *file*: `put` writes
    it and keeps a path, `get` reads it and hands back a fresh tensor.

    The index is in memory and the payload is not. That asymmetry is the whole
    design: an index entry is a key tuple, a path and an int, so ten thousand of
    them are a few megabytes, while the ten thousand arrays they name would be
    terabytes.

    Not thread-safe by inheritance from its callers -- `runtime.RENDER_LOCK`
    serialises every render -- but it takes a lock anyway, because uploads and
    exports reach the spill path from threads that lock is not holding.
    """

    def __init__(self, name: str, share: float) -> None:
        self.name = name
        #: This store's slice of `_budget_bytes()`. A share rather than an
        #: absolute figure so raising one store's allowance visibly lowers the
        #: others', the same argument `device.py` makes for the memory pool.
        self.share = share
        # **Its own directory, per instance, not per name.** Production has one
        # engine and would not notice; `tests/checks/` builds several in one
        # process, and the keys do not carry an engine identity -- a checkpoint
        # key is (image id, boundary, geometry, device, signature), all of which
        # two engines in one process can match exactly. Sharing a directory would
        # then let one engine answer the other's `get`, which is the stale hit
        # this whole subsystem is written around. Two in-memory dicts could never
        # do that; two directories with the same name can.
        self._dir = (
            ROOT / f"{_dirname(name)}-{uuid.uuid4().hex[:8]}"
            if ROOT is not None else None
        )
        if self._dir is not None:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                # A discarded engine takes its files with it. Without this, a
                # process that builds engines in a loop -- again, the check suite
                # -- leaves every one of their spills on the disk until exit.
                weakref.finalize(self, shutil.rmtree, self._dir,
                                 ignore_errors=True)
            except OSError:
                self._dir = None
        #: key -> (path, bytes), oldest first.
        self._idx: "dict[object, tuple[Path, int]]" = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evicted = 0
        #: Bytes ever written. Reported in the UI because it is the number that
        #: says what this is costing the drive, which no live size can.
        self.written = 0
        self._lk = threading.Lock()
        _STORES.append(weakref.ref(self))

    # -- capacity ---------------------------------------------------------- #

    @property
    def cap(self) -> int:
        """This store's byte ceiling, recomputed per call.

        Derived rather than frozen at construction so that
        `FILM_GRAIN_DISK_CACHE_GB` behaves like every other `FILM_GRAIN_*`
        override in this codebase -- a test that sets it gets what it asked for
        -- and so that a disk filling up during a long export session shrinks
        the cache instead of overrunning the volume.
        """
        return int(_budget_bytes() * self.share)

    @property
    def enabled(self) -> bool:
        return self._dir is not None and self.cap > 0

    # -- storage ----------------------------------------------------------- #

    def _path(self, key) -> Path:
        """A filename for `key`.

        A hash rather than an encoding of the key, because keys here carry
        floats, device strings and absolute tile coordinates and a filename has
        to be short and portable. Collisions are not defended against: blake2b
        at 128 bits is far past the point where a birthday collision across the
        few thousand entries a session produces is worth code.
        """
        h = hashlib.blake2b(repr(key).encode(), digest_size=16).hexdigest()
        return self._dir / f"{h}.npy"

    def get(self, key, device: torch.device | None = None) -> torch.Tensor | None:
        """The stored tensor, read back onto `device`, or `None`.

        A **fresh tensor every call**, never a shared one. Two callers holding
        the same cached object is exactly how an in-place stage corrupts a cache
        entry, and the old in-RAM dicts were one careless `mul_` away from it;
        reading from disk makes independence structural rather than a rule
        anybody has to remember.

        On CPU the array is mapped rather than read (`mmap_mode="c"`), so the
        pages are file-backed and the kernel can evict them under pressure --
        the resident cost of a hit is then whatever is actually being touched
        rather than the whole 184MB. Copy-on-write, so a stage that does write
        into it gets a private page and the file on disk stays the value that
        was cached. On an accelerator the bytes have to be copied to the device
        anyway, so there is nothing to map.
        """
        if self._dir is None:
            self.misses += 1
            return None
        with self._lk:
            ent = self._idx.get(key)
            if ent is None:
                self.misses += 1
                return None
            path, _ = ent
            # Move to the end: plain `dict` has preserved insertion order since
            # 3.7 and re-inserting is how you promote in one, which is all
            # `OrderedDict.move_to_end` was doing for these caches.
            self._idx[key] = self._idx.pop(key)
        try:
            cpu = device is None or device.type == "cpu"
            arr = np.load(path, mmap_mode="c" if cpu else None)
        except (OSError, ValueError):
            # The file vanished or is unreadable -- a full disk truncating a
            # write, or someone clearing the folder underneath us. Drop the
            # index entry and report a miss; the caller recomputes, which is
            # always a legal answer.
            self._forget(key)
            self.misses += 1
            return None
        t = torch.from_numpy(arr)
        if device is not None and device.type != "cpu":
            t = t.to(device)
        self.hits += 1
        return t

    def put(self, key, t: torch.Tensor) -> None:
        """Write `t` under `key`, evicting oldest-first to stay under `cap`."""
        if not self.enabled:
            return
        cap = self.cap
        arr = t.detach().to("cpu").contiguous().numpy()
        n = int(arr.nbytes)
        # An entry larger than the whole budget is not stored rather than
        # immediately evicting itself. Same reasoning the in-memory version
        # carried: a single huge tile would otherwise empty the store on every
        # pass and pay the bookkeeping for the privilege.
        if n > cap:
            return
        path = self._path(key)
        tmp = path.with_suffix(".npy.part")
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, arr, allow_pickle=False)
            # Atomic within the directory, so a reader can never see a partial
            # file. It costs nothing here and removes a whole class of "the
            # cache returned half a frame" that would be near-impossible to
            # reproduce.
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        with self._lk:
            prev = self._idx.pop(key, None)
            if prev is not None:
                # Re-putting is legal -- a caller that stored on a miss and
                # stores again later is doing nothing wrong -- but counting the
                # bytes twice is not, and the symptom is a total that climbs
                # while the entry count stays flat.
                self._bytes -= prev[1]
            self._idx[key] = (path, n)
            self._bytes += n
            self.written += n
            while self._bytes > cap and len(self._idx) > 1:
                old_key = next(iter(self._idx))
                self._unlink(old_key)

    # -- eviction ---------------------------------------------------------- #

    def _unlink(self, key) -> None:
        """Drop one entry. Caller holds `_lk`."""
        ent = self._idx.pop(key, None)
        if ent is None:
            return
        self._bytes -= ent[1]
        self.evicted += 1
        try:
            ent[0].unlink()
        except OSError:
            pass

    def _forget(self, key) -> None:
        with self._lk:
            ent = self._idx.pop(key, None)
            if ent is not None:
                self._bytes -= ent[1]

    def drop_if(self, pred) -> int:
        """Delete every entry whose key satisfies `pred`. Returns the count.

        Both caches that use this store evict by *generation* -- a parameter
        state that can never be asked for again -- rather than waiting for the
        byte cap to notice. That reasoning is unchanged by the move to disk and
        is documented where it is decided (`checkpoint.py`, `global_grain.py`);
        this is only the mechanism.
        """
        with self._lk:
            doomed = [k for k in self._idx if pred(k)]
            for k in doomed:
                self._unlink(k)
        return len(doomed)

    def clear(self) -> None:
        with self._lk:
            for k in list(self._idx):
                self._unlink(k)
            self.evicted = 0
            self._bytes = 0

    # -- reporting --------------------------------------------------------- #

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._idx)

    def __iter__(self):
        """Iterate the keys, oldest first -- the `OrderedDict` this replaced.

        Kept because the cache checks in `tests/checks/` inspect *which* keys
        survived an eviction sweep rather than only how many, which is the whole
        point of those tests: a stale-cache bug renders a plausible texture, so
        counting entries cannot tell a correct sweep from a lucky one.

        A list copy rather than the live view, so a caller may evict while
        iterating -- which is exactly what a sweep does.
        """
        return iter(list(self._idx))

    def __contains__(self, key) -> bool:
        return key in self._idx

    def stats(self) -> dict:
        return {
            "name": self.name,
            "bytes": self._bytes,
            "entries": len(self._idx),
            "cap": self.cap,
            "hits": self.hits,
            "misses": self.misses,
            "evicted": self.evicted,
            "written": self.written,
        }


class Spill:
    """One array whose only copy is a file, mapped in when something reads it.

    The other half of this module, and a different shape from `DiskStore`
    because it answers a different question. A store is a *cache*: a miss is
    legal and the caller recomputes. A `Spill` holds something that cannot be
    recomputed -- the decoded pixels of the photograph the user uploaded -- so
    it is a place to keep a value, not a place to maybe find one.

    `array` returns a copy-on-write memmap, so the pages are file-backed and
    evictable and the resident cost is whatever is actually being touched. That
    is what takes a 24MP source from 288MB of anonymous memory the OS can only
    swap to 288MB of clean file pages it can simply drop -- and it is why this
    is a memmap rather than a `np.load` on demand: an explicit read would put
    the whole array back in RAM on every access, which is the thing being fixed.

    Mode ``"c"`` rather than ``"r"`` for one specific reason: `torch.from_numpy`
    refuses a read-only array, and `imageio._interp` -- which every resample in
    the app goes through -- calls exactly that. Copy-on-write gives a writable
    view whose unwritten pages stay shared with the file.
    """

    __slots__ = ("path", "shape", "dtype", "nbytes", "_arr", "_fallback")

    #: All live spills, for `stats()`. A `Spill` is released explicitly by its
    #: owner (`Upload.release`), so this is a plain list of what exists rather
    #: than anything with a policy.
    _live: "list[Spill]" = []

    def __init__(self, kind: str, arr: np.ndarray) -> None:
        self.shape = tuple(arr.shape)
        self.dtype = arr.dtype
        self.nbytes = int(arr.nbytes)
        self.path: Path | None = None
        # Held in RAM only when there is no disk to hold it -- see `_make_root`.
        # Keeping the array rather than failing is what makes the disk cache an
        # optimisation rather than a dependency.
        self._fallback: np.ndarray | None = None
        self._arr = None
        d = ROOT / "frames" if ROOT is not None else None
        if d is None:
            self._fallback = arr
            Spill._live.append(self)
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{_dirname(kind)}-{uuid.uuid4().hex[:12]}.npy"
            with open(path, "wb") as fh:
                np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
            self.path = path
        except OSError:
            self._fallback = arr
        Spill._live.append(self)

    @property
    def array(self) -> np.ndarray:
        """The array, mapped from disk. Cheap enough to call per access."""
        if self._fallback is not None:
            return self._fallback
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="c")
        return self._arr

    def unmap(self) -> None:
        """Drop the mapping, keeping the file.

        The mapping itself is a few kilobytes of page table, so this is not
        about the memmap object. It is about the *pages*: dropping the map is
        what lets the kernel reclaim every clean page it faulted in, which is
        the "flush RAM after the render" half of the arrangement. The next
        `array` re-maps and the pages come back from the SSD as they are
        touched.
        """
        self._arr = None

    def release(self) -> None:
        """Delete the file. The array is gone after this; nothing may read it."""
        self._arr = None
        self._fallback = None
        if self.path is not None:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.path = None
        try:
            Spill._live.remove(self)
        except ValueError:
            pass

    @classmethod
    def unmap_all(cls) -> None:
        for s in list(cls._live):
            s.unmap()

    @classmethod
    def total_bytes(cls) -> int:
        return sum(s.nbytes for s in cls._live)

    @classmethod
    def count(cls) -> int:
        return len(cls._live)


class Blob:
    """Encoded bytes on disk -- a finished export waiting to be downloaded.

    The third shape in this module, and it exists because `JOBS` had no ceiling
    at all: a finished export's bytes were kept on the job dict for the life of
    the process, and a 24MP 16-bit PNG is ~140MB. Ten exports in a session was
    1.4GB of RAM that nothing would ever free -- not a cache with a bad policy,
    an outright leak, and the only one of the five holders this change touched
    that was a bug rather than a trade.

    Bytes rather than an array, so this is neither a `DiskStore` (which may
    answer `None`) nor a `Spill` (which maps a typed array). What it gives the
    controller is a `path`, which is what `FileResponse` wants: the file is then
    streamed to the client by the OS and never passes through Python's heap on
    the way out, which is the second half of the saving.
    """

    __slots__ = ("path", "nbytes", "_data")

    _live: "list[Blob]" = []

    def __init__(self, kind: str, data: bytes) -> None:
        self.nbytes = len(data)
        self.path: Path | None = None
        self._data: bytes | None = None
        d = ROOT / "exports" if ROOT is not None else None
        if d is not None:
            try:
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{_dirname(kind)}-{uuid.uuid4().hex[:12]}.bin"
                path.write_bytes(data)
                self.path = path
            except OSError:
                self._data = data
        else:
            self._data = data
        Blob._live.append(self)

    @property
    def data(self) -> bytes:
        """The bytes, read back. Only for the no-disk fallback path and tests --
        the download route serves `path` directly and must not call this."""
        if self._data is not None:
            return self._data
        try:
            return self.path.read_bytes()
        except OSError:
            return b""

    def release(self) -> None:
        self._data = None
        if self.path is not None:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.path = None
        try:
            Blob._live.remove(self)
        except ValueError:
            pass

    @classmethod
    def total_bytes(cls) -> int:
        return sum(b.nbytes for b in cls._live)

    @classmethod
    def count(cls) -> int:
        return len(cls._live)


def _mac_rss() -> int:
    """Current resident size on macOS, via `libproc`. 0 if it cannot be read.

    `resource.getrusage` only offers a *peak* and there is no `psutil` in the
    Pipfile, so this asks the kernel directly. `PROC_PIDTASKINFO` fills a
    96-byte `proc_taskinfo` whose first two fields are the virtual and resident
    sizes; the rest is declared so the struct is the size the call checks for.
    """
    import ctypes
    import ctypes.util

    class _TaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
        ] + [(f"_i{i}", ctypes.c_int32) for i in range(12)]

    lib = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
    ti = _TaskInfo()
    n = lib.proc_pidinfo(
        os.getpid(), 4, ctypes.c_uint64(0), ctypes.byref(ti),
        ctypes.sizeof(ti),
    )
    return int(ti.pti_resident_size) if n == ctypes.sizeof(ti) else 0


def _win_mem() -> tuple[int, int]:
    """(current, peak) working set on Windows. (0, 0) if it cannot be read."""
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
        return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)
    return 0, 0


def memory() -> dict:
    """This process's resident size now, and the highest it has reached.

    Reported beside the disk figures because it is the number the user actually
    asked about, and because the two are only meaningful together: "3GB on the
    SSD" says nothing on its own about whether the app is behaving.

    **Both numbers, not one, and the current one is the headline.** `ru_maxrss`
    is a high-water mark that never comes down, so on its own it cannot show
    memory being *given back* -- which is the entire subject of this module, and
    would make the readout unable to display the thing it exists to display. The
    peak is kept beside it because it is the honest answer to "how bad does this
    get", and because a render's own working set is a real cost that a
    current-only figure taken between renders would hide.

    No dependency: `psutil` is not in the Pipfile and this is not worth adding
    one for. Every platform's quirk is absorbed here rather than in the client --
    `ru_maxrss` is bytes on macOS and kilobytes on Linux, and neither has a
    current figure, so macOS goes through `libproc` and Linux through
    `/proc/self/statm`. Anything unreadable reports 0, which the client renders
    as "--" rather than as zero bytes.
    """
    cur = peak = 0
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(raw if sys.platform == "darwin" else raw * 1024)
    except Exception:
        pass
    try:
        if sys.platform == "darwin":
            cur = _mac_rss()
        elif sys.platform == "win32":
            cur, peak = _win_mem()
        else:
            with open("/proc/self/statm") as fh:
                cur = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    return {"rss": cur, "peak": peak}


def stats() -> dict:
    """Everything the cache readout in the UI reports.

    One endpoint's worth of numbers, assembled here rather than in the
    controller, because what counts as "the cache" is this module's business and
    a controller that itemised the stores would need editing every time one is
    added.
    """
    stores = [s.stats() for s in _live_stores()]
    frames = {"name": "frames", "bytes": Spill.total_bytes(),
              "entries": Spill.count()}
    exports = {"name": "exports", "bytes": Blob.total_bytes(),
               "entries": Blob.count()}
    parts = stores + [frames, exports]
    return {
        "enabled": ROOT is not None,
        "root": str(ROOT) if ROOT is not None else None,
        "budget": _budget_bytes(),
        "bytes": sum(x["bytes"] for x in parts),
        "written": sum(x["written"] for x in stores),
        "memory": memory(),
        "parts": parts,
    }
