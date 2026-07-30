"""fetch_v2: coalesced, persistent-connection expert fetcher for lazy-K3.

Drop-in replacement for k3loader's expert fetch path:
    fetch_expert_raw(layer, eid) -> {"w1": (packed u8, scale u8), "w2": ..., "w3": ...}
    fetch_experts(layer, eids, workers=4, dequant=...)  (same signature/semantics)

Differences vs baseline (k3loader._range_fetch):
  * ONE Range request per expert (17.55 MB contiguous span) instead of 6.
  * The HF resolve -> CDN 302 redirect is resolved ONCE per shard and cached
    (signed URL, ~1h validity; auto re-resolve on 403/expiry).
  * Persistent keep-alive connections to the CDN host, small pool (default 4).
  * Optional multi-expert span coalescing in fetch_experts() when selected
    experts are file-adjacent (lexicographic eid order, zero-gap layout).
  * Optional httpx HTTP/2 backend (BACKEND="httpx") for benchmarking.

Expert layout facts (verified against tensor_inventory_offsets.json for all
82432 experts): each expert's 6 tensors are contiguous in-shard in order
w1_p, w1_s, w2_p, w2_s, w3_p, w3_s with fixed sizes; each MoE layer's 896
experts occupy ONE shard, back-to-back with zero gaps, sorted by str(eid).
"""
import atexit
import collections
import http.client
import json
import mmap
import os
import re
import ssl
import stat
import sys
import threading

try:
    # Only ever called behind a Darwin guard, for F_NOCACHE.
    import fcntl
except ImportError:  # Windows has no fcntl module
    fcntl = None
import time
import urllib.parse
import concurrent.futures

import numpy as np

try:
    from cache_writer import AsyncCacheWriter, atomic_publish
    import model_source
    import positional_io
    import runtime_platform
except ImportError:  # imported as tools.fetch_v2 instead of a top-level module
    from .cache_writer import AsyncCacheWriter, atomic_publish
    from . import model_source, positional_io, runtime_platform

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV_PATH = os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")
BASE_HOST = model_source.hf_host()
BASE_PATH = model_source.hf_path()
ECACHE = os.path.join(ROOT, "k3-experts")
os.makedirs(ECACHE, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
IDX_PATH = os.path.join(_HERE, "expert_index.npy")
IDX_META_PATH = os.path.join(_HERE, "expert_index.meta.json")

# ---- fixed intra-expert layout (offset, nbytes, shape) --------------------
EXPERT_SPAN = 17547264
_P, _S = 5505024, 344064
LAYOUT = [
    ("w1_p", 0,            _P, (3072, 1792)),
    ("w1_s", _P,           _S, (3072, 112)),
    ("w2_p", _P + _S,      _P, (3584, 1536)),
    ("w2_s", 2 * _P + _S,  _S, (3584, 96)),
    ("w3_p", 2 * (_P + _S), _P, (3072, 1792)),
    ("w3_s", 3 * _P + 2 * _S, _S, (3072, 112)),
]

USER_AGENT = "k3-lazy-fetch/2.0"
MAX_CONNS = int(os.environ.get("K3_FETCH_CONNS", "4"))
BACKEND = os.environ.get("K3_FETCH_BACKEND", "httpclient")  # or "httpx"
MAX_COALESCE = int(os.environ.get("K3_FETCH_COALESCE", "4"))  # experts per merged range

# ---- local read path (K3_EXPERT_READ) -------------------------------------
# "mmap"  : np.memmap views over the .bin cache; the pages are demand-faulted
#           later, inside the GEMV kernel. Measured 0.87-0.93 GB/s.
# "pread" : the whole layer's selected experts are pread() in parallel by a
#           persistent worker pool into reused page-aligned buffers, so the
#           kernel only ever touches resident memory. Measured ~7.0 GB/s.
EXPERT_READ = os.environ.get("K3_EXPERT_READ", "pread")
# Real F_NOCACHE grid on the 8 TB M1 Max SSD: 8 workers sustained 7.51
# logical GB/s versus 6.91/6.69/6.91 for 4/6/12.  Keep the env override: newer
# chips and different SSD widths must rerun tools/bench_expert_io_grid.py.
PREAD_WORKERS = runtime_platform.configured_cpu_workers(
    "K3_PREAD_WORKERS", 8
)
_IS_DARWIN = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")


def _pread_nocache_default(platform):
    return "1" if platform == "darwin" else "0"


# Darwin's proven default uses F_NOCACHE before the read. Linux stays buffered
# by default; an explicit override discards only these completed expert ranges
# with POSIX_FADV_DONTNEED rather than issuing a Darwin fcntl command number.
PREAD_NOCACHE = os.environ.get(
    "K3_PREAD_NOCACHE", _pread_nocache_default(sys.platform)
) == "1"
# free-slot high-water mark; prefill unions (up to 5 x 16 experts/layer) grow the
# pool transiently and are trimmed back to this after the layer is done.
PREAD_MAX_FREE = int(os.environ.get("K3_PREAD_MAX_FREE", "40"))
# how many read_layer() results stay valid at once (buffer recycling depth).
PREAD_DEPTH = int(os.environ.get("K3_PREAD_DEPTH", "2"))
# X4 opt-in: replace one concurrent.futures.Future per raw expert with a
# persistent fixed-width worker ring.  The default remains the proven
# ThreadPoolExecutor path until repeated physical-read layer benchmarks win on
# each hardware/SSD combination.  K3_PREAD_WORKERS remains the capability-keyed
# width knob; never inherit this M1's value blindly on M5.
PREAD_RING = os.environ.get("K3_PREAD_RING", "0") == "1"
# Cache-miss requests historically publish their downloaded bytes synchronously.
# Q13 is opt-in until a real miss-path/full-token A/B establishes that moving
# writes off the request thread beats the extra retained-buffer pressure.
ASYNC_CACHE_WRITE = os.environ.get("K3_ASYNC_CACHE_WRITE", "0") == "1"
CACHE_WRITE_QUEUE = max(1, int(os.environ.get("K3_CACHE_WRITE_QUEUE", "4")))
CACHE_WRITE_WORKERS = max(1, int(os.environ.get("K3_CACHE_WRITE_WORKERS", "1")))
if CACHE_WRITE_WORKERS > CACHE_WRITE_QUEUE:
    CACHE_WRITE_WORKERS = CACHE_WRITE_QUEUE
F_NOCACHE = 48          # <sys/fcntl.h>, Darwin: bypass the unified buffer cache

stats = {
    "expert_http": 0, "expert_disk": 0, "http_bytes": 0, "http_s": 0.0,
    "requests": 0, "new_conns": 0, "conn_s": 0.0, "resolves": 0,
    "resolve_s": 0.0, "retries": 0, "coalesced_spans": 0,
    "pread_experts": 0, "pread_bytes": 0, "pread_s": 0.0, "pread_slots": 0,
    "prefetch_hits": 0, "prefetch_wasted": 0, "npz_experts": 0,
    "grouped_calls": 0, "grouped_yields": 0, "grouped_fallbacks": 0,
    "pread_ring_batches": 0, "pread_ring_jobs": 0,
    "pread_ring_canceled": 0, "pread_ring_failures": 0,
    "pread_retries": 0, "pread_local_failures": 0,
    "cache_write_enqueued": 0, "cache_write_sync": 0,
    "cache_write_sync_fallbacks": 0, "cache_write_completed": 0,
    "cache_write_failures": 0, "cache_write_bytes": 0,
}
_stats_lock = threading.Lock()
_cache_observer = None
_cache_writer_instance = None
_cache_writer_lock = threading.Lock()
_cache_writer_atexit_registered = False


def _apply_pread_nocache(fd):
    """Before-read Darwin cache bypass; command 48 is invalid on Linux."""
    if not (_IS_DARWIN and PREAD_NOCACHE):
        return False
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return True


def _drop_linux_pread_cache(fd):
    """After-read Linux cache eviction for the explicit nocache override."""
    if not (_IS_LINUX and PREAD_NOCACHE):
        return False
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return False
    try:
        advise(fd, 0, 0, dontneed)
    except OSError:
        return False
    return True


def _initial_raw_presence():
    """Validate raw expert spans once; hot-path membership is then syscall-free."""
    present = set()
    try:
        entries = os.scandir(ECACHE)
    except FileNotFoundError:
        return present
    with entries:
        for ent in entries:
            if not ent.name.endswith(".bin"):
                continue
            try:
                info = ent.stat(follow_symlinks=False)
                if (
                    stat.S_ISREG(info.st_mode)
                    and info.st_size == EXPERT_SPAN
                ):
                    present.add(ent.name)
            except OSError:
                continue
    return present


_raw_presence = _initial_raw_presence()
_raw_presence_lock = threading.Lock()


def _bump(**kw):
    with _stats_lock:
        for k, v in kw.items():
            stats[k] += v


def set_cache_observer(observer):
    """Install a cheap callback invoked after an expert file is published."""
    global _cache_observer
    _cache_observer = observer


def _cache_write_published(path, nbytes):
    # Atomic rename is the visibility boundary. Presence and telemetry are
    # published only afterward, from the same writer thread.
    with _raw_presence_lock:
        _raw_presence.add(os.path.basename(path))
    _bump(cache_write_completed=1, cache_write_bytes=nbytes)
    if _cache_observer is not None:
        _cache_observer(path)


def _cache_write_error(path, stage, exc):
    # The demand request may already have returned its downloaded bytes. Keep
    # persistence failures observable; an unpublished path remains absent and
    # will be fetched again on the next demand.
    del path, stage, exc
    _bump(cache_write_failures=1)


def _shutdown_cache_writer_at_exit():
    try:
        status = shutdown_cache_writes(raise_errors=False)
        if status["failures"]:
            print(
                "[fetch_v2] asynchronous cache write failures: "
                f"{status['failures']} (inspect cache_write_status())",
                flush=True,
            )
    except Exception as exc:
        print(f"[fetch_v2] cache writer shutdown failed: {exc}", flush=True)


def _get_cache_writer():
    global _cache_writer_instance, _cache_writer_atexit_registered
    with _cache_writer_lock:
        if _cache_writer_instance is None:
            _cache_writer_instance = AsyncCacheWriter(
                max_pending=CACHE_WRITE_QUEUE,
                workers=CACHE_WRITE_WORKERS,
                name="k3-cache-write",
            )
        if not _cache_writer_atexit_registered:
            atexit.register(_shutdown_cache_writer_at_exit)
            _cache_writer_atexit_registered = True
        return _cache_writer_instance


def cache_write_status():
    """Return live async-publication state without blocking."""
    if _cache_writer_instance is None:
        return {
            "submitted": 0, "completed": 0, "publish_failures": 0,
            "callback_failures": 0, "failures": 0, "bytes": 0,
            "pending": 0, "pending_peak": 0, "backpressure_events": 0,
            "backpressure_s": 0.0, "recent_failures": [],
            "accepting": ASYNC_CACHE_WRITE, "stopped": False,
            "max_pending": CACHE_WRITE_QUEUE, "workers": CACHE_WRITE_WORKERS,
            "alive_workers": 0,
        }
    return _cache_writer_instance.snapshot()


def flush_cache_writes(raise_errors=True):
    """Drain accepted cache writes; synchronous mode is an immediate no-op."""
    if _cache_writer_instance is None:
        return cache_write_status()
    return _cache_writer_instance.flush(raise_errors=raise_errors)


def shutdown_cache_writes(raise_errors=True):
    """Permanently stop and drain this process's asynchronous cache writer."""
    if _cache_writer_instance is None:
        return cache_write_status()
    return _cache_writer_instance.shutdown(raise_errors=raise_errors)


# ---------------------------------------------------------------------------
# expert index: (layer, eid) -> (shard name, absolute byte start)
# ---------------------------------------------------------------------------
_index = None          # dict (L, E) -> (shard_num, start)
_shard_template = None
_index_lock = threading.Lock()


def _build_index():
    inv = json.load(open(INV_PATH))
    pat = re.compile(r"language_model\.model\.layers\.(\d+)\.block_sparse_moe"
                     r"\.experts\.(\d+)\.w1\.weight_packed$")
    rows, template = [], None
    for name, t in inv.items():
        m = pat.match(name)
        if not m:
            continue
        L, E = int(m.group(1)), int(m.group(2))
        start = 8 + t["hlen"] + t["offsets"][0]
        snum = int(t["shard"].split("-")[1])
        rows.append((L, E, snum, start, start + EXPERT_SPAN))
        if template is None:
            template = re.sub(r"(?<=model-)\d+", "{:05d}", t["shard"], count=1)
    arr = np.array(sorted(rows), dtype=np.int64)
    tmp = IDX_PATH + f".tmp{os.getpid()}"
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp, IDX_PATH)
    json.dump({"shard_template": template}, open(IDX_META_PATH, "w"))
    return arr, template


def _get_index():
    global _index, _shard_template
    if _index is not None:
        return _index
    with _index_lock:
        if _index is None:
            if os.path.exists(IDX_PATH) and os.path.exists(IDX_META_PATH):
                arr = np.load(IDX_PATH)
                _shard_template = json.load(open(IDX_META_PATH))["shard_template"]
            else:
                arr, _shard_template = _build_index()
            _index = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in arr}
    return _index


def expert_span(layer, eid):
    """(shard_name, abs_start, abs_end) for one expert's contiguous 6-tensor span."""
    snum, start = _get_index()[(layer, eid)]
    return _shard_template.format(snum), start, start + EXPERT_SPAN


# ---------------------------------------------------------------------------
# redirect resolver: shard -> signed CDN URL (cached, thread-safe)
# ---------------------------------------------------------------------------
_ssl_ctx = ssl.create_default_context()
_resolved = {}          # shard -> (host, path_with_query, expires_epoch)
_resolve_lock = threading.Lock()


def _resolve(shard, force=False):
    now = time.time()
    with _resolve_lock:
        ent = _resolved.get(shard)
        if ent and not force and ent[2] - now > 300:
            return ent
    t0 = time.time()
    c = http.client.HTTPSConnection(BASE_HOST, timeout=30, context=_ssl_ctx)
    try:
        c.request("HEAD", BASE_PATH + shard,
                  headers={"User-Agent": USER_AGENT})
        r = c.getresponse()
        r.read()
        if r.status not in (301, 302, 303, 307, 308):
            raise IOError(f"resolve {shard}: expected redirect, got {r.status}")
        loc = r.getheader("Location")
    finally:
        c.close()
    u = urllib.parse.urlsplit(loc)
    q = urllib.parse.parse_qs(u.query)
    exp = int(q.get("Expires", [now + 2700])[0])
    path = u.path + ("?" + u.query if u.query else "")
    ent = (u.netloc, path, exp)
    with _resolve_lock:
        _resolved[shard] = ent
    _bump(resolves=1, resolve_s=time.time() - t0)
    return ent


# ---------------------------------------------------------------------------
# persistent connection pool (http.client backend)
# ---------------------------------------------------------------------------
class _Pool:
    def __init__(self, max_conns=MAX_CONNS):
        self._sem = threading.BoundedSemaphore(max_conns)
        self._idle = []          # [(host, conn)]
        self._lock = threading.Lock()

    def _get(self, host):
        self._sem.acquire()
        with self._lock:
            for i, (h, c) in enumerate(self._idle):
                if h == host:
                    self._idle.pop(i)
                    return c
            # evict one idle conn to a stale host if pool is saturated with them
            if self._idle:
                _, c = self._idle.pop(0)
                c.close()
        t0 = time.time()
        c = http.client.HTTPSConnection(host, timeout=120, context=_ssl_ctx)
        c.connect()
        _bump(new_conns=1, conn_s=time.time() - t0)
        return c

    def _put(self, host, conn, reusable):
        with self._lock:
            if reusable:
                self._idle.append((host, conn))
        if not reusable:
            try:
                conn.close()
            except Exception:
                pass
        self._sem.release()

    def range_get(self, shard, start, size, retries=6):
        """Fetch [start, start+size) of shard via persistent conn; returns bytes."""
        last = None
        for attempt in range(retries):
            host, path, _ = _resolve(shard, force=attempt >= 2)
            conn = self._get(host)
            reusable = False
            try:
                t0 = time.time()
                conn.request("GET", path, headers={
                    "Range": f"bytes={start}-{start + size - 1}",
                    "User-Agent": USER_AGENT,
                })
                r = conn.getresponse()
                if r.status == 403:          # signed URL expired
                    r.read()
                    _resolve(shard, force=True)
                    raise IOError("403 signed-url expired")
                if r.status != 206:
                    r.read()
                    raise IOError(f"status {r.status}")
                buf = r.read(size)
                if len(buf) != size:
                    raise IOError(f"short read {len(buf)}/{size}")
                reusable = not r.will_close
                _bump(requests=1, http_bytes=size, http_s=time.time() - t0)
                return buf
            except Exception as e:
                last = e
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                _bump(retries=1)
                if attempt == retries - 1:
                    raise
                time.sleep(min(1.5 * attempt, 6.0))
            finally:
                if conn is not None:
                    self._put(host, conn, reusable)
                else:
                    self._sem.release()
        raise last


_pool = _Pool()

# ---------------------------------------------------------------------------
# optional httpx HTTP/2 backend
# ---------------------------------------------------------------------------
_httpx_client = None
_httpx_lock = threading.Lock()


def _get_httpx():
    global _httpx_client
    if _httpx_client is None:
        with _httpx_lock:
            if _httpx_client is None:
                import httpx
                _httpx_client = httpx.Client(
                    http2=True, timeout=120.0,
                    limits=httpx.Limits(max_connections=MAX_CONNS,
                                        max_keepalive_connections=MAX_CONNS),
                    headers={"User-Agent": USER_AGENT})
    return _httpx_client


def _httpx_range_get(shard, start, size, retries=6):
    last = None
    for attempt in range(retries):
        host, path, _ = _resolve(shard, force=attempt >= 2)
        try:
            t0 = time.time()
            r = _get_httpx().get(
                f"https://{host}{path}",
                headers={"Range": f"bytes={start}-{start + size - 1}"})
            if r.status_code == 403:
                _resolve(shard, force=True)
                raise IOError("403 signed-url expired")
            if r.status_code != 206:
                raise IOError(f"status {r.status_code}")
            buf = r.content
            if len(buf) != size:
                raise IOError(f"short read {len(buf)}/{size}")
            _bump(requests=1, http_bytes=size, http_s=time.time() - t0)
            return buf
        except Exception as e:
            last = e
            _bump(retries=1)
            if attempt == retries - 1:
                raise
            time.sleep(min(1.5 * attempt, 6.0))
    raise last


def _range_get(shard, start, size):
    if BACKEND == "httpx":
        return _httpx_range_get(shard, start, size)
    return _pool.range_get(shard, start, size)


# ---------------------------------------------------------------------------
# threaded pread reader (K3_EXPERT_READ=pread)
# ---------------------------------------------------------------------------
# Why this exists: the mmap path hands the GEMV kernel views over file-backed
# pages that are still on disk, so the kernel demand-faults them while it
# computes — measured 0.87-0.93 GB/s against 7.0 GB/s for a threaded pread pool
# on the same files.  Here the whole layer's 16 experts (281 MB) are read in
# parallel before the kernel starts, into page-aligned buffers that are recycled
# across layers.  F_NOCACHE keeps 25.8 GB/token of expert traffic from evicting
# the page cache the resident spine relies on.
#
# EXPERT_SPAN and every intra-expert offset are exact multiples of 16 KB
# (17547264 = 1071*16K, _P = 336*16K, _S = 21*16K), so an anonymous-mmap slot is
# always correctly aligned for an unbuffered read and metal_moe's zero-copy
# contiguity test (base + off) still holds on these buffers.
#
# CAVEAT for tools/metal_moe.py: it caches MTLBuffer wraps keyed by base pointer
# and assumes one pointer == one expert forever. Slots here are recycled, so the
# same pointer carries a different expert a couple of layers later. The wrap
# aliases the same memory so the GPU does read the fresh bytes, but the pin LRU
# in metal_moe becomes meaningless — use metal_moe.raw_from_cache() (persistent
# per-expert mmaps) rather than these buffers if that path is ever wired up.


class _Slot:
    """One recyclable page-aligned EXPERT_SPAN buffer + its numpy views."""
    __slots__ = ("mm", "mv", "arr", "raw")

    def __init__(self):
        self.mm = mmap.mmap(-1, EXPERT_SPAN)
        self.mv = memoryview(self.mm)
        self.arr = np.ndarray((EXPERT_SPAN,), dtype=np.uint8, buffer=self.mm)
        d = {}
        for name, off, nb, shape in LAYOUT:
            w, kind = name.split("_")
            d.setdefault(w, {})[kind] = self.arr[off:off + nb].reshape(shape)
        # stable dict: the views never move, so this is built once per slot
        self.raw = {w: (v["p"], v["s"]) for w, v in d.items()}
        _bump(pread_slots=1)

    def read(self, path):
        source = positional_io.open_positional(path)
        try:
            fd = source.fileno()
            _apply_pread_nocache(fd)
            got = source.read_into(self.mv, 0)
            if got != EXPERT_SPAN:
                raise IOError(f"short read {got}/{EXPERT_SPAN} from {path}")
            _drop_linux_pread_cache(fd)
        finally:
            source.close()


class GroupReadError(IOError):
    """A grouped local read failed after the pipeline had already started.

    Callers must discard partial output and take the ordinary fetch path.  The
    generator settles every outstanding future and retires its slots before
    propagating this exception.
    """


class LocalExpertReadError(IOError):
    """A known-valid local expert failed both its pooled read and retry."""


_RING_QUEUED = 0
_RING_RUNNING = 1
_RING_DONE = 2
_RING_CANCELED = 3
_RING_FAILED = 4


class _RingBatch:
    """One shared completion record for a submitted expert set.

    A ThreadPoolExecutor allocates one full ``Future`` per expert.  The ring
    instead allocates one condition and compact parallel state arrays per
    layer/prefetch set.  Handles are intentionally tiny and implement only the
    ``cancel``/``result`` contract used by the reader.
    """

    __slots__ = (
        "slots", "paths", "states", "errors", "condition", "remaining",
        "successes", "canceled", "failures", "read_s", "stats_published",
        "next_index",
    )

    def __init__(self, slots, paths):
        self.slots = tuple(slots)
        self.paths = tuple(paths)
        self.states = [_RING_QUEUED] * len(self.slots)
        self.errors = [None] * len(self.slots)
        self.condition = threading.Condition()
        self.remaining = len(self.slots)
        self.successes = 0
        self.canceled = 0
        self.failures = 0
        self.read_s = 0.0
        self.stats_published = False
        self.next_index = 0

    def _publish_if_complete_locked(self):
        if self.remaining or self.stats_published:
            return None
        self.stats_published = True
        return {
            "pread_experts": self.successes,
            "pread_bytes": self.successes * EXPERT_SPAN,
            "pread_s": self.read_s,
            "pread_ring_canceled": self.canceled,
            "pread_ring_failures": self.failures,
        }

    @staticmethod
    def _publish(metrics):
        if metrics is not None:
            _bump(**metrics)

    def claim(self):
        """Claim the next uncancelled expert for one persistent worker-drain."""
        with self.condition:
            while (self.next_index < len(self.states)
                   and self.states[self.next_index] != _RING_QUEUED):
                self.next_index += 1
            if self.next_index == len(self.states):
                return None
            index = self.next_index
            self.next_index += 1
            self.states[index] = _RING_RUNNING
            return index

    def finish(self, index, error, elapsed):
        metrics = None
        with self.condition:
            if self.states[index] != _RING_RUNNING:
                raise AssertionError("ring worker finished a non-running job")
            self.read_s += elapsed
            if error is None:
                self.states[index] = _RING_DONE
                self.successes += 1
            else:
                self.states[index] = _RING_FAILED
                self.errors[index] = error
                self.failures += 1
            self.remaining -= 1
            metrics = self._publish_if_complete_locked()
            self.condition.notify_all()
        self._publish(metrics)

    def cancel(self, index):
        metrics = None
        with self.condition:
            if self.states[index] != _RING_QUEUED:
                return False
            self.states[index] = _RING_CANCELED
            self.canceled += 1
            self.remaining -= 1
            metrics = self._publish_if_complete_locked()
            self.condition.notify_all()
        self._publish(metrics)
        return True

    def cancel_all_queued(self):
        """Cancel every expert not yet claimed by a worker."""
        for index in range(len(self.states)):
            self.cancel(index)

    def result(self, index, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.states[index] in (_RING_QUEUED, _RING_RUNNING):
                if deadline is None:
                    self.condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                self.condition.wait(remaining)
            state = self.states[index]
            if state == _RING_DONE:
                return self.slots[index]
            if state == _RING_CANCELED:
                raise concurrent.futures.CancelledError()
            if state == _RING_FAILED:
                raise self.errors[index]
            raise AssertionError(f"unexpected ring state {state}")


class _RingHandle:
    """Future-compatible view of one index in a shared ``_RingBatch``."""

    __slots__ = ("_batch", "_index")

    def __init__(self, batch, index):
        self._batch = batch
        self._index = index

    def cancel(self):
        return self._batch.cancel(self._index)

    def result(self, timeout=None):
        return self._batch.result(self._index, timeout=timeout)

    def cancelled(self):
        with self._batch.condition:
            return self._batch.states[self._index] == _RING_CANCELED


class _PreadWorkerRing:
    """Persistent FIFO worker ring for complete-expert reads.

    Workers sleep on one condition and pull one drain ticket per participating
    worker, then claim experts from the batch's compact shared arrays.  Thus a
    16-expert layer queues at most ``workers`` records rather than 16 jobs or
    Futures. Cancellation marks unclaimed indices terminal; running reads remain
    non-cancelable, matching ``Future.cancel`` and the reader's two-phase drain.
    """

    def __init__(self, workers=PREAD_WORKERS, name="k3pread-ring"):
        if workers <= 0:
            raise ValueError("ring workers must be positive")
        self.workers = workers
        self._queue = collections.deque()
        self._condition = threading.Condition()
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"{name}-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, slots, paths):
        slots = list(slots)
        paths = list(paths)
        if len(slots) != len(paths):
            raise ValueError("slot/path count mismatch")
        if not slots:
            return []
        batch = _RingBatch(slots, paths)
        handles = [
            _RingHandle(batch, index) for index in range(len(slots))
        ]
        with self._condition:
            if self._closed:
                raise RuntimeError("expert worker ring is closed")
            self._queue.extend(
                batch for _ in range(min(self.workers, len(slots)))
            )
            self._condition.notify_all()
        # One scheduling-stat lock acquisition for the entire expert set.
        _bump(pread_ring_batches=1, pread_ring_jobs=len(slots))
        return handles

    def _worker(self):
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                batch = self._queue.popleft()
            while True:
                index = batch.claim()
                if index is None:
                    break
                started = time.perf_counter()
                error = None
                try:
                    batch.slots[index].read(batch.paths[index])
                except BaseException as exc:
                    error = exc
                batch.finish(index, error, time.perf_counter() - started)

    def close(self, cancel_pending=True):
        queued = []
        with self._condition:
            if self._closed:
                threads = list(self._threads)
            else:
                self._closed = True
                queued = list(self._queue) if cancel_pending else []
                if cancel_pending:
                    self._queue.clear()
                threads = list(self._threads)
                self._condition.notify_all()
        for batch in queued:
            batch.cancel_all_queued()
        for thread in threads:
            thread.join()


class _PreadReader:
    """Persistent worker pool + recycled slot pool. One instance per process."""

    def __init__(self, use_ring=None, workers=None):
        self._workers = workers or PREAD_WORKERS
        self._use_ring = PREAD_RING if use_ring is None else bool(use_ring)
        self._ring = (
            _PreadWorkerRing(self._workers)
            if self._use_ring else None
        )
        # The legacy path creates this eagerly. Ring mode keeps a lazy executor
        # only for legacy .npz fallbacks, so the all-.bin steady state really
        # uses exactly the fixed ring threads.
        self._ex = (
            None if self._use_ring
            else concurrent.futures.ThreadPoolExecutor(
                self._workers, thread_name_prefix="k3pread"
            )
        )
        self._free = []                       # idle _Slot objects
        self._gens = collections.deque()      # [[slot, ...]] youngest last
        self._pending = {}                    # layer -> {eid: (slot, future)}
        self._lock = threading.Lock()

    def _fallback_executor(self):
        if self._ex is not None:
            return self._ex
        with self._lock:
            if self._ex is None:
                self._ex = concurrent.futures.ThreadPoolExecutor(
                    self._workers, thread_name_prefix="k3pread-fallback"
                )
            return self._ex

    # -- slot bookkeeping ---------------------------------------------------
    def _take(self, n):
        with self._lock:
            out = [self._free.pop() for _ in range(min(n, len(self._free)))]
        out += [_Slot() for _ in range(n - len(out))]
        return out

    def _give(self, slots):
        with self._lock:
            self._free.extend(slots)
            drop = len(self._free) - PREAD_MAX_FREE
            extra = [self._free.pop() for _ in range(drop)] if drop > 0 else []
        del extra                             # unmaps the surplus prefill slots

    def _retire(self, slots):
        """Publish this call's slots; recycle the ones PREAD_DEPTH calls old."""
        old = []
        with self._lock:
            self._gens.append(slots)
            while len(self._gens) > max(1, PREAD_DEPTH):
                old.extend(self._gens.popleft())
        if old:
            self._give(old)

    # -- reads --------------------------------------------------------------
    def _job(self, slot, path):
        t0 = time.time()
        slot.read(path)
        _bump(pread_experts=1, pread_bytes=EXPERT_SPAN, pread_s=time.time() - t0)
        return slot

    def _submit(self, layer, eids):
        slots = self._take(len(eids))
        if self._ring is not None:
            handles = self._ring.submit(
                slots, [_cache_path_bin(layer, e) for e in eids]
            )
            return {
                e: (slot, handle)
                for e, slot, handle in zip(eids, slots, handles)
            }
        return {
            e: (
                slot,
                self._fallback_executor().submit(
                    self._job, slot, _cache_path_bin(layer, e)
                ),
            )
            for e, slot in zip(eids, slots)
        }

    def prefetch(self, layer, eids):
        """Speculatively start layer `layer`'s reads. At most one layer is kept
        outstanding; a superseded prefetch is drained and its slots recycled."""
        eids = [e for e in eids if _has_bin(layer, e)]
        if not eids:
            return
        with self._lock:
            busy = layer in self._pending or len(self._pending) >= 1
        if busy:
            return
        jobs = self._submit(layer, eids)
        with self._lock:
            self._pending[layer] = jobs

    def _drain(self, jobs):
        """Wait out unwanted in-flight reads and hand their slots back."""
        spare = []
        # Cancel every queued loser *before* waiting for any running loser.
        # The old single loop waited on the first active future; while it was
        # blocked, workers pulled the still-queued speculative futures behind
        # it, often completing the entire wrong prefetch before demand could be
        # submitted.  Two phases bound waste to the jobs already inside pread().
        running = []
        for slot, fut in jobs.values():
            if not fut.cancel():
                running.append(fut)
            spare.append(slot)
        for fut in running:
            try:
                fut.result()
            except Exception:
                pass
        _bump(prefetch_wasted=len(spare))
        self._give(spare)

    def drop_prefetch(self):
        with self._lock:
            pend, self._pending = self._pending, {}
        for jobs in pend.values():
            self._drain(jobs)

    def read_layer(self, layer, eids):
        """{eid: {"w1": (packed, scale), ...}} for the whole selected set, plus
        the list of eids with nothing on disk (those go back to the HTTP path).

        .bin spans go through the pread pool; legacy .npz entries keep the old
        np.load path but are submitted to the same pool first, so they overlap
        with the raw reads instead of running serially after them."""
        want, other = [], []
        for e in eids:
            (want if _has_bin(layer, e) else other).append(e)
        npz = {
            e: self._fallback_executor().submit(_cache_load, layer, e)
            for e in other
        }
        missing = []
        with self._lock:
            pend = self._pending.pop(layer, None)
        jobs = {}
        if pend:
            hits = [e for e in want if e in pend]
            _bump(prefetch_hits=len(hits))
            for e in hits:
                jobs[e] = pend.pop(e)
            if pend:
                self._drain(pend)             # speculation misses -> free slots
            want = [e for e in want if e not in jobs]
        if want:
            jobs.update(self._submit(layer, want))
        raw, slots = {}, []
        local_failures = []
        for e, (slot, fut) in jobs.items():
            try:
                fut.result()
                raw[e] = slot.raw
            except Exception as first_error:
                path = _cache_path_bin(layer, e)
                if _raw_file_valid(layer, e):
                    # A transient local pread error must not masquerade as a
                    # network cache miss. Retry once outside the worker pool;
                    # a complete retry overwrites any partial first read.
                    started = time.time()
                    try:
                        slot.read(path)
                    except Exception as retry_error:
                        _bump(pread_local_failures=1)
                        local_failures.append(
                            (e, path, first_error, retry_error)
                        )
                    else:
                        _bump(
                            pread_experts=1,
                            pread_bytes=EXPERT_SPAN,
                            pread_s=time.time() - started,
                            pread_retries=1,
                        )
                        raw[e] = slot.raw
                else:
                    # The file genuinely disappeared or changed after startup;
                    # only this case is eligible for legacy/HTTP recovery.
                    _drop_bin(layer, e)
                    missing.append(e)
            slots.append(slot)
        self._retire(slots)
        _bump(expert_disk=len(raw))
        # A stale presence record degrades to the ordinary npz/HTTP path; the
        # miss list below owns recovery rather than taking down the run.
        for e, fut in npz.items():
            hit = fut.result()
            if hit is None:
                missing.append(e)
            else:
                raw[e] = hit
                _bump(npz_experts=1)
        if local_failures:
            e, path, first_error, retry_error = local_failures[0]
            raise LocalExpertReadError(
                f"local expert L{layer}-E{e} failed two reads even though "
                f"{path} remains a regular {EXPERT_SPAN}-byte file; refusing "
                "to disguise the local I/O failure as an HTTP cache miss "
                f"(first: {type(first_error).__name__}: {first_error}; "
                f"retry: {type(retry_error).__name__}: {retry_error})"
            ) from retry_error
        return raw, missing

    def read_layer_groups(self, layer, eids, group_size):
        """Yield ``(eids, raw)`` groups while later real experts stay in flight.

        ``begin_layer_groups`` proves every requested expert has a raw .bin
        before this generator is created.  All jobs are submitted up front in
        caller order, so a weight-priority caller gets its first useful group
        as soon as those futures land while the pool continues later reads.

        Slots are retired only after the generator finishes or is closed.  This
        keeps every yielded raw view valid through all Metal dispatches and
        makes exception/fallback cleanup capability-neutral.
        """
        eids = list(eids)
        if group_size <= 0 or len(set(eids)) != len(eids):
            raise ValueError("group_size must be positive and eids unique")
        with self._lock:
            pend = self._pending.pop(layer, None)
        jobs = {}
        if pend:
            hits = [e for e in eids if e in pend]
            _bump(prefetch_hits=len(hits))
            for e in hits:
                jobs[e] = pend.pop(e)
            if pend:
                self._drain(pend)
        want = [e for e in eids if e not in jobs]
        if want:
            jobs.update(self._submit(layer, want))

        slots = [jobs[e][0] for e in eids]
        settled = set()
        successful = set()
        _bump(grouped_calls=1)
        try:
            for start in range(0, len(eids), group_size):
                group_eids = eids[start:start + group_size]
                raw = {}
                for e in group_eids:
                    slot, fut = jobs[e]
                    try:
                        fut.result()
                    except Exception as first_error:
                        path = _cache_path_bin(layer, e)
                        if _raw_file_valid(layer, e):
                            started = time.time()
                            try:
                                slot.read(path)
                            except Exception as retry_error:
                                settled.add(e)
                                _bump(pread_local_failures=1)
                                raise LocalExpertReadError(
                                    f"local expert L{layer}-E{e} failed two "
                                    f"grouped reads even though {path} remains "
                                    f"a regular {EXPERT_SPAN}-byte file; "
                                    "refusing HTTP fallback "
                                    f"(first: {type(first_error).__name__}: "
                                    f"{first_error}; retry: "
                                    f"{type(retry_error).__name__}: "
                                    f"{retry_error})"
                                ) from retry_error
                            _bump(
                                pread_experts=1,
                                pread_bytes=EXPERT_SPAN,
                                pread_s=time.time() - started,
                                pread_retries=1,
                            )
                        else:
                            _drop_bin(layer, e)
                            settled.add(e)
                            _bump(grouped_fallbacks=1)
                            raise GroupReadError(
                                f"grouped read failed for L{layer}-E{e}"
                            ) from first_error
                    settled.add(e)
                    successful.add(e)
                    raw[e] = slot.raw
                _bump(grouped_yields=1)
                yield group_eids, raw
        finally:
            # A Metal failure, explicit generator close, or later read failure
            # must not recycle a slot while pread is still writing into it.
            running = []
            for e in eids:
                if e in settled:
                    continue
                _slot, fut = jobs[e]
                if not fut.cancel():
                    running.append((e, fut))
                settled.add(e)
            # Cancel every queued group before blocking on running reads. This
            # is the same lifetime rule as _drain(): otherwise workers can turn
            # all later groups into I/O while cleanup waits on the first one.
            for e, fut in running:
                try:
                    fut.result()
                    successful.add(e)
                except Exception:
                    # Keep a still-valid local file eligible for a fresh local
                    # retry on the next request. Only a file that actually
                    # vanished or changed is allowed to become a cache miss.
                    if not _raw_file_valid(layer, e):
                        _drop_bin(layer, e)
            self._retire(slots)
            _bump(expert_disk=len(successful))

    def shutdown(self):
        """Drain owned work and stop worker threads (primarily tests/tools)."""
        self.drop_prefetch()
        if self._ring is not None:
            self._ring.close(cancel_pending=True)
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
        self._ring = None
        self._ex = None
        with self._lock:
            self._pending.clear()
            self._free.clear()
            self._gens.clear()


_reader = None
_reader_lock = threading.Lock()


def reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                _reader = _PreadReader()
    return _reader


def prefetch_layer(layer, eids):
    """Start a layer's expert reads early (K3_EXPERT_READ=pread only)."""
    if EXPERT_READ == "pread":
        reader().prefetch(layer, list(eids))


def drop_prefetch():
    if EXPERT_READ == "pread" and _reader is not None:
        _reader.drop_prefetch()


def begin_layer_groups(layer, eids, group_size):
    """Return a local grouped-read iterator, or ``None`` for safe fallback.

    Grouping is intentionally local-pread-only.  A legacy .npz, absent expert,
    HTTP fetch, non-positive group size, or duplicate ID takes the unchanged
    ``fetch_experts`` path before any grouped work is submitted.
    """
    eids = list(eids)
    if (EXPERT_READ != "pread" or group_size <= 0
            or len(set(eids)) != len(eids)
            or any(not _has_bin(layer, e) for e in eids)):
        return None
    return reader().read_layer_groups(layer, eids, group_size)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _slice_expert(buf, base=0):
    """Slice one expert's 6 tensors out of a fetched buffer."""
    out = {}
    for name, off, nb, shape in LAYOUT:
        w, kind = name.split("_")
        a = np.frombuffer(buf, dtype=np.uint8, count=nb,
                          offset=base + off).reshape(shape)
        out.setdefault(w, {})[kind] = a
    return {w: (d["p"], d["s"]) for w, d in out.items()}


def _cache_path(layer, eid):
    return os.path.join(ECACHE, f"L{layer}-E{eid}.npz")


def _cache_path_bin(layer, eid):
    # raw format: the expert's 17,547,264-byte shard span verbatim (w1_p w1_s w2_p
    # w2_s w3_p w3_s) — zero-parse, mmap-able; npz kept as read fallback
    return os.path.join(ECACHE, f"L{layer}-E{eid}.bin")


def _has_bin(layer, eid):
    # Startup validated every entry's size. Individual set operations are
    # atomic under CPython; writes still use a lock so alternate runtimes have
    # an uncomplicated synchronization contract.
    return f"L{layer}-E{eid}.bin" in _raw_presence


def _raw_file_valid(layer, eid):
    """Revalidate a raw expert after a local read error without following links."""
    try:
        info = os.lstat(_cache_path_bin(layer, eid))
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == EXPERT_SPAN


def _drop_bin(layer, eid):
    """Forget a stale cache entry after an actual open/read failure."""
    with _raw_presence_lock:
        _raw_presence.discard(f"L{layer}-E{eid}.bin")


def _cache_store_raw(layer, eid, span):
    path = _cache_path_bin(layer, eid)
    if ASYNC_CACHE_WRITE:
        writer = _get_cache_writer()
        if writer.submit(
            path,
            span,
            on_published=_cache_write_published,
            on_error=_cache_write_error,
        ):
            _bump(cache_write_enqueued=1)
            return
        # A caller racing explicit shutdown still preserves the cache object.
        _bump(cache_write_sync_fallbacks=1)
    atomic_publish(path, span)
    _bump(cache_write_sync=1)
    _cache_write_published(path, len(span))


def _cache_load(layer, eid):
    path = _cache_path_bin(layer, eid)
    if os.path.exists(path) and os.path.getsize(path) == EXPERT_SPAN:
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        _bump(expert_disk=1)
        return _slice_expert(buf)
    path = _cache_path(layer, eid)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    _bump(expert_disk=1)
    return {w: (z[w + "_p"], z[w + "_s"]) for w in ("w1", "w2", "w3")}


def fetch_expert_raw(layer, eid):
    """Drop-in for k3loader.fetch_expert_raw: dict w -> (packed u8, scale u8).
    One coalesced Range request over a persistent connection on cache miss."""
    hit = _cache_load(layer, eid)
    if hit is not None:
        return hit
    shard, start, _ = expert_span(layer, eid)
    buf = _range_get(shard, start, EXPERT_SPAN)
    ws = _slice_expert(buf)
    _cache_store_raw(layer, eid, buf)
    _bump(expert_http=1)
    return ws


def _fetch_group(shard, group):
    """group: list of (layer, eid, start) sorted, file-contiguous. One request."""
    base = group[0][2]
    size = group[-1][2] + EXPERT_SPAN - base
    buf = _range_get(shard, base, size)
    if len(group) > 1:
        _bump(coalesced_spans=1)
    out = {}
    for layer, eid, start in group:
        ws = _slice_expert(buf, base=start - base)
        _cache_store_raw(layer, eid, buf[start - base:start - base + EXPERT_SPAN])
        _bump(expert_http=1)
        out[eid] = ws
    return out


def _finish(raw, dequant):
    if not dequant:
        return {e: {w: (np.ascontiguousarray(p), np.ascontiguousarray(s))
                    for w, (p, s) in ws.items()} for e, ws in raw.items()}
    import torch
    from mxfp4 import dequant_mxfp4
    return {e: {w: torch.from_numpy(dequant_mxfp4(p, s))
                for w, (p, s) in ws.items()} for e, ws in raw.items()}


def fetch_experts(layer, eids, workers=None, dequant=True, coalesce=True):
    """Parallel fetch of a routed set; signature-compatible with k3loader.
    Coalesces file-adjacent misses into single multi-expert Range requests.

    With K3_EXPERT_READ=pread the .bin-cached experts of the whole layer are
    read up front by the worker pool; the returned arrays are views into
    recycled buffers and stay valid for K3_PREAD_DEPTH (default 2) further
    fetch_experts() calls — long enough for the caller's MoE kernel to run."""
    workers = workers or MAX_CONNS
    raw, misses = {}, []
    if EXPERT_READ == "pread":
        raw, misses = reader().read_layer(layer, list(eids))
        if not misses:                       # the all-local steady state
            return _finish(raw, dequant)
    else:
        for e in eids:
            hit = _cache_load(layer, e)
            if hit is not None:
                raw[e] = hit
            else:
                misses.append(e)
    if misses:
        spans = sorted((expert_span(layer, e)[1], e) for e in misses)
        shard = expert_span(layer, misses[0])[0]
        groups, cur = [], [(layer, spans[0][1], spans[0][0])]
        for s, e in spans[1:]:
            if coalesce and s == cur[-1][2] + EXPERT_SPAN and len(cur) < MAX_COALESCE:
                cur.append((layer, e, s))
            else:
                groups.append(cur)
                cur = [(layer, e, s)]
        groups.append(cur)
        with concurrent.futures.ThreadPoolExecutor(min(workers, len(groups))) as ex:
            for fut in [ex.submit(_fetch_group, shard, g) for g in groups]:
                raw.update(fut.result())
    return _finish(raw, dequant)


def close():
    """Close pooled connections (safe to call between batches)."""
    global _httpx_client
    # Surface background persistence errors at an ordinary lifecycle boundary
    # without permanently disabling the writer; atexit performs final shutdown.
    flush_cache_writes(raise_errors=True)
    with _pool._lock:
        for _, c in _pool._idle:
            try:
                c.close()
            except Exception:
                pass
        _pool._idle.clear()
    if _httpx_client is not None:
        _httpx_client.close()
        _httpx_client = None
