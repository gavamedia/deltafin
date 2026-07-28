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
import collections
import fcntl
import http.client
import json
import mmap
import os
import re
import ssl
import sys
import threading
import time
import urllib.parse
import concurrent.futures

import numpy as np

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV_PATH = os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")
BASE_HOST = os.environ.get("K3_HF_HOST", "huggingface.co")
BASE_PATH = os.environ.get("K3_HF_PATH", "/moonshotai/Kimi-K3/resolve/main/")
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
PREAD_WORKERS = int(os.environ.get("K3_PREAD_WORKERS", "6"))
# Cache-bypass for expert reads. On Darwin this is essential: a 64 GB Mac cannot let
# 25.8 GB/token of expert traffic evict the page cache the resident spine depends on.
# On Linux the calculus INVERTS on a large-RAM host — RAM beyond the spine is the best
# expert cache available, so bypassing the page cache there throws away free hit-rate.
# Hence: default ON for Darwin, OFF elsewhere; K3_PREAD_NOCACHE always overrides.
PREAD_NOCACHE = os.environ.get(
    "K3_PREAD_NOCACHE", "1" if sys.platform == "darwin" else "0") == "1"
# free-slot high-water mark; prefill unions (up to 5 x 16 experts/layer) grow the
# pool transiently and are trimmed back to this after the layer is done.
PREAD_MAX_FREE = int(os.environ.get("K3_PREAD_MAX_FREE", "40"))
# how many read_layer() results stay valid at once (buffer recycling depth).
PREAD_DEPTH = int(os.environ.get("K3_PREAD_DEPTH", "2"))
F_NOCACHE = 48          # <sys/fcntl.h>, Darwin: bypass the unified buffer cache


def _drop_cache(fd, nbytes=0):
    """Ask the OS not to retain this fd's pages. Darwin: F_NOCACHE before the read.
    Linux: posix_fadvise(DONTNEED) after it (there is no pre-read equivalent, and
    fcntl cmd 48 is UNDEFINED on Linux -> OSError(EINVAL) in every pread worker,
    which is what this guard exists to prevent). Any other platform: no-op.
    Advisory everywhere; never allowed to break a read that already succeeded."""
    try:
        if sys.platform == "darwin":
            fcntl.fcntl(fd, F_NOCACHE, 1)
        elif hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass    # safe swallow: cache hinting is an optimisation, not correctness

stats = {
    "expert_http": 0, "expert_disk": 0, "http_bytes": 0, "http_s": 0.0,
    "requests": 0, "new_conns": 0, "conn_s": 0.0, "resolves": 0,
    "resolve_s": 0.0, "retries": 0, "coalesced_spans": 0,
    "pread_experts": 0, "pread_bytes": 0, "pread_s": 0.0, "pread_slots": 0,
    "prefetch_hits": 0, "prefetch_wasted": 0, "npz_experts": 0,
}
_stats_lock = threading.Lock()


def _bump(**kw):
    with _stats_lock:
        for k, v in kw.items():
            stats[k] += v


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
        fd = os.open(path, os.O_RDONLY)
        try:
            # Darwin's F_NOCACHE must be set BEFORE the read (it changes how the read
            # is serviced); Linux's fadvise(DONTNEED) only evicts pages that already
            # exist, so it must run AFTER. Same intent, opposite placement.
            if PREAD_NOCACHE and sys.platform == "darwin":
                _drop_cache(fd)
            off = 0
            while off < EXPERT_SPAN:
                n = os.preadv(fd, [self.mv[off:]], off)
                if n <= 0:
                    raise IOError(f"short read {off}/{EXPERT_SPAN} from {path}")
                off += n
            if PREAD_NOCACHE and sys.platform != "darwin":
                _drop_cache(fd, EXPERT_SPAN)
        finally:
            os.close(fd)


class _PreadReader:
    """Persistent worker pool + recycled slot pool. One instance per process."""

    def __init__(self):
        self._ex = concurrent.futures.ThreadPoolExecutor(
            PREAD_WORKERS, thread_name_prefix="k3pread")
        self._free = []                       # idle _Slot objects
        self._gens = collections.deque()      # [[slot, ...]] youngest last
        self._pending = {}                    # layer -> {eid: (slot, future)}
        self._lock = threading.Lock()

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
        return {e: (s, self._ex.submit(self._job, s, _cache_path_bin(layer, e)))
                for e, s in zip(eids, slots)}

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
        for slot, fut in jobs.values():
            if not fut.cancel():
                try:
                    fut.result()
                except Exception:
                    pass
            spare.append(slot)
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
        npz = {e: self._ex.submit(_cache_load, layer, e) for e in other}
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
        err = None
        for e, (slot, fut) in jobs.items():
            try:
                fut.result()
                raw[e] = slot.raw
            except Exception as ex:           # unreadable .bin -> other path
                err = err or ex
                missing.append(e)
            slots.append(slot)
        self._retire(slots)
        _bump(expert_disk=len(raw))
        if err is not None and not raw:
            raise err
        for e, fut in npz.items():
            hit = fut.result()
            if hit is None:
                missing.append(e)
            else:
                raw[e] = hit
                _bump(npz_experts=1)
        return raw, missing


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
    try:
        return os.stat(_cache_path_bin(layer, eid)).st_size == EXPERT_SPAN
    except OSError:
        return False


def _cache_store_raw(layer, eid, span):
    path = _cache_path_bin(layer, eid)
    tmp = path + f".tmp{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "wb") as f:
        f.write(span)
    os.replace(tmp, path)


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
