"""Weight loading for lazy-K3: local resident spine + on-demand HTTP expert fetch with disk cache."""
import json, os, threading, time, urllib.request, concurrent.futures
import numpy as np
import torch
from mxfp4 import dequant_mxfp4

try:
    import model_source
except ImportError:  # imported as tools.k3loader
    from . import model_source

ROOT = (
    os.environ.get("DELTAFIN_ROOT")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
INV = json.load(open(os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")))
BASE = model_source.base_url()
RES = os.path.join(ROOT, "k3-resident/tensors")
ECACHE = os.path.join(ROOT, "k3-experts")
os.makedirs(ECACHE, exist_ok=True)

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "U8": torch.uint8}

stats = {"expert_http": 0, "expert_disk": 0, "http_bytes": 0, "http_s": 0.0}
_stats_lock = threading.Lock()
_runtime_stats = stats
_cache_totals_lock = threading.Lock()


def _initial_cache_totals():
    """Scan the expert directory once, never once per generated token."""
    files = {}
    experts = set()
    try:
        entries = os.scandir(ECACHE)
    except FileNotFoundError:
        return files, experts, 0
    with entries:
        for ent in entries:
            if not ent.name.endswith((".bin", ".npz")):
                continue
            try:
                size = ent.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            files[ent.name] = size
            experts.add(ent.name.rsplit(".", 1)[0])
    return files, experts, sum(files.values())


_cache_file_sizes, _cache_experts, _cache_bytes = _initial_cache_totals()


def register_cache_file(path):
    """Increment the cached count/bytes after an atomic cache-file publish.

    Both the legacy .npz writer below and fetch_v2's raw .bin writer call this.
    Keeping per-file sizes makes a replacement idempotent and thread-safe.
    """
    global _cache_bytes
    name = os.path.basename(path)
    if not name.endswith((".bin", ".npz")):
        return
    try:
        size = os.stat(path).st_size
    except OSError:
        return
    with _cache_totals_lock:
        old = _cache_file_sizes.get(name, 0)
        _cache_file_sizes[name] = size
        _cache_experts.add(name.rsplit(".", 1)[0])
        _cache_bytes += size - old


def set_runtime_stats(source):
    """Point cache_report at a drop-in fetcher's live counters."""
    global _runtime_stats
    _runtime_stats = source


def cache_totals():
    with _cache_totals_lock:
        return len(_cache_experts), _cache_bytes


def _range_fetch(name, retries=6):
    t = INV[name]
    start = 8 + t["hlen"] + t["offsets"][0]
    size = t["offsets"][1] - t["offsets"][0]
    req = urllib.request.Request(
        BASE + t["shard"], headers={"Range": f"bytes={start}-{start+size-1}"})
    for attempt in range(retries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                buf = r.read()
            if len(buf) != size:
                raise IOError(f"short read {len(buf)}/{size}")
            with _stats_lock:
                stats["http_bytes"] += size
                stats["http_s"] += time.time() - t0
            return buf
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def load_resident(name):
    """Load a spine tensor from the local download (falls back to HTTP if not yet there)."""
    t = INV[name]
    path = os.path.join(RES, name)
    size = t["offsets"][1] - t["offsets"][0]
    if os.path.exists(path) and os.path.getsize(path) == size:
        buf = open(path, "rb").read()
    else:
        buf = _range_fetch(name)
    ten = torch.frombuffer(bytearray(buf), dtype=_DT[t["dtype"]]).reshape(t["shape"])
    return ten


def _expert_prefix(layer, eid):
    return f"language_model.model.layers.{layer}.block_sparse_moe.experts.{eid}"


def fetch_expert_raw(layer, eid):
    """Return dict wname -> (packed U8 np, scale U8 np). Disk-cached as raw packed bytes (~17.5MB)."""
    path = os.path.join(ECACHE, f"L{layer}-E{eid}.npz")
    if os.path.exists(path):
        with _stats_lock:
            stats["expert_disk"] += 1
        z = np.load(path)
        return {w: (z[w + "_p"], z[w + "_s"]) for w in ("w1", "w2", "w3")}
    pfx = _expert_prefix(layer, eid)
    out = {}
    for w in ("w1", "w2", "w3"):
        tp = INV[pfx + f".{w}.weight_packed"]
        ts = INV[pfx + f".{w}.weight_scale"]
        p = np.frombuffer(_range_fetch(pfx + f".{w}.weight_packed"), dtype=np.uint8).reshape(tp["shape"])
        s = np.frombuffer(_range_fetch(pfx + f".{w}.weight_scale"), dtype=np.uint8).reshape(ts["shape"])
        out[w] = (p, s)
    tmp = path + f".tmp{os.getpid()}"
    np.savez(tmp, **{w + suf: arr for w, (p, s) in out.items()
                     for suf, arr in (("_p", p), ("_s", s))})
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    register_cache_file(path)
    with _stats_lock:
        stats["expert_http"] += 1
    return out


def fetch_experts(layer, eids, workers=12, dequant=True):
    """Parallel fetch of a routed set. dequant=True -> {eid: {w: fp32 torch tensor}};
    dequant=False -> {eid: {w: (packed u8, scale u8) contiguous np arrays}}."""
    raw = {}
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_expert_raw, layer, e): e for e in eids}
        for f in concurrent.futures.as_completed(futs):
            raw[futs[f]] = f.result()
    if not dequant:
        return {e: {w: (np.ascontiguousarray(p), np.ascontiguousarray(s))
                    for w, (p, s) in ws.items()} for e, ws in raw.items()}
    out = {}
    for e, ws in raw.items():
        out[e] = {w: torch.from_numpy(dequant_mxfp4(p, s)) for w, (p, s) in ws.items()}
    return out


def cache_report():
    n, total = cache_totals()
    gb = total / 1e9
    live = _runtime_stats
    bw = live["http_bytes"] / 1e6 / max(live["http_s"], 1e-9)
    return (f"expert cache: {n} experts / {gb:.1f} GB on disk | this run: "
            f"{live['expert_disk']} disk hits, {live['expert_http']} http fetches "
            f"({bw:.0f} MB/s eff)")
