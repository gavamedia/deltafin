"""Mixed-precision resident-spine loader (K3_SPINE=mixed).

Read side for tools/convert_spine_int4.py --policy. A layer may mix codecs:
each tensor is i4 / i6 (from k3-resident-mixed/tensors) or i8 (falling back to
the existing k3-resident-int8/tensors), and the small bf16 tail (norms, convs,
A_log, dt_bias, res_projs) is read from k3-resident as before.

Same three-part shape as tools/spine_fast.py, which this mirrors deliberately:
  1. one packed readinto for the whole layer (payloads, scales, bf16 tail),
  2. three host->device copies per layer instead of ~56,
  3. a fused Metal dequant kernel per codec that writes straight into the
     template parameter buffer.

The kernels compute float(level) * float(fp16 scale) in fp32 — the same
arithmetic as the numpy oracle in tools/spine_codec.py, hence bit-identical;
`python tools/mixed_spine.py --check` asserts that on device.

Nothing here is active unless K3_SPINE=mixed.
"""
import json
import os
import threading
import time as _time
import concurrent.futures as _cf

import torch

import spine_codec
import spine_io
import runtime_platform

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIXED_DIR = os.environ.get("K3_MIXED_DIR") or os.path.join(ROOT, "k3-resident-mixed/tensors")
_META_PATH = os.path.join(os.path.dirname(MIXED_DIR.rstrip("/")), "meta.json")
READ_THREADS = runtime_platform.configured_cpu_workers(
    "K3_SPINE_READ_THREADS", 4
)
_READ_THREADS_EXPLICIT = "K3_SPINE_READ_THREADS" in os.environ
DEQ = os.environ.get("K3_SPINE_DEQ", "metal")        # metal | torch
ALIGN = 256

_meta = None


def meta():
    global _meta
    if _meta is None:
        try:
            _meta = json.load(open(_META_PATH))
        except OSError:
            _meta = {"format": "mixed-spine", "group": 64, "missing_meta": True}
    return _meta


def group_size():
    return int(meta().get("group", 64))


def available():
    try:
        return any(f.endswith((".i4", ".i6", ".i8")) for f in os.listdir(MIXED_DIR))
    except OSError:
        return False


def codec_of(name, int8_dir):
    """(codec, directory) for one tensor, or (None, None) if it is not quantized."""
    for c in ("i4", "i6", "i8"):
        if os.path.exists(os.path.join(MIXED_DIR, name + spine_codec.EXT[c][0])):
            return c, MIXED_DIR
    if os.path.exists(os.path.join(int8_dir, name + ".i8")):
        return "i8", int8_dir
    return None, None


def describe(int8_dir=None):
    m = meta()
    n = 0
    try:
        n = sum(1 for f in os.listdir(MIXED_DIR) if f.endswith((".i4", ".i6")))
    except OSError:
        pass
    return (f"policy={m.get('policy')} g={m.get('group')} "
            f"layers={m.get('layers')} sub8_tensors={n} "
            f"spine {m.get('spine_gb_int8')}->{m.get('spine_gb_mixed')} GB "
            f"deq={DEQ} read_threads={READ_THREADS} "
            f"io: {spine_io.describe()}"
            f"{'' if metal_available() else ' (metal off: %s)' % _lib_err}")


# --------------------------------------------------------- metal kernels ----
_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// int8, per-row scale — identical arithmetic to spine_fast.deq_f32.
kernel void mx_i8(device float*       out  [[buffer(0)]],
                  device const char*  q    [[buffer(1)]],
                  device const half*  sc   [[buffer(2)]],
                  constant uint&      cols [[buffer(3)]],
                  uint idx [[thread_position_in_grid]])
{
    out[idx] = float(q[idx]) * float(sc[idx / cols]);
}

// int4, group scale. One thread per packed BYTE = two output columns.
// low nibble = even column, 2's complement in [-7, 7].
kernel void mx_i4(device float*        out   [[buffer(0)]],
                  device const uchar*  q     [[buffer(1)]],
                  device const half*   sc    [[buffer(2)]],
                  constant uint&       cols  [[buffer(3)]],
                  constant uint&       cw    [[buffer(4)]],
                  constant uint&       g     [[buffer(5)]],
                  uint idx [[thread_position_in_grid]])
{
    uint half_cw = cw >> 1;
    uint row = idx / half_cw;
    uint c0  = (idx - row * half_cw) * 2;
    if (c0 >= cols) return;                       // zero-padded tail group
    uint ng  = cw / g;
    float s  = float(sc[row * ng + c0 / g]);
    uchar b  = q[idx];
    int lo = int(b & 0x0F); if (lo > 7) lo -= 16;
    int hi = int(b >> 4);   if (hi > 7) hi -= 16;
    device float* o = out + row * cols + c0;
    o[0] = float(lo) * s;
    if (c0 + 1 < cols) o[1] = float(hi) * s;
}

// int6, group scale. One thread per 3 packed BYTES = four output columns.
kernel void mx_i6(device float*        out   [[buffer(0)]],
                  device const uchar*  q     [[buffer(1)]],
                  device const half*   sc    [[buffer(2)]],
                  constant uint&       cols  [[buffer(3)]],
                  constant uint&       cw    [[buffer(4)]],
                  constant uint&       g     [[buffer(5)]],
                  uint idx [[thread_position_in_grid]])
{
    uint quads = cw >> 2;
    uint row = idx / quads;
    uint k   = idx - row * quads;
    uint c0  = k * 4;
    if (c0 >= cols) return;
    uint ng  = cw / g;
    float s  = float(sc[row * ng + c0 / g]);
    device const uchar* b = q + row * (quads * 3) + k * 3;
    int v[4];
    v[0] =  b[0] & 0x3F;
    v[1] = (b[0] >> 6) | ((b[1] & 0x0F) << 2);
    v[2] = (b[1] >> 4) | ((b[2] & 0x03) << 4);
    v[3] =  b[2] >> 2;
    device float* o = out + row * cols + c0;
    for (uint j = 0; j < 4; ++j) {
        if (c0 + j >= cols) break;
        int x = v[j]; if (x > 31) x -= 64;
        o[j] = float(x) * s;
    }
}
"""

_lib = None
_lib_err = None
_lib_lock = threading.Lock()


def metal_available():
    global _lib, _lib_err
    if _lib is not None:
        return True
    if _lib_err is not None:
        return False
    with _lib_lock:
        if _lib is None and _lib_err is None:
            try:
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS not available")
                _lib = torch.mps.compile_shader(_MSL)
            except Exception as e:                       # pragma: no cover
                _lib_err = f"{type(e).__name__}: {e}"
    return _lib is not None


# ------------------------------------------------------- torch fallback -----
def _dequant_torch(codec, q, sc, rows, cols, g, dtype=torch.float32):
    """Device-side dequant with plain torch ops. Same arithmetic as the kernels."""
    if codec == "i8":
        return q.view(torch.int8).reshape(rows, cols).to(torch.float32) \
            * sc.reshape(rows, 1).to(torch.float32)
    cw = spine_codec.padded_cols(cols, g)
    ng = cw // g
    s = sc.reshape(rows, ng).to(torch.float32)
    if codec == "i4":
        b = q.reshape(rows, cw // 2)
        lo = (b & 0x0F).to(torch.int16)
        hi = (b >> 4).to(torch.int16)
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        v = torch.stack((lo, hi), dim=2).reshape(rows, cw)
    else:
        b = q.reshape(rows, cw // 4, 3).to(torch.int16)
        v0 = b[:, :, 0] & 0x3F
        v1 = (b[:, :, 0] >> 6) | ((b[:, :, 1] & 0x0F) << 2)
        v2 = (b[:, :, 1] >> 4) | ((b[:, :, 2] & 0x03) << 4)
        v3 = b[:, :, 2] >> 2
        v = torch.stack((v0, v1, v2, v3), dim=2).reshape(rows, cw)
        v = torch.where(v > 31, v - 64, v)
    out = v.to(torch.float32).reshape(rows, ng, g) * s.unsqueeze(-1)
    return out.reshape(rows, cw)[:, :cols]


def dequant_into(dst, codec, q, sc, g):
    """dst[fp32, (rows, cols), device] <- dequant(q, sc). True if a kernel ran."""
    rows, cols = dst.shape
    if DEQ != "metal" or dst.device.type != "mps" or not metal_available():
        return False
    if codec == "i8":
        _lib.mx_i8(dst, q.view(torch.int8), sc, cols, threads=rows * cols)
        return True
    cw = spine_codec.padded_cols(cols, g)
    if codec == "i4":
        _lib.mx_i4(dst, q, sc, cols, cw, g, threads=rows * (cw // 2))
    else:
        _lib.mx_i6(dst, q, sc, cols, cw, g, threads=rows * (cw // 4))
    return True


# ----------------------------------------------------------- buffer pool ----
_pool = []
_pool_lock = threading.Lock()
_POOL_MAX = 8
KEEP = set()


def _acquire(nbytes):
    with _pool_lock:
        best = -1
        for i, b in enumerate(_pool):
            if len(b) >= nbytes and (best < 0 or len(b) < len(_pool[best])):
                best = i
        if best >= 0:
            return _pool.pop(best)
    return bytearray(max(nbytes, 1))


def _release(buf):
    if buf is None or id(buf) in KEEP:
        return
    with _pool_lock:
        if len(_pool) < _POOL_MAX:
            _pool.append(buf)


def pin(*bufs):
    for b in bufs:
        if b is not None:
            KEEP.add(id(b))


_stage = {}


def _stage_buf(slot, n, dev, dtype):
    """Persistent device staging buffer, one per SLOT. Keying these by dtype
    alone would hand the payload buffer and the bf16-tail buffer the same
    storage (both uint8) — the tail copy then overwrites the head of the
    payload, which is a silent wrong-weights bug, not a slow path. Caught by
    tools/test_mixed_spine.py's layer test (q_proj came back 0.245 off int8
    while every other tensor was exact)."""
    key = (slot, dev.type, dtype)
    b = _stage.get(key)
    if b is None or b.numel() < n:
        b = torch.empty(n, dtype=dtype, device=dev)
        _stage[key] = b
    return b[:n]


def _cpu_direct_views(dev):
    """CPU consumes packed read buffers synchronously; all else stages."""
    return getattr(dev, "type", None) == "cpu"


PHASE = {"read_s": 0.0, "read_bytes": 0, "h2d_s": 0.0, "h2d_bytes": 0,
         "deq_s": 0.0, "other_s": 0.0, "n_layers": 0,
         "bytes_by_codec": {"i4": 0, "i6": 0, "i8": 0}}

_readers = None
_readers_lock = threading.Lock()


def configure_read_threads(threads):
    """Set the reader width before its lazy worker pool is constructed.

    This deliberately matches ``spine_fast.configure_read_threads`` so a
    runtime selector can configure either packed reader through the same
    contract. An explicit call remains authoritative over the lazy
    ``spine_fast`` synchronization below.
    """
    global READ_THREADS, _READ_THREADS_EXPLICIT
    value = int(threads)
    if value < 1:
        raise ValueError("reader threads must be positive")
    if _readers is not None:
        raise RuntimeError("cannot change reader width after pool construction")
    READ_THREADS = value
    _READ_THREADS_EXPLICIT = True


def _sync_runtime_read_threads():
    """Adopt ``spine_fast``'s capability-selected width before first use.

    ``kimi_run`` imports the mixed loader before it imports and tunes
    ``spine_fast``. Looking up the sibling lazily avoids an import cycle and
    lets mixed mode inherit the same automatic width without requiring a
    second runtime-specific branch. ``K3_SPINE_READ_THREADS`` and direct calls
    to :func:`configure_read_threads` remain authoritative.
    """
    global READ_THREADS
    if _READ_THREADS_EXPLICIT or _readers is not None:
        return
    try:
        import spine_fast
    except ImportError:
        return
    value = int(getattr(spine_fast, "READ_THREADS", READ_THREADS))
    if value > 0:
        READ_THREADS = value


def _pool_exec():
    global _readers
    _sync_runtime_read_threads()
    if _readers is None:
        with _readers_lock:
            if _readers is None:
                _readers = _cf.ThreadPoolExecutor(READ_THREADS,
                                                  thread_name_prefix="k3mixed")
    return _readers


# ------------------------------------------------------- layout planning ----
_layouts = {}
_layout_lock = threading.Lock()


def _align(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


def plan_layout(module, prefix, int8_dir, res_dir, inv):
    """items = [(name, full, codec, path_base, (rows,cols), qoff, qbytes,
                 soff_elems, sbytes)] plus the bf16 tail plan."""
    got = _layouts.get(prefix)
    if got is not None:
        return got
    g = group_size()
    items, other = [], []
    qoff, soff = 0, 0                        # soff in fp16 elements
    for name, _ in module.named_parameters():
        if ".experts." in name:
            continue
        full = prefix + name
        codec, d = codec_of(full, int8_dir)
        if codec is None:
            bp = os.path.join(res_dir, full)
            other.append((name, full, bp if os.path.exists(bp) else None))
            continue
        rows, cols = inv[full]["shape"]
        nb, sb = spine_codec.blob_sizes(codec, rows, cols, g)
        items.append((name, full, codec, os.path.join(d, full), (rows, cols),
                      qoff, nb, soff, sb))
        qoff = _align(qoff + nb)
        soff = _align(soff * 2 + sb) // 2
    ooff, oplan = 0, []
    for name, full, bp in other:
        if bp is None:
            continue
        m = inv[full]
        nb = os.path.getsize(bp)
        oplan.append((name, full, m["dtype"], tuple(m["shape"]), ooff, nb, bp))
        ooff = _align(ooff + nb)
    lay = {"items": items, "qtotal": qoff, "stotal": soff, "other": other,
           "oplan": oplan, "ototal": ooff, "g": g}
    with _layout_lock:
        _layouts[prefix] = lay
    return lay


# ---------------------------------------------------- chunked job planning ---
# Which packed buffer a job writes into. The path/offset plan is immutable for
# a layer and mirrors spine_fast.plan_jobs; only destination memoryviews are
# created for each pass.
_QB, _SCB, _OB = 0, 1, 2


def plan_jobs(lay):
    """Return balanced ``spine_io`` jobs for one mixed packed layer."""
    jobs = lay.get("jobs")
    if jobs is not None:
        return jobs
    raw = []
    for _n, _f, codec, base, _sh, qoff, nb, soff, sb in lay["items"]:
        pe, se = spine_codec.EXT[codec]
        raw.append((base + pe, _QB, qoff, nb))
        raw.append((base + se, _SCB, soff * 2, sb))
    for _n, _f, _dt, _sh, ooff, nb, bp in lay["oplan"]:
        raw.append((bp, _OB, ooff, nb))
    chunk = spine_io.CHUNK
    jobs = []
    for path, buffer_id, destination_offset, nbytes in raw:
        if chunk <= 0 or nbytes <= chunk:
            jobs.append(
                (path, 0, buffer_id, destination_offset, nbytes)
            )
            continue
        file_offset = 0
        while file_offset < nbytes:
            size = min(chunk, nbytes - file_offset)
            jobs.append(
                (
                    path,
                    file_offset,
                    buffer_id,
                    destination_offset + file_offset,
                    size,
                )
            )
            file_offset += size
    jobs.sort(key=lambda job: -job[4])
    lay["jobs"] = jobs
    return jobs


def _rdadvise_next(prefix):
    """Issue optional readahead for the next already-planned mixed layer."""
    try:
        head, _, tail = prefix.rpartition("layers.")
        index = int(tail.rstrip("."))
    except (AttributeError, ValueError):
        return
    nxt = _layouts.get(f"{head}layers.{index + 1}.")
    if nxt is None:
        return
    seen = set()
    for path, _file_offset, _buffer_id, _destination_offset, _nbytes in (
        plan_jobs(nxt)
    ):
        if path not in seen:
            seen.add(path)
            spine_io.rdadvise(path)


# ------------------------------------------------------------- read side ----
def read_pack(module, prefix, int8_dir, res_dir, inv, spine=None, load_resident=None):
    """Runs in the preloader thread. Signature matches spine_fast.read_pack."""
    _sync_runtime_read_threads()
    lay = plan_layout(module, prefix, int8_dir, res_dir, inv)
    qbuf = _acquire(lay["qtotal"])
    sbuf = _acquire(lay["stotal"] * 2)
    obuf = _acquire(max(lay["ototal"], 1))

    if spine_io.ENABLED:
        t0 = _time.time()
        try:
            if spine_io.RDADVISE:
                _rdadvise_next(prefix)
            buffers = (
                memoryview(qbuf),
                memoryview(sbuf),
                memoryview(obuf),
            )
            jobs = [
                (
                    path,
                    file_offset,
                    buffers[buffer_id][
                        destination_offset:destination_offset + nbytes
                    ],
                )
                for (
                    path,
                    file_offset,
                    buffer_id,
                    destination_offset,
                    nbytes,
                ) in plan_jobs(lay)
            ]
            spine_io.run_jobs(
                jobs,
                _pool_exec(),
                READ_THREADS,
                nocache=spine_io.stream_tier(prefix),
            )
        except Exception:
            # A sibling worker may still own a destination view while
            # ``run_jobs`` propagates its first failure. Do not publish these
            # buffers back to the reuse pool; their remaining views keep them
            # alive safely until every worker releases ownership.
            raise
        PHASE["read_s"] += _time.time() - t0
        PHASE["read_bytes"] += lay["qtotal"] + lay["ototal"]
        for item in lay["items"]:
            PHASE["bytes_by_codec"][item[2]] += item[6]
        return {"lay": lay, "q": qbuf, "sc": sbuf, "other": obuf}

    qmv, smv = memoryview(qbuf), memoryview(sbuf)

    def _one(it):
        _n, _f, codec, base, _sh, qoff, nb, soff, sb = it
        pe, se = spine_codec.EXT[codec]
        with open(base + pe, "rb") as f:
            f.readinto(qmv[qoff:qoff + nb])
        with open(base + se, "rb") as f:
            f.readinto(smv[soff * 2:soff * 2 + sb])

    t0 = _time.time()
    items = lay["items"]
    if READ_THREADS > 1 and len(items) > 1:
        list(_pool_exec().map(_one, items))
    else:
        for it in items:
            _one(it)
    omv = memoryview(obuf)
    for _n, _f, _dt, _sh, ooff, nb, bp in lay["oplan"]:
        with open(bp, "rb") as f:
            f.readinto(omv[ooff:ooff + nb])
    PHASE["read_s"] += _time.time() - t0
    PHASE["read_bytes"] += lay["qtotal"] + lay["ototal"]
    for it in items:
        PHASE["bytes_by_codec"][it[2]] += it[6]
    return {"lay": lay, "q": qbuf, "sc": sbuf, "other": obuf}


# ------------------------------------------------------------ apply side ----
def apply_pack(module, prefix, pack, dev, dt, inv, dtmap, set_param, load_resident):
    lay = pack["lay"]
    items = lay["items"]
    g = lay["g"]
    PHASE["n_layers"] += 1
    oplan = lay["oplan"]
    direct_cpu = _cpu_direct_views(dev)
    t0 = _time.time()
    if items:
        qh = torch.frombuffer(pack["q"], dtype=torch.uint8)[:lay["qtotal"]]
        sh = torch.frombuffer(pack["sc"], dtype=torch.float16)[:lay["stotal"]]
        if direct_cpu:
            qd, sd = qh, sh
        else:
            qd = _stage_buf("q", lay["qtotal"], dev, torch.uint8)
            sd = _stage_buf("sc", lay["stotal"], dev, torch.float16)
            qd.copy_(qh)
            sd.copy_(sh)
    if oplan:
        oh = torch.frombuffer(pack["other"], dtype=torch.uint8)[:lay["ototal"]]
        if direct_cpu:
            od = oh
        else:
            od = _stage_buf("other", lay["ototal"], dev, torch.uint8)
            od.copy_(oh)
    PHASE["h2d_s"] += _time.time() - t0
    # Preserve this counter's established payload convention (q + bf16 tail,
    # scales excluded), but direct CPU views perform no staging copy.
    PHASE["h2d_bytes"] += (
        0 if direct_cpu else lay["qtotal"] + lay["ototal"]
    )

    t0 = _time.time()
    for name, full, codec, _base, shape, qoff, nb, soff, sb in items:
        rows, cols = shape
        q = qd[qoff:qoff + nb]
        sc = sd[soff:soff + sb // 2]
        p = module.get_parameter(name)
        fits = (p.device.type != "meta" and p.shape == torch.Size(shape)
                and p.dtype == torch.float32 and dt == torch.float32)
        if fits and dequant_into(p.data, codec, q, sc, g):
            continue
        t = _dequant_torch(codec, q, sc, rows, cols, g).to(dt)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    PHASE["deq_s"] += _time.time() - t0

    t0 = _time.time()
    have = {rec[1] for rec in oplan}
    for name, full, sdt, shape, ooff, nb, _bp in oplan:
        src = od[ooff:ooff + nb].view(dtmap[sdt]).reshape(shape)
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != torch.Size(shape) or p.dtype != dt:
            set_param(module, name, src.to(dt).clone())
        else:
            p.data.copy_(src)
    for name, full, path in lay["other"]:
        if full in have:
            continue
        t = load_resident(full).to(dev, dt)
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    PHASE["other_s"] += _time.time() - t0
    if items:
        del qh, sh, qd, sd
    if oplan:
        del oh, od
    if id(pack.get("q")) in KEEP:
        return
    _release(pack["q"])
    _release(pack["sc"])
    _release(pack["other"])
    pack["q"] = pack["sc"] = pack["other"] = None


# ------------------------------------------- single-tensor convenience ------
def load(full, int8_dir, shape, dev="cpu", dtype=torch.float32):
    """Dequantized tensor on `dev`, or None when `full` is not in any quantized
    spine. Drop-in for kimi_run._load_int8()."""
    codec, d = codec_of(full, int8_dir)
    if codec is None:
        return None
    pe, se = spine_codec.EXT[codec]
    rows, cols = int(shape[0]), int(shape[1])
    g = group_size()
    with open(os.path.join(d, full + pe), "rb") as f:
        qb = f.read()
    with open(os.path.join(d, full + se), "rb") as f:
        sb = f.read()
    dev = torch.device(dev)
    q = torch.frombuffer(bytearray(qb), dtype=torch.uint8).to(dev)
    sc = torch.frombuffer(bytearray(sb), dtype=torch.float16).to(dev)
    if dev.type == "mps" and DEQ == "metal" and metal_available():
        out = torch.empty((rows, cols), dtype=torch.float32, device=dev)
        if dequant_into(out, codec, q, sc, g):
            return out.to(dtype)
    return _dequant_torch(codec, q, sc, rows, cols, g).to(dtype)


def phase_report():
    p = PHASE
    if not p["n_layers"]:
        return ""
    rd = p["read_bytes"] / 1e9 / max(p["read_s"], 1e-9)
    hd = p["h2d_bytes"] / 1e9 / max(p["h2d_s"], 1e-9)
    mix = " ".join(f"{k}:{v/1e9:.1f}GB" for k, v in p["bytes_by_codec"].items() if v)
    return (f"[mixed] {p['n_layers']} layers | read {p['read_s']:.1f}s "
            f"({p['read_bytes']/1e9:.1f} GB, {rd:.1f} GB/s, bg thread) | "
            f"h2d {p['h2d_s']:.1f}s ({hd:.1f} GB/s) | deq {p['deq_s']:.1f}s | "
            f"bf16-tail {p['other_s']:.1f}s | {mix}")


# ------------------------------------------------------------- self test ----
def _check(dev=None):
    """GPU dequant == numpy oracle, bit for bit, for every codec and a few
    awkward shapes (odd column counts, zero rows, saturating values)."""
    import numpy as np
    dev = torch.device(dev or ("mps" if torch.backends.mps.is_available() else "cpu"))
    rng = np.random.default_rng(0)
    ok = True
    for (rows, cols) in [(8, 64), (5, 100), (3, 7168), (17, 4096), (2, 63)]:
        w = (rng.standard_normal((rows, cols), dtype=np.float32) * 0.02)
        w[0, :3] = [0.5, -0.5, 0.0]
        if rows > 3:
            w[2] = 0.0
        for codec in spine_codec.CODECS:
            for g in (32, 64):
                if codec == "i8" and g != 32:
                    continue
                packed, s16, deq = spine_codec.quantize(w, codec, g)
                oracle = spine_codec.dequant(codec, packed.tobytes(), s16.tobytes(),
                                             rows, cols, g)
                assert np.array_equal(oracle, deq)
                q = torch.frombuffer(bytearray(packed.tobytes()), dtype=torch.uint8).to(dev)
                sc = torch.frombuffer(bytearray(s16.tobytes()), dtype=torch.float16).to(dev)
                out = torch.empty((rows, cols), dtype=torch.float32, device=dev)
                out.fill_(float("nan"))
                used_metal = dequant_into(out, codec, q, sc, g)
                got = out.cpu().numpy()
                tr = _dequant_torch(codec, q, sc, rows, cols, g).cpu().numpy()
                d1 = float(np.abs(got - oracle).max()) if used_metal else 0.0
                d2 = float(np.abs(tr - oracle).max())
                tag = "metal" if used_metal else "torch-only"
                if d1 or d2:
                    ok = False
                print(f"  [{rows:3d},{cols:5d}] {codec} g={g:3d} {tag:10s} "
                      f"kernel_maxdiff={d1:.1e} torch_maxdiff={d2:.1e} "
                      f"rel_fro={np.linalg.norm(oracle-w)/np.linalg.norm(w):.4f}"
                      + ("" if (d1 == 0 and d2 == 0) else "   <-- MISMATCH"))
    print("CHECK", "PASS (bit-identical to the numpy oracle)" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sys.exit(0 if _check() else 1)
    print(f"[mixed] dir={MIXED_DIR} available={available()}")
    print(f"[mixed] {describe()}")
