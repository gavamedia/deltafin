"""CUDA MoE expert path for Deltafin.

Fused MXFP4 dequant + GEMV + SiTU + weighted reduction on CUDA GPUs.
All tensors stay on GPU — no numpy round-trips in the hot path.
"""
from __future__ import annotations

import ctypes
import os
import struct
import threading

import numpy as np
import torch

try:
    from runtime_platform import native_build_command
except ImportError:
    from .runtime_platform import native_build_command

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get("K3_CUDA_LIB", os.path.join(_HERE, "libcudamoe.so"))

HIDDEN = 3584
INTER = 3072
_P = 5505024
_S = 344064
EXPERT_SPAN = 3 * (_P + _S)

_lib = None
_load_error = None
_lock = threading.Lock()

# Pre-allocated GPU descriptor buffer (reused across calls)
_DESC_BUF = None       # torch.uint8 tensor on CUDA
_DESC_CAP = 0          # max entries the buffer can hold
_DESC_BYTE = 56        # sizeof(ExpertDesc) on CUDA

# Pre-allocated GPU output buffer
_OUT_BUF = None
_OUT_CAP = 0

# GPU expert cache: maps expert_id -> _ExpertUpload (LRU, pinned)
# Auto-sizes from VRAM unless K3_CUDA_EXPERT_CACHE is explicitly set.
_CACHE_ENV = os.environ.get("K3_CUDA_EXPERT_CACHE")
_RESERVED_BYTES = int(11.6e9)  # template arena + pilot gates + spine staging + misc


def _auto_cache_size():
    """Pick a safe expert cache size from device VRAM.

    Budget = 75% of card, subtract fixed reserved overhead, divide by
    expert byte size.  Clamped to [128, 2048].
    """
    if not torch.cuda.is_available():
        return 256
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        budget = int(total * 0.75)
        avail = max(0, budget - _RESERVED_BYTES)
        return max(128, min(avail // EXPERT_SPAN, 2048))
    except Exception:
        return 256


if _CACHE_ENV is not None:
    _GPU_CACHE_MAX = int(_CACHE_ENV)
else:
    _GPU_CACHE_MAX = _auto_cache_size()
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        cache_gb = _GPU_CACHE_MAX * EXPERT_SPAN / 1e9
        print(f"[cuda-moe] VRAM {total_gb:.0f} GiB | "
              f"GPU expert cache: {_GPU_CACHE_MAX} entries ({cache_gb:.1f} GiB)",
              flush=True)

_gpu_cache = {}
_gpu_cache_order = []
_gpu_cache_lock = threading.Lock()

# Optional CPU cross-check: compare CUDA output against fast_moe for first N calls
CHECK = int(os.environ.get("K3_MOE_CHECK", "0"))
CHECK_TOL = float(os.environ.get("K3_MOE_CHECK_TOL", "1e-4"))
_checked = 0
_check_worst = 0.0


def check_stats():
    return {"checked": _checked, "worst_rel": _check_worst}


def _checked_output(out, x_cpu, ids, weights, raw_experts):
    """Compare CUDA output against the CPU reference path.

    out: torch [H] CUDA tensor  |  x_cpu: torch [H] CPU tensor
    ids: [K] ints  |  weights: [K] floats

    Runs on the same raw_experts buffers the GPU just read, in the same
    layer — catches weight-corruption bugs (stale cache aliasing, etc.)
    that produce wrong tokens.
    """
    global _checked, _check_worst
    if _checked >= CHECK:
        return
    _checked += 1
    import fast_moe
    # Build [1, H], [1, K], [1, K] per fast_moe's interface
    x_2d = x_cpu.unsqueeze(0)  # [1, H]
    kid = torch.tensor([ids], dtype=torch.long)       # [1, K]
    kwt = torch.tensor([weights], dtype=torch.float32) # [1, K]
    ref = fast_moe.moe_infer_fast(x_2d, kid, kwt, raw_experts)
    ref_np = ref.detach().to("cpu", torch.float32).numpy()
    out_np = out.detach().to("cpu", torch.float32).numpy()
    denom = max(np.max(np.abs(ref_np)), np.max(np.abs(out_np)), 1e-10)
    rel = float(np.max(np.abs(ref_np - out_np)) / denom)
    _check_worst = max(_check_worst, rel)
    if rel > CHECK_TOL:
        raise RuntimeError(
            f"K3_MOE_CHECK: CUDA vs CPU rel_error {rel:.3e} > {CHECK_TOL:g} "
            f"on call {_checked} (ids={ids})")
    elif _checked <= 3:  # quiet after first few
        print(f"[cuda-moe] check {_checked}: CUDA vs CPU rel_error {rel:.3e} "
              f"(tol={CHECK_TOL:g}) OK", flush=True)


def _gpu_cache_get(eid, raw, device_id=0):
    with _gpu_cache_lock:
        if eid in _gpu_cache:
            _gpu_cache_order.remove(eid)
            _gpu_cache_order.append(eid)
            return _gpu_cache[eid]
        while len(_gpu_cache) >= _GPU_CACHE_MAX:
            victim = _gpu_cache_order.pop(0)
            del _gpu_cache[victim]
        upload = _ExpertUpload(raw, device_id)
        _gpu_cache[eid] = upload
        _gpu_cache_order.append(eid)
        return upload


def gpu_cache_stats():
    with _gpu_cache_lock:
        return {"entries": len(_gpu_cache), "max": _GPU_CACHE_MAX}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    with _lock:
        if _lib is not None or _load_error is not None:
            return _lib
        if not os.path.isfile(_SO):
            _load_error = (
                f"CUDA MoE library not found: {_SO}. Build with: "
                f"nvcc -O3 -shared -o {_SO} {os.path.join(_HERE, 'cuda_moe_kernels.cu')}"
            )
            return None
        try:
            lib = ctypes.CDLL(_SO)
        except OSError as e:
            _load_error = f"{_SO}: {e}"
            return None

        lib.cuda_moe_available.argtypes = [ctypes.c_int]
        lib.cuda_moe_available.restype = ctypes.c_int

        lib.cuda_mxfp4_moe_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.cuda_mxfp4_moe_layer.restype = None

        lib.cuda_moe_zero_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.cuda_moe_zero_output.restype = None

        lib.cuda_int8_deq.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.cuda_int8_deq.restype = None

        lib.cuda_moe_error.restype = ctypes.c_char_p

        if not lib.cuda_moe_available(0):
            _load_error = "CUDA device 0 is unavailable or below sm_75"
            return None

        _lib = lib
        return _lib


def available():
    return _load() is not None


def last_error():
    return _load_error or ""


def describe():
    if not available():
        return f"unavailable ({last_error()})"
    check_info = f", check={CHECK}" if CHECK else ""
    cache_info = f", cache={len(_gpu_cache)}/{_GPU_CACHE_MAX}" if _gpu_cache else ""
    return f"CUDA MoE kernels loaded{check_info}{cache_info}"


# ──────── per-expert upload + GPU cache ────────────────────────────────────

class _ExpertUpload:
    """Holds GPU copies of one expert's three matrices."""

    __slots__ = ("w1_p", "w1_s", "w3_p", "w3_s", "w2_p", "w2_s", "_refs")

    def __init__(self, raw, device_id=0):
        dev = f"cuda:{device_id}"
        refs = []

        def _upload(a):
            # Pinned CPU tensor enables async DMA H2D (~2x throughput)
            arr = np.ascontiguousarray(a)
            cpu = torch.empty(arr.shape, dtype=torch.uint8, pin_memory=True)
            cpu.copy_(torch.from_numpy(arr))
            gpu = cpu.to(device=dev, dtype=torch.uint8, non_blocking=True)
            refs.append(gpu)
            return gpu

        w1p, w1s = raw["w1"]
        w3p, w3s = raw["w3"]
        w2p, w2s = raw["w2"]
        self.w1_p = _upload(w1p).data_ptr()
        self.w1_s = _upload(w1s).data_ptr()
        self.w3_p = _upload(w3p).data_ptr()
        self.w3_s = _upload(w3s).data_ptr()
        self.w2_p = _upload(w2p).data_ptr()
        self.w2_s = _upload(w2s).data_ptr()
        self._refs = refs


# ──────── GPU descriptor builder (pre-allocated buffers) ───────────────────

def _ensure_buffers(num_experts, H, device):
    """Grow pre-allocated GPU buffers if needed."""
    global _DESC_BUF, _DESC_CAP, _OUT_BUF, _OUT_CAP

    if _DESC_BUF is None or _DESC_CAP < num_experts:
        _DESC_BUF = torch.zeros(num_experts * _DESC_BYTE, dtype=torch.uint8, device=device)
        _DESC_CAP = num_experts

    if _OUT_BUF is None or _OUT_CAP < H:
        _OUT_BUF = torch.zeros(H, dtype=torch.float32, device=device)
        _OUT_CAP = H

    return _DESC_BUF, _OUT_BUF


def _build_descriptors(ids, weights, uploads, device, stream):
    """Fill the pre-allocated GPU descriptor buffer with expert pointers.

    Uses struct.pack_into for tight byte-level serialization, avoiding
    ctypes.Structure overhead.  Then does one small H2D copy.
    """
    num_experts = len(ids)
    buf = bytearray(num_experts * _DESC_BYTE)

    for i, eid in enumerate(ids):
        u = uploads[int(eid)]
        off = i * _DESC_BYTE
        struct.pack_into('<Q', buf, off + 0, u.w1_p)
        struct.pack_into('<Q', buf, off + 8, u.w1_s)
        struct.pack_into('<Q', buf, off + 16, u.w3_p)
        struct.pack_into('<Q', buf, off + 24, u.w3_s)
        struct.pack_into('<Q', buf, off + 32, u.w2_p)
        struct.pack_into('<Q', buf, off + 40, u.w2_s)
        struct.pack_into('<f', buf, off + 48, float(weights[i]))
        # bytes 52-55 are padding

    # Pinned + async H2D for descriptor data
    desc_t = torch.frombuffer(buf, dtype=torch.uint8)
    desc_pinned = desc_t.pin_memory()
    desc_dev = desc_pinned.to(device=device, non_blocking=True)
    return desc_dev


# ──────── main MoE inference ──────────────────────────────────────────────

def moe_infer(x, topk_ids, topk_weight, raw_experts, routing_record=None):
    """CUDA MoE inference — all tensors stay on GPU.

    Args:
        x: torch [N, 3584] fp32 on CUDA (or CPU — will be moved)
        topk_ids: torch [N, top_k] int
        topk_weight: torch [N, top_k] fp32
        raw_experts: dict {eid: {w1|w2|w3: (packed_u8, scale_u8)}}
        routing_record: optional dict with 'ids' and 'weights'

    Returns:
        torch [N, 3584] fp32 on same device as x
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA MoE unavailable: {last_error()}")

    # Move x to CUDA if not already
    if x.device.type != "cuda":
        x = x.cuda()
    device = x.device
    device_id = device.index or 0

    N, H = x.shape
    if H != HIDDEN:
        raise ValueError(f"expected hidden={HIDDEN}, got {H}")

    if routing_record is not None:
        ids = routing_record["ids"]
        ws = routing_record["weights"]
    else:
        ids = topk_ids.tolist()
        ws = topk_weight.to(torch.float32).tolist()

    # Pre-allocate GPU buffers
    max_exp = max(len(row) for row in ids)
    _ensure_buffers(max_exp, H, device)

    out_np = np.zeros((N, H), dtype=np.float32) if N > 1 else None
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    x_cpu = x.detach().to("cpu", torch.float32).numpy() if CHECK > 0 else None

    for t in range(N):
        pos_ids = ids[t]
        pos_ws = ws[t]
        num_e = len(pos_ids)

        # Get cached GPU uploads
        uploads = {}
        for eid in set(int(e) for e in pos_ids):
            uploads[eid] = _gpu_cache_get(eid, raw_experts[eid], device_id)

        # Build descriptor array on GPU (async H2D)
        desc_dev = _build_descriptors(pos_ids, pos_ws, uploads, device,
                                       torch.cuda.current_stream())

        # Zero output buffer
        lib.cuda_moe_zero_output(
            ctypes.c_void_p(_OUT_BUF.data_ptr()), H, stream_ptr)

        # Launch fused MoE layer kernel
        lib.cuda_mxfp4_moe_layer(
            ctypes.c_void_p(_OUT_BUF.data_ptr()),
            ctypes.c_void_p(x[t].data_ptr()),
            ctypes.c_void_p(desc_dev.data_ptr()),
            H, INTER, num_e,
            stream_ptr,
        )

        if N > 1:
            out_np[t] = _OUT_BUF.cpu().numpy()
        else:
            result = _OUT_BUF.to(x.dtype)  # stays on GPU

    if CHECK > 0:
        # Sync before CPU comparison
        torch.cuda.synchronize(device)
        if N > 1:
            out_t = torch.from_numpy(out_np).to(device=device, dtype=x.dtype)
            out_slice = out_t[0]
            x_slice = torch.from_numpy(x_cpu[0])
        else:
            out_slice = result
            x_slice = torch.from_numpy(x_cpu)
        # Check the first position only (covers the common N=1 case)
        _checked_output(out_slice, x_slice, ids[0], ws[0], raw_experts)

    if N > 1:
        return torch.from_numpy(out_np).to(device=device, dtype=x.dtype)
    return result


# ──────── int8 spine dequant helper (Phase 4) ────────────────────────────

def dequant_int8(q, sc, out=None):
    """CUDA int8 dequant: out[i] = float(q[i]) * float(sc[i / cols]).

    q: int8 tensor on CUDA [rows, cols]
    sc: fp16 tensor on CUDA [rows]
    out: optional fp32 output on CUDA [rows, cols]
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA MoE unavailable: {last_error()}")
    rows, cols = q.shape
    if out is None:
        out = torch.empty(rows, cols, device=q.device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    lib.cuda_int8_deq(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        rows, cols,
        ctypes.c_void_p(stream_ptr),
    )
    return out
