"""CUDA MoE expert path for Deltafin.

Fused MXFP4 dequant + GEMV + SiTU + weighted reduction on CUDA GPUs.
Drop-in for tools/fast_moe.py's moe_infer_fast — same signature, same semantics.

Phases implemented:
  Phase 1: Per-matrix MXFP4 dequant+GEMV (cuda_mxfp4_gemv)
  Phase 2: Fused expert FFN  (cuda_mxfp4_expert_ffn)
  Phase 3: Grouped MoE layer dispatch (cuda_mxfp4_moe_layer)
  Phase 4: Int8 spine dequant (cuda_int8_deq)  — available but used via spine_fast.py
  Phase 5: Multi-position batching (cuda_mxfp4_moe_positions)

Build:  nvcc -O3 -shared -o libcudamoe.so tools/cuda_moe_kernels.cu
        (or run tools/build_native.py)
"""
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np
import torch

try:
    from runtime_platform import (
        native_build_command,
        NativeLibraryError,
    )
except ImportError:
    from .runtime_platform import native_build_command, NativeLibraryError

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get(
    "K3_CUDA_LIB",
    os.path.join(_HERE, "libcudamoe.so"),
)

HIDDEN = 3584   # routed expert hidden size
INTER = 3072    # intermediate size
_P = 5505024    # packed bytes per matrix
_S = 344064     # scale bytes per matrix
EXPERT_SPAN = 3 * (_P + _S)  # 17,547,264 — one expert blob

_lib = None
_load_error = None
_lock = threading.Lock()
_device = -1
_stream = None

# GPU expert cache: maps expert_id -> _ExpertUpload (LRU, pinned)
# Limits H2D traffic: upload once, reuse across layers and decode tokens.
_GPU_CACHE_MAX = int(os.environ.get("K3_CUDA_EXPERT_CACHE", "512"))
_gpu_cache = {}
_gpu_cache_order = []
_gpu_cache_lock = threading.Lock()


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

        # Set argtypes
        lib.cuda_moe_available.argtypes = [ctypes.c_int]
        lib.cuda_moe_available.restype = ctypes.c_int

        lib.cuda_mxfp4_gemv.argtypes = [
            ctypes.c_void_p,  # y (device)
            ctypes.c_void_p,  # x (device)
            ctypes.c_void_p,  # packed (device)
            ctypes.c_void_p,  # scales (device)
            ctypes.c_int,     # rows
            ctypes.c_int,     # cols
            ctypes.c_void_p,  # stream
        ]
        lib.cuda_mxfp4_gemv.restype = None

        lib.cuda_mxfp4_moe_layer.argtypes = [
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # descs (device)
            ctypes.c_int,     # H
            ctypes.c_int,     # I
            ctypes.c_int,     # num_experts
            ctypes.c_void_p,  # stream
        ]
        lib.cuda_mxfp4_moe_layer.restype = None

        lib.cuda_moe_zero_output.argtypes = [
            ctypes.c_void_p,  # out
            ctypes.c_int,     # n
            ctypes.c_void_p,  # stream
        ]
        lib.cuda_moe_zero_output.restype = None

        lib.cuda_int8_deq.argtypes = [
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # sc
            ctypes.c_int,     # rows
            ctypes.c_int,     # cols
            ctypes.c_void_p,  # stream
        ]
        lib.cuda_int8_deq.restype = None

        lib.cuda_mxfp4_moe_positions.argtypes = [
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # descs (device)
            ctypes.c_void_p,  # expert_counts (device)
            ctypes.c_int,     # H
            ctypes.c_int,     # I
            ctypes.c_int,     # num_positions
            ctypes.c_int,     # max_experts
            ctypes.c_void_p,  # stream
        ]
        lib.cuda_mxfp4_moe_positions.restype = None

        lib.cuda_moe_error.restype = ctypes.c_char_p

        # Probe device 0
        if not lib.cuda_moe_available(0):
            _load_error = f"CUDA device 0 is unavailable or below sm_70"
            return None

        _lib = lib
        return _lib


def available():
    """True if CUDA MoE library loaded and device ready."""
    return _load() is not None


def last_error():
    return _load_error or ""


def _get_stream():
    """Return a CUDA stream pointer from the current torch CUDA stream."""
    global _stream
    if not torch.cuda.is_available():
        return None
    # We use torch's current stream — get its native handle
    s = torch.cuda.current_stream()
    # The stream ctypes pointer is accessible via cuda_stream
    return s.cuda_stream


# ──────── per-expert MXFP4 data management (upload experts to GPU) ────────

def _expert_ptr(packed_np, scale_np, device_id=0):
    """Upload MXFP4 packed+scale for one matrix and return device pointer tuple.

    Returns (dev_packed_ptr, dev_scale_ptr) as ints.
    """
    packed_dev = torch.from_numpy(
        np.ascontiguousarray(packed_np)
    ).to(device=f"cuda:{device_id}", dtype=torch.uint8, non_blocking=True)
    scale_dev = torch.from_numpy(
        np.ascontiguousarray(scale_np)
    ).to(device=f"cuda:{device_id}", dtype=torch.uint8, non_blocking=True)
    return packed_dev.data_ptr(), scale_dev.data_ptr(), (packed_dev, scale_dev)


class _ExpertUpload:
    """Hold GPU copies of one expert's three matrices, keeping tensor refs alive."""

    __slots__ = ("w1_p", "w1_s", "w3_p", "w3_s", "w2_p", "w2_s", "_refs")

    def __init__(self, raw, device_id=0):
        # raw dict: {w1: (packed_np, scale_np), w2: ..., w3: ...}
        refs = []
        w1p, w1s, r1 = _expert_ptr(raw["w1"][0], raw["w1"][1], device_id)
        w3p, w3s, r3 = _expert_ptr(raw["w3"][0], raw["w3"][1], device_id)
        w2p, w2s, r2 = _expert_ptr(raw["w2"][0], raw["w2"][1], device_id)
        self.w1_p = w1p
        self.w1_s = w1s
        self.w3_p = w3p
        self.w3_s = w3s
        self.w2_p = w2p
        self.w2_s = w2s
        self._refs = [*r1, *r3, *r2]


def _moe_infer_cuda(x_np, ids, weights, raw_experts):
    """Run MoE for one position using the phased CUDA kernels.

    Phase 3 (grouped) is attempted first; falls back to Phase 2 (per-expert fused FFN)
    if the grouped path hits a shape issue.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA MoE unavailable: {last_error()}")
    device_id = 0

    # Upload input x
    x_dev = torch.from_numpy(np.ascontiguousarray(x_np)).to(
        device=f"cuda:{device_id}", dtype=torch.float32, non_blocking=True)
    H = x_np.shape[-1]
    I = INTER
    num_experts = len(ids)

    # Upload unique experts to GPU (with caching)
    device_id_local = device_id
    uploads = {}
    for eid_val in set(int(e) for e in ids):
        uploads[eid_val] = _gpu_cache_get(eid_val, raw_experts[eid_val], device_id_local)

    # Output buffer
    out_dev = torch.zeros(H, device=f"cuda:{device_id}", dtype=torch.float32)

    stream_ptr = _get_stream()

    # Build ExpertDesc struct array on device
    import ctypes

    class ExpertDesc(ctypes.Structure):
        _fields_ = [
            ("w1_packed", ctypes.c_void_p),
            ("w1_scales", ctypes.c_void_p),
            ("w3_packed", ctypes.c_void_p),
            ("w3_scales", ctypes.c_void_p),
            ("w2_packed", ctypes.c_void_p),
            ("w2_scales", ctypes.c_void_p),
            ("routing_weight", ctypes.c_float),
        ]

    descs_host = (ExpertDesc * num_experts)()
    for i, eid in enumerate(ids):
        u = uploads[int(eid)]
        descs_host[i] = ExpertDesc(
            u.w1_p, u.w1_s, u.w3_p, u.w3_s, u.w2_p, u.w2_s,
            float(weights[i]),
        )

    # Upload ExpertDesc array to device
    descs_dev = torch.from_numpy(
        np.frombuffer(bytearray(descs_host), dtype=np.uint8).copy()
    ).to(device=f"cuda:{device_id}", dtype=torch.uint8, non_blocking=True)

    # Reset output to zero
    lib.cuda_moe_zero_output(
        ctypes.c_void_p(out_dev.data_ptr()),
        H, stream_ptr)

    # Launch grouped kernel
    lib.cuda_mxfp4_moe_layer(
        ctypes.c_void_p(out_dev.data_ptr()),
        ctypes.c_void_p(x_dev.data_ptr()),
        ctypes.c_void_p(descs_dev.data_ptr()),
        H, I, num_experts,
        stream_ptr,
    )

    # Sync and read back
    torch.cuda.synchronize(device_id)
    out_np = out_dev.cpu().numpy().astype(np.float32)

    # Clean up: tensors will be GC'd when uploads goes out of scope
    return out_np


def moe_infer(
    x, topk_ids, topk_weight, raw_experts, routing_record=None
):
    """CUDA MoE inference — drop-in for fast_moe.moe_infer_fast.

    Args:
        x: torch [N, 3584] fp32 on CUDA (or CPU — will be moved)
        topk_ids: torch [N, top_k] int
        topk_weight: torch [N, top_k] fp32
        raw_experts: dict {eid: {w1|w2|w3: (packed_u8, scale_u8)}}
        routing_record: optional dict with 'ids' and 'weights' (used over topk_*)

    Returns:
        torch [N, 3584] fp32 on same device as x
    """
    device = x.device
    x_np = x.detach().to("cpu", dtype=torch.float32).numpy()
    N, H = x_np.shape

    if routing_record is not None:
        ids = routing_record["ids"]
        ws = routing_record["weights"]
    else:
        ids = topk_ids.tolist()
        ws = topk_weight.to(torch.float32).tolist()

    out_np = np.zeros((N, H), dtype=np.float32)

    # Phase 5: multi-position batching when N > 1
    if N > 1 and _load() is not None:
        try:
            out_np = _moe_infer_positions_cuda(x_np, ids, ws, raw_experts)
            return torch.from_numpy(out_np).to(device=device, dtype=x.dtype)
        except Exception:
            # Fall through to per-position loop
            pass

    for t in range(N):
        out_np[t] = _moe_infer_cuda(x_np[t], ids[t], ws[t], raw_experts)

    return torch.from_numpy(out_np).to(device=device, dtype=x.dtype)


def _moe_infer_positions_cuda(x_np, ids, weights, raw_experts):
    """Phase 5: multi-position grouped MoE for speculative decode."""
    lib = _load()
    device_id = 0
    N = x_np.shape[0]
    H = x_np.shape[-1]
    I = INTER

    # Upload x
    x_dev = torch.from_numpy(np.ascontiguousarray(x_np)).to(
        device=f"cuda:{device_id}", dtype=torch.float32, non_blocking=True)

    # Collect experts across all positions (with caching)
    all_eids = set()
    for pos_ids in ids:
        all_eids.update(int(e) for e in pos_ids)
    uploads = {eid: _gpu_cache_get(eid, raw_experts[eid], device_id) for eid in all_eids}

    max_experts = max(len(pos_ids) for pos_ids in ids)

    import ctypes

    class PosExpertDesc(ctypes.Structure):
        _fields_ = [
            ("w1_packed", ctypes.c_void_p),
            ("w1_scales", ctypes.c_void_p),
            ("w3_packed", ctypes.c_void_p),
            ("w3_scales", ctypes.c_void_p),
            ("w2_packed", ctypes.c_void_p),
            ("w2_scales", ctypes.c_void_p),
            ("routing_weight", ctypes.c_float),
            ("expert_id", ctypes.c_int),
        ]

    total_descs = N * max_experts
    descs_host = (PosExpertDesc * total_descs)()
    for n in range(N):
        for k, eid in enumerate(ids[n]):
            u = uploads[int(eid)]
            idx = n * max_experts + k
            descs_host[idx] = PosExpertDesc(
                u.w1_p, u.w1_s, u.w3_p, u.w3_s, u.w2_p, u.w2_s,
                float(weights[n][k]), int(eid),
            )

    descs_dev = torch.from_numpy(
        np.frombuffer(bytearray(descs_host), dtype=np.uint8).copy()
    ).to(device=f"cuda:{device_id}", dtype=torch.uint8, non_blocking=True)

    expert_counts = torch.tensor(
        [len(pos_ids) for pos_ids in ids],
        device=f"cuda:{device_id}", dtype=torch.int32,
    )

    out_dev = torch.zeros(N * H, device=f"cuda:{device_id}", dtype=torch.float32)
    stream_ptr = _get_stream()

    lib.cuda_mxfp4_moe_positions(
        ctypes.c_void_p(out_dev.data_ptr()),
        ctypes.c_void_p(x_dev.data_ptr()),
        ctypes.c_void_p(descs_dev.data_ptr()),
        ctypes.c_void_p(expert_counts.data_ptr()),
        H, I, N, max_experts,
        stream_ptr,
    )

    torch.cuda.synchronize(device_id)
    out_np = out_dev.cpu().numpy().reshape(N, H).astype(np.float32)
    return out_np


# ──────── int8 spine dequant helper (Phase 4) ────────────────────────────

def dequant_int8(
    q: torch.Tensor,
    sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA int8 dequant: out[i] = float(q[i]) * float(sc[i / cols]).

    q:  int8 tensor on CUDA [rows, cols]
    sc: fp16 tensor on CUDA [rows]  (row scales)
    out: optional fp32 output tensor on CUDA [rows, cols]
    Returns out tensor.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA MoE unavailable: {last_error()}")

    rows, cols = q.shape
    if out is None:
        out = torch.empty(rows, cols, device=q.device, dtype=torch.float32)

    stream_ptr = _get_stream()
    lib.cuda_int8_deq(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        rows, cols,
        stream_ptr,
    )
    return out


def describe():
    """Return a short status string."""
    if not available():
        return f"unavailable ({last_error()})"
    return "CUDA MoE kernels loaded (phases 1-5)"
