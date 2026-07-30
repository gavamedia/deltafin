"""Safe, capability-gated CUDA MXFP4 MoE bridge.

The native library is intentionally stateless: PyTorch owns all device memory,
the current stream is passed explicitly, and every tensor exposed to ctypes is
recorded on that stream.  Expert residency is keyed by
``(model, device, layer, expert)`` and divided into stable per-layer quotas, so
a forward traversal cannot turn a small cache into a cyclic whole-cache flush.
Route-order accumulation combines independent expert outputs deterministically;
the native kernels never use floating-point atomics for routing.

Nothing is loaded at import time.  ``available(device)`` accepts the backend
only after an ABI/shape handshake and deterministic on-device known-answer
tests.  Callers should retain their established CPU fallback for any exception.
"""
from __future__ import annotations

import collections
import ctypes
import math
import numbers
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from runtime_platform import native_library_filename
except ImportError:  # imported as tools.cuda_moe instead of a top-level module
    from .runtime_platform import native_library_filename


ABI_VERSION = 1
POINTER_LAYOUT = 1
HIDDEN = 3584
INTERMEDIATE = 3072
PACKED_BYTES = 5_505_024
SCALE_BYTES = 344_064
EXPERT_SPAN = 17_547_264
DEFAULT_LAYER_STRATA = 92

_HERE = os.path.dirname(os.path.abspath(__file__))
# The platforms tools/build_native.py can produce this library for.  macOS has
# no CUDA at all, so the check stays a positive list rather than a Linux test.
_CUDA_MOE_PLATFORM = (
    sys.platform.startswith("linux") or sys.platform == "win32"
)
_LIBRARY_PATH = os.environ.get(
    "K3_CUDA_MOE_LIB",
    os.path.join(_HERE, native_library_filename("libcudamoe")),
)
_DEFAULT_MODEL_KEY = ("deltafin-default-model",)
_REQUIRED_SYMBOLS = (
    "k3_cuda_moe_abi_version",
    "k3_cuda_moe_shapes",
    "k3_cuda_moe_available",
    "k3_cuda_moe_launch",
    "k3_cuda_mxfp4_gemv",
    "k3_cuda_int8_dequant",
    "k3_cuda_last_error",
)

_OFF = {"0", "off", "false", "no"}
_AUTO = {"", "auto"}
_BYTE_SUFFIXES = {
    "b": 1,
    "k": 1 << 10,
    "kb": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mb": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gb": 1 << 30,
    "gib": 1 << 30,
}


class CudaMoeError(RuntimeError):
    """A CUDA MoE capability, ABI, launch, or lifetime check failed."""


@dataclass
class _CacheEntry:
    key: tuple[Hashable, int, int, int]
    value: Any


@dataclass
class _ResidentExpert:
    """Immutable device span plus the event that makes its bytes visible."""

    tensor: Any
    ready_event: Any


class _LayerStratifiedCache:
    """Fixed per-layer quotas with LRU only inside one layer's allocation.

    Dividing capacity at construction makes admission independent of traversal
    order.  When capacity is smaller than the layer count, a fixed permutation
    selects a stable spread of layers; unselected layers simply do not cache.
    """

    def __init__(self, capacity: int, layer_strata: int = DEFAULT_LAYER_STRATA):
        if capacity < 0:
            raise ValueError("cache capacity cannot be negative")
        if layer_strata <= 0:
            raise ValueError("layer_strata must be positive")
        self.capacity = int(capacity)
        self.layer_strata = int(layer_strata)
        self._layers: dict[
            int, collections.OrderedDict[tuple[Hashable, int, int, int], _CacheEntry]
        ] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejections = 0

    def _slot(self, layer_index: int) -> int:
        return int(layer_index) % self.layer_strata

    def quota(self, layer_index: int) -> int:
        base, remainder = divmod(self.capacity, self.layer_strata)
        slot = self._slot(layer_index)
        # 53 is coprime to 92, the default stratum count.  For a caller-selected
        # count where it is not coprime, the expression is still deterministic;
        # the rank fallback keeps the quota sum exact.
        if self.layer_strata == DEFAULT_LAYER_STRATA:
            rank = (slot * 53 + 17) % self.layer_strata
        else:
            rank = slot
        return base + int(rank < remainder)

    def peek(self, key: tuple[Hashable, int, int, int]) -> Any | None:
        layer = self._layers.get(self._slot(key[2]))
        entry = None if layer is None else layer.get(key)
        return None if entry is None else entry.value

    def get(self, key: tuple[Hashable, int, int, int]) -> Any | None:
        layer = self._layers.get(self._slot(key[2]))
        entry = None if layer is None else layer.get(key)
        if entry is None:
            self.misses += 1
            return None
        layer.move_to_end(key)
        self.hits += 1
        return entry.value

    def put(
        self, key: tuple[Hashable, int, int, int], value: Any
    ) -> tuple[bool, Any | None]:
        quota = self.quota(key[2])
        if quota == 0:
            self.rejections += 1
            return False, None
        slot = self._slot(key[2])
        layer = self._layers.setdefault(slot, collections.OrderedDict())
        old = layer.pop(key, None)
        evicted = None if old is None else old.value
        if old is None and len(layer) >= quota:
            _, victim = layer.popitem(last=False)
            evicted = victim.value
            self.evictions += 1
        layer[key] = _CacheEntry(key, value)
        return True, evicted

    def clear(self) -> None:
        self._layers.clear()

    def __len__(self) -> int:
        return sum(len(layer) for layer in self._layers.values())

    def layer_counts(self) -> dict[int, int]:
        return {
            slot: len(layer)
            for slot, layer in self._layers.items()
            if layer
        }


class _PinnedStage:
    """One reusable pinned expert span, fenced before every CPU overwrite."""

    def __init__(self) -> None:
        self.tensor = None
        self.numpy = None
        self.pending = None

    def prepare(self) -> tuple[Any, np.ndarray]:
        if self.pending is not None:
            # Host writes cannot be ordered with stream.wait_event: the CPU must
            # wait until the prior DMA has stopped reading this pinned storage.
            self.pending.synchronize()
            self.pending = None
        if self.tensor is None:
            self.tensor = torch.empty(
                EXPERT_SPAN, dtype=torch.uint8, pin_memory=True
            )
            if not bool(self.tensor.is_pinned()):
                raise CudaMoeError("CUDA expert staging allocation is not pinned")
            self.numpy = self.tensor.numpy()
        return self.tensor, self.numpy

    def copied_on(self, stream):
        event = torch.cuda.Event(blocking=False, interprocess=False)
        event.record(stream)
        self.pending = event
        return event


class _DeviceState:
    def __init__(
        self,
        index: int,
        *,
        layer_strata: int,
    ) -> None:
        self.index = index
        self.free_bytes = 0
        self.total_bytes = 0
        self.reserve_bytes = 0
        self.capacity = 0
        self.layer_strata = int(layer_strata)
        self.budget_ready = False
        self.cache = _LayerStratifiedCache(0, layer_strata)
        self.stage = _PinnedStage()
        self.lock = threading.RLock()
        self.checked = False
        self.enabled = False
        self.reason: str | None = None
        self.uploads = 0
        self.calls = 0
        self.routes = 0
        self.self_test: dict[str, Any] | None = None


_lib = None
_load_error: str | None = None
_load_lock = threading.Lock()
_states: dict[int, _DeviceState] = {}
_states_lock = threading.Lock()


def _last_error(lib) -> str:
    try:
        value = lib.k3_cuda_last_error()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    except Exception:
        return "native library supplied no error detail"


def _bind_and_handshake(lib) -> None:
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        raise CudaMoeError(
            "incompatible CUDA MoE library; missing " + ", ".join(missing)
        )

    lib.k3_cuda_moe_abi_version.argtypes = []
    lib.k3_cuda_moe_abi_version.restype = ctypes.c_uint32
    lib.k3_cuda_moe_shapes.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.k3_cuda_moe_shapes.restype = None
    lib.k3_cuda_moe_available.argtypes = [ctypes.c_int]
    lib.k3_cuda_moe_available.restype = ctypes.c_int
    lib.k3_cuda_moe_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.k3_cuda_moe_launch.restype = ctypes.c_int
    lib.k3_cuda_mxfp4_gemv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.k3_cuda_mxfp4_gemv.restype = ctypes.c_int
    lib.k3_cuda_int8_dequant.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.k3_cuda_int8_dequant.restype = ctypes.c_int
    lib.k3_cuda_last_error.argtypes = []
    lib.k3_cuda_last_error.restype = ctypes.c_char_p

    observed_abi = int(lib.k3_cuda_moe_abi_version())
    if observed_abi != ABI_VERSION:
        raise CudaMoeError(
            f"CUDA MoE ABI mismatch: found {observed_abi}, need {ABI_VERSION}"
        )
    hidden, intermediate = ctypes.c_int(), ctypes.c_int()
    span, layout = ctypes.c_int64(), ctypes.c_uint32()
    lib.k3_cuda_moe_shapes(
        ctypes.byref(hidden),
        ctypes.byref(intermediate),
        ctypes.byref(span),
        ctypes.byref(layout),
    )
    observed = (hidden.value, intermediate.value, span.value, layout.value)
    expected = (HIDDEN, INTERMEDIATE, EXPERT_SPAN, POINTER_LAYOUT)
    if observed != expected:
        raise CudaMoeError(
            f"CUDA MoE shape/layout mismatch: found {observed}, need {expected}"
        )


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    with _load_lock:
        if _lib is not None or _load_error is not None:
            return _lib
        if not _CUDA_MOE_PLATFORM:
            _load_error = (
                "CUDA MoE native library is supported on Linux and Windows only"
            )
            return None
        if not os.path.isfile(_LIBRARY_PATH):
            _load_error = (
                f"CUDA MoE library not found: {_LIBRARY_PATH}; "
                "run this checkout's tools/build_native.py on a CUDA host"
            )
            return None
        try:
            candidate = ctypes.CDLL(_LIBRARY_PATH)
            _bind_and_handshake(candidate)
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            return None
        _lib = candidate
        return _lib


def _native_check(lib, status: int, operation: str) -> None:
    if int(status) != 0:
        raise CudaMoeError(
            f"{operation} failed with status {int(status)}: {_last_error(lib)}"
        )


def _canonical_cuda_device(device) -> tuple[Any, int]:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"CUDA MoE requires a CUDA device, got {dev}")
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        raise CudaMoeError(
            "CUDA MoE uses libcudart and cannot accept ROCm/HIP pointers"
        )
    if not getattr(version, "cuda", None):
        raise CudaMoeError(
            "this PyTorch build has no CUDA runtime (torch.version.cuda is unset)"
        )
    if not torch.cuda.is_available():
        raise CudaMoeError("PyTorch reports CUDA unavailable")
    count = int(torch.cuda.device_count())
    index = torch.cuda.current_device() if dev.index is None else int(dev.index)
    if index < 0 or index >= count:
        raise ValueError(
            f"CUDA device {index} is outside {count} visible device(s)"
        )
    return torch.device("cuda", index), index


def _parse_bytes(value: str, *, name: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    split = len(text)
    while split > 0 and text[split - 1].isalpha():
        split -= 1
    number, suffix = text[:split], text[split:]
    if suffix not in _BYTE_SUFFIXES:
        raise ValueError(
            f"{name} has unknown suffix {suffix!r}; use B, KiB, MiB, or GiB"
        )
    try:
        parsed = float(number)
    except ValueError as exc:
        raise ValueError(f"{name} must be a byte count") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return int(parsed * _BYTE_SUFFIXES[suffix])


def _cache_budget(
    free_bytes: int,
    total_bytes: int,
    *,
    cache_value: str | None = None,
    reserve_value: str | None = None,
) -> tuple[int, int]:
    """Return ``(capacity, reserve_bytes)`` clamped to current free VRAM."""
    free_bytes = max(0, int(free_bytes))
    total_bytes = max(0, int(total_bytes))
    reserve_text = (
        os.environ.get("K3_CUDA_EXPERT_CACHE_RESERVE", "auto")
        if reserve_value is None
        else str(reserve_value)
    ).strip().lower()
    if reserve_text in _AUTO:
        reserve = max(2 << 30, int(free_bytes * 0.20))
    elif reserve_text.endswith("%"):
        try:
            percent = float(reserve_text[:-1])
        except ValueError as exc:
            raise ValueError(
                "K3_CUDA_EXPERT_CACHE_RESERVE percent is invalid"
            ) from exc
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError(
                "K3_CUDA_EXPERT_CACHE_RESERVE percent must be in [0,100]"
            )
        reserve = int(free_bytes * percent / 100.0)
    else:
        reserve = _parse_bytes(
            reserve_text, name="K3_CUDA_EXPERT_CACHE_RESERVE"
        )
    safe_capacity = max(0, free_bytes - reserve) // EXPERT_SPAN

    cache_text = (
        os.environ.get("K3_CUDA_EXPERT_CACHE", "auto")
        if cache_value is None
        else str(cache_value)
    ).strip().lower()
    if cache_text in _AUTO:
        requested = safe_capacity
    else:
        try:
            requested = int(cache_text)
        except ValueError as exc:
            raise ValueError(
                "K3_CUDA_EXPERT_CACHE must be auto or a non-negative integer"
            ) from exc
        if requested < 0:
            raise ValueError(
                "K3_CUDA_EXPERT_CACHE must be auto or a non-negative integer"
            )
    return min(requested, safe_capacity), reserve


def _new_state(index: int) -> _DeviceState:
    strata_text = os.environ.get(
        "K3_CUDA_EXPERT_CACHE_LAYERS", str(DEFAULT_LAYER_STRATA)
    )
    try:
        strata = int(strata_text)
    except ValueError as exc:
        raise ValueError(
            "K3_CUDA_EXPERT_CACHE_LAYERS must be a positive integer"
        ) from exc
    if strata <= 0:
        raise ValueError(
            "K3_CUDA_EXPERT_CACHE_LAYERS must be a positive integer"
        )
    return _DeviceState(
        index,
        layer_strata=strata,
    )


def _ensure_cache_budget(state: _DeviceState, device) -> None:
    """Snapshot free VRAM at first routing use, after model qualification/load."""
    with state.lock:
        if state.budget_ready:
            return
        with torch.cuda.device(device):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(state.index)
            except TypeError:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
        capacity, reserve = _cache_budget(free_bytes, total_bytes)
        state.free_bytes = int(free_bytes)
        state.total_bytes = int(total_bytes)
        state.reserve_bytes = int(reserve)
        state.capacity = int(capacity)
        state.cache = _LayerStratifiedCache(
            capacity, state.layer_strata
        )
        state.budget_ready = True


def _state_for(index: int) -> _DeviceState:
    state = _states.get(index)
    if state is not None:
        return state
    with _states_lock:
        state = _states.get(index)
        if state is None:
            state = _new_state(index)
            _states[index] = state
    return state


def _ptr(tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(tensor.data_ptr()))


def _stream_ptr(stream) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(stream.cuda_stream))


def _record(stream, *tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


def _mxfp4_reference(packed: np.ndarray, scales: np.ndarray, x: np.ndarray):
    values = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6,
         -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    rows, half = packed.shape
    cols = half * 2
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    weights = values[codes]
    factors = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)
    weights *= np.repeat(factors, 32, axis=1)
    return x @ weights.T


def _self_test(lib, device, index: int) -> dict[str, Any]:
    """Exercise generic GEMV and flattened int8 dequant twice on one device."""
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        # Across the two rows, every E2M1 nibble code 0x0..0xf appears.
        packed_np = np.array(
            [
                [0x10, 0x32, 0x54, 0x76] * 4,
                [0x98, 0xBA, 0xDC, 0xFE] * 4,
            ],
            dtype=np.uint8,
        )
        scale_np = np.array([[127], [126]], dtype=np.uint8)
        x_np = np.stack(
            (
                np.arange(32, dtype=np.float32) / np.float32(32),
                (np.arange(32, dtype=np.float32) - 15) / np.float32(17),
            )
        )
        expected = torch.from_numpy(
            _mxfp4_reference(packed_np, scale_np, x_np)
        )
        packed = torch.from_numpy(packed_np).to(device)
        scales = torch.from_numpy(scale_np).to(device)
        x = torch.from_numpy(x_np).to(device)
        first = torch.empty((2, 2), dtype=torch.float32, device=device)
        second = torch.empty_like(first)
        for output in (first, second):
            status = lib.k3_cuda_mxfp4_gemv(
                _ptr(packed),
                _ptr(scales),
                _ptr(x),
                _ptr(output),
                2,
                32,
                2,
                _stream_ptr(stream),
            )
            _native_check(lib, status, "CUDA MXFP4 known-answer test")
            _record(stream, packed, scales, x, output)

        q = torch.tensor(
            [[-7, -1, 0, 4, 9], [3, -5, 12, 1, -2], [8, 7, -4, 2, 1]],
            dtype=torch.int8,
            device=device,
        )
        sc = torch.tensor(
            [0.5, 0.25, 2.0], dtype=torch.float16, device=device
        )
        dq_first = torch.empty((3, 5), dtype=torch.float32, device=device)
        dq_second = torch.empty_like(dq_first)
        for output in (dq_first, dq_second):
            status = lib.k3_cuda_int8_dequant(
                _ptr(q),
                _ptr(sc),
                _ptr(output),
                output.numel(),
                output.shape[1],
                _stream_ptr(stream),
            )
            _native_check(lib, status, "CUDA int8 known-answer test")
            _record(stream, q, sc, output)

        # Exercise the complete fixed-shape expert entry point, including all
        # six span offsets and SiTU.  Only one weight in each matrix is nonzero,
        # so its analytical result is small despite using the real ABI shape.
        expert_np = np.zeros(EXPERT_SPAN, dtype=np.uint8)
        expert_np[0] = 0x02                         # w1[0,0] = 1
        expert_np[PACKED_BYTES] = 127               # w1 row0/group0 scale
        expert_np[PACKED_BYTES + SCALE_BYTES] = 0x02  # w2[0,0] = 1
        expert_np[2 * PACKED_BYTES + SCALE_BYTES] = 127
        expert_np[2 * (PACKED_BYTES + SCALE_BYTES)] = 0x04  # w3[0,0] = 2
        expert_np[3 * PACKED_BYTES + 2 * SCALE_BYTES] = 127
        expert = torch.from_numpy(expert_np).to(device)
        moe_x = torch.zeros((2, HIDDEN), dtype=torch.float32, device=device)
        moe_x[:, 0] = torch.tensor([1.0, -0.5], device=device)
        moe_outputs = []
        moe_scratches = []
        for _ in range(2):
            gate = torch.empty(
                (2, INTERMEDIATE), dtype=torch.float32, device=device
            )
            up = torch.empty_like(gate)
            output = torch.empty(
                (2, HIDDEN), dtype=torch.float32, device=device
            )
            status = lib.k3_cuda_moe_launch(
                _ptr(expert),
                _ptr(moe_x),
                2,
                _ptr(gate),
                _ptr(up),
                _ptr(output),
                _stream_ptr(stream),
            )
            _native_check(lib, status, "CUDA full-MoE known-answer test")
            _record(stream, expert, moe_x, gate, up, output)
            moe_outputs.append(output)
            moe_scratches.extend((gate, up))

        torch.cuda.synchronize(device)
        gemv_repeat = bool(torch.equal(first, second))
        gemv_error = float(
            (first.cpu() - expected).abs().max().item()
        )
        dq_expected = q.cpu().float() * sc.cpu().float().reshape(-1, 1)
        dq_repeat = bool(torch.equal(dq_first, dq_second))
        dq_exact = bool(torch.equal(dq_first.cpu(), dq_expected))
        moe_repeat = bool(torch.equal(moe_outputs[0], moe_outputs[1]))
        moe_expected = torch.zeros((2, HIDDEN), dtype=torch.float32)
        for row, activation in enumerate((1.0, -0.5)):
            gate_value = activation
            up_value = 2.0 * activation
            moe_expected[row, 0] = (
                4.0
                * math.tanh(gate_value / 4.0)
                / (1.0 + math.exp(-gate_value))
                * (25.0 * math.tanh(up_value / 25.0))
            )
        moe_error = float(
            (moe_outputs[0].cpu() - moe_expected).abs().max().item()
        )
        if not gemv_repeat or gemv_error > 2e-4:
            raise CudaMoeError(
                "CUDA MXFP4 known-answer test failed "
                f"(repeat={gemv_repeat}, max_abs={gemv_error:.3e})"
            )
        if not dq_repeat or not dq_exact:
            raise CudaMoeError(
                "CUDA int8 known-answer test failed "
                f"(repeat={dq_repeat}, exact={dq_exact})"
            )
        if not moe_repeat or moe_error > 2e-5:
            raise CudaMoeError(
                "CUDA full-MoE known-answer test failed "
                f"(repeat={moe_repeat}, max_abs={moe_error:.3e})"
            )
        return {
            "device": index,
            "gemv_repeat_exact": gemv_repeat,
            "gemv_max_abs": gemv_error,
            "int8_repeat_exact": dq_repeat,
            "int8_analytical_exact": dq_exact,
            "moe_repeat_exact": moe_repeat,
            "moe_max_abs": moe_error,
        }


def available(device) -> bool:
    """Return true only after ABI, shape, availability, and on-device KATs."""
    try:
        dev, index = _canonical_cuda_device(device)
        state = _state_for(index)
    except Exception:
        return False
    with state.lock:
        if state.checked:
            return state.enabled
        state.checked = True
        lib = _load()
        if lib is None:
            state.reason = _load_error or "CUDA MoE library did not load"
            return False
        try:
            with torch.cuda.device(dev):
                status = int(lib.k3_cuda_moe_available(index))
                if status != 1:
                    raise CudaMoeError(
                        "native CUDA device probe rejected the device "
                        f"(status {status}): {_last_error(lib)}"
                    )
                state.self_test = _self_test(lib, dev, index)
        except Exception as exc:
            state.reason = f"{type(exc).__name__}: {exc}"
            state.enabled = False
            state.cache.clear()
            return False
        state.enabled = True
        state.reason = None
        return True


def disable(device, reason) -> None:
    """Permanently disable this backend for one CUDA device in this process."""
    try:
        _, index = _canonical_cuda_device(device)
    except Exception:
        return
    state = _state_for(index)
    with state.lock:
        state.checked = True
        state.enabled = False
        state.reason = str(reason)
        state.cache.clear()


def describe(device) -> str:
    try:
        dev, index = _canonical_cuda_device(device)
    except Exception as exc:
        return f"unavailable: {exc}"
    ok = available(dev)
    try:
        state = _state_for(index)
    except Exception as exc:
        return f"unavailable on {dev}: {type(exc).__name__}: {exc}"
    if not ok:
        return f"unavailable on {dev}: {state.reason or _load_error or 'unknown'}"
    if not state.budget_ready:
        return (
            f"CUDA MXFP4 on {dev}; expert cache budget pending first route, "
            "deterministic expert outputs and route-order combine"
        )
    return (
        f"CUDA MXFP4 on {dev}; expert cache {len(state.cache)}/"
        f"{state.capacity}, free-at-budget {state.free_bytes / 2**30:.1f} GiB, "
        f"reserve {state.reserve_bytes / 2**30:.1f} GiB, "
        "deterministic expert outputs and route-order combine"
    )


def stats(device) -> dict[str, Any]:
    dev, index = _canonical_cuda_device(device)
    state = _state_for(index)
    with state.lock:
        return {
            "device": str(dev),
            "enabled": state.enabled,
            "reason": state.reason,
            "budget_ready": state.budget_ready,
            "capacity": state.capacity,
            "entries": len(state.cache),
            "layer_counts": state.cache.layer_counts(),
            "hits": state.cache.hits,
            "misses": state.cache.misses,
            "evictions": state.cache.evictions,
            "admission_rejections": state.cache.rejections,
            "uploads": state.uploads,
            "calls": state.calls,
            "routes": state.routes,
            "free_bytes_at_budget": state.free_bytes,
            "reserve_bytes": state.reserve_bytes,
            "self_test": state.self_test,
        }


def _model_namespace(model_key) -> Hashable:
    value = _DEFAULT_MODEL_KEY if model_key is None else model_key
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("model_key must be hashable") from exc
    return value


def _layer_number(layer_index) -> int:
    if isinstance(layer_index, bool) or not isinstance(
        layer_index, numbers.Integral
    ):
        raise TypeError("layer_index must be a non-negative integer")
    value = int(layer_index)
    if value < 0:
        raise ValueError("layer_index must be a non-negative integer")
    return value


def _expert_ids(eids: Iterable[int]) -> list[int]:
    result, seen = [], set()
    for value in eids:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError("expert IDs must be non-negative integers")
        expert = int(value)
        if expert < 0:
            raise ValueError("expert IDs must be non-negative integers")
        if expert not in seen:
            seen.add(expert)
            result.append(expert)
    return result


def _cache_key(
    model_key: Hashable, device_index: int, layer_index: int, expert_id: int
) -> tuple[Hashable, int, int, int]:
    return model_key, int(device_index), int(layer_index), int(expert_id)


def missing_experts(
    layer_index,
    eids,
    device,
    model_key=None,
) -> list[int]:
    """Return uncached IDs in first-occurrence order; never upload or mutate."""
    dev, index = _canonical_cuda_device(device)
    layer = _layer_number(layer_index)
    model = _model_namespace(model_key)
    experts = _expert_ids(eids)
    state = _state_for(index)
    _ensure_cache_budget(state, dev)
    with state.lock:
        return [
            expert
            for expert in experts
            if state.cache.peek(_cache_key(model, index, layer, expert)) is None
        ]


_RAW_LAYOUT = (
    ("w1", 0, 0, PACKED_BYTES, (3072, 1792)),
    ("w1", 1, PACKED_BYTES, SCALE_BYTES, (3072, 112)),
    ("w2", 0, PACKED_BYTES + SCALE_BYTES, PACKED_BYTES, (3584, 1536)),
    ("w2", 1, 2 * PACKED_BYTES + SCALE_BYTES, SCALE_BYTES, (3584, 96)),
    ("w3", 0, 2 * (PACKED_BYTES + SCALE_BYTES), PACKED_BYTES, (3072, 1792)),
    ("w3", 1, 3 * PACKED_BYTES + 2 * SCALE_BYTES, SCALE_BYTES, (3072, 112)),
)


def _validated_raw_parts(raw) -> list[tuple[int, int, np.ndarray]]:
    if not isinstance(raw, Mapping) or set(raw) != {"w1", "w2", "w3"}:
        raise ValueError("raw expert must contain exactly w1, w2, and w3")
    result = []
    for weight, part, offset, nbytes, shape in _RAW_LAYOUT:
        pair = raw[weight]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"raw expert {weight} must be (packed, scale)")
        array = pair[part]
        if not isinstance(array, np.ndarray):
            raise TypeError(f"raw expert {weight} arrays must be numpy arrays")
        if array.dtype != np.uint8 or tuple(array.shape) != shape:
            raise ValueError(
                f"raw expert {weight} part {part} must be uint8 {shape}, "
                f"got {array.dtype} {tuple(array.shape)}"
            )
        if not array.flags.c_contiguous or array.nbytes != nbytes:
            raise ValueError(
                f"raw expert {weight} part {part} must be C-contiguous "
                f"and {nbytes} bytes"
            )
        result.append((offset, nbytes, array))
    return result


def _upload_expert(state: _DeviceState, device, raw, stream):
    parts = _validated_raw_parts(raw)
    stage, stage_np = state.stage.prepare()
    for offset, nbytes, array in parts:
        np.copyto(
            stage_np[offset:offset + nbytes],
            array.reshape(-1),
            casting="no",
        )
    device_span = torch.empty(
        EXPERT_SPAN, dtype=torch.uint8, device=device
    )
    # Exactly one host-to-device operation carries the complete expert span.
    device_span.copy_(stage, non_blocking=True)
    ready_event = state.stage.copied_on(stream)
    # The pinned host source is protected by _PinnedStage's explicit event;
    # record_stream is a CUDA-tensor allocator contract and does not apply to
    # ordinary/pinned CPU tensors.
    _record(stream, device_span)
    state.uploads += 1
    return _ResidentExpert(device_span, ready_event)


def _validate_x(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "cuda":
        raise ValueError("CUDA MoE x must reside on CUDA")
    if x.dtype != torch.float32:
        raise ValueError("CUDA MoE x must be float32")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != HIDDEN:
        raise ValueError(f"CUDA MoE x must have shape [N,{HIDDEN}]")
    if not x.is_contiguous():
        raise ValueError("CUDA MoE x must be contiguous")


def _nested_rows(value, name: str) -> list[list[Any]]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        return value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a two-dimensional sequence")
    rows = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} rows must be sequences")
        rows.append(list(row))
    return rows


def _validate_routes(
    token_count: int,
    topk_ids,
    topk_weight,
    routing_record=None,
) -> tuple[list[list[int]], list[list[float]]]:
    if routing_record is None:
        ids = _nested_rows(topk_ids, "topk_ids")
        weights = _nested_rows(topk_weight, "topk_weight")
    else:
        if not isinstance(routing_record, Mapping):
            raise TypeError("routing_record must be a mapping")
        if "ids" not in routing_record or "weights" not in routing_record:
            raise ValueError("routing_record must contain ids and weights")
        ids = _nested_rows(routing_record["ids"], "routing_record ids")
        weights = _nested_rows(
            routing_record["weights"], "routing_record weights"
        )
    if len(ids) != token_count or len(weights) != token_count:
        raise ValueError("route row count must equal x token count")
    normalized_ids, normalized_weights = [], []
    route_width = None
    for row_index, (id_row, weight_row) in enumerate(zip(ids, weights)):
        if not id_row or len(id_row) != len(weight_row):
            raise ValueError(
                f"route {row_index} must have equal, non-empty ID/weight rows"
            )
        if route_width is None:
            route_width = len(id_row)
        elif len(id_row) != route_width:
            raise ValueError("every route row must have the same width")
        out_ids, out_weights = [], []
        for expert, weight in zip(id_row, weight_row):
            if isinstance(expert, bool) or not isinstance(
                expert, numbers.Integral
            ):
                raise TypeError("route expert IDs must be integers")
            expert = int(expert)
            if expert < 0:
                raise ValueError("route expert IDs must be non-negative")
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                raise TypeError("route weights must be real numbers")
            weight = float(weight)
            if not math.isfinite(weight):
                raise ValueError("route weights must be finite")
            out_ids.append(expert)
            out_weights.append(weight)
        normalized_ids.append(out_ids)
        normalized_weights.append(out_weights)
    return normalized_ids, normalized_weights


def moe_infer(
    x,
    topk_ids,
    topk_weight,
    raw_experts,
    routing_record=None,
    *,
    layer_index,
    model_key=None,
):
    """Run routed MXFP4 experts on CUDA and return a fresh fp32 CUDA tensor.

    ``raw_experts`` may contain only the IDs reported by ``missing_experts``;
    already resident experts are resolved by the complete cache identity.
    """
    _validate_x(x)
    device, index = _canonical_cuda_device(x.device)
    layer = _layer_number(layer_index)
    model = _model_namespace(model_key)
    ids, weights = _validate_routes(
        x.shape[0], topk_ids, topk_weight, routing_record
    )
    unique = _expert_ids(expert for row in ids for expert in row)
    if not isinstance(raw_experts, Mapping):
        raise TypeError("raw_experts must be a mapping")
    if not available(device):
        state = _state_for(index)
        raise CudaMoeError(state.reason or "CUDA MoE backend unavailable")
    lib = _load()
    if lib is None:  # defensive: available already proved otherwise
        raise CudaMoeError(_load_error or "CUDA MoE library unavailable")
    state = _state_for(index)
    _ensure_cache_budget(state, device)

    with state.lock, torch.cuda.device(device):
        if not state.enabled:
            raise CudaMoeError(state.reason or "CUDA MoE backend disabled")
        stream = torch.cuda.current_stream(device)
        stream_pointer = _stream_ptr(stream)
        cache_keys = {
            expert: _cache_key(model, index, layer, expert)
            for expert in unique
        }
        # Hold strong references to all initial hits before any miss admission
        # can evict one within the same layer quota.
        resident = {
            expert: value
            for expert in unique
            if (value := state.cache.get(cache_keys[expert])) is not None
        }
        missing = [expert for expert in unique if expert not in resident]
        absent_raw = [expert for expert in missing if expert not in raw_experts]
        if absent_raw:
            raise KeyError(
                "raw_experts is missing uncached expert IDs "
                + ", ".join(map(str, absent_raw))
            )

        route_count = sum(len(row) for row in ids)
        expert_outputs = torch.empty(
            (route_count, HIDDEN), dtype=torch.float32, device=device
        )
        flat_routes: dict[int, list[tuple[int, int]]] = {
            expert: [] for expert in unique
        }
        flat_index = 0
        for token, row in enumerate(ids):
            for expert in row:
                flat_routes[expert].append((token, flat_index))
                flat_index += 1

        try:
            for expert in unique:
                resident_expert = resident.get(expert)
                if resident_expert is None:
                    resident_expert = _upload_expert(
                        state, device, raw_experts[expert], stream
                    )
                    admitted, _ = state.cache.put(
                        cache_keys[expert], resident_expert
                    )
                    # Whether admitted or deliberately rejected by a zero quota,
                    # this local reference remains alive through its launch.
                    del admitted
                # A cache hit may have been uploaded on another PyTorch stream.
                # record_stream protects allocator reuse, but only this event
                # wait establishes data readiness across streams.
                stream.wait_event(resident_expert.ready_event)
                expert_span = resident_expert.tensor
                positions = flat_routes[expert]
                token_indices = torch.tensor(
                    [token for token, _ in positions],
                    dtype=torch.long,
                    device=device,
                )
                route_indices = torch.tensor(
                    [route for _, route in positions],
                    dtype=torch.long,
                    device=device,
                )
                expert_x = x.index_select(0, token_indices).contiguous()
                count = len(positions)
                gate = torch.empty(
                    (count, INTERMEDIATE),
                    dtype=torch.float32,
                    device=device,
                )
                up = torch.empty_like(gate)
                batch_output = torch.empty(
                    (count, HIDDEN), dtype=torch.float32, device=device
                )
                status = lib.k3_cuda_moe_launch(
                    _ptr(expert_span),
                    _ptr(expert_x),
                    count,
                    _ptr(gate),
                    _ptr(up),
                    _ptr(batch_output),
                    stream_pointer,
                )
                _native_check(lib, status, "CUDA MXFP4 expert launch")
                # ctypes launches are invisible to PyTorch's allocator.  Record
                # every native source/scratch/destination before any reference
                # can be dropped or cache entry evicted.
                _record(
                    stream,
                    expert_span,
                    x,
                    token_indices,
                    route_indices,
                    expert_x,
                    gate,
                    up,
                    batch_output,
                    expert_outputs,
                )
                expert_outputs.index_copy_(
                    0, route_indices, batch_output
                )

            # Preserve route-slot order with K launches rather than N*K scalar
            # add_ launches. addcmul_ covers every token for one slot at once;
            # the next slot is enqueued afterward on the same explicit stream.
            # No scheduler-dependent atomic operation combines expert results.
            output = torch.zeros_like(x)
            route_width = len(ids[0])
            by_slot = expert_outputs.view(
                x.shape[0], route_width, HIDDEN
            )
            route_weights = torch.tensor(
                weights, dtype=torch.float32, device=device
            ).unsqueeze(-1)
            for slot in range(route_width):
                output.addcmul_(
                    by_slot[:, slot, :],
                    route_weights[:, slot, :],
                )
            _record(
                stream, x, expert_outputs, by_slot, route_weights, output
            )
            state.calls += 1
            state.routes += route_count
            return output
        except Exception as exc:
            # Native/runtime failures are not cached as an innocuous
            # ineligibility: poison this per-device backend and let the caller
            # take its explicit CPU fallback.
            if isinstance(exc, (CudaMoeError, torch.cuda.OutOfMemoryError)):
                state.enabled = False
                state.reason = f"{type(exc).__name__}: {exc}"
                state.cache.clear()
            raise


def _storage_overlaps(*tensors) -> bool:
    ranges = []
    for tensor in tensors:
        begin = int(tensor.data_ptr())
        ranges.append(
            (begin, begin + tensor.numel() * tensor.element_size())
        )
    for left in range(len(ranges)):
        for right in range(left + 1, len(ranges)):
            a0, a1 = ranges[left]
            b0, b1 = ranges[right]
            if max(a0, b0) < min(a1, b1):
                return True
    return False


def _dequant_layout_eligible(dst, q, sc) -> bool:
    if not all(isinstance(value, torch.Tensor) for value in (dst, q, sc)):
        return False
    if dst.device.type != "cuda":
        return False
    if q.device != dst.device or sc.device != dst.device:
        return False
    if dst.dtype != torch.float32 or dst.ndim != 2 or dst.numel() <= 0:
        return False
    if q.dtype != torch.int8 or q.numel() != dst.numel():
        return False
    if sc.dtype != torch.float16 or sc.numel() != dst.shape[0]:
        return False
    if not (dst.is_contiguous() and q.is_contiguous() and sc.is_contiguous()):
        return False

    return not _storage_overlaps(dst, q, sc)


def dequant_int8_into(dst, q, sc) -> bool:
    """Dequantize row-int8 into fp32 ``dst`` on CUDA.

    ``False`` means permanent layout/dtype/device ineligibility only.  Once an
    eligible call reaches the backend, capability or native failures raise so a
    caller never permanently memoizes a transient launch failure as unsupported.
    """
    if not _dequant_layout_eligible(dst, q, sc):
        return False
    device, index = _canonical_cuda_device(dst.device)
    if not available(device):
        state = _state_for(index)
        raise CudaMoeError(state.reason or "CUDA int8 backend unavailable")
    lib = _load()
    state = _state_for(index)
    with state.lock, torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        try:
            status = lib.k3_cuda_int8_dequant(
                _ptr(q),
                _ptr(sc),
                _ptr(dst),
                dst.numel(),
                dst.shape[1],
                _stream_ptr(stream),
            )
            _native_check(lib, status, "CUDA int8 dequant")
            _record(stream, q, sc, dst)
            return True
        except Exception:
            # Dequantization is an optional, independent use of the library.
            # Its caller can retain torch dequantization without invalidating
            # correctly resident MoE experts or disabling native MoE compute.
            raise


__all__ = [
    "CudaMoeError",
    "available",
    "dequant_int8_into",
    "describe",
    "disable",
    "missing_experts",
    "moe_infer",
    "stats",
]
