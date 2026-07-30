"""Fast resident-spine load path (K3_FAST_SPINE=1).

Replaces the per-tensor read -> bytearray -> .to(DEV) -> fp32 dequant -> copy_
chain in kimi_run.py with three changes, each measured on this machine:

  1. PACKED READS.  A layer's ~28 .i8 payloads (and its ~28 .sc scale blobs) are
     read with readinto() straight into ONE reusable host buffer, optionally by a
     few threads.  Measured on the real spine, 634 MB layer, cold-ish:
         read()+bytearray()   4.07 GB/s      (current path: kernel copy + a full
                                              second memcpy for the bytearray)
         readinto pooled x1   5.57 GB/s
         readinto pooled x4   6.94 GB/s
         readinto pooled x8   7.18 GB/s
     The buffer pool matters: bytearray(n) zero-fills, so allocating a fresh
     630 MB per layer would memset 57 GB/token.

  2. ONE host->device transfer per layer instead of ~28 (2,604/token today).
         28 separate .to(DEV)   40.2 GB/s
         1 coalesced .to(DEV)   49.7 GB/s

  3. A CUSTOM METAL DEQUANT KERNEL that fuses int8->fp32 + row-scale multiply +
     the copy into the template buffer into one pass.  torch's row-broadcast
     multiply on MPS is the real bottleneck in the current path -- it runs at
     ~100 GB/s where a plain copy_ of the same traffic runs at 334 GB/s:
         (q.to(f32) * sc.to(f32)) then copy_   10.08 ms   43.7 GB/s   [current]
         torch.mul(q, sc_f32, out=dst)          7.34 ms   60.0 GB/s
         metal deq_f32                          1.48 ms  297.0 GB/s
     (6144x14336 int8 -> fp32.)  The kernel is BIT-EXACT against the current
     expression -- max|diff| = 0.0 on every shape tested, including from a
     non-zero storage offset inside the packed buffer -- because both compute
     float(int8) * float(fp16) in fp32.

Everything here is inert unless K3_FAST_SPINE=1.  Any failure (no MPS, shader
compile error) falls back to the torch expression automatically.
"""
import os
import threading
import time as _time
import concurrent.futures as _cf
import types

import torch

import spine_io
import runtime_platform

# ---------------------------------------------------------------- flags -----
FAST = os.environ.get("K3_FAST_SPINE", "1") == "1"
# fine-grained overrides so the orchestrator can bisect the three changes
_DEQ_EXPLICIT = "K3_SPINE_DEQ" in os.environ
def _default_deq():
    """Pick the native dequant backend for this platform."""
    if not FAST:
        return "torch"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "metal"


DEQ = os.environ.get(
    "K3_SPINE_DEQ", _default_deq()
)  # metal|cuda|mulout|torch
PACK = os.environ.get("K3_SPINE_PACK", "1" if FAST else "0") == "1"  # packed read + 1 H2D
READ_THREADS = runtime_platform.configured_cpu_workers(
    "K3_SPINE_READ_THREADS", 4 if FAST else 1
)
ALIGN = 256          # byte alignment of every slot inside the packed buffer

spine_io.set_process_io_policy()


# --------------------------------------------------------- metal dequant ----
_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// out[i] = float(q[i]) * float(sc[i / cols])   -- identical arithmetic to
// (q.to(float32) * sc.to(float32)) in torch, hence bit-exact.
kernel void deq_f32(device float*       out  [[buffer(0)]],
                    device const char*  q    [[buffer(1)]],
                    device const half*  sc   [[buffer(2)]],
                    constant uint&      cols [[buffer(3)]],
                    uint idx [[thread_position_in_grid]])
{
    out[idx] = float(q[idx]) * float(sc[idx / cols]);
}
"""

_lib = None
_lib_err = None
_lib_lock = threading.Lock()
_cuda_deq_module = None
_cuda_deq_devices = set()
_cuda_deq_errors = {}
_cuda_deq_lock = threading.Lock()


def _cuda_dequant_requested_for_config(
    fast, deq, deq_explicit, cuda_mode, cuda_explicit
):
    enabled = cuda_mode not in {
        "0", "off", "false", "no", "disabled",
    }
    if not enabled:
        return False
    if deq_explicit:
        return deq == "cuda"
    return fast or cuda_explicit


_CUDA_DEQ_EXPLICIT = "K3_CUDA_INT8_DEQUANT" in os.environ
_cuda_deq_mode = os.environ.get(
    "K3_CUDA_INT8_DEQUANT", "auto"
).strip().lower()
_cuda_deq_requested = _cuda_dequant_requested_for_config(
    FAST,
    DEQ,
    _DEQ_EXPLICIT,
    _cuda_deq_mode,
    _CUDA_DEQ_EXPLICIT,
)


def metal_available():
    """Compile the dequant shader once; False (with a reason in metal_error())
    if this build/machine cannot."""
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
            except Exception as e:                     # pragma: no cover
                _lib_err = f"{type(e).__name__}: {e}"
    return _lib is not None


def metal_error():
    return _lib_err


def dequant_into(dst, q, sc):
    """dst[fp32, (rows, cols), on MPS] <- float(q[int8]) * float(sc[fp16, rows]).

    q/sc may be views at arbitrary storage offsets inside a packed buffer.
    Returns True if the metal kernel ran, False if the caller should fall back."""
    if not metal_available():
        return False
    rows, cols = dst.shape
    _lib.deq_f32(dst, q, sc, cols, threads=rows * cols)
    return True


def cuda_dequant_into(dst, q, sc):
    """Try the qualified CUDA int8 dequantizer, otherwise retain torch.

    Import stays inside the CUDA-only call site so a macOS/MPS or CPU process
    neither loads nor probes the optional CUDA library.  Any native or
    qualification error fails closed for the remainder of the process.
    """
    global _cuda_deq_module
    if not _cuda_deq_requested:
        return False
    if dst.device.type != "cuda":
        return False
    device_key = (dst.device.type, dst.device.index)
    if device_key in _cuda_deq_errors:
        return False
    with _cuda_deq_lock:
        if device_key not in _cuda_deq_devices:
            try:
                if _cuda_deq_module is None:
                    import cuda_moe
                    _cuda_deq_module = cuda_moe
                if not _cuda_deq_module.available(dst.device):
                    raise RuntimeError(_cuda_deq_module.describe(dst.device))
                _cuda_deq_devices.add(device_key)
            except Exception as exc:
                _cuda_deq_errors[device_key] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return False
    try:
        if _cuda_deq_module.dequant_int8_into(dst, q, sc):
            return True
        _cuda_deq_errors[device_key] = (
            "native CUDA int8 dequantizer rejected the tensor layout"
        )
        return False
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _cuda_deq_errors[device_key] = error
        return False


def cuda_dequant_error(device=None):
    if device is None:
        return dict(_cuda_deq_errors)
    return _cuda_deq_errors.get((device.type, device.index))


def dequant_torch(q, sc, out=None):
    """Fallback / baseline dequant. `mulout` fuses the multiply with the write
    into the destination (7.34 ms vs 10.08 ms on 6144x14336); `torch` is the
    exact expression kimi_run.py uses today."""
    if out is not None and DEQ == "mulout":
        torch.mul(q, sc.to(torch.float32), out=out)
        return out
    return q.to(torch.float32) * sc.to(torch.float32)


# ----------------------------------------------------------- buffer pool ----
# At most two packs are live at once (the preloader thread fills N+1 while the
# main thread applies N), so this settles at 2 buffers and never reallocates.
_pool = []
_pool_lock = threading.Lock()
_POOL_MAX = 8      # 3 buffers per pack (int8 / scales / bf16) x 2 live packs


def _acquire(nbytes):
    """Best fit, so a 634 MB payload buffer is never handed to a 2 MB scale
    request (which would force a fresh bytearray -- and bytearray(n) zero-fills,
    i.e. 57 GB of memset per token if it happened every layer)."""
    with _pool_lock:
        best = -1
        for i, b in enumerate(_pool):
            if len(b) >= nbytes and (best < 0 or len(b) < len(_pool[best])):
                best = i
        if best >= 0:
            return _pool.pop(best)
    return bytearray(max(nbytes, 1))


# Buffers a caller has taken permanent ownership of (the driver's spine RAM
# cache). They must never re-enter the pool, or the next layer's readinto would
# overwrite cached weights in place — a silent wrong-weights bug, not a slow path.
KEEP = set()


def pin(*bufs):
    """Take ownership of these buffers: the pool will never recycle them."""
    for b in bufs:
        if b is not None:
            KEEP.add(id(b))


def unpin(*bufs):
    """Hand ownership back.  ONLY call this after the pack has been dropped from
    whatever cache pinned it -- an unpinned buffer can be recycled by the next
    layer's read, which would silently rewrite it in place."""
    for b in bufs:
        if b is not None:
            KEEP.discard(id(b))


def _release(buf):
    if buf is None or id(buf) in KEEP:
        return
    with _pool_lock:
        if len(_pool) < _POOL_MAX:
            _pool.append(buf)


# Persistent device staging buffers: without these every layer would allocate
# and free a fresh ~634 MB MPS buffer for the packed int8 payload.
_stage = {}
_stage_generations = {}
_stage_fences = {}


class _StageLease:
    """Identity/generation proof for a direct view into a recycled upload buffer."""

    __slots__ = (
        "key",
        "generation",
        "data_ptr",
        "sync_contract",
        "stream_signature",
    )

    def __init__(
        self,
        key,
        generation,
        data_ptr,
        sync_contract="event",
        stream_signature=None,
    ):
        self.key = key
        self.generation = generation
        self.data_ptr = data_ptr
        self.sync_contract = sync_contract
        self.stream_signature = stream_signature


class _FailedStageFence:
    """Poison a staging buffer whose prior GPU use could not be ordered."""

    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


class _FifoStageFence:
    """Proof that prior use and the next overwrite share one FIFO stream."""

    __slots__ = ("stream_signature", "controller")

    def __init__(self, stream_signature, controller):
        self.stream_signature = stream_signature
        self.controller = controller


def _stage_key(dev, dtype):
    index = dev.index
    if dev.type == "cuda" and index is None:
        # ``torch.device("cuda")`` follows the mutable process current device.
        # Resolve it before indexing global staging state so two GPUs can never
        # alias one upload buffer/fence/generation record.
        index = torch.cuda.current_device()
    return (dev.type, index, dtype)


def _normalize_stage_sync_mode(mode):
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in {"event", "mps_fifo"}:
        raise ValueError(
            "packed KDA stage sync mode must be event or mps_fifo"
        )
    return normalized


def _effective_stage_sync_mode(dev, mode):
    """MPS may opt into FIFO reuse; every other async backend keeps events."""
    normalized = _normalize_stage_sync_mode(mode)
    if normalized == "mps_fifo" and dev.type == "mps":
        return normalized
    return "event"


def _stage_stream_signature(dev):
    """Return the public current-stream identity without creating a stream."""
    accelerator = getattr(torch, "accelerator", None)
    current_stream = getattr(accelerator, "current_stream", None)
    if not callable(current_stream):
        raise RuntimeError(
            "torch.accelerator.current_stream is unavailable"
        )
    stream = current_stream(dev)
    stream_id = getattr(stream, "stream_id", None)
    device_index = getattr(stream, "device_index", None)
    if type(stream_id) is not int or type(device_index) is not int:
        raise RuntimeError("current stream has no stable public identity")
    return (dev.type, device_index, stream_id)


def _mps_fifo_stream_signature(dev):
    signature = _stage_stream_signature(dev)
    if signature[0] != "mps" or signature[2] != 0:
        raise RuntimeError(
            "MPS FIFO staging requires the default stream (stream_id=0)"
        )
    return signature


def stage_storage_capability(dev, sync_mode="event"):
    """Return whether direct staged weights can be reused without a race."""
    try:
        effective_sync = _effective_stage_sync_mode(dev, sync_mode)
    except ValueError as exc:
        return False, str(exc)
    if dev.type == "cpu":
        return True, None
    if dev.type == "mps":
        if effective_sync == "mps_fifo":
            if not callable(getattr(getattr(torch, "mps", None),
                                    "synchronize", None)):
                return False, "torch.mps.synchronize is unavailable"
            try:
                _mps_fifo_stream_signature(dev)
            except Exception as exc:
                return False, str(exc)
            return True, None
        event = getattr(getattr(torch, "mps", None), "Event", None)
        if event is None:
            return False, "torch.mps.Event is unavailable"
        if not callable(getattr(event, "record", None)):
            return False, "torch.mps.Event.record is unavailable"
        if not callable(getattr(event, "wait", None)):
            return False, "torch.mps.Event.wait is unavailable"
        return True, None
    if dev.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        if (
            cuda is None
            or not callable(getattr(cuda, "Event", None))
            or not callable(getattr(cuda, "current_stream", None))
        ):
            return False, "CUDA event/stream ordering primitives are unavailable"
        return True, None
    return False, f"direct staged weights do not support device type {dev.type}"


def _wait_stage_fence(key, dev):
    pending = _stage_fences.pop(key, None)
    if pending is None:
        return
    if isinstance(pending, _FailedStageFence):
        _stage_fences[key] = pending
        raise RuntimeError(
            "packed KDA staging buffer is poisoned after a fence failure"
        ) from pending.error
    if isinstance(pending, _FifoStageFence):
        controller = pending.controller
        try:
            current = _mps_fifo_stream_signature(dev)
            if current != pending.stream_signature:
                raise RuntimeError(
                    "MPS staging stream identity changed before reuse"
                )
            controller.stage_fifo_reuses += 1
            return
        except Exception as stream_error:
            # A stream mismatch invalidates the cheap FIFO proof. A device-wide
            # barrier is conservative but still makes this one reuse safe.
            try:
                torch.mps.synchronize()
                controller.stage_fence_sync_fallbacks += 1
                return
            except Exception as sync_error:
                controller.stage_fence_failures += 1
                poisoned = RuntimeError(
                    "cannot order MPS FIFO staging-buffer reuse"
                )
                poisoned.add_note(
                    f"stream contract failed: {stream_error}"
                )
                poisoned.add_note(
                    f"device synchronize failed: {sync_error}"
                )
                _stage_fences[key] = _FailedStageFence(poisoned)
                controller.state.disable(poisoned)
                raise poisoned from sync_error
    event, controller = pending
    try:
        if dev.type == "mps":
            event.wait()
        elif dev.type == "cuda":
            torch.cuda.current_stream(dev).wait_event(event)
        else:
            raise RuntimeError(
                f"unexpected asynchronous staging device {dev.type}"
            )
        controller.stage_fence_waits += 1
    except Exception as wait_error:
        # Blocking on the recorded event is slower but still proves that reuse
        # is safe. If even that fails, preserve a permanent poison marker so no
        # future upload can silently overwrite live GPU weights.
        try:
            event.synchronize()
            controller.stage_fence_sync_fallbacks += 1
        except Exception as sync_error:
            controller.stage_fence_failures += 1
            poisoned = RuntimeError(
                "cannot order packed KDA staging-buffer reuse"
            )
            poisoned.add_note(f"event wait failed: {wait_error}")
            poisoned.add_note(f"event synchronize failed: {sync_error}")
            _stage_fences[key] = _FailedStageFence(poisoned)
            controller.state.disable(poisoned)
            raise poisoned from sync_error


def _record_stage_fence(lease, dev, controller):
    if dev.type == "cpu":
        return
    if lease.sync_contract == "mps_fifo":
        try:
            current = _mps_fifo_stream_signature(dev)
            if current != lease.stream_signature:
                raise RuntimeError(
                    "MPS staging stream identity changed after use"
                )
            _stage_fences[lease.key] = _FifoStageFence(
                current, controller
            )
            controller.stage_fifo_records += 1
            return
        except Exception as stream_error:
            # The operator was admitted only on the lease's default stream.
            # If that contract changed before release, finish all MPS work now
            # rather than leaving an unprovable marker for the next overwrite.
            try:
                torch.mps.synchronize()
                controller.stage_fence_sync_fallbacks += 1
                return
            except Exception as sync_error:
                controller.stage_fence_failures += 1
                poisoned = RuntimeError(
                    "cannot fence MPS FIFO staging-buffer lifetime"
                )
                poisoned.add_note(
                    f"stream contract failed: {stream_error}"
                )
                poisoned.add_note(
                    f"device synchronize failed: {sync_error}"
                )
                _stage_fences[lease.key] = _FailedStageFence(poisoned)
                controller.state.disable(poisoned)
                raise poisoned from sync_error
    try:
        if dev.type == "mps":
            event = torch.mps.Event()
            event.record()
        elif dev.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(dev))
        else:
            raise RuntimeError(
                f"unexpected asynchronous staging device {dev.type}"
            )
        _stage_fences[lease.key] = (event, controller)
        controller.stage_fence_records += 1
    except Exception as record_error:
        # A device-wide synchronization is conservative, but it keeps the
        # experimental path correct on runtimes whose event implementation is
        # present yet broken. Failure of both mechanisms poisons the buffer.
        try:
            if dev.type == "mps":
                torch.mps.synchronize()
            elif dev.type == "cuda":
                torch.cuda.synchronize(dev)
            else:
                raise
            controller.stage_fence_sync_fallbacks += 1
        except Exception as sync_error:
            controller.stage_fence_failures += 1
            poisoned = RuntimeError(
                "cannot fence packed KDA staging-buffer lifetime"
            )
            poisoned.add_note(f"event record failed: {record_error}")
            poisoned.add_note(f"device synchronize failed: {sync_error}")
            _stage_fences[lease.key] = _FailedStageFence(poisoned)
            controller.state.disable(poisoned)
            raise poisoned from sync_error


def _stage_buf(n, dev, dtype):
    key = _stage_key(dev, dtype)
    b = _stage.get(key)
    if b is None or b.numel() < n:
        b = torch.empty(n, dtype=dtype, device=dev)
        _stage[key] = b
    return b[:n]


_CPU_DIRECT_ALL = (True, True, True)
_CPU_DIRECT_SCALE_OTHER = (False, True, True)
_CPU_DIRECT_NONE = (False, False, False)


def _cpu_direct_view_flags(dev, packed_qkv=None):
    """Return direct-view eligibility for (q, scales, bf16 tail).

    CPU kernels consume ordinary dense/arena inputs synchronously, so the
    completed read buffers are already their final packed source and a second
    CPU staging copy is redundant.  The opt-in packed ``stage`` controller is
    the one exception: it retains q beyond ``apply_pack`` and therefore keeps
    an owned q staging copy.  It copies scales into its arena during apply and
    never retains the bf16 tail, so those two inputs remain direct.

    Every accelerator or unknown device type fails closed to the established
    staging path.  PyTorch exposes ROCm devices through ``type == "cuda"``.
    """
    if getattr(dev, "type", None) != "cpu":
        return _CPU_DIRECT_NONE
    if (
        packed_qkv is not None
        and getattr(packed_qkv, "storage_mode", None) == "stage"
    ):
        return _CPU_DIRECT_SCALE_OTHER
    return _CPU_DIRECT_ALL


def _stage_weight_buf(n, dev, sync_mode="event"):
    """Acquire the int8 upload buffer only after its prior direct use is safe."""
    key = _stage_key(dev, torch.int8)
    _wait_stage_fence(key, dev)
    b = _stage.get(key)
    if b is None or b.numel() < n:
        b = torch.empty(n, dtype=torch.int8, device=dev)
        _stage[key] = b
    generation = _stage_generations.get(key, 0) + 1
    _stage_generations[key] = generation
    sync_contract = _effective_stage_sync_mode(dev, sync_mode)
    stream_signature = (
        _mps_fifo_stream_signature(dev)
        if sync_contract == "mps_fifo" else None
    )
    lease = _StageLease(
        key,
        generation,
        b.data_ptr(),
        sync_contract,
        stream_signature,
    )
    return b[:n], lease


def _stage_lease_is_current(lease):
    if lease is None:
        return False
    current = _stage.get(lease.key)
    return bool(
        current is not None
        and _stage_generations.get(lease.key) == lease.generation
        and current.data_ptr() == lease.data_ptr
    )


# ----------------------------------------- dynamic packed-q8 KDA projections ---
# Every role below consumes the same, unchanged ``hidden_states`` tensor inside
# KimiDeltaAttention.forward.  The second-stage f_b/g_b projections and o_proj
# deliberately stay dense because their inputs differ.
_CORE_PROJECTION_SPECS = (
    ("q", "q_proj"),
    ("k", "k_proj"),
    ("v", "v_proj"),
)
_OPTIONAL_PROJECTION_SPECS = (
    ("f_a", "f_a_proj"),
    ("b", "b_proj"),
)


class DynamicQ8State:
    """Shared kill switch for a packed KDA projection group.

    A backend failure disables every controller in the group process-wide. The
    failing controller first reconstructs its current dense weights, while
    subsequent layers take the ordinary dequant/copy branch in ``apply_pack``.
    """

    def __init__(self, on_disable=None):
        self.enabled = True
        self.reason = None
        self._on_disable = on_disable

    def disable(self, reason):
        if not self.enabled:
            return
        self.enabled = False
        self.reason = f"{type(reason).__name__}: {reason}"
        if self._on_disable is not None:
            self._on_disable(self.reason)


class DynamicPackedQ8QKV:
    """Capability-gated row-int8 storage for KDA projections.

    The default ``arena`` mode copies each current layer out of its recycled
    upload buffer into template-owned storage. Experimental ``stage`` mode
    instead consumes an ordered, generation-checked view of that upload buffer
    directly and fences asynchronous reuse at the end of attention. Dense
    parameters remain available as a cold fallback in either mode.
    """

    _QBUF = "_k3_q8_qkv_weight_arena"
    _SBUF = "_k3_q8_qkv_scale_arena"
    _CTRL = "_k3_q8_qkv_controller"

    def __init__(
        self,
        module,
        dev,
        state,
        backend,
        arenas=None,
        fuse_qkv=None,
        storage_mode="arena",
        stage_sync_mode="event",
    ):
        dev = torch.device(dev)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        attention = getattr(module, "self_attn", None)
        if attention is None:
            raise ValueError("dynamic q8 KDA bundle requires module.self_attn")
        if getattr(attention, self._CTRL, None) is not None:
            raise ValueError("dynamic q8 KDA bundle is already installed")
        if hasattr(attention, self._QBUF) or hasattr(attention, self._SBUF):
            raise ValueError("dynamic q8 KDA arena names are already in use")
        backend_matmul = getattr(backend, "matmul", None)
        resolved_backend = (
            backend
            if callable(backend_matmul)
            and callable(getattr(backend, "supports_tokens", None))
            else None
        )
        resolved_matmul = (
            backend_matmul if resolved_backend is not None else backend
        )
        # Validate before allocating or mutating the module. A failed
        # installation must not leave hundreds of MiB of arenas and a half-built
        # controller attached to a reusable template.
        if not callable(resolved_matmul):
            raise ValueError("packed KDA backend has no callable matmul")
        storage_mode = str(storage_mode).strip().lower()
        if storage_mode not in {"arena", "stage"}:
            raise ValueError(
                "packed KDA storage mode must be arena or stage"
            )
        stage_sync_mode = _normalize_stage_sync_mode(stage_sync_mode)
        stage_sync_contract = _effective_stage_sync_mode(
            dev, stage_sync_mode
        )
        if storage_mode == "stage":
            if arenas is not None:
                raise ValueError(
                    "direct staged KDA storage cannot use persistent arenas"
                )
            capable, reason = stage_storage_capability(
                dev, stage_sync_mode
            )
            if not capable:
                raise ValueError(reason)

        projections = {}
        role_names = {}
        projection_specs = list(_CORE_PROJECTION_SPECS)
        # K3 uses full-rank g_proj. Retain support for the low-rank model
        # variant by packing only its first-stage g_a_proj.
        if getattr(attention, "g_proj", None) is not None:
            projection_specs.append(("g", "g_proj"))
        elif getattr(attention, "g_a_proj", None) is not None:
            projection_specs.append(("g_a", "g_a_proj"))
        projection_specs.extend(_OPTIONAL_PROJECTION_SPECS)
        q_cursor = 0
        scale_cursor = 0
        slots = {}
        for role, attribute in projection_specs:
            projection = getattr(attention, attribute, None)
            required = role in {"q", "k", "v"}
            if projection is None or not hasattr(projection, "weight"):
                if not required:
                    continue
                raise ValueError(f"KDA attention has no {role}_proj.weight")
            if projection.bias is not None:
                if not required:
                    # A packed weight-only call cannot reproduce an optional
                    # biased projection. Leave it on the ordinary dense path.
                    continue
                raise ValueError(f"{role}_proj must be bias-free")
            shape = tuple(projection.weight.shape)
            if len(shape) != 2:
                raise ValueError(f"{role}_proj.weight must be a matrix")
            rows, cols = shape
            q_cursor = _align(q_cursor)
            scale_cursor = _align(scale_cursor * 4) // 4
            slots[role] = (q_cursor, rows * cols, scale_cursor, rows, cols)
            q_cursor += rows * cols
            scale_cursor += rows
            projections[role] = projection
            role_names[role] = f"self_attn.{attribute}.weight"

        roles = tuple(projections)
        names_to_roles = {
            name: role for role, name in role_names.items()
        }

        q_size = _align(q_cursor)
        scale_size = _align(scale_cursor * 4) // 4
        if storage_mode == "stage":
            q_arena = None
            scale_arena = torch.empty(
                scale_size,
                dtype=torch.float32,
                device=dev,
            )
        elif arenas is None:
            q_arena = torch.empty(q_size, dtype=torch.int8, device=dev)
            scale_arena = torch.empty(
                scale_size,
                dtype=torch.float32,
                device=dev,
            )
        else:
            q_arena, scale_arena = arenas
            q_device_mismatch = (
                q_arena.device.type != dev.type
                or (
                    dev.index is not None
                    and q_arena.device.index != dev.index
                )
            )
            scale_device_mismatch = (
                scale_arena.device.type != dev.type
                or (
                    dev.index is not None
                    and scale_arena.device.index != dev.index
                )
            )
            if (q_arena.dtype != torch.int8 or q_arena.numel() < q_size
                    or q_device_mismatch):
                raise ValueError("shared packed KDA weight arena is incompatible")
            if (scale_arena.dtype != torch.float32
                    or scale_arena.numel() < scale_size
                    or scale_device_mismatch):
                raise ValueError("shared packed KDA scale arena is incompatible")
        if q_arena is not None:
            attention.register_buffer(self._QBUF, q_arena, persistent=False)
        attention.register_buffer(self._SBUF, scale_arena, persistent=False)
        setattr(attention, self._CTRL, self)

        self.dev = dev
        self.storage_mode = storage_mode
        self.stage_sync_mode = stage_sync_mode
        self.stage_sync_contract = stage_sync_contract
        self.backend = resolved_backend
        self.module = module
        self.attention = attention
        self.state = state
        self.matmul = resolved_matmul
        self.projections = projections
        self.roles = roles
        self.role_names = role_names
        self.names_to_roles = names_to_roles
        self.trigger_role = roles[0]
        self._cold_weights = {
            role: projection.weight
            for role, projection in projections.items()
        }
        self.slots = slots
        self.loaded = set()
        self.dense_loaded = set()
        self._dense_materialized = False
        self._fused_hidden = None
        self._fused_hidden_version = None
        self._fused_outputs = {}
        self._stage_q_arena = None
        self._stage_lease = None
        self._stage_records = {}
        self.stage_bind_count = 0
        self.stage_bind_failures = 0
        self.stage_stale_rejections = 0
        self.stage_weight_copy_bytes = 0
        self.stage_scale_copy_bytes = 0
        self.stage_fence_records = 0
        self.stage_fence_waits = 0
        self.stage_fence_sync_fallbacks = 0
        self.stage_fence_failures = 0
        self.stage_fifo_records = 0
        self.stage_fifo_reuses = 0
        self.stage_full_shape_probes = 0
        self.stage_full_shape_passes = 0
        self.stage_release_count = 0
        self.last_stage_generation = None
        self._stage_full_shape_validated = False
        self.packed_project_calls = 0
        self.packed_operator_calls = 0
        self.fused_operator_calls = 0
        self.separate_operator_calls = 0
        self.dense_crossover_project_calls = 0
        self.dense_crossover_materializations = 0
        self.dense_crossover_releases = 0
        self.runtime_failures = 0
        requested_fusion = (
            bool(getattr(backend, "fuse_qkv", False))
            if fuse_qkv is None else bool(fuse_qkv)
        )
        self.fusion_eligible, self.fusion_reason = (
            self._fusion_layout_status()
        )
        self.fuse_qkv = requested_fusion and self.fusion_eligible
        if not requested_fusion:
            self.fusion_reason = (
                getattr(backend, "fusion_reason", None)
                or "fusion not requested"
            )
        self._fusion_scope_active = False
        self._fusion_scope_hidden = None
        self._prior_attention_instance_forward = attention.__dict__.get(
            "forward", None
        )
        self._original_attention_forward = attention.forward
        self._prior_instance_forwards = {}
        self._original_forwards = {}
        for role, projection in projections.items():
            self._prior_instance_forwards[role] = projection.__dict__.get(
                "forward", None)
            self._original_forwards[role] = projection.forward

            def packed_forward(this_projection, hidden, _role=role,
                               _controller=self):
                return _controller.project(_role, hidden)

            projection.forward = types.MethodType(packed_forward, projection)

        def scoped_attention_forward(
            this_attention, *args, _controller=self, **kwargs
        ):
            _controller._clear_fused_outputs()
            _controller._fusion_scope_active = True
            _controller._fusion_scope_hidden = None
            try:
                return _controller._original_attention_forward(
                    *args, **kwargs
                )
            finally:
                # Cached projection values are valid only inside this audited
                # Kimi attention call, whose eligible roles all consume the
                # same unchanged hidden_states tensor. Inference tensors expose
                # no version counter, so never let the cache escape that scope.
                _controller._fusion_scope_active = False
                _controller._fusion_scope_hidden = None
                _controller._clear_fused_outputs()
                _controller.release_dense_crossover()
                _controller.release_stage_binding()

        attention.forward = types.MethodType(
            scoped_attention_forward, attention
        )

    @property
    def enabled(self):
        return self.state.enabled

    @property
    def nbytes(self):
        q_arena = getattr(self.attention, self._QBUF, None)
        scale_arena = getattr(self.attention, self._SBUF)
        return (
            (
                q_arena.numel() * q_arena.element_size()
                if q_arena is not None else 0
            )
            + scale_arena.numel() * scale_arena.element_size()
        )

    @property
    def persistent_weight_bytes(self):
        q_arena = getattr(self.attention, self._QBUF, None)
        if q_arena is None:
            return 0
        return q_arena.numel() * q_arena.element_size()

    @property
    def persistent_scale_bytes(self):
        scale_arena = getattr(self.attention, self._SBUF)
        return scale_arena.numel() * scale_arena.element_size()

    def _require_stage_current(self):
        if self.storage_mode != "stage":
            return
        if (
            self._stage_q_arena is None
            or self._stage_lease is None
            or not _stage_lease_is_current(self._stage_lease)
        ):
            self.stage_stale_rejections += 1
            error = RuntimeError(
                "packed KDA staging lease is stale or unbound"
            )
            self.state.disable(error)
            raise error
        if self.stage_sync_contract == "mps_fifo":
            try:
                current = _mps_fifo_stream_signature(self.dev)
                if current != self._stage_lease.stream_signature:
                    raise RuntimeError(
                        "MPS staging stream identity changed before use"
                    )
            except Exception as exc:
                self.stage_stale_rejections += 1
                error = RuntimeError(
                    "packed KDA MPS FIFO stream contract is stale"
                )
                error.add_note(str(exc))
                self.state.disable(error)
                raise error from exc

    def _views(self, role):
        _owned_qoff, qn, soff, rows, cols = self.slots[role]
        if self.storage_mode == "stage":
            self._require_stage_current()
            qoff = self._stage_records[role][0]
            q_arena = self._stage_q_arena
        else:
            qoff = self.slots[role][0]
            q_arena = getattr(self.attention, self._QBUF)
        scale_arena = getattr(self.attention, self._SBUF)
        return (
            q_arena[qoff:qoff + qn].view(rows, cols),
            scale_arena[soff:soff + rows],
        )

    def _fusion_layout_status(self):
        records = [self.slots[role] for role in self.roles]
        columns = {record[4] for record in records}
        if len(columns) != 1:
            return False, "packed projection input widths differ"
        prior = records[0]
        for record in records[1:]:
            if record[0] != prior[0] + prior[1]:
                return False, "packed weight arena has alignment gaps"
            if record[2] != prior[2] + prior[3]:
                return False, "packed scale arena has alignment gaps"
            prior = record
        return True, None

    def _fused_views(self):
        first = self.slots[self.roles[0]]
        last = self.slots[self.roles[-1]]
        _owned_qoff, _qn, soff, _rows, cols = first
        if self.storage_mode == "stage":
            self._require_stage_current()
            qoff = self._stage_records[self.roles[0]][0]
            last_record = self._stage_records[self.roles[-1]]
            q_end = last_record[0] + last_record[1]
            q_arena = self._stage_q_arena
        else:
            qoff = first[0]
            q_end = last[0] + last[1]
            q_arena = getattr(self.attention, self._QBUF)
        scale_end = last[2] + last[3]
        scale_arena = getattr(self.attention, self._SBUF)
        split_rows = tuple(self.slots[role][3] for role in self.roles)
        total_rows = sum(split_rows)
        return (
            q_arena[qoff:q_end].view(total_rows, cols),
            scale_arena[soff:scale_end],
            split_rows,
        )

    def _clear_fused_outputs(self):
        self._fused_hidden = None
        self._fused_hidden_version = None
        self._fused_outputs.clear()

    @staticmethod
    def _tensor_version(tensor):
        try:
            return tensor._version
        except RuntimeError:
            return None

    def _take_fused_output(self, role, hidden):
        if (
            self._fused_hidden is not hidden
            or self._fused_hidden_version != self._tensor_version(hidden)
        ):
            self._clear_fused_outputs()
            return None
        output = self._fused_outputs.pop(role, None)
        if output is not None:
            self.packed_project_calls += 1
        if not self._fused_outputs:
            self._clear_fused_outputs()
        return output

    def arenas(self):
        return (
            getattr(self.attention, self._QBUF, None),
            getattr(self.attention, self._SBUF),
        )

    def consumes(self, name):
        return name in self.names_to_roles

    def stage_descriptor(self, items):
        """Validate and describe one ordered direct-view KDA upload slice."""
        if self.storage_mode != "stage":
            raise ValueError("stage descriptor requested for arena storage")
        consumed = [
            item for item in items if item[0] in self.names_to_roles
        ]
        expected_names = [self.role_names[role] for role in self.roles]
        actual_names = [item[0] for item in consumed]
        if actual_names != expected_names:
            raise ValueError(
                "packed KDA stage order/coverage mismatch: "
                f"{actual_names!r} != {expected_names!r}"
            )
        records = {}
        previous_end = None
        for item, role in zip(consumed, self.roles):
            name, _full, shape, qoff, qbytes, scoff, scale_rows = item
            _owned_qoff, expected_bytes, _soff, rows, cols = self.slots[role]
            if name != self.role_names[role]:
                raise ValueError(f"packed KDA stage role mismatch for {role}")
            if tuple(shape) != (rows, cols) or qbytes != expected_bytes:
                raise ValueError(
                    f"packed KDA stage shape changed for {role}: "
                    f"{tuple(shape)}"
                )
            if scale_rows != rows:
                raise ValueError(
                    f"packed KDA stage scale count changed for {role}"
                )
            if previous_end is not None and qoff != previous_end:
                raise ValueError(
                    "packed KDA stage weight slice has alignment gaps"
                )
            records[role] = (
                qoff, qbytes, scoff, rows, cols
            )
            previous_end = qoff + qbytes
        first = records[self.roles[0]]
        return {
            "roles": tuple(self.roles),
            "records": records,
            "q_start": first[0],
            "q_end": previous_end,
        }

    def reject_stage_layout(self, error):
        """Atomically reject direct views so apply_pack densifies every role."""
        if self.storage_mode != "stage":
            raise ValueError("stage rejection requested for arena storage")
        self.stage_bind_failures += 1
        self._stage_q_arena = None
        self._stage_lease = None
        self._stage_records = {}
        self.loaded.clear()
        self.state.disable(error)

    def bind_stage(self, qweight_stage, scale_stage, descriptor, lease):
        """Bind all roles without copying int8 weights out of the upload arena."""
        if self.storage_mode != "stage":
            raise ValueError("direct staging is disabled for this controller")
        if not self.enabled:
            return False
        try:
            if not isinstance(descriptor, dict):
                raise ValueError("packed KDA stage descriptor is missing")
            if tuple(descriptor.get("roles", ())) != self.roles:
                raise ValueError("packed KDA stage descriptor roles changed")
            records = descriptor.get("records")
            if not isinstance(records, dict) or set(records) != set(self.roles):
                raise ValueError("packed KDA stage descriptor is incomplete")
            if qweight_stage.dtype != torch.int8:
                raise TypeError("packed KDA stage weight dtype is not int8")
            if scale_stage.dtype != torch.float16:
                raise TypeError("packed KDA stage scale dtype is not float16")
            if not qweight_stage.is_contiguous():
                raise ValueError("packed KDA stage weight buffer is not contiguous")
            if qweight_stage.device != scale_stage.device:
                raise ValueError("packed KDA stage buffers use different devices")
            if qweight_stage.device.type != self.dev.type or (
                self.dev.index is not None
                and qweight_stage.device.index != self.dev.index
            ):
                raise ValueError("packed KDA stage buffer device changed")
            if lease is None or lease.key != _stage_key(
                self.dev, torch.int8
            ):
                raise ValueError("packed KDA stage lease identity changed")
            if lease.sync_contract != self.stage_sync_contract:
                raise ValueError("packed KDA stage sync contract changed")
            if (
                qweight_stage.data_ptr() != lease.data_ptr
                or not _stage_lease_is_current(lease)
            ):
                raise RuntimeError("packed KDA stage lease is already stale")
            if self.stage_sync_contract == "mps_fifo":
                current = _mps_fifo_stream_signature(self.dev)
                if current != lease.stream_signature:
                    raise RuntimeError(
                        "packed KDA MPS FIFO upload stream changed"
                    )
            q_start = descriptor.get("q_start")
            q_end = descriptor.get("q_end")
            if (
                type(q_start) is not int
                or type(q_end) is not int
                or q_start < 0
                or q_end <= q_start
                or q_end > qweight_stage.numel()
            ):
                raise ValueError("packed KDA stage weight bounds are invalid")
            scale_arena = getattr(self.attention, self._SBUF)
            expected_qoff = q_start
            for role in self.roles:
                qoff, qbytes, scoff, rows, cols = records[role]
                expected = self.slots[role]
                if (
                    qoff != expected_qoff
                    or qoff + qbytes > q_end
                    or qbytes != expected[1]
                    or rows != expected[3]
                    or cols != expected[4]
                    or scoff < 0
                    or scoff + rows > scale_stage.numel()
                ):
                    raise ValueError(
                        f"packed KDA stage record is invalid for {role}"
                    )
                expected_qoff = qoff + qbytes
                scale_offset = expected[2]
                scale_arena[
                    scale_offset:scale_offset + rows
                ].copy_(scale_stage[scoff:scoff + rows])
            if expected_qoff != q_end:
                raise ValueError(
                    "packed KDA stage descriptor has trailing weight bytes"
                )
        except Exception as exc:
            self.reject_stage_layout(exc)
            return False
        self._stage_q_arena = qweight_stage
        self._stage_lease = lease
        self._stage_records = records
        self.loaded.update(self.roles)
        self.stage_bind_count += 1
        self.stage_scale_copy_bytes += sum(
            self.slots[role][3] * 2
            for role in self.roles
        )
        self.last_stage_generation = lease.generation
        return True

    def release_stage_binding(self):
        """Fence GPU use, then drop all aliases to the recyclable upload arena."""
        if self.storage_mode != "stage" or self._stage_lease is None:
            return
        lease = self._stage_lease
        try:
            _record_stage_fence(lease, self.dev, self)
        finally:
            self._stage_q_arena = None
            self._stage_lease = None
            self._stage_records = {}
            self.stage_release_count += 1

    def begin_load(self):
        self.release_dense_crossover()
        self.release_stage_binding()
        self.loaded.clear()
        self.dense_loaded.clear()
        self._dense_materialized = False
        self._clear_fused_outputs()

    def load(self, name, qweight, scale):
        """Copy one eligible KDA role into owned storage; return True if consumed."""
        if not self.enabled:
            return False
        role = self.names_to_roles.get(name)
        if role is None:
            return False
        if self.storage_mode == "stage":
            # bind_stage validates and installs the entire six-role bundle
            # atomically before this per-item loop begins.
            return role in self.loaded
        try:
            destination_q, destination_scale = self._views(role)
            if destination_q.shape != qweight.shape:
                raise ValueError(
                    f"packed {role} shape changed: "
                    f"{tuple(qweight.shape)} != {tuple(destination_q.shape)}")
            if scale.numel() != destination_scale.numel():
                raise ValueError(
                    f"packed {role} scale count changed: "
                    f"{scale.numel()} != {destination_scale.numel()}")
            # These are real copies, not aliases of _stage_buf(). That staging
            # storage is overwritten as soon as the next layer is applied.
            destination_q.copy_(qweight)
            destination_scale.copy_(scale)
        except (NotImplementedError, RuntimeError, TypeError, ValueError,
                MemoryError) as exc:
            # The caller still owns qweight/scale and will immediately run the
            # ordinary dense branch for this role. Preserve every earlier role
            # first, then make all remaining roles dense process-wide.
            self._materialize_dense()
            self.state.disable(exc)
            return False
        self.loaded.add(role)
        return True

    def mark_dense(self, name):
        """Record that apply_pack installed this layer's dense projection."""
        role = self.names_to_roles.get(name)
        if role is not None:
            self.dense_loaded.add(role)

    def _materialize_dense(self, roles=None):
        selected = self.roles if roles is None else roles
        for role in selected:
            if role not in self.loaded:
                continue
            qweight, scale = self._views(role)
            projection = self.projections[role]
            weight = projection.weight
            if (weight.device.type == "meta"
                    or tuple(weight.shape) != tuple(qweight.shape)
                    or weight.dtype != torch.float32):
                dense = qweight.to(torch.float32) * scale[:, None]
                projection.weight = torch.nn.Parameter(
                    dense, requires_grad=False)
            else:
                torch.mul(
                    qweight.to(torch.float32),
                    scale[:, None],
                    out=weight.data,
                )

    def _ensure_dense_current_layer(self):
        if self._dense_materialized:
            return
        self._materialize_dense()
        self._dense_materialized = True
        self.dense_crossover_materializations += 1
        self._clear_fused_outputs()

    def release_dense_crossover(self):
        """Drop transient fallback projections after the layer finishes.

        Production templates start with meta packed-role Parameters. A prompt
        wider than the packed-GEMV crossover temporarily materializes about
        1.32 GiB for K3's Q/K/V/G/F-A/B bundle; retaining those beside the
        packed arenas would erase much of the residency benefit during decode.
        """
        if not self._dense_materialized or not self.enabled:
            return
        released = False
        for role, projection in self.projections.items():
            cold_weight = self._cold_weights[role]
            if (
                cold_weight.device.type == "meta"
                and projection.weight is not cold_weight
            ):
                projection.weight = cold_weight
                released = True
        self._dense_materialized = False
        if released:
            self.dense_crossover_releases += 1
        self._clear_fused_outputs()

    def _supports_tokens(self, token_count):
        if self.backend is None:
            return True
        return bool(self.backend.supports_tokens(token_count))

    def finish_load(self):
        """Finish one layer, accepting all-packed or safely densifying hybrids."""
        missing = set(self.roles) - self.loaded - self.dense_loaded
        if not missing:
            if self.enabled and self.dense_loaded:
                # A partial int8 spine is valid: apply_pack has already installed
                # the dense role(s). Rebuild the packed subset so this whole
                # current layer can take nn.Linear after the shared kill switch.
                self._materialize_dense()
                error = RuntimeError(
                    "packed KDA layer includes dense projection(s): "
                    + ", ".join(sorted(self.dense_loaded))
                )
                self.state.disable(error)
            return
        self._materialize_dense(self.loaded)
        error = RuntimeError(
            "KDA packed bundle is incomplete; no packed or dense value for "
            + ", ".join(sorted(missing))
        )
        self.state.disable(error)
        # A role not recorded by either path may still contain a prior layer's
        # weight (or meta storage), so continuing would silently corrupt output.
        raise error

    def project(self, role, hidden):
        if not self.enabled or role not in self.loaded:
            return self._original_forwards[role](hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        if not self._supports_tokens(flat.shape[0]):
            # The raw weight-only kernel is a GEMV win, not a universal GEMM
            # replacement. Rebuild the current layer once and keep the backend
            # enabled for later decode-sized calls/layers.
            self._ensure_dense_current_layer()
            self.dense_crossover_project_calls += 1
            return self._original_forwards[role](hidden)
        scoped = self._fusion_scope_active
        if scoped and self._fusion_scope_hidden is None:
            if role == self.trigger_role:
                # Establish the scope from the tensor actually passed to
                # q_proj. This remains correct when Kimi first unpads and
                # replaces its original hidden_states object.
                self._fusion_scope_hidden = hidden
            else:
                scoped = False
        else:
            scoped = scoped and self._fusion_scope_hidden is hidden
        if scoped and role != self.trigger_role:
            cached = self._take_fused_output(role, hidden)
            if cached is not None:
                return cached
        qweight, scale = self._views(role)
        try:
            if scoped and role == self.trigger_role:
                probing_stage_shape = (
                    self.storage_mode == "stage"
                    and not self._stage_full_shape_validated
                )
                if probing_stage_shape:
                    self.stage_full_shape_probes += 1
                if self.fuse_qkv:
                    fused_qweight, fused_scale, split_rows = (
                        self._fused_views()
                    )
                    output = self.matmul(
                        flat, fused_qweight, fused_scale
                    )
                    output = output.view(
                        *hidden.shape[:-1],
                        sum(split_rows),
                    )
                    projected = output.split(split_rows, dim=-1)
                    operator_calls = 1
                    self.fused_operator_calls += 1
                else:
                    projected = tuple(
                        self.matmul(flat, *self._views(projected_role)).view(
                            *hidden.shape[:-1],
                            self._views(projected_role)[0].shape[0],
                        )
                        for projected_role in self.roles
                    )
                    operator_calls = len(self.roles)
                    self.separate_operator_calls += operator_calls
                if probing_stage_shape:
                    self._stage_full_shape_validated = True
                    self.stage_full_shape_passes += 1
                self._clear_fused_outputs()
                self._fused_hidden = hidden
                self._fused_hidden_version = self._tensor_version(hidden)
                self._fused_outputs = {
                    fused_role: fused_output
                    for fused_role, fused_output in zip(
                        self.roles[1:], projected[1:]
                    )
                }
                self.packed_project_calls += 1
                self.packed_operator_calls += operator_calls
                return projected[0]
            output = self.matmul(flat, qweight, scale)
            output = output.view(*hidden.shape[:-1], qweight.shape[0])
            self.packed_project_calls += 1
            self.packed_operator_calls += 1
            self.separate_operator_calls += 1
            return output
        except (NotImplementedError, RuntimeError, TypeError) as exc:
            # The packed operator may be present in the dispatcher yet reject
            # a future device generation, shape, or PyTorch ABI. Rebuild every
            # current bundled weight before switching the template to dense.
            self.runtime_failures += 1
            self._clear_fused_outputs()
            self._materialize_dense()
            self.state.disable(exc)
            return self._original_forwards[role](hidden)

    def status(self):
        return {
            "enabled": self.enabled,
            "storage_mode": self.storage_mode,
            "stage_sync_mode": self.stage_sync_mode,
            "stage_sync_contract": self.stage_sync_contract,
            "persistent_weight_bytes": self.persistent_weight_bytes,
            "persistent_scale_bytes": self.persistent_scale_bytes,
            "stage_bound": self._stage_lease is not None,
            "stage_bind_count": self.stage_bind_count,
            "stage_bind_failures": self.stage_bind_failures,
            "stage_stale_rejections": self.stage_stale_rejections,
            "stage_weight_copy_bytes": self.stage_weight_copy_bytes,
            "stage_scale_copy_bytes": self.stage_scale_copy_bytes,
            "stage_fence_records": self.stage_fence_records,
            "stage_fence_waits": self.stage_fence_waits,
            "stage_fence_sync_fallbacks": (
                self.stage_fence_sync_fallbacks
            ),
            "stage_fence_failures": self.stage_fence_failures,
            "stage_fifo_records": self.stage_fifo_records,
            "stage_fifo_reuses": self.stage_fifo_reuses,
            "stage_full_shape_probes": self.stage_full_shape_probes,
            "stage_full_shape_passes": self.stage_full_shape_passes,
            "stage_release_count": self.stage_release_count,
            "last_stage_generation": self.last_stage_generation,
            "packed_project_calls": self.packed_project_calls,
            "packed_operator_calls": self.packed_operator_calls,
            "fused_operator_calls": self.fused_operator_calls,
            "separate_operator_calls": self.separate_operator_calls,
            "dense_crossover_project_calls": (
                self.dense_crossover_project_calls
            ),
            "dense_crossover_materializations": (
                self.dense_crossover_materializations
            ),
            "dense_crossover_releases": self.dense_crossover_releases,
            "runtime_failures": self.runtime_failures,
            "fuse_qkv": self.fuse_qkv,
            "fusion_eligible": self.fusion_eligible,
            "fusion_reason": self.fusion_reason,
            "packed_roles": list(self.roles),
            "backend": (
                self.backend.status() if self.backend is not None else None
            ),
        }

    def uninstall(self):
        """Restore the original modules after a transactional setup failure."""
        for role, projection in self.projections.items():
            previous = self._prior_instance_forwards[role]
            if previous is None:
                projection.__dict__.pop("forward", None)
            else:
                projection.forward = previous
        previous_attention = self._prior_attention_instance_forward
        if previous_attention is None:
            self.attention.__dict__.pop("forward", None)
        else:
            self.attention.forward = previous_attention
        if getattr(self.attention, self._CTRL, None) is self:
            delattr(self.attention, self._CTRL)
        for name in (self._QBUF, self._SBUF):
            if name in self.attention._buffers:
                del self.attention._buffers[name]


def install_dynamic_q8_qkv(
    module,
    dev,
    state,
    backend,
    arenas=None,
    fuse_qkv=None,
    storage_mode="arena",
    stage_sync_mode="event",
):
    return DynamicPackedQ8QKV(
        module,
        dev,
        state,
        backend,
        arenas=arenas,
        fuse_qkv=fuse_qkv,
        storage_mode=storage_mode,
        stage_sync_mode=stage_sync_mode,
    )


def dynamic_q8_qkv(module):
    attention = getattr(module, "self_attn", None)
    return getattr(attention, DynamicPackedQ8QKV._CTRL, None)


# Phase counters (always accumulated; printed by kimi_run under K3_PROFILE).
PHASE = {"read_s": 0.0, "read_bytes": 0, "h2d_s": 0.0, "h2d_bytes": 0,
         "deq_s": 0.0, "other_s": 0.0, "n_layers": 0}

_readers = None
_readers_lock = threading.Lock()


def _pool_exec():
    global _readers
    if _readers is None:
        with _readers_lock:
            if _readers is None:
                _readers = _cf.ThreadPoolExecutor(READ_THREADS,
                                                  thread_name_prefix="k3spine")
    return _readers


# ------------------------------------------------------- layout planning ----
# The set of files a layer needs, their sizes and their slot offsets never
# change between tokens, so plan once per prefix.  Saves ~5,200 stat() calls
# and ~2,600 open() calls per token.
_layouts = {}
_layout_lock = threading.Lock()


def _align(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


def plan_layout(module, prefix, int8_dir, res_dir, inv, spine):
    """-> dict(items=[(name, full, shape, qoff, qbytes, scoff_elems, rows)],
              qtotal, sctotal_elems, other=[(name, full, path_or_None)])"""
    key = prefix
    got = _layouts.get(key)
    if got is not None:
        return got
    items, other = [], []
    qoff = 0
    scoff = 0                     # in fp16 elements
    packed_qkv = dynamic_q8_qkv(module)
    if (
        packed_qkv is not None
        and packed_qkv.storage_mode == "stage"
    ):
        # Direct-view fusion requires Q/K/V/G/F-A/B to occupy one gap-free
        # int8 slice in exactly the controller's role order. This changes only
        # the in-memory/read plan; checkpoint files remain untouched.
        parameters = list(module.named_parameters())
        by_name = dict(parameters)
        consumed_names = [
            packed_qkv.role_names[role] for role in packed_qkv.roles
        ]
        consumed_set = set(consumed_names)
        parameter_iter = [
            (name, by_name[name])
            for name in consumed_names if name in by_name
        ]
        parameter_iter.extend(
            (name, parameter)
            for name, parameter in parameters
            if name not in consumed_set
        )
    else:
        # Preserve the established arena-mode layout and iteration order.
        parameter_iter = module.named_parameters()
    for name, _ in parameter_iter:
        if ".experts." in name:
            continue
        full = prefix + name
        ip = os.path.join(int8_dir, full + ".i8")
        if spine == "int8" and os.path.exists(ip):
            rows, cols = inv[full]["shape"]
            nb = rows * cols
            items.append((name, full, (rows, cols), qoff, nb, scoff, rows))
            qoff = _align(qoff + nb)
            scoff = _align(scoff * 2 + rows * 2) // 2
            continue
        bp = os.path.join(res_dir, full)
        other.append((name, full, bp if os.path.exists(bp) else None))
    # The bf16 leftovers (norms, conv1d, A_log, dt_bias, res_proj) are tiny but
    # numerous -- ~8 per layer, ~744 per forward. As separate blocking .to(DEV)
    # calls they measured 2.6 s/forward, almost all per-call overhead, so they
    # get the same treatment: one packed buffer, one transfer.
    ooff = 0
    oplan = []
    for name, full, bp in other:
        if bp is None:
            continue
        meta = inv[full]
        nb = os.path.getsize(bp)
        oplan.append((name, full, meta["dtype"], tuple(meta["shape"]), ooff, nb, bp))
        ooff = _align(ooff + nb)
    lay = {"items": items, "qtotal": qoff, "sctotal": scoff, "other": other,
           "oplan": oplan, "ototal": ooff, "int8_dir": int8_dir}
    if (
        packed_qkv is not None
        and packed_qkv.storage_mode == "stage"
    ):
        try:
            lay["packed_qkv_stage"] = packed_qkv.stage_descriptor(items)
        except (TypeError, ValueError) as exc:
            lay["packed_qkv_stage_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
    with _layout_lock:
        _layouts[key] = lay
    return lay


# ---------------------------------------------------- chunked job planning ---
# Which packed buffer a job writes into.  The plan (paths, file offsets, byte
# counts, destination offsets) is identical on every token, so it is built once
# per layer and only the memoryview slices are materialised per call.
_QB, _SCB, _OB = 0, 1, 2


def plan_jobs(lay):
    """-> [(path, file_off, buf_id, dest_off, nbytes)] split at spine_io.CHUNK
    and sorted longest-first for a balanced greedy makespan."""
    jobs = lay.get("jobs")
    if jobs is not None:
        return jobs
    int8_dir = lay["int8_dir"]
    raw = []
    for _n, full, _sh, qoff, nb, scoff, rows in lay["items"]:
        base = os.path.join(int8_dir, full)
        raw.append((base + ".i8", _QB, qoff, nb))
        raw.append((base + ".sc", _SCB, scoff * 2, rows * 2))
    for _n, _f, _dt, _sh, ooff, nb, bp in lay["oplan"]:
        raw.append((bp, _OB, ooff, nb))
    ch = spine_io.CHUNK
    jobs = []
    for path, bid, doff, nb in raw:
        if ch <= 0 or nb <= ch:
            jobs.append((path, 0, bid, doff, nb))
            continue
        o = 0
        while o < nb:
            k = min(ch, nb - o)
            jobs.append((path, o, bid, doff + o, k))
            o += k
    jobs.sort(key=lambda j: -j[4])
    lay["jobs"] = jobs
    return jobs


def _rdadvise_next(prefix):
    """Kick the kernel's readahead for the layer AFTER the one we are about to
    read, so its pages are already in flight when the preloader gets there.
    Only fires once a layer has been seen before (its layout is cached), i.e.
    from the second forward pass on -- prefill is unaffected."""
    try:
        head, _, tail = prefix.rpartition("layers.")
        idx = int(tail.rstrip("."))
    except (ValueError, AttributeError):
        return
    nxt = _layouts.get(f"{head}layers.{idx + 1}.")
    if nxt is None:
        return
    seen = set()
    for path, _fo, _b, _d, _n in plan_jobs(nxt):
        if path not in seen:
            seen.add(path)
            spine_io.rdadvise(path)


# ------------------------------------------------------------- read side ----
def read_pack(module, prefix, int8_dir, res_dir, inv, spine, load_resident):
    """Runs in the preloader thread.  Returns a pack the main thread applies."""
    lay = plan_layout(module, prefix, int8_dir, res_dir, inv, spine)
    items = lay["items"]
    qbuf = _acquire(lay["qtotal"])
    scbuf = _acquire(lay["sctotal"] * 2)
    obuf = _acquire(max(lay["ototal"], 1))

    if spine_io.ENABLED:
        t0 = _time.time()
        if spine_io.RDADVISE:
            _rdadvise_next(prefix)
        bufs = (memoryview(qbuf), memoryview(scbuf), memoryview(obuf))
        jobs = [(path, fo, bufs[bid][doff:doff + nb])
                for path, fo, bid, doff, nb in plan_jobs(lay)]
        spine_io.run_jobs(jobs, _pool_exec(), READ_THREADS,
                          nocache=spine_io.stream_tier(prefix))
        PHASE["read_s"] += _time.time() - t0
        PHASE["read_bytes"] += lay["qtotal"] + lay["ototal"]
        return {"lay": lay, "q": qbuf, "sc": scbuf, "other": obuf}

    qmv, scmv = memoryview(qbuf), memoryview(scbuf)

    def _one(it):
        _, full, _, qoff, nb, scoff, rows = it
        base = os.path.join(int8_dir, full)
        with open(base + ".i8", "rb") as f:
            f.readinto(qmv[qoff:qoff + nb])
        with open(base + ".sc", "rb") as f:
            f.readinto(scmv[scoff * 2:scoff * 2 + rows * 2])

    t0 = _time.time()
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
    return {"lay": lay, "q": qbuf, "sc": scbuf, "other": obuf}


# ------------------------------------------------------------ apply side ----
def apply_pack(module, prefix, pack, dev, dt, inv, dtmap, set_param, load_resident):
    """One H2D transfer for the whole layer, then a fused dequant straight into
    each parameter's existing buffer."""
    lay = pack["lay"]
    items = lay["items"]
    packed_qkv = dynamic_q8_qkv(module)
    if packed_qkv is not None:
        packed_qkv.begin_load()
    stage_lease = None
    direct_q, direct_scales, direct_other = _cpu_direct_view_flags(
        dev, packed_qkv
    )
    use_metal = DEQ == "metal" and dev.type == "mps" and metal_available()
    PHASE["n_layers"] += 1
    oplan = lay["oplan"]
    # All three host->device transfers go first, back to back. cpu->mps copy_ is
    # blocking, so a transfer wedged between kernel dispatches would stall the
    # main thread on the GPU queue; doing them up front lets the dequant kernels
    # pipeline against the next layer instead.
    t0 = _time.time()
    if items:
        qh = torch.frombuffer(pack["q"], dtype=torch.int8)[:lay["qtotal"]]
        sch = torch.frombuffer(pack["sc"], dtype=torch.float16)[:lay["sctotal"]]
        stage_sync_mode = (
            packed_qkv.stage_sync_mode
            if (
                packed_qkv is not None
                and packed_qkv.storage_mode == "stage"
            )
            else "event"
        )
        if direct_q:
            qd = qh
        else:
            qd, stage_lease = _stage_weight_buf(
                lay["qtotal"], dev, stage_sync_mode
            )
            qd.copy_(qh)
        if direct_scales:
            scd = sch
        else:
            scd = _stage_buf(lay["sctotal"], dev, torch.float16)
            scd.copy_(sch)
    if oplan:
        oh = torch.frombuffer(pack["other"], dtype=torch.uint8)[:lay["ototal"]]
        if direct_other:
            od = oh
        else:
            od = _stage_buf(lay["ototal"], dev, torch.uint8)
            od.copy_(oh)
    PHASE["h2d_s"] += _time.time() - t0
    # Preserve this counter's established payload convention (q + bf16 tail,
    # scales excluded), but count only bytes actually copied into staging.
    PHASE["h2d_bytes"] += (
        (0 if direct_q else lay["qtotal"])
        + (0 if direct_other else lay["ototal"])
    )
    t0 = _time.time()
    if (
        packed_qkv is not None
        and packed_qkv.storage_mode == "stage"
    ):
        stage_error = lay.get("packed_qkv_stage_error")
        if stage_error is not None:
            packed_qkv.reject_stage_layout(RuntimeError(stage_error))
        elif not items:
            packed_qkv.reject_stage_layout(
                RuntimeError("packed KDA stage has no int8 payload")
            )
        else:
            descriptor = lay.get("packed_qkv_stage")
            if descriptor is None:
                try:
                    descriptor = packed_qkv.stage_descriptor(items)
                    lay["packed_qkv_stage"] = descriptor
                except (TypeError, ValueError) as exc:
                    packed_qkv.reject_stage_layout(exc)
            if packed_qkv.enabled and descriptor is not None:
                packed_qkv.bind_stage(
                    qd, scd, descriptor, stage_lease
                )
    for name, full, shape, qoff, nb, scoff, rows in items:
        q = qd[qoff:qoff + nb].view(shape)
        sc = scd[scoff:scoff + rows]
        if packed_qkv is not None and packed_qkv.load(name, q, sc):
            continue
        p = module.get_parameter(name)
        fits = (p.device.type != "meta" and p.shape == torch.Size(shape)
                and p.dtype == torch.float32 and dt == torch.float32)
        if fits and use_metal:
            dequant_into(p.data, q, sc)
            if packed_qkv is not None and packed_qkv.consumes(name):
                packed_qkv.mark_dense(name)
            continue
        if fits and dev.type == "cuda" and cuda_dequant_into(p.data, q, sc):
            if packed_qkv is not None and packed_qkv.consumes(name):
                packed_qkv.mark_dense(name)
            continue
        if fits and DEQ == "mulout":
            torch.mul(q, sc.view(rows, 1).to(torch.float32), out=p.data)
            if packed_qkv is not None and packed_qkv.consumes(name):
                packed_qkv.mark_dense(name)
            continue
        t = (q.to(torch.float32) * sc.view(rows, 1).to(torch.float32)).to(dt)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
    PHASE["deq_s"] += _time.time() - t0
    t0 = _time.time()
    have = {rec[1] for rec in oplan}
    for name, full, sdt, shape, ooff, nb, _bp in oplan:
        src = od[ooff:ooff + nb].view(dtmap[sdt]).reshape(shape)
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != torch.Size(shape) or p.dtype != dt:
            set_param(module, name, src.to(dt).clone())
        else:
            p.data.copy_(src)                         # copy_ does the bf16->fp32
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
    for name, full, path in lay["other"]:
        if full in have:
            continue
        t = load_resident(full).to(dev, dt)            # not on disk: rare fallback
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
    if packed_qkv is not None:
        packed_qkv.finish_load()
    PHASE["other_s"] += _time.time() - t0
    # torch's cpu->mps copy_ is blocking (non_blocking=False), so the host
    # buffers are free the moment .copy_() returned.
    if items:
        del qh, sch, qd, scd
    if oplan:
        del oh, od
    # A pinned pack is owned by the caller's cache and will be applied again on
    # the next token: leave its buffers intact and in place.
    if id(pack.get("q")) in KEEP:
        return
    _release(pack["q"])
    _release(pack["sc"])
    _release(pack["other"])
    pack["q"] = pack["sc"] = pack["other"] = None


def describe():
    bits = [f"pack={int(PACK)}", f"deq={DEQ}", f"read_threads={READ_THREADS}"]
    if DEQ == "metal":
        bits.append("metal_ok=" + ("1" if metal_available() else f"0({metal_error()})"))
    if spine_io.ENABLED:
        bits.append("| io: " + spine_io.describe())
    return " ".join(bits)


def phase_report():
    p = PHASE
    if not p["n_layers"]:
        return ""
    rd = p["read_bytes"] / 1e9 / max(p["read_s"], 1e-9)
    hd = p["h2d_bytes"] / 1e9 / max(p["h2d_s"], 1e-9)
    return (f"[spine] {p['n_layers']} layers | read {p['read_s']:.1f}s "
            f"({p['read_bytes']/1e9:.1f} GB, {rd:.1f} GB/s, bg thread) | "
            f"h2d {p['h2d_s']:.1f}s ({hd:.1f} GB/s) | deq {p['deq_s']:.1f}s | "
            f"bf16-tail {p['other_s']:.1f}s")
