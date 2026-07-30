#!/usr/bin/env python3
"""lazy-K3: real Kimi-K3 inference on macOS or Linux by layer-streaming.

Uses Moonshot's own modeling_kimi_linear.py (audited) with a pure-PyTorch fla shim.
Per forward pass, each of the 93 decoder layers is materialized from the local
resident-spine download, routed experts are fetched on demand (HTTP Range, disk
cached) and dequantized from MXFP4, the layer runs on the selected MPS, CUDA, or
CPU device, then its weights are freed. Router selections can optionally be logged to
router_trace.jsonl with K3_TRACE=buffered (or K3_TRACE=sync for debugging).
The published baseline is a 64 GB M1 Max; tunable choices are not dispatched
from that product name.
"""
import argparse, atexit, codecs, functools, gc, json, math, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))    # fla shim, k3loader, mxfp4
# modeling files imported via tools/k3pkg package

import positional_io  # noqa: E402
import runtime_platform  # noqa: E402
import packed_q8  # noqa: E402
import routing_record as routing_records  # noqa: E402
import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)
# Eight is the measured M1 Max default, not a cross-chip conclusion. Keep it
# stable there when available, bound the automatic choice to this process's
# effective CPUs, and retain the override for per-machine CPU/power sweeps.
TORCH_THREADS = runtime_platform.configured_cpu_workers("K3_TORCH_THREADS", 8)
torch.set_num_threads(TORCH_THREADS)

import k3loader  # noqa: E402
import importlib  # noqa: E402
import apple_silicon  # noqa: E402

from k3pkg import modeling_kimi_linear as ml

CFG_JSON = json.load(open(os.path.join(ROOT, "k3-meta/config.json")))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None)
if Cfg is None:
    Cfg = importlib.import_module("k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG_JSON)
config._attn_implementation = "eager"
BASE_MOE_TOP_K = int(config.num_experts_per_token)
MOE_TOP_K = int(os.environ.get("K3_MOE_TOP_K", BASE_MOE_TOP_K))
if not 1 <= MOE_TOP_K <= BASE_MOE_TOP_K:
    raise ValueError(
        f"K3_MOE_TOP_K must be in [1,{BASE_MOE_TOP_K}], got {MOE_TOP_K}")
if MOE_TOP_K != BASE_MOE_TOP_K:
    config.num_experts_per_token = MOE_TOP_K
    print(f"[quality] routed experts reduced {BASE_MOE_TOP_K}->{MOE_TOP_K}; "
          "this is an explicit approximate speed dial", flush=True)
H = config.hidden_size
NL = config.num_hidden_layers
PFX = "language_model.model."
# Sensible defaults, no env vars required: use the GPU when there is one, and
# use the int8 spine when it has been built. Both remain overridable.
INT8_DIR = os.path.join(ROOT, "k3-resident-int8/tensors")
APPLE_CAPS = apple_silicon.snapshot()


def _mps_available():
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _cuda_available():
    return bool(torch.cuda.is_available())


def _resolve_device():
    requested = os.environ.get("K3_DEV")
    # Reject typos before touching MPS/CUDA. In particular, a value such as
    # "cudax" must not initialize the CUDA runtime merely because it shares a
    # prefix with a valid device.
    normalized = runtime_platform.normalize_requested_device(requested)
    mps_available = (
        _mps_available()
        if normalized is None or normalized == "mps" else False
    )
    # Preserve the established Mac path: once MPS is available, do not
    # initialize or even query an unrelated CUDA runtime.
    need_cuda = (
        not mps_available
        and (normalized is None or normalized.startswith("cuda"))
    )
    cuda_available = _cuda_available() if need_cuda else False
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    spec = runtime_platform.choose_device_spec(
        requested,
        mps_available=mps_available,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
    )
    if normalized is None:
        if spec.startswith("cuda"):
            print(
                f"[config] MPS unavailable; auto-selected {spec}",
                flush=True,
            )
        elif spec == "cpu":
            print("[config] no MPS or CUDA GPU found — running on CPU (slow)",
                  flush=True)
    return torch.device(spec)


def _device_synchronize():
    runtime_platform.synchronize_device(torch, DEV)


def _auto_spine():
    try:
        if any(f.endswith(".i8") for f in os.listdir(INT8_DIR)):
            return "int8"
    except FileNotFoundError:
        pass
    return "bf16"


DEV = _resolve_device()                                      # cpu | mps | cuda[:N]
SPINE = os.environ.get("K3_SPINE") or _auto_spine()           # bf16 | int8 | mixed
if SPINE == "bf16" and "K3_SPINE" not in os.environ:
    print("[config] int8 spine not found — using bf16 (2x the per-token I/O). "
          "Build it with: python tools/convert_spine_int8.py", flush=True)

# --- mixed-precision spine (K3_SPINE=mixed; DEFAULT IS UNCHANGED) ------------
# The 53 GB int8 spine is re-read every token, so bytes are seconds. A mixed
# spine keeps int8 where the error lands on token identity and drops selected
# ROLES to 6 or 4 bits; tools/mixed_spine.py reads the mix (falling back to the
# int8 spine for every tensor the policy left alone) and dequantizes each codec
# with its own Metal kernel. Build one with:
#     python tools/convert_spine_int4.py --policy moe4
# Per-role error measurements that justify a policy: tools/spine_sensitivity.py.
MIXED = SPINE == "mixed"
QUANT = SPINE in ("int8", "mixed")      # "a quantized spine of some kind"
if MIXED:
    import mixed_spine  # noqa: E402
    if not mixed_spine.available():
        sys.exit(f"K3_SPINE=mixed but no mixed spine at {mixed_spine.MIXED_DIR}. "
                 f"Build one with: python tools/convert_spine_int4.py --policy moe4")
    print(f"[spine] mixed: {mixed_spine.describe()}", flush=True)
    if os.environ.get("K3_PROFILE", "0") == "1":
        import atexit
        atexit.register(lambda: print(mixed_spine.phase_report(), flush=True))
# K3_APPROX=1 = "approx mode": approximate numerics (fp16 weights) + n-gram
# speculation. Output stays coherent but near-tie tokens may differ from the
# fp32 reference — never use for oracle runs. Speed effect is unproven until a
# quiet-machine A/B; if it measures faster it can earn a faster name.
APPROX = os.environ.get("K3_APPROX", "0") == "1"
DT = torch.float16 if (APPROX or os.environ.get("K3_DTYPE", "fp32") == "fp16") else torch.float32


def _mode_enabled(value, *, automatic):
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return automatic
    if normalized in apple_silicon.TRUE_MODES:
        return True
    if normalized in apple_silicon.FALSE_MODES:
        return False
    raise ValueError("mode must be auto, on/1/true, or off/0/false")


# The raw row-int8 operator is private and backend registrations differ between
# PyTorch builds. Discover it by dispatcher + an analytically exact call at the
# real KDA projection shape; never infer CUDA/MPS/CPU support from a chip name.
_INT8_LM_HEAD_MODE = os.environ.get("K3_INT8_LM_HEAD", "auto")
_INT8_KDA_QKV_REQUESTED_EARLY = _mode_enabled(
    os.environ.get("K3_INT8_KDA_QKV", "0"),
    automatic=False,
)
_INT8_KDA_STORAGE_MODE = os.environ.get(
    "K3_INT8_KDA_STORAGE", "arena"
).strip().lower()
if _INT8_KDA_STORAGE_MODE not in {"arena", "stage"}:
    raise ValueError("K3_INT8_KDA_STORAGE must be arena or stage")
_INT8_KDA_STAGE_SYNC_MODE = os.environ.get(
    "K3_INT8_KDA_STAGE_SYNC", "event"
).strip().lower().replace("-", "_")
if _INT8_KDA_STAGE_SYNC_MODE not in {"event", "mps_fifo"}:
    raise ValueError(
        "K3_INT8_KDA_STAGE_SYNC must be event or mps_fifo"
    )
_INT8_LM_HEAD_REQUESTED = _mode_enabled(
    _INT8_LM_HEAD_MODE,
    # Preserve the established, fully exercised MPS head default. CPU has
    # decisive microbench evidence but remains opt-in until a full active-path
    # Linux/CPU sequence gate lands.
    automatic=DEV.type == "mps",
)
# Keep the existing MPS int8 pilot-router default independent from the head and
# KDA switches.  The CPU path remains dense here until its variable-T router
# workload has its own crossover gate; the new CPU head/KDA paths are opt-in.
_PILOT_INT8_CAPABILITY_NEEDED = (
    DEV.type == "mps"
    and os.environ.get("K3_PILOT", "1") == "1"
    and os.environ.get("K3_PILOT_GATE_DT", "int8") == "int8"
)
_PACKED_Q8_NEEDED = (
    QUANT
    and (
        (
            DT == torch.float32
            and (
                _INT8_LM_HEAD_REQUESTED
                or _INT8_KDA_QKV_REQUESTED_EARLY
            )
        )
        or _PILOT_INT8_CAPABILITY_NEEDED
    )
)
_KDA_PROJECTION_SIZE = (
    int(config.linear_attn_config["head_dim"])
    * int(config.linear_attn_config["num_heads"])
)
_PACKED_Q8_BACKEND = packed_q8.discover(
    torch,
    DEV,
    # Every current consumer supplies fp32 activations.  This is intentionally
    # independent of the main model dtype because the pilot router calls
    # ``h.float()`` even in approximate/fp16 runs.
    torch.float32,
    (_KDA_PROJECTION_SIZE, H),
    mode=(
        os.environ.get("K3_NATIVE_INT8", "auto")
        if _PACKED_Q8_NEEDED else "off"
    ),
    max_tokens=os.environ.get("K3_PACKED_Q8_MAX_T", "auto"),
    fusion_mode=os.environ.get("K3_INT8_QKV_FUSE", "auto"),
)
INT8_LM_HEAD = bool(
    _INT8_LM_HEAD_REQUESTED
    and QUANT
    and DT == torch.float32
    and _PACKED_Q8_BACKEND.available
)
if _INT8_LM_HEAD_REQUESTED and not INT8_LM_HEAD:
    print(
        "[lm-head] packed-int8 request unavailable: "
        + (
            _PACKED_Q8_BACKEND.reason
            if QUANT and DT == torch.float32
            else "requires K3_SPINE=int8 and fp32 activations"
        )
        + "; using dequantized dense weights",
        flush=True,
    )
PREFILL_LAST_LOGIT = os.environ.get("K3_PREFILL_LAST_LOGIT", "1") == "1"


class RouterTrace:
    """Router trace with a zero-work fast mode and one flush per model pass.

    The original hot path serialized and flushed 92 JSON records per pass.
    Performance runs now default to no tracing. ``K3_TRACE=buffered`` retains
    the records but writes them as one block after the layer walk; ``sync`` is
    the old crash-resilient, per-layer behavior.
    """

    def __init__(self, path, mode="off"):
        aliases = {"0": "off", "1": "buffered", "true": "buffered",
                   "false": "off", "on": "buffered"}
        self.mode = aliases.get(str(mode).lower(), str(mode).lower())
        if self.mode not in ("off", "buffered", "sync"):
            raise ValueError("K3_TRACE must be off, buffered, or sync")
        self._f = (open(path, "a", encoding="utf-8")
                   if self.mode != "off" else None)
        self._pending = []
        atexit.register(self.close)

    @property
    def enabled(self):
        return self._f is not None

    def record(self, step, layer, ids, weights):
        if self._f is None:
            return
        weight_rows = (weights if isinstance(weights, list)
                       else weights.view(-1).tolist())
        line = json.dumps({
            "step": step,
            "layer": layer,
            "ids": ids,
            "w": [round(x, 5) for row in weight_rows
                  for x in (row if isinstance(row, list) else [row])],
        }) + "\n"
        if self.mode == "sync":
            self._f.write(line)
            self._f.flush()
        else:
            self._pending.append(line)

    def end_pass(self):
        if self._f is not None and self._pending:
            self._f.writelines(self._pending)
            self._pending.clear()
            self._f.flush()

    def close(self):
        if self._f is not None:
            self.end_pass()
            self._f.close()
            self._f = None


TRACE = RouterTrace(os.path.join(ROOT, "k3-meta/router_trace.jsonl"),
                    os.environ.get("K3_TRACE", "off"))
TIMES = {"resident_io": 0.0, "expert_fetch": 0.0, "compute": 0.0, "moe_kernel": 0.0,
         "preload_wait": 0.0}   # time the main thread blocks on the preloader
PROFILE = os.environ.get("K3_PROFILE", "0") == "1"
PROF = {"kda": 0.0, "mla": 0.0, "n_kda": 0, "n_mla": 0}
EVENT_SCHEMA = "deltafin.run_event.v1"


class EventLog:
    """Optional machine-readable run evidence.

    Human stdout remains useful while debugging, but it is deliberately not the
    benchmark API: formatted text used to round every phase to whole seconds and
    could not say which tokens a speculative pass emitted. Each event carries
    both wall and monotonic nanosecond clocks and is flushed before inference
    proceeds, so a failed run still leaves its last completed phase on disk.
    """

    def __init__(self, path):
        self._f = None
        if path:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            self._f = open(path, "x", encoding="utf-8")

    def emit(self, event, **fields):
        if self._f is None:
            return
        rec = {
            "schema": EVENT_SCHEMA,
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.perf_counter_ns(),
            **fields,
        }
        self._f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        self._f.flush()

    @property
    def enabled(self):
        return self._f is not None

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


class IncrementalTokenDecoder:
    """Decode K3's token bytes once while preserving UTF-8 split boundaries."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._finished = False

    def append(self, token_id):
        if self._finished:
            raise RuntimeError("cannot append after decoder.finish()")
        model = getattr(self._tokenizer, "model", None)
        if model is None or not hasattr(model, "decode_single_token_bytes"):
            # K3 always takes the byte-exact path. This keeps the helper usable
            # with a conventional tokenizer whose pieces are already Unicode.
            return self._tokenizer.decode([int(token_id)])
        raw = model.decode_single_token_bytes(int(token_id))
        return self._utf8.decode(raw, final=False)

    def finish(self):
        if self._finished:
            return ""
        self._finished = True
        return self._utf8.decode(b"", final=True)


def _performance_env():
    prefixes = ("K3_", "PYTORCH_", "OMP_", "MKL_", "VECLIB_", "METAL_", "MLX_")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def _load_int8(full):
    """Return dequantized fp32 tensor on DEV from the int8 spine, or None.
    Under K3_SPINE=mixed this resolves through the mixed spine first (i4/i6),
    which itself falls back to these same int8 blobs."""
    if MIXED:
        return mixed_spine.load(full, INT8_DIR, k3loader.INV[full]["shape"],
                                dev=DEV, dtype=DT)
    op = os.path.join(INT8_DIR, full + ".i8")
    if not os.path.exists(op):
        return None
    shape = k3loader.INV[full]["shape"]
    q = torch.frombuffer(bytearray(open(op, "rb").read()), dtype=torch.int8).reshape(shape)
    sc = torch.frombuffer(bytearray(open(os.path.join(INT8_DIR, full + ".sc"), "rb").read()),
                          dtype=torch.float16).reshape(shape[0], 1)
    return (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)


def _load_int8_packed(full):
    """Load row-int8 weights and fp32 row scales without dequantizing.

    ``torch.from_file`` avoids materializing a second 1.17 GB Python byte string
    for the output head. The returned tensors own their device copies, so the
    short-lived CPU mappings can be dropped immediately.
    """
    op = os.path.join(INT8_DIR, full + ".i8")
    sp = os.path.join(INT8_DIR, full + ".sc")
    if not os.path.exists(op) or not os.path.exists(sp):
        return None
    rows, cols = k3loader.INV[full]["shape"]
    q = torch.from_file(
        op, shared=False, size=rows * cols, dtype=torch.int8
    ).reshape(rows, cols).to(DEV)
    sc = torch.from_file(
        sp, shared=False, size=rows, dtype=torch.float16
    ).to(device=DEV, dtype=torch.float32)
    return q, sc


def _lm_head_forward(hidden):
    """Capability-gated native-int8 head with a one-time dense fallback."""
    global _LM_Q, _LM_SC, _LM_W, _LM_PACKED_CALLS, _LM_DENSE_CALLS
    global _LM_DENSE_CROSSOVERS, _LM_DISABLE_REASON
    flat = hidden.reshape(-1, H)
    if _LM_Q is not None:
        if _PACKED_Q8_BACKEND.supports_tokens(flat.shape[0]):
            try:
                output = _PACKED_Q8_BACKEND.matmul(
                    flat, _LM_Q, _LM_SC
                )
                _LM_PACKED_CALLS += 1
                return output.view(*hidden.shape[:-1], -1)
            except (NotImplementedError, RuntimeError, TypeError) as exc:
                # A newer/older PyTorch build may expose the schema without a
                # working backend for this device/shape. Preserve support
                # rather than making a product-name assumption.
                _LM_DISABLE_REASON = f"{type(exc).__name__}: {exc}"
                print(f"[lm-head] native int8 unavailable ({exc}); "
                      "falling back to dequantized dense weights", flush=True)
        else:
            _LM_DENSE_CROSSOVERS += 1
            _LM_DISABLE_REASON = (
                f"activation rows {flat.shape[0]} exceed measured packed-q8 "
                f"limit {_PACKED_Q8_BACKEND.max_tokens}"
            )
            print(
                f"[lm-head] {_LM_DISABLE_REASON}; materializing dense head",
                flush=True,
            )
        # Materialize before dropping the owned packed tensors, so a failure
        # cannot leave the head with neither representation.
        dense = _LM_Q.to(torch.float32) * _LM_SC[:, None]
        _LM_W = dense
        _LM_Q = _LM_SC = None
    _LM_DENSE_CALLS += 1
    return hidden @ _LM_W.T


def _lm_head_runtime_status():
    return {
        "requested": _INT8_LM_HEAD_REQUESTED,
        "eligible": INT8_LM_HEAD,
        "packed_resident": _LM_Q is not None,
        "dense_resident": _LM_W is not None,
        "packed_calls": _LM_PACKED_CALLS,
        "dense_calls": _LM_DENSE_CALLS,
        "dense_crossovers": _LM_DENSE_CROSSOVERS,
        "disable_reason": _LM_DISABLE_REASON,
        "backend": _PACKED_Q8_BACKEND.status(),
    }


def materialize_resident(module, prefix):
    t0 = time.time()
    missing = []
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue  # routed experts stay meta until selected
        full = prefix + name
        t = _load_int8(full) if QUANT else None
        if t is None:
            try:
                t = k3loader.load_resident(full).to(DEV, DT)
            except KeyError:
                missing.append(full)
                continue
        set_param(module, name, t)
    TIMES["resident_io"] += time.time() - t0
    if missing:
        raise RuntimeError(f"missing resident tensors: {missing[:5]}")


def dematerialize(module):
    for name, p in list(module.named_parameters()):
        if p.device.type != "meta":
            set_param(module, name, torch.empty_like(p, device="meta"))


_TAIL = None


def _new_tail():
    tail = nn.Module()
    with torch.device("meta"):
        tail.output_attn_res_norm = ml.KimiRMSNorm(
            H, eps=config.rms_norm_eps)
        tail.output_attn_res_proj = nn.Linear(H, 1, bias=False)
        tail.norm = ml.KimiRMSNorm(H, eps=config.rms_norm_eps)
    materialize_resident(tail, PFX)
    return tail


def _tail_module():
    """Keep the tiny final norms/AttnRes projection resident across passes."""
    global _TAIL
    if _TAIL is None:
        _TAIL = _new_tail()
    return _TAIL


# --- double-buffered layer loading: a worker thread reads layer N+1's blobs
# (file I/O releases the GIL) while layer N computes; main thread does the
# tensor creation / dequant / device transfer ---------------------------------
import concurrent.futures as _cf  # noqa: E402
_PRELOADER = _cf.ThreadPoolExecutor(1)
PRELOAD = os.environ.get("K3_PRELOAD", "1") == "1"


# --- adaptive RAM budget: pin as many resident layers as this machine affords --
# budget = total RAM - OS/apps reserve (max(10GB, 18%)); ~40% of the spendable
# budget goes to pinned layers (loaded once, never freed); the rest is left to
# the page cache, which holds hot expert .bin files and self-scales with RAM.
# Template-layer buffer reuse: all KDA layers share one shape class, all MLA
# layers another. Two persistent materialized templates + copy_() per layer
# kills the MPS alloc/free churn measured at 1317 -> 288 ms/layer.
TEMPLATES = os.environ.get("K3_TEMPLATES", "1") == "1"
# A positive explicit pin request used to be silently ignored by the default
# template mode. Honor the request by selecting private per-layer modules.
# The default remains templates/no private pins on the current M1 path; larger
# machines can opt into pins without a product-name branch.
_PIN_LAYERS_RAW = os.environ.get("K3_PIN_LAYERS")
if TEMPLATES and _PIN_LAYERS_RAW not in (None, "", "0"):
    if int(_PIN_LAYERS_RAW) > 0:
        TEMPLATES = False
        print(
            "[config] K3_PIN_LAYERS requests private resident layers; "
            "disabling shared templates for this explicit experiment",
            flush=True,
        )
# The dense, KDA+MoE, and MLA+MoE templates execute serially and are populated
# immediately before use. Overlay their parameter views on one max-sized MPS
# arena instead of keeping all three allocations resident (N2).
TEMPLATE_ARENA_MODE = os.environ.get("K3_TEMPLATE_ARENA", "auto").lower()
TEMPLATE_ARENA = (
    TEMPLATES and TEMPLATE_ARENA_MODE not in apple_silicon.FALSE_MODES
)
_TEMPLATE_ARENA_STORAGE = None
_TEMPLATE_ARENA_INFO = {}

# --- fast resident-spine path (K3_FAST_SPINE=1, default off) ------------------
# Packed readinto + one H2D per layer + a bit-exact Metal dequant kernel.
# See tools/spine_fast.py for the measurements that motivate each piece.
import spine_fast  # noqa: E402
import spine_io  # noqa: E402

_SPINE_DEQ = spine_fast.effective_dequant_backend(DEV)
if DEV.type == "mps" and _SPINE_DEQ == "metal":
    spine_fast.metal_available()          # compile once, on the main thread
if spine_fast.FAST or _SPINE_DEQ != "torch":
    print(f"[spine] fast path: {spine_fast.describe(DEV)}", flush=True)

# N1: keep every eligible same-input KDA first-stage projection as row-int8 and
# call a capability-proven native weight-only matmul instead of dequantizing
# Q/K/V/G/F-A/B into the shared fp32 template arena. The historical QKV
# environment/API name is retained for compatibility. Capability and exception
# gates retain the dense path everywhere else.
# Eligibility is based on runtime capabilities, never an OS/product name.
_INT8_KDA_QKV_REQUESTED = _INT8_KDA_QKV_REQUESTED_EARLY
_INT8_KDA_QKV_REASONS = []
if SPINE != "int8":
    _INT8_KDA_QKV_REASONS.append("requires K3_SPINE=int8")
if DT != torch.float32:
    _INT8_KDA_QKV_REASONS.append("requires fp32 activations")
if not _PACKED_Q8_BACKEND.available:
    _INT8_KDA_QKV_REASONS.append(
        _PACKED_Q8_BACKEND.reason
        or "no usable packed row-int8 backend"
    )
if not TEMPLATES:
    _INT8_KDA_QKV_REASONS.append("requires shared templates")
if not spine_fast.PACK:
    _INT8_KDA_QKV_REASONS.append("requires K3_SPINE_PACK=1")
if _INT8_KDA_STORAGE_MODE == "stage":
    _stage_capable, _stage_reason = spine_fast.stage_storage_capability(
        DEV, _INT8_KDA_STAGE_SYNC_MODE
    )
    if not _stage_capable:
        _INT8_KDA_QKV_REASONS.append(
            _stage_reason or "direct staging lifetime cannot be ordered"
        )
INT8_KDA_QKV = _INT8_KDA_QKV_REQUESTED and not _INT8_KDA_QKV_REASONS


def _int8_kda_qkv_disabled(reason):
    print(
        f"[kda-qkv] native packed-int8 unavailable ({reason}); "
        "falling back to dequantized dense projections",
        flush=True,
    )


_INT8_KDA_QKV_STATE = (
    spine_fast.DynamicQ8State(_int8_kda_qkv_disabled)
    if INT8_KDA_QKV else None
)
_INT8_KDA_QKV_CONTROLLERS = []
if _INT8_KDA_QKV_REQUESTED and not INT8_KDA_QKV:
    print(
        "[kda-qkv] packed-int8 request unavailable: "
        + "; ".join(_INT8_KDA_QKV_REASONS)
        + "; using dequantized dense projections",
        flush=True,
    )


def _int8_kda_qkv_runtime_status():
    """Report what actually happened, not just the pre-build capability gate."""
    state = _INT8_KDA_QKV_STATE
    controllers = _INT8_KDA_QKV_CONTROLLERS
    packed_project_calls = sum(
        getattr(controller, "packed_project_calls", 0)
        for controller in controllers
    )
    packed_operator_calls = sum(
        getattr(controller, "packed_operator_calls", 0)
        for controller in controllers
    )
    return {
        "requested": _INT8_KDA_QKV_REQUESTED,
        "eligible": INT8_KDA_QKV,
        "storage_mode": _INT8_KDA_STORAGE_MODE,
        "stage_sync_mode": _INT8_KDA_STAGE_SYNC_MODE,
        "controllers_installed": len(controllers),
        "enabled_at_end": bool(
            INT8_KDA_QKV
            and state is not None
            and state.enabled
            and controllers
        ),
        "packed_project_calls": packed_project_calls,
        "packed_operator_calls": packed_operator_calls,
        "fused_operator_calls": sum(
            getattr(controller, "fused_operator_calls", 0)
            for controller in controllers
        ),
        "separate_operator_calls": sum(
            getattr(controller, "separate_operator_calls", 0)
            for controller in controllers
        ),
        "dense_crossover_project_calls": sum(
            getattr(controller, "dense_crossover_project_calls", 0)
            for controller in controllers
        ),
        "dense_crossover_materializations": sum(
            getattr(controller, "dense_crossover_materializations", 0)
            for controller in controllers
        ),
        "dense_crossover_releases": sum(
            getattr(controller, "dense_crossover_releases", 0)
            for controller in controllers
        ),
        "runtime_failures": sum(
            getattr(controller, "runtime_failures", 0)
            for controller in controllers
        ),
        "persistent_weight_bytes": sum(
            getattr(controller, "persistent_weight_bytes", 0)
            for controller in controllers
        ),
        "persistent_scale_bytes": sum(
            getattr(controller, "persistent_scale_bytes", 0)
            for controller in controllers
        ),
        "stage_bind_count": sum(
            getattr(controller, "stage_bind_count", 0)
            for controller in controllers
        ),
        "stage_bind_failures": sum(
            getattr(controller, "stage_bind_failures", 0)
            for controller in controllers
        ),
        "stage_stale_rejections": sum(
            getattr(controller, "stage_stale_rejections", 0)
            for controller in controllers
        ),
        "stage_weight_copy_bytes": sum(
            getattr(controller, "stage_weight_copy_bytes", 0)
            for controller in controllers
        ),
        "stage_scale_copy_bytes": sum(
            getattr(controller, "stage_scale_copy_bytes", 0)
            for controller in controllers
        ),
        "stage_fence_records": sum(
            getattr(controller, "stage_fence_records", 0)
            for controller in controllers
        ),
        "stage_fence_waits": sum(
            getattr(controller, "stage_fence_waits", 0)
            for controller in controllers
        ),
        "stage_fence_sync_fallbacks": sum(
            getattr(controller, "stage_fence_sync_fallbacks", 0)
            for controller in controllers
        ),
        "stage_fence_failures": sum(
            getattr(controller, "stage_fence_failures", 0)
            for controller in controllers
        ),
        "stage_fifo_records": sum(
            getattr(controller, "stage_fifo_records", 0)
            for controller in controllers
        ),
        "stage_fifo_reuses": sum(
            getattr(controller, "stage_fifo_reuses", 0)
            for controller in controllers
        ),
        "stage_full_shape_probes": sum(
            getattr(controller, "stage_full_shape_probes", 0)
            for controller in controllers
        ),
        "stage_full_shape_passes": sum(
            getattr(controller, "stage_full_shape_passes", 0)
            for controller in controllers
        ),
        "backend": _PACKED_Q8_BACKEND.status(),
        "controllers": [
            controller.status() for controller in controllers
        ],
        "disable_reason": state.reason if state is not None else None,
    }


# --- attention / norm fast paths (K3_KDA_RECUR, K3_SHORTCONV, K3_COMPILE) -----
# All default to the behaviour above; see tools/attn_fast.py for the per-op
# measurements that motivate each one.
import attn_fast  # noqa: E402

attn_fast.install(ml)
if attn_fast.ACTIVE:
    print(f"[attn] {attn_fast.describe(DEV)}", flush=True)


def _ram_budget_layers():
    if TEMPLATES:
        return 0  # one shared module cannot hold different layers concurrently
    if os.environ.get("K3_PIN_LAYERS") is not None:
        return int(os.environ["K3_PIN_LAYERS"])
    explicit_budget = float(os.environ.get("K3_RAM_GB", 0))
    if explicit_budget < 0:
        raise ValueError("K3_RAM_GB must be non-negative")

    linux_memory = None
    if sys.platform.startswith("linux"):
        linux_memory = runtime_platform.linux_memory_limits()
        total_bytes = linux_memory.effective_total_bytes
    else:
        total_bytes = APPLE_CAPS.physical_memory_bytes or 0
    if not total_bytes:
        print("[ram] physical/cgroup memory limit unavailable; pinning no layers",
              flush=True)
        return 0
    total_gb = total_bytes / 2**30
    reserve = max(10.0, 0.18 * total_gb)
    reserve_bytes = int(reserve * 2**30)
    if linux_memory is not None:
        safe_bytes = runtime_platform.safe_linux_host_budget(
            linux_memory, reserve_bytes
        )
    else:
        safe_bytes = APPLE_CAPS.safe_unified_budget(
            host_reserve_bytes=reserve_bytes,
        )
    safe_gb = safe_bytes / 2**30
    if explicit_budget and linux_memory is not None:
        # An explicit target may tune within the safe envelope, but cannot
        # escape a container/cgroup limit or consume another process's RAM.
        budget = min(explicit_budget, safe_gb)
    else:
        budget = explicit_budget or safe_gb

    cuda_note = ""
    if DEV.type == "cuda":
        try:
            cuda_free, cuda_total = torch.cuda.mem_get_info(DEV)
        except (RuntimeError, TypeError) as exc:
            print(f"[ram] CUDA free-memory query failed ({exc}); "
                  "pinning no private layers", flush=True)
            return 0
        cuda_cap_gb = runtime_platform.cuda_free_memory_budget(
            cuda_free, cuda_total
        ) / 2**30
        budget = min(budget, cuda_cap_gb)
        cuda_note = (
            f", CUDA {cuda_free/2**30:.1f}/{cuda_total/2**30:.1f} GB free/total "
            f"-> {cuda_cap_gb:.1f} GB cap"
        )

    head_gb = 1.18 if INT8_LM_HEAD else (4.7 if DT == torch.float32 else 2.35)
    overhead = 8.0 + head_gb + 2.0   # process + lm_head + transients
    per_layer = (113.5 / NL) * (2 if DT == torch.float32 else 1)    # fp32=2x int8 bytes, fp16=1x
    n = max(0, int(0.4 * (budget - overhead) / per_layer))
    budget_source = (
        f"explicit budget {budget:.1f} GB (safe query: {safe_gb:.1f} GB)"
        if explicit_budget
        else f"safe budget {safe_gb:.1f} GB"
    )
    scope = "effective host/cgroup" if linux_memory is not None else "total"
    print(f"[ram] {scope} {total_gb:.0f} GB, {budget_source}{cuda_note} -> pinning "
          f"{min(n, NL)} of {NL} layers ({min(n, NL) * per_layer:.1f} GB at {DT})", flush=True)
    return min(n, NL)


PIN_N = _ram_budget_layers()


def _read_resident_bytes(module, prefix):
    out = {}
    for name, _ in module.named_parameters():
        if ".experts." in name:
            continue
        full = prefix + name
        if SPINE == "int8":
            op = os.path.join(INT8_DIR, full + ".i8")
            if os.path.exists(op):
                out[full] = ("i8", open(op, "rb").read(),
                             open(os.path.join(INT8_DIR, full + ".sc"), "rb").read())
                continue
        path = os.path.join(k3loader.RES, full)
        if os.path.exists(path):
            out[full] = ("bf16", open(path, "rb").read())
    return out


def _apply_resident(module, prefix, blobs):
    t0 = time.time()
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue
        full = prefix + name
        rec = blobs.get(full)
        if rec is None:
            t = k3loader.load_resident(full).to(DEV, DT)
        elif rec[0] == "i8":
            shape = k3loader.INV[full]["shape"]
            q = torch.frombuffer(bytearray(rec[1]), dtype=torch.int8).reshape(shape)
            sc = torch.frombuffer(bytearray(rec[2]), dtype=torch.float16).reshape(shape[0], 1)
            t = (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)
        else:
            meta = k3loader.INV[full]
            t = torch.frombuffer(bytearray(rec[1]),
                                 dtype=k3loader._DT[meta["dtype"]]).reshape(meta["shape"]).to(DEV, DT)
        set_param(module, name, t)
    TIMES["resident_io"] += time.time() - t0


# --- MoE expert lazy materialization + router trace ---------------------------
_step_ctx = {"layer": -1, "step": -1}
_orig_moe_infer = ml.KimiSparseMoeBlock.moe_infer


FAST_MOE = os.environ.get("K3_FAST_MOE", "1") == "1"
fast_moe = runtime_platform.import_when_enabled(FAST_MOE, "fast_moe")

# --- MoE compute backend (K3_MOE=cpu|metal|cuda) -----------------------------
# cpu   : tools/fast_moe_batch.py when its native library is present, otherwise
#         tools/fast_moe.py. The batch path keeps a persistent worker ring and
#         is bit-identical; K3_CPU_MOE_BATCH=0 retains the legacy fallback.
# metal : tools/metal_moe.py, the whole layer's selected experts as one GPU
#         command buffer. Same signature, same semantics; matched the CPU path to
#         2.5e-7 on real experts. Falls back to cpu (loudly) if Metal is missing.
# cuda  : tools/cuda_moe.py after a versioned ABI/shape check and an on-device
#         known-answer test. Expert residency can then avoid both disk reads and
#         host-to-device copies. Any qualification/runtime failure disables the
#         device and replays the complete route through the established CPU path.
# K3_MOE_CHECK=N cross-checks the first N calls against the CPU kernel and raises
# on disagreement — see tools/metal_moe.py. K3_METAL_BINDLESS=0 picks the
# per-expert dispatch mode instead of the Tier-2 argument-buffer one.
_MOE_BACKEND_EXPLICIT = os.environ.get("K3_MOE")
MOE_BACKEND = runtime_platform.choose_moe_backend(
    _MOE_BACKEND_EXPLICIT, DEV.type
)
_MOE_FN = fast_moe.moe_infer_fast if fast_moe is not None else None
CPU_BATCH_ACTIVE = False
_CPU_BATCH_MODE = os.environ.get("K3_CPU_MOE_BATCH", "auto").strip().lower()
if FAST_MOE and _CPU_BATCH_MODE not in ("0", "off", "false", "no"):
    try:
        import fast_moe_batch  # noqa: E402
        workers = fast_moe_batch.pool_init()
        if workers <= 0:
            raise RuntimeError("native worker pool created no workers")
        _MOE_FN = fast_moe_batch.moe_infer_fast
        CPU_BATCH_ACTIVE = True
    except Exception as exc:
        # The batch dylib is an optional acceleration artifact. A source-only
        # checkout or a future CPU can always retain the established kernel.
        print(f"[config] persistent CPU MoE ring unavailable "
              f"({type(exc).__name__}: {exc}); using legacy CPU GEMV", flush=True)
_CPU_MOE_FN = _MOE_FN
cuda_moe = None
_CUDA_MOE_ACTIVE = False
_CUDA_MOE_FALLBACK_REPORTED = False
_CUDA_MODEL_KEY = f"{os.path.realpath(k3loader.RES)}:{H}:{NL}"


def _disable_cuda_moe(reason):
    """Fail closed once, retaining a complete and independently tested path."""
    global _CUDA_MOE_ACTIVE, _CUDA_MOE_FALLBACK_REPORTED
    global MOE_BACKEND, _MOE_FN, FAST_MOE
    if cuda_moe is not None:
        try:
            cuda_moe.disable(DEV, str(reason))
        except Exception:
            pass
    _CUDA_MOE_ACTIVE = False
    MOE_BACKEND = "cpu"
    _MOE_FN = _CPU_MOE_FN
    FAST_MOE = _CPU_MOE_FN is not None
    if not _CUDA_MOE_FALLBACK_REPORTED:
        print(f"[config] CUDA MoE disabled ({reason}); "
              "falling back to the established CPU expert path", flush=True)
        _CUDA_MOE_FALLBACK_REPORTED = True


def _activate_cuda_moe():
    """Qualify the optional CUDA bridge without weakening any kill switch."""
    global cuda_moe, _MOE_FN, _CUDA_MOE_ACTIVE, FAST_MOE
    if not FAST_MOE:
        _disable_cuda_moe("K3_FAST_MOE=0")
    elif DEV.type != "cuda":
        _disable_cuda_moe(f"selected device is {DEV}, not CUDA")
    elif DT != torch.float32:
        # The native ABI consumes fp32. In particular, never reinterpret an
        # approximate-mode fp16 pointer as fp32.
        _disable_cuda_moe(
            f"native CUDA MoE requires fp32 input, current dtype is {DT}")
    else:
        try:
            import cuda_moe as _cuda_moe  # noqa: E402
            cuda_moe = _cuda_moe
            if not cuda_moe.available(DEV):
                raise RuntimeError(cuda_moe.describe(DEV))
            _MOE_FN = cuda_moe.moe_infer
            _CUDA_MOE_ACTIVE = True
            FAST_MOE = True       # CUDA consumes raw MXFP4, never dequantized
            print(f"[config] MoE backend: {cuda_moe.describe(DEV)}", flush=True)
        except Exception as exc:
            _disable_cuda_moe(f"{type(exc).__name__}: {exc}")


if MOE_BACKEND == "cuda":
    _activate_cuda_moe()
if MOE_BACKEND == "metal":
    import metal_moe  # noqa: E402
    if metal_moe.available():
        _MOE_FN = metal_moe.moe_infer
        FAST_MOE = True          # metal consumes raw MXFP4, never dequantized
        print(f"[config] MoE backend: metal "
              f"({'bindless' if metal_moe.stats()['bindless'] else 'per-expert'}, "
              f"pin_max={metal_moe.PIN_MAX}, check={metal_moe.CHECK})", flush=True)
    else:
        MOE_BACKEND = "cpu"
        print(f"[config] K3_MOE=metal unavailable ({metal_moe.last_error()}) "
              f"— falling back to cpu", flush=True)
if MOE_BACKEND == "cpu" and CPU_BATCH_ACTIVE:
    print(f"[config] CPU MoE: persistent worker ring "
          f"({fast_moe_batch.pool_threads()} threads, "
          f"{fast_moe_batch.native_isa()})", flush=True)
elif MOE_BACKEND == "cpu" and fast_moe is not None:
    print(f"[config] CPU MoE: legacy per-call worker pool "
          f"({fast_moe.native_isa()})", flush=True)

fetch_v2 = None
if os.environ.get("K3_FETCH", "v2") == "v2":
    import fetch_v2
    k3loader.fetch_experts = fetch_v2.fetch_experts  # 6.4x: coalesced + keep-alive
    fetch_v2.set_cache_observer(k3loader.register_cache_file)
    k3loader.set_runtime_stats(fetch_v2.stats)

# K3_EXPERT_READ=pread (see tools/fetch_v2.py) reads the layer's whole selected
# set through a threaded pread pool instead of demand-faulting mmap pages inside
# the GEMV kernel. K3_EXPERT_PREFETCH=1 additionally starts layer L+1's reads
# from the previous token's selections while layer L computes — separate flag,
# because it is speculative (39.7% measured recall) and costs wasted bandwidth.
PREAD = fetch_v2 is not None and fetch_v2.EXPERT_READ == "pread"
EXPERT_PREFETCH = PREAD and os.environ.get("K3_EXPERT_PREFETCH", "0") == "1"

# K3_PILOT=1 replaces that previous-token oracle with a router-lookahead
# prediction: layer L+1's router run on layer L's pre-MoE hidden state. See
# tools/pilot.py. Default 0 = nothing below changes.
import pilot  # noqa: E402
import grouped_moe  # noqa: E402

GROUPED_MOE_ACTIVE = (
    grouped_moe.enabled()
    and MOE_BACKEND == "metal"
    and PREAD
)
if grouped_moe.enabled():
    if GROUPED_MOE_ACTIVE:
        print(f"[config] grouped Metal MoE: {grouped_moe.describe()}", flush=True)
    else:
        print("[config] K3_MOE_GROUP_SIZE requested but requires "
              "K3_MOE=metal, K3_FETCH=v2, and K3_EXPERT_READ=pread; "
              "using the established path", flush=True)

_LAST_SEL = {}   # layer -> ids selected for the most recent token (prefetch oracle)
_PREV_SEL = {}   # snapshot of _LAST_SEL taken when the current pass started


def prefetch_prev_token():
    """Fire-and-forget: fetch the previous token's full per-layer expert sets
    (39.7% measured next-token recall); misses stream to disk while layers compute."""
    if PREAD:
        # Under the pread path this would issue 25.8 GB of real reads from a
        # background thread, fighting the foreground layer for the same disk.
        # The per-layer K3_EXPERT_PREFETCH hook replaces it.
        return
    import threading
    snap = dict(_LAST_SEL)

    def run():
        for li in sorted(snap):
            try:
                ids = snap[li]
                if _CUDA_MOE_ACTIVE:
                    ids = cuda_moe.missing_experts(
                        li, ids, DEV, model_key=_CUDA_MODEL_KEY)
                if ids:
                    k3loader.fetch_experts(li, ids, dequant=False)
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()


# Expert reads are the term that grows with the speculative batch: a T=D+1 pass
# reads the UNION of every position's selection, so `uniq/pos` below is the
# sublinearity factor the depth cost model turns on. 1.00 = free, T = worst case.
EXPERT_SEL = {"layer_calls": 0, "uniq": 0, "pos": 0}


def _issue_next_expert_prefetch(li):
    """Issue speculation only after this layer's demand reads have landed."""
    uncached = None
    if _CUDA_MOE_ACTIVE:
        def uncached(layer, candidates):
            try:
                return cuda_moe.missing_experts(
                    layer, candidates, DEV, model_key=_CUDA_MODEL_KEY)
            except Exception:
                # Prefetch is optional and runs before the current layer's
                # backend call. Do not change backend state here: returning
                # the full list is always safe, and the demand path remains
                # the qualification gate.
                return candidates

    if pilot.enabled() and fetch_v2 is not None:
        if uncached is None:
            pilot.issue_prefetch(li + 1, fetch_v2, pread=PREAD)
        else:
            pilot.issue_prefetch(
                li + 1, fetch_v2, pread=PREAD, filter_ids=uncached)
    elif EXPERT_PREFETCH:
        nxt = _PREV_SEL.get(li + 1)
        if nxt:
            nxt = (
                uncached(li + 1, list(nxt))
                if uncached is not None else nxt
            )
            if nxt:
                fetch_v2.prefetch_layer(li + 1, nxt)


def _slow_moe_infer(self, x, topk_ids, topk_weight, raw, ids):
    """Run the framework reference and return every expert to meta storage."""
    try:
        for e, weights in raw.items():
            ex = self.experts[e]
            for wn in ("w1", "w2", "w3"):
                # Dequantized expert tensors are born on the CPU. The reference
                # fallback must follow the resident activation to MPS/CUDA
                # rather than installing cross-device parameters.
                set_param(
                    ex,
                    wn + ".weight",
                    weights[wn].to(device=x.device, dtype=x.dtype),
                )
        return _orig_moe_infer(self, x, topk_ids, topk_weight)
    finally:
        for e in ids:
            for wn in ("w1", "w2", "w3"):
                set_param(self.experts[e], wn + ".weight",
                          torch.empty(0, device="meta"))


def moe_infer_lazy(self, x, topk_ids, topk_weight):
    li = _step_ctx["layer"]
    rows = topk_ids.tolist()                    # [positions][top_k]
    # Every fast raw-MXFP4 backend consumes the same ordered CPU route that
    # fetch scheduling already needs. This extends X12 to CPU expert fallback
    # on CPU/CUDA/ROCm accelerator-spine runs without any device/product branch.
    # When fast MoE is disabled, do not materialize weights here: TRACE=off and
    # the established slow expert path retain their prior zero-extra-work flow.
    routing_record = (
        routing_records.materialize(
            topk_ids, topk_weight, ids=rows)
        if FAST_MOE else None
    )
    flat = [e for r in rows for e in r]
    ids = sorted(set(flat))
    EXPERT_SEL["layer_calls"] += 1
    EXPERT_SEL["uniq"] += len(ids)
    EXPERT_SEL["pos"] += len(rows)
    _LAST_SEL[li] = ids
    if pilot.enabled():
        pilot.on_actual(li, rows)               # score the prediction made at li-1
    if GROUPED_MOE_ACTIVE:
        grouped = grouped_moe.try_infer(
            x, routing_record, li, fetch_v2, metal_moe)
        if grouped is not None:
            out, timing = grouped
            TIMES["expert_fetch"] += timing.get("fetch_wait_s", 0.0)
            TIMES["moe_kernel"] += timing.get("kernel_s", 0.0)
            _issue_next_expert_prefetch(li)
            TRACE.record(_step_ctx["step"], li, flat,
                         routing_record["weights"])
            return out
    fetch_ids = ids
    if _CUDA_MOE_ACTIVE:
        try:
            fetch_ids = cuda_moe.missing_experts(
                li, ids, DEV, model_key=_CUDA_MODEL_KEY)
        except Exception as exc:
            _disable_cuda_moe(
                f"cache query failed: {type(exc).__name__}: {exc}")
            fetch_ids = ids
    t0 = time.time()
    raw = (
        k3loader.fetch_experts(li, fetch_ids, dequant=not FAST_MOE)
        if fetch_ids else {}
    )
    TIMES["expert_fetch"] += time.time() - t0
    # Speculative reads are issued only AFTER this layer's demand reads have
    # landed: the pread pool is FIFO, so a prefetch queued first would put the
    # next layer's speculation in front of this layer's blocking reads.
    _issue_next_expert_prefetch(li)
    TRACE.record(_step_ctx["step"], li, flat,
                 routing_record["weights"] if routing_record else topk_weight)
    if FAST_MOE:
        try:
            # CUDA launches are asynchronous. Synchronize only while profiling;
            # normal decoding keeps the established overlap opportunity.
            if _CUDA_MOE_ACTIVE and PROFILE:
                _device_synchronize()
            tk = time.time()
            if _CUDA_MOE_ACTIVE:
                out = _MOE_FN(
                    x, topk_ids, topk_weight, raw,
                    routing_record=routing_record,
                    layer_index=li,
                    model_key=_CUDA_MODEL_KEY)
            else:
                out = _MOE_FN(
                    x, topk_ids, topk_weight, raw,
                    routing_record=routing_record)
            if _CUDA_MOE_ACTIVE and PROFILE:
                _device_synchronize()
            TIMES["moe_kernel"] += time.time() - tk
            return out
        except Exception as exc:
            if not _CUDA_MOE_ACTIVE:
                raise
            # `raw` can contain only cache misses. It is never safe to hand
            # that partial set to the CPU backend, so fetch the whole route.
            _disable_cuda_moe(
                f"runtime failure: {type(exc).__name__}: {exc}")
            t_retry = time.time()
            if _CPU_MOE_FN is not None:
                raw = k3loader.fetch_experts(li, ids, dequant=False)
            else:
                raw = k3loader.fetch_experts(li, ids, dequant=True)
            TIMES["expert_fetch"] += time.time() - t_retry
            tk = time.time()
            if _CPU_MOE_FN is not None:
                out = _CPU_MOE_FN(
                    x, topk_ids, topk_weight, raw,
                    routing_record=routing_record)
            else:
                out = _slow_moe_infer(
                    self, x, topk_ids, topk_weight, raw, ids)
            TIMES["moe_kernel"] += time.time() - tk
            return out
    return _slow_moe_infer(self, x, topk_ids, topk_weight, raw, ids)


ml.KimiSparseMoeBlock.moe_infer = moe_infer_lazy

# Router lookahead hooks the MoE block's entry, which is the one point in the
# graph that sits after layer L's attention and before any expert read.
_orig_moe_forward = ml.KimiSparseMoeBlock.forward


def moe_forward_pilot(self, hidden_states):
    if pilot.enabled():
        pilot.on_moe_entry(self, hidden_states, _step_ctx["layer"])
    return _orig_moe_forward(self, hidden_states)


ml.KimiSparseMoeBlock.forward = moe_forward_pilot

# --- multi-token speculation (K3_SPEC_DEPTH, default 1 = unchanged) -----------
# Drafts D tokens and verifies all D+1 positions in ONE forward pass, so the
# 53 GB spine read is amortised over up to D+1 emitted tokens. The hooks below
# are inert while capture is disarmed, which is always the case at depth 1 and
# during prefill. See tools/spec_decode.py for the partial-accept rollback
# argument (the crux) and the numerics-parity note.
import spec_decode  # noqa: E402

spec_decode.install(ml, lambda: _step_ctx["layer"],
                    conv_kernel_size=config.linear_attn_config["short_conv_kernel_size"],
                    compiled=attn_fast.COMPILE != "0")
if spec_decode.enabled():
    print(f"[spec] {spec_decode.describe()}", flush=True)


def _pilot_load(full):
    """The resident loader the layer templates use — so a cached gate is the
    exact tensor the model routes with (int8-dequantized under K3_SPINE=int8)."""
    t = _load_int8(full) if QUANT else None
    if t is None:
        t = k3loader.load_resident(full).to(DEV, DT)
    return t


# --- embeddings via memmap (row reads only) -----------------------------------
class LazyEmbed:
    """bf16 embedding rows from a persistent local file, else HTTP Range."""
    NAME = PFX + "embed_tokens.weight"

    def __init__(self):
        self.path = os.path.join(ROOT, "k3-resident/tensors", self.NAME)
        self.meta = k3loader.INV[self.NAME]
        self.rowbytes = H * 2
        self._source = None
        self._ensure_source()

    def _ensure_source(self):
        if self._source is None:
            try:
                self._source = positional_io.open_positional(self.path)
            except FileNotFoundError:
                pass
        return self._source

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def _local_rows(self, tids):
        """Read sorted unique ids, coalescing adjacent rows into one read."""
        rows = {}
        uniq = sorted(set(tids))
        i = 0
        while i < len(uniq):
            j = i + 1
            while j < len(uniq) and uniq[j] == uniq[j - 1] + 1:
                j += 1
            first, count = uniq[i], j - i
            want = count * self.rowbytes
            try:
                data = self._source.read(want, first * self.rowbytes)
            except OSError as exc:
                raise IOError(f"short embedding read of {want} bytes") from exc
            for k, tid in enumerate(uniq[i:j]):
                lo = k * self.rowbytes
                rows[tid] = data[lo:lo + self.rowbytes]
            i = j
        return b"".join(rows[tid] for tid in tids)

    def _row(self, tid):
        if self._ensure_source() is not None:
            try:
                return self._source.read(self.rowbytes, tid * self.rowbytes)
            except OSError as exc:
                raise IOError(
                    f"short embedding read of {self.rowbytes} bytes"
                ) from exc
        m = self.meta
        start = 8 + m["hlen"] + m["offsets"][0] + tid * self.rowbytes
        import urllib.request
        req = urllib.request.Request(
            k3loader.BASE + m["shard"],
            headers={"Range": f"bytes={start}-{start+self.rowbytes-1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def __call__(self, ids):
        tids = [int(t) for t in ids]
        buf = (self._local_rows(tids) if self._ensure_source() is not None
               else b"".join(self._row(tid) for tid in tids))
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(len(tids), H)
        return t.to(DEV, DT).unsqueeze(0)  # [1, T, H]


def build_layers():
    global _INT8_KDA_QKV_CONTROLLERS
    layers = []
    if not TEMPLATES:
        with torch.device("meta"):
            for i in range(NL):
                layers.append(ml.KimiDecoderLayer(config, i).eval())
        return layers
    with torch.device("meta"):
        l0 = ml.KimiDecoderLayer(config, 0).eval()        # dense KDA (unique shape)
        tpl_kda = ml.KimiDecoderLayer(config, 1).eval()   # KDA + MoE class
        tpl_mla = ml.KimiDecoderLayer(config, 3).eval()   # MLA + MoE class
    if INT8_KDA_QKV and _INT8_KDA_QKV_STATE.enabled:
        installed = []
        try:
            # The reusable KDA template executes 68 layers per pass. Layer 0
            # executes once and stays on the dense fallback: giving it another
            # 337.7 MiB packed arena is a poor memory/I/O trade, while attempting
            # to register one MPS arena on both modules caused the original
            # full-model candidate to disable itself before its first call.
            # A future backend with an explicitly supported shared-storage
            # contract can reconsider layer 0 independently.
            if tpl_kda.is_linear_attn:
                installed.append(
                    spine_fast.install_dynamic_q8_qkv(
                        tpl_kda,
                        DEV,
                        _INT8_KDA_QKV_STATE,
                        _PACKED_Q8_BACKEND,
                        storage_mode=_INT8_KDA_STORAGE_MODE,
                        stage_sync_mode=_INT8_KDA_STAGE_SYNC_MODE,
                    )
                )
        except (RuntimeError, ValueError, MemoryError) as exc:
            for controller in reversed(installed):
                controller.uninstall()
            _INT8_KDA_QKV_STATE.disable(exc)
        else:
            _INT8_KDA_QKV_CONTROLLERS = installed
            packed_bytes = sum(
                controller.nbytes for controller in installed
            )
            print(
                f"[kda-bundle] native packed-int8 active for "
                f"{len(installed)} KDA template(s), "
                f"{packed_bytes/2**20:.1f} MiB persistent storage, "
                f"device={DEV.type}, max_T={_PACKED_Q8_BACKEND.max_tokens}, "
                f"fused_bundle={int(_PACKED_Q8_BACKEND.fuse_qkv)}, "
                f"storage={_INT8_KDA_STORAGE_MODE}, "
                f"stage_sync={_INT8_KDA_STAGE_SYNC_MODE}",
                flush=True,
            )
    if TEMPLATE_ARENA:
        _bind_template_arena(((0, l0), (1, tpl_kda), (3, tpl_mla)))
    for i in range(NL):
        layers.append(l0 if i == 0 else
                      tpl_kda if config.is_kda_layer(i) else tpl_mla)
    return layers


def _template_layout(layer_idx, module):
    """Real checkpoint shapes and aligned offsets for one template."""
    align_elems = max(1, 256 // torch.empty((), dtype=DT).element_size())
    offset = 0
    layout = []
    packed_qkv = spine_fast.dynamic_q8_qkv(module)
    for name, p in module.named_parameters():
        if ".experts." in name:
            continue
        if packed_qkv is not None and packed_qkv.consumes(name):
            # The patched projection reads its shared packed arena directly.
            # Leave the dense Parameter on meta unless a guarded fallback must
            # reconstruct it after a backend failure.
            continue
        full = f"{PFX}layers.{layer_idx}.{name}"
        meta = k3loader.INV.get(full)
        shape = tuple(meta["shape"]) if meta is not None else tuple(p.shape)
        offset = (offset + align_elems - 1) // align_elems * align_elems
        n = math.prod(shape)
        layout.append((name, shape, offset, n))
        offset += n
    offset = (offset + align_elems - 1) // align_elems * align_elems
    return layout, offset


def _bind_template_arena(indexed_modules):
    """Bind all serial template parameters to overlapping views of one arena."""
    global _TEMPLATE_ARENA_STORAGE, _TEMPLATE_ARENA_INFO
    plans = []
    for layer_idx, module in indexed_modules:
        layout, elems = _template_layout(layer_idx, module)
        plans.append((layer_idx, module, layout, elems))
    max_elems = max(row[3] for row in plans)
    arena_bytes = max_elems * torch.empty((), dtype=DT).element_size()
    decision = apple_silicon.gate_experiment(
        TEMPLATE_ARENA_MODE,
        capabilities=APPLE_CAPS,
        require_metal=DEV.type == "mps",
        min_max_buffer_bytes=arena_bytes if DEV.type == "mps" else 0,
    )
    if not decision.enabled:
        print(
            f"[templates] shared arena unavailable ({decision.reason}); "
            "using separate template allocations",
            flush=True,
        )
        return False
    try:
        storage = torch.empty(max_elems, dtype=DT, device=DEV)
    except (RuntimeError, MemoryError) as exc:
        print(
            f"[templates] shared arena allocation failed ({exc}); "
            "using separate template allocations",
            flush=True,
        )
        return False
    for _layer_idx, module, layout, _elems in plans:
        for name, shape, offset, n in layout:
            set_param(module, name, storage[offset:offset + n].view(shape))
    _TEMPLATE_ARENA_STORAGE = storage
    separate = sum(row[3] for row in plans)
    _TEMPLATE_ARENA_INFO = {
        "arena_bytes": arena_bytes,
        "separate_bytes": separate * storage.element_size(),
        "saved_bytes": (separate - max_elems) * storage.element_size(),
        "layers": [row[0] for row in plans],
        "capability_gate": decision.reason,
    }
    print(
        f"[templates] shared arena {_TEMPLATE_ARENA_INFO['arena_bytes']/2**30:.2f} "
        f"GiB, saves {_TEMPLATE_ARENA_INFO['saved_bytes']/2**30:.2f} GiB "
        "across dense/KDA/MLA templates",
        flush=True,
    )
    return True


def copy_resident(module, prefix, blobs):
    """Like _apply_resident but copies into the module's EXISTING buffers
    (first touch still allocates; A_log's checkpoint shape [128] replaces the
    constructor's [96] once, then copies match)."""
    t0 = time.time()
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue
        full = prefix + name
        rec = blobs.get(full)
        if rec is None:
            t = k3loader.load_resident(full).to(DEV, DT)
        elif rec[0] == "i8":
            shape = k3loader.INV[full]["shape"]
            q = torch.frombuffer(bytearray(rec[1]), dtype=torch.int8).reshape(shape)
            sc = torch.frombuffer(bytearray(rec[2]), dtype=torch.float16).reshape(shape[0], 1)
            qd, scd = q.to(DEV), sc.to(DEV)
            # K3_SPINE_DEQ=metal: fuse int8->fp32 + row-scale + the copy_ into one
            # bit-exact kernel writing straight into the template buffer.
            if (spine_fast.DEQ != "torch" and p.device.type == DEV.type
                    and p.shape == torch.Size(shape) and p.dtype == torch.float32
                    and DT == torch.float32):
                if spine_fast.DEQ == "metal" and spine_fast.dequant_into(
                        p.data, qd, scd.view(-1)):
                    continue
                torch.mul(qd, scd.to(torch.float32), out=p.data)
                continue
            t = (qd.to(torch.float32) * scd.to(torch.float32)).to(DT)
        else:
            meta = k3loader.INV[full]
            t = torch.frombuffer(bytearray(rec[1]),
                                 dtype=k3loader._DT[meta["dtype"]]).reshape(meta["shape"]).to(DEV, DT)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    TIMES["resident_io"] += time.time() - t0


# --- resident read/apply dispatch --------------------------------------------
# K3_SPINE_PACK=1 routes a layer through spine_fast (one packed readinto, one
# H2D, fused dequant). Default 0 = the existing per-tensor path, unchanged.
# --- resident spine RAM cache (K3_SPINE_CACHE_GB) ----------------------------
# The spine is re-read from disk every token (53 GB int8). Once compute got fast
# enough, that read stopped hiding behind it and showed up as preload_wait. Any
# layer we can hold in RAM never touches the disk again.
# The original inline version sized itself from "free + inactive" and, at
# 29.7 GB on this 64 GB box, took a decode token from 14 s to 122 s: `inactive`
# counted anonymous pages that can only be reclaimed by COMPRESSING them, so
# claiming them walked the machine into the memory compressor. It has been
# replaced by tools/spine_cache.py, which (a) computes headroom only from pages
# the kernel can reclaim for free and (b) watches host_statistics64's compressor
# and swap counters at runtime and hands memory back when they move. Default is
# still OFF (K3_SPINE_CACHE_GB unset) — byte-for-byte today's behaviour.
import spine_cache  # noqa: E402

_SPINE_CACHE = spine_cache.SpineCache(spine_fast, n_layers=NL)
if _SPINE_CACHE.enabled:
    print(f"[spine-cache] {spine_cache.describe_env()}", flush=True)


def _pack_bytes(pack):
    n = 0
    try:
        for v in (pack.values() if isinstance(pack, dict) else pack):
            for item in (v if isinstance(v, (tuple, list)) else (v,)):
                n += len(item) if isinstance(item, (bytes, bytearray, memoryview)) \
                    else getattr(item, "nbytes", 0)
    except Exception:
        return 0
    return n


# --- page-cache resident tier (K3_SPINE_RESIDENT_GB + K3_SPINE_STREAM_NOCACHE)
# Decode scans the 53 GB spine cyclically, the worst case for LRU: a page cache
# smaller than the working set gets a ~0% hit rate. Pinning a fixed subset and
# keeping the streaming tier out of cache turns that into a hit rate equal to
# the fraction that fits, using clean file pages rather than swappable heap.
_RESIDENT_SPEC = os.environ.get("K3_SPINE_RESIDENT_GB")
_RESIDENT_GB = float(_RESIDENT_SPEC or 0)
if _RESIDENT_SPEC is not None and _RESIDENT_GB >= 0 and spine_io.STREAM_TIER and QUANT:
    _rset, _rbytes = spine_cache.resident_tier(INT8_DIR, PFX, NL, _RESIDENT_GB * 1e9)
    spine_io.set_resident_tier(_rset)
    print(f"[spine] page-cache resident tier: {len(_rset)}/{NL} layers, "
          f"{_rbytes/1e9:.1f} GB; the rest use streaming cache advice",
          flush=True)


def _spine_read(module, prefix):
    cached = _SPINE_CACHE.get(prefix)
    if cached is not None:
        return cached
    if MIXED:            # always packed: mixed_spine has no per-tensor read path
        pack = mixed_spine.read_pack(module, prefix, INT8_DIR, k3loader.RES,
                                     k3loader.INV)
    elif spine_fast.PACK:
        pack = spine_fast.read_pack(module, prefix, INT8_DIR, k3loader.RES,
                                    k3loader.INV, SPINE, k3loader.load_resident)
    else:
        pack = _read_resident_bytes(module, prefix)
    if _SPINE_CACHE.enabled and spine_fast.PACK and not MIXED and isinstance(pack, dict):
        # not MIXED: the cache pins buffers through spine_fast's pool, and a
        # mixed pack comes from mixed_spine's — pinning the wrong pool would let
        # the next layer's readinto overwrite cached weights in place.
        if not _SPINE_CACHE.admit(prefix, pack, _pack_bytes(pack)):
            _SPINE_CACHE.poll()
    return pack


def _spine_apply(module, prefix, pack):
    if MIXED:
        t0 = time.time()
        mixed_spine.apply_pack(module, prefix, pack, DEV, DT, k3loader.INV,
                               k3loader._DT, set_param, k3loader.load_resident)
        TIMES["resident_io"] += time.time() - t0
        return
    if spine_fast.PACK:
        t0 = time.time()
        spine_fast.apply_pack(module, prefix, pack, DEV, DT, k3loader.INV,
                              k3loader._DT, set_param, k3loader.load_resident)
        TIMES["resident_io"] += time.time() - t0
        return
    (copy_resident if TEMPLATES else _apply_resident)(module, prefix, pack)


def causal_mask(T, past=0, dtype=None):
    """Return [1, 1, T, past+T] without allocating a full context square."""
    dtype = dtype or DT
    width = past + T
    m = torch.zeros(1, 1, T, width, dtype=dtype, device=DEV)
    future = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=DEV), diagonal=1)
    m[:, :, :, past:].masked_fill_(future, torch.finfo(dtype).min)
    return m


def forward_pass(layers, cache, hidden, step, verbose=True):
    """hidden: [1, T, H] fp32. Returns logits [1, T, vocab]."""
    T = hidden.shape[1]
    if pilot.PILOT:
        pilot.init(config, DEV, _pilot_load, PFX,
                   load_packed=_load_int8_packed if QUANT else None,
                   native_int8=(
                       _PACKED_Q8_BACKEND.matmul
                       if (
                           _PACKED_Q8_BACKEND.available
                           and DEV.type == "mps"
                       )
                       else False
                   ))
        pilot.begin_pass(fetch_v2)
        if PREAD:
            fetch_v2.drop_prefetch()    # nothing from the previous pass is valid
            if pilot.ASYNC_DRAIN:
                pilot.install_async_drain(fetch_v2.reader())
    elif EXPERT_PREFETCH:
        global _PREV_SEL
        _PREV_SEL = dict(_LAST_SEL)     # last token's routing = this pass's oracle
        fetch_v2.drop_prefetch()        # nothing from the previous pass is valid
    past = cache.get_seq_length() or 0
    mask = causal_mask(T, past) if T > 1 else None
    block_residual = hidden.new_zeros(T, 0, H)

    def _next_unpinned(j):
        while j < NL and j < PIN_N and getattr(layers[j], "_k3_res", False):
            j += 1
        return j

    nxt = _next_unpinned(0)
    fut = (_PRELOADER.submit(_spine_read, layers[nxt], f"{PFX}layers.{nxt}.")
           if PRELOAD and nxt < NL else None)
    for i, layer in enumerate(layers):
        _step_ctx["layer"] = i
        if TEMPLATES:
            layer.layer_idx = i
            layer.self_attn.layer_idx = i
        pinned = i < PIN_N and getattr(layer, "_k3_res", False)
        if not pinned:
            if PRELOAD and fut is not None and i == nxt:
                _tw = time.time()
                blobs = fut.result()
                TIMES["preload_wait"] += time.time() - _tw
                j = _next_unpinned(i + 1)
                fut = (_PRELOADER.submit(_spine_read, layers[j], f"{PFX}layers.{j}.")
                       if j < NL else None)
                nxt = j
                _spine_apply(layer, f"{PFX}layers.{i}.", blobs)
            else:
                if TEMPLATES or spine_fast.PACK:
                    pfx = f"{PFX}layers.{i}."
                    _spine_apply(layer, pfx, _spine_read(layer, pfx))
                else:
                    materialize_resident(layer, f"{PFX}layers.{i}.")
            if i < PIN_N:
                layer._k3_res = True   # pinned from now on
        if pilot.enabled():
            pilot.arm(layer)
        if PROFILE and DEV.type in ("mps", "cuda"):
            _device_synchronize()
        t0 = time.time()
        hidden, block_residual = layer(
            hidden, attention_mask=mask, position_ids=None,
            past_key_values=cache, use_cache=True, block_residual=block_residual)
        if PROFILE and DEV.type in ("mps", "cuda"):
            _device_synchronize()
        dt_layer = time.time() - t0
        TIMES["compute"] += dt_layer
        if PROFILE:
            k = "kda" if layer.is_linear_attn else "mla"
            PROF[k] += dt_layer
            PROF["n_" + k] += 1
        if not TEMPLATES and not (i < PIN_N):
            dematerialize(layer)
        if verbose and (i % 10 == 0 or i == NL - 1):
            print(f"    layer {i:2d}/92 done  (res_io {TIMES['resident_io']:.0f}s "
                  f"exp {TIMES['expert_fetch']:.0f}s comp {TIMES['compute']:.0f}s)",
                  flush=True)
    TRACE.end_pass()
    if PROFILE:
        mk = TIMES["moe_kernel"]
        print(f"[prof] KDA {PROF['kda']:.1f}s/{PROF['n_kda']} MLA {PROF['mla']:.1f}s/{PROF['n_mla']} "
              f"| moe_kernel {mk:.1f}s | fetch {TIMES['expert_fetch']:.1f}s "
              f"| apply {TIMES['resident_io']:.1f}s"
              f"| preload_wait {TIMES['preload_wait']:.1f}s", flush=True)
        rep = spine_fast.phase_report()
        if rep:
            print(rep, flush=True)
        rep = _SPINE_CACHE.report()
        if rep:
            print(rep, flush=True)
        rep = pilot.report(fetch_v2.stats if fetch_v2 is not None else None)
        if rep:
            print(rep, flush=True)
        if grouped_moe.enabled():
            print(f"[grouped-moe] {grouped_moe.STATS}", flush=True)
    # tail: output attn-res -> final norm -> lm_head
    tail = _tail_module()
    apply_res = getattr(ml, "_apply_attn_res", None) or ml.KimiDecoderLayer._apply_attn_res
    flat = apply_res(hidden.view(-1, H), block_residual,
                     tail.output_attn_res_proj, tail.output_attn_res_norm)
    hidden = tail.norm(flat.view(1, T, H))
    t0 = time.time()
    _ensure_lm_head_loaded()
    TIMES["resident_io"] += time.time() - t0
    # Generation consumes only the final prompt logit. Speculative/decode
    # passes still need every position for verification.
    head_hidden = (hidden[:, -1:, :] if PREFILL_LAST_LOGIT and step == 0
                   and T > 1 else hidden)
    logits = _lm_head_forward(head_hidden)
    return logits


_LM_W = None
_LM_Q = None
_LM_SC = None
_LM_PACKED_CALLS = 0
_LM_DENSE_CALLS = 0
_LM_DENSE_CROSSOVERS = 0
_LM_DISABLE_REASON = None


def _ensure_lm_head_loaded():
    """Load exactly one head representation and preserve one-time fallback."""
    global _LM_W, _LM_Q, _LM_SC
    if INT8_LM_HEAD and _LM_Q is None and _LM_W is None:
        packed = _load_int8_packed("language_model.lm_head.weight")
        if packed is not None:
            _LM_Q, _LM_SC = packed
    if _LM_Q is None and _LM_W is None:
        # Legacy path: resident across tokens, 2.35-4.7 GB on DEV.
        _LM_W = (
            _load_int8("language_model.lm_head.weight")
            if QUANT else None
        )
        if _LM_W is None:
            _LM_W = k3loader.load_resident(
                "language_model.lm_head.weight"
            ).to(DEV, DT)


# --- n-gram speculative 2-token decode ----------------------------------------
# Resident I/O + compute dominate a token and amortize across batch positions,
# so an accepted free n-gram draft yields 2 tokens for ~1.2x one pass.
def ngram_draft(ids, max_n=6, min_n=2):
    for n in range(min(max_n, len(ids) - 1), min_n - 1, -1):
        suf = ids[-n:]
        for j in range(len(ids) - n - 1, -1, -1):
            if ids[j:j + n] == suf:
                return ids[j + n]
    return None


def snapshot_states(cache):
    """Retain pre-pass cache objects without cloning their storage.

    KDA recurrence and ShortConvolution return new storage. MLA's geometric
    slab may reuse storage, but only writes beyond the old view; the old prefix
    remains immutable and its shape records the rollback length. The contracts
    are exercised by tools/test_snapshot_refs.py and tools/test_kv_slab.py.
    Keeping the old objects is therefore exact and avoids ~475 MB of clones.
    """
    snap = {"rec": {}, "conv": {}, "mla": {}}
    for i in range(NL):
        if cache.recurrent_states[i] is not None:
            snap["rec"][i] = cache.recurrent_states[i]
        if cache.conv_states[i] is not None:
            snap["conv"][i] = cache.conv_states[i]
        if cache.key_cache[i] is not None:
            snap["mla"][i] = (cache.key_cache[i], cache.value_cache[i])
    return snap


def restore_states(cache, snap, keep=0):
    """Undo a speculative pass. `keep` extra MLA positions are retained (used
    only by the deep-spec `rerun` path, which re-feeds the accepted prefix);
    keep=0 is the shipped all-the-way-back behaviour."""
    for i, t in snap["rec"].items():
        cache.recurrent_states[i] = t
    for i, c in snap["conv"].items():
        cache.conv_states[i] = c
    for i, (old_key, old_value) in snap["mla"].items():
        if keep:
            L = old_key.shape[2]
            if hasattr(cache, "truncate_mla"):
                cache.truncate_mla(i, L + keep)
            else:
                cache.key_cache[i] = cache.key_cache[i][
                    :, :, :L + keep].contiguous()
                cache.value_cache[i] = cache.value_cache[i][
                    :, :, :L + keep].contiguous()
        else:
            if hasattr(cache, "restore_mla"):
                cache.restore_mla(i, old_key, old_value)
            else:
                cache.key_cache[i] = old_key
                cache.value_cache[i] = old_value


EOS_ID = 163586  # <|end_of_msg|> — K3's generation stop token


def _spec_step_deep(layers, cache, embed, ctx, pending, s):
    """One K3_SPEC_DEPTH>1 pass. Returns (new_tokens, tag).

    Verifies D+1 positions in a single forward pass and accepts the longest
    correct prefix: with `argm[i] = argmax(logits[0, i])`, argm[0] is the true
    token after `pending` and is always correct, argm[i] is correct iff drafts
    0..i-1 were all correct. So k = #leading drafts matching argm, and the
    emitted run is argm[0..k] — k+1 tokens, the last of which (the model's
    argmax at the mismatch) is the free correct token.

    The pass consumed D+1 positions but only k+1 are real, so the cache is
    rolled back to exactly k+1; see tools/spec_decode.py for why that is exact
    for the KDA fold as well as the MLA KV.
    """
    D = spec_decode.next_depth()
    drafts = spec_decode.draft(ctx, D, ngram_draft)
    if not drafts:
        logits = forward_pass(layers, cache, embed([pending]), step=s, verbose=False)
        tok = int(logits[0, -1].argmax())
        spec_decode.record(0, 0, 1)
        return [tok], " spec-nodraft"

    rerun = spec_decode.ROLLBACK == "rerun"
    snap = snapshot_states(cache) if rerun else None
    mla_len = spec_decode.snapshot_mla(cache, NL)
    replay_armed = False
    try:
        if not rerun:
            # Replay consumes the captured KDA inputs. The reference rerun path
            # restores the pre-pass tensor objects and re-executes instead, so
            # arming there only pins hundreds of MiB without being read.
            spec_decode.arm()
            replay_armed = True
        logits = forward_pass(layers, cache, embed([pending] + drafts),
                              step=s, verbose=False)
        argm = logits[0].argmax(-1).tolist()      # one device sync, D+1 entries
        k = 0
        while k < len(drafts) and argm[k] == drafts[k]:
            k += 1
        new = argm[:k + 1]
        if k < len(drafts) and not rerun:
            spec_decode.rollback_replay(cache, k + 1, mla_len)
    finally:
        if replay_armed:
            spec_decode.release()
    if k < len(drafts) and rerun:
        # Reference path: back to the pre-pass state with the certified
        # whole-tensor restore, then re-feed exactly the accepted tokens.
        # Costs a second spine read; used to validate the replay path.
        restore_states(cache, snap)
        spec_decode.STATS["reruns"] += 1
        forward_pass(layers, cache, embed([pending] + new[:-1]),
                     step=s, verbose=False)
    spec_decode.record(k, len(drafts), len(new))
    return new, f" spec+{len(new)} ({k}/{len(drafts)} draft)"


def _generation_runtime(fn):
    """Run an entire generation with inference tensors and no cyclic-GC polls."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        disable_gc = os.environ.get("K3_DISABLE_GC", "1") == "1"
        restore_gc = disable_gc and gc.isenabled()
        if restore_gc:
            gc.disable()
        try:
            with torch.inference_mode():
                return fn(*args, **kwargs)
        finally:
            if restore_gc:
                gc.enable()
    return wrapped


@_generation_runtime
def generate(layers, cache, embed, ids, max_new, spec=None, on_token=None,
             verbose_prefill=False, log=lambda *a: None):
    """Greedy generation (+ certified-lossless n-gram speculation).

    Shared by the CLI and the OpenAI-compatible server. Calls on_token(token_id)
    as each token is emitted. Returns at most ``max_new`` tokens and never
    streams anything after EOS_ID, including inside a speculative burst."""
    if spec is None:
        spec = os.environ.get("K3_SPEC", "1") == "1"
    if max_new <= 0:
        return []
    generated = []

    def emit(t):
        generated.append(t)
        if on_token:
            on_token(t)

    _step_ctx["step"] = 0
    logits = forward_pass(layers, cache, embed(ids), step=0, verbose=verbose_prefill)
    first = int(logits[0, -1].argmax())
    emit(first)
    if first == EOS_ID:
        return generated
    for _k in EXPERT_SEL:      # the union factor that matters is the decode one
        EXPERT_SEL[_k] = 0
    s = 1
    deep = spec and spec_decode.enabled()
    while len(generated) < max_new:
        _step_ctx["step"] = s
        if os.environ.get("K3_PREFETCH", "1") == "1":
            prefetch_prev_token()
        t0 = time.perf_counter_ns()
        tag = ""
        if deep:
            new, tag = _spec_step_deep(layers, cache, embed,
                                       ids + generated, generated[-1], s)
            if EOS_ID in new:   # never stream past EOS (a pass emits up to D+1)
                new = new[:new.index(EOS_ID) + 1]
        else:
            draft = ngram_draft(ids + generated) if spec else None
            if draft is not None:
                snap = snapshot_states(cache)
                logits = forward_pass(layers, cache, embed([generated[-1], draft]),
                                      step=s, verbose=False)
                # Read both verifier decisions in one device-to-host transfer.
                # The first position decides accept/rollback; on acceptance the
                # second ID is already present instead of forcing another sync.
                verified = logits[0].argmax(-1).tolist()
                n1 = verified[0]
                if n1 == draft:
                    new = verified[:2]
                    tag = " spec+2"
                else:
                    restore_states(cache, snap)
                    logits = forward_pass(layers, cache, embed([generated[-1]]),
                                          step=s, verbose=False)
                    new = [int(logits[0, -1].argmax())]
                    tag = " spec-miss"
            else:
                logits = forward_pass(layers, cache, embed([generated[-1]]),
                                      step=s, verbose=False)
                new = [int(logits[0, -1].argmax())]
        # A verifier can accept several tokens in one pass. Bound the burst
        # before invoking callbacks or structured logging so max_new remains a
        # real API contract and throughput cannot be inflated by over-emission.
        new = new[:max_new - len(generated)]
        for t in new:
            emit(t)
        log(s, tag, t0, list(generated))
        s += 1
        if EOS_ID in new:
            break
    return generated


TOTAL_EXPERTS = 82432
EXPERT_SPAN = 17547264


def check_expert_pool():
    """Streaming is the fallback, not the goal. Warn clearly when the expert pool
    isn't fully local, because every novel prompt pays for it over the network."""
    import shutil
    n, _ = k3loader.cache_totals()
    if n >= TOTAL_EXPERTS:
        return
    missing = TOTAL_EXPERTS - n
    need = missing * EXPERT_SPAN
    free = shutil.disk_usage(ROOT).free
    print("=" * 72)
    print(f"  STREAMING MODE — {n:,} of {TOTAL_EXPERTS:,} experts are local "
          f"({n/TOTAL_EXPERTS*100:.1f}%)")
    print()
    print("  Experts that aren't on disk get fetched from Hugging Face while you")
    print("  generate. Every token needs 25.8 GB of expert data:")
    print("      from local disk   ~4 s      ->  roughly 60-76 s per token")
    print("      over the network  minutes   ->  roughly 3+ min per token")
    print()
    if free - need > 100e9:
        print(f"  Downloading the rest is a one-time cost ({need/1e12:.2f} TB, you have "
              f"{free/1e12:.2f} TB free)")
        print("  and makes every prompt run at full speed:")
        print()
        print("      python tools/fetch_experts_all.py        # resumable, run anytime")
    else:
        short = (need + 100e9 - free) / 1e12
        print(f"  Finishing the download needs {need/1e12:.2f} TB + 100 GB headroom, but only")
        print(f"  {free/1e12:.2f} TB is free — about {short:.2f} TB short. Freeing that space is")
        print("  the single biggest speedup available here. Partial helps too:")
        print()
        print("      python tools/fetch_experts_all.py --layers 1-40")
    print("=" * 72, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=None,
                    help="cap on generated tokens; default: run until the model "
                         "finishes (chat mode) or until Ctrl-C — both stop cleanly")
    ap.add_argument("--chat", action="store_true", help="use the K3 chat template")
    ap.add_argument("--events-jsonl",
                    help="write an exclusive, flushed JSONL run-event stream "
                         "(used by tools/bench.py)")
    args = ap.parse_args()
    events = EventLog(args.events_jsonl)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "k3-meta"), trust_remote_code=True)
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      tokenize=True, add_generation_prompt=True)
    else:
        ids = tok.encode(args.prompt)
    if args.max_new is not None and args.max_new < 0:
        ap.error("--max-new must be non-negative")
    max_new = (
        args.max_new if args.max_new is not None else 1_000_000
    )  # effectively: until EOS or Ctrl-C
    run_started_ns = time.perf_counter_ns()
    events.emit(
        "run_start",
        argv=sys.argv,
        prompt=args.prompt,
        chat=args.chat,
        max_new=max_new,
        input_token_ids=ids,
        config={
            "device": str(DEV),
            "spine": SPINE,
            "dtype": str(DT),
            "approx": APPROX,
            "templates": TEMPLATES,
            "template_arena": bool(_TEMPLATE_ARENA_STORAGE is not None),
            "int8_kda_qkv_requested": _INT8_KDA_QKV_REQUESTED,
            "int8_kda_qkv_eligible": INT8_KDA_QKV,
            "int8_kda_storage": _INT8_KDA_STORAGE_MODE,
            "int8_kda_stage_sync": _INT8_KDA_STAGE_SYNC_MODE,
            "int8_lm_head_requested": _INT8_LM_HEAD_REQUESTED,
            "int8_lm_head_eligible": INT8_LM_HEAD,
            "packed_q8_backend": _PACKED_Q8_BACKEND.status(),
            "mla_cpu_sdpa": attn_fast.cpu_sdpa_status(),
            "preload": PRELOAD,
            "pin_layers": PIN_N,
            "fast_moe": FAST_MOE,
            "moe_backend": MOE_BACKEND,
            "moe_top_k": MOE_TOP_K,
            "base_moe_top_k": BASE_MOE_TOP_K,
            "cpu_moe_batch": CPU_BATCH_ACTIVE,
            "moe_group_size": grouped_moe.GROUP_SIZE,
            "apple_silicon_capability_fingerprint": APPLE_CAPS.fingerprint(),
            "performance_environment": _performance_env(),
        },
    )
    check_expert_pool()
    print(f"prompt tokens ({len(ids)}): {ids}", flush=True)

    layers = build_layers()
    cache = ml.KimiDynamicCache(config)
    embed = LazyEmbed()
    prefill_started_ns = time.perf_counter_ns()
    if args.max_new is None and not args.chat:
        print("note: raw completions have no natural end — press Ctrl-C to stop "
              "cleanly, or pass --max-new N", flush=True)
    print(f"=== prefill: {len(ids)} tokens through 93 layers ===", flush=True)
    state = {"first": True}
    generated = []   # mirrored via on_token so Ctrl-C still has the text
    stream_decoder = IncrementalTokenDecoder(tok)
    step_deltas = []

    def on_token(t):
        generated.append(t)
        delta = stream_decoder.append(t)
        step_deltas.append(delta)
        if state["first"]:
            state["first"] = False
            state["logged_count"] = 1
            duration_ns = time.perf_counter_ns() - prefill_started_ns
            text = tok.decode([t])
            events.emit("prefill_done", duration_ns=duration_ns,
                        emitted_token_ids=[t], emitted_token_text=[text])
            print(f"[prefill done in {duration_ns/1e9:.6f}s] "
                  f"first token: {t!r} = {delta!r}", flush=True)
            step_deltas.clear()

    def log(s, tag, t0, gen):
        duration_ns = time.perf_counter_ns() - t0
        start = state.get("logged_count", 0)
        emitted = gen[start:]
        state["logged_count"] = len(gen)
        delta = "".join(step_deltas)
        step_deltas.clear()
        if events.enabled:
            # Evidence mode intentionally pays for a cumulative decode; the
            # default performance path never does quadratic prefix rendering.
            events.emit("decode_step", step=s, duration_ns=duration_ns, tag=tag,
                        emitted_token_ids=emitted,
                        emitted_token_text=[tok.decode([t]) for t in emitted],
                        cumulative_token_ids=gen, cumulative_text=tok.decode(gen))
        print(f"[token {s}: {duration_ns/1e9:.6f}s{tag}] "
              f"+{delta!r}", flush=True)
        print("   ", k3loader.cache_report(), flush=True)

    status = "ok"
    try:
        generate(layers, cache, embed, ids, max_new,
                 on_token=on_token, verbose_prefill=True, log=log)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[stopped by Ctrl-C]", flush=True)
    except BaseException as exc:
        events.emit("run_error", error_type=type(exc).__name__, message=str(exc),
                    duration_ns=time.perf_counter_ns() - run_started_ns,
                    emitted_token_ids=generated,
                    runtime={
                        "int8_kda_qkv": _int8_kda_qkv_runtime_status(),
                        "int8_lm_head": _lm_head_runtime_status(),
                        "mla_cpu_sdpa": attn_fast.cpu_sdpa_status(),
                    })
        events.close()
        raise
    decoder_tail = stream_decoder.finish()
    if decoder_tail:
        print(f"[decoder tail] +{decoder_tail!r}", flush=True)
    emitted_with_eos = list(generated)
    if EOS_ID in generated:
        generated = generated[:generated.index(EOS_ID)]

    print("\n=== RESULT ===")
    print("completion:", tok.decode(generated))
    print("token ids:", generated)
    rep = spec_decode.report()
    if rep:
        print(rep)
    es = EXPERT_SEL
    if es["layer_calls"]:
        tk = config.num_experts_per_token
        print(f"[experts] decode: {es['uniq']/es['layer_calls']:.1f} unique/layer "
              f"over T={es['pos']/es['layer_calls']:.2f} positions "
              f"= {es['uniq']/(es['layer_calls']*tk):.2f}x a single-token "
              f"(top-{tk}) read")
    duration_ns = time.perf_counter_ns() - prefill_started_ns
    completion = tok.decode(generated)
    events.emit("run_end", status=status, duration_ns=duration_ns,
                emitted_token_ids=emitted_with_eos,
                completion_token_ids=generated, completion_text=completion,
                phase_seconds=TIMES,
                runtime={
                    "int8_kda_qkv": _int8_kda_qkv_runtime_status(),
                    "int8_lm_head": _lm_head_runtime_status(),
                    "mla_cpu_sdpa": attn_fast.cpu_sdpa_status(),
                })
    events.close()
    print(f"total {duration_ns/1e9:.6f}s | times {TIMES}")
    print(k3loader.cache_report())
    TRACE.close()


if __name__ == "__main__":
    main()
