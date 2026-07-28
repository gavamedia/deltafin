#!/usr/bin/env python3
"""lazy-K3: real Kimi-K3 inference on a 64GB M1 Max by layer-streaming.

Uses Moonshot's own modeling_kimi_linear.py (audited) with a pure-PyTorch fla shim.
Per forward pass, each of the 93 decoder layers is materialized from the local
resident-spine download, routed experts are fetched on demand (HTTP Range, disk
cached) and dequantized from MXFP4, the layer runs in fp32 on CPU, then its
weights are freed. Router selections are logged to router_trace.jsonl.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))    # fla shim, k3loader, mxfp4
# modeling files imported via tools/k3pkg package

import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)
torch.set_num_threads(8)

import k3loader  # noqa: E402
import importlib  # noqa: E402

from k3pkg import modeling_kimi_linear as ml

CFG_JSON = json.load(open(os.path.join(ROOT, "k3-meta/config.json")))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None)
if Cfg is None:
    Cfg = importlib.import_module("k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG_JSON)
config._attn_implementation = "eager"
H = config.hidden_size
NL = config.num_hidden_layers
PFX = "language_model.model."
# Sensible defaults, no env vars required: use the GPU when there is one, and
# use the int8 spine when it has been built. Both remain overridable.
INT8_DIR = os.path.join(ROOT, "k3-resident-int8/tensors")


def _auto_dev():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    print("[config] no MPS or CUDA GPU found — running on CPU (slow). "
          "Deltafin targets Apple Silicon.", flush=True)
    return "cpu"


def _auto_spine():
    try:
        if any(f.endswith(".i8") for f in os.listdir(INT8_DIR)):
            return "int8"
    except FileNotFoundError:
        pass
    return "bf16"


DEV = torch.device(os.environ.get("K3_DEV") or _auto_dev())   # cpu | mps | cuda
SPINE = os.environ.get("K3_SPINE") or _auto_spine()           # bf16 | int8
if SPINE == "bf16" and "K3_SPINE" not in os.environ:
    print("[config] int8 spine not found — using bf16 (2x the per-token I/O). "
          "Build it with: python tools/convert_spine_int8.py", flush=True)
# K3_APPROX=1 = "approx mode": approximate numerics (fp16 weights) + n-gram
# speculation. Output stays coherent but near-tie tokens may differ from the
# fp32 reference — never use for oracle runs. Speed effect is unproven until a
# quiet-machine A/B; if it measures faster it can earn a faster name.
APPROX = os.environ.get("K3_APPROX", "0") == "1"
DT = torch.float16 if (APPROX or os.environ.get("K3_DTYPE", "fp32") == "fp16") else torch.float32
TRACE = open(os.path.join(ROOT, "k3-meta/router_trace.jsonl"), "a")
TIMES = {"resident_io": 0.0, "expert_fetch": 0.0, "compute": 0.0, "moe_kernel": 0.0,
         "preload_wait": 0.0}   # time the main thread blocks on the preloader
PROFILE = os.environ.get("K3_PROFILE", "0") == "1"
PROF = {"kda": 0.0, "mla": 0.0, "n_kda": 0, "n_mla": 0}


def set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def _load_int8(full):
    """Return dequantized fp32 tensor on DEV from the int8 spine, or None."""
    op = os.path.join(INT8_DIR, full + ".i8")
    if not os.path.exists(op):
        return None
    shape = k3loader.INV[full]["shape"]
    q = torch.frombuffer(bytearray(open(op, "rb").read()), dtype=torch.int8).reshape(shape)
    sc = torch.frombuffer(bytearray(open(os.path.join(INT8_DIR, full + ".sc"), "rb").read()),
                          dtype=torch.float16).reshape(shape[0], 1)
    return (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)


def materialize_resident(module, prefix):
    t0 = time.time()
    missing = []
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue  # routed experts stay meta until selected
        full = prefix + name
        t = _load_int8(full) if SPINE == "int8" else None
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

# --- fast resident-spine path (K3_FAST_SPINE=1, default off) ------------------
# Packed readinto + one H2D per layer + a bit-exact Metal dequant kernel.
# See tools/spine_fast.py for the measurements that motivate each piece.
import spine_fast  # noqa: E402

if spine_fast.FAST or spine_fast.DEQ == "metal":
    spine_fast.metal_available()          # compile once, on the main thread
    print(f"[spine] fast path: {spine_fast.describe()}", flush=True)

# --- attention / norm fast paths (K3_KDA_RECUR, K3_SHORTCONV, K3_COMPILE) -----
# All default to the behaviour above; see tools/attn_fast.py for the per-op
# measurements that motivate each one.
import attn_fast  # noqa: E402

attn_fast.install(ml)
if attn_fast.ACTIVE:
    print(f"[attn] {attn_fast.describe()}", flush=True)


def _ram_budget_layers():
    if TEMPLATES:
        return 0  # templates and per-layer pinning are mutually exclusive
    if os.environ.get("K3_PIN_LAYERS") is not None:
        return int(os.environ["K3_PIN_LAYERS"])
    import subprocess
    try:  # macOS
        total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    except (OSError, subprocess.CalledProcessError, ValueError):  # Linux
        total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    reserve = max(10.0, 0.18 * total_gb)
    budget = float(os.environ.get("K3_RAM_GB", 0)) or (total_gb - reserve)
    overhead = 8.0 + (4.7 if DT == torch.float32 else 2.35) + 2.0   # process+lm_head+transients
    per_layer = (113.5 / NL) * (2 if DT == torch.float32 else 1)    # fp32=2x int8 bytes, fp16=1x
    n = max(0, int(0.4 * (budget - overhead) / per_layer))
    print(f"[ram] total {total_gb:.0f} GB, budget {budget:.1f} GB -> pinning "
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
import fast_moe  # noqa: E402

# --- MoE compute backend (K3_MOE=cpu|metal) ----------------------------------
# cpu   : tools/fast_moe.py, the fused MXFP4 GEMV in libmxfp4gemv.dylib (default,
#         unchanged behaviour).
# metal : tools/metal_moe.py, the whole layer's selected experts as one GPU
#         command buffer. Same signature, same semantics; matched the CPU path to
#         2.5e-7 on real experts. Falls back to cpu (loudly) if Metal is missing.
# K3_MOE_CHECK=N cross-checks the first N calls against the CPU kernel and raises
# on disagreement — see tools/metal_moe.py. K3_METAL_BINDLESS=0 picks the
# per-expert dispatch mode instead of the Tier-2 argument-buffer one.
MOE_BACKEND = os.environ.get("K3_MOE", "metal").lower()
_MOE_FN = fast_moe.moe_infer_fast
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

fetch_v2 = None
if os.environ.get("K3_FETCH", "v2") == "v2":
    import fetch_v2
    k3loader.fetch_experts = fetch_v2.fetch_experts  # 6.4x: coalesced + keep-alive

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
                k3loader.fetch_experts(li, snap[li], dequant=False)
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()


def moe_infer_lazy(self, x, topk_ids, topk_weight):
    li = _step_ctx["layer"]
    rows = topk_ids.tolist()                    # [positions][top_k]
    flat = [e for r in rows for e in r]
    ids = sorted(set(flat))
    _LAST_SEL[li] = ids
    if pilot.enabled():
        pilot.on_actual(li, rows)               # score the prediction made at li-1
    t0 = time.time()
    raw = k3loader.fetch_experts(li, ids, dequant=not FAST_MOE)
    TIMES["expert_fetch"] += time.time() - t0
    # Speculative reads are issued only AFTER this layer's demand reads have
    # landed: the pread pool is FIFO, so a prefetch queued first would put the
    # next layer's speculation in front of this layer's blocking reads.
    if pilot.enabled() and fetch_v2 is not None:
        pilot.issue_prefetch(li + 1, fetch_v2, pread=PREAD)
    elif EXPERT_PREFETCH:
        nxt = _PREV_SEL.get(li + 1)
        if nxt:
            fetch_v2.prefetch_layer(li + 1, nxt)
    TRACE.write(json.dumps({"step": _step_ctx["step"], "layer": li,
                            "ids": flat,
                            "w": [round(x, 5) for x in topk_weight.view(-1).tolist()]}) + "\n")
    TRACE.flush()
    if FAST_MOE:
        tk = time.time()
        out = _MOE_FN(x, topk_ids, topk_weight, raw)      # K3_MOE=cpu|metal
        TIMES["moe_kernel"] += time.time() - tk
        return out
    for e, w in raw.items():
        ex = self.experts[e]
        for wn in ("w1", "w2", "w3"):
            set_param(ex, wn + ".weight", w[wn])
    out = _orig_moe_infer(self, x, topk_ids, topk_weight)
    for e in ids:  # free expert weights again
        for wn in ("w1", "w2", "w3"):
            set_param(self.experts[e], wn + ".weight",
                      torch.empty(0, device="meta"))
    return out


ml.KimiSparseMoeBlock.moe_infer = moe_infer_lazy

# Router lookahead hooks the MoE block's entry, which is the one point in the
# graph that sits after layer L's attention and before any expert read.
_orig_moe_forward = ml.KimiSparseMoeBlock.forward


def moe_forward_pilot(self, hidden_states):
    if pilot.enabled():
        pilot.on_moe_entry(self, hidden_states, _step_ctx["layer"])
    return _orig_moe_forward(self, hidden_states)


ml.KimiSparseMoeBlock.forward = moe_forward_pilot


def _pilot_load(full):
    """The resident loader the layer templates use — so a cached gate is the
    exact tensor the model routes with (int8-dequantized under K3_SPINE=int8)."""
    t = _load_int8(full) if SPINE == "int8" else None
    if t is None:
        t = k3loader.load_resident(full).to(DEV, DT)
    return t


# --- embeddings via memmap (row reads only) -----------------------------------
class LazyEmbed:
    """bf16 embedding rows from the local blob when present, else per-row HTTP Range."""
    NAME = PFX + "embed_tokens.weight"

    def __init__(self):
        self.path = os.path.join(ROOT, "k3-resident/tensors", self.NAME)
        self.meta = k3loader.INV[self.NAME]
        self.rowbytes = H * 2

    def _row(self, tid):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                f.seek(tid * self.rowbytes)
                return f.read(self.rowbytes)
        m = self.meta
        start = 8 + m["hlen"] + m["offsets"][0] + tid * self.rowbytes
        import urllib.request
        req = urllib.request.Request(
            k3loader.BASE + m["shard"],
            headers={"Range": f"bytes={start}-{start+self.rowbytes-1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def __call__(self, ids):
        buf = b"".join(self._row(int(t)) for t in ids)
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(len(ids), H)
        return t.to(DEV, DT).unsqueeze(0)  # [1, T, H]


def build_layers():
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
    for i in range(NL):
        layers.append(l0 if i == 0 else
                      tpl_kda if config.is_kda_layer(i) else tpl_mla)
    return layers


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
# layer we can hold in RAM never touches the disk again. Sized from free RAM by
# default; 0 disables. Layers are cached in walk order, so a partial cache still
# eliminates a contiguous prefix of the per-token reads.
def _spine_cache_budget():
    # DEFAULT 0 — MEASURED HARMFUL. Holding 29.7 GB of spine blobs in RAM took a
    # decode token from 14 s to 122 s: on this 64 GB machine it evicted the page
    # cache the expert reads depend on and pushed the system into the memory
    # compressor (58.8M swapouts, 3.8 GB swap). The OS was already caching the
    # spine in "inactive" pages, which the auto-budget wrongly counted as free.
    # Kept as an opt-in knob for machines with genuine RAM headroom (128 GB+).
    env = os.environ.get("K3_SPINE_CACHE_GB")
    if env is not None:
        return float(env) * 1e9
    return 0.0



_SPINE_CACHE = {}
_SPINE_CACHE_BYTES = 0
_SPINE_CACHE_MAX = _spine_cache_budget()
_SPINE_CACHE_FULL = False


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


def _spine_read(module, prefix):
    cached = _SPINE_CACHE.get(prefix)
    if cached is not None:
        return cached
    if spine_fast.PACK:
        pack = spine_fast.read_pack(module, prefix, INT8_DIR, k3loader.RES,
                                    k3loader.INV, SPINE, k3loader.load_resident)
    else:
        pack = _read_resident_bytes(module, prefix)
    global _SPINE_CACHE_BYTES, _SPINE_CACHE_FULL
    if _SPINE_CACHE_MAX and not _SPINE_CACHE_FULL:
        n = _pack_bytes(pack)
        if n and _SPINE_CACHE_BYTES + n <= _SPINE_CACHE_MAX:
            # take ownership of the pooled buffers, or the next layer's readinto
            # would overwrite this cache entry in place
            if spine_fast.PACK and isinstance(pack, dict):
                spine_fast.pin(pack.get("q"), pack.get("sc"), pack.get("other"))
            _SPINE_CACHE[prefix] = pack
            _SPINE_CACHE_BYTES += n
        else:
            _SPINE_CACHE_FULL = True
            print(f"[spine] RAM cache full: {len(_SPINE_CACHE)}/{NL} layers, "
                  f"{_SPINE_CACHE_BYTES/1e9:.1f} GB", flush=True)
    return pack


def _spine_apply(module, prefix, pack):
    if spine_fast.PACK:
        t0 = time.time()
        spine_fast.apply_pack(module, prefix, pack, DEV, DT, k3loader.INV,
                              k3loader._DT, set_param, k3loader.load_resident)
        TIMES["resident_io"] += time.time() - t0
        return
    (copy_resident if TEMPLATES else _apply_resident)(module, prefix, pack)


def _dev_sync():
    if DEV.type == "mps":
        torch.mps.synchronize()
    elif DEV.type == "cuda":
        torch.cuda.synchronize()


def causal_mask(T, dtype=None):
    dtype = dtype or DT
    m = torch.zeros(1, 1, T, T, dtype=dtype, device=DEV)
    m.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEV), 1),
                   torch.finfo(dtype).min)
    return m


def forward_pass(layers, cache, hidden, step, verbose=True):
    """hidden: [1, T, H] fp32. Returns logits [1, T, vocab]."""
    T = hidden.shape[1]
    if pilot.PILOT:
        pilot.init(config, DEV, _pilot_load, PFX)
        pilot.begin_pass(fetch_v2)
        if PREAD:
            fetch_v2.drop_prefetch()    # nothing from the previous pass is valid
            if pilot.ASYNC_DRAIN:
                pilot.install_async_drain(fetch_v2.reader())
    elif EXPERT_PREFETCH:
        global _PREV_SEL
        _PREV_SEL = dict(_LAST_SEL)     # last token's routing = this pass's oracle
        fetch_v2.drop_prefetch()        # nothing from the previous pass is valid
    mask = causal_mask(T + (cache.get_seq_length() or 0))[:, :, -T:, :] if T > 1 else None
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
        if PROFILE:
            _dev_sync()
        t0 = time.time()
        hidden, block_residual = layer(
            hidden, attention_mask=mask, position_ids=None,
            past_key_values=cache, use_cache=True, block_residual=block_residual)
        if PROFILE:
            _dev_sync()
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
    if PROFILE:
        mk = TIMES["moe_kernel"]
        print(f"[prof] KDA {PROF['kda']:.1f}s/{PROF['n_kda']} MLA {PROF['mla']:.1f}s/{PROF['n_mla']} "
              f"| moe_kernel {mk:.1f}s | fetch {TIMES['expert_fetch']:.1f}s "
              f"| apply {TIMES['resident_io']:.1f}s"
              f"| preload_wait {TIMES['preload_wait']:.1f}s", flush=True)
        rep = spine_fast.phase_report()
        if rep:
            print(rep, flush=True)
        rep = pilot.report(fetch_v2.stats if fetch_v2 is not None else None)
        if rep:
            print(rep, flush=True)
    # tail: output attn-res -> final norm -> lm_head
    tail = nn.Module()
    with torch.device("meta"):
        tail.output_attn_res_norm = ml.KimiRMSNorm(H, eps=config.rms_norm_eps)
        tail.output_attn_res_proj = nn.Linear(H, 1, bias=False)
        tail.norm = ml.KimiRMSNorm(H, eps=config.rms_norm_eps)
    materialize_resident(tail, PFX)
    apply_res = getattr(ml, "_apply_attn_res", None) or ml.KimiDecoderLayer._apply_attn_res
    flat = apply_res(hidden.view(-1, H), block_residual,
                     tail.output_attn_res_proj, tail.output_attn_res_norm)
    hidden = tail.norm(flat.view(1, T, H))
    t0 = time.time()
    global _LM_W
    if _LM_W is None:  # resident across tokens: 2.35-4.7GB on DEV, loaded once
        _LM_W = _load_int8("language_model.lm_head.weight") if SPINE == "int8" else None
        if _LM_W is None:
            _LM_W = k3loader.load_resident("language_model.lm_head.weight").to(DEV, DT)
    TIMES["resident_io"] += time.time() - t0
    logits = hidden @ _LM_W.T
    dematerialize(tail)
    return logits


_LM_W = None


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
    snap = {"rec": {}, "conv": {}, "mla": {}}
    for i in range(NL):
        if cache.recurrent_states[i] is not None:
            snap["rec"][i] = cache.recurrent_states[i].clone()
        if cache.conv_states[i] is not None:
            snap["conv"][i] = tuple(c.clone() for c in cache.conv_states[i])
        if cache.key_cache[i] is not None:
            snap["mla"][i] = cache.key_cache[i].shape[2]
    return snap


def restore_states(cache, snap):
    for i, t in snap["rec"].items():
        cache.recurrent_states[i] = t
    for i, c in snap["conv"].items():
        cache.conv_states[i] = c
    for i, L in snap["mla"].items():
        cache.key_cache[i] = cache.key_cache[i][:, :, :L].contiguous()
        cache.value_cache[i] = cache.value_cache[i][:, :, :L].contiguous()


EOS_ID = 163586  # <|end_of_msg|> — K3's generation stop token


def generate(layers, cache, embed, ids, max_new, spec=None, on_token=None,
             verbose_prefill=False, log=lambda *a: None):
    """Greedy generation (+ certified-lossless n-gram speculation).

    Shared by the CLI and the OpenAI-compatible server. Calls on_token(token_id)
    as each token is emitted. Returns the emitted token list; a speculative
    accept may emit one token past EOS_ID — callers trim at EOS_ID."""
    if spec is None:
        spec = os.environ.get("K3_SPEC", "1") == "1"
    generated = []

    def emit(t):
        generated.append(t)
        if on_token:
            on_token(t)

    _step_ctx["step"] = 0
    logits = forward_pass(layers, cache, embed(ids), step=0, verbose=verbose_prefill)
    emit(int(logits[0, -1].argmax()))
    s = 1
    while len(generated) < max_new:
        _step_ctx["step"] = s
        if os.environ.get("K3_PREFETCH", "1") == "1":
            prefetch_prev_token()
        t0 = time.time()
        draft = ngram_draft(ids + generated) if spec else None
        tag = ""
        if draft is not None:
            snap = snapshot_states(cache)
            logits = forward_pass(layers, cache, embed([generated[-1], draft]),
                                  step=s, verbose=False)
            n1 = int(logits[0, 0].argmax())
            if n1 == draft:
                emit(n1)
                emit(int(logits[0, 1].argmax()))
                tag = " spec+2"
            else:
                restore_states(cache, snap)
                logits = forward_pass(layers, cache, embed([generated[-1]]),
                                      step=s, verbose=False)
                emit(int(logits[0, -1].argmax()))
                tag = " spec-miss"
        else:
            logits = forward_pass(layers, cache, embed([generated[-1]]), step=s, verbose=False)
            emit(int(logits[0, -1].argmax()))
        log(s, tag, t0, list(generated))
        s += 1
        if EOS_ID in generated[-2:]:
            break
    return generated


TOTAL_EXPERTS = 82432
EXPERT_SPAN = 17547264


def check_expert_pool():
    """Streaming is the fallback, not the goal. Warn clearly when the expert pool
    isn't fully local, because every novel prompt pays for it over the network."""
    import shutil
    cache = os.path.join(ROOT, "k3-experts")
    try:
        n = sum(1 for f in os.listdir(cache) if f.endswith((".bin", ".npz")))
    except FileNotFoundError:
        n = 0
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
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "k3-meta"), trust_remote_code=True)
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      tokenize=True, add_generation_prompt=True)
    else:
        ids = tok.encode(args.prompt)
    check_expert_pool()
    print(f"prompt tokens ({len(ids)}): {ids}", flush=True)

    layers = build_layers()
    cache = ml.KimiDynamicCache(config)
    embed = LazyEmbed()
    t_start = time.time()
    max_new = args.max_new or 1_000_000   # effectively: until EOS or Ctrl-C
    if args.max_new is None and not args.chat:
        print("note: raw completions have no natural end — press Ctrl-C to stop "
              "cleanly, or pass --max-new N", flush=True)
    print(f"=== prefill: {len(ids)} tokens through 93 layers ===", flush=True)
    state = {"first": True}
    generated = []   # mirrored via on_token so Ctrl-C still has the text

    def on_token(t):
        generated.append(t)
        if state["first"]:
            state["first"] = False
            print(f"[prefill done in {time.time()-t_start:.0f}s] "
                  f"first token: {t!r} = {tok.decode([t])!r}", flush=True)

    def log(s, tag, t0, gen):
        print(f"[token {s}: {time.time()-t0:.0f}s{tag}] {tok.decode(gen)!r}", flush=True)
        print("   ", k3loader.cache_report(), flush=True)

    try:
        generate(layers, cache, embed, ids, max_new,
                 on_token=on_token, verbose_prefill=True, log=log)
    except KeyboardInterrupt:
        print("\n[stopped by Ctrl-C]", flush=True)
    if EOS_ID in generated:
        generated = generated[:generated.index(EOS_ID)]

    print("\n=== RESULT ===")
    print("completion:", tok.decode(generated))
    print(f"total {time.time()-t_start:.0f}s | times {TIMES}")
    print(k3loader.cache_report())
    TRACE.close()


if __name__ == "__main__":
    main()
