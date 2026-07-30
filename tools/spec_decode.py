"""Multi-token speculative decode for lazy-K3.

WHY
---
The 53 GB int8 spine is re-read from disk once per FORWARD PASS, not once per
token: ~7.5 s of a ~15 s decode token.  A T=D+1 pass pays that read exactly
once.  So the whole game is "emit as many verified tokens as possible per
pass".  Today's shipped speculation drafts one token and verifies a T=2 batch;
this module generalises that to D drafts verified in one T=D+1 pass, accepting
the longest correct prefix (standard speculative decoding, greedy verifier =
lossless by construction).

THE CRUX: PARTIAL-ACCEPT ROLLBACK
---------------------------------
A pass consumes T=D+1 positions but on a partial accept only k+1 of them are
real.  The cache has to be left exactly as if only those k+1 had been fed.

  * MLA (full attention) is trivially truncatable: `KimiDynamicCache.update`
    cats along dim=2, and this checkpoint is NoPE (`mla_use_nope`, rotary_emb
    is None), so a key/value row depends only on its own hidden state.  Cutting
    to `pre_len + k + 1` is exact.

  * KDA (linear attention) is a fold.  `cache.recurrent_states[i]` after the
    pass is S_T; there is no S_{k+1} to truncate to, and the delta rule is not
    invertible in fp32.  Whole-tensor restore (what `restore_states` does) can
    only go back to S_0.

Three ways out; this file implements the first two.

  replay  (K3_SPEC_ROLLBACK=replay, default)
      Capture the *inputs* to each KDA layer's recurrence during the pass --
      q,k,v,g,beta [B,T,H,*] plus the pre-pass `initial_state` reference and the
      three raw short-conv inputs.  That is ~340 KB x T per KDA layer (~120 MB
      at T=5 over 69 layers), versus 6.3 MB x 69 = 434 MB for one state clone.
      On a partial accept, re-run the SAME fla kernel on the first k+1 columns
      from the same initial_state.
        Why it is exact: `_kda_core` is a plain sequential loop over t whose
        step depends only on (S, position-t inputs); every per-position
        transform (l2norm, sigmoid, gate, scale) is elementwise in t.  So the
        first k+1 iterations of the T-column run and the k+1-column run are the
        same arithmetic in the same order on the same bits -> bit-identical
        S_{k+1}.  (Verified numerically in tools/test_spec_replay.py.)
      The conv state needs no replay at all: ShortConvolution's cache is
      literally "the last W raw inputs", so the state after position j is
      cat([pre_state, x])[:, :, j+1 : j+1+W] -- a pure window of tensors we
      already hold.  Same identity is used by both the conv1d and the mulsum
      branches of the shim, so this is path-independent.
      Cost: one recurrence (no projections, no spine, no experts) over <=D+1
      columns on the 69 KDA layers, only when 0 <= k < D.

  rerun   (K3_SPEC_ROLLBACK=rerun)
      Snapshot with the existing certified `snapshot_states`, and on a partial
      accept `restore_states` + one extra forward pass over the k+1 accepted
      tokens.  Uses no new machinery -- it is exactly today's "spec-miss" redo
      generalised -- so it is the reference the `replay` path is checked
      against.  Costs a second spine read whenever k < D.

  (not implemented) strict all-or-nothing: accept only k==D.  Acceptance decays
  geometrically in D, so it wastes the whole point of a deeper draft.

NUMERICS PARITY
---------------
`fla/ops/kda/__init__.py` picks the CPU recurrence path only when
`q.shape[1] <= 4 and K3_KDA_RECUR == "cpu"`. Under the shipped default
(K3_KDA_RECUR unset -> "device") every T stays on its selected device and there
is nothing to reconcile. But under K3_KDA_RECUR=cpu an MPS T=1 or
T=2 decode runs the recurrence on CPU while a T>=5 pass would silently switch
to MPS -- different fp32 reduction order, so a near-tie token could flip and the
"same tokens as K3_SPEC=0" guarantee would be lost.  K3_SPEC_RECUR_MATCH=1
(default) therefore moves the recurrence inputs to CPU itself whenever that
would have happened, reproducing the shim's CPU branch exactly (it also returns
the state CPU-resident, as that branch does).

The portable short-convolution path keeps cache state bit-identical and output
inside a strict few-ulp gate for one T-position call versus T repeated
one-position calls. It is automatic through T=9 on CPU and forceable elsewhere.
Dense projections around it can also select shape-dependent floating-point
reductions, so production gates compare emitted token IDs with fresh sequential
controls rather than claiming every batched hidden tensor is byte-identical.

FLAGS
-----
K3_SPEC_DEPTH       1 (default, = today's code path, untouched) .. 8
K3_SPEC_ADAPT       0 (default) | 1   shrink depth when drafts keep missing
K3_SPEC_ROLLBACK    replay (default) | rerun
K3_SPEC_MIN_N       2 (default) shortest n-gram suffix allowed as a draft source
K3_SPEC_MAX_N       6 (default) longest
K3_SPEC_RECUR_MATCH 1 (default) | 0   see NUMERICS PARITY
K3_SPEC_FUZZ        0 (default) | 1   correctness harness: poison drafts from a
                    rotating index so k cycles over 0..D-1 and every rollback
                    depth is exercised. Output MUST be unchanged; that is the
                    whole claim being tested.
"""
import math
import os

import torch

DEPTH = max(1, min(8, int(os.environ.get("K3_SPEC_DEPTH", "1"))))
ADAPT = os.environ.get("K3_SPEC_ADAPT", "0") == "1"
ROLLBACK = os.environ.get("K3_SPEC_ROLLBACK", "replay").lower()
if ROLLBACK not in {"replay", "rerun"}:
    raise ValueError("K3_SPEC_ROLLBACK must be replay or rerun")
MIN_N = int(os.environ.get("K3_SPEC_MIN_N", "2"))
MAX_N = int(os.environ.get("K3_SPEC_MAX_N", "6"))
RECUR_MATCH = os.environ.get("K3_SPEC_RECUR_MATCH", "1") == "1"
FUZZ = os.environ.get("K3_SPEC_FUZZ", "0") == "1"   # correctness harness only

STATS = {"passes": 0, "drafted": 0, "accepted": 0, "emitted": 0,
         "full_accepts": 0, "replays": 0, "reruns": 0, "nodraft": 0}

_FORCE_CPU = False          # set by install(): mirror the shim's CPU branch
_CONV_W = 4                 # short_conv_kernel_size; refreshed by install()
_CAP = None                 # None = capture disarmed; else {layer_idx: _Rec}
_layer_idx = lambda: -1     # noqa: E731  injected by install()


def enabled():
    return DEPTH != 1


def describe():
    return (f"depth={DEPTH} adapt={int(ADAPT)} rollback={ROLLBACK} "
            f"ngram={MIN_N}..{MAX_N} recur_match={int(_FORCE_CPU)}"
            + (" FUZZ" if FUZZ else ""))


# --- capture ------------------------------------------------------------------
class _Rec:
    __slots__ = ("fn", "kw", "conv")

    def __init__(self):
        self.fn = None
        self.kw = None
        self.conv = []      # [(x, pre_cache)] in call order: q, k, v


def arm():
    """Start recording KDA recurrence inputs. Stays armed after the forward pass
    returns, because `rollback_replay` reads the records; the caller must
    `release()` in a finally."""
    global _CAP
    if _CAP is not None:
        raise RuntimeError("spec_decode capture is already armed")
    _CAP = {}


def release():
    """Drop the captured tensors. They pin ~0.6 GB at T=5: the 69 pre-pass
    recurrent states (6.3 MB each) plus ~340 KB x T of per-layer inputs."""
    global _CAP
    _CAP = None


def install(ml, layer_idx_fn, conv_kernel_size=4, compiled=False):
    """Wrap the three fla entry points KDA state flows through.

    Nothing here changes behaviour while capture is disarmed (`_CAP is None`),
    which is always the case at K3_SPEC_DEPTH=1 and during prefill.
    """
    global _FORCE_CPU, _CONV_W, _layer_idx, ROLLBACK
    _layer_idx = layer_idx_fn
    _CONV_W = conv_kernel_size

    try:
        import fla.ops.kda as _kda_ops
        _FORCE_CPU = RECUR_MATCH and getattr(_kda_ops, "RECUR", "mps") == "cpu"
    except Exception:
        _FORCE_CPU = False

    if compiled and ROLLBACK == "replay":
        # Under K3_COMPILE the KDA body is traced by dynamo; a Python-side
        # append of intermediate tensors would be baked into the graph or break
        # it. `rerun` needs no capture at all, so degrade to that.
        print("[spec] K3_COMPILE is on -> K3_SPEC_ROLLBACK=rerun "
              "(capture cannot be traced)", flush=True)
        ROLLBACK = "rerun"

    def _wrap_kernel(orig):
        def kernel(*a, **kw):
            if _CAP is None:               # capture disarmed
                return orig(*a, **kw)      # -> byte-for-byte the shipped path
            if a:
                # A positional call would leave nothing to replay from and the
                # rollback would silently keep an over-advanced state. The
                # modeling code calls these keyword-only; fail loudly if not.
                raise RuntimeError("spec_decode: KDA kernel called positionally "
                                   "during a speculative pass; cannot capture")
            rec = _CAP.setdefault(_layer_idx(), _Rec())
            rec.fn, rec.kw = orig, dict(kw)
            return _run(orig, kw)
        return kernel

    ml.chunk_kda = _wrap_kernel(ml.chunk_kda)
    ml.fused_recurrent_kda = _wrap_kernel(ml.fused_recurrent_kda)

    _orig_sc = ml.ShortConvolution.forward

    def sc_forward(self, *a, **kw):
        if _CAP is not None:
            x = kw.get("x", a[0] if a else None)
            pre = kw.get("cache", a[3] if len(a) > 3 else None)
            rec = _CAP.setdefault(_layer_idx(), _Rec())
            rec.conv.append((x, pre))
        return _orig_sc(self, *a, **kw)

    ml.ShortConvolution.forward = sc_forward


def _run(fn, kw):
    """Call an fla KDA kernel, keeping the T<=4 CPU branch's arithmetic when the
    batch is too long to trigger it (see NUMERICS PARITY).

    Only ever reached with capture armed, i.e. inside a speculative verify pass
    or its replay -- prefill and K3_SPEC_DEPTH=1 decode never come through here.
    """
    q = kw.get("q")
    if not (_FORCE_CPU and torch.is_tensor(q) and q.device.type == "mps"):
        return fn(**kw)
    # Route to CPU for EVERY T, not just T>4: the verify pass and its replay
    # must land on one arithmetic path, and hard-coding the shim's own "<= 4"
    # threshold here would silently break if that threshold ever moved.
    dev, vdt = q.device, kw["v"].dtype
    ckw = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kw.items()}
    o, S = fn(**ckw)
    # the shim's own CPU branch returns o on the original device and leaves the
    # recurrent state CPU-resident; match both.
    return o.to(dev, vdt), S


# --- rollback -----------------------------------------------------------------
def snapshot_mla(cache, NL):
    return {i: cache.key_cache[i].shape[2] for i in range(NL)
            if cache.key_cache[i] is not None}


def truncate_mla(cache, mla_len, keep):
    for i, L in mla_len.items():
        if hasattr(cache, "truncate_mla"):
            cache.truncate_mla(i, L + keep)
        else:
            cache.key_cache[i] = cache.key_cache[i][
                :, :, :L + keep].contiguous()
            cache.value_cache[i] = cache.value_cache[i][
                :, :, :L + keep].contiguous()


def _replay_layer(rec, keep):
    """State after `keep` positions of the captured pass. Returns (S, convs)."""
    kw = dict(rec.kw)
    for name in ("q", "k", "v", "g", "beta"):
        t = kw.get(name)
        if torch.is_tensor(t):
            kw[name] = t[:, :keep]
    kw["output_final_state"] = True
    _, S = _run(rec.fn, kw)
    convs = []
    for x, pre in rec.conv:
        B, T, D = x.shape
        xt = x.transpose(1, 2).to(torch.float32)                 # [B, D, T]
        W = pre.shape[-1] if torch.is_tensor(pre) else _CONV_W
        base = pre.to(torch.float32) if torch.is_tensor(pre) \
            else xt.new_zeros(B, D, W)
        full = torch.cat([base, xt], dim=-1)                     # [B, D, W+T]
        convs.append(full[:, :, keep:keep + W].contiguous())
    return S, tuple(convs)


def rollback_replay(cache, keep, mla_len):
    """Leave `cache` exactly as if only the first `keep` positions of the
    captured pass had been fed."""
    truncate_mla(cache, mla_len, keep)
    for li, rec in _CAP.items():
        if rec.fn is None:
            # Convs ran but the recurrence was not captured -> the state moved
            # and we cannot rebuild it. Never skip silently.
            assert not rec.conv, f"layer {li}: short-conv captured, recurrence not"
            continue
        assert len(rec.conv) == 3, (
            f"layer {li}: captured {len(rec.conv)} short-conv calls, expected "
            f"q/k/v — refusing to roll back a state I cannot rebuild")
        S, convs = _replay_layer(rec, keep)
        cache.recurrent_states[li] = S
        cache.conv_states[li] = convs
    STATS["replays"] += 1


# --- drafting -----------------------------------------------------------------
_fuzz_n = 0
_FUZZ_TOK = 96562          # arbitrary in-vocab id, only used by K3_SPEC_FUZZ


def draft(ctx, depth, lookup):
    """D speculative tokens; each extends the context the next lookup sees."""
    out = []
    cur = list(ctx)
    for _ in range(depth):
        t = lookup(cur, MAX_N, MIN_N)
        if t is None:
            break
        out.append(t)
        cur.append(t)
    if FUZZ and out:
        # Correctness harness: poison the draft from a rotating position so that
        # k cycles over 0..D-1 across passes and every partial-accept rollback
        # depth is exercised. A rejected draft may cost time but must NEVER
        # change a token, so a fuzzed run has to emit the reference sequence.
        global _fuzz_n
        j = _fuzz_n % len(out)
        _fuzz_n += 1
        out = out[:j] + [_FUZZ_TOK + i for i in range(len(out) - j)]
    return out


# --- adaptive depth -----------------------------------------------------------
_ema = None


def next_depth():
    """Effective draft length for the coming pass.

    A rejected draft position costs only compute (the spine read is already
    paid), so aim one position past what recent history justifies; a run of
    total misses drives this to 1 and it climbs back as soon as drafts land.
    """
    if not ADAPT or _ema is None:
        return DEPTH
    return max(1, min(DEPTH, int(math.ceil(_ema)) + 1))


def record(accepted, drafted, emitted):
    global _ema
    STATS["passes"] += 1
    STATS["drafted"] += drafted
    STATS["accepted"] += accepted
    STATS["emitted"] += emitted
    if drafted and accepted == drafted:
        STATS["full_accepts"] += 1
    if not drafted:
        STATS["nodraft"] += 1
    _ema = float(accepted) if _ema is None else 0.5 * _ema + 0.5 * accepted


def report():
    s = STATS
    if not s["passes"]:
        return ""
    rate = (s["accepted"] / s["drafted"]) if s["drafted"] else 0.0
    return (f"[spec] {describe()} | passes {s['passes']} "
            f"tokens {s['emitted']} ({s['emitted']/s['passes']:.2f}/pass) | "
            f"drafts {s['accepted']}/{s['drafted']} accepted ({rate*100:.0f}%) "
            f"| full {s['full_accepts']} nodraft {s['nodraft']} "
            f"replays {s['replays']} reruns {s['reruns']}")
