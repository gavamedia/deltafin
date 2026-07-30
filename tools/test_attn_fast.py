"""Decode-path numeric gate for tools/attn_fast.py.

ab_gate.py only ever runs a PREFILL (T=5), so it does not touch the T=1
branches K3_SHORTCONV / K3_KDA_RECUR select. This runs 4 decode steps through
one real-shape KDA layer and one real-shape MLA layer (random weights, fixed
seeds) and reports the worst per-tensor relative difference against the
reference path, including the recurrent and conv states.

    python tools/test_attn_fast.py                 # shortconv / recur
    python tools/test_attn_fast.py compile [mode]  # + K3_COMPILE=attn
"""
import sys, json, time, importlib
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/tools")
import torch, torch.nn as nn
torch.set_grad_enabled(False)
from k3pkg import modeling_kimi_linear as ml
import fla.modules as fm
import fla.ops.kda as fk
import attn_fast
CFG = json.load(open(ROOT + "/k3-meta/config.json"))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None) or importlib.import_module(
    "k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG); config._attn_implementation = "eager"
DEV = torch.device(os.environ.get("K3_DEV") or (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu")); H = config.hidden_size


def set_param(root, dotted, tensor):
    obj = root
    for p in dotted.split(".")[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, dotted.split(".")[-1], nn.Parameter(tensor, requires_grad=False))


torch.manual_seed(0)
with torch.device("meta"):
    kda = ml.KimiDeltaAttention(config, layer_idx=1).eval()
    mla = ml.KimiMLAAttention(config, layer_idx=4).eval()
for m in (kda, mla):
    for n, p in list(m.named_parameters()):
        g = torch.Generator().manual_seed(abs(hash(n)) % 2**31)
        set_param(m, n, (torch.randn(p.shape, generator=g) * 0.02).to(DEV))
set_param(kda, "A_log", (torch.randn(128, generator=torch.Generator().manual_seed(7)) * 0.1).to(DEV))
XS = [(torch.randn(1, 1, H, generator=torch.Generator().manual_seed(100 + i))).to(DEV)
      for i in range(4)]


def run():
    c = ml.KimiDynamicCache(config)
    c.key_cache[4] = (torch.randn(1, 96, 5, 192,
                      generator=torch.Generator().manual_seed(11)) * 0.02).to(DEV)
    c.value_cache[4] = (torch.randn(1, 96, 5, 128,
                        generator=torch.Generator().manual_seed(12)) * 0.02).to(DEV)
    outs = []
    for x in XS:
        outs.append(kda(hidden_states=x, cache_params=c).float().cpu())
        outs.append(mla(hidden_states=x, past_key_values=c).float().cpu())
    outs.append(c.recurrent_states[1].float().cpu())
    outs.append(torch.cat([t.float().cpu().flatten() for t in c.conv_states[1]]))
    return outs


NAMES = [f"{k}{i}" for i in range(4) for k in ("kda_out", "mla_out")] + ["rec_state", "conv_state"]


def cmp(tag, a, b):
    worst, wname, wscale = 0.0, "", 1.0
    for n, x, y in zip(NAMES, a, b):
        d = float((x - y).abs().max()); s = float(x.abs().max())
        if d / max(s, 1e-30) > worst / max(wscale, 1e-30):
            worst, wname, wscale = d, n, s
    print(f"  {tag:34s} worst rel {worst/max(wscale,1e-30):.2e} on {wname} "
          f"(|diff| {worst:.3e} vs scale {wscale:.4f})"
          f"{'  BIT-EXACT' if worst == 0 else ''}", flush=True)


fm.SHORTCONV = "conv1d"
fk.RECUR = "cpu"
print("reference (conv1d, cpu recurrence)", flush=True)
ref = run()
print(f"  ref output scale = {float(ref[0].abs().max()):.4f}")

fm.SHORTCONV = "mulsum"
cmp("K3_SHORTCONV=mulsum", ref, run())
fm.SHORTCONV = "conv1d"

fk.RECUR = "device"
cmp("K3_KDA_RECUR=device", ref, run())

fm.SHORTCONV = "mulsum"
cmp("both", ref, run())

if len(sys.argv) > 1 and sys.argv[1] == "compile":
    attn_fast.COMPILE = "attn"
    attn_fast.COMPILE_MODE = sys.argv[2] if len(sys.argv) > 2 else "default"
    t0 = time.time()
    attn_fast.install(ml)
    got = run()
    print(f"  [compile] first pass {time.time()-t0:.1f}s mode={attn_fast.COMPILE_MODE} "
          f"graphs={attn_fast.STATS['graphs']}", flush=True)
    cmp("both + K3_COMPILE=attn", ref, got)
    t0 = time.time(); run(); print(f"  [compile] warm pass {time.time()-t0:.2f}s")
