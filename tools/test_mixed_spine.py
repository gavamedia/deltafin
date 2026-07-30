#!/usr/bin/env python3
"""Low-level validation for the retired mixed-precision spine artifacts.

These codecs remain available for format research, but the supported Deltafin
runtime rejects them because they change target weights. Two low-level checks
remain useful without enabling the inference path:

  1. KERNEL     every Metal dequant kernel is bit-identical to the numpy oracle
                in tools/spine_codec.py (mixed_spine._check).
  2. TENSOR     for each converted tensor, ||mixed - int8|| / ||int8|| lands in
                the band its codec predicts (i4 ~0.108, i6 ~0.024, i8 exactly 0).
                A mis-indexed slice inside the packed layer buffer would show up
                here as ~1.4 (two unrelated tensors), not as a plausible error.

  python tools/test_mixed_spine.py
  K3_MIXED_DIR=... python tools/test_mixed_spine.py
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spine_codec          # noqa: E402
import mixed_spine          # noqa: E402

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT8_DIR = os.path.join(ROOT, "k3-resident-int8/tensors")
INV = json.load(open(os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")))

BAND = {"i4": (0.06, 0.16), "i6": (0.015, 0.045), "i8": (0.0, 0.0)}


def load_int8(full, dev, dtype=torch.float32):
    shape = INV[full]["shape"]
    with open(os.path.join(INT8_DIR, full + ".i8"), "rb") as f:
        q = torch.frombuffer(bytearray(f.read()), dtype=torch.int8).reshape(shape)
    with open(os.path.join(INT8_DIR, full + ".sc"), "rb") as f:
        sc = torch.frombuffer(bytearray(f.read()), dtype=torch.float16).reshape(shape[0], 1)
    return (q.to(dev).to(torch.float32) * sc.to(dev).to(torch.float32)).to(dtype)


def rel(a, b):
    return float(torch.linalg.norm((a - b).float()) / (torch.linalg.norm(b.float()) + 1e-30))


def test_tensors(dev, limit, layer=None):
    names = sorted({f.rsplit(".", 1)[0] for f in os.listdir(mixed_spine.MIXED_DIR)
                    if f.endswith((".i4", ".i6"))})
    if layer is not None:
        names = [n for n in names if f".layers.{layer}." in n]
    if not names:
        print("  no converted tensors found — nothing to compare")
        return True, {}
    step = max(1, len(names) // limit)
    names = names[::step][:limit]
    by_codec, ok = {}, True
    for n in names:
        if not os.path.exists(os.path.join(INT8_DIR, n + ".i8")):
            continue
        codec, _ = mixed_spine.codec_of(n, INT8_DIR)
        m = mixed_spine.load(n, INT8_DIR, INV[n]["shape"], dev=dev)
        r = rel(m, load_int8(n, dev))
        by_codec.setdefault(codec, []).append(r)
        lo, hi = BAND[codec]
        if not (lo <= r <= hi + 1e-12):
            ok = False
            print(f"  OUT OF BAND {codec} {n}: rel={r:.4f} not in [{lo},{hi}]")
        del m
    for c, v in sorted(by_codec.items()):
        v = np.array(v)
        print(f"  {c}: {v.size:3d} tensors, rel vs int8 min {v.min():.4f} / "
              f"median {np.median(v):.4f} / max {v.max():.4f}  "
              f"(band {BAND[c][0]}-{BAND[c][1]})")
    return ok, by_codec


def test_layer(idx, dev):
    """Materialize one real decoder layer through the packed path."""
    import kimi_run as kr
    if not kr.MIXED:
        print("  K3_SPINE is not 'mixed' — layer test skipped")
        return True
    layers = kr.build_layers()
    mod = layers[idx]
    prefix = f"{kr.PFX}layers.{idx}."
    pack = kr._spine_read(mod, prefix)
    kr._spine_apply(mod, prefix, pack)
    got = {n: p.detach().clone() for n, p in mod.named_parameters()
           if p.device.type != "meta" and ".experts." not in n}
    ok, worst = True, ("", 0.0)
    for n, t in sorted(got.items()):
        full = prefix + n
        codec, _ = mixed_spine.codec_of(full, INT8_DIR)
        if codec is None:                     # bf16 tail (norms, convs, ...)
            ref = kr.k3loader.load_resident(full).to(t.device, t.dtype)
            r = rel(t, ref)
            if r > 1e-6:
                ok = False
                print(f"  bf16 tail MISMATCH {n}: rel={r:.3e}")
            continue
        ref = load_int8(full, t.device, t.dtype)
        r = rel(t, ref)
        lo, hi = BAND[codec]
        flag = "" if lo <= r <= hi + 1e-12 else "   <-- OUT OF BAND"
        if flag:
            ok = False
        if r > worst[1]:
            worst = (n, r)
        print(f"  {codec} {n:52s} rel_vs_int8={r:.4f}{flag}")
    print(f"  layer {idx}: {len(got)} params, worst {worst[0]} {worst[1]:.4f}")
    kr.dematerialize(mod)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="decoder layer index for the layer-level test "
                         "(default: the lowest converted layer)")
    ap.add_argument("--tensors", type=int, default=40, help="tensors to spot-check")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--skip-layer", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.dev or ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[mixed] dir={mixed_spine.MIXED_DIR}")
    print(f"[mixed] {mixed_spine.describe()}")
    print("\n1. KERNEL bit-exactness vs the numpy oracle")
    ok1 = mixed_spine._check(dev)

    print("\n2. TENSOR error bands vs the int8 spine")
    ok2, _ = test_tensors(dev, args.tensors)

    ok3 = True
    if not args.skip_layer:
        idx = args.layer
        if idx is None:
            ls = sorted({int(f.split(".layers.")[1].split(".")[0])
                         for f in os.listdir(mixed_spine.MIXED_DIR)
                         if ".layers." in f and f.endswith((".i4", ".i6"))})
            idx = ls[0] if ls else 1
        print(f"\n3. LAYER {idx} through read_pack/apply_pack")
        ok3 = test_layer(idx, dev)

    print("\nRESULT " + ("PASS" if (ok1 and ok2 and ok3) else "FAIL"))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
