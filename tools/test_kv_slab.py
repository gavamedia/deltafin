#!/usr/bin/env python3
"""N20 correctness oracle for geometric MLA KV slabs and rollback."""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_cache  # noqa: E402
from k3pkg import modeling_kimi_linear as ml  # noqa: E402

kv_cache.install(ml)
KimiDynamicCache = ml.KimiDynamicCache


class Config:
    linear_attn_config = None
    num_hidden_layers = 2


def ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", choices=("cpu", "mps"),
        default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable")
    dev = args.device
    torch.manual_seed(20260728)

    with torch.inference_mode():
        cache = KimiDynamicCache(Config())
        first_k = torch.randn(1, 2, 5, 3, device=dev)
        first_v = torch.randn(1, 2, 5, 4, device=dev)
        got_k, got_v = cache.update(first_k, first_v, 0)
        assert torch.equal(got_k, first_k)
        assert torch.equal(got_v, first_v)
        assert cache._kv_capacity[0] >= 16

        snap_k, snap_v = got_k, got_v
        snap_k_bits, snap_v_bits = snap_k.clone(), snap_v.clone()
        extra_k = torch.randn(1, 2, 3, 3, device=dev)
        extra_v = torch.randn(1, 2, 3, 4, device=dev)
        got_k, got_v = cache.update(extra_k, extra_v, 0)
        assert ptr(got_k) == ptr(snap_k)
        assert torch.equal(got_k, torch.cat((first_k, extra_k), dim=2))
        assert torch.equal(got_v, torch.cat((first_v, extra_v), dim=2))
        assert torch.equal(snap_k, snap_k_bits)
        assert torch.equal(snap_v, snap_v_bits)

        cache.truncate_mla(0, 6)
        assert torch.equal(
            cache.key_cache[0],
            torch.cat((first_k, extra_k[:, :, :1]), dim=2))

        cache.restore_mla(0, snap_k, snap_v)
        assert torch.equal(cache.key_cache[0], first_k)
        replacement_k = torch.randn(1, 2, 2, 3, device=dev)
        replacement_v = torch.randn(1, 2, 2, 4, device=dev)
        got_k, got_v = cache.update(replacement_k, replacement_v, 0)
        expected_k = torch.cat((first_k, replacement_k), dim=2)
        expected_v = torch.cat((first_v, replacement_v), dim=2)
        assert torch.equal(got_k, expected_k)
        assert torch.equal(got_v, expected_v)

        # Force a capacity growth, then restore the old allocation and append.
        old_k, old_v = got_k, got_v
        old_k_bits, old_v_bits = old_k.clone(), old_v.clone()
        grow_k = torch.randn(1, 2, 20, 3, device=dev)
        grow_v = torch.randn(1, 2, 20, 4, device=dev)
        grown_k, _ = cache.update(grow_k, grow_v, 0)
        assert ptr(grown_k) != ptr(old_k)
        assert torch.equal(old_k, old_k_bits)
        assert torch.equal(old_v, old_v_bits)
        cache.restore_mla(0, old_k, old_v)
        final_k = torch.randn(1, 2, 1, 3, device=dev)
        final_v = torch.randn(1, 2, 1, 4, device=dev)
        restored_k, restored_v = cache.update(final_k, final_v, 0)
        assert torch.equal(restored_k, torch.cat((old_k_bits, final_k), dim=2))
        assert torch.equal(restored_v, torch.cat((old_v_bits, final_v), dim=2))

    # Training/grad-enabled use retains the upstream functional torch.cat path.
    training_cache = KimiDynamicCache(Config())
    train_k = torch.randn(1, 2, 2, 3, device=dev, requires_grad=True)
    train_v = torch.randn(1, 2, 2, 4, device=dev, requires_grad=True)
    one_k, _ = training_cache.update(train_k, train_v, 0)
    two_k, _ = training_cache.update(train_k, train_v, 0)
    assert ptr(one_k) != ptr(two_k)
    assert two_k.shape[2] == 4

    if dev == "mps":
        torch.mps.synchronize()
    print(
        f"PASS KV slab on {dev}: in-capacity append, truncate, "
        "pre/post-growth restore, and grad fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
