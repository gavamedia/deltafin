#!/usr/bin/env python3
"""Unit tests for the tracked geometric KV-cache runtime shim."""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_cache  # noqa: E402


class _Config:
    num_hidden_layers = 2


class _PristineCache:
    """Small stand-in for Moonshot's unmodified dynamic cache API."""

    def __init__(self, config):
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        self.upstream_updates = 0
        self.upstream_reorders = 0

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        self.upstream_updates += 1
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key_states), dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_states), dim=2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx):
        self.upstream_reorders += 1
        for layer_idx, key in enumerate(self.key_cache):
            if key is not None:
                self.key_cache[layer_idx] = key.index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[
                    layer_idx
                ].index_select(0, beam_idx)
        return "upstream-result"

    def get_seq_length(self, layer_idx=0):
        key = self.key_cache[layer_idx]
        return 0 if key is None else key.shape[2]


def _ptr(tensor):
    return tensor.untyped_storage().data_ptr()


class GeometricCacheInstallTests(unittest.TestCase):
    def test_pristine_cache_is_patched_with_complete_slab_bookkeeping(self):
        modeling = types.ModuleType("synthetic_pristine_modeling")
        modeling.KimiDynamicCache = _PristineCache

        self.assertFalse(hasattr(_PristineCache, "_k3_geometric_slab"))
        self.assertFalse(hasattr(_PristineCache, "truncate_mla"))
        self.assertTrue(kv_cache.install(modeling))
        self.assertFalse(kv_cache.install(modeling))

        patched = modeling.KimiDynamicCache
        self.assertIsNot(patched, _PristineCache)
        self.assertTrue(issubclass(patched, _PristineCache))
        self.assertTrue(patched._k3_geometric_slab)
        self.assertIs(patched._k3_original_cache_class, _PristineCache)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "K3_KV_SLAB": "1",
                    "K3_KV_SLAB_GROWTH": "1.5",
                },
            ),
            torch.inference_mode(),
        ):
            cache = patched(_Config())
            first_k = torch.arange(16, dtype=torch.float32).reshape(
                2, 1, 4, 2
            )
            first_v = torch.arange(24, dtype=torch.float32).reshape(
                2, 1, 4, 3
            )
            first_view_k, first_view_v = cache.update(
                first_k, first_v, 0
            )
            self.assertEqual(cache.upstream_updates, 0)
            self.assertEqual(cache._kv_capacity[0], 16)
            first_ptr = _ptr(first_view_k)

            fill_k = torch.arange(
                48, dtype=torch.float32
            ).reshape(2, 1, 12, 2) + 100
            fill_v = torch.arange(
                72, dtype=torch.float32
            ).reshape(2, 1, 12, 3) + 200
            full_k, full_v = cache.update(fill_k, fill_v, 0)
            self.assertEqual(_ptr(full_k), first_ptr)
            self.assertEqual(cache._kv_capacity[0], 16)
            full_k_bits = full_k.clone()
            full_v_bits = full_v.clone()

            extra_k = torch.tensor([[[[900.0, 901.0]]],
                                    [[[902.0, 903.0]]]])
            extra_v = torch.tensor([[[[910.0, 911.0, 912.0]]],
                                    [[[913.0, 914.0, 915.0]]]])
            grown_k, grown_v = cache.update(extra_k, extra_v, 0)
            expected_k = torch.cat((first_k, fill_k, extra_k), dim=2)
            expected_v = torch.cat((first_v, fill_v, extra_v), dim=2)
            self.assertNotEqual(_ptr(grown_k), first_ptr)
            self.assertEqual(cache._kv_capacity[0], 24)
            self.assertTrue(torch.equal(grown_k, expected_k))
            self.assertTrue(torch.equal(grown_v, expected_v))
            self.assertEqual(cache.get_seq_length(0), 17)

            snapshot_k, snapshot_v = grown_k, grown_v
            snapshot_k_bits = snapshot_k.clone()
            snapshot_v_bits = snapshot_v.clone()
            snapshot_ptr = _ptr(snapshot_k)
            cache.truncate_mla(0, 6)
            self.assertEqual(cache.get_seq_length(0), 6)
            self.assertEqual(cache._kv_capacity[0], 24)
            self.assertEqual(_ptr(cache.key_cache[0]), snapshot_ptr)
            self.assertTrue(
                torch.equal(cache.key_cache[0], snapshot_k_bits[:, :, :6])
            )

            cache.restore_mla(0, snapshot_k, snapshot_v)
            self.assertEqual(cache.get_seq_length(0), 17)
            self.assertEqual(cache._kv_capacity[0], 24)
            self.assertTrue(torch.equal(cache.key_cache[0], snapshot_k_bits))
            self.assertTrue(
                torch.equal(cache.value_cache[0], snapshot_v_bits)
            )

            # Roll back across the allocation-growth boundary. The shim must
            # adopt the older backing allocation and regrow from its exact
            # contents rather than copying from the abandoned speculative one.
            cache.restore_mla(0, full_k, full_v)
            self.assertIs(cache._key_storage[0], full_k)
            self.assertIs(cache._value_storage[0], full_v)
            self.assertEqual(cache._kv_capacity[0], 16)
            self.assertTrue(torch.equal(cache.key_cache[0], full_k_bits))
            self.assertTrue(torch.equal(cache.value_cache[0], full_v_bits))
            snapshot_k, snapshot_v = cache.update(extra_k, extra_v, 0)
            snapshot_k_bits = snapshot_k.clone()
            snapshot_v_bits = snapshot_v.clone()
            self.assertEqual(cache._kv_capacity[0], 24)
            self.assertTrue(torch.equal(snapshot_k, expected_k))
            self.assertTrue(torch.equal(snapshot_v, expected_v))

            beam_idx = torch.tensor([1, 0, 1], dtype=torch.long)
            result = cache.reorder_cache(beam_idx)
            self.assertEqual(result, "upstream-result")
            self.assertEqual(cache.upstream_reorders, 1)
            self.assertTrue(
                torch.equal(
                    cache.key_cache[0],
                    snapshot_k_bits.index_select(0, beam_idx),
                )
            )
            self.assertTrue(
                torch.equal(
                    cache.value_cache[0],
                    snapshot_v_bits.index_select(0, beam_idx),
                )
            )
            self.assertIs(cache._key_storage[0], cache.key_cache[0])
            self.assertIs(cache._value_storage[0], cache.value_cache[0])
            self.assertEqual(cache._kv_capacity[0], 17)
            self.assertIsNone(cache._key_storage[1])
            self.assertIsNone(cache._value_storage[1])
            self.assertEqual(cache._kv_capacity[1], 0)


if __name__ == "__main__":
    unittest.main()
