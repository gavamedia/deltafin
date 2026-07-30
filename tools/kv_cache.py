"""Tracked geometric MLA KV-cache extension for Moonshot's model class.

Modeling sources downloaded by ``setup_k3.py`` remain third-party, ignored
files. This module installs Deltafin's inference-only cache policy at runtime
instead of modifying those files in place, so setup/upgrade cannot silently
erase the optimization.
"""
from __future__ import annotations

import math
import os

import torch


def install(modeling_module) -> bool:
    """Replace ``KimiDynamicCache`` with an API-compatible slab subclass."""
    original = getattr(modeling_module, "KimiDynamicCache", None)
    if original is None:
        raise RuntimeError("modeling module has no KimiDynamicCache")
    if getattr(original, "_k3_geometric_slab", False):
        return False
    for method in ("update", "reorder_cache", "get_seq_length"):
        if not callable(getattr(original, method, None)):
            raise RuntimeError(
                f"KimiDynamicCache lacks required method {method}"
            )

    class GeometricKimiDynamicCache(original):
        _k3_geometric_slab = True
        _k3_original_cache_class = original

        def __init__(self, config):
            super().__init__(config)
            count = len(self.key_cache)
            self._kv_slab = os.environ.get("K3_KV_SLAB", "1") == "1"
            self._kv_growth = max(
                1.1,
                float(os.environ.get("K3_KV_SLAB_GROWTH", "1.5")),
            )
            self._key_storage = [None for _ in range(count)]
            self._value_storage = [None for _ in range(count)]
            self._kv_capacity = [0 for _ in range(count)]

        def update(
            self,
            key_states,
            value_states,
            layer_idx,
            cache_kwargs=None,
        ):
            # Preserve upstream autograd behavior. In inference, append only
            # the new rows to a geometrically grown backing allocation.
            if not self._kv_slab or torch.is_grad_enabled():
                return super().update(
                    key_states, value_states, layer_idx, cache_kwargs
                )

            old_key = self.key_cache[layer_idx]
            old_value = self.value_cache[layer_idx]
            old_len = 0 if old_key is None else old_key.shape[2]
            needed = old_len + key_states.shape[2]
            key_store = self._key_storage[layer_idx]
            value_store = self._value_storage[layer_idx]
            capacity = self._kv_capacity[layer_idx]

            if old_key is not None and key_store is not None:
                same_backing = (
                    old_key.untyped_storage().data_ptr()
                    == key_store.untyped_storage().data_ptr()
                    and old_value.untyped_storage().data_ptr()
                    == value_store.untyped_storage().data_ptr()
                )
                if not same_backing:
                    # A speculative restore can return to an older allocation.
                    # Treat that exact view as the new source and regrow.
                    key_store = value_store = None
                    capacity = old_len

            shapes_match = (
                key_store is not None
                and value_store is not None
                and key_store.shape[:2] == key_states.shape[:2]
                and key_store.shape[3:] == key_states.shape[3:]
                and value_store.shape[:2] == value_states.shape[:2]
                and value_store.shape[3:] == value_states.shape[3:]
                and key_store.device == key_states.device
                and value_store.device == value_states.device
                and key_store.dtype == key_states.dtype
                and value_store.dtype == value_states.dtype
            )
            if not shapes_match or capacity < needed:
                capacity = (
                    max(needed, int(math.ceil(capacity * self._kv_growth)))
                    if capacity
                    else max(16, needed, int(math.ceil(needed * 1.25)))
                )
                key_store = key_states.new_empty(
                    (*key_states.shape[:2], capacity, *key_states.shape[3:])
                )
                value_store = value_states.new_empty(
                    (
                        *value_states.shape[:2],
                        capacity,
                        *value_states.shape[3:],
                    )
                )
                if old_len:
                    key_store[:, :, :old_len].copy_(old_key)
                    value_store[:, :, :old_len].copy_(old_value)
                self._key_storage[layer_idx] = key_store
                self._value_storage[layer_idx] = value_store
                self._kv_capacity[layer_idx] = capacity

            key_store[:, :, old_len:needed].copy_(key_states)
            value_store[:, :, old_len:needed].copy_(value_states)
            self.key_cache[layer_idx] = key_store[:, :, :needed]
            self.value_cache[layer_idx] = value_store[:, :, :needed]
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def truncate_mla(self, layer_idx: int, length: int) -> None:
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
            if key is None or length < 0 or length > key.shape[2]:
                raise ValueError(
                    f"cannot truncate layer {layer_idx} KV to {length}"
                )
            self.key_cache[layer_idx] = key[:, :, :length]
            self.value_cache[layer_idx] = value[:, :, :length]

        def restore_mla(self, layer_idx, key, value) -> None:
            key_store = self._key_storage[layer_idx]
            value_store = self._value_storage[layer_idx]
            same_backing = (
                key_store is not None
                and value_store is not None
                and key.untyped_storage().data_ptr()
                == key_store.untyped_storage().data_ptr()
                and value.untyped_storage().data_ptr()
                == value_store.untyped_storage().data_ptr()
            )
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
            if not same_backing:
                self._key_storage[layer_idx] = key
                self._value_storage[layer_idx] = value
                self._kv_capacity[layer_idx] = key.shape[2]

        def reorder_cache(self, beam_idx):
            result = super().reorder_cache(beam_idx)
            for layer_idx, key in enumerate(self.key_cache):
                if key is not None:
                    self._key_storage[layer_idx] = key
                    self._value_storage[layer_idx] = self.value_cache[
                        layer_idx
                    ]
                    self._kv_capacity[layer_idx] = key.shape[2]
            return result

    GeometricKimiDynamicCache.__name__ = "KimiDynamicCache"
    GeometricKimiDynamicCache.__qualname__ = "KimiDynamicCache"
    GeometricKimiDynamicCache.__module__ = modeling_module.__name__
    modeling_module.KimiDynamicCache = GeometricKimiDynamicCache
    return True


__all__ = ["install"]
