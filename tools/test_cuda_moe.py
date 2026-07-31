#!/usr/bin/env python3
"""Tests for tools/cuda_moe.py and Deltafin's stateless CUDA source.

The portable cases do not compile, load, or execute native code. A Linux host
with a locally built Deltafin CUDA artifact additionally runs generated,
weight-free end-to-end bridge cases; no downloaded model or third-party native
submission is involved.
"""
from __future__ import annotations

import ctypes
import contextlib
import inspect
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import cuda_moe  # noqa: E402


REAL_CUDA = (
    cuda_moe._CUDA_MOE_PLATFORM
    and torch.cuda.is_available()
    and os.path.isfile(cuda_moe._LIBRARY_PATH)
)


def key(model, device, layer, expert):
    return cuda_moe._cache_key(model, device, layer, expert)


class TestLayerStratifiedCache(unittest.TestCase):
    def test_complete_identity_prevents_cross_layer_model_device_alias(self):
        cache = cuda_moe._LayerStratifiedCache(32, layer_strata=2)
        identities = (
            key("model-a", 0, 0, 7),
            key("model-a", 0, 1, 7),
            key("model-b", 0, 0, 7),
            key("model-a", 1, 0, 7),
        )
        for index, identity in enumerate(identities):
            admitted, evicted = cache.put(identity, f"value-{index}")
            self.assertTrue(admitted)
            self.assertIsNone(evicted)
        for index, identity in enumerate(identities):
            self.assertEqual(cache.peek(identity), f"value-{index}")

    def test_eviction_is_confined_to_the_same_layer_stratum(self):
        cache = cuda_moe._LayerStratifiedCache(4, layer_strata=2)
        layer_one = key("m", 0, 1, 9)
        cache.put(layer_one, "keep")
        cache.put(key("m", 0, 0, 1), "old")
        cache.put(key("m", 0, 0, 2), "new")
        admitted, evicted = cache.put(key("m", 0, 0, 3), "newest")
        self.assertTrue(admitted)
        self.assertEqual(evicted, "old")
        self.assertEqual(cache.peek(layer_one), "keep")
        self.assertIsNone(cache.peek(key("m", 0, 0, 1)))

    def test_sub_layer_count_capacity_admits_stable_subset(self):
        cache = cuda_moe._LayerStratifiedCache(3, layer_strata=8)
        admitted_layers = [
            layer for layer in range(8) if cache.quota(layer) == 1
        ]
        self.assertEqual(admitted_layers, [0, 1, 2])
        for layer in range(8):
            cache.put(key("m", 0, layer, 1), f"L{layer}")
        self.assertEqual(len(cache), 3)
        before = dict(cache.layer_counts())
        for _ in range(4):
            for layer in range(3, 8):
                admitted, _ = cache.put(
                    key("m", 0, layer, 2), f"later-L{layer}"
                )
                self.assertFalse(admitted)
        self.assertEqual(cache.layer_counts(), before)
        for layer in admitted_layers:
            self.assertEqual(cache.peek(key("m", 0, layer, 1)), f"L{layer}")

    def test_default_remainder_is_spread_and_exact(self):
        cache = cuda_moe._LayerStratifiedCache(17)
        admitted = [
            layer
            for layer in range(cuda_moe.DEFAULT_LAYER_STRATA)
            if cache.quota(layer)
        ]
        self.assertEqual(len(admitted), 17)
        self.assertGreater(max(admitted) - min(admitted), 50)
        self.assertEqual(
            sum(
                cache.quota(layer)
                for layer in range(cuda_moe.DEFAULT_LAYER_STRATA)
            ),
            17,
        )

    def test_zero_capacity_is_valid(self):
        cache = cuda_moe._LayerStratifiedCache(0)
        admitted, evicted = cache.put(key("m", 0, 3, 4), object())
        self.assertFalse(admitted)
        self.assertIsNone(evicted)
        self.assertEqual(len(cache), 0)


class TestBudgetAndValidation(unittest.TestCase):
    def test_auto_budget_uses_current_free_not_total_vram(self):
        free = 10 << 30
        total = 80 << 30
        capacity, reserve = cuda_moe._cache_budget(
            free, total, cache_value="auto", reserve_value="2GiB"
        )
        self.assertEqual(reserve, 2 << 30)
        self.assertEqual(capacity, (free - reserve) // cuda_moe.EXPERT_SPAN)
        self.assertLess(
            capacity, (total - reserve) // cuda_moe.EXPERT_SPAN
        )

    def test_explicit_capacity_is_clamped_and_zero_is_valid(self):
        free = 3 << 30
        maximum = (free - (1 << 30)) // cuda_moe.EXPERT_SPAN
        capacity, _ = cuda_moe._cache_budget(
            free, 99 << 30, cache_value="999999", reserve_value="1GiB"
        )
        self.assertEqual(capacity, maximum)
        zero, _ = cuda_moe._cache_budget(
            free, 99 << 30, cache_value="0", reserve_value="1GiB"
        )
        self.assertEqual(zero, 0)

    def test_invalid_budget_values_fail_closed(self):
        for value in ("-1", "many", "1.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    cuda_moe._cache_budget(
                        4 << 30,
                        4 << 30,
                        cache_value=value,
                        reserve_value="0B",
                    )
        with self.assertRaises(ValueError):
            cuda_moe._cache_budget(
                4 << 30,
                4 << 30,
                cache_value="auto",
                reserve_value="101%",
            )

    def test_cache_budget_is_finalized_on_first_use_only(self):
        state = cuda_moe._DeviceState(0, layer_strata=2)
        self.assertFalse(state.budget_ready)
        with (
            mock.patch.object(
                torch.cuda, "device", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(9 << 30, 12 << 30)
            ) as memory,
            mock.patch.object(
                cuda_moe, "_cache_budget", return_value=(7, 2 << 30)
            ) as budget,
        ):
            cuda_moe._ensure_cache_budget(state, "cuda:0")
            cuda_moe._ensure_cache_budget(state, "cuda:0")
        self.assertTrue(state.budget_ready)
        self.assertEqual(state.capacity, 7)
        self.assertEqual(state.cache.capacity, 7)
        memory.assert_called_once()
        budget.assert_called_once()

    def test_rocm_is_rejected_before_cuda_runtime_probe(self):
        with (
            mock.patch.object(torch.version, "hip", "6.0"),
            mock.patch.object(
                torch.cuda,
                "is_available",
                side_effect=AssertionError("must not probe CUDA"),
            ),
        ):
            with self.assertRaisesRegex(cuda_moe.CudaMoeError, "ROCm/HIP"):
                cuda_moe._canonical_cuda_device("cuda:0")

    def test_route_validation_accepts_record_and_preserves_order(self):
        record = {
            "ids": [[7, 2, 7], [3, 9, 1]],
            "weights": [[0.5, 0.25, 0.125], [1, -0.5, 0]],
        }
        ids, weights = cuda_moe._validate_routes(
            2, None, None, record
        )
        self.assertEqual(ids, record["ids"])
        self.assertEqual(weights[0], [0.5, 0.25, 0.125])

    def test_route_length_and_values_are_rejected_before_native(self):
        bad = (
            (2, [[1]], [[1.0]], None),
            (1, [[1, 2]], [[1.0]], None),
            (1, [[-1]], [[1.0]], None),
            (1, [[1]], [[float("nan")]], None),
        )
        for token_count, ids, weights, record in bad:
            with self.subTest(ids=ids, weights=weights):
                with self.assertRaises((TypeError, ValueError)):
                    cuda_moe._validate_routes(
                        token_count, ids, weights, record
                    )

    def test_raw_expert_validation_rejects_wrong_layout(self):
        tiny = np.zeros((1, 1), dtype=np.uint8)
        raw = {name: (tiny, tiny) for name in ("w1", "w2", "w3")}
        with self.assertRaises(ValueError):
            cuda_moe._validated_raw_parts(raw)

    def test_cpu_input_fails_before_capability_or_ctypes(self):
        x = torch.zeros(1, cuda_moe.HIDDEN, dtype=torch.float32)
        with mock.patch.object(
            cuda_moe, "available", side_effect=AssertionError("must not probe")
        ):
            with self.assertRaisesRegex(ValueError, "reside on CUDA"):
                cuda_moe.moe_infer(
                    x,
                    [[1]],
                    [[1.0]],
                    {},
                    layer_index=1,
                )

    def test_dequant_false_is_layout_only_and_does_not_probe(self):
        dst = torch.empty((2, 3), dtype=torch.float32)
        q = torch.empty((2, 3), dtype=torch.int8)
        sc = torch.ones(2, dtype=torch.float16)
        with mock.patch.object(
            cuda_moe, "available", side_effect=AssertionError("must not probe")
        ):
            self.assertFalse(cuda_moe.dequant_int8_into(dst, q, sc))

    def test_dequant_storage_overlap_helper(self):
        storage = torch.empty(32, dtype=torch.uint8)
        self.assertTrue(
            cuda_moe._storage_overlaps(storage[:20], storage[12:])
        )
        self.assertFalse(
            cuda_moe._storage_overlaps(storage[:8], storage[8:16])
        )


class _FakeFunction:
    def __init__(self, implementation=lambda *args: 0):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class _FakeLibrary:
    def __init__(self, shapes=(3584, 3072, 17_547_264, 1), abi=1):
        self.k3_cuda_moe_abi_version = _FakeFunction(lambda: abi)

        def write_shapes(hidden, intermediate, span, layout):
            hidden._obj.value = shapes[0]
            intermediate._obj.value = shapes[1]
            span._obj.value = shapes[2]
            layout._obj.value = shapes[3]

        self.k3_cuda_moe_shapes = _FakeFunction(write_shapes)
        self.k3_cuda_moe_available = _FakeFunction(lambda device: 1)
        self.k3_cuda_moe_launch = _FakeFunction()
        self.k3_cuda_mxfp4_gemv = _FakeFunction()
        self.k3_cuda_int8_dequant = _FakeFunction()
        self.k3_cuda_last_error = _FakeFunction(lambda: b"fake")


class TestABIAndPublicContract(unittest.TestCase):
    def test_handshake_accepts_only_exact_contract(self):
        cuda_moe._bind_and_handshake(_FakeLibrary())
        for library in (
            _FakeLibrary(abi=2),
            _FakeLibrary(shapes=(3585, 3072, 17_547_264, 1)),
            _FakeLibrary(shapes=(3584, 3072, 17_547_264, 2)),
        ):
            with self.assertRaises(cuda_moe.CudaMoeError):
                cuda_moe._bind_and_handshake(library)

    def test_missing_symbol_is_rejected(self):
        library = _FakeLibrary()
        del library.k3_cuda_moe_launch
        with self.assertRaisesRegex(cuda_moe.CudaMoeError, "missing"):
            cuda_moe._bind_and_handshake(library)

    def test_public_signatures(self):
        self.assertEqual(
            list(inspect.signature(cuda_moe.missing_experts).parameters),
            ["layer_index", "eids", "device", "model_key"],
        )
        signature = inspect.signature(cuda_moe.moe_infer)
        self.assertEqual(
            list(signature.parameters),
            [
                "x",
                "topk_ids",
                "topk_weight",
                "raw_experts",
                "routing_record",
                "layer_index",
                "model_key",
            ],
        )
        self.assertEqual(
            signature.parameters["layer_index"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertIn(
            "k3_cuda_moe_launch(", inspect.getsource(cuda_moe._self_test)
        )

    def test_missing_experts_is_read_only_and_ordered(self):
        cache = cuda_moe._LayerStratifiedCache(184)
        model, index, layer = "checkpoint-a", 2, 11
        cache.put(key(model, index, layer, 7), object())

        class State:
            def __init__(self):
                self.cache = cache
                self.lock = __import__("threading").RLock()

        with (
            mock.patch.object(
                cuda_moe,
                "_canonical_cuda_device",
                return_value=("cuda:2", index),
            ),
            mock.patch.object(cuda_moe, "_state_for", return_value=State()),
            mock.patch.object(cuda_moe, "_ensure_cache_budget"),
        ):
            result = cuda_moe.missing_experts(
                layer, [7, 4, 4, 9], "cuda:2", model_key=model
            )
        self.assertEqual(result, [4, 9])
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    @unittest.skipUnless(
        REAL_CUDA,
        "requires a built Deltafin CUDA library and visible CUDA device",
    )
    def test_real_available_runs_on_device_known_answer_suite(self):
        device = torch.device("cuda", 0)
        self.assertTrue(cuda_moe.available(device), cuda_moe.describe(device))
        result = cuda_moe.stats(device)["self_test"]
        self.assertTrue(result["gemv_repeat_exact"])
        self.assertTrue(result["int8_repeat_exact"])
        self.assertTrue(result["moe_repeat_exact"])


@unittest.skipUnless(
    REAL_CUDA,
    "requires a built Deltafin CUDA library and visible CUDA device",
)
class TestRealCudaBridge(unittest.TestCase):
    """Generated sparse experts exercise the Python/native lifetime boundary."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", 0)
        if not cuda_moe.available(cls.device):
            raise AssertionError(cuda_moe.describe(cls.device))
        cls.state = cuda_moe._state_for(0)

    def setUp(self):
        torch.cuda.synchronize(self.device)

    def _reset_cache(self, capacity):
        with self.state.lock:
            self.state.cache = cuda_moe._LayerStratifiedCache(
                capacity, cuda_moe.DEFAULT_LAYER_STRATA
            )
            self.state.capacity = capacity
            self.state.budget_ready = True
            self.state.enabled = True
            self.state.reason = None

    @staticmethod
    def _raw_expert(*, gate=1, up=2, down=1):
        def matrix(packed_shape, scale_shape, code):
            packed = np.zeros(packed_shape, dtype=np.uint8)
            scale = np.zeros(scale_shape, dtype=np.uint8)
            packed[0, 0] = np.uint8(code)
            scale[0, 0] = np.uint8(127)
            return packed, scale

        return {
            "w1": matrix((3072, 1792), (3072, 112), gate),
            "w2": matrix((3584, 1536), (3584, 96), down),
            "w3": matrix((3072, 1792), (3072, 112), up),
        }

    # MXFP4 e2m1 magnitudes, indexed by the packed nibble code.
    E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    @classmethod
    def _analytical(cls, values, *, gate=1, up=2, down=1):
        """Expected output for _raw_expert built from the same nibble codes.

        These parameters are packed codes, exactly as _raw_expert takes them,
        and are decoded through e2m1 here.  Passing a code straight through as
        a magnitude is what made this reference disagree with the kernel: e2m1
        maps code 1 to 0.5 and code 2 to 1.0, not to 1.0 and 2.0.
        """
        gate_weight = cls.E2M1[gate]
        up_weight = cls.E2M1[up]
        down_weight = cls.E2M1[down]
        expected = torch.zeros(
            (len(values), cuda_moe.HIDDEN), dtype=torch.float32
        )
        for row, value in enumerate(values):
            g = gate_weight * value
            u = up_weight * value
            expected[row, 0] = down_weight * (
                4.0 * math.tanh(g / 4.0) / (1.0 + math.exp(-g))
                * (25.0 * math.tanh(u / 25.0))
            )
        return expected

    def _x(self, values):
        x = torch.zeros(
            (len(values), cuda_moe.HIDDEN),
            dtype=torch.float32,
            device=self.device,
        )
        x[:, 0] = torch.tensor(values, device=self.device)
        return x

    def test_public_int8_dequant_is_exact_and_rejects_overlap(self):
        q = torch.tensor(
            [
                [-128, -31, -1, 0, 1, 17, 63, 127],
                [7, -9, 11, -13, 15, -17, 19, -21],
                [3, 5, 8, 13, 21, 34, 55, 89],
            ],
            dtype=torch.int8,
            device=self.device,
        )
        scales = torch.tensor(
            [0.5, -1.25, 0.03125],
            dtype=torch.float16,
            device=self.device,
        )
        dst = torch.empty_like(q, dtype=torch.float32)
        expected = q.float() * scales.float().unsqueeze(1)
        self.assertTrue(cuda_moe.dequant_int8_into(dst, q, scales))
        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.equal(dst, expected))

        byte_count = q.numel()
        shared = torch.empty(
            byte_count * dst.element_size(),
            dtype=torch.uint8,
            device=self.device,
        )
        overlapping_dst = shared.view(torch.float32).view_as(dst)
        overlapping_q = shared[:byte_count].view(torch.int8).view_as(q)
        self.assertFalse(
            cuda_moe.dequant_int8_into(
                overlapping_dst, overlapping_q, scales
            )
        )

    def test_n1_cache_hit_and_output_non_alias(self):
        self._reset_cache(184)
        x = self._x([1.0])
        ids = torch.tensor([[7]], dtype=torch.long, device=self.device)
        weights = torch.ones((1, 1), device=self.device)
        raw = self._raw_expert()
        model = "real-cuda-n1"
        self.assertEqual(
            cuda_moe.missing_experts(11, [7], self.device, model), [7]
        )
        first = cuda_moe.moe_infer(
            x, ids, weights, {7: raw},
            layer_index=11, model_key=model,
        )
        self.assertEqual(
            cuda_moe.missing_experts(11, [7], self.device, model), []
        )
        second = cuda_moe.moe_infer(
            x, ids, weights, {},
            layer_index=11, model_key=model,
        )
        torch.cuda.synchronize(self.device)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(
            first.cpu(), self._analytical([1.0]), atol=2e-5, rtol=2e-5
        )
        self.assertTrue(torch.equal(first, second))

    def test_n2_duplicate_routes_and_zero_capacity(self):
        self._reset_cache(0)
        x = self._x([1.0, -0.5])
        ids = torch.tensor(
            [[7, 7], [7, 7]], dtype=torch.long, device=self.device
        )
        weights = torch.tensor(
            [[0.25, 0.75], [0.4, 0.6]],
            dtype=torch.float32,
            device=self.device,
        )
        output = cuda_moe.moe_infer(
            x, ids, weights, {7: self._raw_expert()},
            layer_index=12, model_key="real-cuda-n2",
        )
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(
            output.cpu(),
            self._analytical([1.0, -0.5]),
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertEqual(
            cuda_moe.missing_experts(
                12, [7], self.device, "real-cuda-n2"
            ),
            [7],
        )

    def test_same_id_in_two_layers_keeps_distinct_weights(self):
        self._reset_cache(184)
        x = self._x([1.0])
        ids = torch.tensor([[5]], dtype=torch.long, device=self.device)
        weights = torch.ones((1, 1), device=self.device)
        model = "real-cuda-layer-identity"
        first = cuda_moe.moe_infer(
            x, ids, weights, {5: self._raw_expert(down=1)},
            layer_index=13, model_key=model,
        )
        other = cuda_moe.moe_infer(
            x, ids, weights, {5: self._raw_expert(down=4)},
            layer_index=14, model_key=model,
        )
        repeated = cuda_moe.moe_infer(
            x, ids, weights, {},
            layer_index=13, model_key=model,
        )
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, other))
        torch.testing.assert_close(
            other.cpu(),
            self._analytical([1.0], down=4),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_one_slot_eviction_keeps_inflight_expert_alive(self):
        self._reset_cache(1)
        layer = next(
            candidate
            for candidate in range(1, 93)
            if self.state.cache.quota(candidate) == 1
        )
        x = self._x([1.0])
        ids = torch.tensor([[3, 4]], dtype=torch.long, device=self.device)
        weights = torch.tensor(
            [[0.5, 0.5]], dtype=torch.float32, device=self.device
        )
        output = cuda_moe.moe_infer(
            x,
            ids,
            weights,
            {
                3: self._raw_expert(down=1),
                4: self._raw_expert(down=4),
            },
            layer_index=layer,
            model_key="real-cuda-one-slot",
        )
        torch.cuda.synchronize(self.device)
        expected = (
            self._analytical([1.0], down=1)
            + self._analytical([1.0], down=4)
        ) * 0.5
        torch.testing.assert_close(
            output.cpu(), expected, atol=2e-5, rtol=2e-5
        )
        self.assertEqual(
            len(self.state.cache), 1
        )
        self.assertEqual(
            len(cuda_moe.missing_experts(
                layer,
                [3, 4],
                self.device,
                "real-cuda-one-slot",
            )),
            1,
        )

    def test_cache_hit_waits_for_upload_from_another_stream(self):
        self._reset_cache(184)
        torch.cuda.synchronize(self.device)
        x = self._x([1.0])
        torch.cuda.synchronize(self.device)
        ids = torch.tensor([[9]], dtype=torch.long, device=self.device)
        weights = torch.ones((1, 1), device=self.device)
        stream_a = torch.cuda.Stream(device=self.device)
        stream_b = torch.cuda.Stream(device=self.device)
        model = "real-cuda-cross-stream"
        with torch.cuda.stream(stream_a):
            first = cuda_moe.moe_infer(
                x, ids, weights, {9: self._raw_expert()},
                layer_index=15, model_key=model,
            )
        with torch.cuda.stream(stream_b):
            second = cuda_moe.moe_infer(
                x, ids, weights, {},
                layer_index=15, model_key=model,
            )
        stream_b.synchronize()
        torch.testing.assert_close(
            second.cpu(), self._analytical([1.0]), atol=2e-5, rtol=2e-5
        )
        stream_a.synchronize()
        self.assertTrue(torch.equal(first, second))


class TestCudaSourceInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(TOOLS, "cuda_moe_kernels.cu")
        with open(path, "r", encoding="utf-8") as handle:
            cls.source = handle.read()

    def test_exported_manifest_is_complete(self):
        for symbol in cuda_moe._REQUIRED_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertIn(f'extern "C"', self.source)
                self.assertIn(symbol, self.source)

    def test_native_code_owns_no_cuda_allocations_or_atomic_combine(self):
        compact = self.source.replace(" ", "")
        self.assertNotIn("cudaMalloc(", compact)
        self.assertNotIn("cudaMallocAsync(", compact)
        self.assertNotIn("atomicAdd(", compact)
        self.assertIn("expert_output", self.source)
        self.assertIn("Route-order", cuda_moe.__doc__)

    def test_stream_is_explicit_and_every_launch_is_checked(self):
        self.assertIn("void* stream_pointer", self.source)
        self.assertIn("cudaPeekAtLastError()", self.source)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertIn(
            "static_cast<int64_t>(rows) * static_cast<int64_t>(batches)",
            self.source,
        )
        self.assertIn("1 + (count - 1) / 256", self.source)

    def test_int8_kernel_uses_flattened_64_bit_index(self):
        self.assertIn("int8_dequant_kernel", self.source)
        self.assertIn("const int64_t index", self.source)
        self.assertIn("const int64_t row = index / cols", self.source)

    def test_cuda_compatibility_uses_stable_project_owned_primitives(self):
        self.assertIn("mxfp4_quiet_nan()", self.source)
        self.assertIn("__int_as_float(0x7fffffff)", self.source)
        self.assertNotIn("CUDART_NAN_F", self.source)
        self.assertIn("cudaDevAttrComputeMode", self.source)
        self.assertIn("cudaDevAttrComputeCapabilityMajor", self.source)
        self.assertIn("cudaDevAttrComputeCapabilityMinor", self.source)
        self.assertNotIn(".computeMode", self.source)
        self.assertNotIn("CUDART_VERSION", self.source)

    def test_expert_layout_constants_match_python(self):
        normalized = self.source.replace("'", "")
        self.assertIn(
            f"constexpr int64_t kExpertSpan = {cuda_moe.EXPERT_SPAN};",
            normalized,
        )
        self.assertIn(
            f"constexpr int kHidden = {cuda_moe.HIDDEN};", normalized
        )
        self.assertIn(
            f"constexpr int kIntermediate = {cuda_moe.INTERMEDIATE};",
            normalized,
        )

    def test_python_bridge_orders_cross_stream_cache_hits(self):
        source = inspect.getsource(cuda_moe.moe_infer)
        self.assertIn(
            "stream.wait_event(resident_expert.ready_event)", source
        )
        self.assertIn("resident_expert.tensor", source)
        self.assertIn("output.addcmul_(", source)
        self.assertNotIn("output[token].add_(", source)

    def test_int8_runtime_failure_does_not_clear_moe_cache(self):
        source = inspect.getsource(cuda_moe.dequant_int8_into)
        self.assertNotIn("state.cache.clear", source)
        self.assertNotIn("state.enabled = False", source)
        eligibility = inspect.getsource(cuda_moe._dequant_layout_eligible)
        self.assertIn("q.dtype != torch.int8", eligibility)
        self.assertIn("_storage_overlaps(dst, q, sc)", eligibility)


if __name__ == "__main__":
    unittest.main(verbosity=2)
