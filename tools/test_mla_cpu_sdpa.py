#!/usr/bin/env python3
"""Correctness/lifecycle gate for the opt-in CPU fp32 MLA SDPA path.

This deliberately does not time anything. It uses K3's production attention
dimensions and KimiDynamicCache, but synthetic q/k/v plus a deterministic
small vocabulary head rather than a full-model checkpoint run.
"""
from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attn_fast  # noqa: E402
import kv_cache  # noqa: E402
from k3pkg import modeling_kimi_linear as ml  # noqa: E402

kv_cache.install(ml)
torch.set_grad_enabled(False)


HEADS = 96
QK_DIM = 192
VALUE_DIM = 128
SCALE = QK_DIM ** -0.5


class CacheConfig:
    linear_attn_config = None
    num_hidden_layers = 1


MODULE = SimpleNamespace(num_key_value_groups=1, training=False)


def normal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        shape, generator=generator, dtype=torch.float32
    ) / math.sqrt(shape[-1])


def causal_mask(tokens: int, context: int) -> torch.Tensor | None:
    if tokens == 1:
        return None
    past = context - tokens
    mask = torch.zeros(1, 1, tokens, context, dtype=torch.float32)
    future = torch.triu(
        torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1
    )
    mask[:, :, :, past:].masked_fill_(
        future, torch.finfo(torch.float32).min
    )
    return mask


def state_signature(*tensors: torch.Tensor):
    return [
        (
            tensor._version,
            tensor.shape,
            tensor.stride(),
            tensor.clone(),
        )
        for tensor in tensors
    ]


def assert_state_signature(
    case: unittest.TestCase,
    expected,
    *tensors: torch.Tensor,
):
    for (version, shape, stride, bits), tensor in zip(expected, tensors):
        case.assertEqual(tensor._version, version)
        case.assertEqual(tensor.shape, shape)
        case.assertEqual(tensor.stride(), stride)
        case.assertTrue(torch.equal(tensor, bits))


def append_and_attend(
    case: unittest.TestCase,
    cache: ml.KimiDynamicCache,
    function,
    query: torch.Tensor,
    key_rows: torch.Tensor,
    value_rows: torch.Tensor,
):
    keys, values = cache.update(key_rows, value_rows, 0)
    before = state_signature(query, keys, values)
    output, _ = function(
        MODULE,
        query,
        keys,
        values,
        causal_mask(query.shape[-2], keys.shape[-2]),
        scaling=SCALE,
        dropout=0.0,
    )
    assert_state_signature(case, before, query, keys, values)
    case.assertEqual(cache.value_cache[0].shape[-1], VALUE_DIM)
    case.assertEqual(cache._value_storage[0].shape[-1], VALUE_DIM)
    return output


class CpuSdpaGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = {
            "schema": "deltafin.c25.cpu-sdpa-correctness.v1",
            "quality_scope": (
                "synthetic deterministic q/k/v at real K3 MLA dimensions "
                "through production KimiDynamicCache and a deterministic "
                "small emitted-token surrogate; not a full checkpoint run"
            ),
        }
        cls.old_requested = attn_fast.CPU_SDPA_REQUESTED
        cls.old_compile = attn_fast.COMPILE
        cls.old_installed = attn_fast._CPU_SDPA_INSTALLED
        cls.old_install_reason = attn_fast._CPU_SDPA_INSTALL_REASON
        cls.had_original = "cpu_sdpa_attention" in attn_fast._ORIG
        cls.old_original = attn_fast._ORIG.get("cpu_sdpa_attention")
        cls.old_model_forward = ml.eager_attention_forward
        attn_fast.CPU_SDPA_REQUESTED = True
        attn_fast.COMPILE = "0"
        attn_fast._CPU_SDPA_INSTALLED = False
        attn_fast._CPU_SDPA_INSTALL_REASON = None
        attn_fast._ORIG.pop("cpu_sdpa_attention", None)
        attn_fast.install(ml)
        cls.original = staticmethod(cls.old_model_forward)
        cls.candidate = staticmethod(ml.eager_attention_forward)

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("K3_CPU_SDPA_TEST_JSON")
        if output:
            cls.evidence["tests_complete"] = all(
                name in cls.evidence for name in (
                    "default_off",
                    "failure_gates",
                    "lifecycle",
                    "module_sequence",
                    "runtime_shape_keyed",
                )
            )
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(cls.evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote correctness artifact: {path}")
        attn_fast.CPU_SDPA_REQUESTED = cls.old_requested
        attn_fast.COMPILE = cls.old_compile
        attn_fast._CPU_SDPA_INSTALLED = cls.old_installed
        attn_fast._CPU_SDPA_INSTALL_REASON = cls.old_install_reason
        ml.eager_attention_forward = cls.old_model_forward
        if cls.had_original:
            attn_fast._ORIG["cpu_sdpa_attention"] = cls.old_original
        else:
            attn_fast._ORIG.pop("cpu_sdpa_attention", None)
        attn_fast._reset_cpu_sdpa_for_tests()

    def setUp(self):
        attn_fast._reset_cpu_sdpa_for_tests()

    def test_default_off_install_is_identity(self):
        original_requested = attn_fast.CPU_SDPA_REQUESTED
        original_installed = attn_fast._CPU_SDPA_INSTALLED
        original_reason = attn_fast._CPU_SDPA_INSTALL_REASON
        original_compile = attn_fast.COMPILE
        function = lambda *args, **kwargs: (args, kwargs)
        namespace = SimpleNamespace(eager_attention_forward=function)
        try:
            attn_fast.CPU_SDPA_REQUESTED = False
            attn_fast._CPU_SDPA_INSTALLED = False
            attn_fast._CPU_SDPA_INSTALL_REASON = None
            attn_fast.COMPILE = "0"
            attn_fast.install(namespace)
            self.assertIs(namespace.eager_attention_forward, function)
            self.assertFalse(attn_fast.cpu_sdpa_status()["installed"])
            self.__class__.evidence["default_off"] = {
                "installed": False,
                "function_identity_preserved": True,
            }
        finally:
            attn_fast.CPU_SDPA_REQUESTED = original_requested
            attn_fast._CPU_SDPA_INSTALLED = original_installed
            attn_fast._CPU_SDPA_INSTALL_REASON = original_reason
            attn_fast.COMPILE = original_compile

    def test_contract_is_runtime_and_shape_keyed(self):
        query = normal((1, HEADS, 1, QK_DIM), 1)
        key = normal((1, HEADS, 32, QK_DIM), 2)
        value = normal((1, HEADS, 32, VALUE_DIM), 3)
        key_label, reason = attn_fast._cpu_sdpa_key(
            MODULE, query, key, value, None, 0.0
        )
        self.assertIsNone(reason)
        self.assertIn(f"torch={torch.__version__}", key_label)
        self.assertIn(f"threads={torch.get_num_threads()}", key_label)
        self.assertIn("T=1", key_label)
        self.assertIn("L=32-127", key_label)
        for product_name in ("darwin", "linux", "apple", "m1", "m5"):
            self.assertNotIn(product_name, key_label.lower())

        old_compile = attn_fast.COMPILE
        try:
            attn_fast.COMPILE = "attn"
            _, reason = attn_fast._cpu_sdpa_key(
                MODULE, query, key, value, None, 0.0
            )
            self.assertEqual(reason, "K3_COMPILE is active")
        finally:
            attn_fast.COMPILE = old_compile

        query_t3 = normal((1, HEADS, 3, QK_DIM), 4)
        _, reason = attn_fast._cpu_sdpa_key(
            MODULE,
            query_t3,
            key,
            value,
            causal_mask(3, 32),
            0.0,
        )
        self.assertEqual(reason, "T=3 is outside the tested set")
        self.assertIsNone(attn_fast._context_bucket(31))
        self.assertIsNone(attn_fast._context_bucket(1025))
        self.__class__.evidence["runtime_shape_keyed"] = {
            "key": key_label,
            "contains_os_or_product_name": False,
            "compile_bypasses": True,
            "untested_token_width_bypasses": True,
            "out_of_range_context_bypasses": True,
        }

    def test_opt_in_install_wraps_exactly_once(self):
        original_requested = attn_fast.CPU_SDPA_REQUESTED
        original_installed = attn_fast._CPU_SDPA_INSTALLED
        original_reason = attn_fast._CPU_SDPA_INSTALL_REASON
        original_compile = attn_fast.COMPILE
        had_original = "cpu_sdpa_attention" in attn_fast._ORIG
        saved_original = attn_fast._ORIG.get("cpu_sdpa_attention")
        function = lambda *args, **kwargs: (args, kwargs)
        namespace = SimpleNamespace(eager_attention_forward=function)
        try:
            attn_fast._ORIG.pop("cpu_sdpa_attention", None)
            attn_fast.CPU_SDPA_REQUESTED = True
            attn_fast._CPU_SDPA_INSTALLED = False
            attn_fast._CPU_SDPA_INSTALL_REASON = None
            attn_fast.COMPILE = "0"
            attn_fast.install(namespace)
            wrapped = namespace.eager_attention_forward
            self.assertIsNot(wrapped, function)
            self.assertTrue(attn_fast.cpu_sdpa_status()["installed"])
            attn_fast.install(namespace)
            self.assertIs(namespace.eager_attention_forward, wrapped)
        finally:
            attn_fast.CPU_SDPA_REQUESTED = original_requested
            attn_fast._CPU_SDPA_INSTALLED = original_installed
            attn_fast._CPU_SDPA_INSTALL_REASON = original_reason
            attn_fast.COMPILE = original_compile
            if had_original:
                attn_fast._ORIG["cpu_sdpa_attention"] = saved_original
            else:
                attn_fast._ORIG.pop("cpu_sdpa_attention", None)

    def test_real_mla_module_emitted_sequence(self):
        config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=HEADS,
            num_key_value_heads=HEADS,
            attention_dropout=0.0,
            q_lora_rank=None,
            qk_rope_head_dim=64,
            kv_lora_rank=16,
            v_head_dim=VALUE_DIM,
            qk_nope_head_dim=128,
            mla_use_nope=True,
            mla_use_output_gate=True,
            _attn_implementation="eager",
        )
        module = ml.KimiMLAAttention(config, layer_idx=0).eval()
        generator = torch.Generator().manual_seed(300)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    ) * 0.02
                )
        embedding = normal((29, config.hidden_size), 301)
        token_head = normal((29, config.hidden_size), 302)
        prompt_ids = torch.arange(32) % embedding.shape[0]
        prompt_hidden = embedding[prompt_ids].unsqueeze(0)
        reference_cache = ml.KimiDynamicCache(CacheConfig())
        candidate_cache = ml.KimiDynamicCache(CacheConfig())

        old_requested = attn_fast.CPU_SDPA_REQUESTED
        try:
            attn_fast.CPU_SDPA_REQUESTED = False
            reference_prompt = module(
                prompt_hidden, past_key_values=reference_cache
            )
            attn_fast.CPU_SDPA_REQUESTED = True
            candidate_prompt = module(
                prompt_hidden, past_key_values=candidate_cache
            )
            self.assertTrue(
                torch.equal(candidate_prompt, reference_prompt)
            )

            reference_token = int(
                (reference_prompt[:, -1] @ token_head.T).argmax()
            )
            candidate_token = int(
                (candidate_prompt[:, -1] @ token_head.T).argmax()
            )
            self.assertEqual(candidate_token, reference_token)
            reference_tokens = [reference_token]
            candidate_tokens = [candidate_token]
            worst_max_abs = 0.0
            minimum_margin = float("inf")

            for _ in range(6):
                reference_hidden = embedding[
                    [reference_token]
                ].unsqueeze(0)
                candidate_hidden = embedding[
                    [candidate_token]
                ].unsqueeze(0)
                attn_fast.CPU_SDPA_REQUESTED = False
                reference = module(
                    reference_hidden,
                    past_key_values=reference_cache,
                )
                attn_fast.CPU_SDPA_REQUESTED = True
                candidate = module(
                    candidate_hidden,
                    past_key_values=candidate_cache,
                )
                worst_max_abs = max(
                    worst_max_abs,
                    float((candidate - reference).abs().max()),
                )
                torch.testing.assert_close(
                    candidate, reference, rtol=3e-5, atol=2e-7
                )
                reference_logits = reference[:, -1] @ token_head.T
                candidate_logits = candidate[:, -1] @ token_head.T
                top2 = reference_logits.topk(2, dim=-1).values
                minimum_margin = min(
                    minimum_margin,
                    float((top2[..., 0] - top2[..., 1]).min()),
                )
                reference_token = int(reference_logits.argmax())
                candidate_token = int(candidate_logits.argmax())
                self.assertEqual(candidate_token, reference_token)
                reference_tokens.append(reference_token)
                candidate_tokens.append(candidate_token)
                self.assertTrue(torch.equal(
                    candidate_cache.key_cache[0],
                    reference_cache.key_cache[0],
                ))
                self.assertTrue(torch.equal(
                    candidate_cache.value_cache[0],
                    reference_cache.value_cache[0],
                ))
                self.assertEqual(
                    candidate_cache.value_cache[0].shape[-1],
                    VALUE_DIM,
                )
        finally:
            attn_fast.CPU_SDPA_REQUESTED = old_requested

        status = attn_fast.cpu_sdpa_status()
        require_native = os.environ.get(
            "K3_REQUIRE_CPU_SDPA_TEST", "0"
        ) == "1"
        if require_native:
            self.assertEqual(status["canary_passes"], 1)
            self.assertEqual(status["canary_failures"], 0)
            self.assertEqual(status["fast_calls"], 6)
        self.assertEqual(reference_tokens, candidate_tokens)
        self.__class__.evidence["module_sequence"] = {
            "actual_kimi_mla_forward": True,
            "heads": HEADS,
            "qk_head_dim": QK_DIM,
            "value_head_dim": VALUE_DIM,
            "shrunk_hidden_size": config.hidden_size,
            "shrunk_kv_lora_rank": config.kv_lora_rank,
            "prompt_tokens": 32,
            "decode_steps": 6,
            "reference_token_ids": reference_tokens,
            "candidate_token_ids": candidate_tokens,
            "emitted_token_ids_exact": (
                reference_tokens == candidate_tokens
            ),
            "cache_payload_exact": True,
            "persistent_value_width": VALUE_DIM,
            "persistent_padding_bytes": 0,
            "worst_module_output_max_abs": worst_max_abs,
            "minimum_reference_argmax_margin": minimum_margin,
            "status": status,
        }

    def test_allocator_and_operator_failures_fall_back_and_poison_key(self):
        query = normal((1, HEADS, 1, QK_DIM), 10)
        key = normal((1, HEADS, 32, QK_DIM), 11)
        value = normal((1, HEADS, 32, VALUE_DIM), 12)
        reference, _ = self.original(
            MODULE, query, key, value, None, scaling=SCALE
        )
        before = state_signature(query, key, value)

        with mock.patch.object(
            attn_fast.F, "pad",
            side_effect=RuntimeError("injected allocator failure"),
        ):
            fallback, _ = self.candidate(
                MODULE, query, key, value, None, scaling=SCALE
            )
        self.assertTrue(torch.equal(fallback, reference))
        assert_state_signature(self, before, query, key, value)
        status = attn_fast.cpu_sdpa_status()
        self.assertEqual(status["canary_failures"], 1)
        self.assertEqual(status["eager_fallback_calls"], 1)
        self.assertFalse(status["keys"][0]["enabled"])

        # A poisoned key must never retry allocation/operator work.
        with mock.patch.object(
            attn_fast.F, "pad",
            side_effect=AssertionError("disabled key retried"),
        ):
            fallback_again, _ = self.candidate(
                MODULE, query, key, value, None, scaling=SCALE
            )
        self.assertTrue(torch.equal(fallback_again, reference))
        self.assertEqual(
            attn_fast.cpu_sdpa_status()["disabled_key_bypasses"], 1
        )
        allocator_status = attn_fast.cpu_sdpa_status()

        # Separately prove a post-canary operator error disables the key and
        # returns eager output without mutating q/k/v.
        attn_fast._reset_cpu_sdpa_for_tests()
        first, _ = self.candidate(
            MODULE, query, key, value, None, scaling=SCALE
        )
        status = attn_fast.cpu_sdpa_status()
        require_native = os.environ.get(
            "K3_REQUIRE_CPU_SDPA_TEST", "0"
        ) == "1"
        if require_native:
            self.assertEqual(status["canary_passes"], 1)
        if status["canary_passes"]:
            self.assertEqual(first.argmax().item(), reference.argmax().item())
            with mock.patch.object(
                attn_fast.F, "scaled_dot_product_attention",
                side_effect=RuntimeError("injected operator failure"),
            ):
                fallback_after_canary, _ = self.candidate(
                    MODULE, query, key, value, None, scaling=SCALE
                )
            self.assertTrue(torch.equal(fallback_after_canary, reference))
            status = attn_fast.cpu_sdpa_status()
            self.assertEqual(status["runtime_failures"], 1)
            self.assertFalse(status["keys"][0]["enabled"])
            assert_state_signature(self, before, query, key, value)
        self.__class__.evidence["failure_gates"] = {
            "allocator_failure": allocator_status,
            "post_canary_operator_failure": (
                attn_fast.cpu_sdpa_status()
            ),
            "fallback_bit_exact": True,
            "state_nonmutation_exact": True,
        }

    def test_cache_lifecycle_and_emitted_sequence(self):
        reference_cache = ml.KimiDynamicCache(CacheConfig())
        candidate_cache = ml.KimiDynamicCache(CacheConfig())
        prompt_key = normal((1, HEADS, 31, QK_DIM), 100)
        prompt_value = normal((1, HEADS, 31, VALUE_DIM), 101)
        reference_cache.update(prompt_key, prompt_value, 0)
        candidate_cache.update(prompt_key, prompt_value, 0)

        head = normal((23, HEADS * VALUE_DIM), 102)
        reference_tokens = []
        candidate_tokens = []
        worst_max_abs = 0.0
        minimum_reference_margin = float("inf")

        def run_chunk(tokens: int, seed: int):
            nonlocal worst_max_abs, minimum_reference_margin
            query = normal((1, HEADS, tokens, QK_DIM), seed)
            key_rows = normal(
                (1, HEADS, tokens, QK_DIM), seed + 1
            )
            value_rows = normal(
                (1, HEADS, tokens, VALUE_DIM), seed + 2
            )
            reference = append_and_attend(
                self, reference_cache, self.original,
                query, key_rows, value_rows,
            )
            candidate = append_and_attend(
                self, candidate_cache, self.candidate,
                query, key_rows, value_rows,
            )
            torch.testing.assert_close(
                candidate, reference, rtol=3e-5, atol=2e-7
            )
            worst_max_abs = max(
                worst_max_abs,
                float((candidate - reference).abs().max()),
            )
            reference_logits = (
                reference.reshape(1, tokens, -1) @ head.T
            )
            candidate_logits = (
                candidate.reshape(1, tokens, -1) @ head.T
            )
            reference_ids = reference_logits.argmax(-1)
            candidate_ids = candidate_logits.argmax(-1)
            top2 = reference_logits.topk(2, dim=-1).values
            minimum_reference_margin = min(
                minimum_reference_margin,
                float((top2[..., 0] - top2[..., 1]).min()),
            )
            self.assertTrue(torch.equal(candidate_ids, reference_ids))
            reference_tokens.extend(reference_ids.flatten().tolist())
            candidate_tokens.extend(candidate_ids.flatten().tolist())
            return reference, candidate

        run_chunk(1, 200)  # L=32, in-capacity append
        run_chunk(2, 210)  # additive causal mask

        reference_snapshot = (
            reference_cache.key_cache[0],
            reference_cache.value_cache[0],
        )
        candidate_snapshot = (
            candidate_cache.key_cache[0],
            candidate_cache.value_cache[0],
        )
        reference_snapshot_bits = tuple(
            tensor.clone() for tensor in reference_snapshot
        )
        candidate_snapshot_bits = tuple(
            tensor.clone() for tensor in candidate_snapshot
        )

        first_replay = run_chunk(8, 220)  # forces geometric slab growth
        for tensor, bits in zip(
            reference_snapshot, reference_snapshot_bits
        ):
            self.assertTrue(torch.equal(tensor, bits))
        for tensor, bits in zip(
            candidate_snapshot, candidate_snapshot_bits
        ):
            self.assertTrue(torch.equal(tensor, bits))

        reference_cache.restore_mla(0, *reference_snapshot)
        candidate_cache.restore_mla(0, *candidate_snapshot)
        second_replay = run_chunk(8, 220)
        self.assertTrue(torch.equal(first_replay[0], second_replay[0]))
        torch.testing.assert_close(
            first_replay[1], second_replay[1],
            rtol=0.0, atol=0.0,
        )

        reference_cache.truncate_mla(0, 33)
        candidate_cache.truncate_mla(0, 33)
        run_chunk(9, 230)

        # Beam duplication exercises reorder and then intentionally leaves the
        # tested B=1 key. The wrapper must call eager exactly for B=2.
        beam = torch.tensor([0, 0])
        reference_cache.reorder_cache(beam)
        candidate_cache.reorder_cache(beam)
        query = normal((2, HEADS, 1, QK_DIM), 240)
        key_rows = normal((2, HEADS, 1, QK_DIM), 241)
        value_rows = normal((2, HEADS, 1, VALUE_DIM), 242)
        reference = append_and_attend(
            self, reference_cache, self.original,
            query, key_rows, value_rows,
        )
        candidate = append_and_attend(
            self, candidate_cache, self.candidate,
            query, key_rows, value_rows,
        )
        self.assertTrue(torch.equal(candidate, reference))
        self.assertEqual(reference_tokens, candidate_tokens)

        status = attn_fast.cpu_sdpa_status()
        require_native = os.environ.get(
            "K3_REQUIRE_CPU_SDPA_TEST", "0"
        ) == "1"
        if require_native:
            self.assertGreaterEqual(status["canary_passes"], 4)
            self.assertEqual(status["canary_failures"], 0)
        self.assertGreaterEqual(
            status["bypass_reasons"].get(
                "batch size is not one", 0
            ),
            1,
        )
        self.assertFalse(
            status["tested_contract"]["persistent_padded_kv"]
        )
        self.__class__.evidence["lifecycle"] = {
            "append": True,
            "in_capacity_append": True,
            "geometric_growth": True,
            "restore_replay": True,
            "truncate_rollback": True,
            "reorder": True,
            "cache_state_nonmutation_exact": True,
            "persistent_value_width": VALUE_DIM,
            "persistent_padding_bytes": 0,
            "reference_token_ids": reference_tokens,
            "candidate_token_ids": candidate_tokens,
            "emitted_token_ids_exact": (
                reference_tokens == candidate_tokens
            ),
            "minimum_reference_argmax_margin": (
                minimum_reference_margin
            ),
            "worst_attention_max_abs": worst_max_abs,
            "status": status,
        }
        print(
            "emitted token surrogate:",
            candidate_tokens,
            "native canaries:",
            status["canary_passes"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
