#!/usr/bin/env python3
"""Exactness and layout gates for the T=1 MLA query-storage alias."""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attn_fast  # noqa: E402
from k3pkg import modeling_kimi_linear as ml  # noqa: E402


HEADS = 4
QK_NOPE = 4
QK_ROPE = 2
QK = QK_NOPE + QK_ROPE
VALUE = 4
HIDDEN = 12
KV_RANK = 3


class CacheConfig:
    linear_attn_config = None
    num_hidden_layers = 1


def same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
    )


def config(q_lora_rank, output_gate):
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        attention_dropout=0.0,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=QK_ROPE,
        kv_lora_rank=KV_RANK,
        v_head_dim=VALUE,
        qk_nope_head_dim=QK_NOPE,
        mla_use_nope=True,
        mla_use_output_gate=output_gate,
        _attn_implementation="eager",
    )


def module(q_lora_rank=5, output_gate=True):
    result = ml.KimiMLAAttention(
        config(q_lora_rank, output_gate),
        layer_idx=0,
    )
    generator = torch.Generator().manual_seed(
        1700
        + (0 if q_lora_rank is None else q_lora_rank)
        + int(output_gate),
    )
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.03
            )
    return result


def hidden(batch, tokens, seed):
    return torch.randn(
        batch,
        tokens,
        HIDDEN,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    )


def legacy_query(raw, module):
    batch, tokens, _ = raw.shape
    heads = raw.view(
        batch,
        tokens,
        module.num_heads,
        module.q_head_dim,
    ).transpose(1, 2)
    q_pass, q_rot = torch.split(
        heads,
        [module.qk_nope_head_dim, module.qk_rope_head_dim],
        dim=-1,
    )
    return torch.cat((q_pass, q_rot), dim=-1)


def downloaded_forward():
    function = ml.KimiMLAAttention.forward
    seen = set()
    while getattr(function, "_k3_mla_query_alias", False):
        if id(function) in seen:
            raise AssertionError("MLA forward ownership cycle")
        seen.add(id(function))
        function = getattr(function, "_k3_original", None)
        if function is None:
            raise AssertionError("MLA alias wrapper lost its original")
    return function


@contextmanager
def capture_query_and_projection(module):
    captured = {"raw": [], "query": []}
    projection = (
        module.q_b_proj
        if module.q_lora_rank is not None
        else module.q_proj
    )

    def projection_hook(_module, _inputs, output):
        captured["raw"].append(output)

    handle = projection.register_forward_hook(projection_hook)
    original_attention = ml.eager_attention_forward

    def attention(
        attention_module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    ):
        captured["query"].append(query)
        return original_attention(
            attention_module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )

    attention._k3_mla_query_alias_safe = True
    attn_fast._MLA_QUERY_ALIAS_SAFE_PROVIDERS.add(attention)
    ml.eager_attention_forward = attention
    try:
        yield captured
    finally:
        ml.eager_attention_forward = original_attention
        attn_fast._MLA_QUERY_ALIAS_SAFE_PROVIDERS.discard(attention)
        handle.remove()


class StridedProjection(nn.Module):
    """Return the normal values through a deliberately noncontiguous view."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.inner = nn.Linear(
            input_features,
            output_features,
            bias=False,
        )

    def forward(self, value):
        dense = self.inner(value)
        interleaved = torch.stack((dense, -dense), dim=-1)
        flattened = interleaved.flatten(-2)
        result = flattened[..., ::2]
        if result.is_contiguous():
            raise AssertionError("fixture unexpectedly became contiguous")
        return result


class MlaQueryAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_forward = ml.KimiMLAAttention.forward
        cls._previous_requested = attn_fast.MLA_QUERY_ALIAS_REQUESTED
        cls._previous_installed = attn_fast._MLA_QUERY_ALIAS_INSTALLED
        cls._previous_reason = attn_fast._MLA_QUERY_ALIAS_REASON
        attn_fast.MLA_QUERY_ALIAS_REQUESTED = True
        attn_fast.install(ml)
        if not attn_fast._MLA_QUERY_ALIAS_INSTALLED:
            raise AssertionError(attn_fast._MLA_QUERY_ALIAS_REASON)

    @classmethod
    def tearDownClass(cls):
        ml.KimiMLAAttention.forward = cls._previous_forward
        attn_fast.MLA_QUERY_ALIAS_REQUESTED = cls._previous_requested
        attn_fast._MLA_QUERY_ALIAS_INSTALLED = cls._previous_installed
        attn_fast._MLA_QUERY_ALIAS_REASON = cls._previous_reason

    def test_t1_alias_matches_old_cat_and_has_canonical_layout(self):
        for q_lora_rank in (None, 5):
            for output_gate in (False, True):
                for batch in (1, 2, 3):
                    with self.subTest(
                        q_lora_rank=q_lora_rank,
                        output_gate=output_gate,
                        batch=batch,
                    ):
                        attention = module(
                            q_lora_rank,
                            output_gate,
                        ).eval()
                        source = hidden(batch, 1, 1800 + batch)
                        before = source.clone()
                        with capture_query_and_projection(
                            attention
                        ) as captured:
                            with torch.no_grad():
                                output = attention(source)

                        self.assertEqual(len(captured["raw"]), 1)
                        self.assertEqual(len(captured["query"]), 1)
                        raw = captured["raw"][0]
                        query = captured["query"][0]
                        old = legacy_query(raw, attention)
                        self.assertTrue(torch.equal(query, old))
                        self.assertEqual(query.stride(), old.stride())
                        self.assertEqual(
                            query.stride(),
                            (HEADS * QK, QK, QK, 1),
                        )
                        self.assertTrue(query.is_contiguous())
                        self.assertTrue(same_storage(query, raw))
                        self.assertFalse(same_storage(old, raw))
                        self.assertTrue(torch.equal(source, before))
                        self.assertEqual(
                            output.shape,
                            (batch, 1, HIDDEN),
                        )

    def test_full_forward_matches_downloaded_original_matrix(self):
        original = downloaded_forward()
        for q_lora_rank in (None, 5):
            for output_gate in (False, True):
                candidate = module(
                    q_lora_rank,
                    output_gate,
                ).eval()
                reference = copy.deepcopy(candidate).eval()
                for batch in (1, 2, 3):
                    for tokens in (1, 2, 8):
                        with self.subTest(
                            q_lora_rank=q_lora_rank,
                            output_gate=output_gate,
                            batch=batch,
                            tokens=tokens,
                        ):
                            value = hidden(
                                batch,
                                tokens,
                                1820 + batch * 10 + tokens,
                            )
                            candidate_input = value.clone()
                            reference_input = value.clone()
                            mask = torch.zeros(
                                batch,
                                1,
                                tokens,
                                tokens,
                                dtype=torch.float32,
                            )
                            future = torch.triu(
                                torch.ones(
                                    tokens,
                                    tokens,
                                    dtype=torch.bool,
                                ),
                                diagonal=1,
                            )
                            mask.masked_fill_(
                                future,
                                torch.finfo(torch.float32).min,
                            )
                            positions = torch.arange(tokens).expand(
                                batch,
                                tokens,
                            )
                            with torch.no_grad():
                                observed = candidate(
                                    candidate_input,
                                    attention_mask=mask,
                                    position_ids=positions,
                                    fixture_unused_kwarg=True,
                                )
                                expected = original(
                                    reference,
                                    reference_input,
                                    attention_mask=mask,
                                    position_ids=positions,
                                    fixture_unused_kwarg=True,
                                )
                            self.assertTrue(torch.equal(
                                observed,
                                expected,
                            ))
                            self.assertTrue(torch.equal(
                                candidate_input,
                                value,
                            ))
                            self.assertTrue(torch.equal(
                                reference_input,
                                value,
                            ))

    def test_exactly_one_query_cat_is_removed(self):
        original_cat = torch.cat

        def count_for(attention, tokens):
            calls = []

            def counted(*args, **kwargs):
                calls.append((args, kwargs))
                return original_cat(*args, **kwargs)

            with mock.patch.object(torch, "cat", side_effect=counted):
                with torch.no_grad():
                    attention(hidden(1, tokens, 1850 + tokens))
            return len(calls)

        self.assertEqual(count_for(module().eval(), 1), 1)
        self.assertEqual(count_for(module().train(), 1), 2)
        self.assertEqual(count_for(module().eval(), 2), 2)

    def test_unreviewed_model_forward_fails_closed(self):
        class ChangedAttention:
            def forward(self, hidden_states):
                return hidden_states

        namespace = SimpleNamespace(
            KimiMLAAttention=ChangedAttention,
        )
        original = ChangedAttention.forward
        installed = attn_fast._MLA_QUERY_ALIAS_INSTALLED
        reason = attn_fast._MLA_QUERY_ALIAS_REASON
        try:
            self.assertFalse(
                attn_fast._install_mla_query_alias(namespace)
            )
            self.assertIs(ChangedAttention.forward, original)
            self.assertFalse(attn_fast._MLA_QUERY_ALIAS_INSTALLED)
            self.assertTrue(
                "unreviewed model" in attn_fast._MLA_QUERY_ALIAS_REASON
                or "initializer source unavailable"
                in attn_fast._MLA_QUERY_ALIAS_REASON,
                msg=attn_fast._MLA_QUERY_ALIAS_REASON,
            )
        finally:
            attn_fast._MLA_QUERY_ALIAS_INSTALLED = installed
            attn_fast._MLA_QUERY_ALIAS_REASON = reason

    def test_explicit_opt_out_leaves_forward_untouched(self):
        class Attention:
            def forward(self, hidden_states):
                return hidden_states

        namespace = SimpleNamespace(KimiMLAAttention=Attention)
        original = Attention.forward
        requested = attn_fast.MLA_QUERY_ALIAS_REQUESTED
        installed = attn_fast._MLA_QUERY_ALIAS_INSTALLED
        reason = attn_fast._MLA_QUERY_ALIAS_REASON
        try:
            attn_fast.MLA_QUERY_ALIAS_REQUESTED = False
            self.assertFalse(
                attn_fast._install_mla_query_alias(namespace)
            )
            self.assertIs(Attention.forward, original)
            self.assertFalse(attn_fast._MLA_QUERY_ALIAS_INSTALLED)
            self.assertEqual(
                attn_fast._MLA_QUERY_ALIAS_REASON,
                "disabled by K3_MLA_QUERY_ALIAS",
            )
        finally:
            attn_fast.MLA_QUERY_ALIAS_REQUESTED = requested
            attn_fast._MLA_QUERY_ALIAS_INSTALLED = installed
            attn_fast._MLA_QUERY_ALIAS_REASON = reason

        code = """
import os
import sys
sys.path.insert(0, os.path.abspath("tools"))
import attn_fast
from k3pkg import modeling_kimi_linear as ml
original = ml.KimiMLAAttention.forward
attn_fast.install(ml)
assert ml.KimiMLAAttention.forward is original
assert not attn_fast._MLA_QUERY_ALIAS_INSTALLED
assert attn_fast._MLA_QUERY_ALIAS_REASON == (
    "disabled by K3_MLA_QUERY_ALIAS"
)
"""
        environment = dict(os.environ)
        environment["K3_MLA_QUERY_ALIAS"] = "0"
        environment["K3_COMPILE"] = "0"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_multitoken_training_and_unknown_provider_keep_cat(self):
        attention = module().eval()
        for tokens in (2, 8):
            with self.subTest(tokens=tokens):
                source = hidden(1, tokens, 1900 + tokens)
                with capture_query_and_projection(
                    attention
                ) as captured:
                    with torch.no_grad():
                        attention(source)
                raw = captured["raw"][0]
                query = captured["query"][0]
                self.assertTrue(
                    torch.equal(query, legacy_query(raw, attention))
                )
                self.assertFalse(same_storage(query, raw))

        # kimi_run intentionally disables global grad tracking at import.
        # Keep this fallback contract independent of test/import ordering.
        with torch.enable_grad():
            training = module().train()
            source = hidden(1, 1, 1910).requires_grad_(True)
            with capture_query_and_projection(training) as captured:
                output = training(source)
            raw = captured["raw"][0]
            query = captured["query"][0]
            self.assertFalse(same_storage(query, raw))
            output.sum().backward()
            self.assertIsNotNone(source.grad)
            self.assertTrue(torch.isfinite(source.grad).all())

            eval_grad = module().eval()
            source = hidden(1, 1, 1911).requires_grad_(True)
            with capture_query_and_projection(eval_grad) as captured:
                output = eval_grad(source)
            self.assertFalse(same_storage(
                captured["query"][0],
                captured["raw"][0],
            ))
            output.sum().backward()
            self.assertIsNotNone(source.grad)
            self.assertTrue(torch.isfinite(source.grad).all())

        provider_name = "k3_alias_mutating_provider_fixture"
        provider_query = []

        def provider(
            attention_module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        ):
            provider_query.append(query)
            return ml.eager_attention_forward(
                attention_module,
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )

        ml.ALL_ATTENTION_FUNCTIONS[provider_name] = provider
        unknown = module().eval()
        unknown.config._attn_implementation = provider_name
        raw_values = []
        projection = unknown.q_b_proj
        handle = projection.register_forward_hook(
            lambda _module, _inputs, output: raw_values.append(output)
        )
        try:
            with torch.no_grad():
                unknown(hidden(1, 1, 1920))
        finally:
            handle.remove()
            del ml.ALL_ATTENTION_FUNCTIONS[provider_name]
        self.assertEqual(len(provider_query), 1)
        self.assertEqual(len(raw_values), 1)
        self.assertTrue(torch.equal(
            provider_query[0],
            legacy_query(raw_values[0], unknown),
        ))
        self.assertFalse(same_storage(
            provider_query[0],
            raw_values[0],
        ))

        original_eager = ml.eager_attention_forward
        mutation = {}

        def mutating_eager(
            attention_module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        ):
            mutation["query_storage"] = query.untyped_storage().data_ptr()
            query.add_(7.0)
            return original_eager(
                attention_module,
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )

        # A copied public-looking marker must not confer trust. Only the exact
        # fingerprinted provider or a wrapper registered by attn_fast may
        # receive aliased projection storage.
        mutating_eager._k3_mla_query_alias_safe = True
        poisoned = module().eval()
        poisoned_raw = []
        handle = poisoned.q_b_proj.register_forward_hook(
            lambda _module, _inputs, output: poisoned_raw.append(
                (output, output.clone())
            )
        )
        ml.eager_attention_forward = mutating_eager
        try:
            with torch.no_grad():
                poisoned(hidden(1, 1, 1921))
        finally:
            ml.eager_attention_forward = original_eager
            handle.remove()
        self.assertEqual(len(poisoned_raw), 1)
        raw, before = poisoned_raw[0]
        self.assertTrue(torch.equal(raw, before))
        self.assertNotEqual(
            mutation["query_storage"],
            raw.untyped_storage().data_ptr(),
        )

    def test_noncontiguous_projection_falls_back_exactly(self):
        attention = module(q_lora_rank=None).eval()
        replacement = StridedProjection(
            HIDDEN,
            HEADS * QK,
        )
        with torch.no_grad():
            replacement.inner.weight.copy_(attention.q_proj.weight)
        attention.q_proj = replacement

        source = hidden(2, 1, 1930)
        with capture_query_and_projection(attention) as captured:
            with torch.no_grad():
                attention(source)
        self.assertEqual(len(captured["raw"]), 1)
        raw = captured["raw"][0]
        query = captured["query"][0]
        self.assertFalse(raw.is_contiguous())
        self.assertTrue(torch.equal(
            query,
            legacy_query(raw, attention),
        ))
        self.assertFalse(same_storage(query, raw))

    def test_decode_sequence_matches_the_old_path_bit_for_bit(self):
        original = downloaded_forward()
        for q_lora_rank in (None, 5):
            for output_gate in (False, True):
                with self.subTest(
                    q_lora_rank=q_lora_rank,
                    output_gate=output_gate,
                ):
                    candidate = module(
                        q_lora_rank,
                        output_gate,
                    ).eval()
                    reference = copy.deepcopy(candidate).eval()
                    candidate_cache = ml.KimiDynamicCache(
                        CacheConfig()
                    )
                    reference_cache = ml.KimiDynamicCache(
                        CacheConfig()
                    )
                    sequence = [hidden(1, 3, 2000)]
                    sequence.extend(
                        hidden(1, 1, 2001 + step)
                        for step in range(7)
                    )
                    with torch.no_grad():
                        for step, value in enumerate(sequence):
                            candidate_input = value.clone()
                            reference_input = value.clone()
                            candidate_output = candidate(
                                candidate_input,
                                past_key_values=candidate_cache,
                            )
                            reference_output = original(
                                reference,
                                reference_input,
                                past_key_values=reference_cache,
                            )
                            self.assertTrue(torch.equal(
                                candidate_output,
                                reference_output,
                            ))
                            self.assertTrue(torch.equal(
                                candidate_input,
                                value,
                            ))
                            self.assertTrue(torch.equal(
                                reference_input,
                                value,
                            ))
                            self.assertTrue(torch.equal(
                                candidate_cache.key_cache[0],
                                reference_cache.key_cache[0],
                            ))
                            self.assertTrue(torch.equal(
                                candidate_cache.value_cache[0],
                                reference_cache.value_cache[0],
                            ))
                            if step == 3:
                                candidate_cache.truncate_mla(0, 4)
                                reference_cache.truncate_mla(0, 4)
                                self.assertTrue(torch.equal(
                                    candidate_cache.key_cache[0],
                                    reference_cache.key_cache[0],
                                ))
                                self.assertTrue(torch.equal(
                                    candidate_cache.value_cache[0],
                                    reference_cache.value_cache[0],
                                ))

    def test_real_shape_transform_matches_cpu_sdpa_layout_contract(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            for batch in (1, 2, 3):
                with self.subTest(dtype=dtype, batch=batch):
                    raw = torch.arange(
                        batch * 96 * 192,
                        dtype=torch.float32,
                    ).to(dtype).reshape(batch, 1, 96 * 192)
                    alias = raw.view(batch, 96, 1, 192)
                    heads = raw.view(
                        batch,
                        1,
                        96,
                        192,
                    ).transpose(1, 2)
                    q_pass, q_rot = torch.split(
                        heads,
                        [128, 64],
                        dim=-1,
                    )
                    old = torch.cat((q_pass, q_rot), dim=-1)
                    self.assertTrue(torch.equal(alias, old))
                    self.assertEqual(alias.stride(), old.stride())
                    self.assertEqual(
                        attn_fast._tensor_layout(alias),
                        "contiguous",
                    )
                    self.assertTrue(same_storage(alias, raw))
                    self.assertFalse(same_storage(old, raw))

    @unittest.skipUnless(
        bool(
            getattr(
                getattr(torch.backends, "mps", None),
                "is_available",
                lambda: False,
            )()
        ),
        "MPS is unavailable",
    )
    def test_physical_mps_alias_matches_old_query_bits(self):
        attention = module(q_lora_rank=5).eval().to("mps")
        source = hidden(2, 1, 2200).to("mps")
        before = source.clone()
        with capture_query_and_projection(attention) as captured:
            with torch.no_grad():
                output = attention(source)
            torch.mps.synchronize()
        raw = captured["raw"][0]
        query = captured["query"][0]
        old = legacy_query(raw, attention)
        self.assertTrue(torch.equal(query, old))
        self.assertEqual(query.stride(), old.stride())
        self.assertTrue(same_storage(query, raw))
        self.assertFalse(same_storage(old, raw))
        self.assertTrue(torch.equal(source, before))
        self.assertEqual(output.device.type, "mps")
        self.assertEqual(output.shape, (2, 1, HIDDEN))

    @unittest.skipUnless(
        hasattr(torch, "compile"),
        "torch.compile is unavailable",
    )
    def test_compile_shape_transitions_preserve_outputs(self):
        eager = module(q_lora_rank=5).eval()
        compiled_source = copy.deepcopy(eager)
        compiled = torch.compile(
            compiled_source,
            backend="eager",
            dynamic=False,
        )
        with torch.no_grad():
            for index, tokens in enumerate((1, 2, 8, 1)):
                value = hidden(1, tokens, 2300 + index)
                expected = eager(value)
                observed = compiled(value)
                self.assertTrue(torch.equal(observed, expected))

    def test_attn_compile_installer_remains_idempotent(self):
        code = """
import copy
import os
import sys
sys.path.insert(0, os.path.abspath("tools"))
import torch
import attn_fast
from k3pkg import modeling_kimi_linear as ml
from test_mla_query_alias import (
    CacheConfig,
    downloaded_forward,
    hidden,
    module,
)
real_compile = torch.compile
def eager_compile(function, **kwargs):
    return real_compile(
        function,
        backend="eager",
        dynamic=kwargs.get("dynamic", False),
    )
torch.compile = eager_compile
attn_fast.install(ml)
assert getattr(ml.KimiMLAAttention.forward, "_k3_mla_query_alias", False)
attn_fast.install(ml)
assert getattr(ml.KimiMLAAttention.forward, "_k3_mla_query_alias", False)
assert attn_fast._MLA_QUERY_ALIAS_INSTALLED
assert attn_fast._MLA_QUERY_ALIAS_REASON is None
assert not attn_fast._CPU_SDPA_INSTALLED
assert attn_fast._CPU_SDPA_INSTALL_REASON == (
    "K3_COMPILE must be 0 for the guarded SDPA path"
)
candidate = module().eval()
reference = copy.deepcopy(candidate).eval()
original = downloaded_forward()
candidate_cache = ml.KimiDynamicCache(CacheConfig())
reference_cache = ml.KimiDynamicCache(CacheConfig())
with torch.no_grad():
    for index, tokens in enumerate((2, 1, 1)):
        value = hidden(1, tokens, 2400 + index)
        got = candidate(value, past_key_values=candidate_cache)
        expected = original(
            reference,
            value,
            past_key_values=reference_cache,
        )
        assert torch.equal(got, expected)
        assert torch.equal(
            candidate_cache.key_cache[0],
            reference_cache.key_cache[0],
        )
        assert torch.equal(
            candidate_cache.value_cache[0],
            reference_cache.value_cache[0],
        )
assert attn_fast.STATS["graphs"] == 1
"""
        environment = dict(os.environ)
        environment["K3_COMPILE"] = "attn"
        environment["K3_MLA_QUERY_ALIAS"] = "1"
        environment["K3_MLA_CPU_SDPA"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_reload_optout_and_compile_rebuild_do_not_leave_stale_wrappers(self):
        code = """
import importlib
import os
import sys
sys.path.insert(0, os.path.abspath("tools"))
os.environ["K3_MLA_QUERY_ALIAS"] = "1"
os.environ["K3_MLA_CPU_SDPA"] = "0"
os.environ["K3_COMPILE"] = "0"
import attn_fast
from k3pkg import modeling_kimi_linear as ml

canonical = ml.KimiMLAAttention.forward
attn_fast.install(ml)
first_alias = ml.KimiMLAAttention.forward
assert getattr(first_alias, "_k3_mla_query_alias", False)

os.environ["K3_MLA_QUERY_ALIAS"] = "0"
attn_fast = importlib.reload(attn_fast)
attn_fast.install(ml)
assert ml.KimiMLAAttention.forward is canonical
assert not attn_fast._MLA_QUERY_ALIAS_INSTALLED

os.environ["K3_MLA_QUERY_ALIAS"] = "1"
os.environ["K3_COMPILE"] = "attn"
attn_fast = importlib.reload(attn_fast)
attn_fast.install(ml)
first_compiled = ml.KimiMLAAttention.forward

def marker_depth(function):
    depth = 0
    seen = set()
    while getattr(function, "_k3_mla_query_alias", False):
        assert id(function) not in seen
        seen.add(id(function))
        assert (
            getattr(function, "_k3_mla_query_alias_token", None)
            is attn_fast._MLA_QUERY_ALIAS_TOKEN
        )
        depth += 1
        function = function._k3_original
    assert function is canonical
    return depth

assert marker_depth(first_compiled) == 2
attn_fast = importlib.reload(attn_fast)
attn_fast.install(ml)
second_compiled = ml.KimiMLAAttention.forward
assert second_compiled is not first_compiled
assert marker_depth(second_compiled) == 2

attn_fast.MLA_QUERY_ALIAS_REQUESTED = False
attn_fast.install(ml)
disabled_compiled = ml.KimiMLAAttention.forward
assert getattr(disabled_compiled, "_k3_mla_compile_wrapper", False)
assert not getattr(disabled_compiled, "_k3_mla_query_alias", False)
assert disabled_compiled._k3_original is canonical

attn_fast.MLA_QUERY_ALIAS_REQUESTED = True
attn_fast.install(ml)
assert marker_depth(ml.KimiMLAAttention.forward) == 2
"""
        environment = dict(os.environ)
        environment["K3_MLA_QUERY_ALIAS"] = "1"
        environment["K3_MLA_CPU_SDPA"] = "0"
        environment["K3_COMPILE"] = "0"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_cpu_sdpa_gate_with_alias_enabled(self):
        environment = dict(os.environ)
        environment["K3_COMPILE"] = "0"
        environment["K3_MLA_QUERY_ALIAS"] = "1"
        environment["K3_MLA_CPU_SDPA"] = "1"
        completed = subprocess.run(
            [sys.executable, "tools/test_mla_cpu_sdpa.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        self.assertEqual(
            completed.returncode,
            0,
            msg=combined,
        )
        self.assertIn("native canaries: 4", combined)
        self.assertIn("Ran 6 tests", combined)


if __name__ == "__main__":
    unittest.main()
