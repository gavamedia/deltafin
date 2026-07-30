#!/usr/bin/env python3
"""Offline tests for exact cross-tokenizer draft translation."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import universal_draft as ud  # noqa: E402


class CharacterTargetTokenizer:
    def decode(self, ids):
        return "".join(chr(int(token)) for token in ids)

    def encode(self, text, *, allow_special_tokens):
        if allow_special_tokens is not True:
            raise AssertionError("target special-token policy was not explicit")
        return [ord(char) for char in text]


class CharacterAssistantTokenizer:
    eos_token_id = 0

    def __call__(self, text, *, return_tensors, add_special_tokens):
        if return_tensors != "pt" or add_special_tokens is not False:
            raise AssertionError("unexpected assistant tokenization options")
        ids = torch.tensor([[ord(char) for char in text]], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }

    def decode(
        self,
        ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("assistant decode must preserve exact text")
        return "".join(chr(int(token)) for token in ids)


class AppendModel:
    def __init__(self, suffix):
        self.suffix = tuple(ord(char) for char in suffix)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        suffix = torch.tensor(
            [self.suffix[:kwargs["max_new_tokens"]]],
            dtype=kwargs["input_ids"].dtype,
            device=kwargs["input_ids"].device,
        )
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


class ScoredAppendModel(AppendModel):
    def generate(self, **kwargs):
        sequences = super().generate(**kwargs)
        if not kwargs.get("return_dict_in_generate"):
            return sequences
        scores = []
        for index, token in enumerate(
            sequences[0, kwargs["input_ids"].shape[1]:]
        ):
            score = torch.zeros(128)
            score[int(token)] = 10.0 if index == 0 else 0.01
            scores.append(score.unsqueeze(0))
        return type("ScoredOutput", (), {
            "sequences": sequences,
            "scores": tuple(scores),
        })()


class UniversalTextDrafterTests(unittest.TestCase):
    def make_drafter(self, suffix="CDEF"):
        model = AppendModel(suffix)
        drafter = ud.UniversalTextDrafter(
            model,
            CharacterAssistantTokenizer(),
            CharacterTargetTokenizer(),
            "cpu",
            max_assistant_tokens=9,
        )
        return drafter, model

    def test_translation_is_capped_and_records_complete_overhead(self):
        drafter, model = self.make_drafter()
        with mock.patch.object(
            ud.time, "perf_counter", side_effect=[10.0, 10.25]
        ):
            proposal = drafter.propose([ord("A"), ord("B")], 2)
        self.assertEqual(proposal.token_ids, (ord("C"), ord("D")))
        self.assertEqual(proposal.assistant_tokens, 4)
        self.assertEqual(proposal.seconds, 0.25)
        call = model.calls[0]
        self.assertEqual(call["max_new_tokens"], 4)
        self.assertFalse(call["do_sample"])
        self.assertTrue(call["use_cache"])
        self.assertEqual(call["input_ids"].device.type, "cpu")
        self.assertEqual(call["attention_mask"].device.type, "cpu")
        self.assertEqual(drafter.status()["proposals"], 1)
        self.assertEqual(drafter.status()["target_drafts"], 2)

    def test_zero_budget_skips_tokenizers_and_model(self):
        drafter, model = self.make_drafter()
        proposal = drafter.propose([65], 0)
        self.assertEqual(proposal.token_ids, ())
        self.assertEqual(model.calls, [])
        self.assertEqual(drafter.status()["proposals"], 0)

    def test_non_roundtripping_target_history_fails_before_generate(self):
        class RewritingTarget(CharacterTargetTokenizer):
            def encode(self, text, *, allow_special_tokens):
                return [999]

        model = AppendModel("C")
        drafter = ud.UniversalTextDrafter(
            model,
            CharacterAssistantTokenizer(),
            RewritingTarget(),
            "cpu",
        )
        with self.assertRaisesRegex(
            ud.UniversalDraftError, "does not round-trip"
        ):
            drafter.propose([65, 66], 1)
        self.assertEqual(model.calls, [])

    def test_assistant_prefix_rewrite_fails_closed(self):
        drafter, _model = self.make_drafter()

        def rewritten_decode(*_args, **_kwargs):
            return "XBC"

        drafter.assistant_tokenizer.decode = rewritten_decode
        with self.assertRaisesRegex(
            ud.UniversalDraftError, "did not preserve"
        ):
            drafter.propose([65, 66], 1)

    def test_verified_counters_are_target_owned(self):
        drafter, _model = self.make_drafter()
        drafter.record_verified(2, 3)
        status = drafter.status()
        self.assertEqual(status["accepted_drafts"], 2)
        self.assertEqual(status["emitted_tokens"], 3)

    def test_confidence_threshold_crops_before_uncertain_token(self):
        model = ScoredAppendModel("CDEF")
        drafter = ud.UniversalTextDrafter(
            model,
            CharacterAssistantTokenizer(),
            CharacterTargetTokenizer(),
            "cpu",
            max_assistant_tokens=9,
            confidence_threshold=0.3,
        )
        proposal = drafter.propose([ord("A"), ord("B")], 4)
        self.assertEqual(proposal.token_ids, (ord("C"),))
        self.assertTrue(proposal.confidence_stopped)
        # Generation work is counted even when policy discards uncertain text.
        self.assertEqual(proposal.assistant_tokens, 4)
        self.assertEqual(len(model.calls), 1)
        status = drafter.status()
        self.assertEqual(status["confidence_truncations"], 1)
        self.assertEqual(status["confidence_skips"], 0)

    def test_two_draft_admission_probe_ignores_confidence_cutoff(self):
        model = ScoredAppendModel("CDEF")
        drafter = ud.UniversalTextDrafter(
            model,
            CharacterAssistantTokenizer(),
            CharacterTargetTokenizer(),
            "cpu",
            max_assistant_tokens=9,
            confidence_threshold=0.3,
        )
        proposal = drafter.propose([ord("A"), ord("B")], 2)
        self.assertEqual(proposal.token_ids, (ord("C"), ord("D")))
        self.assertFalse(proposal.confidence_stopped)
        self.assertFalse(model.calls[0]["return_dict_in_generate"])

    def test_request_mode_aliases_and_invalid_value(self):
        for value in ("1", "true", "on", "yes", "enabled", "auto"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"K3_UAG_DRAFT": value}, clear=False
            ):
                self.assertTrue(ud.requested())
        for value in ("0", "false", "off", "no", "disabled", ""):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"K3_UAG_DRAFT": value}, clear=False
            ):
                self.assertFalse(ud.requested())
        with mock.patch.dict(
            os.environ, {"K3_UAG_DRAFT": "sometimes"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "K3_UAG_DRAFT"):
                ud.requested()

    def test_default_mode_is_auto_but_cpu_load_fails_softly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ud.mode(), "auto")
            self.assertIsNone(
                ud.load_local_drafter(
                    "/missing", CharacterTargetTokenizer(), "cpu"
                )
            )

    def test_default_model_prefers_complete_1_7b_then_complete_0_6b(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "k3-draft-qwen3-0.6b-base")
            new = os.path.join(root, "k3-draft-qwen3-1.7b-base")
            with mock.patch.object(
                ud.os.path,
                "isfile",
                side_effect=lambda path: path.startswith(old),
            ):
                self.assertEqual(ud._default_model_path(root), old)
            with mock.patch.object(ud.os.path, "isfile", return_value=True):
                self.assertEqual(ud._default_model_path(root), new)

    def test_hybrid_wide_identity_comes_from_config_not_directory_name(self):
        wide = mock.Mock(
            model_type="qwen3",
            hidden_size=2_048,
            intermediate_size=6_144,
            num_hidden_layers=28,
        )
        probe = mock.Mock(
            model_type="qwen3",
            hidden_size=1_024,
            intermediate_size=3_072,
            num_hidden_layers=28,
        )
        self.assertTrue(ud._is_hybrid_wide_config(wide))
        self.assertFalse(ud._is_hybrid_wide_config(probe))

    def test_forced_load_rejects_missing_files_before_transformers(self):
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(
                os.environ,
                {"K3_UAG_DRAFT": "on", "K3_SPEC": "1"},
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(
                ud.UniversalDraftError, "missing"
            ):
                ud.load_local_drafter(
                    root,
                    CharacterTargetTokenizer(),
                    "cpu",
                    model_dir=os.path.join(root, "missing-assistant"),
                )


class HybridTextDrafterTests(unittest.TestCase):
    class FakeDrafter:
        def __init__(self, proposals):
            self._proposals = list(proposals)
            self.calls = []
            self.device = torch.device("cpu")
            self.max_assistant_tokens = 20
            self.confidence_threshold = 0.3

        def propose(self, history, width):
            self.calls.append((tuple(history), width))
            return self._proposals.pop(0)

    def test_narrow_probe_uses_only_small_assistant(self):
        probe = self.FakeDrafter([ud.Proposal((1, 2), 4, 0.1)])
        wide = self.FakeDrafter([])
        hybrid = ud.HybridTextDrafter(probe, wide)
        result = hybrid.propose([9], 2)
        self.assertEqual(result.token_ids, (1, 2))
        self.assertEqual(len(probe.calls), 1)
        self.assertEqual(wide.calls, [])

    def test_raw_consensus_overrides_individual_confidence_crop(self):
        raw = (1, 2, 3, 4)
        probe = self.FakeDrafter([
            ud.Proposal(
                (1, 2), 6, 0.2, True, raw, 0.2
            )
        ])
        wide = self.FakeDrafter([
            ud.Proposal(
                (1, 2, 3), 6, 0.3, True, raw, 0.1
            )
        ])
        hybrid = ud.HybridTextDrafter(probe, wide)
        result = hybrid.propose([9], 4)
        self.assertEqual(result.token_ids, raw)
        self.assertFalse(result.confidence_stopped)
        self.assertEqual(result.assistant_tokens, 12)
        self.assertEqual(hybrid.consensus_selections, 1)

    def test_divergence_requires_real_confidence_margin_for_large_model(self):
        probe = self.FakeDrafter([
            ud.Proposal((1,), 6, 0.2, False, (1, 2, 3), 0.24),
            ud.Proposal((7, 8), 6, 0.2, False, (7, 8, 9), 0.47),
        ])
        wide = self.FakeDrafter([
            ud.Proposal((4, 5, 6), 6, 0.3, False, (4, 5, 6), 0.46),
            ud.Proposal((10,), 6, 0.3, False, (10, 11, 12), 0.46),
        ])
        hybrid = ud.HybridTextDrafter(probe, wide)
        self.assertEqual(hybrid.propose([9], 3).token_ids, (4, 5, 6))
        self.assertEqual(hybrid.propose([9], 3).token_ids, (7, 8))
        self.assertEqual(hybrid.wide_selections, 1)
        self.assertEqual(hybrid.probe_selections, 1)


if __name__ == "__main__":
    unittest.main()
