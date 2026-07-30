#!/usr/bin/env python3
"""Regression tests for speculative emission bounds and capture lifetime."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402
import universal_draft as ud  # noqa: E402


def _logits(token: int, positions: int = 1, vocab: int = 64):
    vocab = max(vocab, token + 1)
    result = torch.zeros(1, positions, vocab)
    result[..., token] = 1
    return result


class GenerateBoundsTests(unittest.TestCase):
    def _run(self, burst, *, max_new=3, first=1):
        streamed = []
        logged = []
        with (
            mock.patch.object(
                kr, "forward_pass", return_value=_logits(first)
            ) as forward,
            mock.patch.object(
                kr, "_spec_step_deep", return_value=(list(burst), " spec-test")
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=True),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=max_new,
                spec=True,
                on_token=streamed.append,
                log=lambda step, tag, start, tokens: logged.append(tokens),
            )
        return generated, streamed, logged, forward.call_count

    def test_deep_burst_cannot_exceed_max_new(self):
        generated, streamed, logged, calls = self._run(
            range(2, 10), max_new=3
        )
        self.assertEqual(generated, [1, 2, 3])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [generated])
        self.assertEqual(calls, 1)

    def test_eos_trims_burst_before_streaming(self):
        generated, streamed, logged, _ = self._run(
            [2, kr.EOS_ID, 3, 4], max_new=10
        )
        self.assertEqual(generated, [1, 2, kr.EOS_ID])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [generated])

    def test_zero_budget_does_not_run_prefill(self):
        generated, streamed, logged, calls = self._run(
            [2, 3], max_new=0
        )
        self.assertEqual(generated, [])
        self.assertEqual(streamed, [])
        self.assertEqual(logged, [])
        self.assertEqual(calls, 0)

    def test_prefill_eos_stops_immediately(self):
        generated, streamed, logged, calls = self._run(
            [2, 3], max_new=10, first=kr.EOS_ID
        )
        self.assertEqual(generated, [kr.EOS_ID])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [])
        self.assertEqual(calls, 1)

    def test_explicit_logger_receives_snapshot_not_generation_state(self):
        snapshots = []

        def mutate_snapshot(step, tag, start, tokens):
            snapshots.append(list(tokens))
            tokens.clear()

        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), _logits(2), _logits(3)],
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=False,
                log=mutate_snapshot,
            )
        self.assertEqual(generated, [1, 2, 3])
        self.assertEqual(snapshots, [[1, 2], [1, 2, 3]])


class GenerationBookkeepingTests(unittest.TestCase):
    def test_speculation_reuses_one_private_rolling_history(self):
        refs = []
        snapshots = []

        def deep_step(
            layers, cache, embed, ctx, pending, step, remaining
        ):
            refs.append(ctx)
            snapshots.append(tuple(ctx))
            return [step + 1], " spec-history"

        prompt = [40, 41]
        with (
            mock.patch.object(kr, "forward_pass", return_value=_logits(1)),
            mock.patch.object(
                kr, "_spec_step_deep", side_effect=deep_step
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=True),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                prompt,
                max_new=4,
                spec=True,
            )
        self.assertEqual(generated, [1, 2, 3, 4])
        self.assertEqual(
            snapshots,
            [(40, 41, 1), (40, 41, 1, 2), (40, 41, 1, 2, 3)],
        )
        self.assertTrue(all(ref is refs[0] for ref in refs))
        self.assertEqual(prompt, [40, 41])

    def test_non_speculative_path_does_not_build_history(self):
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), _logits(2), _logits(3)],
            ),
            mock.patch.object(kr, "ngram_draft") as draft,
            mock.patch.object(kr.spec_decode, "enabled") as deep_enabled,
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.object(kr.time, "perf_counter_ns") as clock,
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=False,
            )
        self.assertEqual(generated, [1, 2, 3])
        draft.assert_not_called()
        deep_enabled.assert_not_called()
        clock.assert_not_called()

    def test_prefetch_environment_is_snapshotted_once_per_request(self):
        original_get = kr.os.environ.get
        keys = []

        def counted_get(key, default=None):
            keys.append(key)
            if key == "K3_PREFETCH":
                return "1"
            return original_get(key, default)

        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[
                    _logits(1),
                    _logits(2),
                    _logits(3),
                    _logits(4),
                ],
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token") as prefetch,
            mock.patch.object(kr, "PREAD", False),
            mock.patch.object(kr.os.environ, "get", side_effect=counted_get),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=4,
                spec=False,
            )
        self.assertEqual(generated, [1, 2, 3, 4])
        self.assertEqual(keys.count("K3_PREFETCH"), 1)
        self.assertEqual(prefetch.call_count, 3)

    def test_pread_path_skips_guaranteed_noop_prefetch_calls(self):
        original_get = kr.os.environ.get
        keys = []

        def counted_get(key, default=None):
            keys.append(key)
            return original_get(key, default)

        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), _logits(2), _logits(3)],
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token") as prefetch,
            mock.patch.object(kr, "PREAD", True),
            mock.patch.object(kr.os.environ, "get", side_effect=counted_get),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=False,
            )
        self.assertEqual(generated, [1, 2, 3])
        self.assertNotIn("K3_PREFETCH", keys)
        prefetch.assert_not_called()


class _ArgmaxVector:
    def __init__(self, values, counters):
        self._values = list(values)
        self._counters = counters

    def tolist(self):
        self._counters["bulk_transfers"] += 1
        return list(self._values)


class _VerifierRows:
    def __init__(self, values, counters):
        self._values = values
        self._counters = counters

    def argmax(self, dim):
        if dim != -1:
            raise AssertionError(f"unexpected argmax dim {dim}")
        self._counters["row_argmax_calls"] += 1
        return _ArgmaxVector(self._values, self._counters)


class _VerifierLogits:
    """Reject scalar-position reads so the test enforces one bulk handoff."""

    def __init__(self, values, counters):
        self._values = values
        self._counters = counters

    def __getitem__(self, index):
        if index != 0:
            self._counters["scalar_position_reads"] += 1
            raise AssertionError(f"unexpected scalar verifier read {index!r}")
        return _VerifierRows(self._values, self._counters)


class ShallowSpecArgmaxTests(unittest.TestCase):
    def test_accept_reads_both_argmax_ids_in_one_bulk_transfer(self):
        counters = {
            "row_argmax_calls": 0,
            "bulk_transfers": 0,
            "scalar_position_reads": 0,
        }
        verifier = _VerifierLogits([7, 8], counters)
        streamed = []
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), verifier, _logits(10)],
            ),
            mock.patch.object(kr, "ngram_draft", return_value=7),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr.spec_decode, "rollback_replay"),
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=True,
                on_token=streamed.append,
            )
        self.assertEqual(generated, [1, 7, 8])
        self.assertEqual(streamed, generated)
        self.assertEqual(
            counters,
            {
                "row_argmax_calls": 1,
                "bulk_transfers": 1,
                "scalar_position_reads": 0,
            },
        )
        restore.assert_not_called()

    def test_miss_uses_first_bulk_id_and_replays_one_cache_row(self):
        counters = {
            "row_argmax_calls": 0,
            "bulk_transfers": 0,
            "scalar_position_reads": 0,
        }
        verifier = _VerifierLogits([9, 63], counters)
        streamed = []
        snapshot = object()
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), verifier, _logits(10)],
            ),
            mock.patch.object(kr, "ngram_draft", return_value=7),
            mock.patch.object(
                kr, "snapshot_states", return_value=snapshot
            ),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(
                kr.spec_decode, "rollback_replay"
            ) as replay,
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=True,
                on_token=streamed.append,
            )
        self.assertEqual(generated, [1, 9, 10])
        self.assertEqual(streamed, generated)
        self.assertEqual(counters["row_argmax_calls"], 1)
        self.assertEqual(counters["bulk_transfers"], 1)
        self.assertEqual(counters["scalar_position_reads"], 0)
        restore.assert_not_called()
        replay.assert_called_once_with(mock.ANY, 1, {})


class ReplayCaptureLifetimeTests(unittest.TestCase):
    def _step(self, rollback):
        logits = torch.zeros(1, 3, 32)
        logits[0, 0, 7] = 1
        logits[0, 1, 8] = 1
        logits[0, 2, 9] = 1
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", rollback),
            mock.patch.object(kr.spec_decode, "next_depth", return_value=2),
            mock.patch.object(kr.spec_decode, "draft", return_value=[7, 8]),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm") as arm,
            mock.patch.object(kr.spec_decode, "release") as release,
            mock.patch.object(kr.spec_decode, "record"),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "forward_pass", return_value=logits),
        ):
            new, _ = kr._spec_step_deep(
                [], object(), lambda ids: ids, [1, 2], 2, 1
            )
        return new, arm.call_count, release.call_count

    def test_replay_arms_and_releases_once(self):
        new, arms, releases = self._step("replay")
        self.assertEqual(new, [7, 8, 9])
        self.assertEqual((arms, releases), (1, 1))

    def test_rerun_never_arms_unused_capture(self):
        new, arms, releases = self._step("rerun")
        self.assertEqual(new, [7, 8, 9])
        self.assertEqual((arms, releases), (0, 0))

    def test_replay_releases_capture_after_forward_failure(self):
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "next_depth", return_value=2),
            mock.patch.object(kr.spec_decode, "draft", return_value=[7, 8]),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm") as arm,
            mock.patch.object(kr.spec_decode, "release") as release,
            mock.patch.object(
                kr, "forward_pass", side_effect=RuntimeError("synthetic")
            ),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "restore_states"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                kr._spec_step_deep(
                    [], object(), lambda ids: ids, [1, 2], 2, 1
                )
        self.assertEqual((arm.call_count, release.call_count), (1, 1))


class ExactExternalDraftVerifierTests(unittest.TestCase):
    def test_wrong_hybrid_consensus_cannot_cross_target_verifier(self):
        class FixedAssistant:
            device = torch.device("cpu")
            max_assistant_tokens = 20
            confidence_threshold = 0.3

            def propose(self, _history, width):
                wrong = tuple(range(100, 100 + width))
                return ud.Proposal(
                    wrong,
                    width,
                    0.01,
                    raw_token_ids=wrong,
                    minimum_confidence=0.99,
                )

        hybrid = ud.HybridTextDrafter(
            FixedAssistant(), FixedAssistant()
        )
        proposal = hybrid.propose([6], 3)
        self.assertEqual(proposal.token_ids, (100, 101, 102))
        verifier = _VerifierLogits(
            [7, 20, 30, 40],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        narrow = _VerifierLogits(
            [7],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        with (
            mock.patch.object(
                kr.spec_decode, "snapshot_mla", return_value={}
            ),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "restore_states"),
            mock.patch.object(
                kr, "forward_pass", side_effect=[verifier, narrow]
            ),
        ):
            new, _tag, accepted = kr._verify_draft_tokens_exact(
                [],
                object(),
                lambda ids: ids,
                6,
                list(proposal.token_ids),
                1,
                remaining=4,
                source="uag",
            )
        self.assertEqual(new, [7])
        self.assertEqual(accepted, 0)
        self.assertNotIn(100, new)

    def test_partial_uag_match_restores_and_reruns_certified_prefix(self):
        counters = {
            "row_argmax_calls": 0,
            "bulk_transfers": 0,
            "scalar_position_reads": 0,
        }
        verifier = _VerifierLogits([7, 20, 30, 40], counters)
        narrow = _VerifierLogits([7, 20], counters)
        snapshot = object()
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(
                kr.spec_decode, "snapshot_mla", return_value={2: 10}
            ),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr.spec_decode, "rollback_replay") as replay,
            mock.patch.object(kr, "snapshot_states", return_value=snapshot),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(
                kr, "forward_pass", side_effect=[verifier, narrow]
            ),
        ):
            new, tag, accepted = kr._verify_draft_tokens_exact(
                [],
                object(),
                lambda ids: ids,
                6,
                [7, 8, 9],
                1,
                remaining=9,
                source="uag",
            )
        self.assertEqual(new, [7, 20])
        self.assertEqual(accepted, 1)
        self.assertIn("uag+2", tag)
        replay.assert_not_called()
        restore.assert_called_once_with(mock.ANY, snapshot)
        self.assertEqual(counters["bulk_transfers"], 2)

    def test_output_budget_truncates_cache_even_on_full_match(self):
        verifier = _VerifierLogits(
            [7, 8, 9, 10],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        narrow = _VerifierLogits(
            [7, 8],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        snapshot = object()
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr.spec_decode, "rollback_replay") as replay,
            mock.patch.object(
                kr, "snapshot_states", return_value=snapshot
            ),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(
                kr, "forward_pass", side_effect=[verifier, narrow]
            ),
        ):
            new, _tag, accepted = kr._verify_draft_tokens_exact(
                [],
                object(),
                lambda ids: ids,
                6,
                [7, 8, 9],
                1,
                remaining=2,
                source="uag",
            )
        self.assertEqual(new, [7, 8])
        self.assertEqual(accepted, 2)
        replay.assert_not_called()
        restore.assert_called_once_with(mock.ANY, snapshot)

    def test_eos_is_the_last_emitted_and_cached_boundary(self):
        verifier = _VerifierLogits(
            [7, kr.EOS_ID, 9, 10],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        narrow = _VerifierLogits(
            [7, kr.EOS_ID],
            {
                "row_argmax_calls": 0,
                "bulk_transfers": 0,
                "scalar_position_reads": 0,
            },
        )
        snapshot = object()
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr.spec_decode, "rollback_replay") as replay,
            mock.patch.object(
                kr, "snapshot_states", return_value=snapshot
            ),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(
                kr, "forward_pass", side_effect=[verifier, narrow]
            ),
        ):
            new, _tag, accepted = kr._verify_draft_tokens_exact(
                [],
                object(),
                lambda ids: ids,
                6,
                [7, 8, 9],
                1,
                remaining=8,
                source="uag",
            )
        self.assertEqual(new, [7, kr.EOS_ID])
        self.assertEqual(accepted, 1)
        replay.assert_not_called()
        restore.assert_called_once_with(mock.ANY, snapshot)

    def test_verifier_failure_restores_pristine_target_state(self):
        snapshot = object()
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr, "snapshot_states", return_value=snapshot),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(
                kr, "forward_pass", side_effect=RuntimeError("synthetic")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                kr._verify_draft_tokens_exact(
                    [],
                    object(),
                    lambda ids: ids,
                    6,
                    [7],
                    1,
                    remaining=2,
                    source="uag",
                )
        restore.assert_called_once_with(mock.ANY, snapshot)

    def test_restore_failure_is_distinct_and_fatal(self):
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm"),
            mock.patch.object(kr.spec_decode, "release"),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(
                kr, "restore_states", side_effect=RuntimeError("restore")
            ),
            mock.patch.object(
                kr, "forward_pass", side_effect=RuntimeError("verifier")
            ),
        ):
            with self.assertRaises(kr.ExactVerifierRestoreError):
                kr._verify_draft_tokens_exact(
                    [],
                    object(),
                    lambda ids: ids,
                    6,
                    [7],
                    1,
                    remaining=2,
                    source="uag",
                )


class UniversalDraftGeneratePolicyTests(unittest.TestCase):
    class Drafter:
        def __init__(self, fail=False, bookkeeping_fail=False):
            self.widths = []
            self.fail = fail
            self.bookkeeping_fail = bookkeeping_fail
            self.failures = 0
            self.verified = []

        def propose(self, history, width):
            self.widths.append(width)
            if self.fail:
                raise RuntimeError("synthetic draft failure")
            return type("Proposal", (), {
                "token_ids": tuple(range(100, 100 + width))
            })()

        def record_verified(self, accepted, emitted):
            if self.bookkeeping_fail:
                raise RuntimeError("synthetic bookkeeping failure")
            self.verified.append((accepted, emitted))

    def test_probe_qualifies_then_expands_to_eight_drafts(self):
        drafter = self.Drafter()
        with (
            mock.patch.object(kr, "forward_pass", return_value=_logits(1)),
            mock.patch.object(
                kr,
                "_verify_draft_tokens_exact",
                side_effect=[
                    ([2, 3, 4], " uag+3", 2),
                    (list(range(5, 14)), " uag+9", 8),
                ],
            ),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(
                os.environ,
                {
                    "K3_PREFETCH": "0",
                    "K3_UAG_PROBE_DRAFTS": "2",
                    "K3_UAG_MAX_DRAFTS": "8",
                },
            ),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=13,
                spec=True,
                universal_drafter=drafter,
            )
        self.assertEqual(generated, list(range(1, 14)))
        self.assertEqual(drafter.widths, [2, 8])
        self.assertEqual(drafter.verified, [(2, 3), (8, 9)])

    def test_confidence_skip_uses_one_target_token_then_retries_wide(self):
        class ConfidenceDrafter(self.Drafter):
            def propose(self, history, width):
                self.widths.append(width)
                if len(self.widths) == 2:
                    return type("Proposal", (), {
                        "token_ids": (),
                        "confidence_stopped": True,
                    })()
                return type("Proposal", (), {
                    "token_ids": tuple(range(100, 100 + width)),
                    "confidence_stopped": False,
                })()

        drafter = ConfidenceDrafter()
        with (
            mock.patch.object(
                kr, "forward_pass", side_effect=[_logits(1), _logits(5)]
            ),
            mock.patch.object(
                kr,
                "_verify_draft_tokens_exact",
                side_effect=[
                    ([2, 3, 4], " uag+3", 2),
                    (list(range(6, 15)), " uag+9", 8),
                ],
            ),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(
                os.environ,
                {
                    "K3_PREFETCH": "0",
                    "K3_UAG_PROBE_DRAFTS": "2",
                    "K3_UAG_MAX_DRAFTS": "8",
                },
            ),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=14,
                spec=True,
                universal_drafter=drafter,
            )
        self.assertEqual(generated, list(range(1, 15)))
        self.assertEqual(drafter.widths, [2, 8, 8])
        self.assertEqual(drafter.verified, [(2, 3), (8, 9)])

    def test_failed_proposal_disables_only_this_request(self):
        drafter = self.Drafter(fail=True)
        notices = []
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), _logits(2), _logits(3)],
            ),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=True,
                on_notice=notices.append,
                universal_drafter=drafter,
            )
        self.assertEqual(generated, [1, 2, 3])
        self.assertEqual(drafter.widths, [1])
        self.assertEqual(drafter.failures, 1)
        self.assertEqual(len(notices), 1)
        self.assertIn("proposal failed safely", notices[0])

    def test_bookkeeping_failure_keeps_already_certified_tokens(self):
        drafter = self.Drafter(bookkeeping_fail=True)
        with (
            mock.patch.object(kr, "forward_pass", return_value=_logits(1)) as fwd,
            mock.patch.object(
                kr,
                "_verify_draft_tokens_exact",
                return_value=([2, 3], " uag+2", 1),
            ),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=True,
                universal_drafter=drafter,
            )
        self.assertEqual(generated, [1, 2, 3])
        self.assertEqual(fwd.call_count, 1)
        self.assertEqual(drafter.failures, 1)

    def test_one_token_request_never_calls_assistant(self):
        drafter = self.Drafter()
        with mock.patch.object(
            kr, "forward_pass", return_value=_logits(1)
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=1,
                spec=True,
                universal_drafter=drafter,
            )
        self.assertEqual(generated, [1])
        self.assertEqual(drafter.widths, [])

    def test_fatal_restore_error_is_not_swallowed_into_t1_fallback(self):
        drafter = self.Drafter()
        with (
            mock.patch.object(kr, "forward_pass", return_value=_logits(1)) as fwd,
            mock.patch.object(
                kr,
                "_verify_draft_tokens_exact",
                side_effect=kr.ExactVerifierRestoreError("fatal"),
            ),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            with self.assertRaises(kr.ExactVerifierRestoreError):
                kr.generate(
                    [],
                    object(),
                    lambda ids: ids,
                    [41],
                    max_new=3,
                    spec=True,
                    universal_drafter=drafter,
                )
        self.assertEqual(fwd.call_count, 1)


class PackedHeadLifecycleTests(unittest.TestCase):
    def test_dense_fallback_is_not_followed_by_packed_reload(self):
        sentinel = torch.ones(2, 2)
        prior = (
            kr.INT8_LM_HEAD,
            kr._LM_Q,
            kr._LM_SC,
            kr._LM_W,
        )
        try:
            kr.INT8_LM_HEAD = True
            kr._LM_Q = None
            kr._LM_SC = None
            kr._LM_W = sentinel
            with mock.patch.object(kr, "_load_int8_packed") as load_packed:
                kr._ensure_lm_head_loaded()
            load_packed.assert_not_called()
            self.assertIs(kr._LM_W, sentinel)
            self.assertIsNone(kr._LM_Q)
        finally:
            (
                kr.INT8_LM_HEAD,
                kr._LM_Q,
                kr._LM_SC,
                kr._LM_W,
            ) = prior


if __name__ == "__main__":
    unittest.main()
