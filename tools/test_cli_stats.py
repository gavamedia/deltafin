#!/usr/bin/env python3
"""Tests for the optional command-line live statistics display."""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402


class LiveDecodeStatsTests(unittest.TestCase):
    def test_cli_flag_is_optional(self):
        parser = kr._build_cli_parser()
        self.assertFalse(parser.parse_args([]).stats)
        self.assertTrue(parser.parse_args(["--stats"]).stats)

    def test_only_chat_without_stats_uses_clean_output(self):
        parser = kr._build_cli_parser()
        self.assertTrue(kr._clean_chat_output(parser.parse_args(["--chat"])))
        self.assertFalse(
            kr._clean_chat_output(
                parser.parse_args(["--chat", "--stats"])
            )
        )
        self.assertFalse(kr._clean_chat_output(parser.parse_args([])))
        with mock.patch.object(kr, "PROFILE", True):
            self.assertFalse(
                kr._clean_chat_output(parser.parse_args(["--chat"]))
            )

    def test_clean_chat_fragments_are_contiguous_and_finish_once(self):
        output = io.StringIO()
        display = kr._CleanChatDisplay(True, stream=output)

        display.queue("The answer")
        display.queue(" is 666.")
        self.assertEqual(output.getvalue(), "")
        display.flush()
        display.finish()
        display.finish("ignored")

        self.assertEqual(
            output.getvalue(),
            "\n=== RESPONSE ===\nThe answer is 666.\n",
        )

    def test_disabled_clean_display_writes_nothing(self):
        output = io.StringIO()
        display = kr._CleanChatDisplay(False, stream=output)
        display.queue("hidden")
        display.flush()
        display.finish("tail")
        self.assertEqual(output.getvalue(), "")

    def test_pending_fragment_survives_finish_after_bookkeeping_error(self):
        output = io.StringIO()
        display = kr._CleanChatDisplay(True, stream=output)
        display.queue("already generated")
        display.finish()
        self.assertEqual(
            output.getvalue(),
            "\n=== RESPONSE ===\nalready generated\n",
        )

    def test_runtime_stdout_defers_diagnostics_only_after_response_starts(self):
        output = io.StringIO()
        display = kr._CleanChatDisplay(True, stream=output)
        runtime = kr._CleanRuntimeStdout(display, stream=output)

        runtime.write("prefill progress\n")
        display.queue("The answer is 666.")
        display.flush()
        runtime.write("[runtime fallback]\n")
        runtime.flush()

        self.assertEqual(
            output.getvalue(),
            "prefill progress\n\n=== RESPONSE ===\nThe answer is 666.",
        )
        self.assertEqual(runtime.drain(), "[runtime fallback]\n")
        self.assertEqual(runtime.drain(), "")

    def test_disabled_display_has_no_accounting_side_effect(self):
        stats = kr.LiveDecodeStats(False)
        self.assertIsNone(stats.record_prefill(2_000_000_000, 5))
        self.assertIsNone(stats.record_decode(4_000_000_000, 2))
        self.assertIsNone(stats.final_line(6_000_000_000))
        self.assertEqual(stats.decode_tokens, 0)
        self.assertEqual(stats.decode_ns, 0)

    def test_live_lines_include_running_speed_and_draft_acceptance(self):
        stats = kr.LiveDecodeStats(True)
        prefill = stats.record_prefill(2_000_000_000, 5)
        self.assertIn("prefill 2.000s", prefill)
        self.assertIn("5 prompt tokens", prefill)

        first = stats.record_decode(
            4_000_000_000,
            2,
            {"accepted_drafts": 2, "target_drafts": 2},
        )
        self.assertIn("decode 2 tok / 4.000s", first)
        self.assertIn("0.5000 tok/s", first)
        self.assertIn("2.000 s/token", first)
        self.assertIn("drafts 2/2 (100%)", first)

        second = stats.record_decode(
            6_000_000_000,
            3,
            {"accepted_drafts": 4, "target_drafts": 5},
        )
        self.assertIn("decode 5 tok / 10.000s", second)
        self.assertIn("0.5000 tok/s", second)
        self.assertIn("2.000 s/token", second)
        self.assertIn("last +3 tok in 6.000s", second)
        self.assertIn("drafts 4/5 (80%)", second)

        final = stats.final_line(12_000_000_000)
        self.assertIn("steady decode 5 tok / 10.000s", final)
        self.assertIn("model total 12.000s", final)

    def test_zero_decode_time_and_tokens_are_safe(self):
        stats = kr.LiveDecodeStats(True)
        line = stats.record_decode(0, 0)
        self.assertIn("0.0000 tok/s", line)
        self.assertIn("0.000 s/token", line)
        final = stats.final_line(0)
        self.assertIn("steady decode 0 tok / 0.000s", final)


if __name__ == "__main__":
    unittest.main()
