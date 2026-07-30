#!/usr/bin/env python3
"""Regression tests for local expert-read retry and fail-closed behavior."""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2  # noqa: E402


class _SyntheticSlot:
    def __init__(self, failures):
        self.failures = int(failures)
        self.calls = 0
        self.raw = {"source": "local"}

    def read(self, path):
        del path
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(f"synthetic local read failure {self.calls}")


class LocalExpertReadTests(unittest.TestCase):
    def _reader(self, slot):
        reader = fetch_v2._PreadReader(use_ring=False, workers=1)
        reader._take = lambda count: [slot] * count
        reader._retire = lambda slots: None
        self.addCleanup(reader.shutdown)
        return reader

    def test_valid_local_file_retries_once_without_becoming_a_miss(self):
        slot = _SyntheticSlot(failures=1)
        reader = self._reader(slot)
        with (
            mock.patch.object(fetch_v2, "_has_bin", return_value=True),
            mock.patch.object(
                fetch_v2, "_raw_file_valid", return_value=True
            ),
            mock.patch.object(
                fetch_v2,
                "_cache_path_bin",
                return_value="/local/L1-E7.bin",
            ),
            mock.patch.object(fetch_v2, "_drop_bin") as drop,
        ):
            raw, missing = reader.read_layer(1, [7])
        self.assertEqual(raw, {7: slot.raw})
        self.assertEqual(missing, [])
        self.assertEqual(slot.calls, 2)
        drop.assert_not_called()

    def test_persistent_valid_local_failure_is_not_sent_to_http(self):
        failure = fetch_v2.LocalExpertReadError("persistent local failure")
        fake_reader = types.SimpleNamespace(
            read_layer=mock.Mock(side_effect=failure)
        )
        with (
            mock.patch.object(fetch_v2, "EXPERT_READ", "pread"),
            mock.patch.object(fetch_v2, "reader", return_value=fake_reader),
            mock.patch.object(fetch_v2, "_fetch_group") as http_fetch,
        ):
            with self.assertRaisesRegex(
                fetch_v2.LocalExpertReadError, "persistent local"
            ):
                fetch_v2.fetch_experts(1, [7], dequant=False)
        http_fetch.assert_not_called()

    def test_file_that_really_disappeared_remains_a_cache_miss(self):
        slot = _SyntheticSlot(failures=2)
        reader = self._reader(slot)
        with (
            mock.patch.object(fetch_v2, "_has_bin", return_value=True),
            mock.patch.object(
                fetch_v2, "_raw_file_valid", return_value=False
            ),
            mock.patch.object(
                fetch_v2,
                "_cache_path_bin",
                return_value="/local/L1-E7.bin",
            ),
            mock.patch.object(fetch_v2, "_drop_bin") as drop,
        ):
            raw, missing = reader.read_layer(1, [7])
        self.assertEqual(raw, {})
        self.assertEqual(missing, [7])
        self.assertEqual(slot.calls, 1)
        drop.assert_called_once_with(1, 7)

    def test_grouped_valid_file_retries_locally_and_yields(self):
        slot = _SyntheticSlot(failures=1)
        reader = self._reader(slot)
        with (
            mock.patch.object(
                fetch_v2, "_raw_file_valid", return_value=True
            ),
            mock.patch.object(
                fetch_v2,
                "_cache_path_bin",
                return_value="/local/L1-E7.bin",
            ),
            mock.patch.object(fetch_v2, "_drop_bin") as drop,
        ):
            groups = list(reader.read_layer_groups(1, [7], 1))
        self.assertEqual(groups, [([7], {7: slot.raw})])
        self.assertEqual(slot.calls, 2)
        drop.assert_not_called()

    def test_grouped_valid_file_failure_cannot_become_http_fallback(self):
        slot = _SyntheticSlot(failures=2)
        reader = self._reader(slot)
        with (
            mock.patch.object(
                fetch_v2, "_raw_file_valid", return_value=True
            ),
            mock.patch.object(
                fetch_v2,
                "_cache_path_bin",
                return_value="/local/L1-E7.bin",
            ),
            mock.patch.object(fetch_v2, "_drop_bin") as drop,
        ):
            with self.assertRaisesRegex(
                fetch_v2.LocalExpertReadError, "refusing HTTP fallback"
            ):
                list(reader.read_layer_groups(1, [7], 1))
        self.assertEqual(slot.calls, 2)
        drop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
