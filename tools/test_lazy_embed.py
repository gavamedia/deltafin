#!/usr/bin/env python3
"""Exact/lifetime tests for the local embedding-row handoff."""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr
import positional_io


class _CountingSource:
    """Wrap a real reader so a test can count or perturb its reads.

    The T=1 fast path used to be injected at os.preadv. It now goes through
    positional_io, which supplies the same positional read on Windows too, so
    the seam moved to the reader object.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.chunk = None
        self.error = None

    def fileno(self):
        return self.inner.fileno()

    def read(self, count, offset):
        return self.inner.read(count, offset)

    def read_into(self, destination, offset):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.chunk is None:
            return self.inner.read_into(destination, offset)
        # Serve the request in short pieces to exercise the completion loop.
        view = memoryview(destination)
        total = len(view)
        done = 0
        while done < total:
            count = min(self.chunk, total - done)
            got = self.inner.read_into(view[done:done + count], offset + done)
            done += got
        return done

    def close(self):
        self.inner.close()


def _bits(tensor):
    return tensor.squeeze(0).contiguous().view(torch.uint16).numpy().tobytes()


class LazyEmbedDirectReadTests(unittest.TestCase):
    def setUp(self):
        self.rowbytes = kr.H * 2
        self.rows = [
            bytes(((row * 37 + offset * 13) & 0xFF)
                  for offset in range(self.rowbytes))
            for row in range(4)
        ]
        self.temp = tempfile.TemporaryDirectory(prefix="deltafin-embed-test-")
        self.path = Path(self.temp.name) / "embed.bin"
        self.path.write_bytes(b"".join(self.rows))
        self.embed = kr.LazyEmbed.__new__(kr.LazyEmbed)
        self.embed.path = str(self.path)
        self.embed.meta = {}
        self.embed.rowbytes = self.rowbytes
        self.embed._source = None
        self.embed._ensure_source()
        self.source = _CountingSource(self.embed._source)
        self.embed._source = self.source

    def tearDown(self):
        self.embed.close()
        self.temp.cleanup()

    def _call(self, ids):
        with (
            mock.patch.object(kr, "DEV", torch.device("cpu")),
            mock.patch.object(kr, "DT", torch.bfloat16),
        ):
            return self.embed(ids)

    def test_t1_read_fills_final_owner_with_exact_bits(self):
        result = self._call([2])
        self.assertEqual(_bits(result), self.rows[2])
        self.assertEqual(self.source.calls, 1)

        # The tensor must retain the bytearray owner when CPU .to() aliases it.
        self.embed.close()
        gc.collect()
        junk = [bytearray(self.rowbytes) for _ in range(8)]
        self.assertEqual(_bits(result), self.rows[2])
        self.assertEqual(len(junk), 8)

    def test_positive_partial_read_is_completed_exactly(self):
        self.source.chunk = max(1, self.rowbytes // 5)
        owner = self.embed._local_row_buffer(1)
        self.assertEqual(bytes(owner), self.rows[1])

    def test_read_propagates_original_os_error(self):
        for error in (
            InterruptedError("synthetic signal-handler error"),
            OSError("synthetic storage error"),
        ):
            with self.subTest(error=type(error).__name__):
                self.source.error = error
                try:
                    with self.assertRaises(type(error)) as caught:
                        self.embed._local_row_buffer(1)
                finally:
                    self.source.error = None
                self.assertIs(caught.exception, error)

    def test_short_read_retains_fail_closed_contract(self):
        with self.assertRaisesRegex(
            IOError, rf"short embedding read 0/{self.rowbytes}"
        ):
            self.embed._local_row_buffer(99)

    def test_absent_local_file_uses_the_remote_path(self):
        self.embed.close()
        self.embed.path = str(Path(self.temp.name) / "missing.bin")
        with mock.patch.object(
            self.embed, "_row", side_effect=[self.rows[3]]
        ) as row:
            result = self._call([3])
        self.assertEqual(_bits(result), self.rows[3])
        self.assertEqual(row.call_count, 1)

    def test_multirow_order_and_duplicates_keep_coalesced_path(self):
        ids = [2, 1, 2, 3]
        with mock.patch.object(
            self.embed,
            "_local_row_buffer",
            side_effect=AssertionError("T>1 must keep the coalesced path"),
        ):
            result = self._call(ids)
        self.assertEqual(_bits(result), b"".join(self.rows[i] for i in ids))

    def test_remote_fallback_never_reads_locally(self):
        self.embed.close()
        self.embed.path = str(Path(self.temp.name) / "missing.bin")
        before = self.source.calls
        with mock.patch.object(
            self.embed, "_row", side_effect=[self.rows[1], self.rows[3]]
        ) as row:
            result = self._call([1, 3])
        self.assertEqual(_bits(result), self.rows[1] + self.rows[3])
        self.assertEqual(row.call_count, 2)
        self.assertEqual(self.source.calls, before)


if __name__ == "__main__":
    unittest.main()
