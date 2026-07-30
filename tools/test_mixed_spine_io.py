#!/usr/bin/env python3
"""Focused byte-ownership tests for mixed-spine low-level I/O."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import mixed_spine
import spine_codec
import spine_io


class _FakeModule:
    NAMES = (
        "low.weight",
        "wide.weight",
        "base.weight",
        "tail.weight",
    )

    def named_parameters(self):
        return iter((name, object()) for name in self.NAMES)


class MixedSpineIOTests(unittest.TestCase):
    PREFIX = "language_model.model.layers.0."

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.mixed_dir = root / "mixed" / "tensors"
        self.int8_dir = root / "int8" / "tensors"
        self.resident_dir = root / "resident" / "tensors"
        for directory in (
            self.mixed_dir,
            self.int8_dir,
            self.resident_dir,
        ):
            directory.mkdir(parents=True)

        self.inv = {
            self.PREFIX + "low.weight": {
                "dtype": "BF16",
                "shape": [2, 64],
            },
            self.PREFIX + "wide.weight": {
                "dtype": "BF16",
                "shape": [1, 64],
            },
            self.PREFIX + "base.weight": {
                "dtype": "BF16",
                "shape": [1, 32],
            },
            self.PREFIX + "tail.weight": {
                "dtype": "F32",
                "shape": [3],
            },
        }
        self.payloads = {}
        self._write_codec("low.weight", "i4", self.mixed_dir)
        self._write_codec("wide.weight", "i6", self.mixed_dir)
        self._write_codec("base.weight", "i8", self.int8_dir)
        tail = bytes(range(12))
        tail_path = self.resident_dir / (self.PREFIX + "tail.weight")
        tail_path.parent.mkdir(parents=True, exist_ok=True)
        tail_path.write_bytes(tail)
        self.payloads[str(tail_path)] = tail

        self.old_mixed = (
            mixed_spine.MIXED_DIR,
            mixed_spine._META_PATH,
            mixed_spine._meta,
            mixed_spine._layouts,
            mixed_spine._pool,
            mixed_spine.READ_THREADS,
            mixed_spine._READ_THREADS_EXPLICIT,
            mixed_spine._readers,
        )
        mixed_spine.MIXED_DIR = str(self.mixed_dir)
        mixed_spine._META_PATH = str(self.mixed_dir.parent / "meta.json")
        mixed_spine._meta = {"format": "mixed-spine", "group": 64}
        mixed_spine._layouts = {}
        mixed_spine._pool = []
        mixed_spine.READ_THREADS = 1
        mixed_spine._READ_THREADS_EXPLICIT = True
        mixed_spine._readers = None

        self.old_io = (
            spine_io.FDCACHE,
            spine_io.CHUNK,
            spine_io.CHUNK_MB,
            spine_io.RDADVISE,
            spine_io.IOPOL,
            spine_io.NOCACHE,
            spine_io.STREAM_TIER,
            spine_io.ENABLED,
            set(spine_io._RESIDENT),
            spine_io._RESIDENT_ACTIVE,
            spine_io.OPENS,
        )
        spine_io.close_all()
        spine_io.OPENS = 0
        spine_io.FDCACHE = True
        spine_io.CHUNK = 7
        spine_io.CHUNK_MB = 7 / float(1 << 20)
        spine_io.RDADVISE = False
        spine_io.IOPOL = False
        spine_io.NOCACHE = False
        spine_io.configure_stream_tier(True)
        spine_io.set_resident_tier(set())

    def tearDown(self):
        readers = mixed_spine._readers
        if readers is not None:
            readers.shutdown(wait=True)
        spine_io.close_all()
        (
            mixed_spine.MIXED_DIR,
            mixed_spine._META_PATH,
            mixed_spine._meta,
            mixed_spine._layouts,
            mixed_spine._pool,
            mixed_spine.READ_THREADS,
            mixed_spine._READ_THREADS_EXPLICIT,
            mixed_spine._readers,
        ) = self.old_mixed
        (
            spine_io.FDCACHE,
            spine_io.CHUNK,
            spine_io.CHUNK_MB,
            spine_io.RDADVISE,
            spine_io.IOPOL,
            spine_io.NOCACHE,
            spine_io.STREAM_TIER,
            spine_io.ENABLED,
            resident,
            spine_io._RESIDENT_ACTIVE,
            spine_io.OPENS,
        ) = self.old_io
        spine_io._RESIDENT.clear()
        spine_io._RESIDENT.update(resident)
        self.tmp.cleanup()

    def _write_codec(self, short_name, codec, directory):
        full = self.PREFIX + short_name
        rows, cols = self.inv[full]["shape"]
        payload_bytes, scale_bytes = spine_codec.blob_sizes(
            codec, rows, cols, 64
        )
        payload = bytes((index * 17 + 3) % 251 for index in range(payload_bytes))
        scales = bytes((index * 29 + 5) % 251 for index in range(scale_bytes))
        payload_ext, scale_ext = spine_codec.EXT[codec]
        payload_path = directory / (full + payload_ext)
        scale_path = directory / (full + scale_ext)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload)
        scale_path.write_bytes(scales)
        self.payloads[str(payload_path)] = payload
        self.payloads[str(scale_path)] = scales

    def _read(self):
        return mixed_spine.read_pack(
            _FakeModule(),
            self.PREFIX,
            str(self.int8_dir),
            str(self.resident_dir),
            self.inv,
        )

    @staticmethod
    def _release(pack):
        mixed_spine._release(pack["q"])
        mixed_spine._release(pack["sc"])
        mixed_spine._release(pack["other"])

    def _assert_payloads(self, pack):
        lay = pack["lay"]
        qview = memoryview(pack["q"])
        scale_view = memoryview(pack["sc"])
        other_view = memoryview(pack["other"])
        for (
            _name,
            _full,
            codec,
            base,
            _shape,
            payload_offset,
            payload_bytes,
            scale_offset,
            scale_bytes,
        ) in lay["items"]:
            payload_ext, scale_ext = spine_codec.EXT[codec]
            self.assertEqual(
                bytes(
                    qview[
                        payload_offset:payload_offset + payload_bytes
                    ]
                ),
                self.payloads[base + payload_ext],
            )
            self.assertEqual(
                bytes(
                    scale_view[
                        scale_offset * 2:scale_offset * 2 + scale_bytes
                    ]
                ),
                self.payloads[base + scale_ext],
            )
        for (
            _name,
            _full,
            _dtype,
            _shape,
            offset,
            nbytes,
            path,
        ) in lay["oplan"]:
            self.assertEqual(
                bytes(other_view[offset:offset + nbytes]),
                self.payloads[path],
            )

    def test_spine_io_path_preserves_every_codec_and_tail_byte(self):
        observed = {}
        original = spine_io.run_jobs

        def capture(jobs, executor, threads, nocache=False):
            observed["lengths"] = [len(job[2]) for job in jobs]
            observed["threads"] = threads
            observed["nocache"] = nocache
            return original(jobs, executor, threads, nocache=nocache)

        with mock.patch.object(spine_io, "run_jobs", side_effect=capture):
            pack = self._read()
        try:
            self._assert_payloads(pack)
            self.assertTrue(observed["nocache"])
            self.assertEqual(observed["threads"], 1)
            self.assertTrue(observed["lengths"])
            self.assertLessEqual(max(observed["lengths"]), spine_io.CHUNK)
            planned = mixed_spine.plan_jobs(pack["lay"])
            self.assertEqual(
                [job[4] for job in planned],
                sorted((job[4] for job in planned), reverse=True),
            )
        finally:
            self._release(pack)

    def test_fd_cache_reuses_the_same_read_only_descriptors(self):
        first = self._read()
        self._assert_payloads(first)
        self._release(first)
        opens_after_first = spine_io.OPENS
        self.assertEqual(opens_after_first, len(self.payloads))

        second = self._read()
        try:
            self._assert_payloads(second)
            self.assertEqual(spine_io.OPENS, opens_after_first)
        finally:
            self._release(second)

    def test_disabled_spine_io_retains_the_original_reader(self):
        spine_io.FDCACHE = False
        spine_io.CHUNK = 0
        spine_io.CHUNK_MB = 0.0
        spine_io.RDADVISE = False
        spine_io.NOCACHE = False
        spine_io.configure_stream_tier(False)
        with mock.patch.object(
            spine_io,
            "run_jobs",
            side_effect=AssertionError("optimized reader should be disabled"),
        ):
            pack = self._read()
        try:
            self._assert_payloads(pack)
        finally:
            self._release(pack)

    def test_low_level_read_error_is_propagated(self):
        with mock.patch.object(
            spine_io,
            "run_jobs",
            side_effect=OSError("synthetic read failure"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic read failure"):
                self._read()

    def test_reader_width_contract_matches_spine_fast(self):
        mixed_spine._READ_THREADS_EXPLICIT = False
        sibling = types.SimpleNamespace(READ_THREADS=6)
        with mock.patch.dict(sys.modules, {"spine_fast": sibling}):
            mixed_spine._sync_runtime_read_threads()
        self.assertEqual(mixed_spine.READ_THREADS, 6)

        mixed_spine.configure_read_threads(3)
        self.assertEqual(mixed_spine.READ_THREADS, 3)
        executor = mixed_spine._pool_exec()
        self.assertIsNotNone(executor)
        with self.assertRaisesRegex(RuntimeError, "after pool construction"):
            mixed_spine.configure_read_threads(2)

    def test_invalid_reader_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            mixed_spine.configure_read_threads(0)


if __name__ == "__main__":
    unittest.main()
