#!/usr/bin/env python3
"""Weight-free tests for the portable positional-read layer.

The point of positional reads here is that one shared file object serves a
whole worker pool without the readers serialising on a file pointer, so the
concurrency test below is the one that matters most: it is exactly the usage
that seek-then-read would corrupt.
"""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import random
import sys
import tempfile
import threading
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import positional_io


CONTENT_SIZE = 1 << 20


def _content(size: int = CONTENT_SIZE) -> bytes:
    return random.Random(0xD37AF1).randbytes(size)


class PositionalReadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory(prefix="deltafin-pread-")
        cls.content = _content()
        cls.path = os.path.join(cls._directory.name, "blob.bin")
        with open(cls.path, "wb") as handle:
            handle.write(cls.content)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_reads_match_at_arbitrary_offsets(self):
        with positional_io.open_positional(self.path) as source:
            for offset, count in (
                (0, 1),
                (0, 4096),
                (1, 4095),          # unaligned start
                (4097, 12345),      # unaligned start and length
                (CONTENT_SIZE - 1, 1),
                (CONTENT_SIZE - 4096, 4096),
                (0, CONTENT_SIZE),  # whole file in one call
            ):
                with self.subTest(offset=offset, count=count):
                    buffer = bytearray(count)
                    got = source.read_into(memoryview(buffer), offset)
                    self.assertEqual(got, count)
                    self.assertEqual(
                        bytes(buffer), self.content[offset:offset + count]
                    )

    def test_read_allocates_and_matches(self):
        with positional_io.open_positional(self.path) as source:
            self.assertEqual(source.read(64, 128), self.content[128:192])
            self.assertEqual(source.read(0, 0), b"")

    def test_reading_past_the_end_is_an_error(self):
        with positional_io.open_positional(self.path) as source:
            with self.assertRaises(OSError):
                source.read_into(memoryview(bytearray(64)), CONTENT_SIZE)
            with self.assertRaises(OSError):
                # Straddles the end: the first bytes exist, the rest do not.
                source.read_into(memoryview(bytearray(64)), CONTENT_SIZE - 32)

    def test_read_loop_reassembles_capped_calls(self):
        # Force the loop that a >1 GiB request would otherwise be needed to
        # reach, so the offset arithmetic across calls is actually exercised.
        original = positional_io.MAX_SINGLE_READ
        positional_io.MAX_SINGLE_READ = 1000
        try:
            with positional_io.open_positional(self.path) as source:
                buffer = bytearray(8192)
                source.read_into(memoryview(buffer), 777)
                self.assertEqual(bytes(buffer), self.content[777:777 + 8192])
        finally:
            positional_io.MAX_SINGLE_READ = original

    def test_concurrent_reads_share_one_file_object(self):
        # A seek-then-read implementation passes every test above and fails
        # this one, which is the reason this module exists.
        chunk = 4096
        slots = CONTENT_SIZE // chunk
        errors: list[str] = []
        barrier = threading.Barrier(8)

        with positional_io.open_positional(self.path) as source:
            def worker(seed: int) -> None:
                rng = random.Random(seed)
                buffer = bytearray(chunk)
                view = memoryview(buffer)
                barrier.wait()
                for _ in range(200):
                    index = rng.randrange(slots)
                    offset = index * chunk
                    source.read_into(view, offset)
                    if bytes(buffer) != self.content[offset:offset + chunk]:
                        errors.append(f"mismatch at {offset}")
                        return

            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                list(pool.map(worker, range(8)))

        self.assertEqual(errors, [])

    def test_destination_must_be_writable(self):
        with positional_io.open_positional(self.path) as source:
            with self.assertRaises((ValueError, TypeError, BufferError)):
                source.read_into(memoryview(b"immutable"), 0)

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            positional_io.open_positional(
                os.path.join(self._directory.name, "absent.bin")
            )

    def test_close_is_idempotent(self):
        source = positional_io.open_positional(self.path)
        source.close()
        source.close()

    def test_supported_on_this_interpreter(self):
        self.assertTrue(positional_io.supported())

    def test_fileno_is_a_descriptor_only_where_one_exists(self):
        with positional_io.open_positional(self.path) as source:
            if os.name == "nt":
                # The overlapped handle is deliberately not a CRT descriptor.
                self.assertIsNone(source.fileno())
            else:
                self.assertIsInstance(source.fileno(), int)
                self.assertEqual(
                    os.fstat(source.fileno()).st_size, CONTENT_SIZE
                )


if __name__ == "__main__":
    unittest.main()
