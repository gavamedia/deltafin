#!/usr/bin/env python3
"""Capability and state checks for the automatic spine cache policy."""

import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import spine_cache
import spine_io


class AutomaticStreamPolicyTests(unittest.TestCase):
    SPINE_BYTES = 54_397_786_304

    def test_low_headroom_darwin_streams_with_capped_clean_tier(self):
        enabled, resident, reason = spine_cache.automatic_stream_policy(
            system="Darwin",
            physical_bytes=64 * spine_cache.GIB,
            recommended_working_set_bytes=52 * spine_cache.GIB,
            spine_bytes=self.SPINE_BYTES,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            resident,
            int(64 * spine_cache.GIB * spine_cache.AUTO_RESIDENT_FRACTION),
        )
        self.assertIn("exceeds", reason)

    def test_higher_memory_darwin_keeps_complete_spine_cacheable(self):
        enabled, resident, reason = spine_cache.automatic_stream_policy(
            system="Darwin",
            physical_bytes=96 * spine_cache.GIB,
            recommended_working_set_bytes=80 * spine_cache.GIB,
            spine_bytes=self.SPINE_BYTES,
        )
        self.assertFalse(enabled)
        self.assertEqual(resident, 0)
        self.assertIn("fits", reason)

    def test_unknown_and_non_darwin_capabilities_fail_closed(self):
        cases = (
            dict(
                system="Linux",
                physical_bytes=64 * spine_cache.GIB,
                recommended_working_set_bytes=0,
                spine_bytes=self.SPINE_BYTES,
            ),
            dict(
                system="Darwin",
                physical_bytes=0,
                recommended_working_set_bytes=0,
                spine_bytes=self.SPINE_BYTES,
            ),
            dict(
                system="Darwin",
                physical_bytes=64 * spine_cache.GIB,
                recommended_working_set_bytes=0,
                spine_bytes=0,
            ),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                enabled, resident, _ = (
                    spine_cache.automatic_stream_policy(**kwargs)
                )
                self.assertFalse(enabled)
                self.assertEqual(resident, 0)

    def test_resident_budget_scales_below_cap(self):
        enabled, resident, _ = spine_cache.automatic_stream_policy(
            system="Darwin",
            physical_bytes=32 * spine_cache.GIB,
            recommended_working_set_bytes=24 * spine_cache.GIB,
            spine_bytes=self.SPINE_BYTES,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            resident,
            int(32 * spine_cache.GIB * spine_cache.AUTO_RESIDENT_FRACTION),
        )


class AutomaticReaderPolicyTests(unittest.TestCase):
    def test_measured_m1_max_reader_shape_is_selected(self):
        threads, fdcache, chunk, reason = (
            spine_cache.automatic_reader_policy(
                system="Darwin",
                physical_bytes=64 * spine_cache.GIB,
                effective_cpu_count=10,
                recommended_working_set_bytes=55_662_788_608,
                max_buffer_length_bytes=41_747_087_360,
                stream_nocache=True,
            )
        )
        self.assertEqual(threads, 6)
        self.assertTrue(fdcache)
        self.assertEqual(chunk, 16 * (1 << 20))
        self.assertIn("qualified", reason)

    def test_newer_mac_does_not_inherit_m1_storage_tuning(self):
        result = spine_cache.automatic_reader_policy(
            system="Darwin",
            physical_bytes=64 * spine_cache.GIB,
            effective_cpu_count=16,
            recommended_working_set_bytes=62 * spine_cache.GIB,
            max_buffer_length_bytes=56 * spine_cache.GIB,
            stream_nocache=True,
        )
        self.assertEqual(result[:3], (None, None, None))
        self.assertIn("no matching", result[3])

    def test_reader_policy_fails_closed_off_darwin_or_without_streaming(self):
        cases = (
            dict(
                system="Linux",
                physical_bytes=64 * spine_cache.GIB,
                effective_cpu_count=10,
                recommended_working_set_bytes=0,
                max_buffer_length_bytes=0,
                stream_nocache=True,
            ),
            dict(
                system="Darwin",
                physical_bytes=64 * spine_cache.GIB,
                effective_cpu_count=10,
                recommended_working_set_bytes=55_662_788_608,
                max_buffer_length_bytes=41_747_087_360,
                stream_nocache=False,
            ),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                result = spine_cache.automatic_reader_policy(**kwargs)
                self.assertEqual(result[:3], (None, None, None))


class StreamTierConfigurationTests(unittest.TestCase):
    def test_runtime_configuration_updates_dispatch_flags(self):
        old = (
            spine_io.STREAM_TIER,
            spine_io.ENABLED,
            spine_io.FDCACHE,
            spine_io.FDCACHE_REASON,
            spine_io.CHUNK,
            spine_io.CHUNK_MB,
            spine_io.RDADVISE,
            spine_io.NOCACHE,
        )
        try:
            spine_io.FDCACHE = False
            spine_io.CHUNK = 0
            spine_io.RDADVISE = False
            spine_io.NOCACHE = False
            spine_io.configure_stream_tier(True)
            self.assertTrue(spine_io.STREAM_TIER)
            self.assertTrue(spine_io.ENABLED)
            spine_io.configure_stream_tier(False)
            self.assertFalse(spine_io.STREAM_TIER)
            self.assertFalse(spine_io.ENABLED)
            with (
                mock.patch.object(
                    spine_io, "_fd_limits", return_value=(65_536, 65_536)
                ),
                mock.patch.object(
                    spine_io, "_open_fd_count", return_value=16
                ),
            ):
                spine_io.configure_reader_shape(
                    fdcache=True,
                    chunk_bytes=16 * (1 << 20),
                    required_fds=3_706,
                )
            self.assertTrue(spine_io.FDCACHE)
            self.assertEqual(spine_io.CHUNK, 16 * (1 << 20))
            self.assertEqual(spine_io.CHUNK_MB, 16.0)
            self.assertTrue(spine_io.ENABLED)
        finally:
            (
                spine_io.STREAM_TIER,
                spine_io.ENABLED,
                spine_io.FDCACHE,
                spine_io.FDCACHE_REASON,
                spine_io.CHUNK,
                spine_io.CHUNK_MB,
                spine_io.RDADVISE,
                spine_io.NOCACHE,
            ) = old

    def test_low_hard_limit_disables_cache_but_keeps_chunked_reader(self):
        old = (
            spine_io.ENABLED,
            spine_io.FDCACHE,
            spine_io.FDCACHE_REASON,
            spine_io.CHUNK,
            spine_io.CHUNK_MB,
        )
        try:
            with (
                mock.patch.object(
                    spine_io, "_fd_limits", return_value=(256, 256)
                ),
                mock.patch.object(
                    spine_io, "_open_fd_count", return_value=12
                ),
            ):
                spine_io.configure_reader_shape(
                    fdcache=True,
                    chunk_bytes=16 * (1 << 20),
                    required_fds=3_706,
                )
            self.assertFalse(spine_io.FDCACHE)
            self.assertIn("RLIMIT_NOFILE=256", spine_io.FDCACHE_REASON)
            self.assertIn("3706", spine_io.FDCACHE_REASON)
            self.assertIn("ulimit -n 8192", spine_io.FDCACHE_REASON)
            self.assertEqual(spine_io.CHUNK, 16 * (1 << 20))
            self.assertTrue(spine_io.ENABLED)
        finally:
            (
                spine_io.ENABLED,
                spine_io.FDCACHE,
                spine_io.FDCACHE_REASON,
                spine_io.CHUNK,
                spine_io.CHUNK_MB,
            ) = old

    @unittest.skipIf(
        spine_io.resource is None,
        "RLIMIT_NOFILE is unavailable on this platform",
    )
    def test_low_soft_limit_is_raised_when_hard_limit_allows(self):
        limits = [(256, 65_536), (8_192, 65_536)]
        with (
            mock.patch.object(spine_io, "_fd_limits", side_effect=limits),
            mock.patch.object(spine_io, "_open_fd_count", return_value=12),
            mock.patch.object(spine_io.resource, "setrlimit") as setrlimit,
        ):
            enabled, reason = spine_io._fdcache_decision(
                True, required_fds=3_706
            )
        self.assertTrue(enabled)
        setrlimit.assert_called_once_with(
            spine_io.resource.RLIMIT_NOFILE, (8_192, 65_536)
        )
        self.assertIn("raised automatically from 256", reason)

    def test_fd_cache_capacity_boundary_is_exact(self):
        opened = 12
        required = 3_706
        minimum = opened + spine_io.FDCACHE_RESERVE + required
        with mock.patch.object(
            spine_io, "_open_fd_count", return_value=opened
        ):
            with mock.patch.object(
                spine_io, "_fd_limits", return_value=(minimum, minimum)
            ):
                enabled_at_limit, reason_at_limit = (
                    spine_io._fdcache_decision(
                        True, required_fds=required
                    )
                )
            with mock.patch.object(
                spine_io,
                "_fd_limits",
                return_value=(minimum - 1, minimum - 1),
            ):
                enabled_below_limit, reason_below_limit = (
                    spine_io._fdcache_decision(
                        True, required_fds=required
                    )
                )

        self.assertTrue(enabled_at_limit, reason_at_limit)
        self.assertFalse(enabled_below_limit, reason_below_limit)
        self.assertIn(
            f"leaves {required} safe descriptors", reason_at_limit
        )
        self.assertIn(
            f"leaves {required - 1} safe descriptors",
            reason_below_limit,
        )

    @unittest.skipIf(
        spine_io.resource is None,
        "RLIMIT_NOFILE is unavailable on this platform",
    )
    def test_real_low_hard_limit_uses_transient_chunked_reads(self):
        """The old process-lifetime cache failed around layer six at limit 256.

        Run in a child because lowering the hard limit is deliberately
        irreversible for that process. More distinct files than the child's
        limit ensure the old unbounded cache would raise EMFILE, while the
        guarded path must repeatedly read every byte with transient fds.
        """
        script = textwrap.dedent(
            """
            import concurrent.futures
            import os
            import pathlib
            import resource
            import sys
            import tempfile

            tools_dir = sys.argv[1]
            temp = tempfile.TemporaryDirectory()
            root = pathlib.Path(temp.name)
            paths = []
            expected = []
            buffers = []
            for index in range(128):
                payload = bytes([index % 251]) * 32
                path = root / str(index)
                path.write_bytes(payload)
                paths.append(str(path))
                expected.append(payload)
                buffers.append(bytearray(len(payload)))

            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            sys.path.insert(0, tools_dir)
            import spine_io

            spine_io.configure_reader_shape(
                fdcache=True,
                chunk_bytes=8,
                required_fds=len(paths),
            )
            assert not spine_io.FDCACHE, spine_io.FDCACHE_REASON
            files = [
                (path, 0, memoryview(buffer))
                for path, buffer in zip(paths, buffers)
            ]
            jobs = spine_io.make_jobs(files)
            assert max(len(job[2]) for job in jobs) == 8
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=6
            ) as executor:
                for _ in range(2):
                    spine_io.run_jobs(jobs, executor, 6)
                    assert [
                        bytes(buffer) for buffer in buffers
                    ] == expected

            assert spine_io.OPENS == 0
            assert not spine_io._FDS[0]
            assert not spine_io._FDS[1]
            """
        )
        env = os.environ.copy()
        env.update(
            {
                "K3_SPINE_FDCACHE": "1",
                "K3_SPINE_CHUNK_MB": "0",
                "K3_SPINE_IOPOL": "0",
                "K3_SPINE_NOCACHE": "0",
                "K3_SPINE_STREAM_NOCACHE": "0",
            }
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                os.path.abspath(os.path.dirname(__file__)),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"child stdout:\n{proc.stdout}\nchild stderr:\n{proc.stderr}",
        )

    def test_empty_resident_boundary_activates_all_streaming(self):
        old = (
            spine_io.STREAM_TIER,
            spine_io.ENABLED,
            set(spine_io._RESIDENT),
            spine_io._RESIDENT_ACTIVE,
        )
        try:
            spine_io.configure_stream_tier(
                True, spine_mode="mixed"
            )
            self.assertTrue(spine_io.stream_tier(
                "language_model.model.layers.0."
            ))
            self.assertTrue(spine_io.stream_tier(
                "language_model.model.layers.91."
            ))
        finally:
            (
                spine_io.STREAM_TIER,
                spine_io.ENABLED,
                resident,
                spine_io._RESIDENT_ACTIVE,
            ) = old
            spine_io._RESIDENT.clear()
            spine_io._RESIDENT.update(resident)

    def test_reenable_without_boundary_fails_closed(self):
        old = (
            spine_io.STREAM_TIER,
            spine_io.ENABLED,
            set(spine_io._RESIDENT),
            spine_io._RESIDENT_ACTIVE,
        )
        try:
            spine_io.configure_stream_tier(
                True, resident_prefixes=()
            )
            self.assertTrue(spine_io.stream_tier("layer"))
            spine_io.configure_stream_tier(False)
            spine_io.configure_stream_tier(True)
            self.assertFalse(spine_io.stream_tier("layer"))
        finally:
            (
                spine_io.STREAM_TIER,
                spine_io.ENABLED,
                resident,
                spine_io._RESIDENT_ACTIVE,
            ) = old
            spine_io._RESIDENT.clear()
            spine_io._RESIDENT.update(resident)

    def test_invalid_chunk_is_rejected(self):
        with self.assertRaises(ValueError):
            spine_io.configure_reader_shape(chunk_bytes=-1)


if __name__ == "__main__":
    unittest.main()
