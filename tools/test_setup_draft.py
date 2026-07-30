#!/usr/bin/env python3
"""Offline tests for the pinned universal-assistant installer."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_draft as setup  # noqa: E402


class DraftSetupTests(unittest.TestCase):
    def fixture(self):
        config = json.dumps({
            "model_type": "qwen3",
            "auto_map": None,
        }).encode()
        weights = b"safe tensor fixture"
        blobs = {
            "config.json": config,
            "model.safetensors": weights,
        }
        hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in blobs.items()
        }
        sizes = {name: len(data) for name, data in blobs.items()}
        return blobs, hashes, sizes

    def test_validate_checks_hash_and_data_only_config_gate(self):
        blobs, hashes, sizes = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            for name, data in blobs.items():
                with open(os.path.join(directory, name), "wb") as target:
                    target.write(data)
            with (
                mock.patch.object(setup, "FILES", hashes),
                mock.patch.object(setup, "SIZES", sizes),
            ):
                self.assertEqual(setup.validate(directory), (True, None))
                with open(
                    os.path.join(directory, "model.safetensors"), "r+b"
                ) as target:
                    target.seek(0)
                    target.write(b"X")
                valid, reason = setup.validate(directory)
        self.assertFalse(valid)
        self.assertIn("hash", reason)

    def test_install_stages_then_atomically_publishes_verified_files(self):
        blobs, hashes, sizes = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, "k3-draft-test")

            def download(name, path, expected, expected_size):
                self.assertEqual(expected, hashes[name])
                self.assertEqual(expected_size, sizes[name])
                with open(path, "xb") as target:
                    target.write(blobs[name])

            with (
                mock.patch.object(setup, "ROOT", root),
                mock.patch.object(setup, "DESTINATION", destination),
                mock.patch.object(setup, "FILES", hashes),
                mock.patch.object(setup, "SIZES", sizes),
                mock.patch.object(setup, "_download", side_effect=download),
                mock.patch.object(setup.sys, "platform", "linux"),
            ):
                setup.install()
                self.assertEqual(setup.validate(destination), (True, None))
            with open(
                os.path.join(destination, "deltafin-manifest.json"),
                encoding="utf-8",
            ) as source:
                manifest = json.load(source)
        self.assertEqual(manifest["revision"], setup.REVISION)
        self.assertEqual(manifest["files"], hashes)
        self.assertEqual(manifest["sizes"], sizes)

    def test_install_refuses_to_replace_invalid_existing_directory(self):
        _blobs, hashes, sizes = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, "k3-draft-test")
            os.mkdir(destination)
            with (
                mock.patch.object(setup, "ROOT", root),
                mock.patch.object(setup, "DESTINATION", destination),
                mock.patch.object(setup, "FILES", hashes),
                mock.patch.object(setup, "SIZES", sizes),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "refusing to replace"
                ):
                    setup.install()

    def test_probe_install_has_its_own_pinned_manifest(self):
        blobs, hashes, sizes = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, "k3-draft-probe-test")

            def download(name, path, expected, expected_size):
                self.assertEqual(expected, hashes[name])
                self.assertEqual(expected_size, sizes[name])
                with open(path, "xb") as target:
                    target.write(blobs[name])

            with (
                mock.patch.object(setup, "ROOT", root),
                mock.patch.object(
                    setup, "PROBE_DESTINATION", destination
                ),
                mock.patch.object(
                    setup, "PROBE_REPOSITORY", "fixture/probe"
                ),
                mock.patch.object(setup, "PROBE_REVISION", "probe-revision"),
                mock.patch.object(setup, "PROBE_FILES", hashes),
                mock.patch.object(setup, "PROBE_SIZES", sizes),
                mock.patch.object(
                    setup, "_download_probe", side_effect=download
                ),
                mock.patch.object(setup.sys, "platform", "linux"),
            ):
                setup.install_probe()
                self.assertEqual(
                    setup.validate(
                        destination, files=hashes, sizes=sizes
                    ),
                    (True, None),
                )
            with open(
                os.path.join(destination, "deltafin-manifest.json"),
                encoding="utf-8",
            ) as source:
                manifest = json.load(source)
        self.assertEqual(manifest["repository"], "fixture/probe")
        self.assertEqual(manifest["revision"], "probe-revision")

    def test_cli_installs_both_assistants(self):
        with (
            mock.patch.object(sys, "argv", ["setup_draft.py"]),
            mock.patch.object(setup, "install") as install,
            mock.patch.object(setup, "install_probe") as install_probe,
        ):
            setup.main()
        install.assert_called_once_with()
        install_probe.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
