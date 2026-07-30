#!/usr/bin/env python3
"""Offline tests for the immutable Kimi-K3 source and setup gates."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model_source  # noqa: E402
import inventory_meta  # noqa: E402
import setup_k3 as setup  # noqa: E402


def _can_create_symlinks() -> bool:
    """Whether this process may create a symbolic link at all.

    Windows requires SeCreateSymbolicLinkPrivilege, which an ordinary account
    only holds with Developer Mode enabled.  The refusal these tests check is
    real on every platform; without the privilege the fixture cannot be built,
    which is a property of the test account, not of the code under test.
    """
    with tempfile.TemporaryDirectory() as directory:
        target = os.path.join(directory, "target")
        with open(target, "wb"):
            pass
        try:
            os.symlink(target, os.path.join(directory, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


SYMLINKS_AVAILABLE = _can_create_symlinks()
requires_symlinks = unittest.skipUnless(
    SYMLINKS_AVAILABLE,
    "creating symbolic links is not permitted for this process",
)


class ModelSourcePinTests(unittest.TestCase):
    def test_default_source_is_an_immutable_revision(self):
        self.assertEqual(
            model_source.MODEL_REVISION,
            "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
        )
        self.assertIn(model_source.MODEL_REVISION, model_source.DEFAULT_HF_PATH)
        self.assertNotIn("/resolve/main/", model_source.DEFAULT_HF_PATH)
        self.assertEqual(
            model_source.base_url(),
            "https://huggingface.co" + model_source.DEFAULT_HF_PATH,
        )
        self.assertEqual(model_source.PINNED_INVENTORY_SIZE, 108_581_016)
        self.assertEqual(
            model_source.PINNED_INVENTORY_SHA256,
            "b287e9659afbfd361b1485721a6703a5bc55bb8399ed074a1e5a29803958b425",
        )
        self.assertEqual(model_source.PINNED_INVENTORY_TENSORS, 497_220)

    def test_explicit_mirror_overrides_remain_authoritative(self):
        with mock.patch.dict(
            os.environ,
            {
                "K3_HF_HOST": "models.example.test",
                "K3_HF_PATH": "mirror/kimi/",
            },
            clear=False,
        ):
            self.assertEqual(model_source.hf_host(), "models.example.test")
            self.assertEqual(model_source.hf_path(), "/mirror/kimi/")
            self.assertEqual(
                model_source.base_url(),
                "https://models.example.test/mirror/kimi/",
            )

    def test_every_default_url_consumer_has_dropped_mutable_main(self):
        for name in (
            "setup_k3.py",
            "fetch_spine.py",
            "fetch_v2.py",
            "k3loader.py",
            "test_mxfp4.py",
        ):
            source = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn(
                "moonshotai/Kimi-K3/resolve/main", source, msg=name
            )


class SetupPinTests(unittest.TestCase):
    def fixture(self, payload=b"verified fixture"):
        name = "fixture.bin"
        specs = {
            name: (len(payload), hashlib.sha256(payload).hexdigest())
        }
        return name, payload, specs

    def directories(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        meta = root / "k3-meta"
        package = root / "tools" / "k3pkg"
        meta.mkdir()
        package.mkdir(parents=True)
        self.addCleanup(temporary.cleanup)
        return meta, package

    @contextlib.contextmanager
    def patched(self, meta, package, specs, base="https://example.test/pin/"):
        with (
            mock.patch.object(setup, "META", str(meta)),
            mock.patch.object(setup, "PKG", str(package)),
            mock.patch.object(setup, "FILE_SPECS", specs),
            mock.patch.object(setup, "BASE", base),
        ):
            yield

    def test_existing_regular_file_must_match_without_network(self):
        name, payload, specs = self.fixture()
        meta, package = self.directories()
        (meta / name).write_bytes(payload)
        with (
            self.patched(meta, package, specs),
            mock.patch.object(
                setup.urllib.request,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ),
        ):
            result = setup.fetch(name)
        self.assertIn("verified", result)

    def test_corrupt_existing_file_is_refused_and_preserved(self):
        name, payload, specs = self.fixture()
        meta, package = self.directories()
        destination = meta / name
        destination.write_bytes(payload + b"x")
        with (
            self.patched(meta, package, specs),
            mock.patch.object(
                setup.urllib.request,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                setup.fetch(name)
        self.assertEqual(destination.read_bytes(), payload + b"x")

    @requires_symlinks
    def test_existing_symlink_is_refused(self):
        name, payload, specs = self.fixture()
        meta, package = self.directories()
        source = meta / "source"
        source.write_bytes(payload)
        os.symlink(source, meta / name)
        with self.patched(meta, package, specs):
            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                setup.fetch(name)

    def test_missing_file_is_hashed_fsynced_and_atomically_published(self):
        name, payload, specs = self.fixture()
        meta, package = self.directories()
        requests = []

        def open_fixture(request, timeout):
            requests.append((request.full_url, timeout))
            return io.BytesIO(payload)

        with (
            self.patched(meta, package, specs),
            mock.patch.object(
                setup.urllib.request,
                "urlopen",
                side_effect=open_fixture,
            ),
        ):
            result = setup.fetch(name)
        self.assertEqual((meta / name).read_bytes(), payload)
        self.assertEqual(requests, [("https://example.test/pin/" + name, 120)])
        self.assertIn("downloaded and verified", result)
        self.assertEqual(list(meta.glob("*.part")), [])

    def test_bad_download_is_never_published(self):
        name, _payload, specs = self.fixture()
        meta, package = self.directories()
        with (
            self.patched(meta, package, specs),
            mock.patch.object(
                setup.urllib.request,
                "urlopen",
                return_value=io.BytesIO(b"corrupt"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "download"):
                setup.fetch(name)
        self.assertFalse((meta / name).exists())
        self.assertEqual(list(meta.glob("*.part")), [])

    def test_package_copy_accepts_only_a_verified_source(self):
        name, payload, specs = self.fixture()
        meta, package = self.directories()
        (meta / name).write_bytes(payload)
        with self.patched(meta, package, specs):
            setup._copy_verified_package_file(name)
        self.assertEqual((package / name).read_bytes(), payload)

        (meta / name).write_bytes(b"corrupt")
        with self.patched(meta, package, specs):
            with self.assertRaisesRegex(RuntimeError, "unverified"):
                setup._copy_verified_package_file(name)
        self.assertEqual((package / name).read_bytes(), payload)

    def test_shard_headers_use_the_same_pinned_base(self):
        header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
        responses = [
            struct.pack("<Q", len(header)),
            header,
        ]
        requests = []

        def open_fixture(request, timeout):
            requests.append(
                (request.full_url, request.get_header("Range"), timeout)
            )
            return io.BytesIO(responses.pop(0))

        with (
            mock.patch.object(
                setup,
                "BASE",
                "https://example.test/resolve/immutable/",
            ),
            mock.patch.object(
                setup.urllib.request,
                "urlopen",
                side_effect=open_fixture,
            ),
        ):
            shard, header_length, contents = setup.shard_header(1)

        expected_url = (
            "https://example.test/resolve/immutable/"
            "model-00001-of-000096.safetensors"
        )
        self.assertEqual(shard, "model-00001-of-000096.safetensors")
        self.assertEqual(header_length, len(header))
        self.assertEqual(contents, {})
        self.assertEqual(
            requests,
            [
                (expected_url, "bytes=0-7", 120),
                (expected_url, f"bytes=8-{7 + len(header)}", 120),
            ],
        )


class InventoryHardeningTests(unittest.TestCase):
    def directories(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        meta = root / "k3-meta"
        meta.mkdir()
        self.addCleanup(temporary.cleanup)
        return root, meta

    def valid_inventory(self):
        return {
            f"tensor_{number:05d}": {
                "dtype": "U8",
                "shape": [1],
                "offsets": [0, 1],
                "shard": f"model-{number:05d}-of-000096.safetensors",
                "hlen": 128 + number,
            }
            for number in range(1, 97)
        }

    def serialized(self, document):
        return json.dumps(document).encode("utf-8")

    @contextlib.contextmanager
    def pinned_fixture(self, payload):
        with (
            mock.patch.object(
                inventory_meta, "EXPECTED_TENSOR_COUNT", 96
            ),
            mock.patch.object(
                model_source, "PINNED_INVENTORY_SIZE", len(payload)
            ),
            mock.patch.object(
                model_source,
                "PINNED_INVENTORY_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ),
        ):
            yield

    def test_verified_inventory_requires_exact_pinned_bytes(self):
        _root, meta = self.directories()
        document = self.valid_inventory()
        payload = self.serialized(document)
        path = meta / inventory_meta.INVENTORY_FILENAME
        path.write_bytes(payload)
        with self.pinned_fixture(payload):
            loaded = inventory_meta.load_verified_inventory(str(meta))
        self.assertEqual(loaded, document)

        changed = payload + b" "
        path.write_bytes(changed)
        with self.pinned_fixture(payload):
            with self.assertRaisesRegex(
                inventory_meta.InventoryValidationError, "bytes"
            ):
                inventory_meta.load_verified_inventory(str(meta))

    @requires_symlinks
    def test_existing_inventory_symlink_is_refused_before_network(self):
        _root, meta = self.directories()
        target = meta / "untrusted.json"
        target.write_text("{}", encoding="utf-8")
        destination = meta / inventory_meta.INVENTORY_FILENAME
        os.symlink(target, destination)
        with (
            mock.patch.object(setup, "META", str(meta)),
            mock.patch.object(
                setup,
                "_build_inventory",
                side_effect=AssertionError("network must not be used"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                setup.ensure_inventory()
        self.assertTrue(destination.is_symlink())

    def test_corrupt_existing_inventory_is_refused_and_preserved(self):
        _root, meta = self.directories()
        destination = meta / inventory_meta.INVENTORY_FILENAME
        payload = b'{"not":"the pinned inventory"}'
        destination.write_bytes(payload)
        with (
            mock.patch.object(setup, "META", str(meta)),
            mock.patch.object(
                setup,
                "_build_inventory",
                side_effect=AssertionError("network must not be used"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to use"):
                setup.ensure_inventory()
        self.assertEqual(destination.read_bytes(), payload)

    def test_tensor_names_cannot_escape_output_root(self):
        root, _meta = self.directories()
        output = root / "tensors"
        output.mkdir()
        self.assertEqual(
            inventory_meta.safe_tensor_output_path(
                str(output), "language_model.layer.weight"
            ),
            str(output / "language_model.layer.weight"),
        )
        for name in (
            "../escape",
            "/tmp/escape",
            r"..\escape",
            "nested/tensor",
            "nested\\tensor",
            ".",
            "..",
        ):
            with self.subTest(name=name):
                with self.assertRaises(
                    inventory_meta.InventoryValidationError
                ):
                    inventory_meta.safe_tensor_output_path(str(output), name)

    def test_shard_filename_must_be_exactly_in_1_through_96(self):
        for shard in (
            "model-00000-of-000096.safetensors",
            "model-00097-of-000096.safetensors",
            "model-00001-of-000095.safetensors",
            "model-1-of-96.safetensors",
            "../model-00001-of-000096.safetensors",
        ):
            document = self.valid_inventory()
            document["tensor_00001"]["shard"] = shard
            with self.subTest(shard=shard):
                with self.assertRaisesRegex(
                    inventory_meta.InventoryValidationError,
                    "shard",
                ):
                    inventory_meta.validate_inventory(
                        document, expected_tensor_count=96
                    )

    def test_inventory_rejects_malformed_typed_fields_and_bounds(self):
        mutations = {
            "dtype": lambda entry: entry.update(dtype=7),
            "unknown dtype": lambda entry: entry.update(dtype="F16"),
            "shape": lambda entry: entry.update(shape="1"),
            "boolean dimension": lambda entry: entry.update(shape=[True]),
            "negative dimension": lambda entry: entry.update(shape=[-1]),
            "offset type": lambda entry: entry.update(offsets=[0, "1"]),
            "boolean offset": lambda entry: entry.update(offsets=[0, True]),
            "negative offset": lambda entry: entry.update(offsets=[-1, 0]),
            "reversed offsets": lambda entry: entry.update(offsets=[2, 1]),
            "size mismatch": lambda entry: entry.update(offsets=[0, 2]),
            "hlen type": lambda entry: entry.update(hlen="129"),
            "boolean hlen": lambda entry: entry.update(hlen=True),
            "negative hlen": lambda entry: entry.update(hlen=-1),
        }
        original = self.valid_inventory()
        for label, mutate in mutations.items():
            document = copy.deepcopy(original)
            mutate(document["tensor_00001"])
            with self.subTest(field=label):
                with self.assertRaises(
                    inventory_meta.InventoryValidationError
                ):
                    inventory_meta.validate_inventory(
                        document, expected_tensor_count=96
                    )

    def test_inventory_rejects_offset_gaps_and_missing_shards(self):
        document = self.valid_inventory()
        document["tensor_00001"]["offsets"] = [1, 2]
        with self.assertRaisesRegex(
            inventory_meta.InventoryValidationError, "gap or overlap"
        ):
            inventory_meta.validate_inventory(
                document, expected_tensor_count=96
            )

        document = self.valid_inventory()
        del document["tensor_00096"]
        with self.assertRaisesRegex(
            inventory_meta.InventoryValidationError, "records"
        ):
            inventory_meta.validate_inventory(
                document, expected_tensor_count=96
            )

    def test_publish_uses_unique_fsynced_temp_and_atomic_replace(self):
        _root, meta = self.directories()
        document = self.valid_inventory()
        payload = self.serialized(document)
        destination = meta / inventory_meta.INVENTORY_FILENAME
        real_fsync = os.fsync
        real_replace = os.replace
        with (
            self.pinned_fixture(payload),
            mock.patch.object(setup.os, "fsync", wraps=real_fsync) as fsync,
            mock.patch.object(
                setup.os, "replace", wraps=real_replace
            ) as replace,
        ):
            setup._publish_inventory(str(destination), document)

        self.assertEqual(destination.read_bytes(), payload)
        # The payload fsync is universal. The directory fsync that persists the
        # rename is POSIX-only: _fsync_directory documents itself as acting
        # "where the platform supports it", and Windows cannot open a directory
        # as a descriptor at all.
        self.assertGreaterEqual(fsync.call_count, 1 if os.name == "nt" else 2)
        self.assertEqual(replace.call_count, 1)
        temporary, published = replace.call_args.args
        self.assertNotEqual(temporary, str(destination) + ".part")
        self.assertEqual(published, str(destination))
        self.assertFalse(os.path.exists(temporary))
        self.assertEqual(list(meta.glob("*.part")), [])

    def test_bad_generated_inventory_is_never_published(self):
        _root, meta = self.directories()
        document = self.valid_inventory()
        payload = self.serialized(document)
        destination = meta / inventory_meta.INVENTORY_FILENAME
        with (
            mock.patch.object(
                inventory_meta, "EXPECTED_TENSOR_COUNT", 96
            ),
            mock.patch.object(
                model_source, "PINNED_INVENTORY_SIZE", len(payload)
            ),
            mock.patch.object(
                model_source, "PINNED_INVENTORY_SHA256", "0" * 64
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                setup._publish_inventory(str(destination), document)
        self.assertFalse(destination.exists())
        self.assertEqual(list(meta.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
