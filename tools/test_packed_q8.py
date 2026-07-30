#!/usr/bin/env python3
"""Disk-free tests for backend-neutral packed row-int8 discovery."""
import pathlib
import sys
import unittest

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import packed_q8


def _exact_test_operator(hidden, qweight, scale):
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = qweight.float() * scale[:, None]
    return torch.stack(
        [(flat * row).sum(dim=-1) for row in weight],
        dim=-1,
    )


class PackedQ8CapabilityTests(unittest.TestCase):
    def test_measured_automatic_token_limits_are_backend_specific(self):
        self.assertEqual(
            packed_q8._parse_max_tokens(
                "auto", device_type="mps", torch_threads=8
            ),
            9,
        )
        self.assertEqual(
            packed_q8._parse_max_tokens(
                None, device_type="cpu", torch_threads=1
            ),
            1,
        )
        self.assertEqual(
            packed_q8._parse_max_tokens(
                None, device_type="cpu", torch_threads=4
            ),
            2,
        )
        self.assertEqual(
            packed_q8._parse_max_tokens(
                None, device_type="cuda", torch_threads=8
            ),
            1,
        )

    def test_operator_resolution_accepts_aten_overload(self):
        sentinel = object()

        class Packet:
            default = sentinel

        class Aten:
            _weight_int8pack_mm = Packet()

        class Ops:
            aten = Aten()

        class TorchWithoutTopLevelAlias:
            ops = Ops()

        self.assertIs(
            packed_q8._resolve_operator(TorchWithoutTopLevelAlias),
            sentinel,
        )

    def test_cpu_requires_dispatch_and_exact_real_shape_call(self):
        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            max_tokens=2,
            operator=_exact_test_operator,
            dispatch_query=lambda _schema, _key: True,
        )
        self.assertTrue(backend.available, backend.reason)
        self.assertTrue(backend.dispatch_registered)
        self.assertTrue(backend.canary["analytical_exact"])
        self.assertEqual(backend.canary["shape"], [128, 64])
        self.assertEqual(backend.canary["tested_columns"][-1], 63)
        self.assertEqual(backend.canary["tested_rows"][-1], 127)
        self.assertTrue(backend.supports_tokens(1))
        self.assertTrue(backend.supports_tokens(2))
        self.assertFalse(backend.supports_tokens(3))
        self.assertTrue(backend.fuse_qkv)
        self.assertTrue(backend.fusion_canary["separate_exact"])
        self.assertEqual(
            backend.fusion_canary["split_rows"],
            [32, 64, 96, 128, 64, 32],
        )

    def test_known_missing_cuda_registration_fails_without_calling(self):
        calls = 0

        def forbidden_operator(*_args):
            nonlocal calls
            calls += 1
            raise AssertionError("known-missing dispatch must not be called")

        backend = packed_q8.discover(
            torch,
            torch.device("cuda"),
            torch.float32,
            (128, 64),
            operator=forbidden_operator,
            dispatch_query=lambda _schema, _key: False,
        )
        self.assertFalse(backend.available)
        self.assertEqual(calls, 0)
        self.assertIn("no native CUDA kernel", backend.reason)

    def test_inexact_canary_fails_closed(self):
        def wrong_operator(hidden, qweight, _scale):
            return torch.zeros(
                hidden.shape[0], qweight.shape[0],
                dtype=hidden.dtype, device=hidden.device,
            )

        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            operator=wrong_operator,
            dispatch_query=lambda _schema, _key: True,
        )
        self.assertFalse(backend.available)
        self.assertIn("canary failed", backend.reason)

    def test_non_fp32_is_not_silently_promoted(self):
        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float16,
            (128, 64),
        )
        self.assertFalse(backend.available)
        self.assertIn("requires fp32", backend.reason)

    def test_fusion_and_backend_can_be_disabled_independently(self):
        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            fusion_mode="off",
            operator=_exact_test_operator,
            dispatch_query=lambda _schema, _key: True,
        )
        self.assertTrue(backend.available)
        self.assertFalse(backend.fuse_qkv)
        self.assertIn("disabled", backend.fusion_reason)

        disabled = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            mode="off",
        )
        self.assertFalse(disabled.available)
        self.assertIn("disabled", disabled.reason)

    def test_transposed_output_shape_requires_its_own_canary(self):
        seen_shapes = []

        def recording_operator(hidden, qweight, scale):
            seen_shapes.append(tuple(qweight.shape))
            return _exact_test_operator(hidden, qweight, scale)

        first_stage = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (96, 64),
            fusion_mode="off",
            operator=recording_operator,
            dispatch_query=lambda _schema, _key: True,
        )
        output_stage = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (64, 96),
            fusion_mode="off",
            operator=recording_operator,
            dispatch_query=lambda _schema, _key: True,
        )

        self.assertTrue(first_stage.available, first_stage.reason)
        self.assertTrue(output_stage.available, output_stage.reason)
        self.assertEqual(first_stage.canary["shape"], [96, 64])
        self.assertEqual(output_stage.canary["shape"], [64, 96])
        self.assertEqual(seen_shapes, [(96, 64), (64, 96)])

    def test_inexact_fused_rows_disable_only_fusion(self):
        def shape_sensitive_operator(hidden, qweight, scale):
            output = _exact_test_operator(hidden, qweight, scale)
            # The base real-shape call has 128 rows. Corrupt only the 416-row
            # unequal six-role concatenated call.
            if qweight.shape[0] == 416:
                output = output.clone()
                output[0, 0] += 1
            return output

        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            operator=shape_sensitive_operator,
            dispatch_query=lambda _schema, _key: True,
        )
        self.assertTrue(backend.available)
        self.assertFalse(backend.fuse_qkv)
        self.assertIsNone(backend.fusion_canary)
        self.assertIn("exactness canary failed", backend.fusion_reason)

    def test_real_cpu_private_operator_is_optional(self):
        operator = packed_q8._resolve_operator(torch)
        probe = getattr(
            getattr(torch, "_C", None),
            "_dispatch_has_kernel_for_dispatch_key",
            None,
        )
        registered = None
        if probe is not None:
            try:
                registered = bool(
                    probe(packed_q8.OPERATOR_SCHEMA, "CPU")
                )
            except Exception:
                registered = None
        if operator is None or registered is False:
            self.skipTest("this PyTorch build has no private CPU packed-q8 op")
        backend = packed_q8.discover(
            torch,
            torch.device("cpu"),
            torch.float32,
            (128, 64),
            operator=operator,
            dispatch_query=(
                (lambda _schema, _key: registered)
                if registered is not None else None
            ),
        )
        if not backend.available:
            self.skipTest(
                "private CPU operator is present but failed its canary: "
                + str(backend.reason)
            )
        self.assertTrue(backend.canary["analytical_exact"])


if __name__ == "__main__":
    unittest.main()
