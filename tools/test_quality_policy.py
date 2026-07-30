#!/usr/bin/env python3
"""Tests for Deltafin's non-negotiable inference-quality boundary."""
from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality_policy  # noqa: E402


class QualityPolicyTests(unittest.TestCase):
    def test_complete_model_top_k_is_the_only_accepted_value(self):
        self.assertEqual(quality_policy.full_moe_top_k(16, None), 16)
        self.assertEqual(quality_policy.full_moe_top_k(16, "16"), 16)
        for value in ("15", "1", "several"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality_policy.full_moe_top_k(16, value)

    def test_approximate_or_reduced_precision_modes_are_rejected(self):
        for value in ("1", "on", "true", "enabled"):
            with self.subTest(approx=value), self.assertRaises(ValueError):
                quality_policy.require_fp32(value, "fp32")
        for value in ("fp16", "float16", "bf16"):
            with self.subTest(dtype=value), self.assertRaises(ValueError):
                quality_policy.require_fp32("0", value)

    def test_explicit_exact_aliases_remain_valid(self):
        for approx in (None, "", "0", "off", "false", "disabled"):
            for dtype in (None, "fp32", "float32"):
                with self.subTest(approx=approx, dtype=dtype):
                    quality_policy.require_fp32(approx, dtype)

    def test_lossy_mixed_spine_is_not_a_supported_runtime_mode(self):
        self.assertEqual(quality_policy.supported_spine("bf16"), "bf16")
        self.assertEqual(quality_policy.supported_spine("INT8"), "int8")
        for value in ("mixed", "int4", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality_policy.supported_spine(value)


if __name__ == "__main__":
    unittest.main()
