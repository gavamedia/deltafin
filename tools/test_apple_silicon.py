#!/usr/bin/env python3
"""Focused tests for capability-based Apple Silicon dispatch."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import apple_silicon as asi  # noqa: E402


def fake_caps(
    *,
    name: str = "provenance-only",
    families: tuple[str, ...] = ("apple7", "metal3"),
    physical: int = 64 << 30,
    recommended: int = 48 << 30,
    max_buffer: int = 32 << 30,
    available: bool = True,
    features: dict[str, bool] | None = None,
) -> asi.CapabilitySnapshot:
    return asi.CapabilitySnapshot(
        system="Darwin",
        machine="arm64",
        physical_memory_bytes=physical,
        macos={"product_version": "test", "build_version": "test"},
        metal={
            "available": available,
            "name": name,
            "supported_families": list(families),
            "features": dict(features or {}),
            "recommended_max_working_set_bytes": recommended,
            "max_buffer_length_bytes": max_buffer,
        },
    )


class CapabilityTests(unittest.TestCase):
    def test_family_gate_never_uses_marketing_name(self):
        misleading_new_name = fake_caps(name="M5 Max", families=("apple7",))
        decision = asi.gate_experiment(
            "auto",
            capabilities=misleading_new_name,
            required_families=("apple10",),
        )
        self.assertFalse(decision.enabled)

        misleading_old_name = fake_caps(name="M1", families=("apple10",))
        decision = asi.gate_experiment(
            "auto",
            capabilities=misleading_old_name,
            required_families=("apple10",),
        )
        self.assertTrue(decision.enabled)

    def test_unknown_capability_is_conservative_but_force_is_explicit(self):
        unknown = fake_caps(available=False, families=())
        automatic = asi.gate_experiment(
            "auto",
            capabilities=unknown,
            require_metal=True,
            required_features=("raytracing",),
        )
        self.assertFalse(automatic.enabled)
        self.assertIn("Metal device", automatic.missing)
        self.assertTrue(
            asi.gate_experiment(
                "force",
                capabilities=unknown,
                require_metal=True,
                required_features=("raytracing",),
            ).enabled
        )

    def test_safe_budget_uses_smallest_confirmed_limit(self):
        caps = fake_caps(physical=128 << 30, recommended=96 << 30)
        budget = caps.safe_unified_budget(
            host_reserve_bytes=20 << 30,
            device_reserve_bytes=8 << 30,
            utilization=0.75,
        )
        self.assertEqual(budget, 66 << 30)

    def test_buffer_and_feature_gates(self):
        caps = fake_caps(
            max_buffer=8 << 30,
            features={"placement_sparse": True, "raytracing": False},
        )
        self.assertTrue(caps.max_buffer_allows(4 << 30))
        self.assertFalse(caps.max_buffer_allows(9 << 30))
        self.assertTrue(
            asi.gate_experiment(
                "on",
                capabilities=caps,
                required_features=("placement_sparse",),
                min_max_buffer_bytes=4 << 30,
            ).enabled
        )
        self.assertFalse(
            asi.gate_experiment(
                "on",
                capabilities=caps,
                required_features=("raytracing",),
            ).enabled
        )

    def test_live_probe_is_safe_and_json_serializable(self):
        caps = asi.refresh()
        encoded = json.dumps(caps.to_dict(), sort_keys=True)
        self.assertIn('"fingerprint"', encoded)
        self.assertEqual(len(caps.fingerprint()), 64)
        self.assertIsInstance(caps.metal.get("available"), bool)
        if sys.platform == "darwin":
            self.assertGreater(caps.physical_memory_bytes or 0, 0)
            self.assertTrue(pathlib.Path(asi.NATIVE_SOURCE).exists())
        if os.name == "nt":
            # Windows answers through GlobalMemoryStatusEx.  Reporting nothing
            # here makes the layer cache pin nothing at all.
            self.assertGreater(caps.physical_memory_bytes or 0, 0)


if __name__ == "__main__":
    unittest.main()
