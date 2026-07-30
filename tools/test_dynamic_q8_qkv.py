#!/usr/bin/env python3
"""Disk-free tests for the capability-gated dynamic packed-q8 KDA projections."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(__file__))
import packed_q8  # noqa: E402
import spine_fast  # noqa: E402


class _Attention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return (
            self.q_proj(hidden),
            self.k_proj(hidden),
            self.v_proj(hidden),
        )


class _Layer(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.self_attn = _Attention(width)


class _BundledAttention(nn.Module):
    """Tiny KimiDeltaAttention call-order model with unequal output rows."""

    DEFAULT_ROWS = {
        "q": 64,
        "k": 128,
        "v": 192,
        "g": 256,
        "f_a": 64,
        "b": 32,
    }

    def __init__(
        self,
        width=64,
        *,
        rows=None,
        low_rank_gate=False,
        repeat_role=None,
        mutate_after_q=False,
        output_projection=False,
    ):
        super().__init__()
        self.rows = dict(self.DEFAULT_ROWS if rows is None else rows)
        self.q_proj = nn.Linear(width, self.rows["q"], bias=False)
        self.k_proj = nn.Linear(width, self.rows["k"], bias=False)
        self.v_proj = nn.Linear(width, self.rows["v"], bias=False)
        self.f_a_proj = nn.Linear(width, self.rows["f_a"], bias=False)
        self.b_proj = nn.Linear(width, self.rows["b"], bias=False)
        self.low_rank_gate = low_rank_gate
        if low_rank_gate:
            self.g_a_proj = nn.Linear(width, self.rows["g"], bias=False)
        else:
            self.g_proj = nn.Linear(width, self.rows["g"], bias=False)
        self.repeat_role = repeat_role
        self.mutate_after_q = mutate_after_q
        self.output_projection = output_projection
        if output_projection:
            self.o_proj = nn.Linear(
                self.rows["g"], width, bias=False
            )

    @property
    def gate_role(self):
        return "g_a" if self.low_rank_gate else "g"

    def _projection(self, role):
        return getattr(self, f"{role}_proj")

    def forward(self, hidden_states):
        # Match KimiDeltaAttention's relevant projection order. The intervening
        # operations in the real module do not mutate hidden_states.
        outputs = [self.q_proj(hidden_states)]
        if self.mutate_after_q:
            hidden_states.add_(1)
        outputs.extend([
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
            self.f_a_proj(hidden_states),
        ])
        if self.repeat_role == "f_a":
            outputs.append(self.f_a_proj(hidden_states))
        outputs.extend([
            self.b_proj(hidden_states),
            self._projection(self.gate_role)(hidden_states),
        ])
        if self.output_projection:
            # Model the real tail's different input width and call position.
            outputs.append(self.o_proj(outputs[-1]))
        return tuple(outputs)

    @property
    def output_roles(self):
        roles = ["q", "k", "v", "f_a"]
        if self.repeat_role == "f_a":
            roles.append("f_a")
        roles.extend(["b", self.gate_role])
        if self.output_projection:
            roles.append("o")
        return tuple(roles)


class _BundledLayer(nn.Module):
    def __init__(self, width=64, **kwargs):
        super().__init__()
        self.self_attn = _BundledAttention(width, **kwargs)


class _Backend:
    def __init__(
        self,
        matmul=None,
        *,
        max_tokens=9,
        fuse_qkv=False,
    ):
        self._matmul = matmul or _reference_mm
        self.max_tokens = max_tokens
        self.fuse_qkv = fuse_qkv
        self.fusion_reason = None
        self.calls = 0

    def matmul(self, hidden, qweight, scale):
        self.calls += 1
        return self._matmul(hidden, qweight, scale)

    def supports_tokens(self, token_count):
        return token_count <= self.max_tokens

    def status(self):
        return {
            "max_tokens": self.max_tokens,
            "fuse_qkv": self.fuse_qkv,
        }


def _reference_mm(hidden, qweight, scale):
    # Match nn.Linear and the production packed path: a rank>2 activation is
    # flattened to one row matrix before GEMM, then its leading dimensions are
    # restored.  Calling ``matmul`` directly on a rank-3 tensor may select a
    # batched-GEMM reduction order and differ by a few fp32 ulps.
    flat = hidden.reshape(-1, hidden.shape[-1])
    output = flat @ (qweight.float() * scale[:, None]).T
    return output.view(*hidden.shape[:-1], qweight.shape[0])


def _row_independent_reference_mm(hidden, qweight, scale):
    """Deterministic test backend whose result cannot depend on output height."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = qweight.float() * scale[:, None]
    output = torch.stack(
        [(flat * row).sum(dim=-1) for row in weight],
        dim=-1,
    )
    return output.view(*hidden.shape[:-1], qweight.shape[0])


def _source_for(index, width=8):
    qweight = (
        torch.arange(width * width, dtype=torch.int16).reshape(width, width)
        .remainder(13).sub(6).to(torch.int8)
        .add(index)
    )
    scale = torch.linspace(
        0.125 * index, 0.25 * index, width, dtype=torch.float16)
    return qweight, scale


def _source_for_shape(index, rows, cols):
    qweight = (
        torch.arange(rows * cols, dtype=torch.int32)
        .reshape(rows, cols)
        .remainder(23)
        .sub(11)
        .add(index)
        .to(torch.int8)
    )
    scale = torch.linspace(
        0.03125 * index,
        0.0625 * index,
        rows,
        dtype=torch.float16,
    )
    return qweight, scale


def _set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def _apply_sources(
    layer,
    packed_sources,
    dense_sources,
    *,
    device=torch.device("cpu"),
):
    """Apply a tiny synthetic spine pack through the production integration."""
    qbuf = bytearray()
    scbuf = bytearray()
    items = []
    for role, (qweight, scale) in packed_sources.items():
        name = f"self_attn.{role}_proj.weight"
        qoff = len(qbuf)
        scoff = len(scbuf) // 2
        qbuf.extend(
            qweight.contiguous().reshape(-1).view(torch.uint8).tolist())
        scbuf.extend(scale.contiguous().view(torch.uint8).tolist())
        rows, cols = qweight.shape
        items.append((
            name,
            "layer." + name,
            (rows, cols),
            qoff,
            rows * cols,
            scoff,
            rows,
        ))

    dense_by_full = {}
    other = []
    for role, dense in dense_sources.items():
        name = f"self_attn.{role}_proj.weight"
        full = "layer." + name
        dense_by_full[full] = dense
        other.append((name, full, None))

    pack = {
        "lay": {
            "items": items,
            "qtotal": len(qbuf),
            "sctotal": len(scbuf) // 2,
            "oplan": [],
            "ototal": 0,
            "other": other,
        },
        "q": qbuf,
        "sc": scbuf,
        "other": bytearray(1),
    }
    spine_fast.apply_pack(
        layer,
        "layer.",
        pack,
        device,
        torch.float32,
        {},
        {},
        _set_param,
        lambda full: dense_by_full[full],
    )
    return pack


class DynamicQ8QKVTests(unittest.TestCase):
    def _controller(
        self,
        matmul=_reference_mm,
        meta=False,
        width=8,
        backend=None,
        storage_mode="arena",
        stage_sync_mode="event",
        device=torch.device("cpu"),
    ):
        if meta:
            with torch.device("meta"):
                layer = _Layer(width)
        else:
            layer = _Layer(width)
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer,
            device,
            state,
            backend if backend is not None else matmul,
            storage_mode=storage_mode,
            stage_sync_mode=stage_sync_mode,
        )
        return layer, state, controller

    @staticmethod
    def _load(controller, width=8):
        source = {}
        for index, role in enumerate(("q", "k", "v"), start=1):
            qweight, scale = _source_for(index, width)
            source[role] = (qweight, scale)
            consumed = controller.load(
                f"self_attn.{role}_proj.weight", qweight, scale)
            assert consumed
        controller.finish_load()
        return source

    @staticmethod
    def _load_bundle(controller):
        source = {}
        controller.begin_load()
        for index, role in enumerate(controller.roles, start=1):
            projection = controller.projections[role]
            rows, cols = projection.weight.shape
            qweight, scale = _source_for_shape(index, rows, cols)
            source[role] = (qweight, scale)
            assert controller.load(
                controller.role_names[role], qweight, scale
            )
        controller.finish_load()
        return source

    def _bundle_controller(
        self,
        *,
        backend=None,
        meta=False,
        low_rank_gate=False,
        repeat_role=None,
        mutate_after_q=False,
        output_projection=False,
        o_backend=None,
        storage_mode="arena",
        stage_sync_mode="event",
        device=torch.device("cpu"),
    ):
        if meta:
            with torch.device("meta"):
                layer = _BundledLayer(
                    low_rank_gate=low_rank_gate,
                    repeat_role=repeat_role,
                    mutate_after_q=mutate_after_q,
                    output_projection=output_projection,
                )
        else:
            layer = _BundledLayer(
                low_rank_gate=low_rank_gate,
                repeat_role=repeat_role,
                mutate_after_q=mutate_after_q,
                output_projection=output_projection,
            )
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer,
            device,
            state,
            backend if backend is not None else _reference_mm,
            storage_mode=storage_mode,
            stage_sync_mode=stage_sync_mode,
            o_backend=o_backend,
        )
        return layer, state, controller

    def test_owned_arenas_survive_recycled_source_mutation(self):
        layer, state, controller = self._controller()
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(2, 3, 8)

        expected = {
            role: _reference_mm(hidden, qweight, scale.float())
            for role, (qweight, scale) in source.items()
        }
        for qweight, scale in source.values():
            qweight.zero_()
            scale.zero_()

        self.assertTrue(state.enabled)
        for role in ("q", "k", "v"):
            got = getattr(layer.self_attn, f"{role}_proj")(hidden)
            torch.testing.assert_close(got, expected[role], rtol=0, atol=0)
        self.assertEqual(controller.packed_project_calls, 3)

    def test_equal_shape_qkv_uses_one_fused_operator_call(self):
        # width=64 makes every weight and scale slot naturally 256-byte
        # aligned, matching the real 12288x7168 KDA layout.
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._controller(
            width=64, backend=backend
        )
        controller.begin_load()
        source = self._load(controller, width=64)
        hidden = torch.randn(2, 3, 64)

        got = dict(zip(("q", "k", "v"), layer.self_attn(hidden)))
        self.assertTrue(state.enabled)
        self.assertTrue(controller.fuse_qkv)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.packed_operator_calls, 1)
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(controller.separate_operator_calls, 0)
        self.assertEqual(controller.packed_project_calls, 3)
        for role, (qweight, scale) in source.items():
            torch.testing.assert_close(
                got[role],
                _row_independent_reference_mm(
                    hidden, qweight, scale.float()
                ),
                rtol=0,
                atol=0,
            )

    def test_unequal_kda_bundle_uses_one_operator_call(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(2, 3, 64)

        with torch.inference_mode():
            # Production Kimi passes hidden_states by keyword in some call
            # paths; the attention-scope wrapper must preserve both conventions.
            outputs = layer.self_attn(hidden_states=hidden)

        self.assertEqual(
            controller.roles,
            ("q", "k", "v", "g", "f_a", "b"),
        )
        self.assertTrue(controller.fusion_eligible)
        self.assertTrue(controller.fuse_qkv)
        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.packed_operator_calls, 1)
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(controller.packed_project_calls, 6)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def test_output_projection_uses_its_independent_backend(self):
        bundle_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        output_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            # Even a fusion-capable backend must not fuse the different-input
            # output projection into the first-stage bundle.
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=bundle_backend,
            o_backend=output_backend,
            output_projection=True,
        )
        source = {}
        for index, role in enumerate(controller.roles, start=1):
            rows, cols = controller.projections[role].weight.shape
            source[role] = _source_for_shape(index, rows, cols)
        _apply_sources(layer, source, {})
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden_states=hidden)

        self.assertEqual(
            controller.bundle_roles,
            ("q", "k", "v", "g", "f_a", "b"),
        )
        self.assertEqual(controller.independent_roles, ("o",))
        self.assertEqual(controller.roles[-1], "o")
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(bundle_backend.calls, 1)
        self.assertEqual(output_backend.calls, 1)
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(controller.separate_operator_calls, 1)
        self.assertEqual(controller.packed_operator_calls, 2)
        self.assertEqual(controller.packed_project_calls, 7)

        by_role = dict(zip(layer.self_attn.output_roles, outputs))
        for role in controller.bundle_roles:
            torch.testing.assert_close(
                by_role[role],
                _row_independent_reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
        expected_o = _row_independent_reference_mm(
            by_role["g"], source["o"][0], source["o"][1].float()
        )
        torch.testing.assert_close(
            by_role["o"], expected_o, rtol=0, atol=0
        )

    def test_output_projection_stays_dense_without_qualified_backend(self):
        bundle_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=bundle_backend,
            o_backend=None,
            output_projection=True,
        )
        self._load_bundle(controller)
        self.assertNotIn("o", controller.roles)
        self.assertNotIn(
            "self_attn.o_proj.weight", controller.names_to_roles
        )
        self.assertNotIn("forward", layer.self_attn.o_proj.__dict__)

        outputs = layer.self_attn(torch.randn(1, 1, 64))
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(bundle_backend.calls, 1)
        self.assertEqual(len(outputs), 7)

    def test_output_crossover_densifies_before_first_stage(self):
        bundle_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        output_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=1,
        )
        layer, state, controller = self._bundle_controller(
            backend=bundle_backend,
            o_backend=output_backend,
            output_projection=True,
            meta=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(bundle_backend.calls, 0)
        self.assertEqual(output_backend.calls, 0)
        self.assertEqual(controller.dense_crossover_materializations, 1)
        self.assertEqual(
            controller.dense_crossover_project_calls,
            len(controller.roles),
        )
        by_role = dict(zip(layer.self_attn.output_roles, outputs))
        for role in controller.bundle_roles:
            torch.testing.assert_close(
                by_role[role],
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            by_role["o"],
            _reference_mm(
                by_role["g"], source["o"][0], source["o"][1].float()
            ),
            rtol=0,
            atol=0,
        )
        for projection in controller.projections.values():
            self.assertEqual(projection.weight.device.type, "meta")

    def test_output_backend_failure_materializes_dense_group(self):
        def reject_output(_hidden, _qweight, _scale):
            raise ValueError("synthetic output-shape rejection")

        bundle_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        output_backend = _Backend(reject_output, max_tokens=9)
        layer, state, controller = self._bundle_controller(
            backend=bundle_backend,
            o_backend=output_backend,
            output_projection=True,
            meta=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 1, 64)
        outputs = layer.self_attn(hidden)
        by_role = dict(zip(layer.self_attn.output_roles, outputs))

        self.assertFalse(state.enabled)
        self.assertIn("output-shape rejection", state.reason)
        self.assertEqual(bundle_backend.calls, 1)
        self.assertEqual(output_backend.calls, 1)
        self.assertEqual(controller.runtime_failures, 1)
        torch.testing.assert_close(
            by_role["o"],
            _reference_mm(
                by_role["g"], source["o"][0], source["o"][1].float()
            ),
            rtol=0,
            atol=0,
        )
        for role, projection in controller.projections.items():
            expected_weight = (
                source[role][0].float()
                * source[role][1].float()[:, None]
            )
            torch.testing.assert_close(
                projection.weight, expected_weight, rtol=0, atol=0
            )

    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "MPS packed-q8 integration requires a real Apple GPU",
    )
    def test_real_mps_unequal_bundle_matches_separate_native_calls(self):
        """Exercise unequal fused rows through the production private op."""
        device = torch.device("mps")
        backend = packed_q8.discover(
            torch,
            device,
            torch.float32,
            (64, 64),
            max_tokens=2,
        )
        if not backend.available:
            self.skipTest(backend.reason or "MPS packed-q8 unavailable")
        self.assertTrue(backend.fuse_qkv, backend.fusion_reason)

        with torch.device("meta"):
            layer = _BundledLayer()
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer,
            device,
            state,
            backend,
        )
        source = self._load_bundle(controller)
        hidden_cpu = (
            torch.arange(2 * 64, dtype=torch.float32)
            .reshape(1, 2, 64)
            .remainder(29)
            .sub(14)
            .div(31)
        )
        hidden = hidden_cpu.to(device)

        with torch.inference_mode():
            fused = layer.self_attn(hidden_states=hidden)
            separate = {}
            flat = hidden.reshape(-1, hidden.shape[-1])
            for role in controller.roles:
                qweight, scale = controller._views(role)
                separate[role] = (
                    backend.matmul(flat, qweight, scale).view(
                        *hidden.shape[:-1], qweight.shape[0]
                    )
                )
            torch.mps.synchronize()

        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(controller.packed_operator_calls, 1)
        self.assertEqual(controller.packed_project_calls, 6)
        for role, got in zip(layer.self_attn.output_roles, fused):
            self.assertTrue(
                torch.equal(got.cpu(), separate[role].cpu()),
                f"fused rows changed native {role} output",
            )
            qweight, scale = source[role]
            torch.testing.assert_close(
                got.cpu(),
                _reference_mm(hidden_cpu, qweight, scale.float()),
                rtol=1e-5,
                atol=1e-3,
            )

    def test_bundle_is_consumed_through_production_apply_pack(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        source = {}
        for index, role in enumerate(controller.roles, start=1):
            rows, cols = controller.projections[role].weight.shape
            source[role] = _source_for_shape(index, rows, cols)

        _apply_sources(layer, source, {})
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.loaded, set(controller.roles))
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def _stage_bundle(
        self,
        *,
        backend=None,
        o_backend=None,
        output_projection=False,
        offset=0,
        meta=True,
        stage_sync_mode="event",
    ):
        layer, state, controller = self._bundle_controller(
            backend=backend,
            o_backend=o_backend,
            output_projection=output_projection,
            meta=meta,
            storage_mode="stage",
            stage_sync_mode=stage_sync_mode,
        )
        source = {}
        for index, role in enumerate(controller.roles, start=1 + offset):
            rows, cols = controller.projections[role].weight.shape
            source[role] = _source_for_shape(index, rows, cols)
        return layer, state, controller, source

    def test_stage_bundle_uses_upload_weights_without_persistent_copy(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller, source = self._stage_bundle(
            backend=backend
        )
        hidden = torch.randn(1, 2, 64)
        expected = {
            role: _row_independent_reference_mm(
                hidden, qweight, scale.float()
            )
            for role, (qweight, scale) in source.items()
        }

        _apply_sources(layer, source, {})
        self.assertNotIn(controller._QBUF, layer.self_attn._buffers)
        self.assertEqual(controller.persistent_weight_bytes, 0)
        self.assertEqual(controller.nbytes, controller.persistent_scale_bytes)
        self.assertIsNotNone(controller._stage_q_arena)
        # Poisoning the caller's source tensors after apply must not affect the
        # device upload generation or its copied fp32 scales.
        for qweight, scale in source.values():
            qweight.fill_(97)
            scale.zero_()

        outputs = layer.self_attn(hidden_states=hidden)
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.stage_bind_count, 1)
        self.assertEqual(controller.stage_weight_copy_bytes, 0)
        self.assertGreater(controller.stage_scale_copy_bytes, 0)
        self.assertEqual(controller.stage_full_shape_probes, 1)
        self.assertEqual(controller.stage_full_shape_passes, 1)
        self.assertEqual(controller.stage_release_count, 1)
        self.assertIsNone(controller._stage_q_arena)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output, expected[role], rtol=0, atol=0
            )

    def test_stage_generations_replace_all_six_roles_atomically(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller, first = self._stage_bundle(
            backend=backend
        )
        hidden = torch.randn(1, 1, 64)
        _apply_sources(layer, first, {})
        first_generation = controller.last_stage_generation
        first_ptr = controller._stage_q_arena.data_ptr()
        first_outputs = layer.self_attn(hidden)

        second = {}
        for index, role in enumerate(controller.roles, start=31):
            rows, cols = controller.projections[role].weight.shape
            second[role] = _source_for_shape(index, rows, cols)
        _apply_sources(layer, second, {})
        second_generation = controller.last_stage_generation
        second_ptr = controller._stage_q_arena.data_ptr()
        second_outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled, state.reason)
        self.assertGreater(second_generation, first_generation)
        self.assertEqual(second_ptr, first_ptr)
        self.assertEqual(controller.stage_bind_count, 2)
        self.assertEqual(controller.stage_release_count, 2)
        for source, outputs in (
            (first, first_outputs),
            (second, second_outputs),
        ):
            for role, output in zip(
                layer.self_attn.output_roles, outputs
            ):
                torch.testing.assert_close(
                    output,
                    _row_independent_reference_mm(
                        hidden,
                        source[role][0],
                        source[role][1].float(),
                    ),
                    rtol=0,
                    atol=0,
                )

    def test_stage_rejects_a_stale_generation_before_operator_use(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller, source = self._stage_bundle(
            backend=backend
        )
        _apply_sources(layer, source, {})
        live = controller._stage_q_arena
        overwritten, _lease = spine_fast._stage_weight_buf(
            live.numel(), torch.device("cpu")
        )
        overwritten.fill_(113)

        with self.assertRaisesRegex(RuntimeError, "lease is stale"):
            layer.self_attn(torch.randn(1, 1, 64))
        self.assertFalse(state.enabled)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(controller.stage_stale_rejections, 1)
        self.assertEqual(controller.stage_release_count, 1)

    def test_stage_crossover_releases_dense_and_rebinds_for_decode(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=1,
            fuse_qkv=True,
        )
        layer, state, controller, source = self._stage_bundle(
            backend=backend
        )
        prompt_hidden = torch.randn(1, 2, 64)
        _apply_sources(layer, source, {})
        prompt_outputs = layer.self_attn(prompt_hidden)

        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(controller.dense_crossover_materializations, 1)
        self.assertEqual(controller.dense_crossover_releases, 1)
        for projection in controller.projections.values():
            self.assertEqual(projection.weight.device.type, "meta")
        for role, output in zip(
            layer.self_attn.output_roles, prompt_outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    prompt_hidden,
                    source[role][0],
                    source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )

        next_source = {}
        for index, role in enumerate(controller.roles, start=41):
            rows, cols = controller.projections[role].weight.shape
            next_source[role] = _source_for_shape(index, rows, cols)
        _apply_sources(layer, next_source, {})
        decode_hidden = torch.randn(1, 1, 64)
        decode_outputs = layer.self_attn(decode_hidden)
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.stage_bind_count, 2)
        for role, output in zip(
            layer.self_attn.output_roles, decode_outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    decode_hidden,
                    next_source[role][0],
                    next_source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )

    def test_stage_operator_failure_materializes_every_role_before_disable(self):
        def failing_mm(_hidden, _qweight, _scale):
            raise RuntimeError("synthetic staged bundle rejection")

        backend = _Backend(failing_mm, max_tokens=9, fuse_qkv=True)
        layer, state, controller, source = self._stage_bundle(
            backend=backend
        )
        hidden = torch.randn(1, 1, 64)
        _apply_sources(layer, source, {})
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertEqual(controller.runtime_failures, 1)
        self.assertEqual(controller.stage_full_shape_probes, 1)
        self.assertEqual(controller.stage_full_shape_passes, 0)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                controller.projections[role].weight.device.type,
                "cpu",
            )

    def test_stage_layout_failure_densifies_complete_bundle(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller, source = self._stage_bundle(
            backend=backend
        )
        wrong_order = {
            role: source[role]
            for role in ("q", "k", "v", "f_a", "b", "g")
        }
        _apply_sources(layer, wrong_order, {})
        hidden = torch.randn(1, 1, 64)
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertEqual(controller.stage_bind_count, 0)
        self.assertEqual(controller.stage_bind_failures, 1)
        self.assertEqual(backend.calls, 0)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def test_stage_plan_layout_places_qualified_roles_in_direct_view_order(self):
        _layer, _state, controller, _source = self._stage_bundle(
            o_backend=_Backend(max_tokens=9),
            output_projection=True,
        )
        prefix = "stage-layout-test."
        with tempfile.TemporaryDirectory() as int8_dir, (
            tempfile.TemporaryDirectory()
        ) as resident_dir:
            inventory = {}
            for role in controller.roles:
                name = controller.role_names[role]
                full = prefix + name
                shape = tuple(controller.projections[role].weight.shape)
                inventory[full] = {"shape": shape}
                open(
                    os.path.join(int8_dir, full + ".i8"), "wb"
                ).close()
            try:
                layout = spine_fast.plan_layout(
                    controller.module,
                    prefix,
                    int8_dir,
                    resident_dir,
                    inventory,
                    "int8",
                )
            finally:
                spine_fast._layouts.pop(prefix, None)

        self.assertEqual(
            [item[0] for item in layout["items"]],
            [controller.role_names[role] for role in controller.roles],
        )
        descriptor = layout["packed_qkv_stage"]
        prior_end = None
        for role in controller.roles:
            qoff, qbytes, _scoff, _rows, _cols = (
                descriptor["records"][role]
            )
            if prior_end is not None:
                self.assertEqual(qoff, prior_end)
            prior_end = qoff + qbytes

    def test_stage_output_projection_consumes_before_attention_fence(self):
        controller = None
        saw_live_stage = []

        def output_mm(hidden, qweight, scale):
            saw_live_stage.append(
                controller is not None
                and controller._stage_q_arena is not None
                and controller._stage_lease is not None
            )
            return _row_independent_reference_mm(
                hidden, qweight, scale
            )

        bundle_backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        output_backend = _Backend(output_mm, max_tokens=9)
        layer, state, controller, source = self._stage_bundle(
            backend=bundle_backend,
            o_backend=output_backend,
            output_projection=True,
        )
        _apply_sources(layer, source, {})
        outputs = layer.self_attn(torch.randn(1, 1, 64))

        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(controller.roles[-1], "o")
        self.assertEqual(bundle_backend.calls, 1)
        self.assertEqual(output_backend.calls, 1)
        self.assertEqual(saw_live_stage, [True])
        self.assertEqual(controller.stage_release_count, 1)
        self.assertIsNone(controller._stage_q_arena)
        self.assertEqual(len(outputs), len(controller.roles))

    def test_stage_key_resolves_implicit_cuda_device_without_gpu_alias(self):
        with mock.patch.object(
            spine_fast.torch.cuda, "current_device", return_value=3
        ) as current_device:
            implicit = spine_fast._stage_key(
                torch.device("cuda"), torch.int8
            )
        self.assertEqual(implicit, ("cuda", 3, torch.int8))
        current_device.assert_called_once_with()

        with mock.patch.object(
            spine_fast.torch.cuda,
            "current_device",
            side_effect=AssertionError("explicit index must stay explicit"),
        ):
            explicit = spine_fast._stage_key(
                torch.device("cuda:2"), torch.int8
            )
        self.assertEqual(explicit, ("cuda", 2, torch.int8))

    def test_mps_fifo_capability_requires_public_default_stream(self):
        default_stream = mock.Mock(stream_id=0, device_index=0)
        side_stream = mock.Mock(stream_id=7, device_index=0)
        with mock.patch.object(
            spine_fast.torch.accelerator,
            "current_stream",
            return_value=default_stream,
        ):
            capable, reason = spine_fast.stage_storage_capability(
                torch.device("mps"), "mps_fifo"
            )
        self.assertTrue(capable, reason)

        with mock.patch.object(
            spine_fast.torch.accelerator,
            "current_stream",
            return_value=side_stream,
        ):
            capable, reason = spine_fast.stage_storage_capability(
                torch.device("mps"), "mps_fifo"
            )
        self.assertFalse(capable)
        self.assertIn("default stream", reason)

    def test_mps_fifo_marker_omits_redundant_event(self):
        _layer, _state, controller = self._controller()
        key = ("mps", 4101, torch.int8)
        signature = ("mps", 0, 0)
        lease = spine_fast._StageLease(
            key, 1, 123, "mps_fifo", signature
        )
        spine_fast._stage_fences.pop(key, None)
        try:
            with mock.patch.object(
                spine_fast,
                "_mps_fifo_stream_signature",
                return_value=signature,
            ), mock.patch.object(
                spine_fast.torch.mps,
                "Event",
                side_effect=AssertionError("event must not be constructed"),
            ):
                spine_fast._record_stage_fence(
                    lease, torch.device("mps"), controller
                )
                self.assertIsInstance(
                    spine_fast._stage_fences[key],
                    spine_fast._FifoStageFence,
                )
                spine_fast._wait_stage_fence(
                    key, torch.device("mps")
                )
        finally:
            spine_fast._stage_fences.pop(key, None)

        self.assertEqual(controller.stage_fifo_records, 1)
        self.assertEqual(controller.stage_fifo_reuses, 1)
        self.assertEqual(controller.stage_fence_records, 0)
        self.assertEqual(controller.stage_fence_waits, 0)
        self.assertEqual(controller.stage_fence_sync_fallbacks, 0)

    def test_mps_fifo_stream_mismatch_synchronizes_before_reuse(self):
        _layer, _state, controller = self._controller()
        key = ("mps", 4102, torch.int8)
        signature = ("mps", 0, 0)
        spine_fast._stage_fences[key] = spine_fast._FifoStageFence(
            signature, controller
        )
        try:
            with mock.patch.object(
                spine_fast,
                "_mps_fifo_stream_signature",
                side_effect=RuntimeError("synthetic side stream"),
            ), mock.patch.object(
                spine_fast.torch.mps, "synchronize"
            ) as synchronize:
                spine_fast._wait_stage_fence(
                    key, torch.device("mps")
                )
            synchronize.assert_called_once_with()
        finally:
            spine_fast._stage_fences.pop(key, None)
        self.assertEqual(controller.stage_fifo_reuses, 0)
        self.assertEqual(controller.stage_fence_sync_fallbacks, 1)
        self.assertEqual(controller.stage_fence_failures, 0)

    def test_mps_fifo_release_introspection_failure_synchronizes(self):
        _layer, _state, controller = self._controller()
        key = ("mps", 4104, torch.int8)
        signature = ("mps", 0, 0)
        lease = spine_fast._StageLease(
            key, 1, 123, "mps_fifo", signature
        )
        spine_fast._stage_fences.pop(key, None)
        try:
            with mock.patch.object(
                spine_fast,
                "_mps_fifo_stream_signature",
                side_effect=OSError("synthetic stream introspection failure"),
            ), mock.patch.object(
                spine_fast.torch.mps, "synchronize"
            ) as synchronize:
                spine_fast._record_stage_fence(
                    lease, torch.device("mps"), controller
                )
            synchronize.assert_called_once_with()
            self.assertNotIn(key, spine_fast._stage_fences)
        finally:
            spine_fast._stage_fences.pop(key, None)
        self.assertEqual(controller.stage_fifo_records, 0)
        self.assertEqual(controller.stage_fence_sync_fallbacks, 1)
        self.assertEqual(controller.stage_fence_failures, 0)

    def test_mps_fifo_failed_fallback_permanently_poisons_stage(self):
        _layer, state, controller = self._controller()
        key = ("mps", 4103, torch.int8)
        signature = ("mps", 0, 0)
        spine_fast._stage_fences[key] = spine_fast._FifoStageFence(
            signature, controller
        )
        try:
            with mock.patch.object(
                spine_fast,
                "_mps_fifo_stream_signature",
                side_effect=RuntimeError("synthetic side stream"),
            ), mock.patch.object(
                spine_fast.torch.mps,
                "synchronize",
                side_effect=RuntimeError("synthetic sync failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cannot order MPS FIFO"
                ):
                    spine_fast._wait_stage_fence(
                        key, torch.device("mps")
                    )
            self.assertIsInstance(
                spine_fast._stage_fences[key],
                spine_fast._FailedStageFence,
            )
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                spine_fast._wait_stage_fence(
                    key, torch.device("mps")
                )
        finally:
            spine_fast._stage_fences.pop(key, None)
        self.assertFalse(state.enabled)
        self.assertEqual(controller.stage_fence_failures, 1)

    def test_mps_fifo_request_retains_cuda_cross_stream_events(self):
        _layer, _state, controller = self._controller()
        device = torch.device("cuda:2")
        key = ("cuda", 2, torch.int8)
        stream = mock.Mock()
        event = mock.Mock()
        lease = spine_fast._StageLease(
            key,
            1,
            123,
            spine_fast._effective_stage_sync_mode(device, "mps_fifo"),
            None,
        )
        spine_fast._stage_fences.pop(key, None)
        try:
            with mock.patch.object(
                spine_fast.torch.cuda, "Event", return_value=event
            ), mock.patch.object(
                spine_fast.torch.cuda,
                "current_stream",
                return_value=stream,
            ):
                spine_fast._record_stage_fence(
                    lease, device, controller
                )
                spine_fast._wait_stage_fence(key, device)
            event.record.assert_called_once_with(stream)
            stream.wait_event.assert_called_once_with(event)
        finally:
            spine_fast._stage_fences.pop(key, None)
        self.assertEqual(lease.sync_contract, "event")
        self.assertEqual(controller.stage_fence_records, 1)
        self.assertEqual(controller.stage_fence_waits, 1)
        self.assertEqual(controller.stage_fifo_records, 0)
        self.assertEqual(controller.stage_fifo_reuses, 0)

    @unittest.skipUnless(
        torch.backends.mps.is_available()
        and os.environ.get("K3_RUN_MPS_STAGE_TESTS") == "1",
        "set K3_RUN_MPS_STAGE_TESTS=1 for the bounded real-MPS stage gate",
    )
    def test_real_mps_stage_generation_fence_and_reuse(self):
        device = torch.device("mps")
        backend = packed_q8.discover(
            torch,
            device,
            torch.float32,
            (64, 64),
            max_tokens=2,
        )
        if not backend.available:
            self.skipTest(backend.reason or "MPS packed-q8 unavailable")
        with torch.device("meta"):
            layer = _BundledLayer()
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer,
            device,
            state,
            backend,
            storage_mode="stage",
        )
        hidden_cpu = torch.randn(1, 1, 64)
        hidden = hidden_cpu.to(device)
        generations = []
        for generation_offset in (0, 47):
            source = {}
            for index, role in enumerate(
                controller.roles, start=1 + generation_offset
            ):
                rows, cols = controller.projections[role].weight.shape
                source[role] = _source_for_shape(index, rows, cols)
            _apply_sources(layer, source, {}, device=device)
            with torch.inference_mode():
                outputs = layer.self_attn(hidden_states=hidden)
            # Do not synchronize or read either output yet. The second
            # apply_pack must wait on the event recorded by the first attention
            # before it overwrites the same staging allocation.
            generations.append((source, outputs))
        torch.mps.synchronize()
        for source, outputs in generations:
            for role, output in zip(
                layer.self_attn.output_roles, outputs
            ):
                torch.testing.assert_close(
                    output.cpu(),
                    _reference_mm(
                        hidden_cpu,
                        source[role][0],
                        source[role][1].float(),
                    ),
                    rtol=1e-5,
                    atol=1e-3,
                )
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(controller.persistent_weight_bytes, 0)
        self.assertEqual(controller.stage_bind_count, 2)
        self.assertGreaterEqual(controller.stage_fence_records, 2)
        self.assertGreaterEqual(controller.stage_fence_waits, 1)
        self.assertEqual(controller.stage_fence_failures, 0)
        self.assertEqual(controller.stage_stale_rejections, 0)

    @unittest.skipUnless(
        torch.backends.mps.is_available()
        and os.environ.get("K3_RUN_MPS_FIFO_TESTS") == "1",
        "set K3_RUN_MPS_FIFO_TESTS=1 for the bounded real-MPS FIFO gate",
    )
    def test_real_mps_fifo_two_unsynchronized_generations(self):
        device = torch.device("mps")
        backend = packed_q8.discover(
            torch,
            device,
            torch.float32,
            (64, 64),
            max_tokens=2,
        )
        if not backend.available:
            self.skipTest(backend.reason or "MPS packed-q8 unavailable")
        with torch.device("meta"):
            layer = _BundledLayer()
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer,
            device,
            state,
            backend,
            storage_mode="stage",
            stage_sync_mode="mps_fifo",
        )
        hidden_cpu = torch.randn(1, 1, 64)
        hidden = hidden_cpu.to(device)
        generations = []
        for generation_offset in (0, 47):
            source = {}
            for index, role in enumerate(
                controller.roles, start=1 + generation_offset
            ):
                rows, cols = controller.projections[role].weight.shape
                source[role] = _source_for_shape(index, rows, cols)
            _apply_sources(layer, source, {}, device=device)
            with torch.inference_mode():
                outputs = layer.self_attn(hidden_states=hidden)
            # Keep both device outputs live and unread. The next overwrite is
            # safe only if upload and packed reads really share one FIFO stream.
            generations.append((source, outputs))
        torch.mps.synchronize()
        for source, outputs in generations:
            for role, output in zip(
                layer.self_attn.output_roles, outputs
            ):
                torch.testing.assert_close(
                    output.cpu(),
                    _reference_mm(
                        hidden_cpu,
                        source[role][0],
                        source[role][1].float(),
                    ),
                    rtol=1e-5,
                    atol=1e-3,
                )
        self.assertTrue(state.enabled, state.reason)
        self.assertEqual(controller.stage_bind_count, 2)
        self.assertEqual(controller.stage_fifo_records, 2)
        self.assertGreaterEqual(controller.stage_fifo_reuses, 1)
        self.assertEqual(controller.stage_fence_records, 0)
        self.assertEqual(controller.stage_fence_waits, 0)
        self.assertEqual(controller.stage_fence_sync_fallbacks, 0)
        self.assertEqual(controller.stage_fence_failures, 0)
        self.assertEqual(controller.stage_stale_rejections, 0)

    def test_low_rank_gate_bundles_g_a_not_second_stage(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            low_rank_gate=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled)
        self.assertIn("g_a", controller.roles)
        self.assertNotIn("g", controller.roles)
        self.assertTrue(controller.consumes(
            "self_attn.g_a_proj.weight"
        ))
        self.assertFalse(controller.consumes(
            "self_attn.g_b_proj.weight"
        ))
        self.assertEqual(backend.calls, 1)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def test_second_use_of_cached_role_runs_fresh_operator(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            repeat_role="f_a",
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled)
        # One bundled call plus one fresh f_a call; a consumed cache entry is
        # never served twice.
        self.assertEqual(backend.calls, 2)
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(controller.separate_operator_calls, 1)
        self.assertEqual(controller.packed_project_calls, 7)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def test_bundle_cache_is_attention_scoped_under_inference_mode(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend
        )
        source = self._load_bundle(controller)
        with torch.inference_mode():
            hidden = torch.randn(1, 64)
            # Direct projection use is outside the audited attention scope and
            # therefore cannot populate cross-projection cache entries.
            layer.self_attn.q_proj(hidden)
            hidden.add_(1)
            got = layer.self_attn.g_proj(hidden)
        expected = _row_independent_reference_mm(
            hidden, source["g"][0], source["g"][1].float()
        )

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 2)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_in_scope_normal_tensor_mutation_invalidates_bundle_cache(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            mutate_after_q=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 64)
        q_hidden = hidden.clone()
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, len(controller.roles))
        self.assertEqual(controller.fused_operator_calls, 1)
        self.assertEqual(
            controller.separate_operator_calls,
            len(controller.roles) - 1,
        )
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            role_hidden = q_hidden if role == "q" else hidden
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    role_hidden,
                    source[role][0],
                    source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )

    def test_bundle_crossover_is_transient_for_every_role(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=1,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(controller.dense_crossover_materializations, 1)
        self.assertEqual(
            controller.dense_crossover_project_calls,
            len(controller.roles),
        )
        self.assertEqual(controller.dense_crossover_releases, 1)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
        for projection in controller.projections.values():
            self.assertEqual(projection.weight.device.type, "meta")

        # The next decode-sized layer re-enters one-call packed execution.
        source = self._load_bundle(controller)
        decode_hidden = torch.randn(1, 1, 64)
        decode_outputs = layer.self_attn(decode_hidden)
        self.assertEqual(backend.calls, 1)
        for role, output in zip(
            layer.self_attn.output_roles, decode_outputs
        ):
            torch.testing.assert_close(
                output,
                _row_independent_reference_mm(
                    decode_hidden,
                    source[role][0],
                    source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )

    def test_bundle_partial_dense_load_disables_atomically(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        controller.begin_load()
        source = {}
        for index, role in enumerate(controller.roles, start=1):
            projection = controller.projections[role]
            rows, cols = projection.weight.shape
            qweight, scale = _source_for_shape(index, rows, cols)
            source[role] = (qweight, scale)
            if role == "b":
                projection.weight = nn.Parameter(
                    qweight.float() * scale.float()[:, None],
                    requires_grad=False,
                )
                controller.mark_dense(controller.role_names[role])
            else:
                self.assertTrue(controller.load(
                    controller.role_names[role], qweight, scale
                ))
        controller.finish_load()
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertEqual(backend.calls, 0)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            expected_weight = (
                source[role][0].float()
                * source[role][1].float()[:, None]
            )
            torch.testing.assert_close(
                controller.projections[role].weight,
                expected_weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

    def test_bundle_optional_copy_failure_preserves_packed_prefix(self):
        backend = _Backend(
            _row_independent_reference_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        source = {}
        for index, role in enumerate(controller.roles, start=1):
            rows, cols = controller.projections[role].weight.shape
            source[role] = _source_for_shape(index, rows, cols)

        original_views = controller._views

        class RejectingCopy:
            shape = source["f_a"][0].shape

            @staticmethod
            def copy_(_source):
                raise RuntimeError("synthetic optional packed copy failure")

        def rejecting_optional_view(role):
            if role == "f_a":
                return RejectingCopy(), original_views(role)[1]
            return original_views(role)

        controller._views = rejecting_optional_view
        _apply_sources(layer, source, {})
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertIn("optional packed copy failure", state.reason)
        self.assertEqual(backend.calls, 0)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                controller.projections[role].weight.device.type,
                "cpu",
            )

    def test_bundle_fused_failure_materializes_every_role(self):
        def failing_mm(_hidden, _qweight, _scale):
            raise RuntimeError("synthetic bundle rejection")

        backend = _Backend(
            failing_mm,
            max_tokens=9,
            fuse_qkv=True,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 64)
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.runtime_failures, 1)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                controller.projections[role].weight.device.type,
                "cpu",
            )

    def test_bundle_second_separate_call_failure_is_atomic(self):
        calls = 0

        def fail_second(hidden, qweight, scale):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic second-call rejection")
            return _row_independent_reference_mm(
                hidden, qweight, scale
            )

        backend = _Backend(
            fail_second,
            max_tokens=9,
            fuse_qkv=False,
        )
        layer, state, controller = self._bundle_controller(
            backend=backend,
            meta=True,
        )
        source = self._load_bundle(controller)
        hidden = torch.randn(1, 2, 64)
        outputs = layer.self_attn(hidden)

        self.assertFalse(state.enabled)
        self.assertEqual(calls, 2)
        self.assertEqual(controller.runtime_failures, 1)
        self.assertEqual(controller.packed_operator_calls, 0)
        for role, output in zip(
            layer.self_attn.output_roles, outputs
        ):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                controller.projections[role].weight.device.type,
                "cpu",
            )

    def test_fusion_cache_never_crosses_hidden_identity(self):
        backend = _Backend(max_tokens=9, fuse_qkv=True)
        layer, state, controller = self._controller(
            width=64, backend=backend
        )
        controller.begin_load()
        source = self._load(controller, width=64)
        first_hidden = torch.randn(1, 64)
        second_hidden = first_hidden.clone()

        q_output = layer.self_attn.q_proj(first_hidden)
        # A different Tensor object must not receive first_hidden's cached k.
        k_output = layer.self_attn.k_proj(second_hidden)
        v_output = layer.self_attn.v_proj(second_hidden)

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 3)
        torch.testing.assert_close(
            q_output,
            _reference_mm(
                first_hidden, source["q"][0], source["q"][1].float()
            ),
            rtol=0,
            atol=0,
        )
        for role, output in (("k", k_output), ("v", v_output)):
            torch.testing.assert_close(
                output,
                _reference_mm(
                    second_hidden,
                    source[role][0],
                    source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )

    def test_inference_tensor_mutation_outside_attention_never_uses_cache(self):
        backend = _Backend(max_tokens=9, fuse_qkv=True)
        layer, state, controller = self._controller(
            width=64, backend=backend
        )
        controller.begin_load()
        source = self._load(controller, width=64)
        with torch.inference_mode():
            hidden = torch.randn(1, 64)
            layer.self_attn.q_proj(hidden)
            hidden.add_(1)
            got = layer.self_attn.k_proj(hidden)
        expected = _reference_mm(
            hidden, source["k"][0], source["k"][1].float()
        )
        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 2)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_token_crossover_materializes_once_without_disabling_backend(self):
        backend = _Backend(max_tokens=1, fuse_qkv=True)
        layer, state, controller = self._controller(
            width=64, backend=backend, meta=True
        )
        controller.begin_load()
        source = self._load(controller, width=64)
        hidden = torch.randn(1, 2, 64)

        outputs = layer.self_attn(hidden)
        for role, got in zip(("q", "k", "v"), outputs):
            torch.testing.assert_close(
                got,
                _reference_mm(
                    hidden, source[role][0], source[role][1].float()
                ),
                rtol=0,
                atol=0,
            )

        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(controller.dense_crossover_materializations, 1)
        self.assertEqual(controller.dense_crossover_project_calls, 3)
        self.assertEqual(controller.dense_crossover_releases, 1)
        for role in ("q", "k", "v"):
            self.assertEqual(
                getattr(layer.self_attn, f"{role}_proj").weight.device.type,
                "meta",
            )

        # A later decode-sized layer can use the same packed backend again.
        controller.begin_load()
        source = self._load(controller, width=64)
        decode_hidden = torch.randn(1, 1, 64)
        outputs = layer.self_attn(decode_hidden)
        for role, got in zip(("q", "k", "v"), outputs):
            torch.testing.assert_close(
                got,
                _reference_mm(
                    decode_hidden,
                    source[role][0],
                    source[role][1].float(),
                ),
                rtol=0,
                atol=0,
            )
        self.assertTrue(state.enabled)
        self.assertEqual(backend.calls, 1)

    def test_fused_operator_error_materializes_all_roles_atomically(self):
        def failing_mm(_hidden, _qweight, _scale):
            raise RuntimeError("synthetic fused backend rejection")

        backend = _Backend(
            failing_mm, max_tokens=9, fuse_qkv=True
        )
        layer, state, controller = self._controller(
            width=64, backend=backend, meta=True
        )
        controller.begin_load()
        source = self._load(controller, width=64)
        hidden = torch.randn(1, 64)

        got = layer.self_attn(hidden)[0]
        torch.testing.assert_close(
            got,
            _reference_mm(hidden, source["q"][0], source["q"][1].float()),
            rtol=0,
            atol=0,
        )
        self.assertFalse(state.enabled)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(controller.runtime_failures, 1)
        for role in ("q", "k", "v"):
            qweight, scale = source[role]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                qweight.float() * scale.float()[:, None],
                rtol=0,
                atol=0,
            )

    def test_operator_error_materializes_dense_and_disables_every_template(self):
        calls = 0

        def failing_mm(hidden, qweight, scale):
            nonlocal calls
            calls += 1
            raise RuntimeError("synthetic backend rejection")

        layer, state, controller = self._controller(failing_mm)
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(1, 2, 8)

        got = layer.self_attn.q_proj(hidden)
        expected = _reference_mm(
            hidden, source["q"][0], source["q"][1].float())
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertEqual(calls, 1)
        self.assertFalse(state.enabled)
        self.assertIn("synthetic backend rejection", state.reason)

        # All current roles were reconstructed, not only the one that failed.
        for role in ("q", "k", "v"):
            qweight, scale = source[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )

        # Once disabled, future layer application must take the normal dense
        # branch instead of silently reusing an old packed layer.
        controller.begin_load()
        self.assertFalse(controller.load(
            "self_attn.q_proj.weight",
            torch.ones(8, 8, dtype=torch.int8),
            torch.ones(8, dtype=torch.float16),
        ))

    def test_incomplete_layer_fails_closed(self):
        layer, state, controller = self._controller()
        controller.begin_load()
        qweight = torch.ones(8, 8, dtype=torch.int8)
        scale = torch.full((8,), 0.5, dtype=torch.float16)
        self.assertTrue(controller.load(
            "self_attn.q_proj.weight", qweight, scale))
        with self.assertRaisesRegex(
                RuntimeError, "no packed or dense value for k, v"):
            controller.finish_load()

        self.assertFalse(state.enabled)
        self.assertIn("no packed or dense value for k, v", state.reason)
        torch.testing.assert_close(
            layer.self_attn.q_proj.weight,
            torch.full((8, 8), 0.5),
            rtol=0,
            atol=0,
        )

    def test_two_templates_can_share_persistent_arenas(self):
        first, state, controller_a = self._controller()
        second = _Layer()
        controller_b = spine_fast.install_dynamic_q8_qkv(
            second,
            torch.device("cpu"),
            state,
            _reference_mm,
            arenas=controller_a.arenas(),
        )

        self.assertEqual(
            controller_a.arenas()[0].data_ptr(),
            controller_b.arenas()[0].data_ptr(),
        )
        self.assertEqual(
            controller_a.arenas()[1].data_ptr(),
            controller_b.arenas()[1].data_ptr(),
        )

    def test_backend_error_replaces_meta_dense_fallback(self):
        def failing_mm(hidden, qweight, scale):
            raise NotImplementedError("synthetic missing kernel")

        layer, state, controller = self._controller(failing_mm, meta=True)
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(1, 8)

        got = layer.self_attn.v_proj(hidden)
        expected = _reference_mm(
            hidden, source["v"][0], source["v"][1].float())
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertFalse(state.enabled)
        for role in ("q", "k", "v"):
            self.assertEqual(
                getattr(layer.self_attn, f"{role}_proj").weight.device.type,
                "cpu",
            )

    def test_hybrid_packed_and_dense_layer_falls_back_fully_dense(self):
        layer, state, controller = self._controller(meta=True)
        sources = {
            role: _source_for(index)
            for index, role in enumerate(("q", "k", "v"), start=1)
        }
        dense_v = (
            sources["v"][0].float() * sources["v"][1].float()[:, None])

        _apply_sources(
            layer,
            {"q": sources["q"], "k": sources["k"]},
            {"v": dense_v},
        )

        self.assertFalse(state.enabled)
        self.assertIn("dense projection(s): v", state.reason)
        hidden = torch.randn(2, 3, 8)
        for role in ("q", "k", "v"):
            qweight, scale = sources[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj")(hidden),
                _reference_mm(hidden, qweight, scale.float()),
                rtol=0,
                atol=0,
            )
        self.assertEqual(controller.packed_project_calls, 0)

    def test_copy_failure_materializes_packed_prefix_then_finishes_dense(self):
        layer, state, controller = self._controller(meta=True)
        sources = {
            role: _source_for(index)
            for index, role in enumerate(("q", "k", "v"), start=1)
        }
        original_views = controller._views

        class RejectingCopy:
            shape = sources["k"][0].shape

            @staticmethod
            def copy_(_source):
                raise RuntimeError("synthetic packed copy failure")

        def rejecting_k_view(role):
            if role == "k":
                return RejectingCopy(), original_views(role)[1]
            return original_views(role)

        controller._views = rejecting_k_view
        _apply_sources(layer, sources, {})

        self.assertFalse(state.enabled)
        self.assertIn("synthetic packed copy failure", state.reason)
        hidden = torch.randn(1, 2, 8)
        for role in ("q", "k", "v"):
            qweight, scale = sources[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj")(hidden),
                _reference_mm(hidden, qweight, scale.float()),
                rtol=0,
                atol=0,
            )
        self.assertEqual(controller.packed_project_calls, 0)

    def test_uninstall_restores_linear_forward_and_removes_buffers(self):
        layer, _state, controller = self._controller()
        controller.uninstall()

        self.assertNotIn("forward", layer.self_attn.q_proj.__dict__)
        self.assertNotIn("forward", layer.self_attn.__dict__)
        self.assertIsNone(spine_fast.dynamic_q8_qkv(layer))
        self.assertNotIn(controller._QBUF, layer.self_attn._buffers)
        self.assertNotIn(controller._SBUF, layer.self_attn._buffers)

    def test_invalid_backend_leaves_module_unmodified(self):
        layer = _Layer()
        state = spine_fast.DynamicQ8State()
        with self.assertRaisesRegex(ValueError, "no callable matmul"):
            spine_fast.install_dynamic_q8_qkv(
                layer, torch.device("cpu"), state, object()
            )
        self.assertIsNone(spine_fast.dynamic_q8_qkv(layer))
        self.assertNotIn(
            spine_fast.DynamicPackedQ8QKV._QBUF,
            layer.self_attn._buffers,
        )
        self.assertNotIn(
            spine_fast.DynamicPackedQ8QKV._SBUF,
            layer.self_attn._buffers,
        )


if __name__ == "__main__":
    unittest.main()
