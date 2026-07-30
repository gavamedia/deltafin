"""Capability-gated raw row-int8 projection backend.

PyTorch's ``aten::_weight_int8pack_mm`` is a private operator whose registered
devices vary by build.  This module centralizes the unsafe details so callers
never infer support from a platform name or from symbol presence alone.

Discovery requires:

1. the Python symbol;
2. a native dispatcher registration for the selected device when the
   dispatcher query is available; and
3. an analytically exact canary at the caller's real matrix shape.

Runtime callers must still catch backend errors and atomically retain a dense
fallback: a dispatcher entry cannot promise every future chip/shape/ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


OPERATOR_SCHEMA = "aten::_weight_int8pack_mm"
_DISPATCH_KEYS = {
    "cpu": "CPU",
    "mps": "MPS",
    "cuda": "CUDA",
}
_OFF = {"0", "false", "off", "no"}
_ON = {"1", "true", "on", "yes", "force"}


def _resolve_operator(torch_module) -> Callable | None:
    operator = getattr(torch_module, "_weight_int8pack_mm", None)
    if operator is not None:
        return operator
    aten = getattr(getattr(torch_module, "ops", None), "aten", None)
    packet = getattr(aten, "_weight_int8pack_mm", None)
    return getattr(packet, "default", packet)


def _synchronize(torch_module, device) -> None:
    if device.type == "mps":
        torch_module.mps.synchronize()
    elif device.type == "cuda":
        torch_module.cuda.synchronize(device)


def _analytical_canary(
    torch_module,
    device,
    dtype,
    shape: tuple[int, int],
    operator: Callable,
) -> dict[str, Any]:
    """Run one exact, nontrivial call without allocating a dense reference."""
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"invalid packed-q8 matrix shape {shape}")

    qweight = scale = hidden = output = actual = expected = None
    try:
        qweight = torch_module.zeros(
            rows, cols, dtype=torch_module.int8, device=device
        )
        scale = torch_module.ones(rows, dtype=dtype, device=device)
        hidden = torch_module.zeros(1, cols, dtype=dtype, device=device)

        # Exercise starts, SIMD/group boundaries, midpoints, and tails in both
        # dimensions. Powers of two keep the sparse reference exactly
        # representable while detecting a backend that silently truncates later
        # rows/columns. This remains tiny logically even at K3's real shape and
        # never constructs a dense floating-point reference.
        candidate_columns = (
            0, 1, 2, 3, 31, 32, cols // 2, cols - 2, cols - 1
        )
        columns = sorted({
            column for column in candidate_columns if 0 <= column < cols
        })
        candidate_rows = (
            0, 1, 2, 31, 32, rows // 2, rows - 2, rows - 1
        )
        tested_rows = sorted({
            row for row in candidate_rows if 0 <= row < rows
        })
        hidden_cycle = (1.0, -2.0, 0.5, 4.0, -0.25, 8.0, 0.125, -16.0)
        scale_cycle = (0.5, 0.25, 2.0, 1.0)
        expected = torch_module.zeros(1, rows, dtype=dtype)
        for column_index, column in enumerate(columns):
            hidden[0, column] = hidden_cycle[
                column_index % len(hidden_cycle)
            ]
        for row_index, row in enumerate(tested_rows):
            total = 0.0
            for column_index, column in enumerate(columns):
                weight = (
                    ((row_index + 3) * (column_index + 5)) % 15
                ) - 7
                if weight == 0:
                    weight = 3
                qweight[row, column] = weight
                total += (
                    hidden_cycle[column_index % len(hidden_cycle)] * weight
                )
            row_scale = scale_cycle[row_index % len(scale_cycle)]
            scale[row] = row_scale
            expected[0, row] = total * row_scale

        output = operator(hidden, qweight, scale)
        _synchronize(torch_module, device)
        actual = output.detach().cpu()
        exact = bool(torch_module.equal(actual, expected))
        result = {
            "shape": [rows, cols],
            "output_shape": list(output.shape),
            "tested_rows": tested_rows,
            "tested_columns": columns,
            "analytical_exact": exact,
            "max_abs": float(
                (actual.float() - expected.float()).abs().max()
            ),
        }
        if not exact:
            raise RuntimeError(
                "packed-q8 analytical real-shape canary was not bit-exact"
            )
        return result
    finally:
        output = actual = expected = hidden = scale = qweight = None
        if device.type == "mps":
            torch_module.mps.empty_cache()
        elif device.type == "cuda":
            torch_module.cuda.empty_cache()


def _parse_max_tokens(
    value: str | int | None,
    *,
    device_type: str,
    torch_threads: int,
) -> int:
    if value is not None and str(value).strip().lower() not in ("", "auto"):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("packed-q8 max tokens must be positive")
        return parsed
    # Measured safe ranges, not architecture claims. CPU is opt-in and stays
    # conservative because the raw GEMV wins decisively at T=1, at T=2 only
    # with multiple worker threads, and can lose to an already-resident dense
    # GEMM at larger T. MPS remained ahead through the exact-sequence-tested
    # T=9 ceiling. A T=13 whole-model probe changed a greedy token even though
    # its isolated packed-head argmax probe passed, so wider batches stay out.
    if device_type == "cpu":
        return 2 if torch_threads >= 2 else 1
    if device_type == "mps":
        return 9
    # CUDA n-gram spec decode produces 2-row batches. The packed int8 head
    # handles T=2 without materializing the dense 4.38 GiB head; T=13 had
    # a verified greedy-token change, so stay well below that.
    if device_type == "cuda":
        return 2
    return 1


def _fusion_canary(
    torch_module,
    device,
    dtype,
    shape: tuple[int, int],
    operator: Callable,
) -> dict[str, Any]:
    """Prove that an unequal six-role bundle equals independent calls."""
    rows, cols = shape
    # MPS currently requires both output rows (N) and input columns (K) to be
    # multiples of 32. A successful real-shape canary therefore guarantees
    # these groups are legal there. The 1:2:3:4:2:1 pattern models KDA's six
    # unequal contiguous roles and catches assumptions that an equal Q/K/V
    # triplet cannot expose.
    row_unit = min(rows, 32)
    split_rows = tuple(
        row_unit * multiplier for multiplier in (1, 2, 3, 4, 2, 1)
    )
    total_rows = sum(split_rows)
    probe_tokens = 2

    # Keep the real input width—the reduction dimension is where a backend is
    # most likely to change algorithms—but use only a few output rows so this
    # capability check stays small even for K3's 12,288-row projections.
    qweight = scale = hidden = fused = separate = None
    fused_cpu = separate_cpu = None
    try:
        qweight = (
            torch_module.arange(
                total_rows * cols,
                dtype=torch_module.int32,
            )
            .remainder(251)
            .sub(125)
            .to(torch_module.int8)
            .view(total_rows, cols)
            .to(device)
        )
        scale = torch_module.linspace(
            0.0031,
            0.0197,
            total_rows,
            dtype=dtype,
        ).to(device)
        hidden = (
            torch_module.arange(
                probe_tokens * cols,
                dtype=dtype,
            )
            .view(probe_tokens, cols)
            .remainder(29)
            .sub(14)
            .div(31)
            .to(device)
        )

        fused = operator(hidden, qweight, scale)
        cursor = 0
        pieces = []
        for group_rows in split_rows:
            end = cursor + group_rows
            pieces.append(
                operator(
                    hidden,
                    qweight[cursor:end],
                    scale[cursor:end],
                )
            )
            cursor = end
        separate = torch_module.cat(
            pieces,
            dim=-1,
        )
        _synchronize(torch_module, device)
        fused_cpu = fused.detach().cpu()
        separate_cpu = separate.detach().cpu()
        exact = bool(torch_module.equal(fused_cpu, separate_cpu))
        result = {
            "shape": [total_rows, cols],
            "split_rows": list(split_rows),
            "tokens": probe_tokens,
            "separate_exact": exact,
            "max_abs": float(
                (fused_cpu.float() - separate_cpu.float()).abs().max()
            ),
        }
        if not exact:
            raise RuntimeError(
                "packed-q8 fused rows differed from independent projections"
            )
        return result
    finally:
        fused = separate = fused_cpu = separate_cpu = None
        hidden = scale = qweight = None
        if device.type == "mps":
            torch_module.mps.empty_cache()
        elif device.type == "cuda":
            torch_module.cuda.empty_cache()


@dataclass
class PackedQ8Backend:
    available: bool
    reason: str | None
    device: Any
    dtype: Any
    shape: tuple[int, int]
    operator: Callable | None
    symbol_present: bool
    dispatch_key: str | None
    dispatch_registered: bool | None
    canary: dict[str, Any] | None
    fusion_canary: dict[str, Any] | None
    max_tokens: int
    fuse_qkv: bool
    fusion_reason: str | None
    default_enabled: bool

    def matmul(self, hidden, qweight, scale):
        if not self.available or self.operator is None:
            raise RuntimeError(self.reason or "packed-q8 backend unavailable")
        return self.operator(hidden, qweight, scale)

    def supports_tokens(self, token_count: int) -> bool:
        return self.available and 0 < token_count <= self.max_tokens

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "operator": OPERATOR_SCHEMA,
            "symbol_present": self.symbol_present,
            "dispatch_key": self.dispatch_key,
            "dispatch_registered": self.dispatch_registered,
            "canary": self.canary,
            "fusion_canary": self.fusion_canary,
            "max_tokens": self.max_tokens,
            "fuse_qkv": self.fuse_qkv,
            "fusion_reason": self.fusion_reason,
            "default_enabled": self.default_enabled,
        }


def discover(
    torch_module,
    device,
    dtype,
    shape,
    *,
    mode: str = "auto",
    max_tokens: str | int | None = None,
    fusion_mode: str = "auto",
    operator: Callable | None = None,
    dispatch_query: Callable[[str, str], bool] | None = None,
) -> PackedQ8Backend:
    """Return a backend record; never raise for an unsupported accelerator."""
    device = torch_module.device(device)
    shape = tuple(int(value) for value in shape)
    normalized_mode = str(mode).strip().lower()
    normalized_fusion = str(fusion_mode).strip().lower()
    torch_threads = int(torch_module.get_num_threads())
    token_limit = _parse_max_tokens(
        max_tokens,
        device_type=device.type,
        torch_threads=torch_threads,
    )
    dispatch_key = _DISPATCH_KEYS.get(device.type)
    default_enabled = device.type == "mps"

    def unavailable(
        reason: str,
        *,
        symbol: bool = False,
        registered: bool | None = None,
    ) -> PackedQ8Backend:
        return PackedQ8Backend(
            available=False,
            reason=reason,
            device=device,
            dtype=dtype,
            shape=shape,
            operator=None,
            symbol_present=symbol,
            dispatch_key=dispatch_key,
            dispatch_registered=registered,
            canary=None,
            fusion_canary=None,
            max_tokens=token_limit,
            fuse_qkv=False,
            fusion_reason=reason,
            default_enabled=default_enabled,
        )

    if normalized_mode in _OFF:
        return unavailable("disabled by K3_NATIVE_INT8")
    if normalized_mode not in _ON | {"auto"}:
        return unavailable(
            "K3_NATIVE_INT8 must be auto, on/1/force, or off/0"
        )
    if dtype != torch_module.float32:
        return unavailable(
            "packed-q8 exact path currently requires fp32 activations"
        )
    if dispatch_key is None:
        return unavailable(
            f"no audited packed-q8 dispatch key for device type {device.type}"
        )

    if operator is None:
        operator = _resolve_operator(torch_module)
    symbol_present = operator is not None
    if not symbol_present:
        return unavailable(
            "packed-q8 operator symbol is absent",
            symbol=False,
        )

    if dispatch_query is None:
        dispatch_query = getattr(
            getattr(torch_module, "_C", None),
            "_dispatch_has_kernel_for_dispatch_key",
            None,
        )
    registered = None
    if dispatch_query is not None:
        try:
            registered = bool(
                dispatch_query(OPERATOR_SCHEMA, dispatch_key)
            )
        except Exception:
            registered = None
    # A known-negative registration is final even under "force": sending a
    # private CPU/MPS op to CUDA and hoping for a fallback is not a capability.
    if registered is False:
        return unavailable(
            f"{OPERATOR_SCHEMA} has no native {dispatch_key} kernel",
            symbol=True,
            registered=False,
        )

    try:
        canary = _analytical_canary(
            torch_module, device, dtype, shape, operator
        )
    except (NotImplementedError, RuntimeError, TypeError, ValueError,
            MemoryError) as exc:
        return PackedQ8Backend(
            available=False,
            reason=f"real-shape canary failed: {type(exc).__name__}: {exc}",
            device=device,
            dtype=dtype,
            shape=shape,
            operator=None,
            symbol_present=True,
            dispatch_key=dispatch_key,
            dispatch_registered=registered,
            canary=None,
            fusion_canary=None,
            max_tokens=token_limit,
            fuse_qkv=False,
            fusion_reason="base backend canary failed",
            default_enabled=default_enabled,
        )

    # Row concatenation was measured and bit-checked only for these native
    # backends.  It remains opt-out and is still runtime exception-guarded.
    fusion_measured = device.type in {"cpu", "mps"}
    if normalized_fusion in _OFF:
        fuse_qkv = False
        fusion_reason = "disabled by K3_INT8_QKV_FUSE"
    elif normalized_fusion not in _ON | {"auto"}:
        fuse_qkv = False
        fusion_reason = (
            "K3_INT8_QKV_FUSE must be auto, on/1/force, or off/0"
        )
    elif not fusion_measured:
        fuse_qkv = False
        fusion_reason = (
            f"no equal-row fusion evidence for {device.type}"
        )
    else:
        try:
            fusion_canary = _fusion_canary(
                torch_module,
                device,
                dtype,
                shape,
                operator,
            )
        except (NotImplementedError, RuntimeError, TypeError, ValueError,
                MemoryError) as exc:
            fuse_qkv = False
            fusion_reason = (
                "fused-row exactness canary failed: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            fuse_qkv = True
            fusion_reason = None

    if not fuse_qkv:
        fusion_canary = None

    return PackedQ8Backend(
        available=True,
        reason=None,
        device=device,
        dtype=dtype,
        shape=shape,
        operator=operator,
        symbol_present=True,
        dispatch_key=dispatch_key,
        dispatch_registered=registered,
        canary=canary,
        fusion_canary=fusion_canary,
        max_tokens=token_limit,
        fuse_qkv=fuse_qkv,
        fusion_reason=fusion_reason,
        default_enabled=default_enabled,
    )
