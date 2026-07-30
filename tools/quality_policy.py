"""Hard quality invariants for the supported Deltafin runtime."""
from __future__ import annotations


_FALSE = {"", "0", "false", "off", "no", "disabled"}
_TRUE = {"1", "true", "on", "yes", "enabled"}


def full_moe_top_k(base_top_k: int, requested: str | None) -> int:
    """Require the model's complete routed-expert count."""
    base_top_k = int(base_top_k)
    if requested is None or not str(requested).strip():
        return base_top_k
    try:
        selected = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"K3_MOE_TOP_K must remain {base_top_k}; got {requested!r}"
        ) from exc
    if selected != base_top_k:
        raise ValueError(
            f"K3_MOE_TOP_K={selected} is unsupported: Deltafin's quality "
            f"invariant requires all {base_top_k} routed experts"
        )
    return base_top_k


def require_fp32(
    approximate: str | None,
    dtype: str | None,
) -> None:
    """Reject the former quality-changing activation precision controls."""
    approx = str(approximate or "0").strip().lower()
    if approx in _TRUE:
        raise ValueError(
            "K3_APPROX is unsupported: Deltafin does not trade output "
            "quality for speed"
        )
    if approx not in _FALSE:
        raise ValueError(
            "K3_APPROX must be off/0/false; approximate inference is "
            "unsupported"
        )
    selected_dtype = str(dtype or "fp32").strip().lower()
    if selected_dtype not in {"fp32", "float32"}:
        raise ValueError(
            f"K3_DTYPE={selected_dtype!r} is unsupported: the supported "
            "runtime keeps fp32 activation numerics"
        )


def supported_spine(selected: str) -> str:
    """Allow only spine representations retained by the quality oracle."""
    normalized = str(selected).strip().lower()
    if normalized == "mixed":
        raise ValueError(
            "K3_SPINE=mixed is unsupported: its 4/6-bit research codecs "
            "change target weights and violate Deltafin's quality invariant"
        )
    if normalized not in {"bf16", "int8"}:
        raise ValueError(
            f"unsupported K3_SPINE={selected!r}; expected bf16 or int8"
        )
    return normalized


__all__ = ["full_moe_top_k", "require_fp32", "supported_spine"]
