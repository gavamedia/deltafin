"""Validation and provenance for Kimi-K3's generated tensor inventory.

The inventory is derived deterministically from the pinned safetensors
headers.  Its exact size and SHA-256 are pinned alongside the small repository
files, and every consumer that writes a tensor-named path verifies it first.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat

try:
    import model_source
except ImportError:  # imported as tools.inventory_meta
    from . import model_source


INVENTORY_FILENAME = "tensor_inventory_offsets.json"
EXPECTED_TENSOR_COUNT = model_source.PINNED_INVENTORY_TENSORS
SHARD_COUNT = 96
MAX_TENSOR_RANK = 16
MAX_DIMENSION = 1 << 31
MAX_HEADER_BYTES = 128 << 20
MAX_SHARD_DATA_BYTES = 1 << 40
HASH_CHUNK = 1 << 20

DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "U8": 1,
}

_SHARD_RE = re.compile(r"model-(\d{5})-of-000096\.safetensors\Z")
_TENSOR_KEYS = {"dtype", "shape", "offsets", "shard", "hlen"}


class InventoryValidationError(RuntimeError):
    """The local tensor inventory cannot be trusted."""


def _is_plain_int(value):
    return type(value) is int


def _regular_file_info(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise InventoryValidationError(f"{path} is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise InventoryValidationError(f"{path} is a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise InventoryValidationError(f"{path} is not a regular file")
    return info


def file_sha256(path):
    _regular_file_info(path)
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    info = _regular_file_info(path)
    return info.st_size, file_sha256(path)


def _read_regular_json(path, label):
    _regular_file_info(path)
    try:
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryValidationError(
            f"{label} {path} is not valid UTF-8 JSON: {exc}"
        ) from exc


def validate_tensor_name(name):
    """Validate a tensor identifier before treating it as a filename."""
    if not isinstance(name, str) or not name:
        raise InventoryValidationError("tensor name must be a non-empty string")
    if "\x00" in name:
        raise InventoryValidationError(f"tensor name {name!r} contains NUL")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise InventoryValidationError(
            f"tensor name {name!r} must be one relative path component"
        )
    if os.path.isabs(name) or os.path.splitdrive(name)[0]:
        raise InventoryValidationError(
            f"tensor name {name!r} must be relative"
        )
    return name


def _validate_shard_name(shard):
    if not isinstance(shard, str):
        raise InventoryValidationError("shard must be a string")
    match = _SHARD_RE.fullmatch(shard)
    if match is None:
        raise InventoryValidationError(
            f"invalid shard filename {shard!r}; expected "
            "model-00001-of-000096.safetensors through "
            "model-00096-of-000096.safetensors"
        )
    number = int(match.group(1))
    if not 1 <= number <= SHARD_COUNT:
        raise InventoryValidationError(
            f"shard number {number} is outside 1..{SHARD_COUNT}"
        )
    return number


def validate_inventory(document, expected_tensor_count=None):
    """Return ``document`` after strict validation of all tensor records."""
    if not isinstance(document, dict):
        raise InventoryValidationError("tensor inventory must be a JSON object")
    expected_count = (
        EXPECTED_TENSOR_COUNT
        if expected_tensor_count is None
        else expected_tensor_count
    )
    if len(document) != expected_count:
        raise InventoryValidationError(
            f"tensor inventory has {len(document)} records; "
            f"expected {expected_count}"
        )

    ranges_by_shard = {number: [] for number in range(1, SHARD_COUNT + 1)}
    hlen_by_shard = {}
    for name, entry in document.items():
        validate_tensor_name(name)
        if not isinstance(entry, dict):
            raise InventoryValidationError(
                f"tensor {name!r} metadata must be an object"
            )
        if set(entry) != _TENSOR_KEYS:
            raise InventoryValidationError(
                f"tensor {name!r} metadata keys are {sorted(entry)}; "
                f"expected {sorted(_TENSOR_KEYS)}"
            )

        dtype = entry["dtype"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise InventoryValidationError(
                f"tensor {name!r} has unsupported dtype {dtype!r}"
            )

        shape = entry["shape"]
        if not isinstance(shape, list) or not 1 <= len(shape) <= MAX_TENSOR_RANK:
            raise InventoryValidationError(
                f"tensor {name!r} shape must be a non-empty list with at most "
                f"{MAX_TENSOR_RANK} dimensions"
            )
        elements = 1
        for dimension in shape:
            if (
                not _is_plain_int(dimension)
                or dimension <= 0
                or dimension > MAX_DIMENSION
            ):
                raise InventoryValidationError(
                    f"tensor {name!r} has invalid shape dimension "
                    f"{dimension!r}"
                )
            elements *= dimension
            if elements * DTYPE_BYTES[dtype] > MAX_SHARD_DATA_BYTES:
                raise InventoryValidationError(
                    f"tensor {name!r} exceeds the permitted size bound"
                )

        offsets = entry["offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_plain_int(value) for value in offsets)
        ):
            raise InventoryValidationError(
                f"tensor {name!r} offsets must be two plain integers"
            )
        start, end = offsets
        if (
            start < 0
            or end < start
            or end > MAX_SHARD_DATA_BYTES
        ):
            raise InventoryValidationError(
                f"tensor {name!r} has invalid offset bounds {offsets!r}"
            )
        expected_bytes = elements * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise InventoryValidationError(
                f"tensor {name!r} spans {end - start} bytes but its "
                f"{dtype} shape requires {expected_bytes}"
            )

        shard_number = _validate_shard_name(entry["shard"])
        hlen = entry["hlen"]
        if (
            not _is_plain_int(hlen)
            or hlen <= 0
            or hlen > MAX_HEADER_BYTES
        ):
            raise InventoryValidationError(
                f"tensor {name!r} has invalid header length {hlen!r}"
            )
        prior_hlen = hlen_by_shard.setdefault(shard_number, hlen)
        if prior_hlen != hlen:
            raise InventoryValidationError(
                f"shard {shard_number} has inconsistent header lengths "
                f"{prior_hlen} and {hlen}"
            )
        ranges_by_shard[shard_number].append((start, end, name))

    for shard_number, ranges in ranges_by_shard.items():
        if not ranges:
            raise InventoryValidationError(
                f"tensor inventory has no records for shard {shard_number}"
            )
        ranges.sort()
        expected_start = 0
        for start, end, name in ranges:
            if start != expected_start:
                raise InventoryValidationError(
                    f"shard {shard_number} has a gap or overlap before "
                    f"tensor {name!r}: expected offset {expected_start}, "
                    f"found {start}"
                )
            expected_start = end
    return document


def read_unbound_inventory(path, expected_tensor_count=None):
    """Read and structurally validate a legacy inventory without trusting it."""
    document = _read_regular_json(path, "tensor inventory")
    return validate_inventory(document, expected_tensor_count)


def load_verified_inventory(meta_directory, expected_tensor_count=None):
    """Load an inventory only when its pinned exact bytes verify."""
    inventory_path = os.path.join(meta_directory, INVENTORY_FILENAME)
    size, sha256 = file_identity(inventory_path)
    if size != model_source.PINNED_INVENTORY_SIZE:
        raise InventoryValidationError(
            f"tensor inventory has {size} bytes; pinned revision "
            f"{model_source.MODEL_REVISION} requires "
            f"{model_source.PINNED_INVENTORY_SIZE}"
        )
    if sha256 != model_source.PINNED_INVENTORY_SHA256:
        raise InventoryValidationError(
            f"tensor inventory SHA-256 is {sha256}; pinned revision "
            f"{model_source.MODEL_REVISION} requires "
            f"{model_source.PINNED_INVENTORY_SHA256}"
        )
    document = _read_regular_json(inventory_path, "tensor inventory")
    validate_inventory(document, expected_tensor_count)
    return document


def safe_tensor_output_path(root, name):
    """Resolve a validated tensor filename, refusing any root escape."""
    validate_tensor_name(name)
    root = os.path.abspath(root)
    destination = os.path.abspath(os.path.join(root, name))
    try:
        inside_root = os.path.commonpath((root, destination)) == root
    except ValueError:
        inside_root = False
    if not inside_root:
        raise InventoryValidationError(
            f"tensor path {destination!r} escapes output root {root!r}"
        )
    return destination


__all__ = [
    "EXPECTED_TENSOR_COUNT",
    "INVENTORY_FILENAME",
    "InventoryValidationError",
    "MAX_HEADER_BYTES",
    "SHARD_COUNT",
    "file_identity",
    "load_verified_inventory",
    "read_unbound_inventory",
    "safe_tensor_output_path",
    "validate_inventory",
    "validate_tensor_name",
]
