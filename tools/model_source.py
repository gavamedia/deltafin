"""Immutable source identity for Moonshot's Kimi-K3 checkpoint.

The default Hugging Face paths deliberately use a commit, never ``main``.
Explicit mirror overrides remain authoritative for installations that carry a
private copy of the same checkpoint; setup still verifies every small
executable/configuration file against the manifest below.
"""
from __future__ import annotations

import os


MODEL_REPOSITORY = "moonshotai/Kimi-K3"
MODEL_REVISION = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
DEFAULT_HF_HOST = "huggingface.co"
DEFAULT_HF_PATH = (
    f"/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/"
)

# Deterministic output of setup_k3.py after reading the 96 safetensors
# headers at MODEL_REVISION.  This pins the generated offset inventory just as
# tightly as the small repository files below.  Explicit mirrors remain
# supported when they carry the same pinned checkpoint bytes and layout.
PINNED_INVENTORY_SIZE = 108_581_016
PINNED_INVENTORY_SHA256 = (
    "b287e9659afbfd361b1485721a6703a5bc55bb8399ed074a1e5a29803958b425"
)
PINNED_INVENTORY_TENSORS = 497_220

# name: (exact byte length, SHA-256)
PINNED_FILES = {
    "LICENSE": (
        3065,
        "20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887",
    ),
    "config.json": (
        7006,
        "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    ),
    "generation_config.json": (
        53,
        "c6648c25e9705af7fba8847e243840d21b5cc63ddeb6297f750a7ddbb6a02836",
    ),
    "tokenizer_config.json": (
        3478,
        "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    ),
    "tokenization_kimi.py": (
        16145,
        "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    ),
    "encoding_k3.py": (
        22827,
        "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333",
    ),
    "tiktoken.model": (
        2795286,
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    ),
    "modeling_kimi_linear.py": (
        51506,
        "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    ),
    "modeling_kimi_k3.py": (
        53444,
        "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2",
    ),
    "configuration_kimi_k3.py": (
        11343,
        "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
    ),
}


def hf_host() -> str:
    """Return the explicit mirror host or the official default."""
    return os.environ.get("K3_HF_HOST", DEFAULT_HF_HOST)


def hf_path() -> str:
    """Return a normalized explicit mirror path or the immutable default."""
    selected = os.environ.get("K3_HF_PATH", DEFAULT_HF_PATH)
    if not selected.startswith("/"):
        selected = "/" + selected
    if not selected.endswith("/"):
        selected += "/"
    return selected


def base_url() -> str:
    """HTTPS base URL used by urllib-based downloaders."""
    return f"https://{hf_host()}{hf_path()}"


__all__ = [
    "DEFAULT_HF_HOST",
    "DEFAULT_HF_PATH",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "PINNED_INVENTORY_SHA256",
    "PINNED_INVENTORY_SIZE",
    "PINNED_INVENTORY_TENSORS",
    "PINNED_FILES",
    "base_url",
    "hf_host",
    "hf_path",
]
