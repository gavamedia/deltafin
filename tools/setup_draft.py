#!/usr/bin/env python3
"""Install Deltafin's optional, lossless universal draft assistants.

Only a fixed set of data/configuration files is downloaded from a pinned
official Qwen revision for each assistant. Every byte is SHA-256 checked before
the temporary directory is atomically published. Nothing from either model
repository is imported or executed; the runtime also requires Transformers'
built-in qwen3 implementation, ``trust_remote_code=False``, and safetensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request


ROOT = (
    os.environ.get("DELTAFIN_ROOT")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DESTINATION = os.path.join(ROOT, "k3-draft-qwen3-1.7b-base")
REPOSITORY = "Qwen/Qwen3-1.7B-Base"
REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
BASE_URL = (
    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
)
FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SIZES = {
    "LICENSE": 11_343,
    "config.json": 727,
    "generation_config.json": 138,
    "merges.txt": 1_671_853,
    "model.safetensors": 3_441_185_608,
    "tokenizer.json": 7_031_645,
    "tokenizer_config.json": 9_678,
    "vocab.json": 2_776_833,
}
PROBE_DESTINATION = os.path.join(ROOT, "k3-draft-qwen3-0.6b-base")
PROBE_REPOSITORY = "Qwen/Qwen3-0.6B-Base"
PROBE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PROBE_BASE_URL = (
    f"https://huggingface.co/{PROBE_REPOSITORY}/resolve/{PROBE_REVISION}/"
)
PROBE_FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
PROBE_SIZES = {
    "LICENSE": 11_343,
    "config.json": 727,
    "generation_config.json": 138,
    "merges.txt": 1_671_853,
    "model.safetensors": 1_192_135_096,
    "tokenizer.json": 7_031_645,
    "tokenizer_config.json": 9_678,
    "vocab.json": 2_776_833,
}
CHUNK = 8 << 20


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def validate(
    directory: str,
    *,
    files: dict[str, str] | None = None,
    sizes: dict[str, int] | None = None,
) -> tuple[bool, str | None]:
    expected_files = FILES if files is None else files
    expected_sizes = SIZES if sizes is None else sizes
    for name, expected in expected_files.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            return False, f"missing {name}"
        expected_size = expected_sizes[name]
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            return (
                False,
                f"{name} size is {actual_size}, expected {expected_size}",
            )
        actual = _sha256(path)
        if actual != expected:
            return False, f"{name} hash is {actual}, expected {expected}"
    config_path = os.path.join(directory, "config.json")
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)
    if config.get("model_type") != "qwen3" or config.get("auto_map"):
        return False, "config is not built-in qwen3 data-only"
    return True, None


def _download(
    name: str,
    destination: str,
    expected: str,
    expected_size: int,
) -> None:
    _download_from(
        BASE_URL, name, destination, expected, expected_size
    )


def _download_from(
    base_url: str,
    name: str,
    destination: str,
    expected: str,
    expected_size: int,
) -> None:
    request = urllib.request.Request(
        base_url + name + "?download=true",
        headers={"User-Agent": "deltafin-draft-setup"},
    )
    digest = hashlib.sha256()
    received = 0
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        open(destination, "xb") as target,
    ):
        while block := response.read(CHUNK):
            target.write(block)
            digest.update(block)
            received += len(block)
            if received > expected_size:
                raise RuntimeError(
                    f"{name} exceeded its pinned {expected_size}-byte size"
                )
            if name == "model.safetensors" and received % (256 << 20) < CHUNK:
                print(f"  {name}: {received / 1e9:.2f} GB", flush=True)
        target.flush()
        os.fsync(target.fileno())
    actual = digest.hexdigest()
    if received != expected_size:
        raise RuntimeError(
            f"{name} size is {received}, expected {expected_size}"
        )
    if actual != expected:
        raise RuntimeError(
            f"{name} failed SHA-256 verification: {actual} != {expected}"
        )
    print(f"  {name}: verified ({received / 1e6:.1f} MB)", flush=True)


def _download_probe(
    name: str,
    destination: str,
    expected: str,
    expected_size: int,
) -> None:
    _download_from(
        PROBE_BASE_URL, name, destination, expected, expected_size
    )


def _install_model(
    *,
    destination: str,
    repository: str,
    revision: str,
    files: dict[str, str],
    sizes: dict[str, int],
    downloader,
) -> None:
    if os.path.isdir(destination):
        valid, reason = validate(destination, files=files, sizes=sizes)
        if valid:
            print(
                f"Universal draft assistant is already installed and verified "
                f"at {destination}"
            )
            return
        raise RuntimeError(
            f"refusing to replace the existing {destination}: {reason}. "
            "Move that directory aside and retry."
        )
    if os.path.lexists(destination):
        raise RuntimeError(
            f"refusing to replace non-directory path {destination}"
        )

    temporary = tempfile.mkdtemp(
        prefix=f".{os.path.basename(destination)}-", dir=ROOT
    )
    try:
        print(
            f"Downloading {repository} at pinned revision {revision}...",
            flush=True,
        )
        for name, expected in files.items():
            downloader(
                name,
                os.path.join(temporary, name),
                expected,
                sizes[name],
            )
        valid, reason = validate(temporary, files=files, sizes=sizes)
        if not valid:
            raise RuntimeError(reason)
        with open(
            os.path.join(temporary, "deltafin-manifest.json"),
            "x",
            encoding="utf-8",
        ) as target:
            json.dump(
                {
                    "repository": repository,
                    "revision": revision,
                    "files": files,
                    "sizes": sizes,
                },
                target,
                indent=2,
                sort_keys=True,
            )
            target.write("\n")
        if sys.platform == "darwin":
            open(
                os.path.join(temporary, ".metadata_never_index"), "x"
            ).close()
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary:
            shutil.rmtree(temporary)


def install() -> None:
    """Install the 1.7B wide-proposal assistant.

    Kept as the single-model entry point for callers and offline tests. The
    command-line installer calls this and :func:`install_probe`.
    """
    _install_model(
        destination=DESTINATION,
        repository=REPOSITORY,
        revision=REVISION,
        files=FILES,
        sizes=SIZES,
        downloader=_download,
    )


def install_probe() -> None:
    """Install the 0.6B admission/fallback assistant."""
    _install_model(
        destination=PROBE_DESTINATION,
        repository=PROBE_REPOSITORY,
        revision=PROBE_REVISION,
        files=PROBE_FILES,
        sizes=PROBE_SIZES,
        downloader=_download_probe,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the installed assistant without network access",
    )
    args = parser.parse_args()
    if args.check:
        checks = (
            (
                "wide assistant",
                DESTINATION,
                REPOSITORY,
                REVISION,
                FILES,
                SIZES,
            ),
            (
                "probe assistant",
                PROBE_DESTINATION,
                PROBE_REPOSITORY,
                PROBE_REVISION,
                PROBE_FILES,
                PROBE_SIZES,
            ),
        )
        for label, destination, repository, revision, files, sizes in checks:
            valid, reason = validate(
                destination, files=files, sizes=sizes
            )
            if not valid:
                raise SystemExit(f"{label} check failed: {reason}")
            print(
                f"{label} check passed: {repository}@{revision}",
                flush=True,
            )
        return
    install()
    install_probe()
    print(
        "Lossless hybrid drafting installed. It will be selected "
        "automatically on MPS; CUDA/CPU can test it with K3_UAG_DRAFT=on, "
        "and K3_UAG_DRAFT=off disables it.",
        flush=True,
    )


if __name__ == "__main__":
    main()
