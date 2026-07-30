#!/usr/bin/env python3
"""One-time setup for a fresh Deltafin clone.

Always downloads the small stuff from huggingface.co/moonshotai/Kimi-K3:
  k3-meta/      config, tokenizer, and Moonshot's modeling files (a few MB)
  tools/k3pkg/  import home for those modeling modules
plus k3-meta/tensor_inventory_offsets.json, built by reading only the 96 shard
HEADERS via range requests. Idempotent; safe to re-run.

Unless ``--no-draft`` is supplied, setup also invokes ``setup_draft.py`` to
install the pinned, hash-verified 0.6B + 1.7B universal assistants (~4.63 GB
total). They only schedule proposed text; K3 derives and verifies every output
token.

Then it installs the weights, in one of two modes:

  --full    (default when the disk allows it, and STRONGLY recommended)
            resident spine (~114 GB) + every routed expert (~1.45 TB).
            Nothing is fetched over the network at inference time, so every
            prompt runs at full speed.

  --stream  resident spine only (~114 GB). Experts are fetched over HTTP as the
            router asks for them. This works, but a token needs 25.8 GB of
            expert data: ~4 s from local NVMe versus MINUTES over the network.
            Only text whose experts happen to be cached runs at full speed, so
            in practice most prompts are several times slower.

With no flag, setup picks --full if there is room and falls back to --stream
with a warning if there isn't.
"""
import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import tempfile
import urllib.request

try:
    import model_source
except ImportError:  # imported as tools.setup_k3
    from . import model_source
try:
    import inventory_meta
except ImportError:  # imported as tools.setup_k3
    from . import inventory_meta

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "k3-meta")
PKG = os.path.join(ROOT, "tools", "k3pkg")
BASE = model_source.base_url()

FILE_SPECS = model_source.PINNED_FILES
FILES = tuple(FILE_SPECS)
PKG_FILES = ["modeling_kimi_linear.py", "modeling_kimi_k3.py", "configuration_kimi_k3.py"]
HASH_CHUNK = 1 << 20


def _file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as source:
        while block := source.read(HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _validate_regular_file(filename, expected_size, expected_sha256):
    """Return ``(valid, reason)`` without following a path-level symlink."""
    try:
        info = os.lstat(filename)
    except FileNotFoundError:
        return False, "missing"
    if stat.S_ISLNK(info.st_mode):
        return False, "is a symbolic link"
    if not stat.S_ISREG(info.st_mode):
        return False, "is not a regular file"
    if info.st_size != expected_size:
        return (
            False,
            f"has {info.st_size} bytes; expected {expected_size}",
        )
    actual = _file_sha256(filename)
    if actual != expected_sha256:
        return (
            False,
            f"SHA-256 is {actual}; expected {expected_sha256}",
        )
    return True, None


def fetch(name):
    try:
        expected_size, expected_sha256 = FILE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"{name!r} is not in the pinned file manifest") from exc
    dst = os.path.join(META, name)
    valid, reason = _validate_regular_file(
        dst, expected_size, expected_sha256
    )
    if valid:
        return f"  {name}: already present and verified"
    if reason != "missing":
        raise RuntimeError(
            f"refusing to replace {dst}: {reason}. Move that path aside "
            "and retry setup."
        )

    req = urllib.request.Request(BASE + name, headers={"User-Agent": "deltafin-setup"})
    fd, temporary = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".part", dir=META
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with (
            urllib.request.urlopen(req, timeout=120) as response,
            os.fdopen(fd, "wb") as target,
        ):
            fd = -1
            while block := response.read(HASH_CHUNK):
                target.write(block)
                digest.update(block)
                received += len(block)
            target.flush()
            os.fsync(target.fileno())
        actual = digest.hexdigest()
        if received != expected_size:
            raise RuntimeError(
                f"{name} download has {received} bytes; "
                f"expected {expected_size}"
            )
        if actual != expected_sha256:
            raise RuntimeError(
                f"{name} download SHA-256 is {actual}; "
                f"expected {expected_sha256}"
            )

        # Do not silently overwrite a path created while the download was in
        # flight. A matching concurrent install is harmless; anything else is
        # preserved for inspection.
        valid, reason = _validate_regular_file(
            dst, expected_size, expected_sha256
        )
        if valid:
            return f"  {name}: concurrently installed and verified"
        if reason != "missing":
            raise RuntimeError(
                f"refusing to replace {dst}: it appeared during download "
                f"and {reason}"
            )
        os.replace(temporary, dst)
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    valid, reason = _validate_regular_file(
        dst, expected_size, expected_sha256
    )
    if not valid:
        raise RuntimeError(f"published {name} failed verification: {reason}")
    return f"  {name}: downloaded and verified"


def _copy_verified_package_file(name):
    expected_size, expected_sha256 = FILE_SPECS[name]
    source = os.path.join(META, name)
    valid, reason = _validate_regular_file(
        source, expected_size, expected_sha256
    )
    if not valid:
        raise RuntimeError(f"refusing to copy unverified {source}: {reason}")

    destination = os.path.join(PKG, name)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".part", dir=PKG
    )
    try:
        with open(source, "rb") as src, os.fdopen(fd, "wb") as target:
            fd = -1
            shutil.copyfileobj(src, target, length=HASH_CHUNK)
            target.flush()
            os.fsync(target.fileno())
        valid, reason = _validate_regular_file(
            temporary, expected_size, expected_sha256
        )
        if not valid:
            raise RuntimeError(
                f"temporary package copy for {name} is invalid: {reason}"
            )
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def shard_header(i):
    shard = f"model-{i:05d}-of-000096.safetensors"

    def rng(start, end):
        expected = end - start + 1
        req = urllib.request.Request(
            BASE + shard, headers={"Range": f"bytes={start}-{end}",
                                   "User-Agent": "deltafin-setup"})
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = r.read(expected + 1)
        if len(payload) != expected:
            raise RuntimeError(
                f"{shard} range {start}-{end} returned "
                f"{len(payload)} bytes; expected {expected}"
            )
        return payload

    n = struct.unpack("<Q", rng(0, 7))[0]
    if not 0 < n <= inventory_meta.MAX_HEADER_BYTES:
        raise RuntimeError(f"{shard} has invalid header length {n}")
    try:
        h = json.loads(rng(8, 8 + n - 1))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{shard} has an invalid JSON header: {exc}") from exc
    if not isinstance(h, dict):
        raise RuntimeError(f"{shard} header must be a JSON object")
    h.pop("__metadata__", None)
    return shard, n, h


def _fsync_directory(directory):
    """Persist a completed atomic rename where the platform supports it."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _build_inventory():
    print("  building tensor inventory from 96 shard headers "
          "(range requests)...")
    document = {}
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for shard, hlen, header in ex.map(
            shard_header, range(1, inventory_meta.SHARD_COUNT + 1)
        ):
            for name, info in header.items():
                if name in document:
                    raise RuntimeError(
                        f"duplicate tensor {name!r} found in {shard}"
                    )
                if not isinstance(info, dict):
                    raise RuntimeError(
                        f"tensor {name!r} in {shard} has invalid metadata"
                    )
                try:
                    document[name] = {
                        "dtype": info["dtype"],
                        "shape": info["shape"],
                        "offsets": info["data_offsets"],
                        "shard": shard,
                        "hlen": hlen,
                    }
                except KeyError as exc:
                    raise RuntimeError(
                        f"tensor {name!r} in {shard} is missing "
                        f"{exc.args[0]!r}"
                    ) from exc
            sys.stderr.write(".")
    sys.stderr.write("\n")
    inventory_meta.validate_inventory(document)
    return document


def _publish_inventory(inv_path, document):
    """Fsync and atomically publish the exact pinned inventory."""
    inventory_meta.validate_inventory(document)
    fd, temporary = tempfile.mkstemp(
        prefix=".tensor_inventory_offsets.",
        suffix=".part",
        dir=os.path.dirname(inv_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            fd = -1
            # Keep the deterministic serialization used to derive the pinned
            # exact size and SHA-256 in model_source.py.
            json.dump(document, target)
            target.flush()
            os.fsync(target.fileno())

        size, sha256 = inventory_meta.file_identity(temporary)
        if size != model_source.PINNED_INVENTORY_SIZE:
            raise RuntimeError(
                f"generated tensor inventory has {size} bytes; pinned "
                f"revision requires {model_source.PINNED_INVENTORY_SIZE}"
            )
        if sha256 != model_source.PINNED_INVENTORY_SHA256:
            raise RuntimeError(
                f"generated tensor inventory SHA-256 is {sha256}; pinned "
                f"revision requires {model_source.PINNED_INVENTORY_SHA256}"
            )

        # Never overwrite a path that appeared while the 96 headers were
        # being fetched.  A matching concurrent setup is harmless.
        if os.path.lexists(inv_path):
            try:
                inventory_meta.load_verified_inventory(
                    os.path.dirname(inv_path)
                )
            except inventory_meta.InventoryValidationError as exc:
                raise RuntimeError(
                    f"refusing to replace {inv_path}: it appeared while "
                    f"the inventory was being built and is invalid: {exc}"
                ) from exc
            return
        os.replace(temporary, inv_path)
        temporary = ""
        _fsync_directory(os.path.dirname(inv_path))
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    # Verify the published path rather than relying only on the temporary.
    inventory_meta.load_verified_inventory(os.path.dirname(inv_path))


def ensure_inventory():
    """Verify an existing pinned inventory or build it transactionally."""
    inv_path = os.path.join(META, inventory_meta.INVENTORY_FILENAME)
    if os.path.lexists(inv_path):
        try:
            document = inventory_meta.load_verified_inventory(META)
        except inventory_meta.InventoryValidationError as exc:
            raise RuntimeError(
                f"refusing to use or replace {inv_path}: {exc}. Move that "
                "path aside and retry setup."
            ) from exc
        print(
            f"  tensor inventory: {len(document)} tensors, pinned and verified"
        )
        return document

    document = _build_inventory()
    _publish_inventory(inv_path, document)
    print(f"  tensor inventory: {len(document)} tensors indexed and verified")
    return document


SPINE_BYTES = 114e9
EXPERTS_BYTES = 82432 * 17547264      # ~1.45 TB
HEADROOM = 100e9


def _remaining_bytes():
    """What still has to be downloaded, so re-running as an upgrade doesn't
    demand space for weights that are already on disk."""
    ecache = os.path.join(ROOT, "k3-experts")
    try:
        have = sum(1 for f in os.listdir(ecache) if f.endswith((".bin", ".npz")))
    except FileNotFoundError:
        have = 0
    experts_left = max(0, 82432 - have) * 17547264
    spine = os.path.join(ROOT, "k3-resident", "tensors")
    try:
        spine_have = sum(os.path.getsize(os.path.join(spine, f))
                         for f in os.listdir(spine))
    except FileNotFoundError:
        spine_have = 0
    return max(0, SPINE_BYTES - spine_have), experts_left


def _exclude_from_spotlight():
    """macOS indexes new files aggressively. 1.5 TB of weight blobs sends
    corespotlightd to ~50% CPU for hours, which contends with inference and
    makes any benchmark meaningless. A .metadata_never_index marker stops it."""
    if sys.platform != "darwin":
        return
    for d in ("k3-experts", "k3-resident", "k3-resident-int8", "k3-resident-int4"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            open(os.path.join(p, ".metadata_never_index"), "a").close()


def install_weights(mode, install_draft=True):
    import shutil as _sh
    import subprocess
    free = _sh.disk_usage(ROOT).free
    spine_left, experts_left = _remaining_bytes()
    need_full = spine_left + experts_left + HEADROOM
    if mode is None:
        mode = "full" if free >= need_full else "stream"
        if mode == "stream":
            print("\n" + "!" * 72)
            print(f"  Not enough disk for the full install: it needs "
                  f"{need_full/1e12:.2f} TB more (weights + 100 GB headroom)")
            print(f"  and this volume has {free/1e12:.2f} TB free — about "
                  f"{(need_full-free)/1e12:.2f} TB short.")
            print("  Falling back to STREAMING, which is SUBSTANTIALLY SLOWER:")
            print("  every prompt whose experts aren't cached fetches them over")
            print("  the network — minutes per token instead of about one.")
            print(f"  Freeing ~{(need_full-free)/1e12:.2f} TB and re-running with")
            print("  --full is the single biggest speedup available.")
            print("!" * 72 + "\n")
    if mode == "full" and free < need_full:
        print(f"--full still needs {need_full/1e12:.2f} TB (of which "
              f"{experts_left/1e12:.2f} TB is experts), but only {free/1e12:.2f} TB is free.")
        print("Free up space, or use --stream for now and run")
        print("tools/fetch_experts_all.py later.")
        sys.exit(1)

    py = sys.executable
    os.makedirs(os.path.join(ROOT, "k3-experts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "k3-resident"), exist_ok=True)
    _exclude_from_spotlight()
    print(f"\n== installing weights: {mode} ==", flush=True)
    subprocess.check_call([py, os.path.join(ROOT, "tools", "fetch_spine.py")])
    if mode == "full":
        subprocess.check_call([py, os.path.join(ROOT, "tools", "fetch_experts_all.py")])
        print("\nFull install complete — no network needed at inference time.")
    else:
        print("\nStreaming install complete (spine only).")
        print("Reminder: experts are fetched over the network on demand, so most")
        print("prompts will be several times slower than a full install. When you")
        print("have ~1.45 TB free:  python tools/fetch_experts_all.py")
    if install_draft:
        print("\n== installing exact speculative assistant ==", flush=True)
        subprocess.check_call(
            [py, os.path.join(ROOT, "tools", "setup_draft.py")]
        )
    print("Optional next step (halves per-token I/O):")
    print("  python tools/convert_spine_int8.py")


def main():
    ap_mode = None
    if "--full" in sys.argv:
        ap_mode = "full"
    elif "--stream" in sys.argv:
        ap_mode = "stream"
    meta_only = "--meta-only" in sys.argv
    install_draft = "--no-draft" not in sys.argv

    os.makedirs(META, exist_ok=True)
    print(f"Deltafin setup -> {META}")
    for name in FILES:
        print(fetch(name))
    for name in PKG_FILES:
        _copy_verified_package_file(name)
    print("  verified modeling files copied into tools/k3pkg/")

    ensure_inventory()
    if meta_only:
        print("Done (--meta-only). Weights not installed.")
        return
    install_weights(ap_mode, install_draft=install_draft)


if __name__ == "__main__":
    main()
