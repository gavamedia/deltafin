#!/usr/bin/env python3
"""Download Kimi-K3's resident (non-routed-expert) tensors via HTTP Range requests.
Writes one raw .bin per tensor under k3-resident/tensors/ (name = tensor name).
Resumable: existing regular files with the right size are skipped."""
import concurrent.futures
import os
import stat
import tempfile
import threading
import time
import urllib.request

try:
    import model_source
except ImportError:  # imported as tools.fetch_spine
    from . import model_source
try:
    import inventory_meta
except ImportError:  # imported as tools.fetch_spine
    from . import inventory_meta

BASE = model_source.base_url()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = inventory_meta.load_verified_inventory(os.path.join(ROOT, "k3-meta"))
OUT = os.path.join(ROOT, "k3-resident/tensors")
os.makedirs(OUT, exist_ok=True)


def _existing_tensor_complete(path, expected_size):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"refusing tensor destination symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(
            f"refusing non-regular tensor destination: {path}"
        )
    return info.st_size == expected_size


work = []
for name, t in INV.items():
    if ".experts." in name:
        continue
    size = t["offsets"][1] - t["offsets"][0]
    path = inventory_meta.safe_tensor_output_path(OUT, name)
    if _existing_tensor_complete(path, size):
        continue
    work.append((name, t, size, path))
work.sort(key=lambda w: (w[1]["shard"], w[1]["offsets"][0]))
total = sum(w[2] for w in work)
print(f"to fetch: {len(work)} tensors, {total/1e9:.1f} GB", flush=True)

done_b = 0
lock = threading.Lock()
t0 = time.time()

def fetch(item):
    global done_b
    name, t, size, path = item
    start = 8 + t["hlen"] + t["offsets"][0]
    req = urllib.request.Request(BASE + t["shard"],
                                 headers={
                                     "Range": f"bytes={start}-{start+size-1}",
                                     "User-Agent": "deltafin-setup",
                                 })
    for attempt in range(6):
        fd = -1
        temporary = ""
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".tensor.",
                suffix=".part",
                dir=OUT,
            )
            received = 0
            with (
                urllib.request.urlopen(req, timeout=120) as response,
                os.fdopen(fd, "wb") as target,
            ):
                fd = -1
                while received < size:
                    chunk = response.read(min(1 << 22, size - received))
                    if not chunk:
                        break
                    target.write(chunk)
                    received += len(chunk)
                if received == size and response.read(1):
                    raise IOError("range response exceeded requested size")
            if received != size:
                raise IOError(
                    f"short read: received {received} of {size} bytes"
                )
            os.replace(temporary, path)
            temporary = ""
            with lock:
                done_b += size
            return
        except Exception as e:
            if attempt == 5:
                raise RuntimeError(
                    f"FAILED {name} after 6 attempts: {e}"
                ) from e
            time.sleep(2 * (attempt + 1))
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

with concurrent.futures.ThreadPoolExecutor(10) as ex:
    futs = [ex.submit(fetch, w) for w in work]
    last = 0
    for i, f in enumerate(concurrent.futures.as_completed(futs)):
        f.result()
        now = time.time()
        if now - last > 30:
            last = now
            mbs = done_b / 1e6 / max(now - t0, 1)
            print(f"progress {done_b/1e9:.1f}/{total/1e9:.1f} GB  {mbs:.0f} MB/s  eta {(total-done_b)/1e6/max(mbs,1)/60:.0f} min", flush=True)
print(f"DONE {done_b/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
