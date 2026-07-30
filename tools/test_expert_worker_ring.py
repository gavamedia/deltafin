#!/usr/bin/env python3
"""Deterministic, disk-free correctness gate for the opt-in X4 pread ring.

The tests exercise the contracts that are easy to accidentally weaken while
removing one ``Future`` per expert: queued-versus-running cancellation, partial
prefetch hits, legacy fallback, grouped-read cleanup, slot lifetime, failures,
batched statistics, and the unchanged default executor path.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import sys
import threading
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2  # noqa: E402


_EID_RE = re.compile(r"-E(\d+)\.bin$")


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  PASS {name}")


class FakeSlot:
    """Small stand-in for the 17.5 MB aligned slot; paths remain real-shaped."""

    def __init__(self, serial, calls, fail_eids=(), started=None, release=None):
        self.serial = serial
        self.calls = calls
        self.fail_eids = set(fail_eids)
        self.started = started
        self.release = release
        self.raw = None

    def read(self, path):
        match = _EID_RE.search(path)
        if match is None:
            raise AssertionError(f"unexpected expert path {path}")
        eid = int(match.group(1))
        self.calls.append(eid)
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(5):
            raise AssertionError("test failed to release blocking fake pread")
        if eid in self.fail_eids:
            raise OSError(f"synthetic read failure E{eid}")
        self.raw = {"eid": eid, "slot": self.serial}
        return self


def fake_take_factory(calls, fail_eids=()):
    made = []

    def take(n):
        slots = [
            FakeSlot(len(made) + index, calls, fail_eids)
            for index in range(n)
        ]
        made.extend(slots)
        return slots

    return take, made


def test_batch_and_stats():
    print("shared batch and batched statistics")
    before = dict(fetch_v2.stats)
    calls = []
    slots = [FakeSlot(index, calls) for index in range(7)]
    ring = fetch_v2._PreadWorkerRing(3, name="test-k3-ring-batch")
    try:
        handles = ring.submit(
            slots, [f"/fake/L45-E{eid}.bin" for eid in range(7)]
        )
        check("one completion batch", len({id(h._batch) for h in handles}) == 1)
        check(
            "results preserve slot identity",
            [handle.result(timeout=2) for handle in handles] == slots,
        )
        check("every expert runs exactly once", sorted(calls) == list(range(7)))
    finally:
        ring.close()
        ring.close()  # lifecycle cleanup is intentionally idempotent

    delta = {key: fetch_v2.stats[key] - before[key] for key in before}
    check("one ring batch recorded", delta["pread_ring_batches"] == 1)
    check("seven compact jobs recorded", delta["pread_ring_jobs"] == 7)
    check("successful reads counted once", delta["pread_experts"] == 7)
    check(
        "successful bytes counted once",
        delta["pread_bytes"] == 7 * fetch_v2.EXPERT_SPAN,
    )
    check("no synthetic cancellation/failure", not delta["pread_ring_canceled"]
          and not delta["pread_ring_failures"])


def test_two_phase_cancel():
    print("running/queued cancellation and slot lifetime")
    calls = []
    started = threading.Event()
    release = threading.Event()
    first = FakeSlot(0, calls, started=started, release=release)
    slots = [first] + [FakeSlot(index, calls) for index in range(1, 6)]
    ring = fetch_v2._PreadWorkerRing(1, name="test-k3-ring-cancel")
    handles = ring.submit(
        slots, [f"/fake/L46-E{eid}.bin" for eid in range(6)]
    )
    check("first expert entered read", started.wait(2))

    reader = object.__new__(fetch_v2._PreadReader)
    returned = []
    reader._give = lambda values: returned.extend(values)
    jobs = {
        eid: (slot, handle)
        for eid, slot, handle in zip(range(6), slots, handles)
    }
    errors = []

    def drain():
        try:
            reader._drain(jobs)
        except BaseException as exc:  # make thread failures visible to main
            errors.append(exc)

    thread = threading.Thread(target=drain, name="test-k3-ring-drain")
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while (not all(handle.cancelled() for handle in handles[1:])
               and time.monotonic() < deadline):
            time.sleep(0.001)
        check("all queued experts canceled before wait",
              all(handle.cancelled() for handle in handles[1:]))
        check("only the already-running read touched a slot", calls == [0])
    finally:
        release.set()
        thread.join(5)
        ring.close()

    check("drain thread completed", not thread.is_alive())
    check("drain propagated no error", not errors)
    check("all slots returned only after writer stopped", returned == slots)
    try:
        handles[1].result(timeout=0)
    except concurrent.futures.CancelledError:
        pass
    else:
        raise AssertionError("canceled handle must raise CancelledError")
    print("  PASS canceled result reports CancelledError")


def test_prefetch_and_fallback():
    print("partial prefetch, demand completion, and legacy fallback")
    old_has_bin = fetch_v2._has_bin
    old_cache_load = fetch_v2._cache_load
    calls = []
    returned = []
    retired = []
    reader = fetch_v2._PreadReader(use_ring=True, workers=2)
    take, made = fake_take_factory(calls)
    reader._take = take
    reader._give = lambda slots: returned.extend(slots)
    reader._retire = lambda slots: retired.extend(slots)
    before_hits = fetch_v2.stats["prefetch_hits"]
    try:
        fetch_v2._has_bin = lambda layer, eid: eid not in (9, 10)
        fetch_v2._cache_load = (
            lambda layer, eid: {"legacy": eid} if eid == 9 else None
        )
        reader.prefetch(47, [1, 2, 3, 4])
        raw, missing = reader.read_layer(47, [2, 4, 5, 9, 10])
        check("partial-prefetch output is exact",
              {eid: raw[eid]["eid"] for eid in (2, 4, 5)}
              == {2: 2, 4: 4, 5: 5})
        check("legacy npz fallback preserved", raw[9] == {"legacy": 9})
        check("absent fallback remains missing", missing == [10])
        check("prefetch hit accounting exact",
              fetch_v2.stats["prefetch_hits"] - before_hits == 2)
        check("demand slots retained", len(retired) == 3)
        check("unused prefetch slots recycled", len(returned) == 2)
        check("no slot alias in one demand result",
              len({raw[eid]["slot"] for eid in (2, 4, 5)}) == 3)
        check("all allocated slots accounted for",
              sorted(map(id, returned + retired)) == sorted(map(id, made)))
    finally:
        fetch_v2._has_bin = old_has_bin
        fetch_v2._cache_load = old_cache_load
        reader.shutdown()


def test_grouped_order_lifetime_and_failure():
    print("group order, lifetime, and failure cleanup")
    calls = []
    retired = []
    reader = fetch_v2._PreadReader(use_ring=True, workers=3)
    take, _made = fake_take_factory(calls)
    reader._take = take
    reader._retire = lambda slots: retired.extend(slots)
    try:
        iterator = reader.read_layer_groups(48, [7, 3, 8, 2, 6], 2)
        first_ids, first_raw = next(iterator)
        check("first group follows caller priority", first_ids == [7, 3])
        check("first group raw mapping exact",
              {eid: first_raw[eid]["eid"] for eid in first_ids}
              == {7: 7, 3: 3})
        check("yielded slots remain live", not retired)
        rest = list(iterator)
        check("later group boundaries preserved",
              [ids for ids, _raw in rest] == [[8, 2], [6]])
        check("all grouped slots retire after exhaustion", len(retired) == 5)
        check("each grouped expert read exactly once",
              sorted(calls) == [2, 3, 6, 7, 8])
    finally:
        reader.shutdown()

    calls = []
    retired = []
    before_failures = fetch_v2.stats["pread_ring_failures"]
    reader = fetch_v2._PreadReader(use_ring=True, workers=2)
    take, made = fake_take_factory(calls, fail_eids={12})
    reader._take = take
    reader._retire = lambda slots: retired.extend(slots)
    old_raw_file_valid = fetch_v2._raw_file_valid
    fetch_v2._raw_file_valid = lambda layer, eid: layer == 49 and eid == 12
    try:
        try:
            list(reader.read_layer_groups(49, [11, 12, 13, 14], 1))
        except fetch_v2.LocalExpertReadError as exc:
            check("failure names exact expert", "L49-E12" in str(exc))
            check("valid local failure is fail-closed",
                  "refusing HTTP fallback" in str(exc))
        else:
            raise AssertionError(
                "valid local ring failure must fail closed"
            )
        check("all failure-path slots retired", retired == made)
        check("failed ring job recorded",
              fetch_v2.stats["pread_ring_failures"] - before_failures == 1)
        check("later reads either finish or cancel",
              set(calls).issubset({11, 12, 13, 14})
              and {11, 12}.issubset(calls))
    finally:
        fetch_v2._raw_file_valid = old_raw_file_valid
        reader.shutdown()


def test_opt_in_boundary():
    print("opt-in/default boundary")
    legacy = fetch_v2._PreadReader(use_ring=False, workers=1)
    ring = fetch_v2._PreadReader(use_ring=True, workers=1)
    try:
        check("explicit legacy keeps ThreadPoolExecutor",
              legacy._ring is None and legacy._ex is not None)
        check("explicit ring has no fallback executor yet",
              ring._ring is not None and ring._ex is None)
    finally:
        legacy.shutdown()
        ring.shutdown()


def main():
    test_batch_and_stats()
    test_two_phase_cancel()
    test_prefetch_and_fallback()
    test_grouped_order_lifetime_and_failure()
    test_opt_in_boundary()
    print("\nALL EXPERT WORKER-RING TESTS PASSED")


if __name__ == "__main__":
    main()
