#!/usr/bin/env python3
"""Correctness checks and microbenchmarks for X1/X2/X3/X5/X6/X7/X8/X9.

This deliberately never runs a K3 layer or a full generation.
"""
import gc
import json
import os
import random
import re
import tempfile
import time

os.environ.setdefault("K3_DEV", "cpu")
os.environ.setdefault("K3_MOE", "cpu")
os.environ.setdefault("K3_FAST_SPINE", "0")
os.environ.setdefault("K3_SPINE_DEQ", "torch")
os.environ.setdefault("K3_FETCH", "v2")
os.environ.setdefault("K3_TRACE", "off")

import torch

import kimi_run as kr


def best_ms(fn, reps=5):
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        vals.append((time.perf_counter_ns() - t0) / 1e6)
    return min(vals)


def check_cache_report():
    cache = kr.k3loader.ECACHE

    def legacy_rescan():
        names = os.listdir(cache)
        n = len([f for f in names if f.endswith(".npz")])
        total = sum(os.path.getsize(os.path.join(cache, f)) for f in names)
        return n, total

    old_ms = best_ms(legacy_rescan, reps=1)
    report = kr.k3loader.cache_report()
    expected = len(kr.k3loader._cache_experts)
    assert report.startswith(f"expert cache: {expected} experts /")
    new_ms = best_ms(lambda: [kr.k3loader.cache_report() for _ in range(1000)])
    return {
        "legacy_one_call_ms": old_ms,
        "incremental_1000_calls_ms": new_ms,
        "experts": expected,
    }


def check_router_trace():
    weights = torch.linspace(0, 1, 16)
    ids = list(range(16))
    rows = 92 * 20
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for mode in ("off", "buffered", "sync"):
            path = os.path.join(td, mode + ".jsonl")
            trace = kr.RouterTrace(path, mode)
            t0 = time.perf_counter_ns()
            for step in range(20):
                for layer in range(92):
                    trace.record(step, layer, ids, weights)
                trace.end_pass()
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            trace.close()
            if mode == "off":
                assert not os.path.exists(path)
                count = 0
            else:
                with open(path, encoding="utf-8") as f:
                    records = [json.loads(line) for line in f]
                count = len(records)
                assert count == rows
                assert records[0]["ids"] == ids
                assert records[-1]["layer"] == 91
            out[mode + "_ms"] = elapsed
            out[mode + "_rows"] = count
    return out


def check_presence_bitmap():
    fv = kr.fetch_v2
    pattern = re.compile(r"L(\d+)-E(\d+)\.bin$")
    rows = []
    for name in sorted(fv._raw_presence):
        match = pattern.fullmatch(name)
        if match:
            rows.append((int(match.group(1)), int(match.group(2)), name))
        if len(rows) == 16:
            break
    assert len(rows) == 16
    queries = rows * 92

    def legacy_stats():
        return [
            os.stat(os.path.join(fv.ECACHE, name)).st_size == fv.EXPERT_SPAN
            for _, _, name in queries
        ]

    def bitmap():
        return [fv._has_bin(layer, expert)
                for layer, expert, _ in queries]

    expected = legacy_stats()
    assert bitmap() == expected
    old_ms = best_ms(legacy_stats, reps=3)
    new_ms = best_ms(bitmap, reps=20)

    layer, expert, name = rows[0]
    fv._drop_bin(layer, expert)
    assert not fv._has_bin(layer, expert)
    with fv._raw_presence_lock:
        fv._raw_presence.add(name)
    assert fv._has_bin(layer, expert)
    return {
        "queries": len(queries),
        "legacy_stat_ms": old_ms,
        "presence_set_ms": new_ms,
    }


def check_incremental_decode():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.join(kr.ROOT, "k3-meta"), trust_remote_code=True)
    rng = random.Random(7)
    for _ in range(100):
        ids = [rng.randrange(tok.vocab_size) for _ in range(rng.randrange(1, 80))]
        dec = kr.IncrementalTokenDecoder(tok)
        got = "".join(dec.append(t) for t in ids) + dec.finish()
        assert got == tok.decode(ids)

    ids = [rng.randrange(tok.vocab_size) for _ in range(1200)]

    def old_prefix_decode():
        last = ""
        for i in range(1, len(ids) + 1):
            last = tok.decode(ids[:i])
        return last

    def incremental():
        dec = kr.IncrementalTokenDecoder(tok)
        pieces = [dec.append(t) for t in ids]
        pieces.append(dec.finish())
        return "".join(pieces)

    old_ms = best_ms(old_prefix_decode, reps=1)
    new_ms = best_ms(incremental)
    assert incremental() == tok.decode(ids)
    return {"prefix_1200_ms": old_ms, "incremental_1200_ms": new_ms}


def check_persistent_tail():
    kr._TAIL = None
    cached = kr._tail_module()
    assert cached is kr._tail_module()
    for name, parameter in cached.named_parameters():
        expected = kr.k3loader.load_resident(kr.PFX + name).to(
            device=kr.DEV, dtype=kr.DT)
        assert torch.equal(parameter, expected)

    def rebuild():
        obj = kr._new_tail()
        kr.dematerialize(obj)

    rebuild_ms = best_ms(rebuild, reps=5)
    cached_1000_ms = best_ms(
        lambda: [kr._tail_module() for _ in range(1000)], reps=5)
    return {
        "resident_bytes": sum(p.numel() * p.element_size()
                              for p in cached.parameters()),
        "rebuild_ms": rebuild_ms,
        "cached_1000_lookups_ms": cached_1000_ms,
    }


def _old_mask(q_len, past):
    total = q_len + past
    m = torch.zeros(1, 1, total, total, dtype=torch.float32)
    m.masked_fill_(
        torch.triu(torch.ones(total, total, dtype=torch.bool), 1),
        torch.finfo(torch.float32).min,
    )
    return m[:, :, -q_len:, :]


def check_rectangular_mask():
    for past in (0, 1, 7, 31):
        for q_len in (2, 3, 8):
            assert torch.equal(kr.causal_mask(q_len, past, torch.float32),
                               _old_mask(q_len, past))
    q_len, past = 4, 2048
    old_ms = best_ms(lambda: _old_mask(q_len, past), reps=3)
    new_ms = best_ms(lambda: kr.causal_mask(q_len, past, torch.float32), reps=10)
    old_elems = (past + q_len) ** 2
    new_elems = q_len * (past + q_len)
    return {
        "old_ms": old_ms,
        "rectangular_ms": new_ms,
        "allocation_element_ratio": old_elems / new_elems,
    }


def check_embedding_fd():
    rowbytes, nrows = 64, 4096
    payload = bytes(range(256)) * (rowbytes * nrows // 256)
    rng = random.Random(11)
    random_ids = [rng.randrange(nrows) for _ in range(64)]
    adjacent_ids = list(range(1000, 1064)) + [1003, 1003]

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "embed.bin")
        with open(path, "wb") as f:
            f.write(payload)
        obj = kr.LazyEmbed.__new__(kr.LazyEmbed)
        obj.path, obj.meta, obj.rowbytes = path, {}, rowbytes
        obj._source = None
        obj._ensure_source()

        def baseline(ids):
            rows = []
            for tid in ids:
                with open(path, "rb") as f:
                    f.seek(tid * rowbytes)
                    rows.append(f.read(rowbytes))
            return b"".join(rows)

        for ids in (random_ids, adjacent_ids):
            assert obj._local_rows(ids) == baseline(ids)
        old_ms = best_ms(lambda: baseline(random_ids), reps=10)
        fd_ms = best_ms(lambda: obj._local_rows(random_ids), reps=10)
        old_adj_ms = best_ms(lambda: baseline(adjacent_ids), reps=10)
        fd_adj_ms = best_ms(lambda: obj._local_rows(adjacent_ids), reps=10)
        obj.close()
    return {
        "random_open_per_row_ms": old_ms,
        "random_persistent_fd_ms": fd_ms,
        "adjacent_open_per_row_ms": old_adj_ms,
        "adjacent_coalesced_ms": fd_adj_ms,
    }


def check_inference_gc():
    before = gc.isenabled()

    @kr._generation_runtime
    def probe():
        x = torch.ones(1)
        return torch.is_inference_mode_enabled(), gc.isenabled(), x.is_inference()

    inside = probe()
    assert inside == (True, False, True)
    assert gc.isenabled() == before

    def tiny_work(ctx):
        with ctx:
            x = torch.ones(64)
            for _ in range(5000):
                x = x.mul(1.00001).add_(0.00001)
            return x

    no_grad_ms = best_ms(lambda: tiny_work(torch.no_grad()), reps=3)
    inference_ms = best_ms(lambda: tiny_work(torch.inference_mode()), reps=3)
    return {
        "inside_inference_mode": inside[0],
        "inside_gc_enabled": inside[1],
        "no_grad_ms": no_grad_ms,
        "inference_mode_ms": inference_ms,
    }


def main():
    results = {
        "X1_cache_report": check_cache_report(),
        "X2_router_trace": check_router_trace(),
        "X3_presence_bitmap": check_presence_bitmap(),
        "X5_persistent_tail": check_persistent_tail(),
        "X6_incremental_decode": check_incremental_decode(),
        "X7_rectangular_mask": check_rectangular_mask(),
        "X8_embedding_fd": check_embedding_fd(),
        "X9_inference_gc": check_inference_gc(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    print("RUNTIME OVERHEADS: PASS")


if __name__ == "__main__":
    main()
