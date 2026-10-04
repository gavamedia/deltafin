# Development reference material

The installed public path is the compiled `deltafin` executable. Historical `tools/*.py` files may remain in the source tree temporarily as test vectors, old benchmark evidence or implementation references while their parity records remain useful. They are not supported launchers, installers, servers, converters or fallbacks, and the native process never imports or executes them. The exact boundary between compiled provider inputs and frozen migration material is recorded in [the tools directory notice](tools/README.md).

## Emulating multi-drive storage

`K3_STORAGE_EMULATE` (diagnostic only) charges every positional read to a
virtual device chosen by the longest matching path prefix. Each device is a
FIFO server with a bandwidth, a per-request latency, an optional
thermal-throttle step and an optional failure point:

```text
K3_STORAGE_EMULATE="int=2.8,lat=0.2@/path/to/model;d1=1.12,lat=0.4,throttle=2:0.3,fail=40@/Volumes/d1"
```

The real read still supplies the bytes; the worker then waits until the
virtual device would have delivered them. Emulated devices can therefore only
be slower than the real one. Keep their total below what the host SSD
sustains, and watch the `late` column in the report: late reads mean the real
SSD, not the emulated topology, set the pace. On a one-SSD host, APFS clones
(`cp -c -R`) of the model tree make free stand-ins for extra drives.

The ignored `home_replay::replay_routes_through_storage` test replays real
routes from `k3-meta/router_trace.jsonl` through the real readers in the
engine's per-layer order: spine one layer ahead, then the routed expert union
as a demand read, plus optional per-layer compute. It is configured with
`DELTAFIN_REPLAY_*` variables (passes, layer stride, verify rows, compute ms,
storage homes, spine homes on/off, worker counts, layout):

```bash
DELTAFIN_REPLAY_STORAGE_HOMES=/Volumes/d1,/Volumes/d2 K3_STORAGE_EMULATE="..." \
  cargo test --release -p deltafin --lib -- --ignored --nocapture replay_routes_through_storage
```

To predict real seconds per token, run every device at a fraction `s` of its
real rate and stretch per-layer compute by `1/s`. Then real time is the
measured pass time × `s` × (92 / replayed layers).


## Measuring EAGLE-3 acceptance offline

`DELTAFIN_EAGLE3_TAP_DUMP=<directory>` (diagnostic only) makes every target
sequence append the completed K3 rows at the EAGLE-3 capture layers
(zero-based 1, 45 and 89, the within-block AttnRes stream) as raw fp32 to
`<directory>/L1.f32`, `L45.f32` and `L89.f32`, and each chunk's row count to
`chunks.txt`. It never changes execution; use an empty directory, since the
files are appended to. Beside a run's `--events-jsonl`, the dump is enough to
replay greedy draft chains offline against what K3 actually emitted, at any
chain length and without spending a K3 verify pass:

```bash
DELTAFIN_EAGLE3_TAP_DUMP=/tmp/taps target/release/deltafin run --chat \
  --prompt "..." --max-new 64 --events-jsonl /tmp/taps/events.jsonl
```

The replay used for the numbers in docs/OPTIMIZATIONS.md is a straight port of
vLLM's drafter kept as a local workbench (`tools/probe_eagle3_acceptance.py`,
untracked like every `tools/probe_*` source under the compiled-only policy).
In-engine acceptance is in every run's `decode_step` events and in the
`dspark` block of `run_end`.
