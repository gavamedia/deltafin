# Configuration

Normal operation needs no environment overrides. The most useful controls are:

| Setting | Default | Meaning |
|---|---|---|
| `--device` / `K3_DEV` | `auto` | `mps`, `cuda`, `cuda:N` or `cpu`, with capability-gated auto selection |
| `--expert-backend` / `K3_MOE` | `auto` | `metal`, `cuda` or `cpu`, with capability-gated auto selection |
| `--spine` / `K3_SPINE` | `auto` | `auto` and `bf16` mean original weights; `int8` is explicit and non-weight-exact |
| `K3_EXPERT_SCALE4` | `auto` | `auto`, `off` or `require` for complete lossless scale4 sidecars |
| `K3_DSPARK` | `auto` | mode of the chat/server proposal slot: `auto` (proposes only while live timing shows a win), `off`, or force-qualified `on`. It applies to whichever drafter fills the slot (EAGLE-3.1 or DSpark); K3 verification is never bypassed |
| `K3_EAGLE3` | `auto` | the EAGLE-3.1 hidden-state chain drafter (`k3-draft-eagle3/`, `lightseekorg/kimi-k3-eagle3.1-mla`). `auto` puts it in the slot instead of DSpark whenever it is installed; `off` leaves the slot to DSpark; `on` refuses to start without it. Admitted by pinned SHA-256; needs about 3.5 GiB of provider memory, against DSpark's 6.8 GiB |
| `K3_DSPARK_MAX_CONTEXT` | `8192` | bounded DSpark draft-state context; full K3 continues above it. EAGLE-3.1 tracks K3's admitted context, up to 131,072 tokens |
| `K3_UAG_DRAFT` | `auto` | optional Qwen raw-completion policy: `auto`, `off` or `on` |
| `K3_SPEC_DEPTH` | `8` | ceiling on draft tokens verified per pass (1-8). The adaptive policy already starts narrow and widens only on evidence, so a lower ceiling just truncates its longer proposals: a 128 GB M4 Max measured `4` at -35% (public issue #25). Leave it unset |
| `K3_PILOT_GATE` | `off` | adaptive admission for PILOT speculative expert reads: `on` gates each layer's reads on trailing measured recall, `measure` scores without suppressing, `off` restores the ungoverned scheduler (see docs/PILOT-GATE.md) |
| `K3_PILOT_GATE_THRESHOLD` | `0.10` | trailing recall below which a layer's speculative reads stop, in `[0,1)` |
| `K3_PILOT_GATE_WARMUP` | `16` | scored samples per layer before the gate may suppress or redirect |
| `--reasoning-effort` / `K3_REASONING_EFFORT` | template default (`max`) | chat thinking depth: `low`, `high` or `max`; the server's per-request `reasoning_effort` field overrides it |
| `K3_TRACE` / `K3_TRACE_PATH` | `off` | native router trace mode and path; CLI flags are preferred |
| `K3_EXPERT_STREAM_NOCACHE` | `auto` | page-cache treatment for streaming expert reads: `auto` purges on the memory-tight macOS reference host and keeps the kernel file cache warm elsewhere; `1`/`0` force either behavior |
| `K3_EXPERT_HEAT` | `on` | persistent expert-heat histogram (`k3-meta/expert_heat.v1.bin`) accumulated from authoritative routes during ordinary runs; advisory, never affects routing |
| `K3_EXPERT_PIN_GB` | `auto` | budget for the permanent learned-expert RAM tier on CPU/Metal. `auto` sizes it at startup to exactly the qualifying histogram roster, capped by live free memory minus an 8 GiB headroom — off while history is thin (< 256 passes) or memory is tight. A decimal-GB value is an explicit ceiling; `0` forces it off. The budget is a ceiling, never a reservation: the tier is admitted only after the Qwen drafter and the resident spine are planned, so it can never push out the drafter, a spine that fits completely, or a prefix requested with `K3_PROVIDER_RESIDENT_LAYERS`. A spine that only partly fits may give way to pins (measured +15–19% against +0.23% for the same bytes as spine). A clamped roster is reported at startup. Candidates are promoted only when a route naturally reads them |
| `K3_STORAGE_HOMES` | none | extra drives holding byte-identical copies of model files, as comma-separated `PATH[@GBPS]` entries. Each path is laid out like the model root: `k3-resident-int8/tensors/` and/or `k3-experts/`, any subset (fill one with `deltafin populate-storage-home`). Every read job goes to the drive holding a copy whose queue will finish it soonest, using rates measured from completed reads (`@GBPS` and `primary@GBPS` only seed them). A drive that fails a read is skipped with back-off and the read is retried on another copy. A missing drive is skipped at startup. Only exact-length regular copies no older than their primary are used. Changes where bytes come from, never which bytes |
| `K3_EXPERT_EARLY_DRAIN` | `off` | start each missing expert's matmul as its own bytes land instead of after the layer's whole miss set has been read (single-position Metal decode). Scheduling only — the router still selects the experts, their fp32 weights, and the order they are reduced in, so output cannot depend on disk timing. Off by default because it measured as a clean null on the reference host; see docs/OPTIMIZATIONS.md before turning it on |

The quality guard rejects fewer than 16 experts, non-fp32 target activations and approximation switches. Original BF16 remains the automatic resident authority. Optional paths must validate their device, ABI, shapes, memory and correctness before activation.
