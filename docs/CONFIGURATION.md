# Configuration

Normal operation needs no environment overrides. The most useful controls are:

| Setting | Default | Meaning |
|---|---|---|
| `--device` / `K3_DEV` | `auto` | `mps`, `cuda`, `cuda:N` or `cpu`, with capability-gated auto selection |
| `--expert-backend` / `K3_MOE` | `auto` | `metal`, `cuda` or `cpu`, with capability-gated auto selection |
| `K3_CUDA_EXPERT_CACHE_GB` | auto | explicit GiB ceiling for the resident CUDA expert cache; the automatic free/5 reserve can leave too little contiguous headroom for transient spine binds on smaller GPUs |
| `--spine` / `K3_SPINE` | `auto` | `auto` and `bf16` mean original weights; `int8` is explicit and non-weight-exact |
| `K3_EXPERT_SCALE4` | `auto` | `auto`, `off` or `require` for complete lossless scale4 sidecars |
| `K3_DSPARK` | `auto` | `auto`, `off` or force-qualified `on`; K3 verification is never bypassed |
| `K3_DSPARK_MAX_CONTEXT` | `8192` | bounded auxiliary draft-state context; full K3 continues above it |
| `K3_UAG_DRAFT` | `auto` | optional Qwen raw-completion policy: `auto`, `off` or `on` |
| `K3_PILOT_GATE` | `on` | adaptive admission for PILOT speculative expert reads: `on` gates each layer's reads on trailing measured recall, `measure` scores without suppressing, `off` restores the ungoverned scheduler (see docs/PILOT-GATE.md) |
| `K3_PILOT_GATE_THRESHOLD` | `0.10` | trailing recall below which a layer's speculative reads stop, in `[0,1)` |
| `K3_PILOT_GATE_WARMUP` | `16` | scored samples per layer before the gate may suppress or redirect |
| `--reasoning-effort` / `K3_REASONING_EFFORT` | template default (`max`) | chat thinking depth: `low`, `high` or `max`; the server's per-request `reasoning_effort` field overrides it |
| `K3_TRACE` / `K3_TRACE_PATH` | `off` | native router trace mode and path; CLI flags are preferred |

The quality guard rejects fewer than 16 experts, non-fp32 target activations and approximation switches. Original BF16 remains the automatic resident authority. Optional paths must validate their device, ABI, shapes, memory and correctness before activation.
