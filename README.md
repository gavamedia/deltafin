<div align="center">

```
   ____       _ _         __ _
        |  _ \  ___| | |_ __ _ / _(_)_ __
        | | | |/ _ \ | __/ _` | |_| | '_ \
        | |_| |  __/ | || (_| |  _| | | | |
        |____/ \___|_|\__\__,_|_| |_|_| |_|
```

### An experiment in running [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) (2.8T parameters) on one local workstation

Deltafin is a small research project that runs a Mixture-of-Experts model far
larger than the machine it sits on. It supports Apple Silicon macOS,
x86-64/aarch64 Linux and x86-64 Windows, automatically selecting MPS, CUDA or
CPU for the resident model path and a qualified Metal, CUDA or CPU MXFP4 expert
kernel. The
maintainer-run
reference is **0.0687 token/s (14.6 seconds/token)** on a modest 64 GB M1 Max;
that is one first-generation machine, not a ceiling for newer hardware.

The README stays focused on installing and running the project. For the
mechanisms, measurements, and fallback rules behind the speed work, see
[How Deltafin's optimizations work](OPTIMIZATIONS.md).

![model](https://img.shields.io/badge/model-Kimi_K3_·_2.8T_MoE-blueviolet)
![hardware](https://img.shields.io/badge/measured_on-M1_Max_·_64GB-silver)
![speed](https://img.shields.io/badge/decode-0.0687_tok%2Fs_·_14.6s%2Ftoken_(M1_Max)-orange)
![platforms](https://img.shields.io/badge/platforms-macOS_arm64_·_Linux_x86--64%2Faarch64_·_Windows_x86--64-informational)
![accelerators](https://img.shields.io/badge/accelerators-MPS_·_CUDA_·_CPU-9cf)
![precision](https://img.shields.io/badge/experts-MXFP4_native-teal)
![mode](https://img.shields.io/badge/decoding-greedy_·_reproducible-green)
![license](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## 1. Install or upgrade

### New installation

Clone the repository and enter it first; every path below is relative to the
Deltafin folder. The only real decision is step 3.

```bash
git clone https://github.com/gavamedia/deltafin.git
cd deltafin

# 1. environment (Python 3.12+)
python3 -m venv venv
./venv/bin/python -m pip install torch
./venv/bin/python -m pip install -r requirements.txt

# 2. build and validate the native libraries for this host
./venv/bin/python tools/build_native.py

# 3. download the model  (see the two modes below)
./venv/bin/python tools/setup_k3.py --full
```

On macOS, install Xcode Command Line Tools first (`xcode-select --install`). On
Linux, install a C/C++ compiler toolchain such as `build-essential` or GCC/Clang.
On Windows, install the Visual Studio "Desktop development with C++" workload;
the native build locates it with `vswhere` and captures the environment
`vcvars64.bat` sets, so `python tools\build_native.py` works from an ordinary
shell rather than only from a Developer Command Prompt.
For an NVIDIA system, install a CUDA-enabled PyTorch build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing the remaining Python packages. Deltafin does not vendor PyTorch or a
CUDA runtime. On Linux and Windows, the native build also detects NVCC when it
is available.
Its CUDA artifact is optional in the default `--cuda=auto` mode: a missing or
incompatible toolkit cannot block the CPU libraries. Use `--cuda=on` to require
and diagnose that artifact, or `--cuda=off` to skip the probe explicitly. When
an NVIDIA driver exposes GPU compute capabilities that the installed NVCC
supports, the build adds native images for those GPUs while retaining portable
PTX. `K3_CUDA_ARCH=sm_89` (or a comma-separated list) is the strict manual
override for cross-builds.

### Upgrade without downloading the model again

Once this release is installed, upgrading is one command from the Deltafin
folder:

```bash
./venv/bin/python tools/upgrade.py
```

The upgrader is deliberately conservative. It refuses a dirty or diverged Git
checkout, fetches the configured upstream, accepts only a fast-forward, keeps
the host-specific PyTorch build unchanged, refreshes the other dependencies,
and rebuilds all native libraries transactionally. It never runs model setup,
conversion, `git clean`, or `git reset`; downloaded models and caches remain
where they are.

#### One-time upgrade from an older release

Older copies do not contain `tools/upgrade.py` yet. From the existing Deltafin
folder, first inspect the checkout:

```bash
git status --short
```

If that prints nothing, bootstrap the new upgrader and then let it finish the
environment and native-library update:

```bash
git pull --ff-only
./venv/bin/python tools/upgrade.py
```

If `git status --short` prints any paths, stop there rather than discarding or
stashing them blindly. Commit or move work you want to keep, then retry. An
upgrade never requires `setup_k3.py`, another 1.7 TB download, or manual
one-library `clang` commands.

### The two modes

| | `--full` (recommended) | `--stream` |
|---|---|---|
| Disk needed | **~1.7 TB** | **~215 GB** |
| Download time | 5–10 hours, resumable | ~30 minutes |
| Speed afterwards | **14.6 s/token median** on our M1 Max | ~3+ min/token for anything not already cached |
| Network at inference | none | constant |

Every token reads 16 experts × 92 layers = **25.8 GB of expert data**. From local
disk that's about 4 seconds; over the network it's minutes. That single fact is
the whole difference between the two columns.

Run `setup_k3.py` with no flag and it picks `--full` when the disk allows,
otherwise falls back to streaming and tells you exactly how much space you'd
need to free.

### Starting with streaming and upgrading later

Streaming is a fine way to try Deltafin without committing 1.7 TB. Whenever you
want the speed, one command finishes the job — no reinstall, no reconfiguration,
and it picks up whatever is already cached:

```bash
./venv/bin/python tools/fetch_experts_all.py          # resumable, run anytime
./venv/bin/python tools/fetch_experts_all.py --dry-run   # just show the numbers
./venv/bin/python tools/fetch_experts_all.py --layers 1-40   # partial is fine too
```

For a streaming install, an idle-time warmer can rank absent experts from
recorded router traces. Its default is a read-only plan; network fetching is
explicit, and it can atomically convert legacy `.npz` entries to the raw fast
format:

```bash
./venv/bin/python tools/warm_expert_cache.py
./venv/bin/python tools/warm_expert_cache.py --convert-npz --fetch 128
```

Deltafin prints a reminder at startup — both for the CLI and the API server —
whenever it's still in streaming mode, showing how much of the pool is local and
what finishing would cost.

### Optional: int8 spine

Halves per-token I/O for the non-expert weights, with no meaningful quality
change in our checks. Takes a few minutes:

```bash
./venv/bin/python tools/convert_spine_int8.py
```

## 2. Use from the command line

```bash
# ask a question; generates until the model finishes its answer
./venv/bin/python tools/kimi_run.py --chat --prompt "What are the three largest moons of Saturn?"

# raw completion (no chat template); runs until you press Ctrl-C, or cap it
./venv/bin/python tools/kimi_run.py --prompt "The capital of France is" --max-new 16
```

Tokens print as they are generated, so you always see the text as it comes.
Ctrl-C stops cleanly at any point and prints the result so far; `--max-new N`
caps the length. One honest warning: K3 thinks before it answers, and at roughly
4.1 tokens per minute a full chat answer can take a while — watching it stream
is part of the experience.

Set `K3_TRACE=buffered` to log router selections to `router_trace.jsonl` for
offline study. Performance runs leave tracing off.

## 3. Use through an OpenAI-compatible server

Deltafin can serve the standard OpenAI API, so chat interfaces, the `openai` SDK
and coding agents can use it by changing a base URL:

```bash
./venv/bin/python tools/serve_openai.py --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "deltafin-kimi-k3",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="deltafin-kimi-k3",
    messages=[{"role": "user", "content": "Hello!"}])

print(r.choices[0].message.content)            # the answer
print(r.choices[0].message.reasoning_content)  # K3's thinking, when present
```

`/v1/chat/completions`, `/v1/completions` and `/v1/models` are implemented, and
streaming (`"stream": true`) works. Most tools that read `OPENAI_BASE_URL` and
`OPENAI_API_KEY` will work by pointing those at the server.

Please read these caveats before pointing anything automated at it:

- **Time.** Answers arrive when they arrive — set your client's timeouts to
  hours, not seconds. Omitting `max_tokens` lets the model finish its answer
  (recommended); raw completions, which never end on their own, default to 256.
  Operators can set a hard ceiling with `K3_SERVER_MAX_TOKENS`.
- **Streaming installs are much slower here.** A chat-template prompt is 60
  tokens or more and prefill touches many experts per layer, so on a partly
  filled cache a chat request can spend hours fetching. With a full install it's
  just normal (slow) inference. The server prints a warning at startup when
  you're in streaming mode.
- **Greedy only.** `temperature` and `top_p` are accepted and ignored, and one
  request runs at a time (a second concurrent request gets a 429).
- **Agents are a curiosity, not a workflow.** Coding assistants work in
  principle, but their long system prompts make prefill expensive.

---

## Reference and technical details

The three sections above cover the normal path: install or upgrade Deltafin,
run it directly, or expose it through the compatible server. The remaining
sections collect configuration, implementation, platform and benchmark detail.

### Configuration

Everything works with no configuration: Deltafin picks a preferred available
device and the int8 spine when it has been built, and says what it chose at
startup. These variables exist for overriding that:

| Variable | Default | Meaning |
|---|---|---|
| `K3_DEV` | auto | `mps` when available, then the first CUDA device visible to the process, otherwise `cpu`; accepts explicit `mps`, `cuda`, `cuda:N` or `cpu` |
| `K3_MOE` | auto | `metal` on MPS; a native `cuda` candidate on CUDA; otherwise `cpu`. Optional GPU backends must pass their own ABI, shape and correctness checks or fall back to CPU |
| `K3_GEMV_LIB` / `K3_BATCH_LIB` | platform default | override the native MXFP4 library paths (`.dylib` on macOS, `.so` on Linux, `.dll` on Windows) |
| `K3_CUDA_MOE_LIB` | `tools/libcudamoe.so` (`.dll` on Windows) | override the optional native CUDA library path |
| `K3_ASSUME_FMA3` | `0` | Windows only: assert FMA3 on a CPU that has it without AVX2, which Windows cannot report |
| `K3_CUDA_ARCH` | auto | add compiler-supported native CUDA images detected from installed GPUs while retaining a portable PTX fallback; accepts strict values such as `sm_89` or `sm_89,sm_120` for cross-builds |
| `K3_CUDA_EXPERT_CACHE` | `auto` | maximum resident GPU experts; `auto` budgets from free VRAM at the first real route, after model allocations, while `0` keeps CUDA compute but disables residency |
| `K3_CUDA_EXPERT_CACHE_RESERVE` | `auto` | VRAM kept outside the expert cache (`auto` is at least 2 GiB and 20% of free VRAM); accepts byte units or a percentage |
| `K3_CUDA_INT8_DEQUANT` | `auto` | qualified one-pass CUDA dequant for int8 resident-spine tensors; `off` retains the selected PyTorch expression |
| `K3_SPINE` | auto | `int8` when built (recommended), else `bf16` |
| `K3_INT8_LM_HEAD` | `auto` | packed row-int8 output head on a qualified native backend; automatic on MPS, forceable elsewhere, with the first real head call guarded by a dense fallback |
| `K3_INT8_KDA_QKV` | `0` | experimental packed row-int8 bundle for KDA Q/K/V/G/F-A/B; one real MPS streamed-weight sequence has passed exactly, while every backend remains capability-gated with a dense fallback |
| `K3_KDA_RECUR` | `device` | keep the KDA recurrence on the selected CPU, CUDA or MPS device; `cpu` preserves the historical MPS-to-CPU experiment and legacy `mps` is accepted as an alias for `device` |
| `K3_MLA_QUERY_ALIAS` | `0` | opt-in T=1 eager-MLA zero-copy query view; source, provider, inference and layout gates retain the established path on any mismatch; removes a redundant allocation, with no end-to-end speed claim yet ([details](OPTIMIZATIONS.md#zero-copy-mla-query-view-at-single-token-decode)) |
| `K3_SHORTCONV` | `auto` | use the portable decode-time four-tap multiply/sum kernel on MPS, CUDA and CPU; `conv1d` remains forceable for backend retesting |
| `K3_SPEC` | `1` | n-gram speculation (lossless) |
| `K3_TEMPLATES` | `1` | template-layer buffer reuse |
| `K3_PRELOAD` / `K3_PREFETCH` | `1` | background layer loading / expert prefetch |
| `K3_GEMV_THREADS` | auto | native CPU MXFP4 workers: up to 4 effective CPUs on macOS and 8 on Linux and Windows, respecting affinity and cgroup quota; override to retune a specific host |
| `K3_METAL_POSITION_BATCH` | `0` | MPS/Metal-specific exact opt-in T>1 position-major MoE; measured +2.0% on accepted speculative passes and should be retuned per Mac |
| `K3_MOE_TOP_K` | `16` | explicit quality/speed dial; fewer routed experts reduce expert bytes and can change output |
| `K3_CPU_MOE_BATCH` | `auto` | exact persistent CPU MXFP4 worker ring; padded counters measured +3.6% at eight threads |
| `K3_ASYNC_CACHE_WRITE` | `0` | opt-in cache-miss write overlap; `K3_CACHE_WRITE_QUEUE` (4) bounds outstanding buffers and `K3_CACHE_WRITE_WORKERS` (1) is retunable per host |
| `K3_APPROX` | `0` | fp16 numerics; not reproducible at near-ties |
| `K3_RAM_GB` / `K3_PIN_LAYERS` | auto | override the RAM budget |
| `K3_PROFILE` | `0` | per-phase timing for each pass |
| `K3_TRACE` | `off` | `buffered` writes one router-trace block per pass; `sync` writes each layer immediately |
| `DELTAFIN_ROOT` | repo root | where caches and weights live |
| `K3_HF_HOST` / `K3_HF_PATH` | Hugging Face | point expert fetching at a mirror |
| `K3_SERVER_MAX_TOKENS` | unlimited | optional hard ceiling on server generations |
| `K3_RESPONSE_MEMO_ENTRIES` | `32` | exact in-process replay cache for identical deterministic API requests; `0` disables |

Less commonly needed experimental controls, including packed-KDA storage and
CPU padded-SDPA qualification, are documented with their evidence and fallback
contracts in [the optimization guide](OPTIMIZATIONS.md). They are intentionally
not promoted as general defaults here.

### Requirements

- Apple Silicon macOS, x86-64/aarch64 Linux, or x86-64 Windows. x86-64 hosts
  require AVX, FMA3, SSE3 and SSSE3; AVX2 is detected and selected at runtime.
  Windows publishes no FMA3 query, so it is inferred from AVX2 there; set
  `K3_ASSUME_FMA3=1` for the rare CPU with FMA3 but no AVX2.
- A C/C++ compiler: Xcode Command Line Tools on macOS, GCC/Clang on Linux, or
  the Visual Studio C++ build tools on Windows.
  Building the optional native CUDA expert path also requires NVCC; inference
  retains the native CPU expert path when it is absent.
- Python 3.12 or newer.
- PyTorch for the selected device. NVIDIA acceleration requires a CUDA-enabled
  PyTorch build; CPU-only Linux remains supported.
- Disk: ~1.7 TB for the full install, ~215 GB for streaming (see
  [Install or upgrade](#1-install-or-upgrade)).
- Network access to Hugging Face.

Host RAM and container/cgroup limits are budgeted automatically. Accelerator
headroom is also checked when private resident-layer pinning is enabled. CUDA
auto-selection uses the first device visible to the process; `K3_DEV=cuda:N`
remains authoritative. More memory lets Deltafin retain more of the resident
spine and expert page cache. See
[Why newer hardware should be faster](#why-newer-hardware-should-be-faster).

### How it works

K3's weights total about 1.56 TB, which is more than most single workstations
can hold in RAM. The observation that makes local inference possible anyway is
that a Mixture-of-Experts model only *touches* a small fraction of itself per
token.

- **The resident spine** (~114 GB: attention, shared experts, latent projections,
  embeddings) is downloaded once and read layer-by-layer from local storage each
  token, quantized to int8 and computed on the selected MPS, CUDA or CPU device.
- **The 82,432 routed experts** (~1.45 TB). For each token K3's router picks 16
  experts per layer, and only those are read. Install them all locally if you can
  (recommended); otherwise Deltafin fetches them from Hugging Face on demand —
  one HTTP range request per expert — into a growing disk cache.
- **The forward pass** runs Moonshot's own modeling code, unmodified. A small
  pure-PyTorch shim stands in for the CUDA-only `fla` kernels it expects.

```mermaid
flowchart LR
    subgraph HF["Hugging Face CDN"]
        W[("96 safetensors shards<br/>1.56 TB · MXFP4")]
    end
    subgraph HOST["Local workstation (macOS or Linux)"]
        subgraph DISK["Local SSD / NVMe"]
            SP[("resident spine<br/>114 GB bf16 → 60 GB int8")]
            EC[("expert cache<br/>raw shard spans")]
        end
        subgraph TOK["per token"]
            R{"router<br/>top-16 of 896<br/>× 92 layers"}
            L["93 decoder layers<br/>MPS / CUDA / CPU"]
            K["fused MXFP4 MoE<br/>Metal / CUDA / CPU SIMD"]
        end
    end
    W -- "one range request<br/>per missing expert" --> EC
    SP -- "double-buffered<br/>layer loader" --> L
    EC -- "mmap" --> K
    R -- "selected experts" --> K
    K --> L
    L -- "logits" --> R
```

### What to expect

The maintainer reference below was measured on **one modest M1 Max (10-core CPU,
32-core GPU, 64 GB, internal NVMe)** with the full model installed locally,
int8 resident weights and output head, Metal MoE, exact fp32 numerics, greedy
decoding, and tracing disabled. It is a transparent baseline from an early
Apple Silicon machine, not a claim that every supported host performs alike.

The current column pools **six exact full-model runs** from balanced ABBA/BAAB
campaigns. Each run used the five-token prompt `The capital of France is`,
verified the three-token completion ` Paris. The`, and discarded the first
decode step before reporting steady throughput. Values are medians; the range
shows how much this I/O-heavy workload moved even on the same quiet machine.

| Metric | First working version | Current M1 Max benchmark | Change |
|---|---|---|---|
| Prefill / first token (5-token prompt) | 2,429 s | **28.0 s median** (24.9–37.9 s) | ~87× |
| Steady decode, experts local | ~20 min/token | **0.0687 token/s** (**14.6 s/token**); 0.0503–0.0779 token/s run range | ~82× |
| Exact three-token generation, model time | — | **56.5 s median** | — |
| Fresh-process wall time for the same run | — | **64.1 s median** | — |
| Decode, experts streamed | ~20 min/token | ~3 min/token | network-bound |

> This is the “measly M1” result: an aging first-generation M1 Max, not a newer
> Max or Ultra. It is a conservative reference point, not a cross-Mac benchmark.
> We expect newer, higher-bandwidth and higher-RAM systems to do better, but will
> label those numbers separately when someone measures them.

#### Community Linux + CUDA result

[Maurice Brown (`trumb`)](https://github.com/trumb) reported an end-to-end run
on an NVIDIA DGX Spark (GB10 Grace-Blackwell, 20-core Cortex-X925, 128 GB unified
LPDDR5X) using Ubuntu 24.04 and GCC 13.3. With no device environment variables
set, Deltafin selected CUDA plus the int8 spine and completed
`The capital of France is` → ` Paris. The Eiffel Tower is located` in **213
seconds** in that reported single run (139.6 s compute, 51.9 s expert fetch,
40.8 s MoE kernel, 37.4 s preload wait). This is a community measurement from
[pull request #2](https://github.com/gavamedia/deltafin/pull/2), not a
maintainer-replicated benchmark. It used the hybrid CUDA-resident/CPU-expert
path and therefore is not a benchmark of the newer native CUDA expert kernel.

The contributor also reported this four-configuration comparison on the same
system; all four runs produced the same expected output:

| Configuration | Total | Compute | Resident I/O | Preload wait |
|---|---:|---:|---:|---:|
| bf16 + CPU | 865 s | 299 s | 53 s | 504 s |
| bf16 + CUDA | 858 s | 160 s | 20 s | 676 s |
| int8 + CPU | 649 s | 327 s | 311 s | 2 s |
| **int8 + CUDA** | **221 s** | **184 s** | **17 s** | **18 s** |

On this 128 GB unified-memory machine, int8 was worth roughly **4× end to end**,
not merely the 2× suggested by halving spine bytes: the 107 GB bf16 spine left
almost no room for the expert page cache, while the roughly 53 GB int8 spine
freed enough cache headroom for preload wait to collapse. That result is a good
reminder that quantization can change the I/O regime, not just arithmetic cost.

#### Recent exact-path improvements

The newest measurements below are balanced A/Bs on the same M1 Max. Token
oracles were checked for every full-model run.

| Change | Measured result | Shipping behavior |
|---|---|---|
| Packed MPS int8 output head | **+17.3%** median steady decode, **+23.1%** prefill, and **+26.8%** wall throughput; resident head storage fell from 4.7 GB to 1.17 GB | enabled when the operator and int8 weights are available, with an exception-guarded dense fallback |
| Reference-only speculative snapshots | 0.001 ms instead of 3.56 ms and no ~475 MB state clone | enabled by default; replay and partial-accept tests preserve the exact future sequence |
| Position-major Metal MoE for accepted drafts | **+4.7%** on a real T=2 layer and **+2.0%** pooled full-model throughput | exact opt-in via `K3_METAL_POSITION_BATCH=1`, pending per-Mac tuning |
| 128-byte-aligned CPU worker counters | **+0.2%** at four threads and **+3.6%** at eight threads | automatic in the persistent CPU fallback |

The median works out to roughly **4.1 tokens per minute**. A representative M1
Max profile is dominated by:

| | |
|---|---|
| waiting on the resident spine read (53 GB) | ~5 s |
| reading the 16 selected experts per layer (25.8 GB) | ~4.3 s |
| applying the spine (transfer + dequant) | ~3 s |
| attention and norms (93 layers) | ~2 s |
| MoE expert matmuls | ~1 s |

Decode is now **bound by disk bandwidth on the resident spine**. Those 53 GB are
re-read every token, and at the ~7 GB/s this access pattern sustains that is about
7.5 s of the 14.6-second median—unavoidable without either more RAM (enough to
hold the spine without displacing the page cache the expert reads need) or a
smaller spine.

#### Why newer hardware should be faster

The maintainer numbers above come from an M1 Max, but the dominant costs vary
with memory, storage, CPU SIMD and accelerator capability rather than a
hard-coded chip name:

- **Memory bandwidth.** Faster unified or system memory lands directly on spine
  loading and expert matmuls. Newer Apple Max/Ultra chips and high-bandwidth
  Linux workstations should have more room than the M1 reference.
- **Accelerator.** More capable Apple GPUs execute the MPS/Metal resident path
  faster. On NVIDIA Linux, CUDA accelerates the resident spine and attention;
  the routed MXFP4 MoE also uses the native CUDA path after its ABI and
  on-device correctness tests pass, with the native CPU kernel as fallback.
- **SSD.** Expert reads are one of the largest slices and run at the throughput
  the local NVMe and filesystem can sustain.
- **RAM.** This often matters most. The 53 GB int8 spine does not fit alongside
  everything else on a 64 GB host, so much of it is re-read every token. With
  more headroom it can remain in page cache, expert preload waits fall, and
  Deltafin pins more layers automatically.
- **CPU ISA and cores.** aarch64 uses NEON. The x86-64 fat binary preflights
  its AVX/FMA3/SSSE3 baseline and selects its bit-exact AVX2/FMA kernel at
  runtime when available.

Runtime selection checks MPS/CUDA availability, native CPU capabilities, host
and cgroup memory limits, and optional operator support rather than matching
marketing names. That keeps newer Apple families—including M5—and new Linux
hosts eligible while optional paths retain safe fallbacks.

**If you try Deltafin on a newer Mac, an NVIDIA Linux workstation, an Ultra, or
a machine with 128 GB or more, we would genuinely like to see your
numbers**—open an issue with the output of `K3_PROFILE=1`, your OS and hardware.

When n-gram speculation accepts a draft, one forward pass emits two tokens, so
repetitive text runs proportionally faster. Speculation is lossless: accepted
drafts reproduce the reference sequence exactly, and a rejected draft restores
the model state bit-for-bit.

The gap between the two decode rows is the whole argument for the full install
([see above](#the-two-modes)): with the experts local, every prompt runs at the
top-row speed instead of only the ones whose experts happen to be cached.

Output is greedy and reproducible: the same prompt yields the same tokens, run
after run.

> `The capital of France is` → ` Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also in Paris. The Louvre has…`

To be clear about the limitations: this is a research artifact, not a practical
chat setup. A 14.6-second median token is a long way from interactive, and long
prompts are expensive because prefill touches many experts. We think it is
interesting mainly as an existence proof, and as a testbed for
streaming-inference techniques.

### Performance design

Deltafin gets most of its speed by moving fewer bytes, overlapping unavoidable
reads with compute, reusing allocations and state, and selecting native kernels
from observed runtime capabilities. The public README intentionally keeps that
description broad.

The detailed guide explains each retained optimization separately—including
the unusual state-snapshot, packed-projection, page-cache, buffer-ownership and
CPU FlashAttention paths—along with its measurements, exactness rules, platform
scope, and fallback:

**[Read: How Deltafin's optimizations work](OPTIMIZATIONS.md)**

The short version is:

- immutable model data stays packed until the operation that consumes it;
- streamed layers and routed experts use separate, overlapping data paths;
- temporary tensors, routing records and cache state are reused when ownership
  makes that safe;
- optional native paths qualify themselves by capability and correctness
  checks, then fall back without changing the model data;
- a structural byte or allocation saving is not reported as a tokens/second
  win until a balanced full-model measurement supports it.

### Supported platforms

| Host | Resident model / attention | Routed-expert MoE |
|---|---|---|
| Apple Silicon macOS | MPS | Metal, with a native CPU alternative |
| NVIDIA Linux, x86-64 or aarch64 | CUDA | qualified native CUDA MXFP4, with native CPU fallback |
| Linux x86-64 (AVX/FMA3/SSSE3 baseline) | CPU | runtime AVX2/FMA or exact 128-bit compatibility MXFP4 |
| Linux aarch64 | CPU | native NEON MXFP4 |
| NVIDIA Windows x86-64 | CUDA | qualified native CUDA MXFP4, with native CPU fallback |
| Windows x86-64 (AVX/FMA3/SSSE3 baseline) | CPU | runtime AVX2/FMA or exact 128-bit compatibility MXFP4 |

The NVIDIA path qualifies the optional CUDA expert library with a versioned
ABI, fixed shape/layout handshake and small on-device known-answer tests before
the first real expert call. Failure leaves the resident CUDA model path intact
and selects the native CPU MXFP4 fallback. `build_native.py` selects `.dylib`,
`.so` or `.dll`, applies the appropriate host ISA flags, validates required
symbols and ABI, and only then installs the artifacts. Optional CUDA build
failure cannot block installation of the required CPU libraries.
The x86 build is one fat binary: it requires the instructions used by the exact
AVX/FMA3/SSSE3 baseline, detects AVX2 once at runtime, and calls the
256-bit kernel only when the CPU and OS support it. GCC and Clang isolate that
kernel with a target attribute and `-mavx` sets the baseline; MSVC accepts the
intrinsics regardless of `/arch` and uses `/arch:AVX` for the same baseline.
The Windows kernels reach POSIX threading through `tools/win_compat.h`, a narrow
shim over SRW locks, condition variables and `_beginthreadex`.

Windows support currently covers the native build and the MXFP4 kernels. The
read path needs one more change, because CPython provides neither `os.preadv`
nor `os.pread` there:

- The resident spine is unaffected in its default configuration, which reads
  with `open()` and `readinto()` on a per-thread handle. Only the opt-in
  `spine_io` tuning flags (`K3_SPINE_FDCACHE`, `K3_SPINE_CHUNK_MB`,
  `K3_SPINE_RDADVISE`, `K3_SPINE_NOCACHE`) take the positional-read path.
- The expert reader defaults to `K3_EXPERT_READ=pread`. Set
  `K3_EXPERT_READ=mmap` on Windows; that path demand-faults `np.memmap` views
  instead, measured at 0.87-0.93 GB/s against 7.0 GB/s for `pread` on the
  reference machine.
- `LazyEmbed` in `kimi_run.py` reads embedding rows with `os.pread` and has no
  local alternative. This is the one remaining blocker for an end-to-end
  Windows run.

Restoring full expert-read throughput needs a Windows positional-read layer,
most directly `ReadFile` with an `OVERLAPPED` offset on handles opened
`FILE_FLAG_OVERLAPPED`, which is the native equivalent of `preadv` and keeps
the concurrent reads the current design depends on.

Validation is intentionally separated from platform support:

| Configuration | Current evidence |
|---|---|
| Apple Silicon / MPS + Metal | maintainer-run end-to-end benchmarks and exact-token gates |
| Linux aarch64 / CUDA + CPU MoE | community end-to-end run and contributor kernel tests on DGX Spark |
| Linux x86-64 / CPU | strict fat-binary compilation plus selected/compatibility exactness under Rosetta; native Linux AVX2 timing is still wanted |
| Windows x86-64 / CPU | MSVC build plus the selected/compatibility and scale-LUT exactness gates run natively; the embedding reader still needs `os.pread`, so no end-to-end run exists yet |
| Native CUDA MXFP4 MoE | implementation is ABI/KAT/fallback-gated; CUDA hardware parity and throughput measurements are still wanted |

### Where this could go

Roughly in order: hardware-wide CUDA expert-kernel tuning, further Metal
expert-kernel tuning, a proper quality harness — average NLL
against the official API — so lossy speed/quality trade-offs can be measured
rather than argued about, smarter expert prefetching, and eventually a native
engine in the spirit of ds4, where most remaining overhead should disappear.

### Thanks

Deltafin leans heavily on work that others published openly. In rough order of
influence:

- **[Maurice Brown (`trumb`)](https://github.com/trumb)** — contributed the
  Linux x86-64/aarch64 and NVIDIA compatibility findings, AVX2 kernel design,
  native-kernel correctness checks, platform-specific build settings and DGX
  Spark measurements in
  **[pull request #2](https://github.com/gavamedia/deltafin/pull/2)**. Those
  findings informed Deltafin's guarded cross-platform implementation.
- **[colibri](https://github.com/JustVugg/colibri)** (JustVugg, Apache-2.0) —
  showed that a 744B MoE can run in 25 GB of RAM, and is where we learned about
  router-lookahead prefetch, learned expert pinning, `F_NOCACHE` and `F_RDADVISE`
  discipline on macOS, and the shard-by-shard conversion pattern. Its M5 Max
  performance report — CPU spin-waits starving the GPU of a shared power budget —
  changed how we schedule work.
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** (Salvatore Sanfilippo,
  MIT) — the clearest expert-streaming design we studied: zero-copy expert
  buffers, masked dispatch, selection-based cache eviction, session persistence,
  and a quality methodology (average NLL against official API outputs) that we
  adopted outright. Its stated philosophy — correctness before speed, hide I/O
  behind compute — is the sensible one, and we tried to follow it.
- **[Moonshot AI](https://huggingface.co/moonshotai/Kimi-K3)** — for releasing
  K3's weights openly with readable modeling code, which Deltafin runs directly,
  and for the Kimi Delta Attention design, whose small recurrent state is what
  makes long context feasible on a laptop at all.
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)**
  (fla-org, MIT) — our KDA shim is a port of semantics from its kernels and
  reference implementations.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** — prior art for
  in-kernel dequantization and MXFP4 handling, and the foundation of most of what
  the local-inference community knows.
- **[PyTorch](https://github.com/pytorch/pytorch)**,
  **[Transformers](https://github.com/huggingface/transformers)**,
  **[ml_dtypes](https://github.com/jax-ml/ml_dtypes)** (our bit-exactness
  reference for e2m1) and **[tiktoken](https://github.com/openai/tiktoken)**.

### License

Deltafin's own code is [MIT](LICENSE). Two things in this repository are not ours:

- [`tools/fla/`](tools/fla) is a pure-PyTorch port of semantics from
  flash-linear-attention (MIT, © 2023–2026 Songlin Yang, Yu Zhang, Zhiyuan Li).
  The attribution is repeated in the file header and in [LICENSE](LICENSE).
- Kimi K3's weights and modeling code belong to Moonshot AI and are distributed
  under Moonshot's own license. They are downloaded at setup, never vendored
  here — please read that license before using them.

Deltafin is an independent project with no affiliation to Moonshot AI.
