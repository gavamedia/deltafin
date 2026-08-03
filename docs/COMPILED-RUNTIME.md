# Deltafin's custom compiled runtime

Deltafin ships one production executable named `deltafin`. Rust owns the
long-lived engine, public commands, HTTP server, model contract, tokenization,
storage, scheduling, cache lifecycle and output. A small versioned C ABI links
that executable directly to reviewed C++/LibTorch provider code and the CPU,
Metal and CUDA kernels that perform tensor arithmetic.

This is not a group of command-line programs calling one another. The provider
archive is linked into the Rust executable, all inference state stays in one
address space, and a token does not cross a process, interpreter or local HTTP
boundary.

## Production ownership

```text
deltafin (one Rust executable and one process)
  ├─ commands, configuration and server              Rust
  ├─ exact K3 tokenizer and chat rendering           Rust
  ├─ authenticated setup, fetch, convert and pack    Rust
  ├─ bounded I/O, residency and route tracing        Rust
  ├─ DSpark/Qwen controllers and cache transactions  Rust
  └─ coarse target transaction                       stable C ABI
       └─ tensor provider                            C++ / LibTorch ATen
            ├─ MPS and custom Metal                  Apple GPU
            ├─ CUDA and custom CUDA MXFP4            NVIDIA GPU
            └─ C / SIMD MXFP4                        CPU
```

The release profile uses fat link-time optimization and one Rust
code-generation unit. The binary audit rejects `libpython` and
`libtorch_python`. LibTorch remains because its CPU, MPS and CUDA providers are
already compiled tensor implementations; Deltafin calls those providers
directly instead of embedding their former frontend.

## Why Rust does not replace every tensor kernel

Rust is the right owner for state machines, bounded storage, validation and
concurrency. It does not make an untuned matrix kernel faster merely by being
Rust. Deltafin reuses the established LibTorch implementation when it is the
fastest exact compiled provider, and writes a custom kernel only when exactness
and representative timing justify it.

That rule has prevented several regressions:

- a one-dispatch custom Metal replacement was **20.5% slower** than the
  LibTorch/MPS provider on the measured shape;
- two expert-major compiled layouts were **1.6–2.2× slower** because register
  pressure reduced occupancy;
- adding KDA's small beta projection to a profitable five-projection bundle
  increased MPS time from 1.225 to 1.465 ms, so beta remains separate;
- a private fused MPS RMS operator was 3.1–6.2× faster in isolation but changed
  values by as much as 4.77e-7, so the exact path rejects it;
- a compiled residual/MLP row batch had the same 4.77e-7 physical-MPS
  difference and is excluded from the production provider.

The useful optimization is not “replace compiled math with different compiled
math.” It is to remove fine-grained dispatch and ownership overhead around the
best compiled math, then fuse only the shapes that actually win.

## The native build

`cargo build --locked --release` performs one reproducible build:

1. Cargo validates the locked Rust dependency graph.
2. On the audited CPU/MPS targets, the build bootstrap authenticates an exact
   PyTorch 2.13.0 artifact by size and SHA-256, extracts only its native C++
   layout and records a manifest for future validation.
3. A compiled Rust build script validates Clang/GCC, the archiver and optional
   NVCC as native executables. On macOS it also validates Apple's root-owned
   Metal toolchain, compiles both reviewed MSL programs to metallibs and embeds
   their bytes. It then compiles the provider archive and platform kernels
   directly.
4. Cargo links that archive and the selected LibTorch libraries into
   `target/release/deltafin`.
5. Runtime search paths point at the authenticated native library directory.

Source builds require Rust 1.85+ and a native C/C++ toolchain. macOS builds
require full Xcode and the component installed by
`xcodebuild -downloadComponent MetalToolchain`; release inference loads the
embedded metallibs and never calls `newLibraryWithSource`. An NVIDIA build
additionally requires a trusted CUDA LibTorch tree and matching NVCC supplied explicitly
through `DELTAFIN_TORCH_ROOT` (or `LIBTORCH`) and
`DELTAFIN_CUDA_MOE=ON`. The bootstrap has no automatic CUDA artifact today;
that limitation is deliberate and fail-closed.

The selected LibTorch package carries dormant general-purpose build metadata,
but the production build never evaluates it. Rust supplies the exact reviewed
include paths, definitions, sources and link libraries. Every selected
compiler/archiver must resolve to a native Mach-O or ELF executable, Python
environment variables and shell-startup hooks are removed, and conventional
interpreter names in the private build `PATH` resolve to a compiled denial
guard. That guard leaves a marker before failing; every native compilation and
archive operation checks the marker. CUDA 12.6 or any CUDA 13.x NVCC is
admitted only as a native executable belonging to the same major toolkit ABI
as LibTorch.

HTTPS setup and upgrades likewise avoid a generic native-package discovery
script. Deltafin carries a narrowly maintained `curl-sys` fork whose upstream
FFI declarations and license remain byte-identical, while its build boundary is
Rust-only and has no build dependencies or helper processes. macOS links the
system libcurl directly. Linux accepts only a root-owned `libcurl.so.4` in a
fixed trusted directory after bounded ELF/SONAME/export validation; both paths
then require libcurl 7.28 or newer with TLS and HTTPS before any transfer.
These static and runtime checks are covered separately so a library cannot
pass merely by having a plausible filename.

The current build targets are macOS arm64 and Linux x86-64/aarch64. Windows is
a future port and is not accepted by the provider build today.

The provider source lives in `native/provider_gate/`. Platform arithmetic also
reuses these audited sources:

| Source | Linked role |
|---|---|
| `tools/fused_gemv.c` | portable single-expert MXFP4 GEMV with aarch64 NEON and x86 SIMD |
| `tools/fused_gemv_batch.c` | persistent CPU workers, batched expert phases and ordered reduction |
| `tools/metal_moe.mm` and `tools/metal/moe_mxfp4.metal` | bindless Apple-GPU expert execution and exact scale4 descriptors |
| `tools/cuda_moe_kernels.cu` | NVIDIA MXFP4 GEMV and MoE execution |
| `native/provider_gate/provider_*.cpp` | coarse K3, KDA, MLA, draft, route and cache operations |

These source files compile into the same executable. They are not public helper
apps and do not create subprocesses.

### One native build and test graph

Production and provider tests now share `deltafin-native-build`. The former
CMake graph, shell runner, shell linkage auditor and Python native builder were
retired after the Rust graph reproduced their useful target admission,
compiler/ISA flags, Metal/CUDA selection, ABI checks and failure boundaries.
The authenticated PyTorch bootstrap remains an internal Rust library called by
that graph; it is not a second command-line application.

The complete provider gate is available through:

```sh
cargo run --locked --package deltafin-xtask -- native-test all
```

Its declarative specifications build the production archive once and reuse
it. Each independent test executable adds only its main and explicitly named
test-flavor translation units. Children receive a minimal environment with no
ambient loader path, bounded output and a deadline. Before any test executes,
the xtask recursively resolves its native dependency graph and refuses an
interpreter library at any depth.

The macOS matrix separately exercises CPU and physical MPS cases, the real
12288×7168 packed-int8 shape, route mailbox, DSpark, KDA/MLA, the 93-layer
target tape and both embedded and explicit development Metal-source modes.
Only that last isolated flavor defines the runtime-source compilation macro;
production always loads build-time metallibs. The CUDA test also contains a
capacity-one residency canary—miss, pinned upload/admit, zero-miss hit, exact
device-to-host byte comparison and cancel—but that physical arm must run on a
real Linux/CUDA host; macOS proves only its fail-closed stub.

## Native boundaries and fixes

### Configuration and device selection

Rust parses configuration once and resolves `auto` from real provider
capability. MPS pairs with Metal experts, CUDA pairs with the qualified CUDA
expert provider, and CPU selects the native SIMD kernel. Explicit unsupported
requests fail early; automatic optional paths may select a correct native
fallback.

Selection is not based on chip marketing names. M5 and future Apple systems can
qualify through the same device/ABI/correctness gates as M1. Linux x86-64 and
aarch64 use the same model contract with their own native providers.

### Exact tokenizer, chat template and decoder

The K3 rank-file tokenizer, special-token trie, Unicode splitting, XTML chat
renderer and incremental UTF-8 decoder are Rust modules. Construction validates
the complete 163,840-entry vocabulary. Ordinary user strings cannot inject
structural control tokens; only renderer-owned segments may enable them.

The tokenizer passed 43,904 reference cases. Large classified chat histories
use stable-order parallel segment encoding after a conservative crossover,
inspired by GigaToken's reviewed batching interface. Small prompts stay on the
lower-overhead sequential path. The streaming decoder keeps incomplete byte
fragments and control markers between token IDs rather than decoding the whole
response repeatedly.

### Authenticated setup and downloads

`deltafin setup`, `setup-k3`, `setup-dspark`, `setup-qwen`, `fetch-weights` and
`warm-expert-cache` are native commands. Their download clients use pinned
repositories, bounded redirects, exact lengths, hashes, strict JSON and
Safetensors schemas, resumable ranges, atomic publication and filesystem
identity rechecks. Downloaded repository code is data to audit, never code to
load.

`warm-expert-cache --convert-npz` also replaces the former NumPy/ZIP migration
utility. Its bounded Rust parser accepts only the six canonical uint8 arrays,
verifies ZIP CRCs and exact NPY shapes, handles C and Fortran storage order,
and emits the historical byte layout. A raw expert is published no-replace,
fsync'd and SHA-256-read back before the source NPZ may be deleted; an existing
raw file is hashed against the source rather than trusted by its length.

The one-shot setup planner calculates peak additional space plus a 100 GiB
floor on the target volume before transfer. It uses no-follow filesystem
queries and creates `.metadata_never_index` in enormous model stores on macOS
to avoid Spotlight walking terabytes of weights. Complete and streaming modes
share the same authenticated inventory, so an interrupted or partial install
can be reused safely.

### Safe native upgrade

`deltafin upgrade` requires a clean checkout and a normal branch with an
upstream. It calculates the Git relation, rejects divergence, applies only a
fast-forward and rebuilds the locked native workspace. It resolves Git and
Cargo to real ELF or Mach-O executables so a path-injected shebang shim cannot
gain execution. Model and cache roots are protected; setup, destructive clean
and hard reset are not upgrade operations.

Each release binary also carries a versioned, inert description of the native
profile that produced it. The authenticated automatic CPU bootstrap is stored
symbolically, while an operator-supplied LibTorch root is stored as an encoded
absolute path. CUDA is recorded by its effective compiled state (`ON` or
`OFF`), not the original `AUTO` request. When CUDA compiled, the effective
architecture set and canonical NVCC path are retained along with explicit
toolkit-location overrides. Upgrade removes
all corresponding ambient variables before invoking Cargo directly, then sets
only the recorded values. This prevents a CUDA installation from becoming a
CPU-only build merely because a later shell omitted its original variables.
Malformed profiles and moved external roots fail closed; changing profiles is
an explicit source-build operation.

The final executable does not exist when `build.rs` runs, so that script cannot
honestly audit the completed loader table. The safe upgrader audits the Cargo-
reported artifact before success, and `deltafin doctor` applies the same
bounded Mach-O/ELF audit to the running binary. This is a transitive audit, not
just a check of the executable: Rust resolves reachable `LC_LOAD_*` and
`DT_NEEDED` records through Mach-O runpaths or ELF RPATH/RUNPATH and `$ORIGIN`,
detects cycles, and reads only bounded loader metadata even when a CUDA library
is several gigabytes. It rejects Python-environment paths and any reachable
dependency whose basename matches `libpython*` or `libtorch_python*`.

Operator-supplied LibTorch/CUDA roots fail closed when a non-system dependency
cannot be resolved inside the recorded roots, or when a dependency symlink
escapes them. Platform-owned macOS and Linux libraries remain loader-managed
and are exempt through narrow system path/soname rules. The audit never runs
`otool`, `ldd`, a shell, or an interpreter; notably, it does not use `ldd` on
an object being inspected.

### Model contract and pack authentication

Rust validates K3 configuration and builds immutable tensor plans. DFSP packs
authenticate their header, ordered directory, payload, chunks, model identity,
source inventory and layout schema. Lengths and offsets use checked arithmetic;
parsers reject excessive allocation claims before allocating.

Original BF16 packs and explicit int8 packs use separate directories and
layout digests, preventing accidental mixing. BF16 is the automatic authority.
The embedding remains canonical BF16 and is read row-selectively rather than
being quantized with the streamed spine.

### Persistent bounded I/O

Rust-owned readers reuse aligned arenas, issue positional reads and coalesce
extents. Spine, authoritative experts and speculative prefetch have separate
bounded ownership. The next spine layer begins loading while the current layer
is executing.

The descriptor budget reads the host soft limit and current descriptor count,
reserves headroom for LibTorch and the server, and limits persistent model
descriptors. This fixed the observed `Too many open files` crash without asking
users to change a system-wide limit. Each reader can close or reopen safely,
and cancellation prevents an arena slot from being reused while an I/O worker
still owns it.

### Coarse provider binding

Validated layer storage crosses a descriptor-driven ABI call. The provider
owns every tensor after binding; Rust never lends a pointer to an asynchronous
operation that can outlive the source allocation. Resident layers form one
immutable ordered prefix. Streamed layers rotate through bounded storage only
after provider generations prove the previous use is finished.

Adjacent matrices may become zero-copy views when the authenticated layout
proves contiguity. Dense gate/up storage therefore avoids a giant concatenation
allocation on the matching packed path. Qualification samples sparse nonzero
values at real K3 shapes rather than accepting an all-zero canary.

### Complete target sequence tape

One provider-owned target sequence advances all 93 layers for 1–64 positions.
Rust binds the next spine generation, asks the provider to prepare the layer,
receives the authoritative top-16 route mailbox, supplies exact expert spans,
and finally commits only the accepted position prefix. KDA, MLA and speculative
state are private until commit; dropping the Rust owner cancels the transaction.

This coarse tape removes thousands of frontend-level tensor dispatches while
retaining LibTorch's compiled operations. Session, layer, spine generation,
row cursor, route order and bit-level fp32 weights are checked at every ABI
transition.

### Provider-owned KDA and MLA state

KDA and MLA caches remain opaque provider resources. A sequence stages new
state under a ticket, then Rust commits the exact accepted prefix or discards
the ticket. Published state cannot be mutated by a failed speculative branch.

Server reuse adds a stronger branch transaction: inspect the published KDA/MLA
boundary, create one exclusive child, route every target commit into it, and
atomically publish or discard it after the response transport resolves. Merely
matching a token-prefix hash is never treated as cache capability.

### Exact next-layer expert prefetch

For complete local CPU/Metal installs, native PILOT keeps a provider-owned
roster containing only next-layer norm and router data. It computes a
scheduling hint from K3's exact current activation. Rust submits sixteen
one-expert reads into a bounded seventeen-slot arena.

At the authoritative next router boundary, losers are cancelled first. Hits
are reused as scattered no-copy spans, misses are read through the normal
authenticated path, and the final array is ordered by the real route union.
The hint has no authority over weights, output or route membership. A bad hint
only wastes optional I/O.

The fix that made this safe was lifetime-oriented: every prefetched span owns
its read ticket until provider consumption, cancellation drains claimed jobs,
and the provider accepts explicit scattered spans rather than requiring a
second 280 MiB copy.

### CUDA pre-I/O planning and residency

CUDA does not use the host scattered-prefetch path. The provider first freezes
an immutable plan for the current authoritative route union and takes a
snapshot of device-resident hits. Only then does Rust open files for the exact
ordered miss list. The finish call consumes that one plan and accepts one raw
span per reported miss; an all-hit tile performs no expert file I/O.

The cache allocates after resident model setup, when live free VRAM is known.
Automatic mode keeps at least 2 GiB or 20% of current free space as reserve,
whichever is larger, and distributes capacity over all 92 expert layers rather
than letting early layers monopolize it. Pinned staging, CUDA events and stream
guards ensure upload buffers cannot be recycled early. Recoverable allocation
failure drains the stream and disables expert residency; terminal stream or
ABI failure poisons the optional CUDA expert provider instead of continuing
with uncertain ownership.

### Draft models remain proposal-only

DSpark and optional Qwen are parsed, mapped and executed through native
provider objects. No model-defined source is imported. DSpark can consume
provider-owned target rows without round-tripping the large activation through
host memory. Its controller snapshots exact draft state, begins with a
two-token economic probe, permits at most seven proposals and disables itself
when live verifier economics no longer repay the cost.

Qwen is reserved for raw continuation. Its 0.6B probe and 1.7B wide model are
loaded only when the request type, device and fixed memory budget admit them.
Its KV reservation follows the target's physically admitted context rather
than charging Qwen's full architectural limit; a separately admitted 0.6B-only
fallback avoids losing useful proposals merely because retaining both models
would exceed the startup peak. The actual retokenized input is checked before
provider allocation. Cross-tokenizer text is always re-encoded with K3 and
checked by the target.

Both paths return untrusted IDs. The target sequence computes the longest
matching prefix and emits at most that prefix plus K3's own next token. State
publication uses the same transactional commit boundary as target-only decode.

### Native server and output transport

The OpenAI-compatible server uses a bounded request parser, serialized
generation permit and bounded response memo. A second concurrent generation
gets HTTP 429 rather than racing provider state, while non-generation routes
remain responsive. Streaming transport separates thinking and answer control
markers without exposing partial marker prefixes.

Native production inference is currently text-only. The authenticated K3
inventory contains 165 vision-tower tensors and three multimodal-projector
tensors, but the native target deliberately excludes them until their complete
execution and parity path exists. The OpenAI request boundary rejects image,
audio and other multimodal content before entering the authoritative target;
it never substitutes an image placeholder and presents that as full-model
vision inference.

Target state is published only after a complete JSON response or terminal SSE
event has been flushed. Disconnects and write failures discard the private
branch. This closes a subtle class of bugs where externally failed output could
advance an internal cache beyond what the client actually received.

### Router traces and cache warming

The route trace is produced inside the native target sequence from the same
mailbox used for expert execution. Buffered mode batches completed passes;
synchronous mode is available for crash investigation. Files must be regular,
non-symlink paths, individual lines are bounded and the complete trace has an
8 GiB limit.

The native warmer parses these exact traces, ranks missing experts and remains
read-only without an explicit `--fetch`. Requested downloads go through the
same authenticated inventory and atomic publication as normal setup.

## Weight fidelity

The original resident BF16 checkpoint is the default for `--spine auto` and
`--spine bf16`. Target activations remain fp32, K3's released MXFP4 experts
remain unchanged, and all 16 selected experts run. No draft or prefetch result
can bypass the authoritative target.

`--spine int8` is an explicit non-weight-exact research option. Its row rounding
and fp16 scales can change resident weights, so the CLI and event stream label
it. Lossless scale4 is different: it codes expert scale sidecars reversibly and
does not change packed MXFP4 values or reconstructed scales.

## Verification boundaries

The native runtime is tested at several levels:

- Rust unit and integration tests for parsers, CLI, storage, downloader,
  tokenizer, server, cache and state machines;
- 18 isolated C/C++ provider executables for ABI reports, device rules,
  KDA/MLA, route mailboxes, DSpark/Qwen, PILOT, CUDA planning and the complete
  target tape;
- bit-exact CPU/Metal/CUDA kernel fixtures where the platform is available;
- synthetic 93-layer sequence tapes with cancellation and accepted-prefix
  commits;
- real-weight exact-token oracles for full-model runs;
- binary linkage audits that reject interpreter libraries;
- benchmark event validation that refuses malformed, poisoned or oracle-mismatched
  runs.

A narrow unit test cannot prove end-to-end speed. Public throughput claims keep
their host, model representation, prompt, token oracle and measurement method.
An optimization that fails exactness or loses a balanced timing remains a
falsifier, not a default.

The compiled migration has an additional parity gate: using the same model
representation, expert set, proposal policy, prompt and output oracle, its
interleaved median may not regress the frozen implementation it replaces.
The former frontend already dispatched compiled tensor providers, so merely
changing the control language does not earn a speed claim; native completion
requires reproducing the profitable packed-I/O and speculative schedule while
removing, or at minimum not adding, dispatch, allocation and copy overhead.

The original-BF16 MPS path also has physical full-target sequence evidence. On
2026-08-01, the independent frozen implementation and the native provider both
encoded `The capital of France is` as
`[1008, 10484, 318, 15383, 387]` and greedily emitted these exact 17 token IDs:

```text
[17374, 13, 646, 606, 142957, 37092, 387, 7081, 306,
 17374, 13, 646, 14715, 91527, 16575, 387, 1280]
```

Both decode to ` Paris. The Eiffel Tower is located in Paris. The Louvre
Museum is also`. The native run traversed all 93 target layers, all 16 routed
experts per layer and the allocator-reclamation boundary that had previously
failed during MLA growth. This is correctness evidence, not a throughput
benchmark, and it does not claim unmeasured physical CUDA parity.

## Historical source material

Some `tools/*.py` files remain in the repository as old experiment records,
reference semantics and test vectors. They are not part of installation or any
supported runtime pipeline. The compiled executable does not import them,
spawn them or use them as a fallback. Their continued presence is a source
audit convenience while native parity evidence is retained, not an alternate
product architecture. [The tools directory notice](tools/README.md) maps every
former public task to its supported native command and distinguishes compiled
provider sources from frozen migration material.
