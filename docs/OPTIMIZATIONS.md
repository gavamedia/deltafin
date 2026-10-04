# How Deltafin's optimizations work

Deltafin tries unusual speed ideas, but it uses a stricter definition of
success than "the output looked plausible." This document describes the paths
that survived implementation and their safety boundaries. Measurements are
kept beside the configuration that produced them; a structural saving is not
called a token-per-second improvement without a full-model measurement.

Entries are grouped by the day they landed, newest first. The invariant below
and the standing sections at the end apply to every entry regardless of date.

## New in this release — 2026-08-02

Deltafin is now a single compiled Rust executable that links reviewed
C++/LibTorch provider code through a versioned C ABI, replacing the previous
set of interpreted entrypoints calling one another.
[The compiled-runtime document](COMPILED-RUNTIME.md) describes that boundary.
Most entries in this document are established optimizations carried across it
unchanged; the release bundles work from 2026-07-28 onward.

The new technology it introduces is narrower:

- **DFSP authenticated spine packs.** A new on-disk spine format. One ordered
  authenticated file per layer commits to source identity, tensor roster,
  dtype, shape, alignment and per-chunk digests, replacing loose per-tensor
  files and the name lookup, descriptor and extent planning each layer paid.
  Original BF16 and explicit int8 use different pack roots, so a pack cannot be
  reinterpreted as the other representation.
- **Lossless scale4 expert storage.** A reversible four-way grouping of the
  MXFP4 scale side stream with an authenticated per-layer sidecar and manifest.
  It does not requantize K3's expert weights. Ten interleaved runs measured a
  2.4% median improvement against ten raw controls on the same oracle.
- **Native DSpark block drafting.** A K3-vocabulary proposal model, installed
  by default, with provider-owned target-row capture so activation bytes never
  cross into Rust and the checkpoint's redundant embedding is never
  materialized.
- **Persistent exact target state for growing server chats.** Provider state
  branches let a request inherit the exact state a prior request built, so a
  growing conversation stops re-prefilling its own history. One fixture reused
  117 of 174 tokens and cut wall time from 281.9 to 128.8 seconds.
- **Deltafin's own exact K3 tokenizer.** The rank-file tokenizer and chat
  rendering are now implemented directly against the validated 163,840-entry
  decoder, so the measured parallel-segment encode no longer requires
  installing a third-party native package.
- **PILOT rebuilt inside the provider.** The former router-lookahead helper
  became an immutable 92-layer roster with a bounded ticket arena, no-copy
  scattered spans across the ABI and two-phase cancellation.

## The invariant before the optimization

The normal target is K3's original resident BF16 checkpoint, fp32 target
activations and the released MXFP4 experts. The router selects all 16 of K3's
required experts at every routed layer. The complete K3 target is the sole
authority for every emitted token.

That rule changes how Deltafin can optimize:

- a draft model may propose token IDs, but K3 must compute and certify them;
- a route predictor may start reads, but the real router determines the expert
  IDs, fp32 weights and reduction order;
- cached state may be reused only when the provider proves it is the exact
  state for the complete token prefix;
- a faster operator must satisfy the relevant exactness oracle before it can
  replace the established operator;
- a memory failure may disable an optional optimization, but cannot select
  fewer experts, lower activation precision or assistant-authored output.

The explicit `--spine int8` path is outside the weight-exact default: it changes
resident weights and is labeled quantized and non-weight-exact. It remains a
useful research/performance configuration, but it is not presented as the
original BF16 target. Lossless scale4 expert storage does preserve values.

## 2026-10-03

*EAGLE-3.1 chain drafting on K3's own hidden states, with verify commits that
replay the KDA recurrence instead of keeping a 453 MiB state per row: chat
decode 2.0× faster on the reference host, byte-identical. Storage homes:
spine and expert reads spread over several drives, chosen per read job from
measured load and rate, with failover. Built and validated by emulation,
because the reference host has one SSD.*

### EAGLE-3.1 hidden-state chain drafting (`K3_EAGLE3`)

**Status: automatic for chat and server when `k3-draft-eagle3/` is installed;
full K3 verifies every token.**

[`lightseekorg/kimi-k3-eagle3.1-mla`](https://huggingface.co/lightseekorg/kimi-k3-eagle3.1-mla)
is a one-layer drafter trained against K3 itself (MXFP4, 93 layers). It reads
K3's within-block AttnRes stream after zero-based layers 1, 45 and 89 (vLLM's
aux ids 2/46/90, the stream it was trained on), normalizes and fuses the three
rows into one hidden state, and runs one MLA decoder layer plus its own
163,840-row head per draft. Each draft feeds the next, so a proposal is a
greedy chain of up to seven tokens.

How it fits the engine:

- The target sequence captures the three rows as one provider-owned BF16
  `[T, 3×7168]` tensor through the same path DSpark uses (five other layers).
  No activation crosses into Rust.
- The drafter's token embedding is byte-identical to K3's, so the engine's
  exact BF16 table supplies every input row. The provider holds only the
  drafter's own 3.43 GiB of weights and a 1,152-byte-per-position cache, about
  half of DSpark's footprint.
- Config and weights are admitted by pinned SHA-256; hashing the 6.03 GB
  checkpoint at startup takes about 3.7 s.
- It takes DSpark's place in the proposal slot and controller, and
  `K3_DSPARK` still sets the mode. Unlike DSpark, a miss narrows the chain to
  three drafts instead of ending drafting for the request, and automatic
  economics wait four verified steps before judging. Its first draft agrees
  with K3 89% of the time, so one early miss is noise.
- A C++ test checks the model against an independent fp32 reference; a Rust
  test drives the C ABI end to end on a CPU canary.

Offline acceptance on real K3 states from a chat (greedy chains replayed
against what K3 emitted; see docs/DEVELOPMENT-REFERENCE.md): **3.26, 4.17 and
4.77 tokens per verify pass at three, five and seven drafts.**

#### Evidence

Same 64-token chat on the 64 GB M1 Max (int8 spine, no resident spine layer,
scale4 experts). Every run below emitted the same 64 token IDs.

| Configuration | Verify path | Decode s/token | Whole run |
|---|---|---|---|
| Plain decode (DSpark loaded, but disabled itself) | none | 11.38 | 1005 s |
| EAGLE-3.1, per-row KDA snapshots refused by live memory | every proposal aborted | 11.68 | 1026 s |
| EAGLE-3.1, full-commit verify, accepted prefix rerun on a mismatch | full commit | 8.26 | 809 s |
| EAGLE-3.1, KDA prefix replay and a startup verify reserve | ordinary prefix commit | **5.62** | 643 s |

The three proposing runs made the same 17 proposals: 45 of 80 drafts
accepted, 3.65 tokens per verify pass. A verify pass costs about
9.2 s + 2.0 s per row here: the fixed part is mostly streaming the 54 GB int8
spine, the per-row part expert reads. A chain drafter amortizes the fixed part
over every accepted token. Fitted per-depth acceptance puts the best chain at
four to five drafts (about 5.1 s/token) against the controller's current
seven after a full match.

### Verify commits replay the KDA recurrence

**Status: automatic for every multi-row ordinary verify; bit-identical.**

An ordinary verify must be able to publish any accepted prefix. It used to keep
every row's KDA state, 6 MiB × 69 layers = 453 MiB per row. An 8-row verify
needed 3.1 GiB of fresh memory, and a busy 64 GB host refused it every time.
That memory refusal is why the first EAGLE run above never verified anything.

A multi-row verify now keeps the state after all of its rows, plus per layer
the decay, key, value and beta rows the recurrence consumed and the three
convolution sources: 19.4 MiB per row. Committing a shorter prefix replays
only the recurrence over the accepted rows, starting from the committed state.
It calls the same per-row update function the pass used, on the same row
tensors, so the state is bit-identical to the one the pass held there.

- It reads nothing and touches no weights.
- Each layer drops its full-sequence state as it replays, so the peak stays
  one staged generation.
- Replay happens before the no-throw publication half, so a failure still
  leaves every cache untouched.
- Exactness is checked against the recurrence's own boundaries at real K3
  width, for every prefix of T = 1..9, on CPU and MPS. The target-sequence
  prefix/continuation tests check it again through `commit_prefix`, with exact
  cache equality.

An 8-row verify now needs about 185 MiB beyond the decode reserve. A plan that
spends every spare byte on pins and spine refused even that at run time, so
the slot drafter's widest verify is withheld at startup, as Qwen's verifier
always was. That turned the full-commit-and-rerun 8.26 s/token into 5.62.

### Chat runs no longer reserve the Qwen drafter

**Status: automatic.**

Qwen serves raw completions only, and only while the slot is not forced on. A
direct chat run, or any run with `K3_DSPARK=on`, can therefore never route a
request to it, yet it reserved 1.7 GiB for it. That memory is now left to the
plan (the Qwen state reads `unused-by-run`); the server, which takes both
request shapes, keeps Qwen. On the reference host the pin tier grew from 156
experts (2.48 GiB) to 280–395.

### Storage homes (`K3_STORAGE_HOMES`)

**Status: implemented and exact; speedup predicted by emulation, not yet
measured on real external drives.**

Every pass on the 64 GB reference host streams the whole 53 GB int8 spine plus
the routed experts from the internal SSD (78.7 GB per T=1 token). A home is
another drive laid out like the model root, holding copies of
`k3-resident-int8/tensors/` and/or `k3-experts/`.
- **Drive choice.** Each read job (a 16 MiB spine chunk, an expert plane or
  file) goes to the holder whose in-flight reads will finish it soonest.
- **Shared load.** All readers share one load counter per drive, and each
  drive's rate is measured from the reads it completes.
- **Failover.** Failed reads retry on another copy; the failing drive sits out
  with back-off.
- **Exactness.** Selection only changes where bytes come from. The real-corpus
  gate (`installed_corpus_reads_identical_bytes_through_storage_homes`) read
  56.4 GB across all 92 layers, both layouts and wide unions, byte-identical.
  The full engine produced the identical France continuation with three drives
  attached.

#### How it was evaluated

Two diagnostic tools, described in DEVELOPMENT-REFERENCE.md:
- `K3_STORAGE_EMULATE` gives reads virtual devices by path prefix.
- The `home_replay` harness replays real routes in the engine's per-layer
  order.

Emulation setup:
- Devices run at 0.4× real speed, keeping four devices' total below what one
  SSD sustains.
  - Internal SSD: 2.8 GB/s (7.0 real).
  - Thunderbolt NVMe: 1.12 GB/s (2.8 real) and 2× the latency.
- Spare "drives" are free APFS clones.
- Every 4th MoE layer is replayed; times below are seconds per replayed pass.

#### Measured in emulation

Pure I/O, T=1 unless noted:

| Configuration | s/pass | vs 1 drive | Ideal |
|---|---|---|---|
| Internal only | 7.48 | 1.00x | — |
| First design: per-batch expert homes, fixed weights, +1 drive | 6.95 | 1.08x | 1.41x |
| First design, +3 drives | 6.27 | 1.19x | 2.21x |
| Per-job choice, experts only (spine not mirrored), +3 | 5.53 | 1.35x | — |
| Per-file choice, spine + experts, +3 | 4.47 | 1.67x | 2.21x |
| **Per-job choice, spine + experts, +1** | **5.39** | **1.39x** | 1.41x |
| **Per-job choice, spine + experts, +3** | **3.56** | **2.10x** | 2.21x |
| Spine + hottest 30% of experts per drive, +3 | 3.64 | 2.05x | 2.21x |
| Spine-only drives, +3 | 4.29 | 1.74x | 2.21x |
| T=8 verify, internal only | 14.82 | 1.00x | — |
| T=8 verify, full copies, +3 | 7.04 | 2.11x | 2.21x |
| T=8 verify, spine + hottest 30%, +3 | 7.27 | 2.04x | 2.21x |
| T=8 verify, spine-only drives, +3 | 11.71 | 1.27x | 2.21x |

Each design step fixed a problem the emulation exposed.
1. **Experts only on the extra drives leave the internal SSD saturated by the
   spine.** At most about 1.47x is possible for plain decoding this way.
2. **Fixed weights cannot see the spine's load on the internal drive.** With
   shared counters and measured rates, expert reads moved off it on their own.
3. **Placing whole files on one drive builds backlogs.** Several workers wait
   on one slow queue while other drives idle. Choosing per job reached
   95% of ideal.

The hottest 30% of experts by heat histogram cover 85.6% of the trace's decode
routes. That is why partial drives match full mirrors.

#### Fault behavior in emulation (+3 drives, T=1)

| Event | s/pass | Result |
|---|---|---|
| One drive throttles to 30% after 2 GB | 4.09 | 95% of the best achievable (3.87); measured 0.32 GB/s and was given less |
| One drive fails every read after 3 GB | 4.19 | 99% of the three-drive ideal; 4 failed reads retried elsewhere, no error surfaced |
| Speeds misconfigured 18x too high | 3.59 | same as correct; measurement overrode the prior |

The first design failed the whole generation on the emulated unplug.

#### Predicted real speed

Per-layer compute (45.9 ms, from the 2026-08-04 phase profile) is put back on
the critical path and stretched to the emulation's scale. That predicts T=1 at:

| Drives | s/token | Speedup |
|---|---|---|
| Internal only | 13.9 | — |
| + 1 Thunderbolt NVMe | 10.7 | 1.30x |
| + 2 | 9.0 | 1.54x |
| + 3 | 8.3 | 1.67x |

Notes on the prediction:
- The one-drive figure is about 16% above today's measured T=1 run (12.0
  s/token), because the model ignores PILOT prefetch overlap. The ratios are
  the useful part.
- Past three drives, compute dominates.
- Draft-verified passes are I/O-bound (PLAN.md: ~23 s per T=8 pass at the SSD
  ceiling). They should keep most of the 2.1x, so the France oracle could reach
  roughly 0.5-0.6 tok/s with three enclosures. Unverified.

#### Still unverified without hardware

- **Real throughput.** Actual Thunderbolt NVMe throughput on M1 Max for
  5.5-16 MiB `F_NOCACHE` reads (2.8 GB/s assumed).
- **Concurrency model.** Real NVMe drives share bandwidth across concurrent
  reads instead of serving them FIFO.
- **Demand vs spine priority.** Not implemented. It matters mainly for
  spine-only drives: 1.74x vs 2.10x.
- **Long-run behavior.** Enclosure thermal behavior over hours, APFS directory
  lookups across 82k files on external volumes, and Spotlight on new volumes
  (`populate-storage-home` writes `.metadata_never_index`).

### The learned pin tier is admitted after the drafter and the spine

**Status: implemented and exact; the order is by measured marginal value.**

The pin tier used to charge its whole ceiling as a fixed host cost *before*
residency and Qwen selection. That was harmless while the matured auto roster
was about 117 experts (1.9 GB). As history grew it reached about 1,700 experts
(27 GiB), and the up-front reservation started to evict things worth far more:
- On the 64 GB M1 Max it pushed out the Qwen drafter (memory-rejected) and
  every resident spine layer: **11.83 s/token** on the France oracle.
- On a 128 GB M4 Max (public issue #25, PR #23) a 32 GB budget displaced a
  spine that would otherwise fit completely, resident layers falling from 93
  to about 34 and throughput by 23%.

The tier is now sized after the pin-free plan is chosen, by binary search over
whole experts. Admission keeps the largest roster whose charge still leaves
the Qwen plan unchanged and the residency memory proof valid. Which spine
layers it must also leave alone depends on why the spine is the size it is:

| Pin-free spine | Pins may displace it? | Why |
|---|---|---|
| Every layer resident | No | streaming it back costs every pass |
| A prefix requested with `K3_PROVIDER_RESIDENT_LAYERS` | No | the operator asked for exactly that |
| Cut short by the memory budget | Yes | +15–19% from a ~1.9 GB roster against +0.23% for 1.7 GB more spine |

The Qwen drafter always outranks pins: up to 2× on draftable text, while pins
are near-neutral once drafting runs.

Same host and memory regime as the 11.83 s/token run, after the change: the
request was clamped to 220 experts (3.5 GiB), Qwen was admitted (13/13 drafts
accepted) and the run took **4.94 s/token, 2.4× faster**, with byte-identical
output. A clamp is reported at startup; a cold histogram still disables the
tier silently. The charge is never more than `candidates × span`, so the tier
cannot exceed what was admitted.

Tests run the real planning chain (`PlanSelector`, the code the engine uses)
on a 128 GiB host where the spine fits and a 64 GiB host where it does not,
check that admission is monotone in the roster size (the property binary
search relies on) across randomized hosts, and separate an explicit
zero-layer request from an over-budget probe, since both report zero layers.

Open follow-up: only a ~2 GB roster was ever timed. On a quiet host with a lot
of free RAM, auto may now admit up to about 27 GiB in place of a partly
resident spine, so A/B auto against a 2–4 GB cap before trusting a large
roster.

## 2026-08-09

*Per-expert early drain: a decode layer's experts start computing as their own
bytes land, without letting arrival order touch the reduction.*

### Expert early drain

**Status: implemented and exact, measured as a null, and therefore default off
(`K3_EXPERT_EARLY_DRAIN=on` enables it) for single-position Metal decode;
scheduling-only and fail-soft. The measurement is at the end of this entry —
read it before turning this on.**

A decode layer's missing experts are separate files. The reader pool fetches
them concurrently, so they land at different times — but the established path
waited for the layer's entire miss set before running a single expert's
matmul. The last expert to arrive gated all sixteen. On the reference host the
expert-read wait is 5.4–7.5 s of a steady chunk and the expert kernel is
0.7–1.0 s, so that kernel second was sitting inside a much larger window with
nothing to do.

Deltafin now starts each expert's independent work the moment that expert's
own bytes are readable:

1. The demand union's reader publishes per-job completion, so Rust can ask
   which experts of an in-flight batch are already whole. A slot qualifies
   only when every read that writes any byte of it finished without error, so
   a scale4 expert counts only after its three raw planes *and* its
   authenticated sidecar record have landed.
2. The caller parks on that signal rather than spinning — the reader workers
   own the device, and a busy caller would only take cycles from the reads it
   is waiting for.
3. Arrived experts are handed to the provider a wave at a time. Tier-resident
   spans go first: they need no I/O at all, so they are always available to
   fill the window the reads are about to occupy. Settled speculative hits
   follow, then demand arrivals.
4. The provider maps each arrived expert onto the route edge that owns it and
   runs that edge's GLU/W2 into its own output slot, committed without
   waiting. The GPU works while the caller returns to its reads.

**The reduction never moves.** Each expert's weighted contribution enters the
layer's routed output exactly once, inside the ordinary finish call, over
every edge in the router's own order, from the per-edge results staging
produced. Arrival order reaches scheduling and nothing else. This is the whole
reason the design is shaped as "stage, then finish" rather than "accumulate as
you go": a running sum fed in completion order would make the emitted token a
function of disk timing, and the same prompt would stop reproducing itself
across runs.

Everything about staging is advisory. The provider declines any shape it does
not handle (multi-row tiles, non-Metal backends, an unqualified bridge) and a
refusal is terminal for that tile rather than an error. The finish call
recomputes any edge that was never staged, and if anything about the staged
state fails to line up — a stale token, a different activation row, a command
buffer that failed — it discards the staging entirely and runs the established
whole-layer call. A layer that stages nothing is byte-for-byte the layer that
existed before.

Dispatch shape turned out to matter as much as the overlap itself. The first
implementation encoded one dispatch per staged expert, which is how the
bridge's direct-bind path has always worked — and it measured as a null,
because the layer's unstaged remainder then paid per-expert dispatch cost in
the finish call and gave back most of what the overlap won. A staged wave now
uses one dispatch per stage over the whole group: `moe_glu_slots` /
`moe_w2_slots` take an argument buffer of the wave's experts plus a map from
argument-buffer entry to output slice, so an arbitrary subset of a layer's
edges runs in a single dispatch without moving anyone's result. Each wave
takes a fresh span of that argument/slot pool rather than rewriting the front,
because an earlier wave's command buffer is deliberately still in flight and
still reading its own entries. The kernels call the same `glu_rows`/`w2_rows`
the whole-layer form calls, over the same bytes into the same slice — which
edges share a dispatch cannot change a value.

The wave size still matters. Each offer costs one provider call and one GPU
submission, so experts are handed over in reader-pool-sized groups, and the
tail below that threshold is deliberately left to the finish call, where it
joins that call's single command buffer instead of paying for another.

Exactness is asserted directly rather than inferred. `provider_moe`'s
staged-equivalence check drives the Metal bridge with exact K3 raw-v1 spans,
stages the edges in an order deliberately unrelated to route order and in
uneven waves, and requires the result to be byte-identical to the whole-layer
call — including for an abandoned token and for a partially staged layer.

Not covered, deliberately: multi-position verifier tiles keep the established
path, so a workload dominated by accepted block drafts sees this only on its
single-position chunks. CUDA is excluded — it has its own provider-owned
residency plan and one contiguous miss slab.

#### Measured (M1 Max 64GB, int8 spine, scale4 experts, 2026-08-09)

Interleaved `deltafin benchmark` on the France oracle, 20 generated tokens per
run, first decode step excluded, three pairs with early drain second and three
with it first — the balance matters, since going first pays a measurable
page-cache penalty on this host. All twelve runs emitted identical token IDs,
which is the invariant this entry exists to protect.

| pairs | median paired delta | mean | sign test | pooled per-step median |
|---|---:|---:|---:|---:|
| 6 (balanced), 108 steady steps per arm | **−0.01%** | +0.20% | 3/6 | +0.33% |

**That is a null, not a win.** The paired standard deviation is 3.07%, so this
campaign resolves roughly ±2.5%; the true effect is somewhere inside that, and
the point estimate is zero. Two earlier balanced six-pair campaigns at six
tokens per run agreed (−0.46% and 0.00% median), including one run of the
first, per-expert-dispatch implementation.

The mechanism is confirmed to be doing what it claims — roughly 11 of each
layer's 16 experts start their matmul before the layer's reads finish, and the
`[phases] expert early drain:` counter reports it — so this is a measurement
of a working optimization, not of a no-op.

Why the win did not materialize: the addressable budget is smaller than the
`expert_kernel` phase suggests. That phase brackets the whole finish call —
the routed-input host materialization, the routed RMS/up projection, the
shared-expert branch, the final merge, and the output transfer — and early
drain moves only the routed-expert GEMV portion of it. What it does move is
roughly cancelled by what it adds: one provider call and one GPU submission
per wave, and a caller wakeup per completed read. On this host, with a matured
pin tier resident (236 experts) leaving fewer misses per layer to overlap,
those two quantities are the same size.

It is kept, off by default, for three reasons: it is exact by construction and
proven so by test; the balance shifts toward it wherever reads are slower or
the resident tier is smaller than this host's; and the per-job read visibility
it introduced (`completed_destination_slots`) is the general primitive any
future consumer of partial batches needs. Extending it to multi-position
verifier tiles is the obvious next step, and should be measured on its own
before being claimed.

## 2026-08-08

*Measured learned-tier results across five workloads, and the automatic
roster-sized budget as the new default.*

### The learned pin tier, measured, and `K3_EXPERT_PIN_GB=auto`

**Status: automatic default; explicit decimal-GB and `0` overrides remain.**

The interleaved A/B the 2026-08-03 entry deferred to has now reported. Five
prompts of deliberately different structure, six balanced-order pairs each at
a fixed 2 GB budget, sixty runs total on the reference M1 Max, every
completion byte-identical between arms, no system stalls, and the candidate
roster verified unchanged across the whole campaign:

| workload | tier off | tier on | median paired delta |
|---|---:|---:|---:|
| narrative continuation | 16.120 s/token | 13.031 s/token | **+19.4%** (6/6 pairs) |
| short factual (France) | 13.167 s/token | 11.014 s/token | **+15.5%** (6/6 pairs) |
| instructional | 11.896 s/token | 11.926 s/token | +1.4% (4/6) |
| code continuation | 4.130 s/token | 4.112 s/token | +0.2% (3/6) |
| factual, off-roster (gold) | 11.268 s/token | 11.475 s/token | −2.0% (1/6) |

The France result independently replicates an earlier +16.9% campaign against
a different, larger candidate roster. The pattern is mechanistic, not
accidental: the tier pays when a workload routes into the learned roster and
decode is expert-read-bound; it is neutral when routing overlap is absent or
when block drafting already dominates (the code prompt decodes at 4.1 s/token
on speculation and leaves little demand-read stall to remove). The two
regressions are within noise and carried a draft-acceptance divergence
between arms. Hits are not savings: the instructional prompt produced the
most tier hits of any workload and no measurable win, so per-workload benefit
is claimed only where it was measured.

That asymmetry — double-digit upside on expert-read-bound natural language,
no measured downside elsewhere — is what justifies a default. `auto` sizes
the budget at bootstrap to exactly what the qualifying roster needs
(candidates × span), capped by live free memory minus an 8 GiB reserved
headroom, and resolves to off while the histogram is below its 256-pass
confidence ramp or the host is tight. On the reference host the matured
roster is 117 experts ≈ 1.86 GB, so `auto` reproduces the measured
configuration almost exactly; a fresh install runs tier-off until ordinary
use accumulates enough history. Explicit budgets bypass the cap, and the
16 GB configuration remains documented as harmful on 64 GiB hosts: it evicts
the entire resident spine prefix and measured slower than no tier at all.

## 2026-08-03

*Persistent cross-run expert heat and the opt-in learned RAM pin tier.*

### Learned expert residency from a persistent route histogram

**Status: histogram on by default and advisory; the RAM tier is explicit
(`K3_EXPERT_PIN_GB`) and off pending an interleaved measurement.**

K3 routes 16 of 896 experts per routed layer, and one decode token touches
roughly 25.8 GB of expert spans — far beyond any per-token cache. The existing
router-trace/warm flow can already learn placement, but only through an
explicit trace-then-fetch step, and only onto disk. Deltafin now accumulates
that signal automatically: every committed pass folds its authoritative routes
into a small decayed histogram at `k3-meta/expert_heat.v1.bin` (~330 KB),
merged across runs and processes under a sidecar lock with atomic replacement.

When a byte budget is set, bootstrap freezes a candidate roster from the
histogram and charges the roster's exact ceiling as a fixed host cost before
resident-spine selection. Promotion is sticky, never speculative: a candidate
enters the permanent RAM tier only when an authoritative route has already
read and authenticated its bytes, and later routes to it are served from that
tier as page-aligned scattered spans through the same provider boundary the
PILOT prefetch path proved. Ordinary demand I/O covers everything else; CUDA
keeps its provider-owned device cache.

This is learned storage placement without learned inference. The histogram
observes the mailbox after the router publishes; the tier changes only where
already-selected expert bytes are copied from. Route IDs, fp32 weights and
reduction order are untouched, and the second-route integration test asserts
tier-served spans are byte-identical to the tier-free read. A missing, stale,
corrupt or locked histogram degrades to "nothing learned yet" — recording is
infallible on the hot path and flushes are best-effort by contract.

Counting is presence-per-mailbox rather than per row, with an 8,192-pass
half-life, a 256-pass confidence ramp and a 2× uniform-frequency floor. Those
guards exist because our own holdout study measured naive frequency pinning at
**41.1%** held-out hit rate versus **51.9%** in-sample at a 40 GB tier, with
the gap widening as the tier grows: the tail of a frequency ranking is exactly
the part that does not generalize. Realistic budgets on the 64 GiB reference
host are smaller than that study's tier, so the honest expectation was a
lower hit rate over the best-generalizing head. The interleaved A/B this
entry deferred to has since reported; the 2026-08-08 entry above carries the
measured five-workload results and the automatic default they justified.

## 2026-08-02

*The compiled Rust runtime: one executable owning the engine, model contract,
tokenization, storage, scheduling, cache lifecycle and server, with the new
spine pack format, lossless scale4 expert sidecars, persistent server target
state and the provider-side PILOT predictor.*

### Exact next-layer expert prefetch with PILOT

**Status: automatic for a complete local expert corpus with CPU or Metal
experts; scheduling-only and fail-soft.**

K3 routes 16 of 896 experts in each of 92 MoE layers. Waiting for the next
router before starting those reads leaves storage idle during the current
layer's expert compute. Reading a guessed full union without a hard memory
bound can be worse: one layer of sixteen raw experts is roughly 281 MiB, and
keeping both guessed and authoritative unions would double it.

Deltafin's native PILOT path uses a narrower contract:

1. At bootstrap, after its exact provider-memory reserve has been admitted,
   the provider builds an immutable 92-layer roster. Each entry retains only
   the next-layer normalization and router tensors needed for scheduling. It
   never retains streamed expert slabs or aliases caller memory.
2. When layer `L` reaches its authoritative expert boundary, the provider has
   K3's exact post-attention residual for that position. It applies layer
   `L+1`'s retained norm/router and returns at most one canonical ascending
   16-ID hint.
3. Rust submits one authenticated one-expert ticket per hinted ID through a
   dedicated reader. The arena has 17 slots: sixteen live expert spans plus one
   bounded caller/worker transition slot.
4. Layer `L` continues executing while these reads progress.
5. At layer `L+1`, the ordinary K3 router publishes the authoritative IDs,
   fp32 weight bits and order. Predicted losers are cancelled before draining.
6. Correct predicted spans remain owned by their tickets and cross the ABI as
   no-copy scattered slices. Exact misses are fetched through the ordinary
   authenticated demand reader. The final span array follows authoritative
   canonical IDs, never prediction order.

The predictor never supplies route weights and never skips the real router. A
miss or an optional read failure is ordinary demand I/O. A malformed ABI report
is treated more seriously: the current target transaction is cancelled rather
than allowing advisory state to cross an unverified boundary.

#### Why the implementation is memory-safe

The first tempting version would concatenate all surviving prefetched experts
into a second contiguous slab. That copies hundreds of MiB and briefly owns
both versions. The provider now accepts an explicit array of scattered spans
on CPU and Metal. Each span's read ticket lives through the synchronous
provider call, and arena reuse is impossible until every claimed job drains.

Cancellation has two phases. Unclaimed jobs are atomically marked cancelled;
claimed reads finish into their still-owned arena slots and drain before those
slots return to the pool. This avoids use-after-reuse while still making losing
predictions cheap when the filesystem has not started them.

CUDA deliberately does not share this path. Its device cache has better
knowledge and uses the plan described next.

### Persistent exact target state for growing server chats

**Status: automatic, one owner and one retained boundary; strict extensions
only.**

OpenAI-compatible chat clients usually resend the entire conversation. Without
state reuse, every turn re-prefills the old history. Deltafin's native server
can retain the exact provider state the prior request already built, but only
through a complete transaction—not from a prefix hash alone.

#### Publication protocol

1. K3 completes prompt prefill and leaves its ordinary one-token-lag boundary:
   all but the trailing uncommitted token are committed.
2. The provider creates an exclusive child branch. Decode and speculative
   commits address this child, not the published parent.
3. Rust records model/configuration identity, complete logical token tape,
   committed position, provider cache generation and trailing token.
4. After a complete JSON body or final server-sent event is successfully
   flushed, the child may become the new published boundary.
5. A disconnect, write failure or incomplete response discards the child and
   restores the prompt parent. A provider failure that cannot prove restoration
   poisons the engine rather than serving from uncertain state.

The retained slot is considered only after 117 committed tokens and below the
current native exact-context admission bound. The next prompt must be a strict
extension of the complete logical tape. Equal prompts, edits, truncations,
forks, raw completions, changed configuration, stale cache generations or
provider mismatch consume the slot and reset to a fresh target.

Deltafin does not attempt longest-common-prefix rollback for recurrent KDA.
That narrower rule loses some reuse opportunities but has a simple proof: one
published state transfers exclusively to one direct extension.

#### It avoids more than time to first token

Reuse removes all old-prefix layer work before generation begins. Every output
token then proceeds from the retained state, so total request wall time falls
by the avoided prefill, not only the first-token timestamp. It does not make an
individual later decode transaction faster; it prevents the growing chat from
repeating earlier transactions.

One exact fixture reused 117 of 174 tokens. Time to first token fell from
**245.3 to 92.7 seconds**, and four-token wall time fell from **281.9 to 128.8
seconds**, with identical four-token output. The retained state in that earlier
compact-cache fixture occupied 0.451 GiB. These results demonstrate skipped
prefill, not steady decode throughput.

#### Context and memory truth

The K3 checkpoint advertises an architectural one-million-token context. The
current native provider uses an exact expanded fp32 MLA representation with a
512 MiB per-layer storage ceiling, so its presently admitted physical context
is much smaller and is reported at startup. Every geometric growth stages a
new generation while the old one remains live; Rust charges old, new and
intermediate allocator scratch against a fresh memory snapshot. A future exact
latent representation may raise the admitted bound, but the architectural
number is not presented as available memory today.

An absorbed compact-fp32 prototype was deliberately rejected as that future
representation. Although it stored fp32 latent rows, moving K3's `kv_b`
projection across the score and value contractions reassociated reductions. A
deterministic 33-position falsifier found 309 differing fp32 output elements
on CPU and 279 on MPS, with a maximum absolute difference of
`1.49012e-08`. Those values are small, but later layers can amplify them into a
different top-16 route or token. Production cache creation and both target
tapes therefore require the expanded representation; compact MLA remains a
synthetic research path until an implementation preserves the authoritative
operation order or establishes a stronger exact-output proof.

### Lossless scale4 expert storage

**Status: optional conversion; automatic only on a qualified exact consumer.**

K3's expert weights already use MXFP4. Deltafin does not requantize them.
Scale4 targets only the scale side stream: it groups four scale values into a
reversible representation and records an authenticated per-layer sidecar.

The native converter:

- requires the complete raw expert corpus;
- processes layers transactionally and resumes complete authenticated layers;
- writes source identity, layout and digests into a manifest;
- publishes the activation marker only after all 82,432 experts validate;
- retains every original raw expert file.

The sidecars add about 40.25 GiB but reduce scale bytes consumed by the matching
Metal path. Missing/corrupt/incomplete manifests stay on raw experts. CPU uses
raw by default where it measured faster; CUDA currently consumes raw-v1 expert
spans.

In the interleaved M1 campaign, ten scale4 runs reached **0.2734 token/s (3.66
s/token median)** versus ten raw controls at **0.2669 token/s (3.75 s/token
median)**, a **2.4% median improvement**. All twenty runs matched the same
17-token oracle.

### Authenticated spine packs and contiguous views

**Status: native pack command and automatic authenticated reader.**

Loose tensor files impose name lookup, descriptor and extent-planning work on
every layer. DFSP groups a layer's tensors into one authenticated ordered file.
Its header commits to:

- model/source identity and spine representation;
- exact ordered tensor roster, dtype, shape and alignment;
- per-chunk and complete payload digests;
- directory and layout schema.

The reader validates sizes and checked offsets before mapping or allocating.
Original BF16 and explicit int8 use different pack roots and digests. A pack
cannot be reinterpreted as the other representation.

When adjacent matrices have authenticated compatible layout, the provider
creates views into one owner instead of concatenating. The dense MoE gate/up
pair is the most important example: avoiding a huge temporary allocation saves
memory traffic and allocator pressure even though the tensor operator remains
the established compiled implementation.

### Native router traces and expert-cache warming

**Status: trace off by default; native plan/fetch command explicit.**

`--router-trace PATH --router-trace-mode buffered` records the authoritative
layer, route IDs and fp32 weight bits from the same provider mailbox used for
execution. Buffered mode reserves at most 5 MiB and writes completed passes;
`sync` is a slower crash-investigation mode. Paths are relative to the model
root unless absolute, must be regular non-symlink files and cannot grow beyond
8 GiB.

`deltafin warm-expert-cache` parses one or more traces, validates bounded JSONL,
counts/ranks missing experts and prints a read-only plan by default. Only
`--fetch N` authorizes downloads. Those downloads use the pinned K3 inventory,
bounded native HTTPS workers, resumable authenticated ranges and atomic file
publication.

This is learned storage placement without learned inference: a route history
can decide what to fetch while idle, but has no effect on the current router or
output.

### Native command and server consolidation

**Status: one supported executable.**

Setup, fetching, conversion, packing, inference, serving, benchmarking,
diagnostics and upgrade share the same Rust model contracts. That removes
repeated model discovery and fine-grained language-boundary calls, but the
larger benefit is correctness: one owner knows exactly which reader slabs,
provider tensors, state branches and response bytes remain live.

The OpenAI server serializes target generation. A concurrent generation gets a
bounded HTTP 429 rather than racing the provider; `/v1/models` and other
non-generation handling remain responsive. Request bodies, memo entries, output events and trace lines all
have explicit limits.

No local HTTP service connects inference components. Network HTTP exists only
at the public API boundary and in authenticated remote downloads.

## 2026-07-30

*Idle-warmed exact server tokenization; exact hybrid drafting and
hardened inference; sub-4-second exact decoding with universal drafts;
exact speed paths and a refreshed M1 benchmark; clean streamed chat
output.*

### GigaToken-inspired native batch tokenization

**Status: automatic inside the exact Rust K3 tokenizer for large classified
chat histories.**

[GigaToken](https://github.com/marcelroed/gigatoken), by
[Marcel Rød](https://github.com/marcelroed), showed a useful interface: encode
independent rows in parallel, preserve input order and return native token
batches. Deltafin first tested the upstream native package as a server adapter.
That experiment established two facts:

- very small prompts can lose to initialization and fan-out overhead;
- large, segmented histories have enough independent work to repay it.

The production runtime now implements K3's exact rank-file tokenizer itself in
Rust and applies the measured idea without installing the upstream wheel.
Trusted XTML structure is already split into independently classified segments;
ordinary user text never becomes structural input. If there are at least eight
segments and 128 KiB of text, at most eight scoped workers encode disjoint
segments. Results rejoin in input order. Smaller requests use the sequential
path.

Both paths use the same validated 163,840-entry decoder and fixed Unicode
expression. A worker failure rejects the whole call; no partial token list can
reach prefill. This is an automatic crossover, not a server process that spins
up later, so there is no KV transition or one-time context re-encode.

#### Isolated evidence

On the reference M1 Max, the earlier server batching experiment measured:

| Rendered input | Ordinary segmented encode | Native batch encode |
|---|---:|---:|
| 453 characters, 38 segments, 90 tokens | 0.438 ms | 0.155 ms |
| 100 synthetic turns | 67.225 ms | 9.735 ms |
| 1,000 synthetic turns | 664.987 ms | 95.986 ms |

These are once-per-request CPU preprocessing measurements. They do not make
prefill or generation 6–7× faster; they save only tokenization time. The native
automatic threshold retains the small-request path where fan-out does not pay.

### Native DSpark block drafting

**Status: installed by default; automatic for qualified chat/server requests;
full K3 always verifies. When EAGLE-3.1 is installed it takes this slot instead
(2026-10-03); `K3_EAGLE3=off` returns it to DSpark.**

[Inferact's Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) is a
K3-vocabulary proposal model trained for block drafting. The checkpoint is
6.635 GiB on disk and approximately 4.49 GiB at runtime in Deltafin because the
provider reuses K3's authoritative embedding rather than materializing the
checkpoint's redundant copy.

Deltafin downloads five inert pinned files, verifies the complete Safetensors
schema and never imports repository code. Rust owns admission and token-state
transactions; the C++/LibTorch provider owns model tensors and runs on MPS or
CUDA when qualified.

#### Target-row capture

DSpark needs information from K3's current trajectory. Copying a large
activation to host memory and converting it at every layer would erase its
benefit. The target sequence can instead expose one provider-owned BF16 tensor
containing the exact rows DSpark consumes. The DSpark provider appends the
accepted prefix directly on the same device; activation bytes never cross into
Rust.

#### Automatic economics

Auto mode begins with a two-token proposal, never more than seven. It requires:

- an installed, authenticated checkpoint;
- a supported provider/device and passing BF16 canary;
- startup memory admission for the complete model and state;
- prompt length below the default 8,192-token auxiliary-state bound;
- a measured target baseline and a live proposal/verifier result that beats it
  by the configured safety margin.

If DSpark misses, fails, exceeds its auxiliary bound or no longer repays its
cost, the controller releases/abandons optional state and full K3 continues.
It may later re-challenge only at a clean conversation boundary, not by
attaching unmatched draft state to a target cache hit.

DSpark snapshots have model identity, exact token ledger and boundary
generation. Proposing cannot mutate committed draft state. After K3 verifies,
the controller commits only the longest matching proposal prefix; target and
draft final boundaries are paired only after the target publishes first.
Losing the DSpark half never discards a valid target boundary.

#### Evidence

A physical M1 creative-chat check measured canonical T=1 target-only at
**0.081220 token/s** and DSpark with its qualified short-convolution path at
**0.095027 token/s**. Both produced the exact same 25 output IDs. Earlier
code-shaped parity fixtures accepted 17 of 23 submitted drafts and matched the
24-ID target oracle, but machine load made those wall times unsuitable as a
headline.

### Optional Qwen universal drafting

**Status: separately installed, automatic only for the qualified raw-completion
workload.**

DSpark and Qwen solve different proposal problems. DSpark shares K3's token
space and is the normal chat/server drafter. The optional Qwen 0.6B/1.7B pair
is useful for raw text continuation, where a small universal model may cheaply
guess a predictable phrase.

Qwen's tokenizer differs from K3. Deltafin therefore treats its output as text:

1. the 0.6B model probes admission and the 1.7B model may generate a wider
   proposal;
2. the resulting text is encoded by K3's exact tokenizer;
3. `[trailing K3 token, proposed K3 IDs...]` enters the ordinary target tape;
4. K3 emits only the longest exact matching prefix plus its own next token;
5. partial matches commit that prefix and rerun a narrower exact boundary when
   needed.

Confidence and model choice influence only how much work is proposed. They do
not choose output. The fixed provider-memory reserve includes model tensors,
KV state and scratch before Qwen is loaded; insufficient headroom keeps it
inactive.

The reserve follows the exact target context that this host can actually
admit, plus one maximum proposal—not Qwen's unrelated 32,768-token
architectural ceiling. The controller checks the real retokenized Qwen input
against that bound before provider allocation. It first tries the
probe-plus-wide plan and, if that complete peak does not fit, retries a
separately proven 0.6B-only plan. On the current 64 GiB host this reduced the
charged MPS peak from 7.94 GiB to 4.91 GiB for both assistants, or 1.70 GiB for
the probe-only fallback. An over-cap or failed proposal still falls back to
ordinary exact K3 decode.

The contrasting planet completion improved from **12.530 s/token** target-only
to **4.682 s/token** with the optional confidence policy and identical 17 IDs.
The France oracle reached **3.826 s/token** in a fresh single confirmation. The
larger model alone had regressed the France prompt, which is why the native
controller retains probe-first request-local economics instead of assuming
“larger drafter is always faster.”

## 2026-07-29

*Portable speed paths and safe upgrades; qualified native CUDA MoE
path; Linux and CUDA compatibility; optional packed-int8 KDA
projections; clearer native-library and upgrade guidance.*

### Cache-aware CUDA expert planning before I/O

**Status: automatic when the compiled CUDA MXFP4 provider qualifies; exact
CPU fallback remains available.**

A conventional cache API asks for all expert bytes, uploads them, then discovers
which experts were already resident. That wastes the read and prevents storage
from overlapping only the true misses. Deltafin moves the cache decision in
front of file I/O:

1. The provider receives the authoritative route union and freezes a one-use
   residency plan.
2. Under the selected CUDA device guard, it snapshots live cache hits and their
   ready events.
3. The ABI returns the complete canonical list of missing expert IDs.
4. Rust reads exactly one authenticated raw-v1 span per reported miss. An
   all-hit tile passes an empty byte slice and performs no expert file I/O.
5. The finish call consumes the same plan, uploads misses through reusable
   pinned host storage, waits on hit/upload events in the active stream and
   executes the original authoritative route order.

The plan contains sequence, spine generation, layer, row cursor, route union
and provider-session identity. It cannot be reused for another layer or target
transaction. Dropping it releases an unpublished plan; attempting to finish
with a different miss roster cancels rather than guessing.

#### Automatic capacity and eviction

The cache does not reserve VRAM before the resident model has loaded. At the
first real route it queries live free memory, then keeps at least 2 GiB or 20%
of that free memory as headroom. Capacity is capped at 92 × 16 expert entries
and divided across permuted layer strata, preventing early layers from
consuming every slot. Each stratum uses recency ordering within its quota.

Pinned staging has one CPU owner and one CUDA ready event. Before the CPU writes
that slab again, it waits for the preceding DMA to stop reading it. Device
entries retain their ready event until the active execution stream has waited.
Recoverable allocation failure drains the stream and disables residency. A
stream/ABI failure that cannot establish safe ownership poisons the optional
CUDA expert path instead of continuing.

### Reference-only speculative cache snapshots

**Status: automatic inside target-verified drafting.**

Speculative verification needs to evaluate several candidate positions and
publish only the accepted prefix. Deep-cloning KDA/MLA state is both slow and
large. Deltafin instead keeps immutable published state plus provider-owned
staged generations:

- beginning a target sequence creates private state;
- every candidate row has an exact commit boundary;
- `commit_prefix(n)` publishes precisely the accepted prefix;
- dropping or cancelling the sequence discards every staged mutation.

The earlier reference snapshot experiment measured **0.001 ms** rather than
**3.56 ms** and avoided an approximately **475 MB** clone. The production
benefit is broader than that microbenchmark: ownership makes rollback cheap
enough for partial draft matches while eliminating giant transient copies.

Rust budgets the complete peak before selecting a wide verifier: one staged
KDA generation plus a small per-row prefix record (see "Verify commits replay
the KDA recurrence", 2026-10-03). If live memory cannot admit the requested
width, a slot proposal is verified as one full-commit transaction and any
other proposal falls back to ordinary exact T=1 decode. It never shortens
target work and presents that as the same verification.

### Resident-prefix selection and page-cache discipline

**Status: automatic from live host/provider memory.**

Keeping one more resident layer can remove a large read every target pass, but
overcommitting unified memory can evict experts or freeze the desktop. Deltafin
models fixed host/provider costs first, including tokenizer data, reader arenas,
embedding rows, target state, verification snapshots, optional DSpark/Qwen and
PILOT. It then selects the largest safe ordered layer prefix.

An ordered prefix matters: provider residency is immutable and sequential, and
claiming isolated cheap layers would complicate lifetimes without eliminating
the surrounding stream. Transient high-water bytes are evaluated for each
candidate prefix so a large first layer is not double-counted against itself.

The policy reads physical/cgroup limits and current free memory. Apple unified
allocations remain bounded by host safety. CUDA does not infer VRAM from host
RAM; provider-owned CUDA expert capacity instead uses a live device query at
the point it can be trusted.

For enormous model stores on macOS, setup writes `.metadata_never_index` before
large transfers. That avoids an unrelated Spotlight scan competing for I/O and
memory after hundreds of gigabytes appear. Deltafin does not kill or reconfigure
system services.

### Bounded asynchronous layer and expert I/O

**Status: automatic.**

The runtime uses persistent Rust worker pools, positional reads and aligned
reusable arenas. Each read batch has an immutable plan. Workers claim jobs
atomically; the caller participates in draining rather than waiting idle.

Spine execution follows a two-stage cadence:

1. finish and authenticate layer `L`'s read;
2. bind it into a provider generation;
3. immediately submit layer `L+1` when it is not already resident;
4. execute layer `L` while the next read proceeds;
5. recycle the prior transient slot only after the provider proves it is no
   longer referenced.

Expert demand reads use canonical route unions so repeated IDs across a
multi-position verifier are read once. Prefetch and CUDA cache hits narrow the
union further without weakening it.

Descriptor caching is bounded by the process soft limit minus existing open
files and explicit runtime headroom. This fixed a reproducible layer-6
`Too many open files` failure caused by keeping one descriptor for every
streamed tensor. The solution is internal; users need not raise `ulimit`.

### Packed MPS output head

**Status: explicit int8-spine performance path with dense fallback.**

The vocabulary head is unusually large. On the int8 research path, the packed
MPS matmul consumes row-int8 weights and scales without materializing the full
dense fp32 matrix. A six-vs-six rerun measured:

- **14.7%** higher median steady decode;
- **2.3%** higher prefill throughput;
- **9.2%** higher wall throughput;
- resident head storage reduced from about 4.7 GB to 1.17 GB.

Provider/shape/scale qualification happens before use and exceptions select
the established dense implementation. This is a real throughput result for
the explicit quantized resident configuration, not the default BF16 target.

### Position-major expert verification

**Status: exact capability-gated provider path.**

Accepted drafts create multiple positions that often share expert IDs. A naive
loop launches expert work position by position. The position-major Metal path
builds the canonical union once, runs compatible positions together and
reduces each row in original router order.

The representative real T=2 layer measured **4.7%** faster and the pooled
full-model experiment measured **2.0%** higher throughput. Exact route/weight
mailboxes and output tensors must match before activation. The provider retains
the ordinary per-position implementation as the qualified fallback.

### Native CPU MXFP4 kernels

**Status: automatic portable fallback.**

The CPU provider dequantizes MXFP4 inside GEMV and avoids a materialized fp32
expert matrix. aarch64 uses NEON. x86-64 has an exact 128-bit
SSSE3/AVX/FMA3 compatibility path and a target-attributed AVX2 path selected
only when the CPU and operating system support it. Those ISA flags apply only
to the kernel translation unit—release builds do not use `-march=native` or
assume AVX2. A host below the compatibility floor receives a classified error
before the first expert dispatch instead of an illegal-instruction crash.

Persistent workers avoid thread creation per expert. Shared control fields and
worker counters are cache-line separated; 128-byte alignment measured **0.2%**
at four threads and **3.6%** at eight threads in the focused kernel test. A
pre-existing fixed-size stack assumption was removed so hosts with more than
16 worker threads cannot overrun it.

The kernel test suite covers one-hot dequantization, float64 GEMV bounds and
bit-equality across single-row, multi-thread and batch variants. Platform
availability and ISA qualification are separate from speed claims on a given
machine.

### Exact response memoization

**Status: bounded and automatic in the native server.**

The server memo keys the complete target request semantics, not merely prompt
text. It can reuse a prior K3-certified response across JSON and streaming wire
formats while rebuilding transport-specific framing. A one-token or option
change misses. Default bounds are 32 entries and 64 MiB; both can be set to
zero.

Memo hits do not enter the model and therefore cannot stage or publish target
state. This separation avoids accidentally treating a response-cache hit as a
new provider KV boundary.

## 2026-07-28

*Initial release; exact Apple Silicon inference speedups; spine I/O
page-cache partitioning; Spotlight exclusion for large model stores;
first measured M1 Max benchmarks.*

This is the foundation the entries above build on: running K3's full
2.8-trillion-parameter MoE on a single Apple Silicon machine at all, then
bringing prefill from 40 to 25 seconds and decode from 16 to 15 seconds. The
storage and residency work introduced here is described in its current form
under 2026-07-29, which is where this document's present text for it was
written.

## Measurement discipline

Deltafin's native benchmark harness launches the current executable, captures
structured events and validates the run before summarizing it. An acceptable
performance comparison should record:

- hardware, operating system and available memory;
- resident representation and expert storage layout;
- device and expert backend;
- prompt token count, generated-token oracle and draft acceptance;
- prefill, first-token, steady decode and wall timing separately;
- tracing, cache warmth and competing system load;
- interleaved arms when environmental drift matters.

The established M1 Max reference used ten raw and ten scale4 arms interleaved.
It reported the median rather than promoting the best endpoint. Its 3.66
s/token result used an explicit int8 resident spine and optional Qwen, so it is
not relabeled as a measurement of the original BF16 target or of DSpark chat.

Long-context server performance must include the context length and whether an
exact prefix was reused. Five-token continuation throughput cannot predict a
near-capacity chat: tokenization, prefill, cache growth, expert unions and draft
acceptance all change with history.

## Rejected ideas remain useful

Some attractive paths were kept out because evidence contradicted them:

- custom compiled code can lose to an established compiled provider;
- a broader fusion can increase register pressure or force unnecessary work;
- a numerically close operator can still violate the chosen exactness rule;
- an int8 or compressed representation can save memory while changing target
  weights;
- an eager parallel tokenizer can lose on short prompts;
- a prediction that reads both guessed and authoritative experts can spend
  more I/O and memory than it hides.

Negative experiments belong in ignored development logs, not in the public
success list. The retained implementations above either preserve the default
target exactly or clearly identify the explicit non-weight-exact configuration
to which their measurements apply.

## Source map

The public implementation is the native workspace:

- Rust engine and ownership: `native/deltafin/src/engine.rs`;
- exact tokenizer/chat/output: `native/deltafin/src/tokenizer.rs`,
  `chat.rs`, `output.rs`;
- bounded storage and packs: `storage.rs`, `spine_runtime.rs`, `experts.rs`,
  `packfile.rs`;
- setup/fetch/convert: `one_shot_setup.rs`, `weight_fetch.rs`,
  `spine_int8.rs`, `expert_scale4/`;
- DSpark, EAGLE-3.1 and Qwen: `dspark_*.rs`, `eagle3_provider.rs`,
  `draft_backend.rs`, `qwen_*.rs` plus matching provider sources
  (`provider_eagle3.*`);
- server/KV publication/memo: `openai/`, provider target-state branch ABI;
- trace, warming and learned residency: `router_trace.rs`, `cache_warm.rs`,
  `expert_heat.rs`;
- tensor provider and PILOT/CUDA cache: `native/provider_gate/`;
- specialized C/Metal/CUDA arithmetic: the compiled sources under `tools/`
  listed in [the compiled-runtime document](COMPILED-RUNTIME.md).

Legacy interpreted files under `tools/` are historical experiments and
reference material only. No supported command, server or inference fallback
executes them.
