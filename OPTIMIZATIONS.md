# How Deltafin's speed paths work

Deltafin is mostly an exercise in moving fewer bytes, moving the necessary
bytes at the right time, and declining an optimization when the running
machine has not proved that it can execute it safely. Kimi K3 is a 2.8-trillion
parameter Mixture-of-Experts model, but one token selects only 16 of 896 routed
experts at each MoE layer. That makes the model runnable on one workstation;
it does not make the remaining I/O small. A single-position local target pass
touches roughly 25.8 GB of routed-expert data and, with the int8 resident spine,
about 53 GB of non-expert weights. Multi-position verification shares the
spine scan but reads the union of experts selected across its positions.

This document describes the paths retained in the runtime. It is deliberately
not a catalogue of every experiment we tried. We do not mean to suggest that
the underlying ideas are all new; the interesting part here is often the
particular ownership, fallback, or data-layout contract needed to make an idea
work with K3.

## Quality is the invariant

Deltafin treats output quality as a hard boundary, not another benchmark
column. A faster result does not qualify if it uses fewer routed experts,
changes the target decoding policy, lets an assistant author output, or enables
an approximate numerical shortcut. Storing and running the full model would
make little sense if inference quietly substituted a smaller or lower-quality
model at the last moment.

The full K3 target therefore remains the sole authority for every emitted
token:

1. Every MoE layer retains K3's complete top-16 routed-expert computation.
2. Greedy target decoding and its output limits remain unchanged.
3. The 0.6B and 1.7B assistants can propose candidate text only. Their
   confidence scores may shorten a proposal or decide which proposal is worth
   checking, but they cannot approve a token.
4. Candidate text is re-tokenized with K3's tokenizer and passed through the
   full K3 target. Only tokens derived and certified by that target pass can be
   emitted.
5. A mismatch, exception, or incomplete verification restores the last
   committed target state and falls back safely. A bad guess may waste time; it
   cannot gain authority over the answer.

The supported runtime enforces the same boundary at startup. It requires all
16 experts and fp32 activation numerics, and rejects the former reduced-expert,
fp16/approximate, and 4/6-bit mixed-spine settings. Representation and scheduling
optimizations—such as packed storage, native kernels, shared buffers, prefetch,
and multi-token verification—must pass their applicable correctness and token
parity gates while retaining a safe fallback. Confidence is used to avoid
wasteful verification, never as a substitute for verification.

This is also the benchmark rule: any measurement that crosses the quality
boundary is excluded, regardless of how impressive its tokens-per-second
number appears.

The status labels used below are important:

| Label | Meaning |
|---|---|
| **Default** | Used automatically in the ordinary configuration. |
| **Capability-gated** | Used only after the live runtime or native library proves the required operator, shape, and device behavior. |
| **Opt-in** | Implemented and correctness-gated, but off by default while its complete-token speed or wider hardware coverage is still being established. |
| **Structural evidence** | Exact byte, allocation, call, or synchronization work was removed, but we do not yet publish a tokens-per-second gain for that change alone. |

## Idle-warmed request-batched server tokenization

**Status: Default-auto for OpenAI-compatible server chats; exact whole-call
fallback.**

An OpenAI-compatible client normally sends the complete message history on
every turn. Deltafin must tokenize that history again before it can prefill
K3. The ordinary Kimi tokenizer first renders the chat into many independent
segments—ordinary user text, structural special tokens, template punctuation
and generation markers—and historically encoded those pieces one at a time.
The calls are individually small, but a growing chat can contain thousands of
them.

The server can now batch that segmented work through
[GigaToken](https://github.com/marcelroed/gigatoken), an in-process native CPU
tokenizer. It does not replace the authoritative K3 tokenizer or concatenate
text across boundaries:

1. K3's unchanged chat template still creates every segment and decides
   whether registered special-token spellings are structural at that position.
2. Each segment is split at the same 400,000-character and
   25,000-non-whitespace-character boundaries as the baseline tokenizer.
3. Ordinary pieces and structural pieces are submitted as two separate native
   batches. A small tape records every original segment, policy and piece
   count.
4. The encoded rows are reassembled in the original tape order. Adjacent
   segments are never merged, because doing so could change BPE boundaries.
5. If ordinary user or tool text literally contains a registered special
   spelling that the native compatibility layer declines, the complete native
   result is discarded and the whole chat tokenization is rerun through K3.
   Any unexpected native error does the same and disables that backend for the
   rest of the server process.

Only a shallow private tokenizer proxy owns the batched segment hook. The live
K3 tokenizer used for decoding and universal drafting is never mutated.
Prompt-token IDs therefore retain one authority just as generated tokens do.

### Why initialization happens after a response

The first valid server chat is already on the warm side of the measured
request-batching crossover, but importing, constructing and fully qualifying a
native backend takes roughly a quarter second on the reference machine. The
default `auto` lifecycle moves that one-time work past the first response:

1. The first chat uses K3's ordinary tokenizer.
2. Only after its response bytes have been sent and flushed does the server
   qualify the backend. It retains target-generation admission during this
   short post-response step, so initialization cannot overlap a K3 pass.
3. A separate chat-tokenizer gate spans the final response flush and
   qualification. A client that submits its next turn immediately cannot slip
   a baseline encode into the interval; it waits for qualification and then
   receives the warm path. A qualification failure takes the exact baseline
   instead.
4. The backend is published only after all compatibility checks pass.

`--gigatoken on` performs those checks synchronously and fails server startup
if they do not pass. `--gigatoken off` never inspects or imports the package.
`K3_SERVER_GIGATOKEN=auto|on|off` provides the equivalent startup setting.
There is no tokenizer daemon, internal HTTP connection, runtime installer,
download, or network access.

Initialization, encoding and shutdown cross one native-call barrier. This
prevents a queued request from using a stale backend after another request has
disabled it and prevents native qualification from racing native encoding.
Shutdown first marks the controller as closing; any in-flight native result is
discarded rather than published afterward. Expected exceptions and PyO3
panic-style non-control-flow `BaseException` failures discard native output
and retain the baseline; process-control exceptions are not swallowed.

### Qualification and artifact boundary

Construction uses only the absolute local `k3-meta/tiktoken.model` rank file,
the explicit `kimi` pretokenizer and K3's complete 256-entry special-token map.
Before publication, the runtime checks:

- the exact 0.10.0 package and a reviewed platform artifact;
- the installed package-tree and native-library SHA-256, preventing a
  same-version rebuild or `sys.path` shadow from silently taking over;
- the complete special-token mapping and vocabulary size;
- multilingual, Unicode, whitespace and special-policy canaries; and
- `decode_single_token_bytes` for all 163,840 token IDs.

The wheel archives themselves are SHA-256-pinned in `requirements.txt`, so pip
cannot substitute a later artifact with the same version or fall back to the
foreign source distribution. Reviewed wheels cover macOS arm64 and
manylinux2014/glibc Linux on x86-64 and aarch64. The Linux packages were
unpacked and statically audited, including their ELF architecture,
dependencies, Python-file identity and RECORD hashes; physical performance was
measured only on macOS arm64. Unsupported runtime artifacts fail to tiktoken in
`auto` and fail closed in explicit `on`.

Before timing, parity probes compared 559,513 output token IDs across growing
multilingual chats, ordinary literal special spellings and the exact
400k/25k boundaries. The following numbers isolate only the already-rendered,
warm segment-encoding step on the reference M1 Max:

| Fixture | Rendered chars | Segments | Tokens | K3 segment loop | Native batch | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Smallest server chat | 453 | 38 | 90 | 0.438 ms | 0.155 ms | 2.83× |
| 100 completed turns + next user message | 124,086 | 3,650 | 38,009 | 67.225 ms | 9.735 ms | 6.91× |
| 1,000 completed turns + next user message | 1,235,586 | 36,050 | 379,109 | 664.987 ms | 95.986 ms | 6.93× |

The template has already built its segments before these timers start, and the
one-time qualification is excluded. This is a modest once-per-request CPU
saving, not a 6–7× model or tokens-per-second claim. Deltafin still creates a
fresh target cache and fully prefills the supplied history on every API
request; that much larger K3 work is unchanged. The path matters increasingly
as clients resend longer histories, while a rolling target KV cache remains
the more consequential future optimization.

Implementation:
[`tools/server_tokenizer.py`](tools/server_tokenizer.py),
[`tools/serve_openai.py`](tools/serve_openai.py),
[`tools/test_server_tokenizer.py`](tools/test_server_tokenizer.py), and
[`tools/test_serve_openai.py`](tools/test_serve_openai.py).

## Cross-tokenizer universal drafting

**Status: Default when the pinned local assistants are installed on the
measured MPS path; target-verified and forceable elsewhere, with target-only
fallback.**

The largest retained speedup comes from doing fewer K3 passes, not from making
one K3 pass dramatically cheaper. The 53 GB int8 spine is read once per forward
pass. If one pass can certify several output tokens, those tokens share the
same spine sweep, expert scheduling and layer traversal.

This deliberately spends roughly 4.63 GB of assistant residency—about 7.2% of
the reference machine's memory—to avoid repeated sweeps of a roughly 53 GB
target spine. It is the aggressive RAM-for-I/O trade that produced the largest
retained gain. New installs pair a 0.6B/1.19 GB model with a 1.7B/3.44 GB model;
either remains usable alone for older installations. The spine scan is shared
across verifier positions, but expert traffic is not free: a multi-position
pass reads the union selected by those positions. Wide drafts help only when
accepted tokens outweigh that larger union and its batched compute.

Deltafin uses the
[0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and
[1.7B](https://huggingface.co/Qwen/Qwen3-1.7B-Base) base forms of Qwen3 as
small local proposal models. This follows the broad idea of
[universal assisted generation](https://huggingface.co/docs/transformers/assisted_decoding):
the assistant and target do not need to share a tokenizer. The implementation
is deliberately conservative:

1. Decode the complete, already-certified K3 history to its canonical text and
   require that K3 tokenizes it back to the identical token IDs.
2. Ask the local assistant for a short greedy continuation. The first version
   rebuilds its small context on each proposal rather than maintaining a
   difficult cross-tokenizer KV boundary.
3. Decode the assistant's complete text and tokenize it with K3. If the exact
   K3 history is no longer a prefix, reject the proposal before touching the
   target.
4. Feed `[pending token, proposed K3 tokens...]` through the ordinary K3 model.
   Only the longest target-matching prefix plus K3's own first
   mismatch/bonus token may be emitted.
5. Retain exactly one target-cache row per emitted token. The final emitted
   token remains uncached as the next pending token. A fully accepted proposal
   keeps its target state directly. A partial universal proposal restores the
   pre-verifier snapshot and reruns only the certified narrow prefix; this
   avoids retaining numerically different state from a wider failed batch.

The assistant therefore cannot emit a token directly. A bad suggestion costs
time, and a partial suggestion pays for a conservative narrow rerun before its
state is committed. Proposal, verifier, rollback, EOS and output-budget
failures restore the pristine target state before a T=1 fallback. A failure in
optional post-verifier assistant bookkeeping keeps the tokens K3 already
certified and disables later proposals instead of re-feeding the old pending
token.

### Request-local admission and width

Wide verification is valuable only when drafts match. Each request starts with
two tokens proposed by the inexpensive 0.6B model. A full two-of-two acceptance
qualifies an eight-draft fast lane; a partial wide acceptance contracts the
next width, and a miss returns to the probe width or target-only decoding. The
hard ceiling is eight drafts, hence nine K3 input rows, because that is the
measured/capability-tested packed-int8 range on the reference backend. The
output budget can only lower the width.

At a qualified wide position, both assistants generate independently. If their
complete K3-tokenized proposals agree, that shared proposal is submitted. If
they differ, the larger model is selected only when its weakest generated-token
probability exceeds the smaller model's by more than 0.02; otherwise the 0.6B
proposal wins. This is scheduling, not model arbitration: the selected text is
still untrusted, and the full K3 pass still derives every emitted token. The
pair exists because the 0.6B model proved stronger on the France admission and
tail while the 1.7B model found the long planet continuation.

For proposals wider than the two-draft admission probe, the assistant's own
selected-token probabilities can shorten a proposal at the first value below
0.30. The narrow probe is always verified directly: a confidence
false-negative there can prevent a useful fast lane, while avoiding one T=3
pass saves little. If a wide proposal is shortened to no K3 tokens, Deltafin
runs one ordinary target token and tries the assistant again at the next
position. It does not treat a low-confidence guess as a request failure or
permanently discard a previously qualified width. The probability therefore
controls only how much untrusted work K3 is asked to verify; it can never admit
a token. This follows the useful principle behind dynamic speculation
lookahead, applied conservatively around Deltafin's cross-tokenizer and
exact-cache constraints.

This policy is based on work per accepted token rather than a machine name.
The assistant is selected automatically only when it is installed and the
chosen device is MPS. CUDA and CPU remain forceable for measurement, but are
not assumed to have the same crossover without end-to-end evidence.
`K3_SPEC=0` is the master target-only parity control, and
`K3_UAG_DRAFT=off` disables only the universal assistant.

### Exact M1 Max result

Three fresh-process runs on the maintainer's 64 GB M1 Max used the five-token
prompt `The capital of France is` and checked all 17 expected output token IDs.
Metadata-only setup first restored the pinned pristine Moonshot modeling files,
so the campaign exercised the tracked runtime cache shim used by a fresh clone.
No speculative pass was omitted as warm-up. In every run the assistant
proposed 13 of the 16 post-prefill tokens, K3 accepted all 13, and the decode
needed exactly three target passes:

| Target pass | Drafts accepted | Tokens emitted |
|---|---:|---:|
| T=3 admission probe | 2 / 2 | 3 |
| T=9 qualified pass | 8 / 8 | 9 |
| T=4 bounded tail | 3 / 3 | 4 |

The pooled median was **0.2660 token/s, or 3.76 seconds/token**. Individual
runs ranged from 0.2640 to 0.2718 token/s (3.79 to 3.68 seconds/token). The
assistant generated 19 tokens per run and took about 0.50 seconds total. This
is a real exact-oracle completion result, but not a universal acceptance rate:
unrelated prose, code and chat histories can accept fewer drafts and therefore
run more slowly.

A deliberately different prompt exposed that limitation. On
`The largest planet in our solar system is`, the earlier 0.6B path accepted
10/26 drafts and needed 10.515 seconds/token. A 1.7B-only confidence-gated path
reached 4.770 and 4.692 seconds/token, but did worse on the France prompt.

The final hybrid policy preserved both cases. On the planet prompt it rejected
one uncertain wide proposal before target verification, accepted 12/12
submitted drafts, and reached **4.682 seconds/token**. The four post-prefill
passes emitted 3, 1, 9, and 3 tokens. A speculation-disabled K3 control took
**12.530 seconds/token** and produced the identical 17 token IDs, a **2.68×**
same-build improvement for that completion. On France, a fresh hybrid run
accepted 13/13 submitted drafts, reproduced the established 17 IDs, and reached
**3.826 seconds/token**. We report these as small robustness checks rather than
as universal acceptance rates or speed guarantees.

`tools/setup_draft.py` downloads only fixed file allowlists from two pinned
official revisions, checks every SHA-256, and publishes each directory
atomically. Runtime loading is local-only, safetensors-only, uses Transformers'
built-in `qwen3` implementation, rejects `auto_map`, and sets
`trust_remote_code=False`.

Implementation:
[`tools/universal_draft.py`](tools/universal_draft.py),
[`tools/setup_draft.py`](tools/setup_draft.py),
[`tools/kimi_run.py`](tools/kimi_run.py), and
[`tools/spec_decode.py`](tools/spec_decode.py).

## Reference-only speculative snapshots

**Status: Default, exact.**

The unusual part of speculative decoding in Deltafin is not the n-gram draft;
it is making a failed draft cheap enough to abandon.

The original safe implementation cloned every KDA recurrent tensor, all three
short-convolution histories, and the MLA key/value views before a speculative
pass. That copied roughly 475 MB merely to create a rollback point. The current
snapshot records references to the old tensor objects instead:

- KDA recurrence and short convolution return new state storage during
  inference, so the previous objects remain immutable.
- MLA uses a geometrically growing slab. Appending writes beyond the old view,
  leaving its prefix unchanged; the old view also records the rollback length.
- A rejected draft restores the old KDA and convolution objects and restores or
  truncates the MLA views.

This turns the measured snapshot operation from 3.56 ms into 0.001 ms and
removes the large clone. The optimization is fundamentally an ownership proof,
not a faster copy.

The tests retain snapshots across accepted drafts, rejected drafts, partial
accepts, cache growth, replay, and reorder operations, then compare the future
token sequence. Deeper speculative passes can capture the KDA recurrence inputs
and replay only the accepted prefix; the whole-reference restore plus rerun
remains the conservative reference path.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/spec_decode.py`](tools/spec_decode.py), and
[`tools/test_snapshot_refs.py`](tools/test_snapshot_refs.py).

## Packed row-int8 output head

**Status: Capability-gated; automatic on the measured MPS path, forceable
elsewhere, with an exact dense fallback.**

The output head is an unusually large matrix: 163,840 vocabulary rows by 7,168
hidden columns. Materializing it as fp32 occupies about 4.7 GB. Deltafin already
has a per-row int8 checkpoint and fp16 row scales, so the faster path keeps the
weights packed and gives them directly to PyTorch's native weight-only int8
operator. Head residency falls to about 1.17 GB, and the separate full-head
dequantization disappears.

There are two additional data-movement details:

1. `torch.from_file` maps the packed checkpoint without first creating another
   1.17 GB Python byte string.
2. During ordinary prefill, generation needs only the final prompt position's
   logits. Deltafin therefore sends only that hidden row through the head.
   Speculative verification still computes every required position.

The native operator is a private PyTorch implementation detail whose device
registrations can change. Deltafin checks symbol and dispatcher availability,
then runs an analytical exact canary at the real KDA projection width. That is
not a head-shaped canary. The first real head call remains exception-guarded. If
it fails, Deltafin materializes the dense head *before* releasing the packed
representation, then uses the established dense calculation from that point
onward.

On the maintainer's M1 Max, a fresh six-vs-six balanced full-model rerun
measured **+14.7% median steady decode**, **+2.3% prefill throughput**, and
**+9.2% fresh-process wall throughput**. Every run matched the same exact
three-token oracle. These numbers describe that MPS configuration, not CPU,
CUDA, or every future PyTorch build.

Implementation:
[`tools/packed_q8.py`](tools/packed_q8.py) and
[`tools/kimi_run.py`](tools/kimi_run.py).

## Six KDA projections in one packed-int8 bundle

**Status: Opt-in, exact sequence-qualified on one real MPS stream; no speed
claim yet.**

A KDA layer applies six first-stage projections to the same hidden input:
Q, K, V, G, F-A, and F-B. The ordinary int8-spine path dequantizes all six into
the fp32 template and launches six dense projections. The experimental bundle
instead lays their row-int8 weights and row scales consecutively and asks the
same weight-only operator to process the combined rows. The output is split
back into the six original, unequal role sizes.

This path has two storage modes:

- **Arena** copies the current layer out of the recyclable upload pack into
  template-owned packed storage. It is simpler because the operator can finish
  after the loader has released its source buffer.
- **Stage** binds the packed operator directly to the current upload arena.
  Generation-stamped leases, device events, and a conservative MPS FIFO
  contract prevent the loader from refilling a buffer while an asynchronous
  operator may still read it. A stale lease, failed fence, changed shape, or
  changed stream contract disables the path.

The controller first checks each projection shape and the unequal-row fusion
contract. If a fused call fails, it can retain separate packed calls; if the
packed backend itself fails, it reconstructs the dense weights and atomically
returns the layer to the normal path. Unsupported CPU, CUDA, MPS, dtype, token
count, shape, or ABI combinations never inherit support from another backend.

One streamed-weight MPS sequence executed 816 logical projections as 136
operator calls and emitted the exact reference tokens. That establishes the
mechanism and rollback behavior, but it is not staged-KDA speed evidence.
CUDA packed execution has not been claimed.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py),
[`tools/packed_q8.py`](tools/packed_q8.py), and
[`tools/test_dynamic_q8_qkv.py`](tools/test_dynamic_q8_qkv.py).

## Treating the page cache as part of the engine

**Status: The int8 spine and expert-read shaping are default; explicit
resident-tier partitioning remains opt-in.**

With a model this large, RAM left to the operating-system page cache is often
more valuable than an equally sized Python heap cache. The int8 resident spine
helps twice: it approximately halves the bytes read for the non-expert weights,
and it leaves much more memory for recently used spine pages and routed
experts. This is why its effect can exceed the arithmetic suggestion of “half
the I/O” on a machine whose bf16 spine nearly fills RAM.

The checkpoint uses symmetric int8 rows with fp16 row scales. Deltafin selects
it automatically when it has been built, and retains bf16 as the fallback. In
the original quality checks, the top-five next-token candidates kept their
order and the top logit moved by 0.07%. This is a quantized quality result, not
a claim that int8 and bf16 tensors are bit-identical.

The read policy then separates two very different streams:

- Routed experts are a high-volume stream with little reason to evict useful
  spine pages. On macOS, the proven local path applies `F_NOCACHE` to those
  reads. Linux does not receive a Darwin command number; it stays buffered by
  default and has a separate, explicit best-effort `POSIX_FADV_DONTNEED` path.
- The spine is a cyclic scan. When an explicit resident tier is configured,
  a fixed subset of layers remains cacheable and the rest uses streaming cache
  advice. A fixed subset avoids the pathological LRU pattern in which a
  smaller-than-spine cache repeatedly evicts the page needed next.

On the reference Mac, demand-faulting expert mmaps delivered about 0.87 GB/s,
while the parallel read path delivered about 6.85 GB/s; the corresponding
expert-read slice fell from roughly 40 seconds to 4.3 seconds per token.
Those are historical I/O-path measurements, not promises for another SSD or
filesystem.

Implementation:
[`tools/convert_spine_int8.py`](tools/convert_spine_int8.py),
[`tools/spine_io.py`](tools/spine_io.py),
[`tools/spine_cache.py`](tools/spine_cache.py), and
[`tools/fetch_v2.py`](tools/fetch_v2.py).

## Zero-copy Metal expert weights

**Status: Default when MPS and the Metal library are available; numerically
checked with a CPU fallback.**

Each routed expert is stored as one 17,547,264-byte span containing its six
MXFP4 payload and scale tensors at fixed offsets. On Apple Silicon, the Metal
MoE path can wrap a page-aligned local span with
`newBufferWithBytesNoCopy`. The GPU then reads the same pages that the local
expert loader filled; it does not expand the expert into fp16 or fp32 and does
not make a second weight staging copy.

“Zero-copy” here applies to the expert weights. The current bridge still moves
the small activation and result across its synchronous CPU/Metal boundary.

The difficult part is lifetime. Read slots are recycled, and a later expert can
occupy the same virtual address. Deltafin pins the Python object that owns each
mapping for as long as Metal retains a wrapper. If an address appears with a
different owner, the stale wrapper is dropped before reuse. If the same live
slot receives new bytes, command completion and the host fill establish the
required order. Non-contiguous legacy cache entries are copied into reusable
page-aligned anonymous mappings and then use the same Metal path.

One command buffer performs the selected experts' gate/up projections, SiTU,
down projections, and weighted reduction. If Metal initialization, shader
compilation, shape validation, or execution fails, the runtime reports the
reason and selects the native CPU MXFP4 path. The Metal and CPU implementations
have been checked on real experts with token-oracle coverage; they are not
described as byte-identical floating-point implementations.

Implementation:
[`tools/metal_moe.py`](tools/metal_moe.py),
[`tools/metal_moe.mm`](tools/metal_moe.mm), and
[`tools/metal/moe_mxfp4.metal`](tools/metal/moe_mxfp4.metal).

## Position-major Metal MoE

**Status: Opt-in and exact-token qualified.**

Speculative verification gives the MoE layer several positions at once, but
the original Metal bridge submitted and waited once per position. The
position-major path flattens those routes in their original position and
router order, resolves each unique expert span once, and encodes the positions
in one command buffer. It uses a separate scratch pool so an occasional
multi-position pass cannot permanently enlarge the ordinary one-token buffers.

The reduction order for each position remains the model's original top-k order.
Missing support for the multi-position ABI falls back to the established
one-position loop. Balanced full-model measurements on the reference M1 Max
showed **+2.0% pooled throughput** on accepted speculative passes. It remains
off by default because the crossover depends on the Mac, runtime, and accepted
batch shape.

Implementation:
[`tools/metal_moe.py`](tools/metal_moe.py),
[`tools/metal_moe.mm`](tools/metal_moe.mm), and
[`tools/test_metal_position_batch.py`](tools/test_metal_position_batch.py).

## Direct CPU views into resident-spine read buffers

**Status: Default on the synchronous CPU apply path; exact structural
evidence.**

Packed int8 and mixed-codec spine reads already land in writable pooled host
buffers. The old CPU path copied their weights, scales, and bf16 tails into a
second CPU staging allocation before immediately consuming them. CPU
dequantization is synchronous, so the original read view already has the
necessary lifetime.

The direct-view path consumes that source in place and releases it only after
apply completes. Accelerator paths retain their device staging. The
experimental packed-KDA stage controller also keeps a conservative owned
weight copy where its operator may outlive the ordinary CPU apply call.

For a fully streamed 93-layer int8 pass, the exact accounting removes
**50.71 GiB of redundant host copies**. Source poisoning, retained-buffer,
mixed-codec, device-selection, and fallback tests cover the lifetime boundary.
This is structural evidence only; no complete-token speed percentage is
attached to it.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py),
[`tools/mixed_spine.py`](tools/mixed_spine.py), and
[`tools/test_cpu_spine_direct_views.py`](tools/test_cpu_spine_direct_views.py).

## In-place SiTU, scratch reuse, and ordered combine

**Status: Default on the portable CPU MXFP4 batch path; bit-exact structural
evidence.**

The CPU expert path produces disposable gate and up vectors, transforms them
with K3's SiTU activation, runs the down projection, and then multiplies each
expert row by its routing weight before adding it in router order. A direct
NumPy expression allocates several full-size temporaries at each step.

Deltafin keeps NumPy's established fp32 ufunc order but supplies explicit
destinations:

- the gate and up arrays are overwritten after their last original use;
- the not-yet-written prefix of the expert-output allocation serves as SiTU
  scratch when its shape permits; and
- each disposable down-projection row is scaled in place before the same
  ordered add into the output.

Real-expert and focused numerical tests are bit-exact. The known T=1
allocation accounting removes **57.5 MiB per 92-layer pass**. Timing on a quiet
host is still needed before assigning a throughput number.

Implementation:
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py) and
[`tools/test_situ_inplace.py`](tools/test_situ_inplace.py).

## Reusing the short-convolution source as its cache

**Status: Default in inference; exact structural evidence.**

KDA has three width-four causal depthwise convolutions. Both supported formulas
already create the source consisting of the last three cached values followed
by the new input. The previous implementation then concatenated cache and
input a second time solely to return the last four values as the new cache.

In inference, the first source's final four values are the same values, so the
runtime returns that tail. At T=1 the returned cache can be the source
allocation itself. With gradients enabled, the code deliberately keeps the
old independent allocation: backward may retain the first source, and a caller
is allowed to mutate the returned training cache without invalidating it.

Across 207 T=1 convolution calls, this removes **87.328 MiB of allocation per
model pass**. CPU and MPS real-width tests cover both convolution formulas,
cache identity, immutable retained snapshots, and mutation-before-backward.
That byte figure is not a tokens-per-second claim.

The related decode kernel treats a four-tap depthwise convolution as the small
operation it really is: build the sliding windows once, multiply by the four
weights, and reduce. This avoids asking a generic grouped-convolution engine to
schedule 12,288 tiny groups. It is automatic through the measured T=9 CPU
range and for the established T=1 accelerator path. `conv1d` remains
forceable, and larger accelerator batches retain the conservative crossover
until independently measured.

Implementation:
[`tools/fla/modules/__init__.py`](tools/fla/modules/__init__.py) and
[`tools/test_shortconv_cache_reuse.py`](tools/test_shortconv_cache_reuse.py).

## One routing record for every consumer

**Status: Default and exact; structural synchronization evidence.**

Expert fetching needs selected IDs on the CPU. The CPU and Metal MoE backends,
weighted reduction, trace writer, and lookahead accounting need the same IDs
and weights. Repeated `tolist()` calls are not just Python work when the router
lives on an accelerator: each conversion can introduce a device
synchronization.

The driver now materializes one ordered CPU record, reusing the ID rows it
already created for fetch scheduling and converting the fp32 route weights
once. That read-only record flows through the selected backend and optional
trace. Legacy direct callers can still materialize locally, while
`K3_FAST_MOE=0` with tracing disabled avoids weight-list work entirely.

Tie order, dtype conversion, multi-position routing, backend fallback, and
trace serialization are tested. We do not attach a throughput percentage to
this synchronization removal.

Implementation:
[`tools/routing_record.py`](tools/routing_record.py),
[`tools/kimi_run.py`](tools/kimi_run.py), and
[`tools/test_routing_record_portability.py`](tools/test_routing_record_portability.py).

## Zero-copy MLA query view at single-token decode

**Status: Opt-in, exact within its qualified contract; structural evidence
only.**

The downloaded MLA forward reshapes each query projection, splits its adjacent
128- and 64-wide columns, then concatenates those unchanged columns back in the
same order. For T=1, the contiguous projection already admits the canonical
`[B,96,1,192]` view. `K3_MLA_QUERY_ALIAS=1` reuses that storage instead of
allocating the query concatenation.

Across K3's 24 MLA layers, this removes 24 concatenation outputs and 1,769,472
bytes (1.6875 MiB) of temporary payload from an ordinary one-token pass. This
is allocation accounting, not measured DRAM traffic or a tokens-per-second
result.

The wrapper is installed only when the downloaded initializer, forward, and
eager-attention provider match reviewed source fingerprints. At runtime it
also requires inference mode, gradients off, T=1, and the exact registered
read-only provider. A contiguous projection with the canonical stride takes
the alias; an unexpected projection layout uses the established split-and-
concatenate construction without running the projection twice. Training,
T>1, changed source, or another provider calls the original forward.

Tests cover bit-exact full forwards, cache-bearing decode sequences, T=1/2/8
transitions, noncontiguous fallback, provider mutation, reload and opt-out
behavior, compilation wrapping, and CPU-SDPA composition. A physical MPS check
runs when that backend is available. The default remains off pending quiet
full-token timing and wider physical-device measurements.

Implementation:
[`tools/attn_fast.py`](tools/attn_fast.py) and
[`tools/test_mla_query_alias.py`](tools/test_mla_query_alias.py).

## CPU value padding to unlock fused SDPA

**Status: Opt-in, default off, fingerprinted per live CPU/runtime shape.**

K3's MLA uses 192-wide query/key heads but 128-wide value heads. On the tested
CPU build, ordinary unequal-width scaled-dot-product attention decomposed into
the math implementation. Appending 64 exact-zero value channels selected
PyTorch's fused CPU FlashAttention operator; slicing those channels from the
output restores the original width.

The production candidate pads per call, so the cache remains 128-wide. This
avoids the persistent alternative's 50% larger value cache and 20% larger
combined key/value cache. Eligibility includes PyTorch version, reported CPU
capability, thread counts, dtype, batch, dimensions, T, context bucket, mask,
and layout. On the first real call for each key, the profiler must observe the
native CPU FlashAttention operator. A missing operator, unsupported shape,
allocation failure, changed mask, training mode, or runtime exception disables
that key and executes eager attention.

Across the measured T=1/2/8/9 and context 32/128/512/1024 cells, the padded
attention call had a **1.23x geometric-mean isolated attention speedup**. This
is not a full-layer or tokens-per-second result. MPS already had a native
unequal-width attention operator in the local tests, and CUDA has not inherited
the CPU result.

Implementation:
[`tools/attn_fast.py`](tools/attn_fast.py),
[`tools/test_mla_cpu_sdpa.py`](tools/test_mla_cpu_sdpa.py).

## Fused MXFP4 dequantization and GEMV

**Status: Default native CPU expert path, with architecture and runtime
dispatch.**

The routed weights store two E2M1 values per byte and one E8M0 scale for every
group of 32 columns. A conventional implementation expands a matrix to fp32
and then multiplies it, creating eight times as much weight traffic before the
GEMV even begins. Deltafin decodes each packed group inside the row loop and
accumulates directly into the output.

An immutable, 64-byte-aligned 8 KiB lookup table covers the complete 256-value
E8M0 scale domain. That replaces repeated exponent-table synthesis while
retaining the synthesis implementation as a build-time correctness oracle.

The same source has guarded implementations for:

- NEON on aarch64;
- a 128-bit x86-64 compatibility path using the required
  AVX/FMA3/SSE3/SSSE3 baseline; and
- an internal target-attributed 256-bit AVX2/FMA path selected only when the
  CPU and operating system report it usable.

The x86 library is therefore one fat binary rather than an AVX2-only artifact.
The direct implementation details are internal, not a promised public native
ABI. The build validates the supported entry points before installing a
library.

MXFP4 columns are groups of 32, and the supported matrices preserve that
contract. The tests include odd output-row counts; they do **not** establish
arbitrary column tails. Full-domain scale/nibble checks, single and
multithreaded results, awkward row counts, and batch paths are compared with
the independent reference. Native Linux AVX2 timing is still wanted, so no
Linux AVX2 speed number is borrowed from translated or other-host evidence.

Implementation:
[`tools/fused_gemv.c`](tools/fused_gemv.c),
[`tools/neon_compat_x86.h`](tools/neon_compat_x86.h),
[`tools/build_native.py`](tools/build_native.py), and
[`tools/test_fused_gemv_portability.py`](tools/test_fused_gemv_portability.py).

## Qualified CUDA expert residency and fused MXFP4 MoE

**Status: Automatically considered on a CUDA resident device, but enabled only
after ABI, layout and on-device known-answer checks. The implementation is new;
maintainer-replicated CUDA token throughput and whole-sequence parity evidence
are still wanted.**

Moving the resident spine and attention to CUDA while bringing every routed
expert back to the CPU leaves an obvious boundary in the middle of each MoE
layer. The CUDA expert path removes that boundary without first expanding
MXFP4 weights to fp32. Its kernel reads the same packed E2M1 nibbles and E8M0
group scales as the CPU and Metal paths, performs gate/up GEMVs, applies SiTU,
and performs the down GEMV on the selected CUDA device.

This path is intentionally not enabled merely because `torch.cuda.is_available`
returns true. The bridge checks a versioned native ABI, the model's fixed expert
dimensions, the exact 17,547,264-byte expert span and a pointer-layout version.
It then runs small deterministic MXFP4-GEMV and int8-dequant known-answer cases
on the requested device. A missing library, stale build, unsupported device,
wrong dtype or shape, failed launch, or failed answer leaves CUDA active for
the resident model but selects the established CPU expert implementation.
Approximate fp16 mode never crosses an fp32 raw-pointer ABI.

Expert identity is stricter than a local expert number. K3 reuses IDs 0–895 in
every MoE layer, so a cache indexed only by expert ID can silently execute one
layer's weights in another layer. Deltafin keys residency by model namespace,
CUDA device, exact layer and expert ID. Capacity comes from current free VRAM
at the first real route—after model allocations—with an explicit safety reserve
rather than from a fixed number of entries.
The 92 MoE layers receive stable strata: each layer can evict only within its
own quota, and a capacity smaller than 92 admits a deterministic spread of
layers. Sequential traversal therefore cannot turn the cache into a
whole-model global-LRU thrash.

Each cache miss is assembled in reusable pinned host storage and transferred as
one contiguous expert span, rather than six independent weight/scale copies.
CUDA events guard pinned-slab reuse, and tensors are recorded on the stream that
consumes them so neither the host staging pool nor PyTorch's allocator can
recycle storage under an asynchronous operation. State is per device; the
native translation unit itself owns no device allocation or retained pointer.

The reduction is also deliberately conservative. One block owns an expert
output element and reduces a fixed tree; there is no cross-block `atomicAdd`.
Each selected expert produces its own output, and the route weights are combined
in their original top-k order. Multi-position calls remain on CUDA rather than
round-tripping every position through the CPU. The returned tensor is fresh,
not an alias of reusable scratch.

Residency is consulted before disk demand reads. Already-resident experts are
not fetched again, and router-lookahead prefetch filters the same hits before
issuing speculative I/O. If a cache query or native call fails, the driver
disables that device's CUDA expert path, fetches the **complete** routed set
again, and invokes the CPU backend. It never hands a partial miss-only mapping
to a fallback that expects the full route.

The same qualified bridge includes a flat-grid CUDA kernel for resident-spine
int8 dequantization. It writes `int8 × fp16 row scale` directly into an existing
fp32 destination, avoiding the pair of temporary conversions in the ordinary
PyTorch expression. This subpath has its own local circuit breaker and respects
an explicit `K3_SPINE_DEQ=torch` or `mulout` request; a dequant rejection does
not disable an otherwise healthy CUDA MoE backend.

Linux builds keep CUDA optional. Default `--cuda=auto` probes NVCC only on
Linux and installs the required CPU libraries even when CUDA compilation,
validation or installation fails. `--cuda=on` makes all requested artifacts a
single fail-closed transaction; `--cuda=off` performs no CUDA compiler probe.
The build asks NVCC which real and virtual architectures it supports, parses
all installed GPU capabilities defensively, and adds a native SASS image for
each compatible capability. A conservative `compute_75` PTX image remains in
the binary for forward portability whenever NVCC supports it; a future toolkit
that drops it uses its lowest supported virtual target. Native detection never
replaces that PTX fallback. If an automatically selected native target fails
compilation or validation, the builder retries the portable target. A strict
`K3_CUDA_ARCH` override supports cross-builds without allowing malformed
values to leak into compiler flags.

CUDA header compatibility follows the same narrow approach. The MXFP4 NaN
sentinel uses a project-owned device helper rather than a toolkit macro, and
device availability uses the stable attribute API instead of fields removed
from CUDA 13. Once the runtime device is resolved, spine-dequant startup warms
Metal only for MPS; CUDA remains lazy, and the CPU path touches neither
accelerator backend.

No CUDA speed number is claimed here yet. The expected wins—GPU expert compute,
fewer CPU/GPU boundaries, one-copy uploads and avoided disk reads on cache
hits—are mechanisms, not a benchmark. Publication-quality evidence needs
synchronized event timing, cache hit/upload counters, cold and warm runs,
matched output, and medians on real NVIDIA systems.

Implementation:
[`tools/cuda_moe.py`](tools/cuda_moe.py),
[`tools/cuda_moe_kernels.cu`](tools/cuda_moe_kernels.cu),
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/spine_fast.py`](tools/spine_fast.py), and
[`tools/test_cuda_moe.py`](tools/test_cuda_moe.py).

## Persistent CPU workers and batch-wide activation preparation

**Status: Default when the batch native library validates; legacy native GEMV
fallback otherwise.**

Running 16 experts requires 48 matrices: gate, up, and down for each expert.
The first native wrapper crossed Python and created/joined worker threads for
each matrix. The batch library keeps one worker ring alive and dispatches the
whole gate/up phase, then the whole down phase.

On the AVX2 implementation, the activation uses a lane order convenient for
the packed weight decode. The batch path prepares each distinct activation
once for the whole dispatch. In phase A, 32 matrices normally share one hidden
vector; permuting it 32 times would turn a kernel-local trick into avoidable
host work.

Workers claim fixed row units, so each row retains the same accumulation order
regardless of which worker executes it. The hot generation, cursor, and
completion counters are each aligned to 128 bytes to avoid false sharing.
That counter padding measured **+0.2% at four threads** and **+3.6% at eight
threads** in the focused worker test. Worker selection respects process
affinity and Linux cgroup CPU quotas, uses a conservative default of up to four
workers on macOS and eight on Linux, and retains explicit overrides plus hard
native bounds. This also closes the older risk of indexing fixed thread
storage with an oversized worker count.

The NumPy SiTU remains outside the native batch for the exact default because
platform `tanhf`/`expf` are not bit-identical to NumPy's established
transcendentals. If batch initialization or symbol/ABI validation fails, the
runtime uses the legacy native path.

Implementation:
[`tools/fused_gemv_batch.c`](tools/fused_gemv_batch.c),
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py), and
[`tools/runtime_platform.py`](tools/runtime_platform.py).

## Live CPU worker-width calibration

**Status: Exact opt-in; default off pending quiet validation on more hosts.**

A fixed native worker count is a poor universal rule for a bandwidth-bound
MXFP4 kernel. Core types, memory channels, process affinity, container quotas,
other traffic, and the particular CPU implementation can all move the
crossover. `K3_GEMV_AUTOTUNE=1` lets the persistent CPU backend measure its
first real, fully fetched top-16 expert set instead.

The candidate widths come from the process's effective affinity and Linux
cgroup quota, are capped by the native pool limit, and include the conservative
default plus a few widths below and above it. Pool construction is outside the
kernel score, and an unscored gate/up dispatch primes each newly rebuilt pool
before timing. Two reversed trials then time the existing gate/up and down
GEMV phases, and every candidate's fp32 phase output must be bit-identical.
The winner must be the same in both trials and beat the configured default by
at least 5% in each. The finally installed winning pool is primed and checked
once more before it is published for later layers.

The calibration is deliberately narrow:

- an explicit `K3_GEMV_THREADS` always wins;
- Metal and CUDA never enter it, including CUDA's emergency CPU replay;
- unknown or high host load, an unsupported shape, a pool-width mismatch,
  output drift, disagreement, insufficient margin, exception, or the
  cooperative 350 ms deadline retains the configured default (or the positive
  worker count the native pool actually created if that default is no longer
  available); and
- the result is process-local. It is not copied from an M1 to an M5, from ARM
  to x86, or into another process. If affinity or a cgroup limit changes after
  startup, restart the process or use an explicit `K3_GEMV_THREADS`.

The final measured phase-B output is already the exact first-call result, so a
successful calibration does not replay the GEMV merely to install its winner.
The deadline includes installing the winning pool; a failure may still require
one cooperative fallback rebuild before normal inference resumes.

Existing exploratory real-expert campaigns repeatedly selected eight or ten
workers instead of four and found sizeable held-out layer gains. A current-code
campaign selected eight and produced a 1.240x held-out geometric mean with all
outputs exact, but normalized host load reached 1.108 against the 0.40 gate.
Those numbers motivate the opt-in; they are not an accepted tokens-per-second
claim. The default stays off until a complete run remains quiet and Linux
x86-64, Linux aarch64, and newer Apple hardware receive the same treatment.

Implementation:
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py),
[`tools/runtime_platform.py`](tools/runtime_platform.py), and
[`tools/test_python_portability.py`](tools/test_python_portability.py).

## Packed spine reads and Metal dequantization

**Status: Packed reads are default; the custom dequantizer is
capability-gated to MPS with a Torch fallback.**

A resident layer contains many int8 payload and scale files. Reading each into
a fresh `bytearray`, transferring it separately, dequantizing into a temporary,
and finally copying into the layer template creates unnecessary allocation,
memset, transfer, and dispatch work.

The packed path reads directly into a small pool of reusable aligned host
buffers, overlaps at most the current and next layer, and coalesces the device
staging. On MPS, a small Metal kernel performs `int8 -> fp32`, fp16 row-scale
conversion, multiplication, and the final write into the template in one pass.
Persistent staging buffers prevent a new hundreds-of-megabytes allocation for
every layer.

On the reference M1 Max, the Torch row-broadcast expression moved the tested
tensor at about 43 GB/s, while the fused Metal kernel reached about 297 GB/s;
the broader per-layer load path moved from 118 ms to 21 ms. The kernel was
bit-exact on every tested tensor and non-zero packed-buffer offset because both
paths evaluate `float(int8) * float(fp16)` in fp32. A missing MPS device,
shader-compilation error, incompatible destination, or non-fp32 mode uses the
Torch expression.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py).

## Reused layer templates and one serial arena

**Status: Default; the shared-arena extension is capability-gated.**

Materializing and destroying a complete decoder module 93 times asks the
framework allocator to manage far more state than the computation requires.
K3 has only three relevant resident shape classes: one first-layer dense KDA
shape, one KDA+MoE shape reused by the remaining 68 KDA layers, and one MLA+MoE
shape reused by 24 MLA layers. Deltafin creates one template for each class and
copies the current layer's streamed weights into its stable parameter views.

The templates execute serially. Where the selected device can allocate the
required single buffer, their non-overlapping parameter views can therefore
share one maximum-sized arena rather than retaining the sum of three
allocations. On MPS, the runtime checks Metal's maximum buffer length first.
A missing capability query, a too-small limit, or an allocation failure keeps
the separate template allocations.

Correctness depends on not reusing a view before the previous asynchronous
consumer is ordered. The packed loader, staging lifetimes, speculative state,
and layer index are tested while alternating real shape classes. Stable shapes
and parameter identities also make optional compilation experiments possible,
but no compilation speed is claimed here.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py) and
[`tools/apple_silicon.py`](tools/apple_silicon.py).

## Keeping KDA recurrence on the selected device

**Status: Default; the historical CPU hop remains forceable.**

An earlier MPS path copied roughly 240 KB of KDA inputs to the CPU, ran the
small recurrent update, and copied the result back. The arithmetic itself was
small, but every transfer drained the accelerator queue. The current default
runs the recurrence on the device that already owns Q/K/V: MPS, CUDA, or CPU.

In a focused MPS measurement, the recurrence took 1.17 ms on-device versus
3.12 ms with the CPU round trip. We do not transfer that result to CUDA or
another Mac. `K3_KDA_RECUR=cpu` retains the historical MPS comparison path,
and speculative verification explicitly keeps its replay on the same
arithmetic route as the original call.

Implementation:
[`tools/fla/ops/kda/__init__.py`](tools/fla/ops/kda/__init__.py) and
[`tools/attn_fast.py`](tools/attn_fast.py).

## Coalesced expert ranges and raw cache files

**Status: Default for streaming misses and local expert storage.**

All six tensors for a K3 routed expert are contiguous in its original shard.
Deltafin verified this for all 82,432 experts and records the fixed offsets.
A remote miss can therefore use one 17.55 MB HTTP range request rather than
six requests. Connections stay alive, the Hugging Face redirect is cached for
its validity window, and file-adjacent selected experts may share a request
without downloading an unselected expert-sized gap. This measured about 6.4x
faster than the original per-tensor remote fetch under the reference
conditions.

The cache file is the exact shard span, published atomically. A local hit needs
no archive parsing, decompression, or checksum traversal before compute; its
six arrays are views at known offsets. An in-memory presence set also avoids a
filesystem metadata lookup for every routed selection.

This HTTP traffic exists only for Hugging Face downloads in a streaming
installation. A full local installation running `tools/kimi_run.py` does not
use HTTP to communicate between Deltafin components.

Implementation:
[`tools/fetch_v2.py`](tools/fetch_v2.py) and
[`tools/k3loader.py`](tools/k3loader.py).

## Parallel expert reads, layer double-buffering, and router lookahead

**Status: Parallel expert reads and spine double-buffering are default;
lookahead is exact with respect to model output and fails closed.**

Handing file-backed views directly to a GEMV can make the compute thread take
page faults one at a time. Instead, a persistent reader pool fills reusable,
page-aligned slots for the selected expert union before the kernel begins.
Slots are bounded and recycled only after their consumer finishes.

At the resident-spine level, one loader thread reads layer `N+1` while the
selected device computes layer `N`. The driver publishes layers in model order,
and any read exception is surfaced before a partially loaded layer can run.

Expert selection is depth-serial: the real router for layer `N+1` normally
cannot run until layer `N` finishes. The lookahead path uses layer `N`'s
pre-MoE hidden state with a cached copy of the next router to predict the next
expert set and begin those reads. Packed int8 router weights are used only when
the live native operator qualifies; otherwise the predictor can retain a dense
representation.

The prediction never decides the computation. The real next-layer router
remains authoritative, a correct prediction turns a demand read into a wait on
already-running work, and a wrong prediction merely wastes I/O. Failed
predictor initialization disables lookahead. Previous-token selection remains
a simpler fallback experiment, but it is not treated as a correctness source.

Implementation:
[`tools/fetch_v2.py`](tools/fetch_v2.py),
[`tools/pilot.py`](tools/pilot.py), and
[`tools/kimi_run.py`](tools/kimi_run.py).

## Lossless n-gram speculative decoding

**Status: Default zero-model fallback, exact-token gated.**

Deltafin searches the generated token history for the longest matching suffix
and uses the following historical token as a free draft. A two-position
forward pass verifies that draft with K3 itself. If it matches, the pass emits
the draft and the next verified token. If it does not, the shared exact
verifier replays the captured KDA inputs for the one committed position and
truncates MLA to the same boundary. The whole-reference restore plus narrow
rerun remains the conservative alternative.

This is especially useful in Deltafin because one forward pass rereads the
resident spine once. A verified T=2 pass can share that fixed work across two
emitted tokens. It is less magical for routed experts: the pass must still read
the union of experts selected by both positions.

Every emitted token is chosen by K3's greedy verifier. Rollback tests poison
drafts deliberately, restore each cache family, and compare future logits and
tokens. Deeper n-gram drafts remain forceable, but their batch/union crossover
is not inferred from the successful universal-assistant result.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/spec_decode.py`](tools/spec_decode.py), and
[`tools/test_spec_replay.py`](tools/test_spec_replay.py).

## Runtime and memory auto-selection

**Status: Default, conservative.**

Cross-platform speed depends on selecting what the host can actually execute,
not matching a product name:

- MPS is preferred when available, then the first CUDA device visible to the
  process, then CPU. An explicit `cuda:N`, `mps`, or `cpu` remains
  authoritative.
- Metal MoE is selected only with an MPS resident device and a working Metal
  library. CUDA MoE is selected only on a CUDA resident device after its
  versioned ABI, shape/layout and on-device known-answer checks; failure keeps
  the resident CUDA path and uses native CPU MXFP4 experts.
- aarch64 selects NEON; x86-64 checks the required compatibility features and
  selects AVX2 only at runtime.
- worker counts respect CPU affinity and Linux cgroup quotas.
- RAM budgets respect host memory and Linux cgroup limits. CUDA expert
  residency budgets from current free VRAM, retains a safety reserve and uses
  stable per-layer quotas instead of treating total VRAM as free memory.
- the int8 spine is selected when present; optional operators still pass their
  own dispatcher, shape, dtype, and first-call gates.

Unknown memory limits and failed capability probes are handled conservatively.
This is what keeps a path measured on an older M1 eligible for a newer Mac
without assuming that every M5, CUDA device, container, or PyTorch build has
the same crossover.

Implementation:
[`tools/runtime_platform.py`](tools/runtime_platform.py),
[`tools/apple_silicon.py`](tools/apple_silicon.py), and
[`tools/kimi_run.py`](tools/kimi_run.py). CUDA-specific qualification and cache
details are in
[`tools/cuda_moe.py`](tools/cuda_moe.py).

## Small fixed-cost cleanups

**Status: Default unless noted.**

Several smaller changes are worth retaining even when they do not justify a
headline:

- Generation runs under `torch.inference_mode()`. Cyclic garbage collection is
  temporarily disabled for the generation and restored in `finally`, avoiding
  periodic object-graph scans while still cleaning up normally afterward.
- Router tracing is off in performance runs. Buffered tracing writes one block
  after a model pass instead of synchronously flushing one JSON record per
  layer.
- Persistent objects are reused for the final tail, packed read pools, device
  staging, pointer/shape arrays, and native workers.
- Cache presence is indexed once and updated at the atomic publication
  boundary rather than rescanning or repeatedly `stat`-ing the 82,432-file
  expert pool.
- A successful two-position speculative verification transfers both argmax
  IDs to the host together. This avoids a second accelerator synchronization
  before emitting the already-verified next token.
- The common local one-token embedding lookup uses positional vector I/O to
  fill the final mutable `torch.frombuffer` owner directly. On macOS and Linux
  this avoids an intermediate immutable allocation and one 14,336-byte
  userspace copy. Multi-token/coalesced reads, remote rows, unsupported
  platforms, partial reads, and errors retain exact guarded paths.
- Speculative decoding keeps one private prompt-plus-output history instead of
  rebuilding `ids + generated` on every pass. Non-speculative requests create
  no history copy. The prefetch setting is snapshotted once before decode, and
  the default threaded-`pread` path skips a whole-token prefetch helper that is
  guaranteed to return immediately.
- When no step logger is requested, generation skips both the per-pass clock
  read and cumulative-token snapshot. The API server also installs no token
  callback for non-streaming requests, and streaming keeps only the
  incremental decoder state rather than an unused second token list.

These changes remove host overhead and allocation noise; they do not carry
separate tokens-per-second claims.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/serve_openai.py`](tools/serve_openai.py),
[`tools/fetch_v2.py`](tools/fetch_v2.py), and
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py).

## Optional cache-write overlap

**Status: Opt-in, default off.**

In a streaming installation, a downloaded expert must eventually be written
to its local cache. `K3_ASYNC_CACHE_WRITE=1` lets inference keep using the
immutable downloaded bytes while a bounded background writer publishes the
cache entry.

The bound covers queued *and active* payloads, so a slow disk cannot retain an
unlimited amount of model data. Mutable inputs are snapshotted before enqueue;
immutable `bytes` can be retained directly. Publication writes a temporary
file beside the destination and atomically replaces the final name. Shutdown
stops admission, drains accepted work, joins workers, and exposes any failure.
When the queue is full, producers apply backpressure rather than silently
dropping cache data.

This can affect the cache-miss path only. It does not speed a complete local
installation, and it remains off until the retained-buffer and write-contention
tradeoff is measured on the target storage.

Implementation:
[`tools/cache_writer.py`](tools/cache_writer.py),
[`tools/fetch_v2.py`](tools/fetch_v2.py), and
[`tools/test_async_cache_write.py`](tools/test_async_cache_write.py).

## Exact memoization for repeated API requests

**Status: Default in the optional API server, bounded and disableable.**

The OpenAI-compatible server is greedy and serializes access to one immutable
model instance. The exact tuple of request mode, prompt token IDs, and output
limit therefore identifies a deterministic response. A small LRU can return
the stored token IDs, text, and finish reason for that *identical* request.
Similar prompts, changed limits, and different modes are misses; no prefix is
guessed.

This is not a normal tokens-per-second improvement. A hit bypasses inference
because the same deterministic request was already completed in that server
process. Set `K3_RESPONSE_MEMO_ENTRIES=0` to disable it.

The server itself is optional. Normal CLI inference is in-process and does not
send HTTP requests to another Deltafin component.

Implementation:
[`tools/response_memo.py`](tools/response_memo.py),
[`tools/serve_openai.py`](tools/serve_openai.py), and
[`tools/test_response_memo.py`](tools/test_response_memo.py).

## What the evidence does and does not say

Deltafin uses three different acceptance bars:

1. **Exact paths** must preserve bytes, operation order where required, cache
   lifetime, rollback, and emitted tokens.
2. **Numerical paths** must stay inside a stated error gate and preserve the
   sequence oracle; they are never casually described as bit-exact.
3. **Quantized quality paths** need explicit logit/token or task-quality
   evidence and a clear fallback.

An allocation reduction is not automatically a speedup, a fast isolated
kernel is not automatically a faster token, and a result on MPS does not prove
CUDA or CPU behavior. That is why the default-off KDA staging path, CPU padded
SDPA, and position-major Metal path are documented here without being silently
enabled everywhere.

The public end-to-end baseline and its measurement method remain in
[`README.md`](README.md). This file explains the mechanisms behind it; it does
not combine isolated wins into a synthetic throughput number.
