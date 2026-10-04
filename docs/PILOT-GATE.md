# PILOT gate: adaptive per-layer prefetch admission and predictor selection

Status: implemented and live-validated (design first landed with the
implementation; this document is the reviewable rationale, with measured
results at the end).

## Problem

PILOT speculative expert prefetch (run layer L+1's bit-identical cached router
on layer L's in-progress hidden state, start background reads for the predicted
top-16) averages ~50% recall but collapses at layers sitting at or immediately
after MLA (full-attention) boundaries — measured 1.2% at layer 13, ~20-30% at
layers 12/24/25/36 on the France/planet oracle prompts. The mechanism is input
staleness: the snapshot PILOT routes on is missing the large global update the
next MLA attention step is about to apply. At those layers nearly every
prefetched expert read (17.5 MB each) is wasted SSD bandwidth on the very
channel that bounds decode speed.

Hardcoding "skip layer 13" would bake in two prompts' worth of evidence and go
stale on the next quantization or model revision. Instead: keep scoring every
prediction on every token (free — both ID lists already exist), and gate only
the *disk reads* per layer on the trailing measured hit rate, with a cheaper
fallback predictor for layers where PILOT is measurably worse than naive
alternatives.

## Correctness envelope

Unchanged and non-negotiable: everything here decides only what to prefetch
speculatively. The authoritative mailbox route from the real router still
selects every executed expert; `read_expert_tile_with_prefetch` still treats
prefetched bytes as an opportunistic cache keyed by exact expert ID, layer,
and layout, and every miss goes down the ordinary checked demand path. A wrong
gate decision costs bandwidth (prefetching garbage) or latency (suppressing a
good prefetch) — never output. The governor is advisory-only by construction:
it holds no I/O resources, its state is never consulted by the demand path,
and it cannot panic the run (out-of-range inputs are skipped, not asserted).

## Where the signal lives

All integration is in `finish_expert_mailbox` (engine.rs), the single choke
point where three things meet per MoE layer per pass:

1. the authoritative routes for layer L (the mailbox), which score any
   outstanding prediction *for* L and seed the prev-token predictor;
2. the provider's PILOT hint for L+1 (`take_prefetch_hint`, taken at L's final
   tile) — the C++ side (`prepare_pilot_rows`) computes the prediction
   unconditionally during routing, and taking the hint is a pure fail-soft
   consume, so scoring continues even for layers whose reads are gated off:
   recovery needs no special path;
3. the decision to submit speculative reads (`try_schedule_expert_prefetch`),
   which is the only thing the gate ever suppresses or redirects.

No ABI or provider_runtime.cpp change is required.

## Scoring

Per layer (1..=92; layer 0 is dense) and per predictor, an EMA of per-row
recall with a sample counter:

- A pass's sample is `mean over rows of |predicted ∩ row_top16| / 16`.
  Denominator 16 per row — not the pass's route union — so prefill chunks,
  speculative multi-row verify passes, and single-row decode produce
  comparable samples. For single-row decode (the case that matters) this is
  exactly the recall the hand-instrumentation measured. Draft rows that later
  fail verification still count; that noise is bounded and advisory.
- `ema += 0.2 * (sample - ema)`, first sample initializes the EMA directly.
  Effective window ~10 samples; `samples` saturates.
- The PILOT predictor's outstanding hint is recorded when taken (whether or
  not reads were issued) and scored one layer later when its target mailbox
  arrives; a mismatched or missing target (sequence teardown) drops it
  unscored. A provider fail-soft miss (no hint) contributes no sample.
- The prev-token predictor needs no lookahead: when layer L's mailbox arrives,
  its remembered previous route set is scored against the same rows, then
  replaced by the newest row's 16 experts (sorted). Across sequence
  boundaries this is precisely "previous token's routing" for decode; across
  request boundaries it degrades to one noisy sample per layer.

## Gating and predictor selection

At layer L's final tile, for target layer L+1:

- Warmup: until a predictor has `K3_PILOT_GATE_WARMUP` samples for that layer
  it cannot be judged. While PILOT is unwarmed the legacy behavior (prefetch
  the PILOT hint) is preserved; the prev-token predictor is eligible for
  selection only once warmed.
- Selection: the warmed predictor with the higher EMA wins the layer. PILOT
  wins ties (it is the stronger predictor globally and the incumbent).
- Gate: if the winner's EMA is below `K3_PILOT_GATE_THRESHOLD`, no speculative
  reads are issued for that layer this pass. Scoring continues regardless, so
  a layer whose recall recovers (different prompt regime, model update)
  automatically resumes prefetching — the same threshold in both directions,
  no hysteresis; flapping is harmless and flip logging is capped.
- The prev-token plan is the remembered 16 IDs (sorted unique, already
  ABI-validated < 896); the PILOT plan is the hint's canonical union (16..=32
  IDs). Both flow through the existing `try_schedule_expert_prefetch` bounds
  checks via `ExpertPrefetchPlan`, the small type that generalizes the
  provider hint.

Expected effect at the measured pathology: layer 13 (PILOT 1.2%) falls to the
prev-token predictor (~30% class); if even that sat below threshold the layer
would go dark on reads while remaining fully scored. Healthy layers (~50-70%)
see byte-identical scheduling to today.

One layer gets net-new coverage: layer 1, the first routed layer, is
structurally invisible to PILOT because its hint would have to come from the
dense layer 0, which produces no mailbox — so its ~272 MB of expert reads
were demand-read cold on every pass. In `on` mode the governor offers the
prev-token plan for layer 1 at sequence start (`plan_sequence_start`),
overlapping those reads with layer 0's compute and both layers' spine binds,
under the same warmup/threshold discipline as every other layer. `measure`
mode declines this plan deliberately: it must keep the legacy read schedule
byte-identical to stay a clean A/B baseline.

## Configuration

Following the `RuntimeConfig` conventions (env-resolved, fail closed on
garbage, echoed on the `[config] resolved:` line):

- `K3_PILOT_GATE` = `on` | `measure` | `off` (default).
  `measure` scores and reports but never suppresses or redirects (A/B
  baseline, and the tool for the open prompt-diversity question).
  `off` restores legacy behavior exactly (no governor is constructed).
- `K3_PILOT_GATE_THRESHOLD` = EMA recall below which a layer's speculative
  reads stop. Finite, in [0,1). Default 0.10.
- `K3_PILOT_GATE_WARMUP` = samples per layer before the gate may act.
  Integer in 1..=100000. Default 16.

## Observability

- Gate flips log one line each (`[pilot-gate] layer 13: reads suppressed
  (pilot 1.2%, prev-token 8.9%, 24 samples)`), capped per layer against
  threshold flapping.
- End of run (`deltafin run`), mirroring the Qwen telemetry block: a summary
  line (passes scored, experts admitted/suppressed at the gate, prev-token
  plans, currently suppressed layers), plus a per-layer table under `--stats`
  of both EMAs, both sample counts, the standing preferred predictor
  (`pilot` / `prev-token` / `warming`; an individual pass can still fall
  through to the other predictor when the preferred one has nothing to offer,
  e.g. on a provider hint miss), and the gate state. The server surface keeps
  flip logs only for now.

## Measured (M1 Max 64GB, int8 spine, scale4 experts, 2026-08-03)

Live measure-mode runs (32 new tokens each) on "France", "planet", and a
`def fibonacci(n):` code prompt confirmed the collapse map is architectural,
not content-driven:

| layer | france | planet | code | prev-token (best) |
|---|---|---|---|---|
| 1 | — (no hint possible) | — | — | 7-8% |
| 12 | 11.5% | 29.7% | 20.2% | ~28% |
| 13 | 1.9% | 1.1% | 1.1% | ~25-29% |
| 24 | 29.5% | 20.2% | 19.3% | ~22% |
| 36 | 33.8% | 34.1% | 33.8% | ~29% |

Mean pilot recall is 59-70% per prompt (per-row decode EMA); expert reads are
58-67% of steady-state pass time. Layer 1's prev-token predictor sits at 7-8%
on every prompt, so the sequence-start plan is correctly suppressed at the
default threshold — dormant machinery unless a workload with stable layer-1
routing appears.

Interleaved A/B on the France prompt (2 reps per arm, 15 chunks each,
outputs byte-identical across all 10 runs — the advisory invariant held under
every arm including total suppression):

- `off` (legacy ungoverned prefetch) vs `on@0.10` vs `on@0.35`: pooled
  per-chunk medians 10.31 / 10.41 / 10.41 s/token post-warmup —
  statistically indistinguishable at this power (~±1-1.5%). The gate's
  addressable traffic is just 0.25% (t=0.10) to 2.5% (t=0.35) of
  speculative bytes, so a null here is expected, not a measurement failure.
- `off` vs `on@0.99` (all speculation suppressed once warmed): the
  no-speculation arm is slower by ~+0.40 s per steady single-token chunk —
  about 4% of steady-state decode, ~1.5% of these runs' total wall-clock.
  Every clean on99 steady chunk (10.45-10.67 s) exceeded every clean off
  steady chunk (9.96-10.15 s); run-level significance ~t=2.7-3.1 with a
  12/14 sign test (p≈0.013), and the entire deficit localizes in the
  expert-read wait phase (6.0-6.2 → 6.5-6.8 s/chunk) while kernel,
  attention, and bind phases stay flat — the mechanism, not just the
  average, points at speculation. (These figures were independently
  re-derived from the raw logs by two adversarial verification passes,
  which also caught and corrected an earlier mis-derived "1.9%" version of
  this number.)

Interpretation: the SSD is near-continuously busy with demand expert reads
plus spine streaming, so speculative prefetch can only exploit the small
compute-only windows; that budget is worth ~4% of steady decode here, and
59-70% recall already captures it. Across the four-point tuning curve (no
speculation → aggressively gated → gated → full), everything except "none"
is indistinguishable, so the default configuration sits at the measured
maximum within this experiment's power; no gate setting can add measurable
speed on this machine. The gate therefore shipped as `on` / threshold 0.10
/ warmup 16: same speed as every alternative measured, plus the redirects at
the collapsed layers, the per-layer diagnostic surface, and self-correction
if a future quantization or model revision shifts the map. A later clean,
freeze-free France-oracle A/B measured full speculation (`off`) at 0.2901
tok/s against the gate's 0.2847 with byte-identical output (+1.9%), so since
2026-08-06 the default is `off`; `on` and `measure` remain available. Recall-improving
extensions (delayed snapshots, learned correctors) inherit the same ~4%
ceiling; larger decode wins on this machine live outside prefetch
scheduling entirely (fewer bytes per miss, bigger caches, spine-stream
reduction).

## Re-measuring this

Decode wall-clock on this machine swings 20%+ between identical runs, so the
numbers above came from short runs and are correspondingly soft — the
speculation figure rests on four runs of one 32-token prompt. To harden them,
or to measure a new predictor, use the sustained A/B harness.

The harness (`scripts/ab_pilot.sh` plus its analyzer) is **local measurement
tooling and is deliberately not tracked**: the compiled-only publication
policy forbids shipping interpreted sources, so these commands apply to a
working copy that has it, not to a fresh clone. What follows is recorded here
because the measurement design — not the script — is the part worth keeping.

```bash
./scripts/ab_pilot.sh -n 10                   # overnight: 10 A/B pairs
./scripts/ab_pilot.sh -n 1 -t 200             # one pair, shorter
./scripts/ab_pilot.sh -a gate -n 10           # governor vs legacy instead
./scripts/ab_pilot.sh --smoke-only            # preflight only
```

It runs both arms inside each iteration, with the order drawn randomly and
then inverted on the next iteration so every pair is balanced — a fixed order
is a real confound, since going first measured a 7-9% page-cache penalty here,
roughly twice the arm effect, and the first campaign always ran the control
first. Iteration counts are forced even for the same reason (an odd count
leaves one arm leading once more than the other), and the analyzer
independently re-checks the balance and drops a surplus iteration if a
campaign gets interrupted mid-pair.

It verifies that every run generated byte-identical text before trusting any
timing. Each iteration contributes one independent sample rather than
cross-pairing runs (which is what made the first analysis's error bars
several times too tight), and that sample is normalized to seconds/token per
chunk and reduced with the **median**, not the mean: the engine periodically
grows its attention cache, stalling one chunk by ~2 minutes, and a single such
chunk supplied 79% of an iteration's mean in validation. Chunks known to
overlap a cache-growth retry are identified from the engine's own log lines
and excluded outright before the median is even taken; the count excluded is
always reported, never silent.

A fail-closed memory-admission refusal — the engine correctly declining to
grow the cache rather than a bug — is retried up to 5 times, 5 minutes apart,
by default (`--memory-retries`, `--memory-retry-wait`); a genuinely different
error is never retried. An overnight campaign here lost 5 of 12 runs to this
before retries existed. `corespotlightd`/`mds_stores` (Spotlight) were found
holding several GB of memory since a reboot 45 minutes before the campaign
started — a plausible, but not confirmed, contributor: no kernel memory-
pressure/jetsam events fired all night, and `mdfind` later showed zero
indexed content in this repo's own unexcluded directories, so it may simply
be Spotlight's baseline resident cost rather than active work on this repo
specifically. `deltafin setup` now excludes the whole model root from
Spotlight automatically regardless (see below); it costs nothing and can
only help if this ever recurs.

The engine now also prevents rather than survives these refusals, two ways.
First: every admission failure ever recorded — across four campaigns — was
the full-speculation arm, never the suppressed arm, and the mechanism is the
speculative-prefetch arena, which lazily materializes up to ~1.1 GiB of
slabs it retains for the whole run while measured admission margins were
0–340 MiB; a failed growth proof now sheds those advisory slabs (they
re-grow lazily on the next speculative read) and re-proves immediately
before waiting on anyone else's memory. Second: a bounded request
(`--max-new`, envelope ≤ 512 tokens) proves its whole context ceiling at its
first chunk, so every later growth boundary takes the committed-capacity
fast path with no live mid-run proof — pressure arriving mid-run (opening
another application) can no longer fail a generation that was admissible
when it began; a declined envelope just falls back to incremental growth.
Verified live: the envelope pre-admits at request start and the output is
byte-identical to the pre-fix binary (same preflight sha).

**Open finding, not yet resolved**: tracing individual outlier chunks (not
tied to cache growth) turned up several, scattered through one campaign, where
wall-clock differed by 6-28s between arms while the profiled per-layer
compute phases for the same window differed by only 1-5s — meaning the gap
sits entirely outside the instrumented path, and it was directionally
consistent (the fully-suppressed arm always slower, never the reverse) rather
than random. This could be the real, concentrated cost of zero speculative
overlap on specific expensive positions, or an engine/OS scheduling quirk
particular to the never-prefetch code path — undetermined. It isn't tied to
any log marker, so nothing here excludes it automatically; more iterations
average over it, but a smaller mean-vs-median gap than iteration 3's (which
growth-exclusion did resolve) may still appear and should not be read as
disqualifying the run.

**Measurement overhead**: `--stats` used to also enable full per-layer phase
profiling (`[phase layer N/92] ...`, ~93 lines/token) as a side effect of
`collect_profile = stats || events.is_some()`, even though the cheap
cumulative summary line needs neither that data nor per-layer timer
collection. Every A/B campaign run so far paid for that verbosity without
asking for it — plausibly a contributor to the open finding above, since
writing that much text competes for the same disk the benchmark measures.
`--layer-profile` is now a separate, independent flag (verified live: `--stats`
alone prints zero `[phase layer]` lines; `--stats --layer-profile` prints
both; `--layer-profile` alone prints the phase lines without the summary
line). The harness needs no change to benefit — it only ever passed `--stats`,
never `--layer-profile`, so future runs get the lighter default for free.
Whole-task wall-clock (`wall_s`/`internal_s` in the manifest) was always
completely external to any of this — it wraps the process from the shell and
depends on nothing the engine prints — and remains the right sanity check for
a single dramatic outlier run; per-chunk analysis exists because a growth-
event stall is only ~3-7% of a whole run's total (not the order-of-magnitude
spike that would make eyeballing whole-task numbers alone sufficient) and
because it multiplies the usable sample size roughly 150x over one number per
run, which matters most exactly when iteration counts are constrained by
overnight practicalities.

Budget roughly `tokens x 12s x 2` per iteration plus ~3 min prefill per run,
plus whatever memory retries add (rare once ambient pressure is under
control). A long prompt is a trap: a 405-word prompt measured ~20 min of
prefill per run, so the built-in prompt is deliberately short and context
grows during generation instead. `-n 10 -t 150` is a reasonable overnight
target once ambient memory pressure is controlled.

## Out of scope (deliberately)

- Delayed snapshots for bad layers (idea #3) — needs C++ scheduling changes;
  the gate's telemetry will identify the layers worth the invasive change.
- The learned per-layer corrector (idea #4) — offline harvest/train first via
  the router-trace infrastructure; nothing engine-side until it beats the
  stale baseline on held-out traces.
- Per-predictor kill switches and server-side telemetry endpoints — add if
  A/B practice demands them.
