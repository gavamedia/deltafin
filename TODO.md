# Deltafin TODO — prioritized

Updated 2026-10-07. Built from a review of the open PRs and issues on
`gavamedia/deltafin` plus every planning doc in this repo (PLAN.md,
SPEED-ROADMAP.md, BRAINSTORM-SPEED.md, docs/*, research/, experiments/),
cross-checked against git history and the code.

Guiding rule: maximize decode tok/s; the only hard boundary is K3 output accuracy.

## Where things stand

- Headline: **0.2901 tok/s** (M1 Max, France oracle, Qwen + int8 spine + scale4,
  `K3_PILOT_GATE=off`). Chat is roughly 0.08–0.10 tok/s.
- Time per token (chat, 10.69 s/token): expert reads **60.5%**, resident attention 17%,
  bind/upload 10%, expert kernel 6%. Decode is expert-read-bound. Issue #25 sees the
  same on an M4 Max, and so does #33.
- The public repo (`441fbfd`, Aug 6) is **2 commits behind internal**: 775a0af (auto pin
  tier) and dd8327b (admission fix). On top of that, **~2,400 lines of finished WIP have
  been uncommitted since Aug 9** (per-expert early drain plus the CI changes).
- Community data point (#33): an M5 Max fork reaches **~1.0 tok/s** by spreading expert
  reads over four drives. Going from 1 drive to 4 gave about **1.9×**, with byte-identical
  MXFP4 experts.

---

## P0 — Housekeeping that unblocks everything (hours, do first)

1. ~~**Commit the WIP.**~~ Done 2026-10-03.
   - Commit early drain (`K3_EXPERT_EARLY_DRAIN`, default off): it is exact, but the
     measurement was null.
   - Commit the CI legs and `tools/expert_pin_ab_prompts.json` as their own commits.
2. ~~**Merge PR #26** (expert_heat flush-test flake, #24).~~ Fixed internally 2026-10-03
   (312fa3e, plus a second racy spine timing test); the public PR itself is untouched.
   - Test-only and correct. Replaces the 1,000-iteration `yield_now` loop with a 30 s
     deadline and 1 ms sleeps.
   - The same flaky loop still exists in internal `expert_heat.rs:839`.
3. ~~**Merge PR #30** (empty `prompt` → 400, #27).~~ Fixed internally 2026-10-03
   (2d01473); the public PR itself is untouched.
   - A one-combinator fix with a regression test, and correct.
   - Still unfixed internally (`openai/server.rs:598`).
4. **Publish internal → public** (yours to do; Claude leaves the public repo alone):
   everything since 441fbfd, including the #24/#27 fixes.
5. **Close PR #14 (native Windows) with thanks, and redirect it.**
   - It targets the Python runtime (`build_native.py`, `spine_io.py`, `fetch_v2.py`,
     `kimi_run.py`, `cuda_moe.py`), and all of those have been deleted since.
   - Invite a re-port against the Rust/C++ runtime. Its `win_compat.h` shim and the
     `ReadFile`+`OVERLAPPED` positional-read design (11.5 GB/s) carry over directly.
6. **Ask the issue reporters to retest after publishing.**
   - **#20**: the CUDA expert cache was fixed in f590b7d.
   - **#22**: MPS server poisoned after a long follow-up request. dd8327b's envelope
     admission and slab shedding probably cover part of it.
   - **#25** and **#33**: reply with thanks.
7. ~~**Fix stale docs.**~~ Done 2026-10-03 (PILOT-GATE default, `K3_SPEC_DEPTH` warning;
   SPEED-ROADMAP.md is local-only and was updated in place).
   - PILOT-GATE.md:191 still says the default is `on`.
   - SPEED-ROADMAP.md shows #5 and #3 unchecked, but both are done.
   - Remove the `K3_EXPERT_STREAM_NOCACHE=0` "free win" note (see P2-14).

## P1 — Biggest speed levers

8. **Multi-drive expert striping (C144/C145). This is the top lever.**
   - **Status 2026-10-03: built and validated by emulation; real drives still
     needed to confirm speed.**
     - `K3_STORAGE_HOMES` spreads spine and expert reads over every drive
       holding a copy. Drives are chosen per read job from shared load counters
       and measured rates, with failover. `deltafin populate-storage-home`
       makes verified copies: the spine first, then the hottest experts.
     - Emulated on this machine at 0.4x scale with real routes. One extra
       drive gives 1.39x, three give 2.10x (ideal 2.21x); T=8 verification
       gives 2.11x.
     - Throttling, unplugs and wrong speed settings are absorbed at 95-99% of
       the best achievable.
     - Partial drives (spine + hottest 30% of experts, about 500 GB) match full
       mirrors.
     - Predicted real T=1: about 1.30x / 1.54x / 1.67x with 1/2/3 extra drives.
       Compute dominates beyond that.
     - Exactness: 56.4 GB of real reads came back byte-identical, and the full
       engine produced identical text.
     - Details are in OPTIMIZATIONS.md (2026-10-03).
     - **Follow-ups:**
       - Measure on real enclosures.
       - Add demand-over-spine priority (it lifts spine-only drives from 1.74x
         toward 2.10x).
       - Once storage stops being the limit, revisit compute overlap
         (#16, C141).
   - **Why:**
     - The docs name more storage bandwidth as the only credible exact path past
       ~0.46 tok/s on the internal SSD (BRAINSTORM "1 tok/s campaign").
     - #33 has now shown it working: 0.55 → 0.96 tok/s from 1 → 4 drives.
   - **Fork code:** #33 offered to send its fork's code as separate PRs. Ask for just:
     - the multi-home expert router (`K3_EXPERT_DIR_B/C`, `K3_EXPERT_HOT_DIR`,
       `K3_SPLIT_ETA`);
     - usage-ranked replica placement;
     - tier balancing.
   - **Do not adopt** their arrival-order tile accumulation (`K3_ARRIVAL_GROUPS`).
     - They report it flips near-tie tokens, which breaks the accuracy rule.
     - Our early drain already keeps the canonical reduction order and showed no gain on M1.
   - **Hardware:** 2–3 Thunderbolt NVMe enclosures for the M1 Max.
     - Target is ~15 GB/s aggregate with the internal SSD. That figure is the C144 gate.
     - Measure what each port really delivers before buying the full set.
   - **Tail latency:** #33 found each layer is paced by the *slowest* of its 16 reads, not by
     total bandwidth. Hedged or duplicate reads to the second copy of an expert fall out of
     this work naturally.
9. ~~**Add expert disk bytes and GB/s to `--stats`** (S).~~ Done 2026-10-03.
   - `[stats] disk:` reports the bytes each reader physically read this run (spine,
     expert demand, expert prefetch), the effective GB/s and GB/token.
   - First reading (France, T=1, drafter memory-rejected): 1,465 GB in 201 s =
     **7.29 GB/s, the SSD's ceiling**. That is 54 GB of spine + 32 GB of experts per token.
     **Prefetch read more expert bytes (311 GB) than demand did (228 GB).**
   - **New follow-up:** on a saturated SSD every mispredicted speculative read
     costs time directly. Count PILOT prefetch bytes that were never consumed
     and A/B gated vs ungated speculation at T=1 without a drafter. The 2026-08-06
     `off` default was measured with the Qwen drafter active.
10. ~~**Fix pin-tier admission (PR #23).**~~ Done 2026-10-03, by measured marginal value.
    - **New order:** Qwen drafter > a spine that fits completely > pinned experts >
      a spine that only partly fits. The tier is sized after spine and Qwen
      selection, by binary search over whole experts (PR #23's approach,
      credited).
    - **On this M1:** the auto roster had grown to about 1,700 experts (27 GiB) as heat
      history matured. Reserved up front, it **pushed out the Qwen drafter**:
      11.83 s/token.
    - **Now:** the same request is clamped to 220 experts (3.5 GiB) and Qwen is
      admitted (13/13 drafts accepted): **4.94 s/token, 2.4x faster**, with
      identical output.
    - **New follow-up:** only a ~2 GB roster was ever measured (+15-19%). On a quiet
      host with lots of free RAM, auto would now pin up to about 27 GiB in place of
      every spine layer. A/B auto against a 2-4 GB cap before trusting the large roster.
11. ~~**EAGLE-3/MTP hidden-state drafter**~~ (SPEED-ROADMAP #7). Done 2026-10-03.
    - EAGLE-3.1 (`lightseekorg/kimi-k3-eagle3.1-mla`) in the chat/server slot,
      `K3_EAGLE3=auto`. Same 64-token chat, byte-identical: **11.38 → 5.62 s/token**
      decode, whole run 1005 → 643 s. A = 3.65 tokens per verify pass in-engine
      (offline 3.26/4.17/4.77 at 3/5/7 drafts).
    - It needed two enablers, now in: verify commits replay the KDA recurrence
      (19.4 MiB a row instead of a 453 MiB state per row), and the slot's widest
      verify is reserved at startup. Without them every proposal was refused by
      live memory, or paid for a rerun (8.26 s/token).
    - **Follow-ups:**
      - The width policy jumps to 7 after a full match. The fitted cost
        (9.2 s + 2.0 s/row) and acceptance favor 4–5, worth about 5%; an online
        width-by-cost controller would also adapt per host.
      - Time a long chat and a code prompt; anchored training claims flat
        acceptance at long context.
      - On a host with resident spine, the fixed per-pass cost shrinks and
        deeper chains pay more.
12. **C120 certified wide verifier.**
    - Bisect the T=13 divergence and lift `MAX_EXACT_DRAFTS` (engine.rs:152) past 8.
    - Worth ~0.42–0.46 vs 0.34–0.36 tok/s, and it gets more valuable once #8 adds
      bandwidth.
    - Unblocks C142 (chained DSpark, T=15) and C130.

## P1 — Accuracy (the one hard constraint)

13. **A quality gate for the default int8 spine.**
    - The default is non-weight-exact.
    - STORAGE.md says validation "continues", but the native runtime has no
      avg-NLL/top-1-agreement harness against BF16 (PLAN §5.7).
    - Build it before any further speed default ships on top of int8.

## P2 — Medium levers and cheap wins

14. **Knob sweep on a quiet host**: `K3_SPINE_RESIDENT_GB` 24/30/34 and
    `K3_EXPERT_READ_THREADS` 4/8/12.
    - **Drop `K3_EXPERT_STREAM_NOCACHE=0`.** #25 measured −52% (base) and −38% (full stack)
      even on 128 GB.
    - Add a docs warning about `K3_SPEC_DEPTH=4`, which cost 35% in #25.
15. ~~**Reserve only the active drafter**~~ (SPEED-ROADMAP #4). Done 2026-10-03.
    - A direct run reserves only a drafter its requests can reach. Chat no longer
      holds Qwen (1.7 GiB went to pins: 156 → 280+ experts). Raw runs never held
      the slot, and the slot holds EAGLE-3.1 (3.5 GiB) or DSpark, never both. The
      server still holds both, because it serves both request shapes.
16. **Overlap expert reads behind compute** (SPEED-ROADMAP #6; C141 conveyor; C126
    deadline-aware prefetch).
    - Note the PILOT data caps the speculative-prefetch budget at about 4%.
17. **C139 in-kernel ANS over MXFP4 experts.**
    - About 10.4% fewer expert bytes, which is direct read time.
    - Large effort: the decode has to be fused into the kernel.
18. **Prefill re-reads each layer's experts 8×** (reported in #33; verify it applies here).
    - TTFT only, not decode. On #33 a 512-token prompt waited 375–799 s for its first token.
19. **Nightly idle-gated A/B harness** (SPEED-ROADMAP).
    - Most 2026-07 candidates were never timed because the host was busy.
    - Makes everything below cheap to settle.
20. **PILOT-GATE.md:270**: unexplained 6–28 s outlier chunks in the suppressed arm.

## P3 — Small exact candidates (each ≤ a few %; batch-test via #19)

21. **C162 fused q/k/v short-conv**: 10–19 ms per pass; needs a bitwise canary.
22. **N5 `K3_METAL_POSITION_BATCH`**: +0.75–4.7%; needs a repeated ABBA.
23. **C148 flat Metal position dispatch**: finish its campaign.
24. **C133 threadgroup-major MXFP4 repack**: the kernel runs at 150 vs 290 GB/s.
25. **C140 TinyLFU dynamic expert cache**: partly superseded by the pin tier.
26. **C124/C137 economic width selector calibration**, **C127 n-gram router** and
    **C150 Qwen slack=1**.
27. **Still unmeasured**:
    - C131 split-K q8, C134 residency sets, C135, C136, C129, X11 fused argmax.
    - SPEED-ROADMAP #10 scale4 coalescing and #11 DVFS.
28. **Re-time any surviving Python-era candidates** in one quiet session, if they still
    apply to the compiled runtime:
    - CPU MoE kernels;
    - C31/C41 overlap;
    - the KDA→Metal ports.

## P4 — Not single-stream speed / platform / later

29. **Windows on the Rust runtime**: a re-port of PR #14.
    - The Windows CPU target already lives (`.github/workflows/windows-native.yml`,
      MSVC toolchain, PE dependency audit, `platform.rs` accepts `windows/x86_64`).
      CUDA on Windows is deliberately hard-blocked in the build graph.
    - Also: HIP CI (the runner runs out of disk) and AMD hardware evidence.

    ### 29a. Windows CUDA enablement (build graph)

    The runtime side is already portable: `provider_device.h:25` handles `_WIN32`,
    `platform.rs` accepts `cuda:N`, and the `cuda-moe`/`bf16-cuda`/`mla`/`kda`
    native-test specs are `platform: Any`. What is missing is the build graph,
    which is Linux-shaped in five places. Order matters; 1–4 unblock an
    operator-supplied `DELTAFIN_TORCH_ROOT` + matching `cu13x` toolkit, 5 makes
    the bootstrap self-serve.

    - [x] 1. **PE runtime-ABI detection.** `detect_gpu_runtime`
      (`native/deltafin-native-build/src/lib.rs:4826`) returns `None` off Linux and
      searches ELF strings. On Windows scan `c10_cuda.dll`/`.lib` for
      `cudart64_12.dll` / `cudart64_13.dll` / `amdhip64.dll` with the existing
      `file_contains` (a raw byte search, PE-safe). Without this the
      `run_production_build` "GPU libraries require an identified runtime ABI"
      panic (`lib.rs:689`) fires even after the explicit blocks are removed.
    - [x] 2. **Open the four explicit gates.** `build_provider_artifacts`
      (`lib.rs:1171`, Windows + CUDA pair panic), `build_cuda_kernel`
      (`lib.rs:3582`, Linux-only and `ON` panic), `validate_explicit_torch_root`
      (`lib.rs:4762`, `ON` non-Linux panic), and keep HIP Linux-only.
    - [x] 3. **Windows-aware NVCC/toolkit discovery.** `discover_nvcc`
      (`lib.rs:3786`) and `cuda_toolkit_root` (`lib.rs:3949`) look for `bin/nvcc`
      (not `nvcc.exe`); `find_on_path` (`lib.rs:4233`) has no PATHEXT handling.
      `find_cuda_provider` (`lib.rs:3500`) and `cuda_runtime_directory_optional`
      (`lib.rs:4007`) look for `libcudart.so*` under Linux layout; Windows needs
      `lib/x64/cudart.lib` + `cudart64_*.dll`. The exact `12.6`/`13.0` gate
      (`lib.rs:3709`) also rejects a newer `13.x` toolkit.
    - [x] 4. **MSVC-shaped nvcc invocation.** `build_cuda_kernel`
      (`lib.rs:3619`) passes `-Xcompiler=-fPIC` (cl.exe warns D9002) and hardcodes
      `.o` outputs (`lib.rs:3600`); add host flags matching the CRT (`/MD`,
      `/EHsc`, `/Zc:__cplusplus`) and use `object_file_name`. The rpath link args
      (`lib.rs:709`, `739`) are unconditional `-Wl,-rpath,...`; Windows `link.exe`
      rejects them, so the loader must find `cudart64_*.dll` beside the exe
      (`deploy_runtime_libraries`, `lib.rs:813`, already copies `torch/lib` DLLs).
    - [ ] 5. **Bootstrap artifact pin.** `native/deltafin-bootstrap/src/lib.rs`
      pins only the CPU Windows wheel (`lib.rs:189`); add a `cu13x` Windows wheel
      pin (sha256 + size + files manifest) and extend `required_libraries`
      (`lib.rs:111`) with `torch_cuda`/`c10_cuda` and the CUDA runtime DLLs.
    - [ ] 6. **Evidence.** Extend `windows-native.yml` with a GPU leg
      (`cuda-moe`, `bf16-cuda`, `provider-precision`), confirm Windows VRAM
      detection (`engine.rs:7917`), and update `PLATFORMS.md` +
      `COMPILED-RUNTIME.md:107`.

    Steps 1–4 landed on `feat/windows-cuda` and were verified on real hardware
    (RTX PRO 4000, Blackwell sm_120, CUDA 13.3): `bf16-cuda` and `cuda-moe`
    native tests PASS on the GPU, and `deltafin doctor` reports one CUDA device
    with both exact kernels compiled and every CUDA canary passing. The CUDA
    LibTorch root is a manually extracted `torch-2.13.0+cu130` wheel supplied
    through `DELTAFIN_TORCH_ROOT`. Step 5 (bootstrap pin) and the GPU CI leg of
    step 6 remain.
30. **Context beyond today's bound.**
    - The expanded fp32 MLA cache is 512 MiB per layer.
    - Exact compact MLA (C30) was rejected because it is not bit-exact.
31. **Continuous batching (C146)**: 1.6–3.4× aggregate across requests, no single-stream gain.
32. **Exploratory**: vision path, ANE, MLX bakeoff, per-chip autotuner, M5 retests of M1
    negatives.

## Done / rejected — do not reopen on M1

**Done:**
- Auto pin tier (775a0af).
- PILOT gate off (eeee5ae).
- Admission fix (dd8327b).
- Compiled runtime and route mailbox.
- scale4.
- `--stats`/`--layer-profile` split.

**Rejected:**

| Candidate | Result |
|---|---|
| Early drain | null |
| C30 compact MLA | not exact |
| C121 | −41% |
| C132 | −9.7% |
| C157 | −20% |
| C160 | 1.6–2.2× slower |
| C149, C152, C154, C155, C156, C153 | no gain or not exact |
| More resident spine | +0.23% |
| NOCACHE=0 | −38 to −52% |
| fp16 MPS | — |
| Top-k cuts | — |
| Uniform int4 spine | — |
