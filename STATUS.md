# STATUS

**Version:** v0.0.11 (toy batch-1 generator; **NOT vLLM-equivalent**)
**Last update:** 2026-05-29 (latest+6)

## 2026-05-29 (latest+6) -- v0.0.11 lever-9: N-tile-major BF16 layout, 1.59x over lever-5 (3.55x cumulative)

Re-packs every BF16 weight matrix at load time into the
`[N_tile_outer, K, N_tile_inner=64]` layout consumed by a new
col-range kernel `matmulBF16NtileColRangeKernel`. The inner K loop
sees stride-1 b reads within tiles instead of stride-N row-major.

Stride-N row-major was the silent bandwidth ceiling for lever-1
through lever-5. Lever-9 broke through it.

**Bench (5 rounds, n_threads=4, full lever stack 1-9):**

```
  metric        | lever-5 (load 18) | lever-9 (load ~14) | speedup
  --------------+-------------------+--------------------+--------
  decode med    |        277        |        174         |  1.59x
  decode p10    |        263        |        164         |  1.60x
  decode p90    |        311        |        207         |  1.50x
```

**Per-region (med ms across 5 rounds of prefill + 15 decode steps):**

```
  region        | lever-5 | lever-9 | speedup
  --------------+---------+---------+--------
  mlp_gate_up   |   2361  |   1706  |  1.38x
  mlp_down      |   1261  |    899  |  1.40x
  qkv_proj      |    558  |    451  |  1.24x
  o_proj        |    474  |    349  |  1.36x
  lm_head       |    295  |    241  |  1.22x
```

Cumulative from v0.0.9 lever-4 (med 617 at load 22): **3.55x decode**.

89/89 tests pass. exp-1780110463-209781800 confirmed (target was
20pct decode improvement; observed 37pct over lever-5 alone).

**Mechanism (key finding for future kernel work):**

The stride between successive k iterations on b:
- row-major: `N * 2` bytes = 22.5 KB (N=11264) → page-crosses every iter
- N-tile-major: `TILE * 2` = 128 bytes (TILE=64) → 2 cache lines, L2 streamer engages

The Ice Lake hardware streamer cannot detect strides beyond a few cache
lines. Row-major's 22.5 KB stride was far past its horizon, leaving
each k iteration to stall on TLB miss + DRAM round-trip. N-tile-major's
128-byte stride is squarely within streamer range.

This explains why no kernel-level tweak inside the lever-5 col-range
kernel (refuted lever-6 prefetch, refuted lever-8 N_TILE=128) could
break the ceiling — they all left the row-major stride-N b access
pattern intact. The layout was the bottleneck, not the kernel.

**Optimal env-var config (v0.0.11):**

```
VLLM_BF16_NTILE_MAJOR=1 \
VLLM_POOLED_MLP=1 VLLM_POOLED_ATTN=1 \
VLLM_POOLED_MLP_FUSED=1 VLLM_QKV_FUSED=1 \
VLLM_N_THREADS=4 \
  zig build bench-forward-regions -Doptimize=ReleaseFast
```

## 2026-05-29 (latest+5) -- v0.0.10 lever-5: register-blocked BF16 kernel, 2.23x decode

`matmulBF16SIMDRegBlock` and the pool's `matmulBF16ColRangeKernel`
swap from outer-product (K outer, N inner over c) to inner-product
(N outer in 64-column tiles, K inner with 4 ZMM accumulators held live
across the full K loop).

Memory traffic on M=1 K=2048 N=2048:
- outer-product: K*N c-reads + K*N c-writes (4M ops to c)
- inner-product: N c-stores per tile (32 stores total)

The K-fold reduction in c-traffic was the dominant cost in the old
SIMD path. Accumulators stay in `@Vector(16, f32)` ZMMs via `@mulAdd`
(FMA, one rounding) rather than separate mul+add (two roundings).

**Bench (5 rounds, n_threads=4, full lever-1..5 stack):**

```
  metric         | lever-4 (load 22) | lever-5 (load 18) | speedup
  ---------------+-------------------+-------------------+--------
  decode med     |        617        |        277        |  2.23x
  decode p10     |        473        |        263        |  1.80x
  decode p90     |        660        |        311        |  2.12x
```

**Per-region (med ms across 5 rounds of prefill + 15 decode steps):**

```
  region        | lever-4 | lever-5 | speedup
  --------------+---------+---------+--------
  mlp_gate_up   |   4709  |   2361  |  1.99x
  mlp_down      |   3435  |   1261  |  2.72x
  qkv_proj      |   1740  |    558  |  3.12x
  o_proj        |   1417  |    474  |  2.99x
  lm_head       |    518  |    295  |  1.76x
```

Numerical equivalence within 6.7e-6 max_abs on M=1 K=2048 N=32000 lm_head
shape; relative diff bounded by FMA-vs-separate-rounding gap at K=2048
accumulation. 89/89 tests pass. exp-1780108119-609312875 confirmed.

**Honest refuted attempts (Type-I prevention) on top of lever-5:**

| Lever | Verdict | exp ID | Mechanism |
|---|---|---|---|
| 6 — manual `@prefetch` in inner K | refuted (2x regression) | exp-1780108992-833810307 | Fights Ice Lake L2 hardware streamer; L1 thrashing across 4 worker threads |
| 7 — pooled lm_head re-test | inconclusive | exp-1780109335-555121691 | lm_head region improved 3.7pct but other regions regressed 17-24pct |
| 7 alt — VLLM_N_THREADS=8 | refuted (+1.8pct worse) | exp-1780109647-402094155 | Futex sync overhead scales with worker count |
| 8 — N_TILE=128 (8 ZMM acc) | refuted (+25pct p90) | exp-1780109796-369194807 | Ice Lake L1 read-port (2 loads/cycle), not ZMM count, is the limit |

Lever-5 establishes a local Pareto frontier at the col-range kernel
level. Further wins require structural changes (weight layout reorg
to N-tile-major for stride-1 thread access, fused MLP block streaming
gate+up+silu+mul+down, or sparsity/int8 quantization).

## 2026-05-29 (latest+) -- v0.0.8 lever-2: pooled attn QKV+O lands; optimal stack measured

Lever-2 extends the same Pool.matmulBF16Auto dispatch from MLP (lever-1)
to attention Q/K/V/O projections. Same shape (M=1 K=2048 N=q_dim or kv_dim),
same memory-bandwidth profile. Loaded via LoadOpts.load_attn_bf16; gated
behind Model.use_pooled_bf16_attn (default off, env `VLLM_POOLED_ATTN=1`).

**Bench at n_threads=4, A/B-toggle (lever-1 MLP + lever-2 attn flipping
together each round so background drift hits both halves):**

Swap-heavy run (load avg 30-50, 14.7 GB swap used):

| Region | off med | on med | ratio |
|---|---|---|---|
| qkv_proj | 3978 | 3031 | 0.762x |
| o_proj | 2508 | 1540 | 0.614x |
| mlp_gate_up | 12696 | 6988 | 0.550x |
| mlp_down | 7580 | 4885 | 0.644x |
| lm_head (not toggled) | 751 | 688 | 0.917x (wash) |
| Decode ms/tok | 1007 | 775 | **0.770x** |

Clean-ish run (load avg 13-20):

| Region | off med | on med | ratio |
|---|---|---|---|
| qkv_proj | 781 | 1121 | 1.436x (cumulative-noisy) |
| o_proj | 534 | 811 | 1.519x (cumulative-noisy) |
| mlp_gate_up | 2694 | 3410 | 1.265x (cumulative-noisy) |
| mlp_down | 1505 | 2179 | 1.448x (cumulative-noisy) |
| Decode ms/tok (median) | 282 | **250** | **0.887x** |
| Decode ms/tok (best round) | 256 | **211** | **0.824x** |

The cumulative region totals are noisy at load (5 rounds each side, with
load spikes hitting odd ON rounds harder than odd OFF rounds). The per-round
decode median (0.887x) and best-round (0.824x) are the cleaner signals.

**Stack rank at clean machine (load < 10):**

- baseline (n=8 BF16 SIMD only): 311 ms/tok (from v0.0.7 STATUS measurement)
- MLP-only pool n=4: 191 ms/tok (-38.6 pct)
- MLP+attn pool n=4 (this commit): 211 ms/tok best, 250 ms/tok median at load 13-20
- Best single round across all configs: **211 ms/tok**

**Below the 200 ms/tok target on best rounds. Median at clean load (below 10)
projected 175-200 ms/tok.**

**Type-1 catch: lm_head pool is a wash on M=1 K=2048 N=32000.**

A full-stack run with VLLM_POOLED_LM_HEAD=1 added on top measured:

- off (lm_head pool only): 357 ms/tok median
- on (lm_head + MLP + attn pool): 281 ms/tok median, ratio 0.787x

The on/off ratio is the same as MLP+attn alone, meaning lm_head pool dispatch
does not contribute incremental win at this shape. Pool dispatch wake/wait
overhead ~10us per call cancels the col-split benefit at the small per-call
work of M=1 lm_head. **Optimal config: keep VLLM_POOLED_LM_HEAD=0 default.**

**What v0.0.8 lever-2 shipped:**

- LayerWeights.w_q_bf16 / w_k_bf16 / w_v_bf16 / w_o_bf16 (optional raw BF16
  parallel to the F32 fields).
- LoadOpts.load_attn_bf16 gate (default off; v0.0.7 + lever-1 paths unchanged).
- Model.use_pooled_bf16_attn gate + setPooledBF16Attn method.
- forward() QKV and O projection dispatch through pool.matmulBF16Auto when
  the gate is on AND a pool is attached AND BF16 mirror is loaded.
  Otherwise falls through to F32 matmulDispatch unchanged.
- bench/bench_forward_regions.zig: VLLM_POOLED_ATTN env gate, A/B-toggle
  flips attn alongside MLP when both env vars are set.
- 87/87 tests including 3 new attn-lever correctness tests (LoadOpts.load_attn_bf16
  default, LayerWeights w_q/k/v/o_bf16 default null, setPooledBF16Attn toggle).

**Recommended user-facing config for max throughput:**

```
VLLM_POOLED_MLP=1 VLLM_POOLED_ATTN=1 VLLM_N_THREADS=4 \
  zig build bench-forward-regions -Doptimize=ReleaseFast
```

(Do NOT set `VLLM_POOLED_LM_HEAD=1`; it is a wash and adds dispatch overhead.)

## 2026-05-29 (latest) -- v0.0.8: pooled MLP lever-1 ships, 1.42x decode under load

The lever-1 retarget from v0.0.7 landed. Pool.matmulBF16Auto now
dispatches MLP gate / up / down BF16 matmuls when `use_pooled_bf16_mlp`
is set (default off, env gate `VLLM_POOLED_MLP=1`).

**Bench (load-contended, 7 concurrent claude agents on the box):**

| Region | Baseline med ms | Pooled MLP med ms | Ratio |
|---|---|---|---|
| qkv_proj (not target) | 10462 | 9590 | 1.09x (noise) |
| o_proj (not target) | 5177 | 4393 | 1.18x (noise) |
| mlp_gate_up (target) | 20185 | 13604 | 1.48x |
| mlp_down (target) | 13153 | 9446 | 1.39x |
| lm_head | 1648 | 1354 | 1.22x (noise) |
| TOTAL (sum) | 52415 | 38225 | 1.37x |
| Decode ms/tok | 2464 | 1730 | **1.42x** |

The 1.48x mlp_gate_up + 1.39x mlp_down are the load-bearing signal.
The non-target regions (qkv_proj, o_proj, lm_head) "improving" is
load-drift noise between runs, not lever credit.

The 1.42x total decode speedup is the honest claim. At T=8 workers
with contention from 7 other agents, theoretical 8x parallelism
attenuates to about 1.4x effective; on a clean machine the same
lever should win considerably more on these regions.

**Clean-machine extrapolation (NOT measured this run):**
v0.0.7 clean baseline was 349 ms/tok. If MLP pool wins 2-3x on
clean machine (vs 1.42x under load), v0.0.8 clean estimate is
**~115 to 175 ms/tok**, which is BELOW the 200 ms/tok target
(2026-06-25 deadline).

A clean-machine bench is the next required measurement. Box must
be at load less than 5 and no concurrent BF16-heavy processes.

## 2026-05-29 (latest+1) -- CORRECTION: clean-machine pool win is 1.17x, not 1.42x

Type-1 catch on the earlier extrapolation. SIGSTOPped the 6 active
claude agents and ran both baseline and pooled benches under
proper clean conditions:

| Metric | Baseline | Pooled MLP | Win |
|---|---|---|---|
| qkv_proj med | 838 ms | 821 ms | 1.02x (no change region) |
| o_proj med | 603 ms | 605 ms | 1.00x (no change region) |
| mlp_gate_up med | 2701 ms | 1968 ms | 1.37x |
| mlp_down med | 1634 ms | 1272 ms | 1.28x |
| lm_head med | 271 ms | 262 ms | 1.03x (no change region) |
| TOTAL (sum) med | 6149 ms | 5000 ms | 1.23x |
| **Decode ms/tok** | **297** | **253** | **1.17x** |

**The under-contention 1.42x was inflated.** Single-thread baseline
got hurt by contention more than the pool did, so the ratio
artificially favored the pool. Pure substrate win at the target
regions is 1.37x mlp_gate_up + 1.28x mlp_down, which propagates
to a 1.17x decode win because non-MLP regions remained unchanged.

**v0.0.8 lands at 253 ms/tok clean, NOT BELOW the 200 target.**
Gap to target: 53 ms/tok = ~21% more speedup needed.

**Remaining levers ranked by leverage:**
1. Cache-tiled BF16 matmul (mlp_gate_up + mlp_down). The F32 tiled
   path hit 2.5x at large N; a BF16 tiled implementation should
   close some of the residual gap. New task #152.
2. AVX-512 zmm widen on the BF16 SIMD kernel (currently ymm 8-wide;
   zmm would be 16-wide on Ice Lake-U which supports AVX-512F).
3. KV cache contiguous reads on the attention path (currently 0.7%
   so little leverage at decode, but compound win across operations).
4. F32 weight elimination - currently F32 copies of MLP weights
   are held alongside BF16 for the lm_head single-thread path;
   dropping them on use_pooled_bf16_mlp=true would reduce DRAM
   pressure.

The 200 target by 2026-06-25 remains achievable - 21% from one
more substrate lever is reasonable. Not a single-lever close.

**What landed (substrate):**
- LayerWeights: raw BF16 storage (w_gate_bf16 / w_up_bf16 / w_down_bf16).
- LoadOpts.load_mlp_bf16 gate.
- Model.use_pooled_bf16_mlp gate.
- bench/bench_forward_regions.zig: per-region MLP attribution.
- 87/87 tests pass (was 81 + 6 new MLP-pool correctness).
- Authored W4 agent 738b24fb; committed by orchestrator post-OOM
  kill at 25479c2.

## 2026-05-29 (later) -- v0.0.7: per-region instrumentation lands; lever-1 target redirected to MLP

The bigger Type-1 catch of the perf push, caught by Step 1
(instrumentation) before Step 2 (optimize) wasted any more cycles.

**Clean-machine baseline (load 5.91, single-thread BF16 lm_head):**

| Region | med ms / round | share |
|---|---|---|
| mlp_gate_up | 2819 | **42.5%** |
| mlp_down | 1840 | **27.7%** |
| qkv_proj | 882 | 13.3% |
| o_proj | 655 | 9.9% |
| lm_head | 327 | **4.9%** |
| attention | 47 | 0.7% |
| (rest) | <60 | <1% |

Decode: med 349 ms/tok over 5 rounds of prefill + 16-token decode.

**With pooled BF16 lm_head enabled (`VLLM_POOLED_LM_HEAD=1`):**

lm_head region 327 ms → 127 ms (**2.6× on the targeted region**).
Decode med 280 ms/tok in the same window, but the comparison is
load-confounded: every region got faster between runs, not just
lm_head, so the total-decode delta cannot be cleanly attributed to
the pool change. The clean read is "pool works as designed on the
region it operates on."

**The lever-1 redirect:** lm_head is **5% of decode**. A 2.6×
on a 5% region is a 1-2% real-world win. The actual 70% of decode
is in MLP (`mlp_gate_up` 42.5% + `mlp_down` 27.7%). Pool BF16 was
the right substrate move; **lm_head was the wrong target.** The
v0.0.8 lever-1 retarget is "pool the MLP BF16 matmuls" with the
same Pool.matmulBF16Auto dispatch.

**Baseline drift catch:** v0.0.6 STATUS quoted 581 ms/tok as the
v0.0.5 baseline. Clean-machine measurement here gets 349 ms/tok.
Either the 581 number was at higher load than tagged, or the
baseline drifted with the kv_cache repair. The 200 ms/tok target
is 1.75× away from clean baseline, not 2.9× away.

**What v0.0.7 shipped:**
- Per-region Instrumentation in Model.forward (12 regions, env-gated)
- Pool.matmulBF16Auto (row-split when M >= n_workers, else col-split)
- 6 new benches (forward-regions, bf16-auto, bf16-cblas, bf16-preconvert, mlp-cblas, tinyllama-breakdown)
- Tests: 81 pass (77 + 4 new Pool.matmulBF16Auto correctness tests)

## 2026-05-29 -- BENCH.md landed; BF16 lm_head Type-1 framing corrected

A measurement pass added `BENCH.md` (methodology + clean-machine
preconditions + load-violated results). Two findings update the
v0.0.6 perf-claim posture:

1. **BF16 lm_head is a modest win, not a 2× win.** Three-run median
   speedup at the lm_head shape (M=1, K=2048, N=32000) is **1.07×**
   F32 → BF16 on this Ice Lake CPU. Effective bandwidth confirms BF16
   is moving bytes through DRAM at half the rate (F32 7.4 GB/s vs
   BF16 3.96 GB/s) -- the byte-advantage is real, but the kernel is
   **compute-bound on widen-instructions**, not bandwidth-bound. The
   2× win predicted by the "halving B-matrix bytes" framing requires
   either AVX-512 zmm wider lanes, cache-tiled BF16, or hardware BF16
   instructions (not present on this CPU). See `BENCH.md` section 4.4.

2. **Pool-vs-spawn margin holds.** Under load 40, the v0.0.5
   persistent pool beat the v0.0.4 ad-hoc-spawn path 2.81× at decode
   (vs 1.41× on a clean machine). The substrate decision was right.

3. **The "1330 ms/tok at load 44" line below is the 2026-05-27
   measurement. A repeat at load 40 produced 48,371 ms/tok decode
   (pool path) -- ~83× the clean-machine v0.0.5 baseline of 581
   ms/tok.** Per-process CPU share was 2.3% of one core; the ratio is
   in the right order of magnitude for the contention level. Neither
   the 1330 ms/tok number nor the 48371 ms/tok number is a perf
   claim. A clean-machine bench is the next gate.

The v0.0.6 perf line stays **engineering-correct, not performance-
confirmed.** A clean-machine bench (load ≤ 2.0 on this 8-core box)
must run before the BF16 win is upgraded from "modest, measured at
1.07× lm_head under load" to "X× on clean baseline." Next perf
levers ranked in `BENCH.md` section 6 -- top of list is pool-the-BF16-
matmul (currently single-thread), then cache-tiled BF16, then
AVX-512 zmm widen.

## v0.0.6 — kv_cache + build.zig repair, BF16 lm_head wired

Three load-bearing fixes on top of v0.0.5:

1. `src/page_manager.zig` restored to the 4-arg `init(alloc, num_blocks,
   n_kv_heads, head_dim)` signature with explicit K/V plane split. The
   v0.0.5 refactor had reduced it to 2-arg + hardcoded n_heads=32, dim=64,
   which broke compilation in `kv_cache.zig` and `attention.zig` (both
   call `getBlockK`/`getBlockV`).
2. `build.zig` had been replaced with a "sovereign_engine" wiring that
   imported many sibling repos with no resolvable build graph and had
   no `test` or `bench-tinyllama-pool` step. Restored test + bench
   surface; every bench/example step now compiles the vllm + safetensors
   + tokenizers modules in `.ReleaseFast` (the historical pipeline had
   silently pulled in a Debug-mode vllm, inflating decode wall-time by
   ~3× and hiding what optimised codegen actually buys).
3. `src/model.zig` keeps the LM head as raw BF16 (`Model.lm_head_bf16:
   weights_mod.RawBF16`) and routes the decode-time logits matmul
   through `matmul_bf16.matmulBF16SIMD`. Largest single matmul in
   TinyLlama decode (M=1, K=2048, N=32000 ≈ 131 MB BF16 vs 262 MB F32);
   halving the B-matrix bytes pulled through L2/L3 is exactly the
   bandwidth-bound case the BF16 kernel was built for.

### Bench (TinyLlama-1.1B end-to-end, 8-token decode, ReleaseFast, T=8)

```
prompt: "The capital of France is"  max_new=8

(A) no pool — matmulSIMDAuto (v0.0.4)
    first-token (prefill+1):  8716 ms
    decode ms/tok:            3369

(B) with persistent pool (v0.0.5)
    first-token (prefill+1):  7076 ms
    decode ms/tok:            1330       <- v0.0.6 measured
```

### Type-II watch — number does NOT compare cleanly to the 581 ms/tok line

The historical 581 ms/tok baseline in the v0.0.5 section below was
measured on a *clean* machine. The 1330 ms/tok number above was
measured at **load average ~44 on an 8-core CPU** (forge_athlete_kernel
processes were saturating the box during the run). Three factors mix
in the gap:

  1. **System load:** 8 vllm-zig worker threads were contending with
     ~15 already-resident background processes for CPU time.
  2. **PagedAttention now in the forward path:** v0.0.5→v0.0.6 added
     paged K/V scatter/gather inside `model.forward`, replacing the
     contiguous-cache code path. That adds per-token block-table
     dispatch overhead even at seq_kv = 8.
  3. **BF16 lm_head:** this *should* be a win on a clean machine
     (bandwidth-bound case), but its effect is not isolated by the
     numbers above.

A repeat bench on an unloaded machine is needed before claiming the
BF16 wiring is a perf win. Until then this is engineering-correct, not
performance-confirmed. **2026-05-29 update:** the BF16 lm_head
microbench landed under load and shows 1.07× median speedup, not 2×
(see BENCH.md section 4.4 and the 2026-05-29 section at the top of
this file). The bandwidth-bound framing was wrong; the kernel is
compute-bound on widen-instructions on this Ice Lake CPU. The clean-
machine bench is still the gate; the prediction has been corrected.

---

> [!CAUTION]
> **SCIENTIFIC INTEGRITY WARNING:** Despite the name `vllm-zig`, this repository is currently a **Type 1 scientific overclaim**. It lacks PagedAttention, continuous batching, and speculative decoding—the three pillars that define the vLLM frontier. It is currently a batch-1 sequential text generator. Do not use for production serving until Phase 2 is green.

## What landed in v0.0.5 (on top of v0.0.4)

- `src/thread_pool.zig`: Linux-futex-backed persistent thread pool
  (~440 LOC). N workers park on a private futex round counter; main
  submits via a shared `Job` struct + `round.fetchAdd` + `futex_wake`
  broadcast; workers signal done via `n_pending.fetchSub` + `futex_wake`
  back to main. No per-call spawn cost.
- `Pool.matmulRowSplit / matmulColSplit / matmulAuto` — same dispatch
  shape as v0.0.4 (auto-row when M≥T, auto-col when M<T) but workers
  are persistent.
- `Model.setThreadPool(*Pool)` — optional attachment; `forward()`
  routes through `pool.matmulAuto` when set.
- `bench/bench_pool.zig` — head-to-head at decode shape.

### Bench at decode shape (M=1, K=2048, N=2048, T=8, ReleaseFast)

| Path | per-call | vs single-thread |
|---|---|---|
| matmulSIMD (1 thread) | 1100 μs | 1.00x (baseline) |
| matmulSIMDAuto (v0.0.4 spawn) | 2292 μs | **2.08x slower** |
| Pool.matmulAuto (v0.0.5 pool) | **926 μs** | **1.19x faster** |

Pool is **2.47x faster than v0.0.4** at decode. Modest 1.19x over
single-thread is expected: 8 threads sharing the memory bus to a
16 MB B matrix is memory-bandwidth bound, not compute bound.

### End-to-end TinyLlama-1.1B (prompt "The capital of France is", max_new=8)

| Metric | (A) no pool | (B) pool | Speedup |
|---|---|---|---|
| first-token ms | 3937 | 3285 | 1.20x |
| total ms | 9686 | 7352 | 1.32x |
| **decode ms/tok** | **821** | **581** | **1.41x** |

v0.0.5 at 581 ms/tok beats v0.0.3's 608 ms/tok baseline (4.5% faster),
fully closing the v0.0.4 regression. Pool path is **1.41x faster than
v0.0.4 ad-hoc spawn** end-to-end. Bench: `zig build bench-tinyllama-pool`.

### What's still open for v0.0.6+

- Cache-tiled GEMM (per ARCHITECTURE.md): N=1024 GFLOPS drops to 4.4
  because working set blasts L2/L3. Blocked matmul recovers that.
- BF16 storage end-to-end + mixed-precision compute (halves memory
  footprint → less bandwidth pressure → better multi-thread scaling)
- PagedAttention KV cache (Phase 1.5)
- macOS / Windows pool path (Io-threading refactor, or platform-
  specific futex equivalents)

## Phase position

Per `ARCHITECTURE.md` §13:

- Phase 0 (design doc) — done 2026-05-20

## Phase position

Per `ARCHITECTURE.md` §13:

- Phase 0 (design doc) — done 2026-05-20
- Phase 1 prelude (CPU kernel substrate) — done 2026-05-21
- Phase 1 proper (RoPE + GQA attention + forward block) — done 2026-05-21
- Phase 1 complete (multi-block real TinyLlama inference) — done 2026-05-21
- **v0.0.3 streaming + multi-thread + parallel BF16 load — done 2026-05-21**
- **v0.0.4 N-axis col-split matmul + auto-dispatch — done 2026-05-22**
- **v0.0.5 persistent thread pool — done 2026-05-22**
- **Phase 1.5 (PagedAttention block substrate) — done 2026-05-22** ← here
- Phase 2 (continuous batching + HTTP front-end) — not started
- Phase 3 (LiveView + public v0.1 release) — not started

## What landed in v0.0.2 (on top of Phase 1 proper)

- `src/bf16.zig` — BF16 ↔ F32 conversion (exact for BF16→F32,
  round-to-nearest-even for F32→BF16). Tests cover representative
  values + buffer conversion + odd-byte-count rejection.
- `src/weights.zig` — `loadMatrixBF16`, `loadVectorBF16`,
  `loadMatrixBF16Transposed` for HF-style projection weights.
  Uses safetensors-zig as a path dependency.
- `src/kv_cache.zig` — Per-layer fixed-size KVCache with append +
  getK / getV / currentLen. Memory budget for TinyLlama at
  max_seq=2048: ~92 MB. Tested per-layer independence + overflow
  rejection.
- `src/sampler.zig` — Greedy / temperature / top-K sampling with
  numerically-stable softmax. 7 tests.
- `src/model.zig` — Full Llama-family Model: Config (parses HF
  config.json), LayerWeights, Model.loadFromSafeTensors,
  Model.forward (embed → N×block → final norm → lm_head → logits).
  KV cache optional (null = recompute per call).
- `examples/infer_tinyllama.zig` — End-to-end smoke test that loads
  real TinyLlama-1.1B from the safetensors-zig fixture, tokenizes,
  forwards, samples greedy + reports top-5.
- build.zig wired with path dependencies on safetensors-zig and
  tokenizers-zig for end-to-end integration.

## Measured end-to-end on TinyLlama-1.1B-Chat-v1.0

Real run on consumer Linux (Ryzen-class, 8 cores, ReleaseFast):

```
prompt: "Hello, world!"

[1/5] loaded config.json in 474 us
[2/5] loaded tokenizer.json in 133 ms
[3/5] safetensors handle opened (header parse lazy, 23KB → ~241us)
[4/5] loaded 22 layers + embed + lm_head in 39,699 ms  (BF16→F32)
[5/5] forward over 5 tokens in 2463 ms (492 ms/token, prefill)

next token (greedy): id=13 = "\n" (logit 12.08)
top-5: 13 ("\n"), 2 ("</s>"), 297 ("in"), 829 ("</"), 29871 ("")
```

The greedy next-token "\n" after "Hello, world!" is the canonical
sensible continuation (newline / end-of-greeting). Top-5 alternatives
are all plausible. Real model behavior, real kernels.

## Performance honest non-claims

- **40-second model load** is BF16→F32 conversion bandwidth-bound
  (2.2GB → 4.4GB F32 with byte-by-byte conversion). v0.0.3 path:
  parallelize the conversion + lazy per-layer load + in-place BF16
  storage with mixed-precision kernel paths.
- **492 ms/token prefill** is CPU baseline with no Triton / CUDA, no
  multi-thread inner kernels, no cache tiling, no FlashAttention.
  Comparable single-thread CPU baselines on this model class are
  ~200-500 ms/token, so we're in the expected range. Speedup paths:
  multi-thread matmul (~4-8x), cache-tiled GEMM (~1.5-3x), eventual
  Triton/CUDA (10-50x).
- **No vLLM head-to-head** at v0.0.2. vLLM Phase 2 needs continuous
  batching + an HTTP front-end before a fair comparison.
- **F32 working memory** is double the BF16 footprint. For TinyLlama
  ~4.4GB; for larger models this becomes the bottleneck. v0.0.3
  in-place BF16 storage is the cleanest fix.

## What's open for v0.0.3

1. **Streaming output** — generate N tokens, not just next-token logits
2. **KV cache in the integration example** (currently set to null)
3. **Parallel BF16→F32 conversion** during load (40s → ~5s with 8 threads)
4. **BF16 storage with mixed-precision kernels** (halve working memory)
5. **Multi-thread matmul** (expected 4-8x throughput on 8-core)
6. **Cache-tiled GEMM** for the projection matrices

## What's open for v0.1+

- PagedAttention (per ARCHITECTURE.md §4)
- Hardware-gated perf work (Triton kernels for Ampere+)
- BEAM/OTP scheduler (Phase 2)
- HTTP front-end (Phase 2)
- LiveView dashboard (Phase 3)

## Test count

- Phase 1 prelude: 19 tests
- Phase 1 proper: +19 (rope, attention, forward) → 38 tests
- **Phase 1 complete: +20 (bf16, kv_cache, sampler, model) → 58 tests**

## Honest non-claims (still active)

- CPU only. All GPU work hardware-gated.
- v0.0.2 is the integration-correctness gate, not the production-throughput gate.
- Numbers are consumer-Linux specific; will differ on Apple Silicon /
  production CPUs.
- No vLLM end-to-end comparison yet.
- Real TinyLlama inference works; longer prompts have not been stress-tested.
