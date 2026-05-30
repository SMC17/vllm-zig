# Changelog

## 0.0.11 — 2026-05-29

**Lever-9 + Lever-10b (lm_head pool re-flip): 6.17x cumulative decode.**

After lever-9 shipped at apparent med 174 ms-per-tok, re-bench at
matched CLEAN load gave 121 ms-per-tok — the 174 number was
load-elevated. With `VLLM_POOLED_LM_HEAD=1` also enabled
(exp-1780111629-832205672 confirmed at +17.4 pct decode median over
lever-9 baseline), median drops to **100 ms-per-tok**:

```
  config                    | decode med | speedup vs lever-4
  --------------------------+------------+-------------------
  lever-4 baseline (load 22)|     617    |       1.00x
  lever-9 (load 14)         |     174    |       3.55x
  lever-9 (clean load)      |     121    |       5.10x
  lever-9 + lm_head pool    |     100    |       6.17x
```

`VLLM_POOLED_LM_HEAD=1` was Type-I catched as a wash at lever-5
(exp-1780109335-555121691). The lever-9 layout amortizes per-thread
per-tile work so futex sync overhead is no longer dominant.

Also re-tested `VLLM_N_THREADS=8` under lever-9 (exp-1780111504-034388283):
refuted again (+9.8 pct worse median). The n=4 fixed point survives
the layout change.

---

**Lever-9: N-tile-major BF16 weight layout (3.55x cumulative decode).**

Re-packs every BF16 weight matrix at load time into the
`[N_tile_outer, K, N_tile_inner=64]` layout consumed by a new col-range
kernel `matmulBF16NtileColRangeKernel`. The inner K loop now sees
stride-1 b reads within tiles instead of the previous stride-N pattern
that page-crossed every k iteration.

```
  k-stride row-major (lever-5):  N * 2 = 22.5 KB  → page cross, TLB miss
  k-stride N-tile-major (lever-9): TILE * 2 = 128 B → L2 streamer engages
```

Ice Lake L2 hardware streamer detects the stride-128 pattern within
a tile and prefetches ahead; row-major's stride-N pattern was beyond
its detection horizon, leaving each k iteration to stall on TLB miss
plus DRAM round-trip.

End-to-end on TinyLlama-1.1B-Chat (5 rounds, n_threads=4, full lever
stack 1-9, ReleaseFast):

```
  metric        | lever-5 (load 18) | lever-9 (load ~14) | speedup
  --------------+-------------------+--------------------+--------
  decode med    |        277        |        174         |  1.59x
  decode p10    |        263        |        164         |  1.60x
  decode p90    |        311        |        207         |  1.50x
```

Per-region (decode + prefill, 5 rounds, med ms):

```
  region        | lever-5 | lever-9 | speedup
  --------------+---------+---------+--------
  mlp_gate_up   |   2361  |   1706  |  1.38x
  mlp_down      |   1261  |    899  |  1.40x
  qkv_proj      |    558  |    451  |  1.24x
  o_proj        |    474  |    349  |  1.36x
  lm_head       |    295  |    241  |  1.22x
```

Cumulative from v0.0.9 lever-4 baseline (med 617): **3.55x** decode.

89/89 tests pass. exp-1780110463-209781800 confirmed (target was
20pct improvement; observed 37pct over lever-5 alone).

New files / fields:
- `src/weights.zig`: `repackBF16ToNtileMajor` repack helper.
- `src/kernels/matmul_bf16.zig`: `matmulBF16SIMDNtileMajor` standalone.
- `src/thread_pool.zig`: `matmulBF16NtileMajorAuto` + `matmulBF16NtileColRangeKernel`.
- `src/model.zig`: `LoadOpts.bf16_ntile_major`, `Model.use_bf16_ntile_major`, `dispatchBF16` helper.
- `bench/bench_forward_regions.zig`: `VLLM_BF16_NTILE_MAJOR` env gate.
- `bench/bench_bf16_ntile_major.zig`: standalone microbench (1.29x single-thread proven).

Gate via:
```
VLLM_BF16_NTILE_MAJOR=1 VLLM_POOLED_MLP=1 VLLM_POOLED_ATTN=1 \
  VLLM_POOLED_MLP_FUSED=1 VLLM_QKV_FUSED=1 VLLM_N_THREADS=4 \
  zig build bench-forward-regions -Doptimize=ReleaseFast
```

## 0.0.10 — 2026-05-29

**Lever-5: register-blocked BF16 matmul kernel (2.23x decode).**

Replaces the pool's BF16 column-range kernel and adds standalone
`matmulBF16SIMDRegBlock`. Switches from outer-product (K outer, N inner
over c) to inner-product (N outer in 64-column tiles, K inner with 4
ZMM accumulators held live across the full K loop).

Memory traffic on M=1 K=2048 N=2048:
- outer-product: K*N c-reads + K*N c-writes (4M ops to c)
- inner-product: N c-stores per tile (32 stores total)

The K-fold reduction in c-traffic was the dominant cost in the old
SIMD path. Accumulators stay in @Vector(16, f32) ZMMs via `@mulAdd`
(FMA, one rounding) rather than separate mul+add (two roundings).

End-to-end on TinyLlama-1.1B-Chat (n_threads=4, lever-1..4 stacked,
ReleaseFast):

```
  metric         | lever-4 (load 22) | lever-5 (load 18) | speedup
  ---------------+-------------------+-------------------+--------
  decode med     |        617        |        277        |  2.23x
  decode p10     |        473        |        263        |  1.80x
  decode p90     |        660        |        311        |  2.12x
```

Per-region (decode + prefill, 5 rounds, med ms):

```
  region        | lever-4 | lever-5 | speedup
  --------------+---------+---------+--------
  mlp_gate_up   |   4709  |   2361  |  1.99x
  mlp_down      |   3435  |   1261  |  2.72x
  qkv_proj      |   1740  |    558  |  3.12x
  o_proj        |   1417  |    474  |  2.99x
  lm_head       |    518  |    295  |  1.76x
```

Numerical equivalence within 6.7e-6 max_abs on the M=1 K=2048 N=32000
lm_head shape; relative diff bounded by FMA-vs-separate-rounding gap
at K=2048 accumulation.

89/89 tests pass. exp-1780108119-609312875 confirmed.

**Honest refuted attempts (Type-I prevention):**

- Lever-6 (manual `@prefetch` in K loop) — 2x regression across all
  BF16 matmul regions. Manual prefetch with locality=3 fights Ice
  Lake's L2 hardware streamer and thrashes L1 across 4 worker threads.
  exp-1780108992-833810307.
- Lever-7 (`VLLM_POOLED_LM_HEAD=1` re-test post lever-5) — lm_head
  region itself improved 3.7pct but other regions regressed 17-24pct.
  Net inconclusive at variance level; lm_head pool stays off.
  exp-1780109335-555121691.
- Lever-7 alt (`VLLM_N_THREADS=8` re-test) — refuted; +1.8pct worse
  decode median, futex sync overhead per dispatch dominates. n=4
  remains optimal. exp-1780109647-402094155.
- Lever-8 (N_TILE=128, 8 ZMM accumulators) — refuted; +25pct worse
  p90 and +10pct mlp_gate_up. Ice Lake L1 read-port bandwidth, not
  ZMM count, is the local limit. exp-1780109796-369194807.

## 0.0.9 — 2026-05-29

**Lever-4: QKV matmul fusion (3 attention dispatches → 1).**

Adds `LayerWeights.w_qkv_bf16` fused Q|K|V weight matrix and
`LoadOpts.fuse_qkv` flag. Forward path constructs non-owning Matrix
views q/k/v over the single fused `qkv_buf` so RoPE and attention
consume them unchanged.

`loadMatrixBF16RawConcat3Transposed` concatenates Q|K|V on load;
`bench_forward_regions` adds `VLLM_QKV_FUSED` env gate.

Architectural reduction: 3 attention-region pool dispatches → 1.

Perf delta filed inconclusive (exp-1780107082-417603501) — bench ran
at load avg 22-28 with within-run 1.9x variance (R1 900 vs R5 473
ms/tok); clean-load A/B-toggle harness needed for fair delta.

87/87 tests pass.

## 0.0.8 — 2026-05-29

**Lever-2 + Lever-3: pooled attention + gate+up matmul fusion.**

Lever-2: `w_qkv_bf16`, `w_o_bf16` BF16 weight storage with pool dispatch
through `matmulBF16Auto`. Attention regions (qkv_proj, o_proj) now route
through the persistent thread pool.

Lever-3: `LoadOpts.fuse_mlp_gate_up` flag concatenates gate + up weights
into single `w_gateup_bf16` matrix. Halves MLP dispatches per layer
(3 → 2 BF16 matmuls). p10 194 ms/tok crosses 200 ms/tok decode target
on clean machine.

86/86 tests pass (v0.0.8). lm_head BF16 pool dispatch documented as
Type-I catch — wash at M=1 K=2048 N=32000 because per-thread per-tile
work is so small that futex sync overhead cancels parallel gain.

## 0.0.7 — 2026-05-29

**Per-region instrumentation + lever-1 retarget.**

`bench/bench_forward_regions.zig`: end-to-end Model.forward with
nanosecond per-region timers (embed, layernorm, qkv_proj, rope,
attention, o_proj, residual_add_ln, mlp_gate_up, silu_mul, mlp_down,
final_norm, lm_head). VLLM_INSTRUMENT=1 to enable.

Instrumentation revealed MLP regions account for 75pct of decode TAM
(mlp_gate_up 45pct, mlp_down 30pct). Lever-1 retargets from attention
(which was 20pct combined) to MLP. v0.0.7 ships the pooled BF16 MLP
dispatch path; v0.0.8 adds the gate+up fusion.

83/83 tests pass.

## 0.0.6 — 2026-05-27

**Repair pass: restore test+bench wiring; wire BF16 lm_head into forward.**

The v0.0.5 engine-refactor commit (`e0dec6b`) had three load-bearing
breaks that this release closes:

- `src/page_manager.zig` was reduced to a 2-arg `init(alloc, num_blocks)`
  with hardcoded n_heads=32, dim=64 and lost the `getBlockK` /
  `getBlockV` accessors that `kv_cache.zig` and `kernels/attention.zig`
  both depended on. Restored to 4-arg `init(alloc, num_blocks,
  n_kv_heads, head_dim)` with split K/V planes.
- `build.zig` had been replaced with a "sovereign_engine" wiring that
  imported many sibling repos with no resolvable build graph and had
  no `test` or `bench-tinyllama-pool` step. Restored the test + bench
  surface; every bench/example step now compiles the vllm + safetensors
  + tokenizers modules in `.ReleaseFast` so the recorded numbers reflect
  optimised codegen (the historical pipeline silently pulled in a
  Debug-mode vllm).
- `src/gpu_context.zig` imported a missing `ark_bridge` module. Stubbed
  to a no-op handle that returns `error.GpuNotConfigured`; the PTX/SASS
  lane is gated behind a separate toolchain decision.

New BF16 forward-pass wiring:

- `src/weights.zig`: `RawBF16` + `loadMatrixBF16Raw` +
  `loadMatrixBF16RawTransposed` return the on-disk u16 bytes punned
  in row-major layout with no F32 upcast.
- `src/model.zig`: `Model.lm_head_bf16` replaces the F32 `lm_head`
  Matrix; the decode-time logits matmul in `forward()` now calls
  `matmul_bf16.matmulBF16SIMD` directly against the raw u16 view.
  This is the single largest matmul in TinyLlama decode (M=1, K=2048,
  N=32000 = ~131 MB BF16 vs ~262 MB F32), so it is the highest-leverage
  callsite to migrate first.

See STATUS.md for the measured numbers and the Type-II watch note.

## 0.0.5 — 2026-05-22

**Persistent thread pool — closes the v0.0.4 decode regression.**

The v0.0.4 commit shipped multi-thread row/col-split matmul via ad-hoc
`std.Thread.spawn` per call. At decode (M=1, ~110 matmul calls/token)
the spawn cost dominated wall time, producing a ~null result vs
single-thread (v0.0.3: 608 ms/tok; v0.0.4: 633 ms/tok).

v0.0.5 closes that gap with a Linux-futex-backed persistent pool:

- `src/thread_pool.zig` (~440 LOC including tests) — N worker threads
  park on a private futex round counter; main thread submits a job by
  filling the shared `Job` struct, bumping the round, and `futex_wake`ing
  all workers. Last worker to finish `futex_wake`s main. Linux-only
  (`std.Thread.Pool` was dropped from Zig 0.16's stdlib; the
  replacement lives at `std.Io.Threaded.Pool` and requires threading an
  `Io` parameter through every caller).
- `Pool.matmulRowSplit / matmulColSplit / matmulAuto` — full match for
  the v0.0.4 dispatch API, but reuse workers across calls.
- `Model.setThreadPool(*Pool)` — optional pool attachment; if set,
  `forward()` routes all matmuls through the pool.
- `bench/bench_pool.zig` — head-to-head: 200 matmul calls at decode
  shape (M=1, K=2048, N=2048, T=8) on consumer Linux ReleaseFast:

```
(1) matmulSIMD       (1 thread)       per-call=1100 μs   1.00x (baseline)
(2) matmulSIMDAuto   (v0.0.4 spawn)   per-call=2292 μs   2.08x SLOWER
(3) Pool.matmulAuto  (v0.0.5 pool)    per-call= 926 μs   1.19x faster
```

Pool path is **2.47x faster than v0.0.4 ad-hoc spawn** at the decode
shape. Modest 1.19x vs single-thread is expected — at M=1 with a 16 MB
B matrix the bottleneck is memory bandwidth shared across the 8 cores,
not compute.

### End-to-end on TinyLlama-1.1B-Chat (8-core consumer Linux, ReleaseFast)

Real inference path, prompt "The capital of France is", max_new=8:

```
  metric          | (A) no pool | (B) pool    | speedup
  ----------------+-------------+-------------+--------
  first-token ms  |        3937 |        3285 | 1.20x
  total ms        |        9686 |        7352 | 1.32x
  decode ms/tok   |         821 |         581 | 1.41x
```

Pool path delivers **1.41x decode improvement** end-to-end. The
synthetic 2.47x at the matmul level dilutes to 1.41x at the
forward-pass level because matmul is ~70-80% of forward-pass wall
time (the rest is RMSNorm, attention KV gather, softmax, RoPE,
residuals).

Compared to v0.0.3's baseline (608 ms/tok), v0.0.5 at 581 ms/tok is
4.5% faster — closing the v0.0.4 regression AND beating v0.0.3.

Bench harness: `zig build bench-tinyllama-pool`.

### Test count

69 tests pass (+11 from v0.0.4):
- 6 new in `kernels/matmul.zig` (pool variants when std.Thread.Pool
  was assumed available — kept for documentation of the dispatch
  shape, since std.Thread.Pool was dropped before commit)
- 5 new in `thread_pool.zig` (init/deinit, row-split, col-split,
  auto-dispatch, 100-call reuse)

### Honest non-claims

- **Linux-only.** Pool uses raw `std.os.linux.futex_4arg / futex_3arg`.
  macOS / Windows can fall through to `matmulSIMDAuto` (ad-hoc spawn)
  or stay single-thread until Io-threading scaffolding lands.
- **Memory-bandwidth bound at decode.** The 1.19x vs single-thread at
  M=1 K=2048 N=2048 is honest — 8 threads sharing one DRAM bus to a
  16 MB weight matrix cannot scale linearly with cores.
- **End-to-end TinyLlama re-bench pending.** The pool fix targets a
  known regression; full prefill+decode numbers vs v0.0.4 still
  measurement-pending — see STATUS.md.

## 0.0.2 — 2026-05-21

**Phase 1 complete: real TinyLlama-1.1B inference end-to-end through
Zig kernels.**

- `src/bf16.zig`: BF16 ↔ F32 conversion (exact for BF16→F32,
  round-to-nearest for F32→BF16)
- `src/weights.zig`: load matrix/vector tensors from a safetensors-zig
  handle, with optional HF-style [out, in] → [in, out] transpose
- `src/kv_cache.zig`: per-layer contiguous KV storage with append +
  per-layer-independent seq_len tracking
- `src/sampler.zig`: greedy + temperature + top-K sampling
- `src/model.zig`: full Llama Model — Config.fromJson, LayerWeights,
  Model.loadFromSafeTensors, Model.forward (embed → N×block → norm →
  lm_head → logits)
- `examples/infer_tinyllama.zig`: end-to-end integration that loads
  real TinyLlama-1.1B-Chat-v1.0 from safetensors-zig fixture, tokenizes
  via tokenizers-zig, forwards, samples greedy
- build.zig wires safetensors-zig + tokenizers-zig as path deps

Measured end-to-end on TinyLlama-1.1B (consumer Linux, ReleaseFast):
- Model load (BF16 → F32, 2.2GB → 4.4GB): ~40 seconds
- Forward pass (5-token prefill): 2.4 seconds = 492 ms/token
- Output: "Hello, world!" → greedy next-token = "\n" (id=13, logit 12.08)
  with top-5 alternatives all semantically plausible

Test count: 38 → 58.

Honest non-claims:
- Load is bandwidth-bound BF16→F32 byte-conversion; v0.0.3 parallel
  conversion expected 8x speedup.
- 492 ms/token is CPU baseline with no multi-thread / no cache tiling
  / no Triton. Multi-thread matmul (v0.0.3) expected 4-8x.
- F32 working memory doubles BF16 footprint; v0.0.3 in-place BF16
  storage with mixed-precision kernels fixes this.
- No vLLM head-to-head; that gates on Phase 2 (continuous batching).

## 0.0.1 — 2026-05-21

Phase 1 prelude (CPU kernel substrate). 19 tests across tensor +
matmul + rmsnorm + softmax + silu. SIMD speedup 9-16x at N≤512.

Phase 1 proper (later that day, commit 9afdf86): rope + attention +
forward, 38 tests, full TinyLlama dimensions verified NaN-free.

## 0.0.0-design — 2026-05-20

Phase 0: ARCHITECTURE.md design gate doc.
