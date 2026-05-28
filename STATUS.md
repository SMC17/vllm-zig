# STATUS

**Version:** v0.0.6 (toy batch-1 generator; **NOT vLLM-equivalent**)
**Last update:** 2026-05-27

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
performance-confirmed.

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
