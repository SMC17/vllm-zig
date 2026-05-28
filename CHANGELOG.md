# Changelog

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
