# STATUS

**Version:** v0.0.1 (Phase 1 prelude)
**Last update:** 2026-05-21

## Phase position

Per `ARCHITECTURE.md` §13:

- **Phase 0 (design doc):** Done 2026-05-20
- **Phase 1 prelude (kernel substrate, CPU):** Done 2026-05-21 ← here
- **Phase 1 (single-sequence prefill+decode on TinyLlama):** Not started
- **Phase 1.5 (PagedAttention + KV cache):** Not started
- **Phase 2 (continuous batching + HTTP):** Not started
- **Phase 3 (LiveView + public v0.1):** Not started

## What landed in Phase 1 prelude (v0.0.1)

- `src/tensor.zig` — Matrix abstraction with init/deinit/at/set/row/
  rowMut/fill/fillRandom/approxEq
- `src/kernels/matmul.zig` — scalar + SIMD f32 row-major matmul.
  SIMD path uses broadcast A-scalar × B-row outer-product pattern.
  Speedup 9-16x at N≤512.
- `src/kernels/rmsnorm.zig` — Llama RMSNorm with SIMD inner loop
- `src/kernels/softmax.zig` — numerically stable softmax
- `src/kernels/silu.zig` — SiLU + Llama-MLP `siluMul`
- `bench/bench_matmul.zig` — scalar vs SIMD bench across N ∈ {64, 128, 256, 512, 1024}
- `examples/forward_skeleton.zig` — composes all kernels into a
  Llama-style transformer block (random weights, no RoPE, smoke test)
- 19 unit tests:
  - 4 Matrix (init zeroes, at/set, rowMut, approxEq)
  - 4 matmul (known result, dim mismatch, SIMD↔scalar agreement on 7
    shape variants, identity)
  - 5 RMSNorm (known result, weight scaling, zero input, dim mismatch,
    matrix-per-row)
  - 6 softmax (uniform, sum-to-1, max-wins, large-magnitude stability,
    known [-1,0,1], empty)
  - 5 SiLU (silu(0), silu(large+), silu(large-), vec apply, siluMul
    mismatch, siluMul elementwise)
- AGPL-3.0 LICENSE

## What's open for Phase 1 proper

1. **RoPE kernel** (Zig port of the Triton reference at ~/triton-rope)
2. **Single-head + multi-head attention** with causal masking
3. **Single-sequence forward pass** end-to-end on a real TinyLlama-1.1B
   model via safetensors-zig weight loading + tokenizers-zig
   text-to-IDs
4. **Sampler** (temperature + top-k + top-p)
5. **Per-decode loop** with KV cache (non-paged at first)

## What's open for Phase 1.5+

- PagedAttention KV cache (per ARCHITECTURE.md §4)
- Hardware-gated perf work (Triton kernels for Ampere+)
- BEAM/OTP scheduler (Phase 2)
- HTTP front-end (Phase 2)

## Honest non-claims (v0.0.1)

- The forward-skeleton example runs but does NOT perform real
  inference. Random weights, no RoPE, no causal mask, no KV cache.
- All measurements are single-thread CPU on consumer x86_64-linux.
  Numbers will differ on Apple Silicon (NEON 4-wide vs AVX-2 8-wide),
  and dramatically on actual GPUs.
- No cache tiling — N=1024 matmul drops to 4.4 GFLOPS because L2/L3
  spillover. Cache-blocked matmul is Phase 2 scope.
- No comparison to vLLM upstream because there is no forward pass yet.
  vLLM head-to-head is a Phase 2+ artifact.
