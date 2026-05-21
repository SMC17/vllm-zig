# Changelog

## 0.0.1 — 2026-05-21

Phase 1 prelude: CPU kernel substrate.

- `src/tensor.zig` — Matrix abstraction
- `src/kernels/matmul.zig` — scalar + SIMD matmul (9-16x speedup)
- `src/kernels/rmsnorm.zig` — Llama RMSNorm with SIMD
- `src/kernels/softmax.zig` — numerically stable softmax
- `src/kernels/silu.zig` — SiLU + siluMul
- 19 unit tests
- `bench/bench_matmul.zig` — scalar vs SIMD bench
- `examples/forward_skeleton.zig` — Llama transformer block wiring smoke test
- AGPL-3.0

## 0.0.0-design — 2026-05-20

Phase 0: ARCHITECTURE.md design gate doc.
