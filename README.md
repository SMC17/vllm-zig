# vllm-zig

Production LLM serving substrate. Zig kernels + (future) BEAM/OTP
scheduler. PagedAttention design + continuous batching.

**Status:** v0.0.5 — multi-thread + streaming + real TinyLlama inference.
See `ARCHITECTURE.md` for the full design and `STATUS.md` for the
phase-by-phase position.

## What's here in v0.0.5

Real end-to-end inference on TinyLlama-1.1B-Chat through pure Zig
kernels, with streaming generation and multi-thread matmul.

### Kernels (single-thread + SIMD)
- **`tensor.zig`** — `Matrix` (f32, row-major, owned slice).
  Approximate-equality helper for SIMD/scalar cross-validation.
- **`kernels/matmul.zig`** — scalar reference, SIMD outer-product,
  multi-thread row-split (M-axis), multi-thread col-split (N-axis),
  and `matmulSIMDAuto` dispatcher choosing row-vs-col split by
  M vs n_threads. 9-16x SIMD speedup over scalar at N≤512.
- **`kernels/rmsnorm.zig`** — Llama-style RMSNorm with SIMD inner
  loop. Per-row, in-place.
- **`kernels/softmax.zig`** — numerically stable softmax (subtract
  max before exp). In-place.
- **`kernels/silu.zig`** — SiLU activation + `siluMul` (Llama-MLP
  pattern: `silu(gate) * up`).
- **`kernels/rope.zig`** — RoPE rotation matching the Triton
  reference at `~/triton-rope`.
- **`kernels/attention.zig`** — GQA causal attention with KV cache.

### Forward pass + weights
- **`src/bf16.zig`** — BF16 ↔ F32 conversion (exact for BF16→F32,
  round-to-nearest-even for F32→BF16).
- **`src/weights.zig`** — `loadMatrixBF16` / `loadVectorBF16` /
  `loadMatrixBF16Transposed`. Uses safetensors-zig as a path
  dependency. **Parallel BF16→F32 load.**
- **`src/kv_cache.zig`** — Per-layer fixed-size KV cache. Memory
  budget for TinyLlama at max_seq=2048: ~92 MB.
- **`src/sampler.zig`** — Greedy / temperature / top-K sampling.
- **`src/model.zig`** — Full Llama-family Model (Config from
  `config.json`, LayerWeights, `loadFromSafeTensors`, `forward` =
  embed → N×block → final norm → lm_head → logits). KV cache
  optional (null = recompute per call).
- **`src/generate.zig`** — Streaming `generate()` that emits each
  decoded token as it's produced via a caller-supplied callback.

### End-to-end on TinyLlama-1.1B-Chat-v1.0 (consumer Linux, 8 cores)

```
prompt: "The capital of France is" (max_new=8, ReleaseFast)

Phase           | v0.0.3       | v0.0.4       | Δ
----------------|--------------|--------------|--------
model load      | 58,012 ms    | 25,219 ms    | 2.3x faster
prefill (7 tok) |  3,715 ms    |  2,363 ms    | 1.6x faster
decode/token    |     608 ms   |     633 ms   | null (regress; see below)
total (8 tok)   |  7,978 ms    |  6,797 ms    | 1.2x faster
```

Real model behavior, real kernels, real tokens streamed out. The 25-
second model load is parallel BF16→F32 (2.3x v0.0.3). The 1.6x
prefill is multi-thread matmul row-split. The v0.0.4 decode
regression (ad-hoc `std.Thread.spawn` cost dominating at M=1) was
closed in v0.0.5 with a Linux-futex-backed persistent worker
pool — see `CHANGELOG.md` and `STATUS.md` for the numbers.

69 unit tests pass at v0.0.5 (+11 from v0.0.4). SIMD ↔ scalar
matmul agreement verified at 7 (M, K, N) shape variants.

## What's NOT here (per ARCHITECTURE.md phase plan)

- **Cache-tiled matmul** (N=1024 GFLOPS drops to 4.4 because the
  working set blasts L2/L3) → v0.0.6
- **PagedAttention KV cache** → Phase 1.5
- **BEAM/OTP scheduler** → Phase 2
- **OpenAI HTTP front-end + continuous batching** → Phase 2
- **CUDA / Triton GPU kernels** → Phase 1.5+, hardware-gated on Ampere+

## Build

```sh
zig build test                  # 69 unit tests at v0.0.5
zig build forward-skeleton      # Wiring smoke test
zig build bench-matmul          # Matmul scalar vs SIMD speedup table
zig build infer-tinyllama       # End-to-end TinyLlama inference
zig build install -Doptimize=ReleaseFast
```

## Honest non-claims

- **No vLLM head-to-head.** vLLM-zig is at end-to-end-correctness on
  CPU; vLLM (the upstream) is years of optimisation + production
  GPU/CUDA scheduling. Fair comparison requires Phase 2 (continuous
  batching + HTTP front-end + GPU kernels). Until then this is a
  kernel + correctness substrate, not a serving substrate.
- **CPU only at v0.0.5.** No GPU code. Per ARCHITECTURE.md, GPU
  paths are hardware-gated on Ampere+.
- **No cache tiling, no PagedAttention.** Each is a
  measured-bottleneck-driven follow-up scheduled in
  ARCHITECTURE.md, not a silent gap.
- **Pre-1.0 substrate** per the Zig 0.16 ecosystem convention. Claims stay
  below release-readiness language until the evidence changes.

## License

AGPL-3.0-or-later. See `LICENSE`. AGPL on server software means SaaS
deployment requires releasing modifications or negotiating a
commercial license. Standard sovereign-stack posture.

## See also

- `ARCHITECTURE.md` — full design doc (16 sections, ~400 lines)
- `STATUS.md` — phase-by-phase status
- [`SMC17/safetensors-zig`](https://github.com/SMC17/safetensors-zig) — weight-loading dependency
- [`SMC17/tokenizers-zig`](https://github.com/SMC17/tokenizers-zig) — tokenization dependency
