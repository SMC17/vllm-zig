# vllm-zig

Production LLM serving substrate. Zig kernels + (future) BEAM/OTP
scheduler. PagedAttention design + continuous batching.

**Status:** v0.0.1 — Phase 1 prelude. CPU kernel substrate landed.
See `ARCHITECTURE.md` for the full design and `STATUS.md` for the
phase-by-phase position.

**Portfolio position:** Artifact #1 (headline) in the 8-artifact
jaw-drop portfolio (see `~/portfolio/README.md`).

## What's here in v0.0.1

CPU kernel substrate that future GPU/Triton kernels will plug into
behind the same API:

- **`tensor.zig`** — `Matrix` (f32, row-major, owned slice). Approximate-
  equality helper for SIMD/scalar cross-validation.
- **`kernels/matmul.zig`** — scalar reference + SIMD outer-product
  matmul (broadcast A-scalar × B-row, accumulate into C-row).
  9-16x SIMD speedup over scalar at N≤512.
- **`kernels/rmsnorm.zig`** — Llama-style RMSNorm with SIMD inner
  loop. Per-row, in-place.
- **`kernels/softmax.zig`** — numerically stable softmax (subtract
  max before exp). In-place.
- **`kernels/silu.zig`** — SiLU activation + `siluMul` (Llama-MLP
  pattern: `silu(gate) * up`).
- **`examples/forward_skeleton.zig`** — composes all kernels into a
  Llama-style transformer block. Random weights, no RoPE, no causal
  mask. Smoke test for the kernel-orchestration pattern, NOT a real
  forward pass.

## Measured (consumer Linux, Ryzen-class, ReleaseFast)

Matmul bench across square shapes:

| N | Scalar (ms) | SIMD (ms) | Speedup | GFLOPS (SIMD) |
|---|---|---|---|---|
| 64 | 0 | 0 | 8.9x | 13.7 |
| 128 | 4 | 0 | 9.5x | 9.9 |
| 256 | 34 | 2 | 15.9x | 15.4 |
| 512 | 369 | 27 | 13.6x | 9.9 |
| 1024 | (skipped) | 484 | — | 4.4 |

19 unit tests pass. SIMD ↔ scalar matmul agreement verified at 7
(M, K, N) shape variants.

## What's NOT here (per ARCHITECTURE.md phase plan)

- **The actual forward pass** with real weights → Phase 1 proper
- **RoPE rotation** kernel (Zig port of the Triton reference at
  `~/triton-rope`) → Phase 1
- **PagedAttention KV cache** → Phase 1.5
- **Causal masking + attention with KV cache** → Phase 1
- **BEAM/OTP scheduler** → Phase 2
- **OpenAI HTTP front-end** → Phase 2
- **Continuous batching** → Phase 2
- **CUDA / Triton GPU kernels** → Phase 1.5+, hardware-gated on Ampere+

## Build

```sh
zig build test                # 19 unit tests across kernel + tensor modules
zig build forward-skeleton    # Wiring smoke test: random-weight transformer block
zig build bench-matmul        # Matmul scalar vs SIMD speedup table
zig build install -Doptimize=ReleaseFast
```

## Honest non-claims

- **No real forward pass yet.** The skeleton runs end-to-end but with
  random weights and no RoPE / no causal mask. It validates that the
  kernels compose; it does not perform inference.
- **CPU only.** No GPU code. Per ARCHITECTURE.md, all phases beyond
  Phase 1 correctness are hardware-gated on Ampere+.
- **No cache tiling.** N=1024 GFLOPS drops to 4.4 because the working
  set blasts through L2/L3. Cache-blocked matmul is Phase 2 work.
- **No multi-thread.** Single-thread inner loops. The end-to-end
  serving throughput claim depends on BEAM/OTP scheduler + multi-GPU
  worker mesh, which is Phase 2-3.
- **Comparable kernels at speedup** but NOT comparable to vLLM
  end-to-end yet. vLLM-zig has zero forward-pass code at v0.0.1;
  vLLM (the upstream we're benching against) has years of optimisation.
  No vLLM head-to-head until Phase 2.

## License

AGPL-3.0-or-later. See `LICENSE`. AGPL on server software means SaaS
deployment requires releasing modifications or negotiating a
commercial license. Standard sovereign-stack posture.

## See also

- `ARCHITECTURE.md` — full design doc (16 sections, ~400 lines)
- `STATUS.md` — phase-by-phase status
- `~/portfolio/README.md` — overall portfolio doctrine
- `~/safetensors-zig/` — weight-loading dependency (artifact #4a)
- `~/tokenizers-zig/` — tokenization dependency (artifact #4b)
- `~/triton-rope/` — RoPE kernel reference (artifact #3)
