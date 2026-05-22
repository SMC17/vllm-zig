# Changelog

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
