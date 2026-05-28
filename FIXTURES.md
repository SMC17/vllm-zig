# Fixtures

The end-to-end TinyLlama inference claim, the "from git clone to inference in 5 commands" reproducibility recipe, and every wall-time number in this repository are bounded by a specific model fixture, hardware class, and bench harness. Those bounds are the contribution.

## `tests/fixtures/tinyllama-real/` — TinyLlama-1.1B-Chat-v1.0

Canonical `TinyLlama/TinyLlama-1.1B-Chat-v1.0` repository contents from HuggingFace Hub:

- `model.safetensors` — 2.2 GB BF16 weights, 201 tensors
- `tokenizer.json` — BPE + SentencePiece
- `config.json` — 22 layers, 32 attention heads, 4 KV heads, head_dim=64, hidden_size=2048, intermediate_size=5632, vocab=32000

TinyLlama-1.1B, not Llama-3.1-8B or larger, for three reasons:

1. Architecturally identical to the Llama-2/3 family at smaller scale. RoPE rotary embeddings, GQA grouped-query attention, SwiGLU FFN, RMSNorm — the same primitives as production-scale Llama, sized down to 1B. The forward pass exercises the same kernel set; wall times scale linearly in layer count.
2. 2.2 GB fits on commodity laptop RAM. Reproduction on a developer machine. 8B+ models require enterprise-class hardware for f32 inference, which makes reproduction harder than the substrate claim warrants.
3. Apache 2.0 licence. No friction for weight download or testing.

Gitignored — too large to commit. `scripts/fetch-tinyllama.sh` bootstraps from HuggingFace Hub.

Curation choices:

- BF16 weights, F32 inference. Conversion happens on weight load. This isolates the kernel-correctness question from the dtype-quantization question. Pure BF16 inference is a separate engineering decision.
- Chat-v1.0 variant, not base. Same architecture; the tokenizer + canonical IDs verified in `tokenizers-zig/FIXTURES.md` use the chat variant.

## Canonical bench prompt

```
The capital of France is
```

`max_new_tokens=8`. Greedy.

Why this exact prompt:

1. Seven tokens of prefill, eight tokens of decode. Short enough that the bench measures both prefill and decode hot paths in reasonable wall time; long enough that prefill is a meaningful fraction (not dominated by single-token init cost).
2. Deterministic output under greedy decoding. The model is expected to output " Paris" (BPE id 8403) as the first generated token. If the forward pass diverges from the reference path, the first generated token will not be 8403. The bench is a correctness test alongside being a throughput test.
3. One canonical prompt, not a sweep. Cross-prompt variance is documented separately. The headline number is calibrated against this exact one.

## Bench harnesses

`bench/bench-tinyllama-pool/` measures end-to-end wall time:

- Model load — parallel BF16→F32
- Prefill — 7-token forward pass
- Decode — 8-token autoregressive loop
- Total

Per-version numbers are captured and committed in CHANGELOG (v0.0.3 → v0.0.4 → v0.0.5 → v0.0.6).

Curation choices:

- Single hardware class for headline numbers. All published wall times in README + CHANGELOG are i7-1065G7 (Ice Lake mobile, 8-core). Different hardware gives different numbers; the headline number is calibrated to one class so version-to-version comparison is apples-to-apples.
- Load-average disclosure. The v0.0.6 number "1330 ms/token" was measured under load average ~44, not a clean machine. Documented in the README so readers adjust expectation against their own conditions.
- No artificial throttling. ReleaseFast, no thermal limits, no container cpu cgroups. The bench machine is the developer machine.

## `tests/test_realmodel.zig` — integration smoke

End-to-end. Loads TinyLlama-1.1B weights through `safetensors-zig`. Tokenizes a test prompt through `tokenizers-zig`. Runs the forward pass through `vllm-zig`. Samples the next token. Verifies it matches the expected canonical token ID.

The 77/77 tests headline includes unit tests on individual kernels. This test is the system integration check. A clean-room reimplementation can match every unit test and still fail this one if composition has a bug.

## Why reproducibility binds to the fixture

A port reproducing the v0.0.6 numbers needs to:

1. Use the same TinyLlama-1.1B-Chat-v1.0 weights from HuggingFace canonical. Different weight versions give different logits.
2. Use the same bench prompt — "The capital of France is", max_new_tokens=8, greedy. Cross-prompt timing varies.
3. Match the BF16-load + F32-inference dtype split. Pure-F32-load benches will be different.
4. Disclose the hardware class. "Half a second per token on Ice Lake mobile" is bound to that hardware; "0.3 seconds on Sapphire Rapids" is a different bench.

Without these anchors, a port can claim "we ran TinyLlama in language X" but cannot claim "we matched vllm-zig v0.0.6's reported wall time." That distinction is the point of the fixture discipline.

## Provenance, licence

TinyLlama-1.1B-Chat-v1.0 weights: Apache 2.0, published by the TinyLlama authors. Gitignored; bootstrap via `huggingface-cli download` at run time.

Bench harnesses (`bench/bench-tinyllama-pool/` and others): AGPL-3.0-or-later.

Canonical bench prompt: original.

Cite fixture and hardware: `vllm-zig v0.0.6 on TinyLlama-1.1B-Chat-v1.0, bench prompt "The capital of France is" with max_new_tokens=8 greedy, i7-1065G7 ReleaseFast`.
