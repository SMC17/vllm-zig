# Test fixtures — what they are, why they bound the claims

The end-to-end TinyLlama inference claim + the "from git clone to
inference in 5 commands" reproducibility recipe + every wall-time
number in this repository is bounded by a specific model fixture,
hardware class, and bench harness. Those bounds are the contribution.

## Fixture inventory

### `tests/fixtures/tinyllama-real/` — TinyLlama-1.1B-Chat-v1.0 weights

**What:** The canonical `TinyLlama/TinyLlama-1.1B-Chat-v1.0` repository
contents from HuggingFace Hub. Specifically:

- `model.safetensors` — 2.2 GB BF16 weights, 201 tensors
- `tokenizer.json` — the BPE+SentencePiece tokenizer
- `config.json` — the model architecture spec (22 layers, 32 attention
  heads, 4 KV heads, head_dim=64, hidden_size=2048, intermediate_size=5632, vocab=32000)

**Why TinyLlama-1.1B specifically, not Llama-3.1-8B or larger:**

1. **Architecturally identical to Llama-2/3-family at smaller scale.**
   RoPE rotary embeddings, GQA grouped-query attention, SwiGLU FFN,
   RMSNorm — same primitives as the production-scale Llama family,
   just sized down to 1B params. The forward pass exercises the same
   kernel set as 7B+ models would; the kernels are identical, the
   wall times scale linearly in layer count.
2. **2.2GB fits on commodity laptop RAM.** Reproducibility on a
   developer machine. 8B+ models require enterprise-class hardware
   for f32 inference, which makes reproduction harder than the
   substrate claim warrants.
3. **MIT-style license (TinyLlama, Apache 2.0).** No friction for
   weight download or testing.

The fixture is gitignored (2.2GB is too large to commit). The
`scripts/fetch-tinyllama.sh` step bootstraps it from HuggingFace Hub.

**Taste decisions:**

1. **BF16 weights, F32 inference.** Inference is in F32 throughout
   (BF16 → F32 conversion on weight load). This isolates the
   kernel-correctness question from the dtype-quantization question.
   v0.1+ would add BF16-throughout inference; for v0.0.x the claim is
   "we converge on the same logits as the reference path in f32" and
   the bench numbers reflect f32 throughput.
2. **The chat-v1.0 variant.** Same architecture as TinyLlama-1.1B-base,
   but with the chat template applied. The tokenizer + canonical IDs
   verified in `tokenizers-zig/FIXTURES.md` use this chat variant.

### `tests/fixtures/test-prompts/` — canonical bench prompts

**What:** A small set of bench prompts used for the per-token wall-time
benches. The canonical bench prompt is:

> `"The capital of France is"`

with `max_new_tokens=8`, sampling greedy.

**Why this exact prompt:**

1. **7 tokens of prefill, 8 tokens of decode.** Short enough that the
   bench measures both prefill and decode hot paths in a reasonable
   wall time; long enough that prefill is a meaningful fraction (not
   dominated by single-token init cost).
2. **Deterministic output under greedy decoding.** The model is
   expected to output "Paris" as the first generated token. If the
   forward pass diverges from the reference path, the first
   generated token will not be 8403 (the BPE id for " Paris"). This
   makes the bench a correctness test alongside being a throughput
   test.
3. **One canonical prompt, not a sweep.** Bench-prompt sweeps are
   informative but the wall-time-per-token bench is calibrated against
   this exact one. Cross-prompt variance is documented separately.

### Bench harnesses

**`bench/bench-tinyllama-pool/`** — measures end-to-end wall time
across:

- Model load (parallel BF16→F32)
- Prefill (7-token forward pass)
- Decode (8-token autoregressive loop)
- Total

Per version (v0.0.3 → v0.0.4 → v0.0.5 → v0.0.6) bench numbers are
captured + committed in CHANGELOG.

**Taste decisions:**

1. **Single hardware class for headline numbers.** All published wall
   times in the README + CHANGELOG are i7-1065G7 (Ice Lake mobile)
   8-core. Different hardware will give different numbers; the
   headline number is calibrated to one hardware class so the version-
   to-version comparison is apples-to-apples.
2. **Load-average disclosure.** The v0.0.6 number "1330 ms/token" was
   measured under load average ~44 (not a clean machine). This is
   documented explicitly in the README so readers can adjust their
   expectation against their own measurement conditions.
3. **No artificial throttling.** ReleaseFast, no thermal limits, no
   container cpu cgroups. Bench machine is the developer machine.

### `tests/test_realmodel.zig` — integration smoke

**What:** End-to-end integration test that loads TinyLlama-1.1B weights
through `safetensors-zig` + tokenizes a test prompt through
`tokenizers-zig` + runs the forward pass through `vllm-zig` + samples
the next token + verifies it matches the expected canonical token ID.

**Why this test:** the 77/77 tests headline number includes unit tests
on individual kernels. This test is the *system integration* check —
it verifies the kernels, the weight loader, the tokenizer, and the
sampler compose correctly. A clean-room reimplementation can match
every unit test and still fail this one if the composition has a bug.

## Why the fixtures matter for reproducibility

A port to another language reproducing the v0.0.6 numbers needs to:

1. **Use the same TinyLlama-1.1B-Chat-v1.0 weights** (HuggingFace
   canonical). Different weight versions will give different logits.
2. **Use the same canonical bench prompt** ("The capital of France is",
   max_new_tokens=8, greedy). Cross-prompt timing varies.
3. **Match the BF16-load + F32-inference dtype split.** Pure-F32-load
   benches will be different.
4. **Disclose the hardware class.** "Half a second per token on Ice
   Lake mobile" is bound to that hardware; reporting "0.3 seconds on
   Sapphire Rapids" is a different bench.

Without these calibration anchors, a port can claim "we ran TinyLlama
in language X" but cannot claim "we matched vllm-zig v0.0.6's reported
wall time." That distinction is the whole point of the fixture
discipline.

## Fixture provenance + license

- TinyLlama-1.1B-Chat-v1.0 weights — published by TinyLlama authors
  under Apache 2.0 via HuggingFace Hub. Gitignored; CI / users bootstrap
  via `huggingface-cli download` at run time.
- Bench harnesses (`bench/bench-tinyllama-pool/`, etc.) — original to
  this project. AGPL-3.0-or-later.
- The canonical bench prompt is original.

When citing benchmark numbers, cite the fixture and hardware: `vllm-zig
v0.0.6 on TinyLlama-1.1B-Chat-v1.0, bench prompt "The capital of France
is" with max_new_tokens=8 greedy, i7-1065G7 ReleaseFast`.
