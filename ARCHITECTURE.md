# vLLM-zig — architecture design (pre-code)

**Status:** DESIGN. No code yet. This document is the gate the build
must pass before any Phase 1 implementation work begins.
**Author:** Sean Collins (stax)
**Last update:** 2026-05-20

---

## 1. Problem

Production LLM serving on GPU is bottlenecked by two coupled problems:

1. **KV-cache memory fragmentation under variable-length sequences.**
   Naive contiguous allocation wastes 60–80% of GPU memory on padding +
   slack. Solved by PagedAttention (Kwon et al., 2023) — block-based
   allocation with virtual-memory-style indirection.

2. **Scheduling heterogeneous request mixes (long prefill + short
   decode).** Prefill is compute-bound and benefits from large batches;
   decode is memory-bound and benefits from many concurrent in-flight
   requests. Naive uniform batching trades both off badly. Solved by
   continuous batching (Yu et al., Orca 2022).

Berkeley vLLM combined PagedAttention + continuous batching, became the
de-facto standard for open-weight LLM serving, ~30K GitHub stars.
SGLang, TensorRT-LLM, mistral.rs, and the LM Studio / Ollama ecosystem
either depend on vLLM or reimplement its core design.

The Python implementation has structural costs that the design itself
doesn't:

- **GIL contention** between scheduler + tokenizer + admission control.
- **Multiprocessing fragility** — worker death cascades through queues.
- **Observability gap** — Python's introspection at high QPS is weak.
- **Cold-start time** — large Python deps + slow weight load.

These are implementation properties, not algorithmic ones. The
algorithmic core is reimplementable in any language with a real ABI to
CUDA.

## 2. Goals and non-goals (v0.x)

### Goals

- **Reproduce the PagedAttention + continuous-batching design** in a
  Zig kernels + BEAM/OTP scheduler architecture.
- **Single-GPU, single-node, fp16 inference** at v0.1.
- **OpenAI-API-compatible HTTP front-end** at v0.2 (chat/completions,
  streaming, tools).
- **Phoenix LiveView observability surface** — request lifecycle,
  KV-cache occupancy, batch composition, kernel latency.
- **Audited benchmark against vLLM** on at least one open-weight model
  (Llama-3.2-3B or Qwen2.5-7B) with publicly published numbers,
  including losses.

### Non-goals at v0.x

- Multi-node distributed inference. v0.3+.
- Tensor / pipeline parallelism across GPUs. v0.3+.
- Quantization (AWQ / GPTQ / fp8). v0.3+ (Hopper-required for fp8).
- Speculative decoding. Future.
- LoRA / adapter swapping at request time. Future.
- KV-cache offload to CPU/disk. Future.
- All model families. v0.x supports Llama-architecture-family only
  (Llama-3, Qwen-2.5, Mistral-7B). Gemma, Phi, GPT-OSS later.

The non-goals are explicit so that scope creep doesn't kill the v0.1
ship date.

## 3. Architectural choice: Zig kernels + BEAM/OTP scheduler

The serving problem decomposes naturally:

- **Tight, latency-critical, numeric** work (KV-cache block management,
  attention kernels, sampling, dequantization) belongs in Zig + CUDA.
- **Coarse-grained, fairness-critical, fail-fast** work (request
  admission, batching, backpressure, retries, fairness across users,
  HTTP, streaming SSE, telemetry, model swapping) belongs in
  BEAM/OTP.

The C ABI between them is the canonical "Zig is C-compat" surface.
Communication patterns:

- **NIF (Native Implemented Function) on dirty schedulers** for sub-ms
  calls into Zig from BEAM. Used for: tokenize batch, prepare batch
  metadata, sample from logits, allocate/free KV blocks.
- **Port + shared memory** for long-running GPU launches. The Zig side
  owns a long-running CUDA stream and consumes a ring buffer of batch
  descriptors. The BEAM side writes batches into the ring and reads
  results.

Decision: start with NIF on dirty-CPU scheduler for v0.1. Move
GPU-launching paths to a Port if NIF latency becomes a problem (BEAM
schedulers can hiccup under NIF load > ~10ms). Both paths are well-trod.

### Why not pure Zig?

- Zig has no first-class concurrent runtime at the supervision-tree
  caliber of OTP. We'd reinvent supervisor logic, request queues,
  fairness scheduling, telemetry. That's ~2 person-years.
- BEAM's hot-code reload + observability via `:observer` / LiveDashboard
  is real production value, not a toy.
- The "vLLM-zig" naming is acceptable because the **kernel layer is
  Zig**. The scheduler being Elixir/OTP is the load-bearing
  architectural choice that makes the design plausible at solo+Claude
  scale.

### Why not pure Elixir?

- Tight numeric inner loops in BEAM are a non-starter. Even Nx/EXLA
  trades BEAM control for the same XLA/CUDA the Python ecosystem
  already has. The advantage we have is replacing the *scheduler +
  supervision* layer, not the kernel layer.

## 4. KV-cache design — PagedAttention faithful port

Block-based KV-cache, 16-token blocks per the original vLLM paper.
Block storage layout: contiguous tensor of shape

```
[total_blocks, 2 (K and V), heads, head_dim, block_size]
```

allocated once at startup, sized to fit GPU memory minus model weights
+ activation buffer + safety margin (10%).

**Block table per sequence.** Each in-flight sequence holds a list of
block indices. As the sequence grows, new blocks are allocated;
returning a sequence at end-of-generation frees its blocks to a free
list.

**Copy-on-write for shared prefixes.** Beam search / parallel sampling
with shared prompt: reference-count blocks. First write fork triggers a
block copy. This is the v0.1.5 optimization, not v0.1.

**Memory ownership.** Block-pool tensor lives in Zig (cudaMalloc'd
once at startup, ownership held by a singleton). BEAM gets a u32 block
ID handle. All block allocation/freeing goes through NIF calls that
update the free-list bitmap. BEAM never dereferences a CUDA pointer.

### Cache attention kernel

Loads K and V from non-contiguous block IDs into shared memory + accums
the softmax(QK^T) V output. This is the PagedAttention kernel from
vLLM. Implementation paths:

- **Phase 1**: Triton implementation (from the broader
  `triton-kernels` portfolio — see artifact #3). The autotuner does the
  work the hand-rolled CUDA would.
- **Phase 1.5**: CUTLASS-based hand-rolled kernel if Triton has
  measurable overhead on Hopper. Decision deferred until Phase 1
  numbers land.
- **Hardware floor**: Ampere+ for honest numbers. Turing (GTX 1650)
  works for correctness testing only; perf claims wait for at least
  RTX 30-series or A100 access.

## 5. Continuous batching scheduler

Orca-style continuous batching. The scheduler holds a request pool and
emits a new batch every iteration (i.e. every forward pass), picking
sequences from the pool by:

1. **Decode sequences** that have available KV blocks and are not
   blocked by exceeding `max_batched_tokens`.
2. **Prefill sequences** that fit in the remaining
   `max_batched_tokens` budget.

Mixed prefill+decode in one batch is the key design — `max_batched_tokens`
is the dial that trades prefill-throughput against decode-latency.

The scheduler runs as a GenServer on the BEAM side. Per-iteration:

```
1. Pop completed sequences from worker out-queue.
2. Stream tokens to HTTP clients for each completed sequence.
3. Free KV blocks for finished sequences (NIF call).
4. Pull new requests from the admission queue (subject to KV-cache
   capacity).
5. Build the next batch descriptor (block tables, position IDs,
   stop conditions).
6. Push the batch descriptor to the worker in-queue.
```

This loop is sub-ms on the BEAM side; the limit is the forward-pass
latency from the worker.

### Fairness

Per-API-key fair-share scheduling. Tracks tokens-served-per-key over a
sliding window and biases admission toward under-served keys when the
queue is non-empty. Implementation: priority queue with computed
priorities, recomputed every N admissions.

This is the unique BEAM/OTP advantage over Python — fairness logic at
this granularity is trivial in Erlang and a maintenance burden in
Python.

## 6. Supervisor tree

```
vllm_zig_sup (one_for_one)
├── kv_cache_sup (rest_for_one)
│   ├── KvCache.Pool             # owns block-pool NIF + free list
│   └── KvCache.Metrics          # block-occupancy gauge
├── worker_sup (one_for_one)
│   ├── Worker.GPU.0             # one GenServer per GPU
│   ├── Worker.GPU.1
│   └── ...
├── scheduler_sup (rest_for_one)
│   ├── Scheduler.Admission      # accepts incoming requests
│   ├── Scheduler.Batcher        # emits batches
│   └── Scheduler.Streamer       # streams tokens back to HTTP
├── http_sup (one_for_one)
│   ├── Plug.Cowboy listener     # OpenAI-compat HTTP endpoint
│   └── ApiKey.Tracker           # per-key usage + fairness
└── telemetry_sup (one_for_one)
    ├── Telemetry.Reporter       # Phoenix LiveView + Prometheus
    └── Telemetry.Aggregator
```

Failure modes:

- **KV-cache corruption** → `kv_cache_sup` is `rest_for_one`. A
  corrupted block pool restarts the workers downstream of it; in-flight
  requests fail with a 503 (clients retry).
- **Worker GPU OOM / CUDA error** → just that worker restarts. Other
  workers continue. The scheduler observes the worker death and
  reassigns its in-flight requests if possible.
- **Scheduler crash** → in-flight requests are NOT lost; the
  `Scheduler.Streamer` is decoupled and finishes streaming whatever the
  worker has already returned.
- **HTTP listener crash** → just the listener restarts; existing
  connections drop (clients retry).

This is the structural reliability win over Python multiprocessing.
Each Python worker death currently risks state loss in the
multiprocessing queue; OTP supervision tree makes the same scenario
safe.

## 7. Request lifecycle

```
1. HTTP POST /v1/chat/completions arrives at Plug.Cowboy.
2. Validation + ApiKey.Tracker auth + admission attempt.
3. Tokenize via tokenizers-zig NIF (CPU-side, fast).
4. Push request to Scheduler.Admission with: token IDs, max_tokens,
   stop sequences, sampling params, stream flag.
5. Scheduler.Batcher pops on its iteration loop, allocates KV blocks
   (NIF), builds batch descriptor.
6. Worker.GPU.N receives the batch descriptor, launches forward pass
   on its CUDA stream, polls completion.
7. Logits return → sample on GPU → token IDs flow back via NIF.
8. Scheduler.Streamer pushes incremental tokens to HTTP via SSE chunked
   transfer encoding.
9. At stop / max_tokens, KV blocks free; final tokens flush; response
   closes.
```

P50 first-token latency target at Phase 2: <100ms on a 7B model on a
single A10 / 4090 / 3090. (Comparable to vLLM at the same hardware
class — this is a parity target, not a beat-claim.)

## 8. Tokenizer integration

`tokenizers-zig` (portfolio artifact #4) provides:

- BPE tokenize / detokenize
- Pre-tokenization normalisation (NFC + GPT-2-style byte splits)
- Per-model special-token handling (chat templates loaded per Llama /
  Qwen / Mistral)

Per-request tokenize at admission time. For high-QPS workloads, batch-
tokenize is a future optimization (`tokenizers-zig` v0.2+).

## 9. Weight loading

`safetensors-zig` (portfolio artifact #4, shipped v0.0.1 2026-05-20)
provides the file-format reader. Loading flow:

1. On startup, mmap the safetensors file.
2. Walk the tensor index, validate shapes against the model architecture.
3. For each tensor, `cudaMemcpyAsync` from the mmap'd host buffer to
   pre-allocated GPU buffers (organized per layer for cache locality).
4. Synchronize on stream. Load time = HBM bandwidth-bound (~30s for a
   7B model on a 50 GB/s PCIe bus).

`safetensors-zig` v0.0.1 has the parse + read paths; v0.1 will add the
mmap-backed view for zero-copy host read. The GPU-side path is
`vllm-zig` work.

### Quantized models

v0.x supports fp16 + bf16 only. AWQ / GPTQ / fp8 require dequant kernels
or fused dequant-into-matmul; deferred to v0.3.

## 10. Hardware target ladder

| Phase | Hardware | Use |
|---|---|---|
| Phase 1 correctness | GTX 1650 Max-Q (Turing, 4GB) | Tiny model (TinyLlama-1.1B), correctness only |
| Phase 1 perf | RTX 30/40-series or A10/A100 (Ampere+) | First honest bench numbers |
| Phase 2 | RTX 4090 or A100 (Ampere/Ada) | Realistic single-GPU production scenario |
| Phase 3 | H100 / B200 | Hopper-specific paths (fp8, TMA, dist-shared-memory) |

Phase 1 perf and beyond are **hardware-gated**, NOT solo-engineering-
gated. The GTX 1650 floor exists today; everything beyond requires
either cloud GPU rental (Lambda / RunPod / Modal) or community-loaned
hardware.

Honest non-claim: the Turing baseline is the **only** measured hardware
at the time of this design doc. Phase 1 perf numbers do not yet exist.

## 11. Comparable upstream

| Project | Language | Pos-Mark |
|---|---|---|
| **vLLM** (Berkeley + Anyscale) | Python + CUDA | The reference. PagedAttention origin. Cite generously. |
| **SGLang** | Python + CUDA | RadixAttention; structured-gen interleave. |
| **TensorRT-LLM** (NVIDIA) | C++ + Python | Vendor-tuned; closed-source kernels. |
| **mistral.rs** (Eric Buehler) | Rust + CUDA/Metal | Cross-platform; Rust ecosystem comparable. |
| **candle** (HuggingFace) | Rust + CUDA/Metal | More framework than server. |
| **llama.cpp** | C + CPU/Metal/CUDA | The CPU/desktop baseline. |
| **mlc-llm** | TVM + multiple backends | Compilation-heavy approach. |

vLLM-zig's positioning: "Same algorithmic design as vLLM, BEAM/OTP
scheduler instead of Python multiprocessing, Zig kernels for the
numeric path. We cite the vLLM paper as prior art. We benchmark
against vLLM and publish losses where they happen."

## 12. Open questions to resolve before Phase 1 code

1. **NIF latency under load.** Dirty-CPU scheduler hiccups at >10ms
   NIF calls. Need to measure end-to-end NIF round-trip on a real
   workload. If >5ms, move GPU-launching to Port.
2. **Block-pool size pin policy.** Static (allocate-everything-at-
   startup) vs grow-on-demand. Static is simpler; grow risks
   fragmentation. v0.1 = static.
3. **Sampling on GPU vs CPU.** GPU sampling needs another Triton
   kernel; CPU sampling needs logits NIF'd back. CPU-sampling is the
   v0.1 default; GPU-sampling is v0.2.
4. **Logging / tracing model.** OpenTelemetry vs `:telemetry` only.
   v0.1 = `:telemetry` only; OTel is v0.2.
5. **Chat-template handling.** Llama-3 vs Qwen vs Mistral have
   different chat templates. Use HuggingFace tokenizer JSON which
   carries the template? Or hardcode per model family? Decision:
   parse the HF tokenizer JSON in `tokenizers-zig` and respect
   `chat_template` field. Mustache-render in Elixir at admission.
6. **Streaming SSE buffering.** Per-token flush risks small-write
   overhead. Batch every 50ms or every 5 tokens, whichever first.
7. **Multi-tenant memory pressure.** When KV-cache fills, which
   sequence(s) to preempt? Vanilla vLLM swaps to CPU; v0.1 has no
   swap, so the policy is "kill the youngest sequence." v0.2 adds
   CPU-swap.

## 13. Phase plan

| Phase | Calendar | Deliverables |
|---|---|---|
| **0 (this doc)** | Done 2026-05-20 | ARCHITECTURE.md, open questions named |
| **1 (months 1-2)** | 2026-Q3 start | Single-sequence prefill+decode on TinyLlama-1.1B fp16, no PagedAttention yet, no continuous batching. Correctness only. |
| **1.5 (month 3)** | 2026-Q3 | KV-cache block pool + PagedAttention kernel. Still single-sequence. |
| **2 (months 4-5)** | 2026-Q4 | Continuous batching scheduler. Llama-3.2-3B target. OpenAI HTTP front-end. First honest bench vs vLLM. |
| **3 (month 6)** | 2026-Q4 | Llama-3.2-8B (memory-pressure stress); Phoenix LiveView dashboard; public v0.1 release. |

**Phase 1 gate to start coding**: this design doc reviewed (a) by self
under the engineering-bar-before-distribution rule, (b) by at least one
external read of the PagedAttention paper and vLLM source code, (c) by
the open-questions list having a written answer for at least #1
(NIF-latency-measurement plan).

## 14. Honest non-claims

- **No code yet.** This is a design gate, not a release.
- **No perf claim against vLLM** at any phase until Phase 2 bench
  numbers land. The claim "Zig kernels + BEAM/OTP scheduler" is an
  *architectural* claim, not a performance one.
- **No multi-node claim.** v0.x is single-node-single-GPU. Multi-GPU
  is a different artifact and a Phase 3+ ambition.
- **No "vLLM killer" framing.** Position as "BEAM-native alternative
  with cite-and-bench discipline." vLLM is the prior art; vLLM-zig is
  the sibling that explores a different scheduler layer.
- **Hardware-gated phases.** Phase 1 perf and beyond require Ampere+
  GPU access. If that access doesn't materialize, the project halts
  honestly at the Turing-correctness floor.
- **Solo + Claude scale.** A 4–6 month timeline is realistic only if
  the design holds together. Open questions #1, #6, and #7 could
  invalidate parts of the plan and extend it by months.

## 15. Audience and inbound mechanics

Per portfolio doctrine:

- **NVIDIA inference team** — see PagedAttention port + Triton kernels
- **Together AI / Modal Labs / Anyscale** — see BEAM scheduler as
  alternative to vLLM Python in their stack
- **Apollo / METR / AISI** — see eval-serving substrate with explicit
  proof gates (every eval that calls a model needs serving)
- **Production-scale inference operators** — see scheduler + observability surface
- **vLLM team (Berkeley)** — see collaboration energy, not competition.
  Reach out post-Phase 2 with bench results + cite-and-bench paper.

Twitter / blog cadence on release: Phase 1.5 milestone (PagedAttention
landed) ships with a thread + diagrams. Phase 2 ships with bench
numbers (honest, including losses). Phase 3 ships with the LiveView
dashboard demo video.

## 16. Open-source license

AGPL-3.0-or-later. Matches the rest of the sovereign-stack portfolio.

The AGPL choice is deliberate for a server: SaaS users have to release
their modifications. If Together AI / Modal / Anyscale want to deploy
vLLM-zig in their commercial stack, they negotiate a commercial license
or contribute upstream. Standard sovereign-stack OSS posture.
