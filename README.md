# vllm-zig

Production LLM serving substrate. Zig kernels + BEAM/OTP scheduler.
PagedAttention design + continuous batching, BEAM supervision tree.

**Status:** DESIGN. No code yet. See `ARCHITECTURE.md` for the full
design doc that gates Phase 1 implementation.

**Portfolio position:** Artifact #1 (headline) in the 8-artifact
jaw-drop portfolio (see `~/portfolio/README.md`).

## Why

vLLM (Berkeley) defined the de facto open-source LLM-serving design
with PagedAttention + continuous batching. The algorithm is sound;
the Python implementation carries structural costs (GIL, MP fragility,
observability gap, cold-start) that are implementation properties,
not algorithmic ones.

`vllm-zig` reimplements the same algorithmic core with:

- **Zig kernels** for the tight numeric path (attention, sampling,
  KV-cache block management)
- **BEAM/OTP supervisor tree** for the scheduler / admission / fairness
  / HTTP / observability layer
- **Phoenix LiveView** as a first-class observability surface

This is an architectural claim — not a performance one until Phase 2
benchmarks land.

## Phase plan

See `ARCHITECTURE.md` §13. Headline:

- **Phase 0:** ARCHITECTURE.md (done 2026-05-20)
- **Phase 1:** Single-sequence prefill+decode on TinyLlama-1.1B
- **Phase 1.5:** PagedAttention + block pool
- **Phase 2:** Continuous batching + OpenAI HTTP front-end + first
  honest bench vs vLLM
- **Phase 3:** Llama-3.2-8B + LiveView dashboard + v0.1 public release

4–6 months solo+Claude scale, hardware-gated at Phase 1 perf and
beyond (Ampere+ required).

## Honest non-claims

- No code yet. This is design-before-code per the engineering-bar
  rule.
- No "vLLM killer" framing. vLLM is the prior art. We cite, we bench,
  we publish losses.
- Hardware-gated at every perf phase. If Ampere+ access doesn't
  materialize, the project halts at the Turing-correctness floor.

## License

AGPL-3.0-or-later. See `LICENSE`.

Server software at AGPL means SaaS users have to release modifications.
Together AI / Modal / Anyscale / similar can deploy via commercial
license negotiation or upstream contribution.

## Citations

Architectural inheritance:

- Kwon et al., *PagedAttention*, SOSP 2023
- Yu et al., *Orca: Continuous Batching*, OSDI 2022

See `ARCHITECTURE.md` for the full citation list + audience-map +
inbound-mechanics plan.
