# vllm-zig Agent Entry Point

Pure-Zig LLM serving **substrate** (paged attention, BF16 kernels, thread pool).

**Not vLLM-equivalent.** Do not brand, title, or claim production serving
parity with Berkeley vLLM / SGLang. STATUS uses explicit non-claims — keep them.

## Before you change anything

1. Read `README.md` and `STATUS.md` (Type-I notes on benches matter).
2. Run the proof gate before claiming tests green.
3. Dependencies live under `deps/` (gitignored checkouts or symlinks):

```text
deps/safetensors-zig  -> path dep (clone or symlink)
deps/tokenizers-zig   -> path dep (clone or symlink)
```

Bootstrap: `./scripts/fetch-deps.sh` (uses `gh` or sibling symlinks).
CI checks private siblings into `deps/` via multi-checkout (see `.github/workflows/ci.yml`).
Private GitHub archive `url`+`hash` is not used: unauthenticated fetch 404s.

## Proof gate

```sh
zig build test
```

Optional (operator intent): `zig build bench-matmul`, `infer-tinyllama`, etc.
Benches need matched load; do not revive load-elevated numbers as truth.

## Allowed work

- docs/status reconciliation with honest non-claims
- unit-tested kernel / page-manager / pool fixes
- proof-gate maintenance
- path-dep → git dependency conversion for clean clone

## Forbidden without explicit operator intent

- production serving claims; multi-GPU/CUDA as "shipped"
- live OpenAI HTTP front-end work
- large weight blobs in the public tree
- vLLM-equivalent branding or README title changes toward that
- publishing latency numbers without matched clean-load protocol

## Escalation

Ask before production serving claims, CUDA path work, or publishing latency
numbers without matched load notes in STATUS.

## Hygiene

Org templates: `SMC17/stax-hygiene`. Highest graph priority: clean-clone deps.
