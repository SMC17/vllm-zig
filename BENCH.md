# BENCH -- vllm-zig perf measurement methodology + v0.0.6 results

**Status:** in-progress, internal substrate work
**Date:** 2026-05-29
**Author:** Sean Collins
**Repo:** vllm-zig @ v0.0.6
**Audience:** the next time anyone (including me) tries to claim a
v0.0.6 perf number, this is the document they read first.

---

## Why this document exists

`STATUS.md` records two TinyLlama-1.1B decode numbers:

- v0.0.5 historical: **581 ms/tok** on a clean machine (held up as the
  v0.0.4→v0.0.5 regression-closer baseline)
- v0.0.6 measured: **1330 ms/tok** on a machine at **load average ~44**
  (16+ `forge_athlete_kernel` processes saturating the 8-core box)

The 1330 ms/tok number is engineering-correct (the BF16 lm_head wiring
is live, kv-cache repair holds, build.zig wiring restores test +
ReleaseFast bench surface) but it is **not** a perf claim. The two
numbers cannot be compared. Three confounders mix in:

1. System load (8 vllm-zig workers contending with 15+ already-resident
   background processes)
2. PagedAttention entered the forward path between v0.0.5 and v0.0.6
3. BF16 lm_head wiring is new -- its effect is not isolated

The 30-day perf gate is **200 ms/tok by 2026-06-25**. Until we have
clean-machine numbers and an isolated BF16 win, that gate is
unmeasured, not unmet.

This document defines:

- The hardware and OS state benches MUST be run against
- The load preconditions a bench MUST satisfy before any number it
  produces gets recorded
- The bench shapes, repeat counts, and statistical reporting
- What the v0.0.6 numbers actually show

If any number on this page would not survive hostile review by a perf
engineer who reads `/proc/loadavg` first, that number does not belong
on this page.

---

## 1. Hardware

```
CPU model:        Intel(R) Core(TM) i7-1065G7 (Ice Lake-U, 10nm)
sockets:          1
cores:            4 (physical)
threads:          8 (SMT-2)
base / max MHz:   1300 / 3900 (Turbo Boost 3.0)
L1d / L1i / core: 48 KiB / 32 KiB
L1d total:        192 KiB
L1i total:        128 KiB
L2 / core:        512 KiB
L2 total:         2 MiB
L3 shared:        8 MiB
SIMD:             AVX-512 (Ice Lake), AVX2, FMA, BF16 instr NOT in this gen
                  (BF16 widening here is software bitshift, not VNNI)
RAM:              16 GiB DDR4 (this generation is DDR4-3200, not DDR5)
NUMA:             single node, CPUs 0-7
OS:               Linux 7.0.3-arch1-1 (kernel reported via uname -r)
governor:         powersave (cpu0 confirmed; turbo not disabled, but
                  effective frequency held at 1.6 GHz under load)
intel_pstate:     no_turbo=0 (turbo allowed; governor decides)
```

**Implications for the 200 ms/tok target.** TinyLlama-1.1B has ~1.1B
params. End-to-end decode at single-token batch-1 is bandwidth-bound:
every forward pass walks the full parameter set once. BF16 storage =
~2.2 GB read per token. DDR4-3200 single-channel theoretical peak =
~25.6 GB/s. Practical sustained on a laptop CPU under realistic load is
roughly half of that. Floor estimate for bandwidth-bound batch-1
decode on this hardware:

```
2.2e9 bytes / 12.8e9 bytes/sec ≈ 172 ms/tok    (best-case bandwidth)
2.2e9 bytes / 8.0e9 bytes/sec  ≈ 275 ms/tok    (realistic sustained)
```

So **200 ms/tok is within ~15% of the bandwidth floor on this CPU.**
Hitting it requires not just BF16 weights end-to-end (currently only
lm_head is BF16), but also enough kernel discipline that compute is
not blocking the memory pipeline. The 581 ms/tok v0.0.5 baseline is
~2.9× the realistic floor. The gap is real perf work, not
microoptimisation.

---

## 2. Load preconditions -- refuse-to-bench rule

A bench number that gets recorded MUST be produced with the box in
the following state:

```
load average 1m   ≤ 2.0     (on this 8-core box)
load average 5m   ≤ 2.5
%CPU busy (4s)    ≤ 25%
%CPU iowait (4s)  ≤ 2%
no zig build or nix build process in `ps aux`
no ollama runner serving a model
no forge_athlete_kernel processes
no other user-owned compute daemon at >1% CPU
```

If any of these fail, the bench MUST either:

- Not run, and `BENCH.md` records "skipped -- load X, 4s-busy Y%"
- Run anyway with an explicit `# LOAD-VIOLATED` banner on every row,
  and the resulting numbers tagged "noise-dominated, not a perf claim"

The second mode exists for triage (do the kernels at least produce
nonzero, finite output? Does the bench harness compile and run end-to-
end?). It does NOT produce comparable numbers. STATUS.md's 1330 ms/tok
line is an example of mode-2 output that the v0.0.6 commit message
unfortunately did not flag clearly enough.

### How to bring load down on this box

The 17-day uptime accumulated these background daemons:

- `forge_athlete_kernel` -- 16+ instances, each ~12% CPU, total ≈ 200% of an
  8-core box. Long-running compute experiments owned by the user. **Not
  to be killed without explicit user authorisation** (per the
  "Son-of-Anton" memory rule: one-shot kill authorisation is not
  general kill authorisation).
- `ollama runner` -- model server. Stop with `pkill ollama` IF the
  user confirms.
- `nix build .#homeConfigurations.stax.activationPackage` -- system
  rebuild in progress. Wait for completion or interrupt with user
  authorisation.
- `zig build` (faiss-zig, sovereign-daw, etc.) -- sibling-repo
  compiles. Wait for completion.

**This session ran benches without quiescing forge_athlete_kernel.**
The numbers below are mode-2 (LOAD-VIOLATED). See section 4 for the
honest result of that decision.

---

## 3. Bench methodology

### 3.1 Prompt + token budget

For end-to-end TinyLlama (`bench-tinyllama-pool`):

```
prompt:      "The capital of France is"
max_new:     8
seed:        none (greedy via argmax; deterministic given identical
             prompt + model weights)
fixture:     ../safetensors-zig/tests/fixtures/tinyllama-real
             (2.2 GB BF16 model.safetensors)
threads:     std.Thread.getCpuCount() -- 8 on this box
build mode:  ReleaseFast (vllm + safetensors + tokenizers all
             compiled in ReleaseFast via mod_fast in build.zig)
```

### 3.2 Microbench shapes

`bench-lmhead-bf16-vs-f32` (new in this run):

```
shapes (M=1, K=2048):
  qkv-proj   N=2048    4M elems    16 MB F32 / 8 MB BF16
  mlp        N=5632   11M elems    44 MB F32 / 22 MB BF16
  lm_head    N=32000  65M elems   262 MB F32 / 131 MB BF16
iters per sample:  30
SEED:              42
single-thread, no pool, no tiling -- isolates dtype-only effect
```

The lm_head shape is the headline case: B-matrix is 262 MB F32 and
131 MB BF16. L3 on this box is 8 MB, so neither fits -- both pass
through DRAM. The BF16 path should pull half the bytes per call. If
the bandwidth-bound claim holds, BF16 should be ~2× faster on lm_head
and roughly comparable on qkv-proj (qkv 16 MB F32 still busts L3 but
not by much).

### 3.3 Reporting

For each (shape, dtype):

- 30 per-call timings, captured one-by-one (NOT iters-then-divide)
- Sorted, then report **median, p10, p90**
- Effective B-matrix bandwidth = bytes(B) / median-second
- Speedup = F32-median / BF16-median

End-to-end TinyLlama bench prints first-token ms, total ms, decode
ms/tok. The bench as written produces a single sample per
(prompt, max_new) run; to get median + p10/p90 we run the harness
multiple times and aggregate externally.

---

## 4. Results -- v0.0.6 under LOAD-VIOLATED conditions

### 4.1 Pre-bench state

The box at the start of the bench window:

```
date:                 2026-05-29T08:30:14-04:00 (pre-bench probe)
uptime:               17 days, 17:15
load avg (1/5/15):    52.51 / 56.57 / 59.40
4s wall %CPU busy:    100.0%
4s wall %iowait:      0.00%
RAM free / avail:     4.5 GiB / 8.5 GiB
swap used:            6.8 GiB of 19 GiB
governor:             powersave (cpu0 at 1.6 GHz, capped from 3.9 GHz max)
contending procs:     16 × forge_athlete_kernel (each ~12% CPU,
                          running since May 24-25, owned by user,
                          not killed -- see section 2)
                      1  × ollama runner (~21% CPU)
                      1  × nix build activationPackage (~24% CPU)
                      2  × zig build (faiss-zig, sovereign-daw,
                          ~18% CPU each -- sibling-repo compiles)
```

Load did descend to ~36 by the time the e2e bench finished (the nix
build and one of the sibling zig builds completed during the run),
but remained ~18× over the bench precondition for the entire window.
Per-process CPU share for `bench-tinyllama-pool` measured at 2.3% of
one logical core (out of 8) during peak -- i.e. roughly 1/35th of the
effective compute the bench would see on an idle machine.

**Load violated by ~25× on the 1m average.** Every number below is
mode-2 (load-violated, noise-dominated). They are NOT v0.0.6 perf
claims.

### 4.2 BF16 vs F32 lm_head microbench

Three back-to-back runs of `bench-lmhead-bf16-vs-f32`. Each run is 30
per-call samples × 3 shapes × 2 dtypes, ITERS=30, SEED=42, single-
thread, no pool, no tiling. The "med" column is the per-call median
across the 30 samples within that run. Three runs gives a feel for
inter-run drift under heavy contention.

```
RUN 1 -- load 1m=39.19
shape                       | f32 med us | f32 p10/p90 | bf16 med us | bf16 p10/p90  | speedup | f32 GB/s | bf16 GB/s
qkv-proj   K=2048 N=2048    |       1797 | 1755/1848   |        1390 | 1308/1459     |   1.29x |    9.33  |    6.03
mlp        K=2048 N=5632    |       6566 | 5108/9016   |        6588 | 3961/7806     |   1.00x |    7.03  |    3.50
lm_head    K=2048 N=32000   |      35423 | 33012/307839|       33131 | 26794/1001741 |   1.07x |    7.40  |    3.96

RUN 2 -- load 1m=38.44
shape                       | f32 med us | f32 p10/p90 | bf16 med us | bf16 p10/p90 | speedup | f32 GB/s | bf16 GB/s
qkv-proj   K=2048 N=2048    |       1897 | 1744/2106   |        1511 | 1370/2561    |   1.26x |    8.84  |    5.55
mlp        K=2048 N=5632    |       5648 | 5299/6095   |        4473 | 4087/5552    |   1.26x |    8.17  |    5.16
lm_head    K=2048 N=32000   |      46081 | 33535/446130|       33973 | 27615/227835 |   1.36x |    5.69  |    3.86

RUN 3 -- load 1m=36.14
shape                       | f32 med us | f32 p10/p90  | bf16 med us | bf16 p10/p90 | speedup | f32 GB/s | bf16 GB/s
qkv-proj   K=2048 N=2048    |       1769 | 1674/692659  |        1319 | 1277/1349    |   1.34x |    9.48  |    6.36
mlp        K=2048 N=5632    |       4776 | 4738/4828    |       57155 | 3631/991676  |   0.08x |    9.66  |    0.40
lm_head    K=2048 N=32000   |      33148 | 29801/7609250|       27357 | 24062/80237  |   1.21x |    7.91  |    4.79

3-run median of per-run medians (the headline rows)
shape                       | f32 med us | bf16 med us | speedup | f32 GB/s | bf16 GB/s
qkv-proj                    |       1797 |        1390 |   1.29x |    9.33  |    6.03
mlp                         |       5648 |        4473 |   1.26x |    8.17  |    5.16
lm_head                     |      35423 |       33131 |   1.07x |    7.40  |    3.96
```

Wide p90/median ratios (lm_head F32 run 3: p90 = 7.6 sec when median
= 33 ms, a 230× spread) are the load-violated signature. Run 3's MLP
BF16 row is the clearest contention symptom: median 57 ms is 12×
worse than F32's 4.7 ms, but p10 = 3.6 ms -- i.e. the kernel did
finish quickly in some samples and got preempted hard in others. The
3-run-median row treats these as drawn from a noisy distribution and
takes the per-shape median across runs.

### 4.3 End-to-end TinyLlama decode

Single run of `bench-tinyllama-pool` under load 1m = 40.86 at start,
descending to ~36 over the 41-minute run. Same v0.0.6 code for both
paths; (A) is the v0.0.4 ad-hoc-spawn matmul, (B) is the v0.0.5
persistent pool. Both paths exercise paged-attention KV-cache and
BF16 lm_head.

```
                       | (A) no pool   | (B) pool      | (B/A speedup)
model load ms          |        751879 |        751879 | (single load, shared between A and B)
first-token (pref+1) ms|        276525 |        232121 |  1.19x
total ms (8 tokens)    |       1227544 |        570721 |  2.15x
decode ms/tok          |        135859 |         48371 |  2.81x
```

For context, the historical clean-machine numbers in `STATUS.md`:

```
                       | clean v0.0.4 | clean v0.0.5 | clean v0.0.6 (load 44) | load-violated v0.0.6 here
decode ms/tok          |          821 |          581 |                   1330 |                     48371
```

The 48,371 ms/tok pool path number is **~83× the clean-machine v0.0.5
baseline of 581 ms/tok**. That ratio is roughly the inverse of the
per-process CPU share the bench got (2.3% of a single core out of 8
cores under load 36-40, so ~30-50× slowdown predicted; observed 83×
is in the right order of magnitude given turbo throttling on top of
the share loss).

The single number for total ms (1,227,544 vs 570,721, 2.15× faster)
and decode ms/tok (135,859 vs 48,371, 2.81× faster) are the only
measurements this run produced. n=1 per cell. The 2.81× pool-vs-spawn
ratio is suggestive but not statistically established. The historical
clean ratio for pool-vs-spawn was 1.41× (STATUS.md v0.0.5 row). The
gap (2.81 under load vs 1.41 clean) is consistent with thread-spawn
costs ballooning under contention because every `Thread.spawn` waits
for scheduler attention while the pool's persistent workers already
hold their slots.

### 4.4 What these numbers actually show

**BF16 lm_head is NOT a 2× win on this CPU.** The 3-run median
speedup at the lm_head shape (M=1, K=2048, N=32000) is **1.07×**,
well short of the 2× implied by "halving the B-matrix bytes pulled
through L2/L3". Effective bandwidth tells the story:

- F32 lm_head sustained ≈ 7.4 GB/s at median
- BF16 lm_head sustained ≈ 3.96 GB/s at median
- Ratio = 1.87×, i.e. BF16 moves bytes through DRAM at roughly half
  the rate of F32 -- exactly the bandwidth advantage predicted

The byte advantage is real. The wall-time advantage is not, because
the BF16 path is **not bandwidth-bound on this CPU under this kernel
implementation**. The widening sequence (u16 → u32 → shift-left-16 →
bitcast to f32) is 4 instructions per BF16 lane in the inner loop;
combined with the FMA, that is roughly 5× the instructions of the
F32 path's single broadcast-multiply-add. At 1.6 GHz under load the
instruction throughput becomes the binding constraint, not memory.

This is a **Type-1 catch on STATUS.md's v0.0.6 BF16-as-perf-win
framing.** The STATUS.md commentary says the BF16 wiring "should be a
win on a clean machine (bandwidth-bound case)". The microbench here
shows it is a modest win (1.07-1.36× at lm_head, 1.00× at mlp in run
1) but the assumed mechanism is wrong: the kernel is compute-bound on
widen-instructions, not bandwidth-bound. The clean-machine number
will still be a win (BF16 is also touching half the page cache, half
the TLB pressure, half the prefetch), but the magnitude prediction
("2×") will not hold without a wider SIMD lane or hardware BF16.

Three concrete fixes to recover the bandwidth-bound regime:

  (a) Wider SIMD (AVX-512 zmm, 16 lanes instead of 8) so one widen
      sequence services twice as many multiply-adds, doubling the
      compute-vs-memory ratio. Ice Lake has AVX-512F. Worth trying.
  (b) Hardware BF16 instructions (AVX-512 BF16 VNNI) -- NOT available
      on this Ice Lake CPU; first appeared in Cooper Lake / Sapphire
      Rapids.
  (c) Cache-tiled BF16 matmul so the widen amortises across more
      multiplies before the L1 panel turns over.

**The pool-vs-spawn 2.81× margin holds, larger under load than on a
clean machine.** This is consistent with v0.0.5's persistent-pool
substrate being the right call. No claim adjustment needed there.

**The 200 ms/tok target is not measured here.** A bench run at
load = 40 cannot speak to it. The honest gap is "we don't know what
v0.0.6 does on a clean machine"; the box has been continuously
saturated for 17 days. Section 8 below tracks that as the primary
outstanding work item.

---

## 5. Gap to 200 ms/tok -- where the time goes

Before any concrete optimisation can be prioritised, we need a
per-layer or per-region breakdown of where decode time is spent in
v0.0.6. Today the bench reports total decode ms/tok with no
breakdown. The honest answer to "where does the time go" is: we don't
know yet without instrumentation.

Rough first-principles breakdown for TinyLlama-1.1B decode on
batch-1, BF16 weights end-to-end (which we don't have -- most weights
are still F32 after load conversion):

```
component                  bytes touched per token
embed lookup                  ~4 KB (one row of embed)
22 × attention proj qkv      22 × 12 MB F32 = 264 MB (would be 132 MB BF16)
22 × attention out proj      22 × 16 MB F32 = 352 MB (176 MB BF16)
22 × MLP gate + up + down    22 × 132 MB F32 ≈ 2.9 GB F32 (1.45 GB BF16)
lm_head                       262 MB F32 OR 131 MB BF16 (← BF16 wired)
KV-cache scatter/gather       proportional to seq_len
RoPE + RMSNorm + softmax      compute-bound, small
                            -----
total per-token bytes        ~3.8 GB F32 / ~1.9 GB BF16
```

This estimate says even with **everything** in BF16, single-channel
DDR4-3200 puts the bandwidth floor near 75 ms/tok (1.9 GB / 25 GB/s),
and realistic sustained near 150 ms/tok. The 200 ms/tok target is
achievable but only with weights end-to-end in BF16, not just lm_head.

---

## 6. Next perf lever ranking -- informed by the microbench

The lm_head microbench inverted my prior. Old #1 (end-to-end BF16
storage) was predicated on BF16 being a 2× win at lm_head shape; the
median measurement is 1.07×. The bandwidth advantage is real but the
kernel is compute-bound on widen-instructions, not bandwidth-bound.
Revised ranking by expected end-to-end ms/tok reduction:

1. **Pool the BF16 matmul path.** Today `matmul_bf16.matmulBF16SIMD`
   is single-thread. The pool only dispatches `matmul.matmul*`
   variants. Adding `Pool.matmulBF16Auto` shaped after
   `matmulRowSplit` / `matmulColSplit` lets 8 cores share the BF16
   widen + multiply work. Single-thread lm_head measured 33 ms; an
   8-thread split at the historical pool scaling factor of ~5×
   effective (the v0.0.5 row was 1.19× over single-thread because
   the F32 path was already memory-bound; the BF16 path is compute-
   bound on widen, so the 8-thread scaling should be closer to 4-6×)
   puts lm_head at 6-8 ms per call. Direct effect on end-to-end
   decode: lm_head is one of ~88 matmuls per token, but it's by far
   the largest; cutting it from 33 ms to 7 ms saves ~26 ms/tok.
   **Estimated end-to-end win: 5-15% of decode wall time at clean-
   machine v0.0.5 baseline (581 ms/tok → ~530 ms/tok).** Cheap to
   implement, no kernel rewrite, drop-in dispatch.

2. **Cache-tiled BF16 matmul.** Currently `matmulBF16SIMD` is the
   straight i-k-j loop. The F32 `matmulSIMDTiled` measured up to
   2.5× at large N (per the inline comments). The same tiling
   applied to BF16 amortises the widen across more multiply-adds
   inside the L1 panel, which is exactly the lever that recovers
   bandwidth-bound behaviour. **Estimated win at lm_head: 1.5-2.5×
   over single-thread BF16 SIMD.** Compose with #1 (pool the tiled
   kernel) for the headline win.

3. **AVX-512 zmm widen path.** Ice Lake has AVX-512F. Doubling the
   SIMD lane from 8 (`@Vector(8, f32)`) to 16 (`@Vector(16, f32)`)
   halves the per-output-element widen cost. Speculative on this
   exact CPU because turbo and AVX-frequency-licensing may eat the
   throughput gain back, but worth a microbench. **Estimated win at
   lm_head: 1.3-1.7× over the 8-lane path, on top of #1 and #2.**

4. **End-to-end BF16 storage for projection + MLP weights.** Still on
   the list, but demoted. The bandwidth advantage is half of memory
   bytes; on a clean machine that translates to roughly half of
   DRAM-pressure time, but the dominant cost in v0.0.5 (581 ms/tok)
   was already not pure DRAM bandwidth (single-thread fp32 lm_head
   was 1.8 ms on clean baseline per the v0.0.5 bench tables; 22
   blocks × ~5 matmuls × ~2-5 ms each ≈ 220-550 ms, mostly compute).
   Wins are real but smaller than #1-3. Land this AFTER #1 / #2 so
   the comparison is fair and pool wins are not double-counted.

5. **PagedAttention overhead audit.** v0.0.6 added per-token block-
   table dispatch to `model.forward`. At seq_kv = 8 this should be
   noise, but it has not been measured. Audit (instrument
   `forward.zig` per-region) before optimising. Likely no perf cost
   but the assumption deserves a number.

6. **Per-region instrumentation in `Model.forward`.** Not a perf
   lever itself; it tells us where time goes so the lever pick is
   informed. Without it #1 above is the right first lever by
   first-principles arithmetic, but a real breakdown moves it from
   "estimated 5-15%" to a measured prediction.

The actual lever pick stays gated on a clean-machine baseline.
Optimising against load-violated numbers is a category error.

---

## 7. Reproduction

```bash
# 1. Verify load
uptime    # 1m load MUST be ≤ 2.0
read _ u1 n1 s1 i1 w1 q1 sq1 _ < /proc/stat ; sleep 4 ; \
  read _ u2 n2 s2 i2 w2 q2 sq2 _ < /proc/stat ; \
  echo "4s busy: $(awk "BEGIN{printf \"%.1f\", \
    ((($u2-$u1)+($n2-$n1)+($s2-$s1)+($q2-$q1)+($sq2-$sq1)))*100/\
    ((($u2-$u1)+($n2-$n1)+($s2-$s1)+($i2-$i1)+($w2-$w1)+($q2-$q1)+($sq2-$sq1)))" )%"

# 2. Microbench (BF16 vs F32 lm_head)
zig build bench-lmhead-bf16-vs-f32

# 3. End-to-end TinyLlama
zig build bench-tinyllama-pool

# 4. (Optional) BF16 SIMD vs Auto path sweep
zig build bench-matmul-bf16-auto
```

---

## 8. Outstanding work

- [ ] Re-run benches on a clean machine (load ≤ 2.0).
- [ ] Instrument `Model.forward` for per-region timing (attention vs
      MLP vs lm_head vs sample) so the next bench produces a
      breakdown not just a total.
- [ ] Pure-v0.0.4 and pure-v0.0.5 reproducer: tag the commits, write
      a script that checks out, builds, runs, and rolls back, so the
      "v0.0.5 was 581 ms/tok" claim becomes verifiable not historical.
- [ ] Re-state the v0.0.6 number in `STATUS.md` once a clean number
      exists. Until then the v0.0.6 perf line stays explicitly
      "engineering-correct, not performance-confirmed".
