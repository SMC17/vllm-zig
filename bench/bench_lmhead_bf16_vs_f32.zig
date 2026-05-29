//! BF16 vs F32 lm_head matmul microbench -- isolates the bandwidth claim
//! from the v0.0.6 STATUS.md ("halving the B-matrix bytes pulled through
//! L2/L3 is exactly the bandwidth-bound case the BF16 kernel was built
//! for"). The end-to-end TinyLlama bench cannot isolate BF16 lm_head from
//! attention / MLP / pool-overhead noise; this bench can.
//!
//! Shapes exercised (M is always 1 -- single-token decode):
//!   - lm_head:            K=2048, N=32000   (65M elems = 130MB F32 / 65MB BF16)
//!   - mlp:                K=2048, N=5632    (11M elems)
//!   - qkv-proj:           K=2048, N=2048    ( 4M elems)
//!
//! For each shape: matmulSIMD (F32 B-matrix) vs matmulBF16SIMD (BF16
//! B-matrix). Same A. Same K, N. Same call shape (single-thread, no
//! pool, no tiling). The ONLY difference is dtype of the B matrix.
//!
//! Reports: ITERS-median and p10/p90 over ITERS samples (timing
//! per-call, not aggregated). Also reports effective B-matrix bandwidth
//! (GB/s) assuming each call reads the entire B once.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("vllm").tensor.Matrix;
const matmul = @import("vllm").matmul;
const matmul_bf16 = @import("vllm").matmul_bf16;
const bf16_mod = @import("vllm").bf16;

const Shape = struct { name: []const u8, K: usize, N: usize };
const SHAPES = [_]Shape{
    .{ .name = "qkv-proj   K=2048 N=2048 ", .K = 2048, .N = 2048 },
    .{ .name = "mlp        K=2048 N=5632 ", .K = 2048, .N = 5632 },
    .{ .name = "lm_head    K=2048 N=32000", .K = 2048, .N = 32000 },
};

const ITERS: usize = 30;
const SEED: u64 = 42;

fn percentile(samples_sorted: []const u64, p: f64) u64 {
    if (samples_sorted.len == 0) return 0;
    const idx_f = p * @as(f64, @floatFromInt(samples_sorted.len - 1));
    const idx: usize = @intFromFloat(@floor(idx_f));
    return samples_sorted[idx];
}

fn ltU64(_: void, a: u64, b: u64) bool {
    return a < b;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    var prng: std.Random.DefaultPrng = .init(SEED);
    const rng = prng.random();

    std.debug.print("=== matmul BF16 vs F32 lm_head shape sweep ===\n", .{});
    std.debug.print("build target: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("iters per sample: {d}, single-thread, no tiling, no pool\n\n", .{ITERS});

    std.debug.print("shape                       | f32 med us | f32 p10/p90 | bf16 med us | bf16 p10/p90 | speedup | f32 GB/s | bf16 GB/s\n", .{});
    std.debug.print("----------------------------+------------+-------------+-------------+--------------+---------+----------+----------\n", .{});

    var samples_f32: [ITERS]u64 = undefined;
    var samples_bf16: [ITERS]u64 = undefined;

    for (SHAPES) |s| {
        var a = try Matrix.init(alloc, 1, s.K);
        a.fillRandom(101);

        // F32 B
        var b_f32 = try Matrix.init(alloc, s.K, s.N);
        for (b_f32.data) |*x| x.* = rng.float(f32);

        // BF16 B (same shape, different dtype, fresh random -- we are
        // measuring path latency, not numerical equality)
        const b_bf16 = try alloc.alloc(u16, s.K * s.N);
        for (b_bf16) |*x| x.* = bf16_mod.f32ToBf16(rng.float(f32));

        var c_f32 = try Matrix.init(alloc, 1, s.N);
        var c_bf16 = try Matrix.init(alloc, 1, s.N);

        // Warmup once each
        try matmul.matmulSIMD(&a, &b_f32, &c_f32);
        try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_bf16);

        // F32 samples -- per-call timing
        var iter: usize = 0;
        while (iter < ITERS) : (iter += 1) {
            const t0 = std.Io.Clock.awake.now(io);
            try matmul.matmulSIMD(&a, &b_f32, &c_f32);
            const ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            samples_f32[iter] = ns;
        }

        // BF16 samples
        iter = 0;
        while (iter < ITERS) : (iter += 1) {
            const t0 = std.Io.Clock.awake.now(io);
            try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_bf16);
            const ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            samples_bf16[iter] = ns;
        }

        std.mem.sort(u64, &samples_f32, {}, ltU64);
        std.mem.sort(u64, &samples_bf16, {}, ltU64);

        const f32_med_us = samples_f32[ITERS / 2] / 1000;
        const f32_p10_us = percentile(&samples_f32, 0.10) / 1000;
        const f32_p90_us = percentile(&samples_f32, 0.90) / 1000;
        const bf16_med_us = samples_bf16[ITERS / 2] / 1000;
        const bf16_p10_us = percentile(&samples_bf16, 0.10) / 1000;
        const bf16_p90_us = percentile(&samples_bf16, 0.90) / 1000;

        const f32_med_ns = samples_f32[ITERS / 2];
        const bf16_med_ns = samples_bf16[ITERS / 2];

        const speedup = @as(f64, @floatFromInt(f32_med_ns)) /
            @as(f64, @floatFromInt(bf16_med_ns));

        // Effective B-matrix bandwidth: one full pass of B per call.
        // F32 B = K*N*4 bytes. BF16 B = K*N*2 bytes.
        const f32_bytes: f64 = @floatFromInt(s.K * s.N * 4);
        const bf16_bytes: f64 = @floatFromInt(s.K * s.N * 2);
        const f32_sec: f64 = @as(f64, @floatFromInt(f32_med_ns)) / 1.0e9;
        const bf16_sec: f64 = @as(f64, @floatFromInt(bf16_med_ns)) / 1.0e9;
        const f32_gbs: f64 = (f32_bytes / f32_sec) / 1.0e9;
        const bf16_gbs: f64 = (bf16_bytes / bf16_sec) / 1.0e9;

        std.debug.print(
            "{s} | {d:>10} | {d:>4}/{d:>4} | {d:>11} | {d:>5}/{d:>5} | {d:6.2}x | {d:7.2} | {d:7.2}\n",
            .{
                s.name,
                f32_med_us,
                f32_p10_us,
                f32_p90_us,
                bf16_med_us,
                bf16_p10_us,
                bf16_p90_us,
                speedup,
                f32_gbs,
                bf16_gbs,
            },
        );
    }
}
