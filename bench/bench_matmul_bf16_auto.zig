//! Sweep matmulBF16Auto vs matmulBF16SIMD across TinyLlama-class
//! shapes. On macOS the Auto path routes large B-matrices through
//! cblas_sgemm; smaller shapes stay on the in-process SIMD path.
//!
//! Shapes exercised (M is always 1 — single-token decode):
//!   - q/k/v projection:   K=2048, N=2048    ( 4M elems)
//!   - mlp gate/up/down:   K=2048, N=5632    (11M elems)
//!   - lm_head:            K=2048, N=32000   (65M elems)
//!   - tiny (under threshold): K=2048, N=256  (524K elems)
//!
//! Build requirement: `-framework Accelerate` on macOS.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("vllm").tensor.Matrix;
const matmul_bf16 = @import("vllm").matmul_bf16;
const bf16_mod = @import("vllm").bf16;

const Shape = struct { name: []const u8, K: usize, N: usize };
const SHAPES = [_]Shape{
    .{ .name = "tiny           K=2048 N=256  ",   .K = 2048, .N = 256 },
    .{ .name = "qkv-proj       K=2048 N=2048 ",   .K = 2048, .N = 2048 },
    .{ .name = "mlp            K=2048 N=5632 ",   .K = 2048, .N = 5632 },
    .{ .name = "lm_head        K=2048 N=32000",   .K = 2048, .N = 32000 },
};

const ITERS: usize = 30;
const SEED: u64 = 42;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    var prng: std.Random.DefaultPrng = .init(SEED);
    const rng = prng.random();

    std.debug.print("=== matmulBF16Auto vs matmulBF16SIMD shape sweep ===\n", .{});
    std.debug.print("build target: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("Auto threshold: {d} elems (route to cblas above)\n\n", .{1_000_000});
    std.debug.print("shape                       | simd ms | auto ms | speedup | b_elems\n", .{});
    std.debug.print("----------------------------+---------+---------+---------+--------\n", .{});

    for (SHAPES) |s| {
        var a = try Matrix.init(alloc, 1, s.K);
        a.fillRandom(101);

        const b_bf16 = try alloc.alloc(u16, s.K * s.N);
        for (b_bf16) |*x| x.* = bf16_mod.f32ToBf16(rng.float(f32));

        var c_simd = try Matrix.init(alloc, 1, s.N);
        var c_auto = try Matrix.init(alloc, 1, s.N);

        // Warmup
        try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_simd);
        try matmul_bf16.matmulBF16Auto(alloc, &a, b_bf16, s.K, s.N, &c_auto);

        // Time SIMD
        const t0 = std.Io.Clock.awake.now(io);
        var iter: usize = 0;
        while (iter < ITERS) : (iter += 1) {
            try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_simd);
        }
        const ns_simd: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        // Time Auto
        const t1 = std.Io.Clock.awake.now(io);
        iter = 0;
        while (iter < ITERS) : (iter += 1) {
            try matmul_bf16.matmulBF16Auto(alloc, &a, b_bf16, s.K, s.N, &c_auto);
        }
        const ns_auto: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        const simd_ms = ns_simd / ITERS / 1_000_000;
        const auto_ms = ns_auto / ITERS / 1_000_000;
        const speedup = @as(f64, @floatFromInt(ns_simd)) / @as(f64, @floatFromInt(ns_auto));
        std.debug.print("{s} | {d:7} | {d:7} | {d:6.2}x | {d}\n", .{ s.name, simd_ms, auto_ms, speedup, s.K * s.N });
    }
}
