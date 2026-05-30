//! Lever-9 N-tile-major BF16 layout — per-shape microbench across the
//! five TinyLlama-1.1B BF16 matmul shapes.
//!
//! Validates that the layout win is consistent across the model's hot
//! matmul shapes, not specific to fused gate+up. Each shape is tested
//! at M=1 (decode regime) with 100 iterations per condition.
//!
//! Shapes covered:
//!   qkv_proj     M=1 K=2048 N=2560  (fused Q+K+V)
//!   o_proj       M=1 K=2048 N=2048
//!   mlp_gate_up  M=1 K=2048 N=11264 (fused gate+up)
//!   mlp_down     M=1 K=5632 N=2048
//!   lm_head      M=1 K=2048 N=32000

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;
const weights = vllm.weights;

const TILE: usize = 64;
const ITERS: usize = 100;

const Shape = struct {
    name: []const u8,
    K: usize,
    N: usize,
};

const SHAPES = [_]Shape{
    .{ .name = "qkv_proj    ", .K = 2048, .N = 2560 },
    .{ .name = "o_proj      ", .K = 2048, .N = 2048 },
    .{ .name = "mlp_gate_up ", .K = 2048, .N = 11264 },
    .{ .name = "mlp_down    ", .K = 5632, .N = 2048 },
    .{ .name = "lm_head     ", .K = 2048, .N = 32000 },
};

fn benchShape(alloc: std.mem.Allocator, io: std.Io, shape: Shape) !void {
    const K = shape.K;
    const N = shape.N;

    var a = try Matrix.init(alloc, 1, K);
    defer a.deinit(alloc);
    a.fillRandom(101);

    var b_f32 = try Matrix.init(alloc, K, N);
    defer b_f32.deinit(alloc);
    b_f32.fillRandom(103);

    const b_rm = try alloc.alloc(u16, K * N);
    defer alloc.free(b_rm);
    for (b_f32.data, 0..) |x, i| {
        const bits: u32 = @bitCast(x);
        b_rm[i] = @truncate(bits >> 16);
    }

    const b_raw: weights.RawBF16 = .{ .data = b_rm, .rows = K, .cols = N };
    const b_nt = try weights.repackBF16ToNtileMajor(alloc, b_raw);
    defer alloc.free(b_nt.data);

    var c_rm = try Matrix.init(alloc, 1, N);
    defer c_rm.deinit(alloc);
    var c_nt = try Matrix.init(alloc, 1, N);
    defer c_nt.deinit(alloc);

    // Warmup
    for (0..10) |_| {
        try matmul_bf16.matmulBF16SIMD(&a, b_rm, K, N, &c_rm);
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_nt.data, K, N, &c_nt);
    }

    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMD(&a, b_rm, K, N, &c_rm);
    }
    const ns_rm: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const t1 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_nt.data, K, N, &c_nt);
    }
    const ns_nt: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const us_rm = ns_rm / ITERS / 1000;
    const us_nt = ns_nt / ITERS / 1000;
    const speedup = @as(f64, @floatFromInt(ns_rm)) / @as(f64, @floatFromInt(ns_nt));

    const bytes_per_call: u64 = K * N * 2; // BF16
    const gbs_rm = @as(f64, @floatFromInt(bytes_per_call)) * 1.0e9 / @as(f64, @floatFromInt(ns_rm / ITERS)) / (1024.0 * 1024.0 * 1024.0);
    const gbs_nt = @as(f64, @floatFromInt(bytes_per_call)) * 1.0e9 / @as(f64, @floatFromInt(ns_nt / ITERS)) / (1024.0 * 1024.0 * 1024.0);

    std.debug.print("  {s} K={d:>4} N={d:>5} | row-maj {d:>5} us {d:>5.1} GB/s | n-tile {d:>5} us {d:>5.1} GB/s | speedup {d:.2}x\n", .{ shape.name, K, N, us_rm, gbs_rm, us_nt, gbs_nt, speedup });

    // Numerical equivalence
    var max_abs: f32 = 0;
    for (c_rm.data, c_nt.data) |x, y| {
        const d = @abs(x - y);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-3) {
        std.debug.print("    ! max_abs_diff={e} exceeds 1e-3 — kernel disagreement on this shape\n", .{max_abs});
    }
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== bench_bf16_ntile_shapes — lever-9 layout per-shape, M=1, single-thread ===\n\n", .{});
    std.debug.print("  shape         K     N    | row-major (lever-5)         | N-tile-major (lever-9)       | speedup\n", .{});
    std.debug.print("  ------------- ----- ---- | --------- us       GB/s     | ------------ us       GB/s   | -------\n", .{});

    for (SHAPES) |shape| {
        try benchShape(alloc, io, shape);
    }

    std.debug.print("\nAggregate GB/s is per-call effective single-thread bandwidth (matmul reads K*N*2 bytes of b).\n", .{});
    std.debug.print("Speedup compares matmulBF16SIMD (lever-5 row-major) vs matmulBF16SIMDNtileMajor (lever-9).\n", .{});
}
