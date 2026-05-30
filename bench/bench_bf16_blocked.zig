//! Lever-12 microbench — comptime-blocked GEMM substrate primitive.
//!
//! Compares the new matmulBF16SIMDBlocked kernel (M-block + N-tile +
//! K-tile, comptime-parameterized) against the lever-9 ntile-major
//! kernel at three shapes:
//!
//!   M=1   K=2048 N=2048   — decode regime (no M to amortize across)
//!   M=8   K=2048 N=2048   — small-batch prefill
//!   M=64  K=2048 N=2048   — large-batch prefill (where M-blocking pays off)
//!
//! Hypothesis: lever-12 lossless at M=1, wins 2x+ at M=64 because each b
//! cache-line load is consumed M_BLOCK=4 times before the next load.
//!
//! Pre-reg exp-1780157294-563607188.

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;
const weights = vllm.weights;
const Pool = vllm.Pool;

const TILE: usize = 64;
const ITERS: usize = 30;

const Shape = struct { M: usize, K: usize, N: usize, label: []const u8 };

const SHAPES = [_]Shape{
    .{ .M = 1, .K = 2048, .N = 2048, .label = "decode    M=1 " },
    .{ .M = 8, .K = 2048, .N = 2048, .label = "small     M=8 " },
    .{ .M = 64, .K = 2048, .N = 2048, .label = "prefill   M=64" },
};

fn benchShape(alloc: std.mem.Allocator, io: std.Io, shape: Shape) !void {
    const M = shape.M;
    const K = shape.K;
    const N = shape.N;

    var a = try Matrix.init(alloc, M, K);
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

    var c_ntile = try Matrix.init(alloc, M, N);
    defer c_ntile.deinit(alloc);
    var c_blk1 = try Matrix.init(alloc, M, N);
    defer c_blk1.deinit(alloc);
    var c_blk4 = try Matrix.init(alloc, M, N);
    defer c_blk4.deinit(alloc);
    var c_pool = try Matrix.init(alloc, M, N);
    defer c_pool.deinit(alloc);

    var pool = try Pool.init(alloc, 4);
    defer pool.deinit();

    // Warmup
    for (0..5) |_| {
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_nt.data, K, N, &c_ntile);
        try matmul_bf16.matmulBF16SIMDBlocked(1, 64, 64, &a, b_nt.data, K, N, &c_blk1);
        try matmul_bf16.matmulBF16SIMDBlocked(4, 64, 64, &a, b_nt.data, K, N, &c_blk4);
        try pool.matmulBF16BlockedAuto(&a, b_nt.data, K, N, &c_pool, 4);
    }

    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_nt.data, K, N, &c_ntile);
    }
    const ns_nt: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const t1 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMDBlocked(1, 64, 64, &a, b_nt.data, K, N, &c_blk1);
    }
    const ns_b1: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const t2 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMDBlocked(4, 64, 64, &a, b_nt.data, K, N, &c_blk4);
    }
    const ns_b4: u64 = @intCast(t2.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const t3 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try pool.matmulBF16BlockedAuto(&a, b_nt.data, K, N, &c_pool, 4);
    }
    const ns_pool: u64 = @intCast(t3.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const us_nt = ns_nt / ITERS / 1_000;
    const us_b1 = ns_b1 / ITERS / 1_000;
    const us_b4 = ns_b4 / ITERS / 1_000;
    const us_pool = ns_pool / ITERS / 1_000;

    const speedup_b1 = @as(f64, @floatFromInt(ns_nt)) / @as(f64, @floatFromInt(ns_b1));
    const speedup_b4 = @as(f64, @floatFromInt(ns_nt)) / @as(f64, @floatFromInt(ns_b4));
    const speedup_pool = @as(f64, @floatFromInt(ns_nt)) / @as(f64, @floatFromInt(ns_pool));

    // Per-row throughput (rows-per-second).
    const rows_per_sec_nt: u64 = @intFromFloat(@as(f64, @floatFromInt(M * ITERS)) * 1.0e9 / @as(f64, @floatFromInt(ns_nt)));
    const rows_per_sec_pool: u64 = @intFromFloat(@as(f64, @floatFromInt(M * ITERS)) * 1.0e9 / @as(f64, @floatFromInt(ns_pool)));

    std.debug.print(
        "  {s} K={d:>4} N={d:>4} | ntile-1T {d:>6} us | blk1-1T {d:>6} ({d:.2}x) | blk4-1T {d:>6} ({d:.2}x) | blk4-4T {d:>6} ({d:.2}x) | rows/s nt-1T {d:>7} vs blk4-4T {d:>7}\n",
        .{ shape.label, K, N, us_nt, us_b1, speedup_b1, us_b4, speedup_b4, us_pool, speedup_pool, rows_per_sec_nt, rows_per_sec_pool },
    );
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== bench_bf16_blocked — lever-12 substrate primitive, single-thread ===\n\n", .{});
    std.debug.print("Compares matmulBF16SIMDBlocked(M_BLOCK, 64, 64) vs lever-9 ntile-major.\n", .{});
    std.debug.print("Hypothesis: lossless at M=1, wins 2x+ at M=64 prefill via M-blocking.\n\n", .{});

    for (SHAPES) |shape| {
        try benchShape(alloc, io, shape);
    }

    std.debug.print("\nM_BLOCK=4 means each b cache-line load is consumed 4 times (4 rows of accumulator)\n", .{});
    std.debug.print("before the next b load — b-traffic per output column drops by 4x in the M-blocked path.\n", .{});
}
