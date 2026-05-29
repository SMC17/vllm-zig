//! Pre-conversion-at-load cblas matmul vs matmulBF16SIMD.
//!
//! Real serving pattern: weights are loaded ONCE at Model init time,
//! so BF16→F32 conversion is amortized across all decode steps. This
//! bench mimics that: B is converted once outside the timing loop,
//! both paths see the SAME memory layout otherwise.
//!
//! Compares:
//!   - matmulBF16SIMD (reads BF16 storage, widens per-row in hot loop)
//!   - cblas_sgemm on pre-converted F32 storage (Accelerate AMX path)
//!
//! The 2× memory cost (BF16 kept for compat + F32 cache) is the real
//! trade. Per-matmul speedup is the lever this bench measures.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("vllm").tensor.Matrix;
const matmul_bf16 = @import("vllm").matmul_bf16;
const bf16_mod = @import("vllm").bf16;

const CBLAS_ORDER = enum(c_int) { row_major = 101, col_major = 102 };
const CBLAS_TRANSPOSE = enum(c_int) { no_trans = 111, trans = 112 };

extern "c" fn cblas_sgemm(
    order: CBLAS_ORDER,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) void;

const Shape = struct { name: []const u8, K: usize, N: usize };
const SHAPES = [_]Shape{
    .{ .name = "qkv-proj  K=2048 N=2048 ", .K = 2048, .N = 2048 },
    .{ .name = "mlp       K=2048 N=5632 ", .K = 2048, .N = 5632 },
    .{ .name = "lm_head   K=2048 N=32000", .K = 2048, .N = 32000 },
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

    std.debug.print("=== Pre-converted cblas vs in-process SIMD (BF16 weights) ===\n", .{});
    std.debug.print("build target: {s}\n\n", .{@tagName(builtin.os.tag)});
    std.debug.print("shape                   | simd_us | cblas_us | speedup | max_diff\n", .{});
    std.debug.print("------------------------+---------+----------+---------+---------\n", .{});

    for (SHAPES) |s| {
        var a = try Matrix.init(alloc, 1, s.K);
        a.fillRandom(101);

        const b_bf16 = try alloc.alloc(u16, s.K * s.N);
        for (b_bf16) |*x| x.* = bf16_mod.f32ToBf16(rng.float(f32));

        // Pre-convert once (simulates load-time conversion)
        const b_f32 = try alloc.alloc(f32, s.K * s.N);
        for (b_bf16, 0..) |w, idx| b_f32[idx] = @bitCast(@as(u32, w) << 16);

        var c_simd = try Matrix.init(alloc, 1, s.N);
        const c_cblas = try Matrix.init(alloc, 1, s.N);

        // Warmup both paths
        try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_simd);
        cblas_sgemm(.row_major, .no_trans, .no_trans, 1, @intCast(s.N), @intCast(s.K), 1.0, a.data.ptr, @intCast(s.K), b_f32.ptr, @intCast(s.N), 0.0, c_cblas.data.ptr, @intCast(s.N));

        // Time SIMD (BF16 storage)
        const t0 = std.Io.Clock.awake.now(io);
        var iter: usize = 0;
        while (iter < ITERS) : (iter += 1) {
            try matmul_bf16.matmulBF16SIMD(&a, b_bf16, s.K, s.N, &c_simd);
        }
        const ns_simd: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        // Time cblas (F32 storage, NO per-call convert)
        const t1 = std.Io.Clock.awake.now(io);
        iter = 0;
        while (iter < ITERS) : (iter += 1) {
            cblas_sgemm(.row_major, .no_trans, .no_trans, 1, @intCast(s.N), @intCast(s.K), 1.0, a.data.ptr, @intCast(s.K), b_f32.ptr, @intCast(s.N), 0.0, c_cblas.data.ptr, @intCast(s.N));
        }
        const ns_cblas: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        // Correctness: BF16 lossy; tolerance per element ~1e-3 for random N=2048..32000 elem dot products
        var max_diff: f32 = 0;
        for (c_simd.data, c_cblas.data) |x, y| {
            const d = @abs(x - y);
            if (d > max_diff) max_diff = d;
        }

        const us_simd = ns_simd / ITERS / 1000;
        const us_cblas = ns_cblas / ITERS / 1000;
        const speedup = @as(f64, @floatFromInt(ns_simd)) / @as(f64, @floatFromInt(ns_cblas));
        std.debug.print("{s} | {d:7} | {d:8} | {d:6.2}x | {e}\n", .{ s.name, us_simd, us_cblas, speedup, max_diff });
    }
}
