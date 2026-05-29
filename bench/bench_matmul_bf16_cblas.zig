//! Head-to-head: vllm-zig matmulBF16SIMD vs Apple Accelerate cblas_sgemm
//! on the lm_head shape (M=1, K=2048, N=32000).
//!
//! macOS-only. Compiled with `-framework Accelerate` linked.
//!
//! cblas_sgemm operates on F32, so we convert B from BF16 storage to a
//! scratch F32 buffer once outside the timing loop (in the real serving
//! path the convert could also be cached). Both timed paths see the
//! same numerical weights.
//!
//! Hypothesis: cblas wins on this rectangular shape by enough margin
//! that the BF16→F32 convert cost is amortized — the right lever for
//! the vllm-zig lm_head op on Apple Silicon.

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

const M_DIM: comptime_int = 1;
const K_DIM: comptime_int = 2048;
const N_DIM: comptime_int = 32_000;
const ITERS: usize = 50;
const SEED: u64 = 42;

pub fn main(init: std.process.Init) !void {
    if (builtin.os.tag != .macos) {
        std.debug.print("This bench requires macOS (Accelerate framework).\n", .{});
        std.process.exit(2);
    }

    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    var prng: std.Random.DefaultPrng = .init(SEED);
    const rng = prng.random();

    var a = try Matrix.init(alloc, M_DIM, K_DIM);
    a.fillRandom(101);

    // B as BF16 (vllm-zig's storage shape: row-major u16)
    const b_bf16 = try alloc.alloc(u16, K_DIM * N_DIM);
    for (b_bf16) |*x| x.* = bf16_mod.f32ToBf16(rng.float(f32));

    // B converted to F32 once for the cblas path
    const b_f32 = try alloc.alloc(f32, K_DIM * N_DIM);
    for (b_bf16, 0..) |w, idx| b_f32[idx] = @bitCast(@as(u32, w) << 16);

    var c_zig = try Matrix.init(alloc, M_DIM, N_DIM);
    const c_cblas = try alloc.alloc(f32, M_DIM * N_DIM);

    // Warmup
    try matmul_bf16.matmulBF16SIMD(&a, b_bf16, K_DIM, N_DIM, &c_zig);
    cblas_sgemm(.row_major, .no_trans, .no_trans, M_DIM, N_DIM, K_DIM, 1.0, a.data.ptr, K_DIM, b_f32.ptr, N_DIM, 0.0, c_cblas.ptr, N_DIM);

    // Time vllm-zig matmulBF16SIMD
    const t0 = std.Io.Clock.awake.now(io);
    var iter: usize = 0;
    while (iter < ITERS) : (iter += 1) {
        try matmul_bf16.matmulBF16SIMD(&a, b_bf16, K_DIM, N_DIM, &c_zig);
    }
    const ns_zig: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // Time cblas_sgemm
    const t1 = std.Io.Clock.awake.now(io);
    iter = 0;
    while (iter < ITERS) : (iter += 1) {
        cblas_sgemm(.row_major, .no_trans, .no_trans, M_DIM, N_DIM, K_DIM, 1.0, a.data.ptr, K_DIM, b_f32.ptr, N_DIM, 0.0, c_cblas.ptr, N_DIM);
    }
    const ns_cblas: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // Correctness: max abs diff between the two outputs
    var max_diff: f32 = 0;
    for (c_zig.data, c_cblas) |x, y| {
        const d = @abs(x - y);
        if (d > max_diff) max_diff = d;
    }

    const us_zig = ns_zig / ITERS / 1000;
    const us_cblas = ns_cblas / ITERS / 1000;
    const ratio = @as(f64, @floatFromInt(ns_zig)) / @as(f64, @floatFromInt(ns_cblas));

    std.debug.print("=== vllm-zig matmulBF16SIMD vs Accelerate cblas_sgemm (lm_head shape) ===\n", .{});
    std.debug.print("config: M={d} K={d} N={d} iters={d}\n", .{ M_DIM, K_DIM, N_DIM, ITERS });
    std.debug.print("vllm-zig matmulBF16SIMD\n", .{});
    std.debug.print("  ns_per_matmul = {d}\n", .{ns_zig / ITERS});
    std.debug.print("  us_per_matmul = {d}\n", .{us_zig});
    std.debug.print("cblas_sgemm (Accelerate AMX path)\n", .{});
    std.debug.print("  ns_per_matmul = {d}\n", .{ns_cblas / ITERS});
    std.debug.print("  us_per_matmul = {d}\n", .{us_cblas});
    std.debug.print("ratio_cblas_vs_zig_simd = {d:.3}\n", .{ratio});
    std.debug.print("max_abs_diff_f32 = {e}\n", .{max_diff});
}
