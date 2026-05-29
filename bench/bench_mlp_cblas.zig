//! MLP shape A/B: matmul.matmulSIMD (vllm-zig kernel, F32 weights) vs
//! cblas_sgemm (OpenBLAS on Linux, Accelerate on macOS) at the TinyLlama
//! MLP shape M=1 K=2048 N=5632 — the 75% TAM region per bench_tinyllama_breakdown.
//!
//! Run: zig build bench-mlp-cblas -Doptimize=ReleaseFast

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.tensor.Matrix;
const matmul = vllm.matmul;

const M: usize = 1;
const K: usize = 2048;
const N: usize = 5632;
const ITERS: usize = 100;
const WARMUP: usize = 10;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
extern "c" fn cblas_sgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) void;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var a = try Matrix.init(alloc, M, K);
    defer a.deinit(alloc);
    var b = try Matrix.init(alloc, K, N);
    defer b.deinit(alloc);
    var c_ours = try Matrix.init(alloc, M, N);
    defer c_ours.deinit(alloc);
    var c_ours16 = try Matrix.init(alloc, M, N);
    defer c_ours16.deinit(alloc);
    var c_blas = try Matrix.init(alloc, M, N);
    defer c_blas.deinit(alloc);

    for (a.data) |*x| x.* = rng.float(f32);
    for (b.data) |*x| x.* = rng.float(f32);

    // Warmup matmulSIMD (8-lane)
    for (0..WARMUP) |_| try matmul.matmulSIMD(&a, &b, &c_ours);

    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| try matmul.matmulSIMD(&a, &b, &c_ours);
    const t1 = std.Io.Clock.awake.now(io);

    // Warmup matmulSIMD16 (16-lane = AVX-512 on Ice Lake)
    for (0..WARMUP) |_| try matmul.matmulSIMD16(&a, &b, &c_ours16);
    const t_s16_0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| try matmul.matmulSIMD16(&a, &b, &c_ours16);
    const t_s16_1 = std.Io.Clock.awake.now(io);
    const ns_simd16: u64 = @intCast(t_s16_0.durationTo(t_s16_1).nanoseconds);

    // cblas warmup + timed
    for (0..WARMUP) |_| {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            @intCast(M), @intCast(N), @intCast(K),
            1.0, a.data.ptr, @intCast(K),
            b.data.ptr, @intCast(N),
            0.0, c_blas.data.ptr, @intCast(N),
        );
    }

    const t2 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            @intCast(M), @intCast(N), @intCast(K),
            1.0, a.data.ptr, @intCast(K),
            b.data.ptr, @intCast(N),
            0.0, c_blas.data.ptr, @intCast(N),
        );
    }
    const t3 = std.Io.Clock.awake.now(io);

    const ns_ours: u64 = @intCast(t0.durationTo(t1).nanoseconds);
    const ns_blas: u64 = @intCast(t2.durationTo(t3).nanoseconds);
    const us_ours_per = ns_ours / ITERS / 1000;
    const us_blas_per = ns_blas / ITERS / 1000;
    const speedup_x100 = (ns_ours * 100) / ns_blas;

    // Correctness check (loose tolerance for FP non-determinism)
    var max_err: f32 = 0;
    for (c_ours.data, c_blas.data) |o, bl| {
        const e = @abs(o - bl);
        if (e > max_err) max_err = e;
    }

    const us_simd16_per = ns_simd16 / ITERS / 1000;
    var max_err16: f32 = 0;
    for (c_ours.data, c_ours16.data) |o8, o16| {
        const e = @abs(o8 - o16);
        if (e > max_err16) max_err16 = e;
    }

    std.debug.print("=== MLP shape M={d} K={d} N={d} ===\n", .{ M, K, N });
    std.debug.print("matmul.matmulSIMD  (8-lane) us/call: {d}\n", .{us_ours_per});
    std.debug.print("matmul.matmulSIMD16(16-lane) us/call: {d}  (vs 8-lane: {d:.2}x)\n", .{ us_simd16_per, @as(f64, @floatFromInt(ns_ours)) / @as(f64, @floatFromInt(ns_simd16)) });
    std.debug.print("cblas_sgemm        us/call: {d}\n", .{us_blas_per});
    std.debug.print("speedup_x100 (8-lane vs cblas): {d}  ({d:.2}x)\n", .{ speedup_x100, @as(f64, @floatFromInt(ns_ours)) / @as(f64, @floatFromInt(ns_blas)) });
    std.debug.print("max element error 8 vs cblas: {d:.6}\n", .{max_err});
    std.debug.print("max element error 16 vs 8   : {d:.6}\n", .{max_err16});
}
