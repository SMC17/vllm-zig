//! Microbench — lever-9 N-tile-major BF16 weight layout vs lever-5
//! row-major register-blocked baseline.
//!
//! Layout transformation at load time:
//!   row-major: b[k * N + j]                                  (current)
//!   N-tile-major: b[n_tile_outer * K * TILE + k * TILE + i] (lever-9)
//!
//! Stride between k iterations on b:
//!   row-major: N * 2 bytes = 22.5 KB (page-crosses → TLB pressure)
//!   N-tile-major: TILE * 2 = 128 bytes (2 cache lines → HW streamer wins)
//!
//! Shape: M=1, K=2048, N=11264 (TinyLlama fused mlp_gate_up).
//! Pre-reg exp-1780110463-209781800.

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;

const M: usize = 1;
const K: usize = 2048;
const N: usize = 11264; // fused gate+up dim
const TILE: usize = 64;
const LANE: usize = 16;
const ITERS: usize = 200;

fn repackToNtileMajor(alloc: std.mem.Allocator, b_rm: []const u16) ![]u16 {
    std.debug.assert(N % TILE == 0);
    const out = try alloc.alloc(u16, K * N);
    const n_tiles = N / TILE;
    var nt: usize = 0;
    while (nt < n_tiles) : (nt += 1) {
        var k: usize = 0;
        while (k < K) : (k += 1) {
            var i: usize = 0;
            while (i < TILE) : (i += 1) {
                out[nt * K * TILE + k * TILE + i] = b_rm[k * N + nt * TILE + i];
            }
        }
    }
    return out;
}

/// Lever-9 kernel: 4-ZMM register block, walks K with stride-1 b access
/// within each 64-col tile.
fn matmulNtileMajor(a: *const Matrix, b: []const u16, c: *Matrix) void {
    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);

    const n_tiles = N / TILE;
    const a_row = a.data[0..K];
    const c_row = c.data[0..N];
    const shift: @Vector(16, u5) = @splat(16);

    var nt: usize = 0;
    while (nt < n_tiles) : (nt += 1) {
        var acc0: Vec_f32 = @splat(0.0);
        var acc1: Vec_f32 = @splat(0.0);
        var acc2: Vec_f32 = @splat(0.0);
        var acc3: Vec_f32 = @splat(0.0);

        const tile_base = nt * K * TILE;

        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec_f32 = @splat(a_row[k]);
            const b_row = b[tile_base + k * TILE ..][0..TILE];

            const b0_u16: Vec_u16 = b_row[0..LANE].*;
            const b1_u16: Vec_u16 = b_row[LANE..][0..LANE].*;
            const b2_u16: Vec_u16 = b_row[2 * LANE ..][0..LANE].*;
            const b3_u16: Vec_u16 = b_row[3 * LANE ..][0..LANE].*;

            const b0_u32: Vec_u32 = b0_u16;
            const b1_u32: Vec_u32 = b1_u16;
            const b2_u32: Vec_u32 = b2_u16;
            const b3_u32: Vec_u32 = b3_u16;

            const b0_f32: Vec_f32 = @bitCast(b0_u32 << shift);
            const b1_f32: Vec_f32 = @bitCast(b1_u32 << shift);
            const b2_f32: Vec_f32 = @bitCast(b2_u32 << shift);
            const b3_f32: Vec_f32 = @bitCast(b3_u32 << shift);

            acc0 = @mulAdd(Vec_f32, a_scalar, b0_f32, acc0);
            acc1 = @mulAdd(Vec_f32, a_scalar, b1_f32, acc1);
            acc2 = @mulAdd(Vec_f32, a_scalar, b2_f32, acc2);
            acc3 = @mulAdd(Vec_f32, a_scalar, b3_f32, acc3);
        }

        const c_base = nt * TILE;
        c_row[c_base..][0..LANE].* = acc0;
        c_row[c_base + LANE ..][0..LANE].* = acc1;
        c_row[c_base + 2 * LANE ..][0..LANE].* = acc2;
        c_row[c_base + 3 * LANE ..][0..LANE].* = acc3;
    }
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== bench_bf16_ntile_major — lever-9 layout vs lever-5 row-major ===\n", .{});
    std.debug.print("shape: M={d} K={d} N={d}, ITERS={d}\n", .{ M, K, N, ITERS });

    var a = try Matrix.init(alloc, M, K);
    defer a.deinit(alloc);
    a.fillRandom(101);

    var b_f32 = try Matrix.init(alloc, K, N);
    defer b_f32.deinit(alloc);
    b_f32.fillRandom(103);

    // Quantise to BF16
    const b_rm = try alloc.alloc(u16, K * N);
    defer alloc.free(b_rm);
    for (b_f32.data, 0..) |x, i| {
        const bits: u32 = @bitCast(x);
        b_rm[i] = @truncate(bits >> 16);
    }

    const b_nt = try repackToNtileMajor(alloc, b_rm);
    defer alloc.free(b_nt);

    var c_rm = try Matrix.init(alloc, M, N);
    defer c_rm.deinit(alloc);
    var c_nt = try Matrix.init(alloc, M, N);
    defer c_nt.deinit(alloc);

    // Warmup
    for (0..10) |_| {
        try matmul_bf16.matmulBF16SIMD(&a, b_rm, K, N, &c_rm);
        matmulNtileMajor(&a, b_nt, &c_nt);
    }

    // Equivalence check
    var max_abs: f32 = 0;
    for (c_rm.data, c_nt.data) |x, y| {
        const d = @abs(x - y);
        if (d > max_abs) max_abs = d;
    }
    std.debug.print("\nNumerical equivalence (vs matmulBF16SIMD): max_abs={e}\n", .{max_abs});
    if (max_abs > 1e-3) {
        std.debug.print("  ! tolerance exceeded — likely accumulation-order divergence (FMA vs not)\n", .{});
    }

    // Lever-5 row-major bench
    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMD(&a, b_rm, K, N, &c_rm);
    }
    const ns_rm: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // Lever-9 N-tile-major bench
    const t1 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        matmulNtileMajor(&a, b_nt, &c_nt);
    }
    const ns_nt: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const us_rm = ns_rm / ITERS / 1000;
    const us_nt = ns_nt / ITERS / 1000;

    std.debug.print("\nthroughput (single-thread, ReleaseFast):\n", .{});
    std.debug.print("  lever-5 row-major  | per-call us = {d}  | total ms = {d}\n", .{ us_rm, ns_rm / 1_000_000 });
    std.debug.print("  lever-9 ntile-maj  | per-call us = {d}  | total ms = {d}\n", .{ us_nt, ns_nt / 1_000_000 });
    const speedup = @as(f64, @floatFromInt(ns_rm)) / @as(f64, @floatFromInt(ns_nt));
    std.debug.print("  speedup            | {d:.2}x\n", .{speedup});

    // Anti-DCE
    std.debug.print("\n(cksum c_rm[0]={d:.4} c_nt[0]={d:.4})\n", .{ c_rm.data[0], c_nt.data[0] });
}
