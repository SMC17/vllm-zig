//! K_TILE parameter sweep for the lever-12 comptime-blocked GEMM
//! kernel at M=64 K=2048 N=2048 (prefill regime).
//!
//! All K_TILE values that the inner-loop schedule visits — 32, 64,
//! 128, 256, 512, 1024, 2048 — are instantiated as separate
//! comptime specializations of matmulBF16SIMDBlocked. The compiler
//! generates a distinct inner loop for each.
//!
//! At K=2048, the active b-tile for ONE n-tile across K_TILE rows is
//! K_TILE * 64 * 2 bytes BF16. L2 (Ice Lake-U) is 512 KB:
//!   K_TILE=64    →   8 KB  (L1-resident)
//!   K_TILE=128   →  16 KB
//!   K_TILE=512   →  64 KB
//!   K_TILE=2048  → 256 KB  (full K, fits L2 but not with multiple n-tiles active)
//!
//! Pre-reg exp-1780164427-830843575.

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;
const weights = vllm.weights;

const K: usize = 2048;
const TILE: usize = 64;
const ITERS: usize = 20;

const K_TILES = [_]usize{ 32, 64, 128, 256, 512, 1024, 2048 };

const Shape = struct { M: usize, N: usize, label: []const u8 };

const SHAPES = [_]Shape{
    .{ .M = 64, .N = 2048, .label = "M=64  N= 2048 (prefill)" },
    .{ .M = 16, .N = 32000, .label = "M=16  N=32000 (lm_head prefill)" },
};

fn sweepShape(alloc: std.mem.Allocator, io: std.Io, shape: Shape) !void {
    const M = shape.M;
    const N = shape.N;

    std.debug.print("\n=== {s} K={d} ===\n", .{ shape.label, K });

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

    var c = try Matrix.init(alloc, M, N);
    defer c.deinit(alloc);

    std.debug.print("  K_TILE | b-tile KB | med us/call | rows/sec\n", .{});
    std.debug.print("  ------ | --------- | ----------- | --------\n", .{});

    var best_ns: u64 = std.math.maxInt(u64);
    var best_ktile: usize = 0;

    inline for (K_TILES) |kt| {
        for (0..3) |_| {
            try matmul_bf16.matmulBF16SIMDBlocked(4, 64, kt, &a, b_nt.data, K, N, &c);
        }
        const t0 = std.Io.Clock.awake.now(io);
        for (0..ITERS) |_| {
            try matmul_bf16.matmulBF16SIMDBlocked(4, 64, kt, &a, b_nt.data, K, N, &c);
        }
        const ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        const us_per_call = ns / ITERS / 1_000;
        const b_tile_kb: usize = (kt * TILE * 2) / 1024;
        const rows_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(M * ITERS)) * 1.0e9 / @as(f64, @floatFromInt(ns)));

        if (ns < best_ns) {
            best_ns = ns;
            best_ktile = kt;
        }

        std.debug.print("  K={d:>5} | {d:>5}     | {d:>9}   | {d:>7}\n", .{ kt, b_tile_kb, us_per_call, rows_per_sec });
    }

    std.debug.print("  best: K_TILE={d} at {d} us/call\n", .{ best_ktile, best_ns / ITERS / 1_000 });
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== K_TILE sweep for matmulBF16SIMDBlocked(4, 64, K_TILE) ===\n", .{});
    std.debug.print("single-thread, ReleaseFast\n", .{});

    for (SHAPES) |shape| {
        try sweepShape(alloc, io, shape);
    }
}
