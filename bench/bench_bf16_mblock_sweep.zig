//! M_BLOCK parameter sweep for the lever-12 comptime-blocked GEMM
//! kernel across batched M values {8, 16, 32, 64} at K=2048 N=2048.
//!
//! Ice Lake AVX-512 has 32 ZMM registers. With N_TILE=64 (4 ZMM acc
//! per row of M-block), the maximum M_BLOCK that fits without spill
//! is 4 (4*4 acc + 4 b + 4 a + 1 shift = 25). M_BLOCK=8 would need
//! 8*4 acc + 4 b + 8 a + 1 = 45, exceeding the register file.
//!
//! Sweep M_BLOCK in {1, 2, 4} per M shape, single-thread.
//! Pre-reg exp-1780164607-908798779.

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;
const weights = vllm.weights;

const K: usize = 2048;
const N: usize = 2048;
const TILE: usize = 64;
const ITERS: usize = 30;

const M_VALUES = [_]usize{ 8, 16, 32, 64 };
const M_BLOCKS = [_]usize{ 1, 2, 4 };

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== M_BLOCK sweep for matmulBF16SIMDBlocked(M_BLOCK, 64, 64) ===\n", .{});
    std.debug.print("K={d} N={d}, single-thread, ReleaseFast\n\n", .{ K, N });

    std.debug.print("  M       | M_BLOCK=1 us | M_BLOCK=2 us | M_BLOCK=4 us | best M_BLOCK | best vs M=1 speedup\n", .{});
    std.debug.print("  ------- | ------------ | ------------ | ------------ | ------------ | -------------------\n", .{});

    for (M_VALUES) |M| {
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

        var times: [3]u64 = .{ 0, 0, 0 };

        inline for (M_BLOCKS, 0..) |mb, idx| {
            for (0..5) |_| {
                try matmul_bf16.matmulBF16SIMDBlocked(mb, 64, 64, &a, b_nt.data, K, N, &c);
            }
            const t0 = std.Io.Clock.awake.now(io);
            for (0..ITERS) |_| {
                try matmul_bf16.matmulBF16SIMDBlocked(mb, 64, 64, &a, b_nt.data, K, N, &c);
            }
            times[idx] = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        }

        var best_idx: usize = 0;
        for (1..3) |i| if (times[i] < times[best_idx]) {
            best_idx = i;
        };
        const speedup = @as(f64, @floatFromInt(times[0])) / @as(f64, @floatFromInt(times[best_idx]));

        const us0 = times[0] / ITERS / 1_000;
        const us1 = times[1] / ITERS / 1_000;
        const us2 = times[2] / ITERS / 1_000;

        std.debug.print("  M={d:>5} | {d:>10}   | {d:>10}   | {d:>10}   |       {d:>2}     | {d:.2}x\n", .{ M, us0, us1, us2, M_BLOCKS[best_idx], speedup });
    }

    std.debug.print("\nIce Lake 32-ZMM file limits M_BLOCK to 4 at N_TILE=64.\n", .{});
    std.debug.print("Speedup column is best vs M_BLOCK=1 (no M-block reuse).\n", .{});
}
