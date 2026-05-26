//! Cache-tiled vs non-tiled matmul — wall time at several N=M=K shapes.
//!
//! N≤256: tiled falls through to non-tiled; expected equal.
//! N=512+: tiled should win as working set exceeds L2.
//! N=1024+: tiled should win by 1.5-2.5x as untiled blasts L2/L3.

const std = @import("std");
const vllm = @import("vllm");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const shapes = [_]usize{ 256, 512, 768, 1024, 1536, 2048 };
    const ITERS = 5;

    std.debug.print("shape | non-tiled ms | tiled ms | speedup | GFLOPS (tiled)\n", .{});
    std.debug.print("------+--------------+----------+---------+----------------\n", .{});

    for (shapes) |N| {
        var a = try vllm.Matrix.init(alloc, N, N);
        defer a.deinit(alloc);
        var b = try vllm.Matrix.init(alloc, N, N);
        defer b.deinit(alloc);
        var c1 = try vllm.Matrix.init(alloc, N, N);
        defer c1.deinit(alloc);
        var c2 = try vllm.Matrix.init(alloc, N, N);
        defer c2.deinit(alloc);

        a.fillRandom(@as(u64, @intCast(N)) * 31);
        b.fillRandom(@as(u64, @intCast(N)) * 37);

        var t_simd_min: u64 = std.math.maxInt(u64);
        var t_tiled_min: u64 = std.math.maxInt(u64);

        for (0..ITERS) |_| {
            const t0 = std.Io.Clock.awake.now(io);
            try vllm.matmul.matmulSIMD(&a, &b, &c1);
            const t1 = std.Io.Clock.awake.now(io);
            const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
            if (ns < t_simd_min) t_simd_min = ns;
        }
        for (0..ITERS) |_| {
            const t0 = std.Io.Clock.awake.now(io);
            try vllm.matmul.matmulSIMDTiled(&a, &b, &c2);
            const t1 = std.Io.Clock.awake.now(io);
            const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
            if (ns < t_tiled_min) t_tiled_min = ns;
        }

        // Correctness check
        std.debug.assert(c1.approxEq(&c2, 1e-2));

        const simd_ms = t_simd_min / 1_000_000;
        const tiled_ms = t_tiled_min / 1_000_000;
        const speedup = if (t_tiled_min > 0)
            @as(f64, @floatFromInt(t_simd_min)) / @as(f64, @floatFromInt(t_tiled_min))
        else
            0.0;
        const flops = 2 * N * N * N;
        const gflops_tiled = if (t_tiled_min > 0)
            @as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(t_tiled_min))
        else
            0.0;
        std.debug.print(
            "{d:>5} | {d:>12} | {d:>8} | {d:.2}x  | {d:.2}\n",
            .{ N, simd_ms, tiled_ms, speedup, gflops_tiled },
        );
    }
}
