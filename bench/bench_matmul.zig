//! Matmul kernel bench — scalar reference vs SIMD outer-product
//! pattern, at several M=N=K shapes representative of transformer
//! workloads.
//!
//! M=N=K=128: small attention sub-block
//! M=N=K=512: per-head Q*K^T at moderate seq
//! M=N=K=1024: full hidden-size matmul on a 1B-class model
//!
//! All numbers are single-thread ReleaseFast on consumer Linux.

const std = @import("std");
const vllm = @import("vllm");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const shapes = [_]usize{ 64, 128, 256, 512, 1024 };
    const ITERS_PER_SHAPE = 5;

    std.debug.print("shape | scalar_ms | simd_ms | speedup | GFLOPS (simd)\n", .{});
    std.debug.print("------+-----------+---------+---------+----------------\n", .{});

    for (shapes) |N| {
        var a = try vllm.Matrix.init(alloc, N, N);
        defer a.deinit(alloc);
        var b = try vllm.Matrix.init(alloc, N, N);
        defer b.deinit(alloc);
        var c_scalar = try vllm.Matrix.init(alloc, N, N);
        defer c_scalar.deinit(alloc);
        var c_simd = try vllm.Matrix.init(alloc, N, N);
        defer c_simd.deinit(alloc);

        a.fillRandom(31);
        b.fillRandom(37);

        // Skip scalar bench at the largest size — N=1024 scalar is ~5s
        // and dominates the bench wall-clock. Run scalar only for N<=512.
        const run_scalar = N <= 512;

        var scalar_trials = [_]u64{0} ** ITERS_PER_SHAPE;
        if (run_scalar) {
            for (0..ITERS_PER_SHAPE) |t| {
                const t0 = std.Io.Clock.awake.now(io);
                try vllm.matmul.matmulScalar(&a, &b, &c_scalar);
                const t1 = std.Io.Clock.awake.now(io);
                scalar_trials[t] = @intCast(t0.durationTo(t1).nanoseconds);
            }
        }

        var simd_trials = [_]u64{0} ** ITERS_PER_SHAPE;
        for (0..ITERS_PER_SHAPE) |t| {
            const t0 = std.Io.Clock.awake.now(io);
            try vllm.matmul.matmulSIMD(&a, &b, &c_simd);
            const t1 = std.Io.Clock.awake.now(io);
            simd_trials[t] = @intCast(t0.durationTo(t1).nanoseconds);
        }

        std.sort.pdq(u64, &scalar_trials, {}, std.sort.asc(u64));
        std.sort.pdq(u64, &simd_trials, {}, std.sort.asc(u64));

        const scalar_ms = scalar_trials[ITERS_PER_SHAPE / 2] / 1_000_000;
        const simd_ms = simd_trials[ITERS_PER_SHAPE / 2] / 1_000_000;
        const simd_ns = simd_trials[ITERS_PER_SHAPE / 2];

        // Speedup: scalar/simd
        const speedup: f64 = if (simd_ns > 0 and run_scalar)
            @as(f64, @floatFromInt(scalar_trials[ITERS_PER_SHAPE / 2])) / @as(f64, @floatFromInt(simd_ns))
        else
            0.0;

        const flops: u64 = 2 * N * N * N;
        const gflops: f64 = @as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(simd_ns));

        if (run_scalar) {
            std.debug.print("{d:>4}  |  {d:>8}  | {d:>6}  | {d:>5.2}x  | {d:>6.2}\n", .{ N, scalar_ms, simd_ms, speedup, gflops });
        } else {
            std.debug.print("{d:>4}  |  (skipped) | {d:>6}  |   --    | {d:>6.2}\n", .{ N, simd_ms, gflops });
        }
    }
}
