//! Persistent thread pool vs ad-hoc spawn — wall-time per matmul call.
//!
//! Submits N=200 matmul calls back-to-back through:
//!   (1) matmulSIMD (single-thread baseline)
//!   (2) matmulSIMDAuto (v0.0.4 ad-hoc spawn)
//!   (3) Pool.matmulAuto (v0.0.5 persistent pool)
//!
//! At decode-shape (M=1, K=2048, N=2048 — TinyLlama-class q/k/v/o projs),
//! the v0.0.4 path spawns 8 threads per call; v0.0.5 reuses 8 workers
//! across all 200 calls.

const std = @import("std");
const vllm = @import("vllm");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const N_CALLS: usize = 200;
    const N_THREADS: usize = 8;

    // Decode-shape: M=1, K=2048, N=2048 (one of TinyLlama's projections)
    const M: usize = 1;
    const K: usize = 2048;
    const N: usize = 2048;

    std.debug.print("Pool vs ad-hoc spawn — {d} matmul calls at M={d} K={d} N={d}, T={d}\n", .{ N_CALLS, M, K, N, N_THREADS });
    std.debug.print("---\n", .{});

    var a = try vllm.Matrix.init(alloc, M, K);
    defer a.deinit(alloc);
    var b = try vllm.Matrix.init(alloc, K, N);
    defer b.deinit(alloc);
    var c = try vllm.Matrix.init(alloc, M, N);
    defer c.deinit(alloc);
    a.fillRandom(41);
    b.fillRandom(43);

    // (1) Single-thread baseline
    {
        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < N_CALLS) : (i += 1) {
            try vllm.matmul.matmulSIMD(&a, &b, &c);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("(1) matmulSIMD       (1 thread)       total={d:>8} ms,  per-call={d:>6} us\n", .{ ns / 1_000_000, ns / N_CALLS / 1_000 });
    }

    // (2) Ad-hoc spawn (v0.0.4)
    {
        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < N_CALLS) : (i += 1) {
            try vllm.matmul.matmulSIMDAuto(&a, &b, &c, N_THREADS);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("(2) matmulSIMDAuto   (v0.0.4 spawn)   total={d:>8} ms,  per-call={d:>6} us\n", .{ ns / 1_000_000, ns / N_CALLS / 1_000 });
    }

    // (3) Persistent pool (v0.0.5)
    {
        var pool = try vllm.Pool.init(alloc, N_THREADS);
        defer pool.deinit();

        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < N_CALLS) : (i += 1) {
            try pool.matmulAuto(&a, &b, &c, N_THREADS);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("(3) Pool.matmulAuto  (v0.0.5 pool)    total={d:>8} ms,  per-call={d:>6} us\n", .{ ns / 1_000_000, ns / N_CALLS / 1_000 });
    }
}
