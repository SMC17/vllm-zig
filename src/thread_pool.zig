//! Persistent thread pool for matmul workers (Linux futex-backed).
//!
//! v0.0.5 — closes the v0.0.4 decode regression where ad-hoc `std.Thread.spawn`
//! per matmul call dominated wall time at M=1 (~30-100μs spawn × ~110 matmul
//! calls per token = ~3-11ms of pure spawn overhead per decode step).
//!
//! Design (Linux-only):
//!   - N worker threads, each parked on a private futex (round counter)
//!   - Main thread submits a job by filling in the shared job struct,
//!     incrementing the round counter, futex_waking all workers
//!   - Each worker reads its assigned [start, end) range, does the kernel
//!     work, atomically decrements `n_pending`, and parks again on the next
//!     round counter
//!   - When the last worker finishes, it futex_wakes the main thread
//!     waiting on `n_pending`
//!
//! Zig 0.16 ecosystem note: std.Thread.Pool was dropped from the stdlib
//! between 0.15 and 0.16. The replacement lives at std.Io.Threaded.Pool
//! but threading an `Io` parameter through the whole forward path was a
//! large refactor we wanted to defer. This module uses raw Linux futex
//! syscalls (std.os.linux.futex_3arg / futex_4arg) directly — Linux-only,
//! no Io required, no per-call spawn cost.
//!
//! Non-Linux platforms: this module is gated on `builtin.os.tag == .linux`
//! at the build wire-up. Callers should fall back to the ad-hoc spawn path
//! (kernels/matmul.zig matmulSIMDAuto) on non-Linux until a portable Io
//! path lands.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("tensor.zig").Matrix;
const matmul_kernels = @import("kernels/matmul.zig");

const linux = std.os.linux;

pub const Error = error{
    OutOfMemory,
    SpawnFailed,
    UnsupportedPlatform,
    DimensionMismatch,
    TooManyWorkers,
};

const max_workers = 32;

const WorkKind = enum(u32) {
    idle = 0,
    row_split = 1,
    col_split = 2,
    shutdown = 3,
};

const Job = struct {
    a: ?*const Matrix = null,
    b: ?*const Matrix = null,
    c: ?*Matrix = null,
    ranges: [max_workers][2]usize = [_][2]usize{.{ 0, 0 }} ** max_workers,
};

pub const Pool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,

    round: std.atomic.Value(u32) align(64),
    n_pending: std.atomic.Value(u32) align(64),
    work_kind: std.atomic.Value(u32) align(64),
    active_workers: std.atomic.Value(u32) align(64),

    job: Job align(64),

    pub fn init(allocator: std.mem.Allocator, n_workers: usize) Error!*Pool {
        if (builtin.os.tag != .linux) return Error.UnsupportedPlatform;
        if (n_workers == 0 or n_workers > max_workers) return Error.TooManyWorkers;

        const pool = try allocator.create(Pool);
        errdefer allocator.destroy(pool);

        pool.* = .{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, n_workers),
            .round = .init(0),
            .n_pending = .init(0),
            .work_kind = .init(@intFromEnum(WorkKind.idle)),
            .active_workers = .init(0),
            .job = .{},
        };
        errdefer allocator.free(pool.threads);

        var i: usize = 0;
        errdefer {
            // If a later spawn fails, signal shutdown to the already-spawned
            // workers and join them.
            pool.work_kind.store(@intFromEnum(WorkKind.shutdown), .release);
            _ = pool.round.fetchAdd(1, .acq_rel);
            futexWakeAll(&pool.round);
            for (pool.threads[0..i]) |t| t.join();
        }
        while (i < n_workers) : (i += 1) {
            pool.threads[i] = std.Thread.spawn(.{}, workerLoop, .{ pool, i }) catch return Error.SpawnFailed;
        }

        return pool;
    }

    pub fn deinit(self: *Pool) void {
        self.work_kind.store(@intFromEnum(WorkKind.shutdown), .release);
        _ = self.round.fetchAdd(1, .acq_rel);
        futexWakeAll(&self.round);
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        const alloc = self.allocator;
        alloc.destroy(self);
    }

    pub fn nWorkers(self: *const Pool) usize {
        return self.threads.len;
    }

    /// Row-split matmul (prefill path, M >= n_workers).
    pub fn matmulRowSplit(self: *Pool, a: *const Matrix, b: *const Matrix, c: *Matrix, n_workers: usize) Error!void {
        if (a.cols != b.rows) return Error.DimensionMismatch;
        if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
        const M = a.rows;
        if (n_workers <= 1 or M <= 1) {
            matmul_kernels.matmulSIMD(a, b, c) catch return Error.DimensionMismatch;
            return;
        }

        @memset(c.data, 0.0);
        const T = @min(@min(n_workers, M), self.threads.len);
        const chunk = (M + T - 1) / T;

        try self.submit(.row_split, a, b, c, T, fillRowRanges, .{ chunk, M });
    }

    /// Column-split matmul (decode path, M=1 typical).
    pub fn matmulColSplit(self: *Pool, a: *const Matrix, b: *const Matrix, c: *Matrix, n_workers: usize) Error!void {
        if (a.cols != b.rows) return Error.DimensionMismatch;
        if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
        if (n_workers <= 1) {
            matmul_kernels.matmulSIMD(a, b, c) catch return Error.DimensionMismatch;
            return;
        }

        @memset(c.data, 0.0);
        const N = b.cols;
        const T = @min(@min(n_workers, N), self.threads.len);
        if (T <= 1) {
            matmul_kernels.matmulSIMD(a, b, c) catch return Error.DimensionMismatch;
            return;
        }
        const chunk = (N + T - 1) / T;

        try self.submit(.col_split, a, b, c, T, fillColRanges, .{ chunk, N });
    }

    /// Auto-dispatch: row-split when M >= n_workers, col-split otherwise.
    pub fn matmulAuto(self: *Pool, a: *const Matrix, b: *const Matrix, c: *Matrix, n_workers: usize) Error!void {
        if (n_workers <= 1) {
            matmul_kernels.matmulSIMD(a, b, c) catch return Error.DimensionMismatch;
            return;
        }
        if (a.rows >= n_workers) return self.matmulRowSplit(a, b, c, n_workers);
        return self.matmulColSplit(a, b, c, n_workers);
    }

    fn submit(
        self: *Pool,
        kind: WorkKind,
        a: *const Matrix,
        b: *const Matrix,
        c: *Matrix,
        T: usize,
        comptime fill: anytype,
        fill_args: anytype,
    ) Error!void {
        // Fill job state. Workers are still parked on the OLD round, so this
        // is safe without locks.
        self.job.a = a;
        self.job.b = b;
        self.job.c = c;
        @call(.auto, fill, .{&self.job.ranges} ++ fill_args);
        self.active_workers.store(@as(u32, @intCast(T)), .release);
        self.n_pending.store(@as(u32, @intCast(T)), .release);
        self.work_kind.store(@intFromEnum(kind), .release);

        // Bump round and wake all workers.
        _ = self.round.fetchAdd(1, .acq_rel);
        futexWakeAll(&self.round);

        // Wait for completion.
        while (true) {
            const pending = self.n_pending.load(.acquire);
            if (pending == 0) break;
            futexWait(&self.n_pending, pending);
        }

        // Reset work_kind back to idle for the next round (workers don't
        // race on this — they've already read it before signaling done).
        self.work_kind.store(@intFromEnum(WorkKind.idle), .release);
    }
};

fn fillRowRanges(ranges: *[max_workers][2]usize, chunk: usize, M: usize) void {
    var t: usize = 0;
    while (t < max_workers) : (t += 1) {
        const start = t * chunk;
        const end = @min(start + chunk, M);
        ranges[t] = .{ start, end };
    }
}

fn fillColRanges(ranges: *[max_workers][2]usize, chunk: usize, N: usize) void {
    var t: usize = 0;
    while (t < max_workers) : (t += 1) {
        const start = t * chunk;
        const end = @min(start + chunk, N);
        ranges[t] = .{ start, end };
    }
}

fn workerLoop(pool: *Pool, id: usize) void {
    var my_round: u32 = 0;
    while (true) {
        // Park on the round counter until main bumps it.
        while (true) {
            const cur = pool.round.load(.acquire);
            if (cur != my_round) break;
            futexWait(&pool.round, my_round);
        }
        my_round = pool.round.load(.acquire);

        const kind: WorkKind = @enumFromInt(pool.work_kind.load(.acquire));
        if (kind == .shutdown) return;

        const active = pool.active_workers.load(.acquire);
        if (id < active) {
            const range = pool.job.ranges[id];
            if (range[0] < range[1]) {
                if (pool.job.a) |a| if (pool.job.b) |b| if (pool.job.c) |c| {
                    switch (kind) {
                        .row_split => matmulRowsKernel(a, b, c, range[0], range[1]),
                        .col_split => matmulColRangeKernel(a, b, c, range[0], range[1]),
                        else => {},
                    }
                };
            }

            // Signal done. If we're the last, wake the main thread.
            const prev = pool.n_pending.fetchSub(1, .acq_rel);
            if (prev == 1) futexWakeAll(&pool.n_pending);
        }
    }
}

fn matmulRowsKernel(a: *const Matrix, b: *const Matrix, c: *Matrix, row_start: usize, row_end: usize) void {
    const N = b.cols;
    const K = a.cols;

    // Route through the cache-tiled kernel for high-N high-K shapes
    // (prefill on TinyLlama-class models). Below this, the row-walk
    // path's hardware prefetch beats the tile-machinery overhead.
    // Threshold mirrors matmulSIMDTiled's crossover.
    //
    // Pool's matmulRowSplit already @memset'd C to zero; matmulSIMDTiledRows
    // zeros its own row range again (cheap) so the accumulation is correct.
    if (N >= 1280 and K >= 512) {
        matmul_kernels.matmulSIMDTiledRows(a, b, c, row_start, row_end) catch unreachable;
        return;
    }

    const Vec = @Vector(8, f32);
    const LANE = 8;
    const n_tail = N - (N % LANE);

    var i = row_start;
    while (i < row_end) : (i += 1) {
        const c_row = c.data[i * N ..][0..N];
        const a_row = a.data[i * K ..][0..K];

        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec = @splat(a_row[k]);
            const b_row = b.data[k * N ..][0..N];

            var j: usize = 0;
            while (j < n_tail) : (j += LANE) {
                const bv: Vec = b_row[j..][0..LANE].*;
                const cv: Vec = c_row[j..][0..LANE].*;
                c_row[j..][0..LANE].* = cv + a_scalar * bv;
            }
            while (j < N) : (j += 1) {
                c_row[j] += a_row[k] * b_row[j];
            }
        }
    }
}

fn matmulColRangeKernel(a: *const Matrix, b: *const Matrix, c: *Matrix, col_start: usize, col_end: usize) void {
    const M = a.rows;
    const N = b.cols;
    const K = a.cols;
    const Vec = @Vector(8, f32);
    const LANE = 8;

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec = @splat(a_row[k]);
            const b_row = b.data[k * N ..][0..N];

            var j = col_start;
            const aligned_end = col_start + ((col_end - col_start) / LANE) * LANE;
            while (j < aligned_end) : (j += LANE) {
                const bv: Vec = b_row[j..][0..LANE].*;
                const cv: Vec = c_row[j..][0..LANE].*;
                c_row[j..][0..LANE].* = cv + a_scalar * bv;
            }
            while (j < col_end) : (j += 1) {
                c_row[j] += a_row[k] * b_row[j];
            }
        }
    }
}

inline fn futexWait(addr: *const std.atomic.Value(u32), expected: u32) void {
    if (builtin.os.tag != .linux) return;
    _ = linux.futex_4arg(
        @ptrCast(&addr.raw),
        .{ .cmd = .WAIT, .private = true },
        expected,
        null,
    );
}

inline fn futexWakeAll(addr: *std.atomic.Value(u32)) void {
    if (builtin.os.tag != .linux) return;
    _ = linux.futex_3arg(
        @ptrCast(&addr.raw),
        .{ .cmd = .WAKE, .private = true },
        std.math.maxInt(i32),
    );
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Pool — init/deinit smoke (no work submitted)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();
    try testing.expectEqual(@as(usize, 4), pool.nWorkers());
}

test "Pool — row-split matches single-thread on 32x32x32" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const N: usize = 32;
    var a = try Matrix.init(testing.allocator, N, N);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, N, N);
    defer b.deinit(testing.allocator);
    var c_single = try Matrix.init(testing.allocator, N, N);
    defer c_single.deinit(testing.allocator);
    var c_pool = try Matrix.init(testing.allocator, N, N);
    defer c_pool.deinit(testing.allocator);

    a.fillRandom(101);
    b.fillRandom(103);

    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    try matmul_kernels.matmulSIMD(&a, &b, &c_single);
    try pool.matmulRowSplit(&a, &b, &c_pool, 4);

    try testing.expect(c_single.approxEq(&c_pool, 1e-4));
}

test "Pool — col-split matches single-thread on 1x32x128 (decode shape)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var a = try Matrix.init(testing.allocator, 1, 32);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 32, 128);
    defer b.deinit(testing.allocator);
    var c_single = try Matrix.init(testing.allocator, 1, 128);
    defer c_single.deinit(testing.allocator);
    var c_pool = try Matrix.init(testing.allocator, 1, 128);
    defer c_pool.deinit(testing.allocator);

    a.fillRandom(107);
    b.fillRandom(109);

    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    try matmul_kernels.matmulSIMD(&a, &b, &c_single);
    try pool.matmulColSplit(&a, &b, &c_pool, 4);

    try testing.expect(c_single.approxEq(&c_pool, 1e-4));
}

test "Pool — auto-dispatch row for M>=T, col for M<T" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    // Row-split path (M=16 >= T=4)
    var a1 = try Matrix.init(testing.allocator, 16, 16);
    defer a1.deinit(testing.allocator);
    var b1 = try Matrix.init(testing.allocator, 16, 16);
    defer b1.deinit(testing.allocator);
    var c1s = try Matrix.init(testing.allocator, 16, 16);
    defer c1s.deinit(testing.allocator);
    var c1p = try Matrix.init(testing.allocator, 16, 16);
    defer c1p.deinit(testing.allocator);
    a1.fillRandom(113);
    b1.fillRandom(127);
    try matmul_kernels.matmulSIMD(&a1, &b1, &c1s);
    try pool.matmulAuto(&a1, &b1, &c1p, 4);
    try testing.expect(c1s.approxEq(&c1p, 1e-4));

    // Col-split path (M=1 < T=4)
    var a2 = try Matrix.init(testing.allocator, 1, 32);
    defer a2.deinit(testing.allocator);
    var b2 = try Matrix.init(testing.allocator, 32, 128);
    defer b2.deinit(testing.allocator);
    var c2s = try Matrix.init(testing.allocator, 1, 128);
    defer c2s.deinit(testing.allocator);
    var c2p = try Matrix.init(testing.allocator, 1, 128);
    defer c2p.deinit(testing.allocator);
    a2.fillRandom(131);
    b2.fillRandom(137);
    try matmul_kernels.matmulSIMD(&a2, &b2, &c2s);
    try pool.matmulAuto(&a2, &b2, &c2p, 4);
    try testing.expect(c2s.approxEq(&c2p, 1e-4));
}

test "Pool — 100 consecutive submits reuse the same workers" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var a = try Matrix.init(testing.allocator, 1, 64);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 64, 64);
    defer b.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 1, 64);
    defer c.deinit(testing.allocator);
    a.fillRandom(139);
    b.fillRandom(149);

    var pool = try Pool.init(testing.allocator, 8);
    defer pool.deinit();

    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        try pool.matmulAuto(&a, &b, &c, 8);
    }
    var any_nonzero = false;
    for (c.data) |v| if (v != 0) {
        any_nonzero = true;
        break;
    };
    try testing.expect(any_nonzero);
}
