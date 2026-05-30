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
const matmul_bf16_kernels = @import("kernels/matmul_bf16.zig");

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
    bf16_row_split = 4,
    bf16_col_split = 5,
};

const Job = struct {
    a: ?*const Matrix = null,
    b: ?*const Matrix = null,
    c: ?*Matrix = null,
    // BF16-weights path: B lives as raw u16 plus row/col dims since it
    // is not a Matrix. When set, b_bf16 is non-null and b is ignored.
    b_bf16: ?[]const u16 = null,
    b_bf16_rows: usize = 0,
    b_bf16_cols: usize = 0,
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

    /// Column-split BF16-weights matmul (decode path, M=1 typical).
    /// A is F32, B is raw BF16 (u16), C is F32. Same threading shape
    /// as matmulColSplit; per-worker kernel widens BF16 -> F32 in
    /// register via matmul_bf16.matmulBF16SIMD's lane sequence.
    pub fn matmulBF16ColSplit(
        self: *Pool,
        a: *const Matrix,
        b: []const u16,
        b_rows: usize,
        b_cols: usize,
        c: *Matrix,
        n_workers: usize,
    ) Error!void {
        if (a.cols != b_rows) return Error.DimensionMismatch;
        if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
        if (b.len != b_rows * b_cols) return Error.DimensionMismatch;
        if (n_workers <= 1) {
            matmul_bf16_kernels.matmulBF16SIMD(a, b, b_rows, b_cols, c) catch return Error.DimensionMismatch;
            return;
        }

        @memset(c.data, 0.0);
        const N = b_cols;
        const T = @min(@min(n_workers, N), self.threads.len);
        if (T <= 1) {
            matmul_bf16_kernels.matmulBF16SIMD(a, b, b_rows, b_cols, c) catch return Error.DimensionMismatch;
            return;
        }
        const chunk = (N + T - 1) / T;

        try self.submitBF16(.bf16_col_split, a, b, b_rows, b_cols, c, T, fillColRanges, .{ chunk, N });
    }

    /// Row-split BF16-weights matmul (prefill path, M >= n_workers).
    /// A is F32, B is raw BF16 (u16), C is F32.
    pub fn matmulBF16RowSplit(
        self: *Pool,
        a: *const Matrix,
        b: []const u16,
        b_rows: usize,
        b_cols: usize,
        c: *Matrix,
        n_workers: usize,
    ) Error!void {
        if (a.cols != b_rows) return Error.DimensionMismatch;
        if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
        if (b.len != b_rows * b_cols) return Error.DimensionMismatch;
        const M = a.rows;
        if (n_workers <= 1 or M <= 1) {
            matmul_bf16_kernels.matmulBF16SIMD(a, b, b_rows, b_cols, c) catch return Error.DimensionMismatch;
            return;
        }

        @memset(c.data, 0.0);
        const T = @min(@min(n_workers, M), self.threads.len);
        const chunk = (M + T - 1) / T;

        try self.submitBF16(.bf16_row_split, a, b, b_rows, b_cols, c, T, fillRowRanges, .{ chunk, M });
    }

    /// Auto-dispatch BF16-weights matmul: row-split when M >= n_workers,
    /// col-split otherwise. Mirrors matmulAuto's policy for the F32 path.
    pub fn matmulBF16Auto(
        self: *Pool,
        a: *const Matrix,
        b: []const u16,
        b_rows: usize,
        b_cols: usize,
        c: *Matrix,
        n_workers: usize,
    ) Error!void {
        if (n_workers <= 1) {
            matmul_bf16_kernels.matmulBF16SIMD(a, b, b_rows, b_cols, c) catch return Error.DimensionMismatch;
            return;
        }
        if (a.rows >= n_workers) return self.matmulBF16RowSplit(a, b, b_rows, b_cols, c, n_workers);
        return self.matmulBF16ColSplit(a, b, b_rows, b_cols, c, n_workers);
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
        self.job.b_bf16 = null;
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

    fn submitBF16(
        self: *Pool,
        kind: WorkKind,
        a: *const Matrix,
        b_bf16: []const u16,
        b_rows: usize,
        b_cols: usize,
        c: *Matrix,
        T: usize,
        comptime fill: anytype,
        fill_args: anytype,
    ) Error!void {
        // Same race shape as submit(): workers are parked on the OLD
        // round so per-round job state is publication-safe via the
        // round.fetchAdd release.
        self.job.a = a;
        self.job.b = null;
        self.job.c = c;
        self.job.b_bf16 = b_bf16;
        self.job.b_bf16_rows = b_rows;
        self.job.b_bf16_cols = b_cols;
        @call(.auto, fill, .{&self.job.ranges} ++ fill_args);
        self.active_workers.store(@as(u32, @intCast(T)), .release);
        self.n_pending.store(@as(u32, @intCast(T)), .release);
        self.work_kind.store(@intFromEnum(kind), .release);

        _ = self.round.fetchAdd(1, .acq_rel);
        futexWakeAll(&self.round);

        while (true) {
            const pending = self.n_pending.load(.acquire);
            if (pending == 0) break;
            futexWait(&self.n_pending, pending);
        }

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
                if (pool.job.a) |a| if (pool.job.c) |c| {
                    switch (kind) {
                        .row_split => if (pool.job.b) |b| matmulRowsKernel(a, b, c, range[0], range[1]),
                        .col_split => if (pool.job.b) |b| matmulColRangeKernel(a, b, c, range[0], range[1]),
                        .bf16_row_split => if (pool.job.b_bf16) |bb| matmulBF16RowsKernel(
                            a,
                            bb,
                            pool.job.b_bf16_rows,
                            pool.job.b_bf16_cols,
                            c,
                            range[0],
                            range[1],
                        ),
                        .bf16_col_split => if (pool.job.b_bf16) |bb| matmulBF16ColRangeKernel(
                            a,
                            bb,
                            pool.job.b_bf16_rows,
                            pool.job.b_bf16_cols,
                            c,
                            range[0],
                            range[1],
                        ),
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

    // 16-lane: see kernels/matmul.zig matmulSIMDColRange docstring.
    const Vec = @Vector(16, f32);
    const LANE = 16;
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
    // 16-lane: see kernels/matmul.zig matmulSIMDColRange docstring.
    const Vec = @Vector(16, f32);
    const LANE = 16;

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

/// BF16-weights row-split per-thread kernel. Walks rows [row_start, row_end)
/// of A and writes full C rows. Widens B (u16) -> f32 in-register via the
/// same shift-left-16 lane sequence as matmul_bf16.matmulBF16SIMD.
fn matmulBF16RowsKernel(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
    row_start: usize,
    row_end: usize,
) void {
    _ = b_rows;
    // 16-lane: see kernels/matmul.zig matmulSIMDColRange docstring.
    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;

    const N = b_cols;
    const K = a.cols;
    const n_tail = N - (N % LANE);

    var i = row_start;
    while (i < row_end) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];

        // Pool's matmulBF16RowSplit @memset'd C to zero before submit;
        // accumulation here is safe.
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec_f32 = @splat(a_row[k]);
            const b_row = b[k * N ..][0..N];

            var j: usize = 0;
            while (j < n_tail) : (j += LANE) {
                const b_u16v: Vec_u16 = b_row[j..][0..LANE].*;
                const b_u32v: Vec_u32 = b_u16v;
                const b_shifted: Vec_u32 = b_u32v << @splat(@as(u5, 16));
                const b_f32v: Vec_f32 = @bitCast(b_shifted);
                const cv: Vec_f32 = c_row[j..][0..LANE].*;
                c_row[j..][0..LANE].* = cv + a_scalar * b_f32v;
            }
            while (j < N) : (j += 1) {
                const bv: f32 = @bitCast(@as(u32, b_row[j]) << 16);
                c_row[j] += a_row[k] * bv;
            }
        }
    }
}

/// BF16-weights column-range per-thread kernel. M is implicit (a.rows);
/// each worker writes columns [col_start, col_end) for ALL rows.
/// This is the decode-shape path: M=1, N=vocab_size, T workers each own
/// a vocab slice.
///
/// v0.0.10 lever-5 (register-blocked inner-product form):
///   - N-tile = 64 columns held in 4 ZMM accumulators across full K loop.
///   - K*N c-reads/writes collapse to 1 store per N-tile (K-fold reduction).
///   - Tail per LANE = 16 (single-accumulator inner-product), then scalar.
/// Pool's submitBF16 @memset C to zero before dispatch, so a final store
/// is correct (no accumulation needed for tiled ranges we fully overwrite).
fn matmulBF16ColRangeKernel(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
    col_start: usize,
    col_end: usize,
) void {
    _ = b_rows;
    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;
    // TILE=64 is the lever-5 fixed point.
    // (Lever-8 doubled to TILE=128 in exp-1780109796-369194807; refuted —
    // L1 read-port saturation cancels the larger register block. Decode med
    // 272 vs 277 was within noise, p90 +25pct worse, mlp_gate_up +10pct.)
    const TILE = 4 * LANE;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const span = col_end - col_start;
    const tile_span = (span / TILE) * TILE;
    const lane_span_after_tile = ((span - tile_span) / LANE) * LANE;
    const j_tile_end = col_start + tile_span;
    const j_lane_end = j_tile_end + lane_span_after_tile;

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        const shift: @Vector(16, u5) = @splat(16);

        // 4-ZMM register-blocked main loop: 64 cols per tile, accumulators
        // stay in registers across K, single store at tile end.
        // (Lever-6 manual prefetch attempted exp-1780108992-833810307 —
        // refuted: 2x regression across all regions because @prefetch
        // locality=3 in the inner K loop thrashes L1 across 4 worker
        // threads and fights Ice Lake's L2 streamer. Hardware prefetcher
        // already saturates this access pattern.)
        var j = col_start;
        while (j < j_tile_end) : (j += TILE) {
            var acc0: Vec_f32 = @splat(0.0);
            var acc1: Vec_f32 = @splat(0.0);
            var acc2: Vec_f32 = @splat(0.0);
            var acc3: Vec_f32 = @splat(0.0);

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_row = b[k * N + j ..][0..TILE];

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

            c_row[j..][0..LANE].* = acc0;
            c_row[j + LANE ..][0..LANE].* = acc1;
            c_row[j + 2 * LANE ..][0..LANE].* = acc2;
            c_row[j + 3 * LANE ..][0..LANE].* = acc3;
        }

        // Single-LANE inner-product tail for remaining 16-column blocks.
        while (j < j_lane_end) : (j += LANE) {
            var acc: Vec_f32 = @splat(0.0);
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_u16: Vec_u16 = b[k * N + j ..][0..LANE].*;
                const b_u32: Vec_u32 = b_u16;
                const b_f32: Vec_f32 = @bitCast(b_u32 << shift);
                acc = @mulAdd(Vec_f32, a_scalar, b_f32, acc);
            }
            c_row[j..][0..LANE].* = acc;
        }

        // Scalar tail for columns not covered by a full LANE.
        while (j < col_end) : (j += 1) {
            var acc: f32 = 0.0;
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const bv: f32 = @bitCast(@as(u32, b[k * N + j]) << 16);
                acc += a_row[k] * bv;
            }
            c_row[j] = acc;
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

const bf16_test_mod = @import("bf16.zig");

fn quantiseToBF16(allocator: std.mem.Allocator, m: *const Matrix) ![]u16 {
    const out = try allocator.alloc(u16, m.data.len);
    for (m.data, 0..) |x, i| out[i] = bf16_test_mod.f32ToBf16(x);
    return out;
}

test "Pool — BF16 col-split matches single-thread on 1x32x128 (decode shape)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var a = try Matrix.init(testing.allocator, 1, 32);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, 32, 128);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(151);
    b_f32.fillRandom(157);

    const b_bf16 = try quantiseToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_single = try Matrix.init(testing.allocator, 1, 128);
    defer c_single.deinit(testing.allocator);
    var c_pool = try Matrix.init(testing.allocator, 1, 128);
    defer c_pool.deinit(testing.allocator);

    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    try matmul_bf16_kernels.matmulBF16SIMD(&a, b_bf16, 32, 128, &c_single);
    try pool.matmulBF16ColSplit(&a, b_bf16, 32, 128, &c_pool, 4);

    try testing.expect(c_single.approxEq(&c_pool, 1e-4));
}

test "Pool — BF16 row-split matches single-thread on 16x32x48 (prefill shape)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var a = try Matrix.init(testing.allocator, 16, 32);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, 32, 48);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(163);
    b_f32.fillRandom(167);

    const b_bf16 = try quantiseToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_single = try Matrix.init(testing.allocator, 16, 48);
    defer c_single.deinit(testing.allocator);
    var c_pool = try Matrix.init(testing.allocator, 16, 48);
    defer c_pool.deinit(testing.allocator);

    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    try matmul_bf16_kernels.matmulBF16SIMD(&a, b_bf16, 32, 48, &c_single);
    try pool.matmulBF16RowSplit(&a, b_bf16, 32, 48, &c_pool, 4);

    try testing.expect(c_single.approxEq(&c_pool, 1e-4));
}

test "Pool — BF16 auto-dispatch row for M>=T, col for M<T" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var pool = try Pool.init(testing.allocator, 4);
    defer pool.deinit();

    // Row path: M=8 >= T=4
    var a1 = try Matrix.init(testing.allocator, 8, 16);
    defer a1.deinit(testing.allocator);
    var b1_f32 = try Matrix.init(testing.allocator, 16, 24);
    defer b1_f32.deinit(testing.allocator);
    a1.fillRandom(173);
    b1_f32.fillRandom(179);
    const b1_bf16 = try quantiseToBF16(testing.allocator, &b1_f32);
    defer testing.allocator.free(b1_bf16);
    var c1s = try Matrix.init(testing.allocator, 8, 24);
    defer c1s.deinit(testing.allocator);
    var c1p = try Matrix.init(testing.allocator, 8, 24);
    defer c1p.deinit(testing.allocator);
    try matmul_bf16_kernels.matmulBF16SIMD(&a1, b1_bf16, 16, 24, &c1s);
    try pool.matmulBF16Auto(&a1, b1_bf16, 16, 24, &c1p, 4);
    try testing.expect(c1s.approxEq(&c1p, 1e-4));

    // Col path: M=1 < T=4
    var a2 = try Matrix.init(testing.allocator, 1, 32);
    defer a2.deinit(testing.allocator);
    var b2_f32 = try Matrix.init(testing.allocator, 32, 64);
    defer b2_f32.deinit(testing.allocator);
    a2.fillRandom(181);
    b2_f32.fillRandom(191);
    const b2_bf16 = try quantiseToBF16(testing.allocator, &b2_f32);
    defer testing.allocator.free(b2_bf16);
    var c2s = try Matrix.init(testing.allocator, 1, 64);
    defer c2s.deinit(testing.allocator);
    var c2p = try Matrix.init(testing.allocator, 1, 64);
    defer c2p.deinit(testing.allocator);
    try matmul_bf16_kernels.matmulBF16SIMD(&a2, b2_bf16, 32, 64, &c2s);
    try pool.matmulBF16Auto(&a2, b2_bf16, 32, 64, &c2p, 4);
    try testing.expect(c2s.approxEq(&c2p, 1e-4));
}

test "Pool — BF16 lm_head-shape col-split matches single-thread (M=1, K=2048, N=32000)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    // Same shape as the real lm_head matmul in Model.forward; verifies
    // the col-split decomposition reconstructs the full single-thread
    // output bit-for-bit up to floating accumulation order.
    const K: usize = 2048;
    const N: usize = 32000;
    var a = try Matrix.init(testing.allocator, 1, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(193);
    b_f32.fillRandom(197);

    const b_bf16 = try quantiseToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_single = try Matrix.init(testing.allocator, 1, N);
    defer c_single.deinit(testing.allocator);
    var c_pool = try Matrix.init(testing.allocator, 1, N);
    defer c_pool.deinit(testing.allocator);

    var pool = try Pool.init(testing.allocator, 8);
    defer pool.deinit();

    try matmul_bf16_kernels.matmulBF16SIMD(&a, b_bf16, K, N, &c_single);
    try pool.matmulBF16ColSplit(&a, b_bf16, K, N, &c_pool, 8);

    // v0.0.10 lever-5: col-range kernel uses @mulAdd (FMA, one rounding)
    // while baseline matmulBF16SIMD does separate mul+add (two roundings).
    // max_abs diff is ~6e-6 (within f32 precision); max_rel can spike to
    // ~2% only on indices where both values are <1e-5 (small-magnitude
    // BF16 cancellation noise). Switch to absolute tolerance for that
    // edge case.
    for (c_single.data, c_pool.data) |x, y| {
        const d = @abs(x - y);
        const m = @max(@abs(x), @abs(y));
        if (m > 1e-3) {
            try testing.expect(d / m < 5e-3);
        } else {
            try testing.expect(d < 1e-4);
        }
    }
}
