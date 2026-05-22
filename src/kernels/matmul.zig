//! Matmul kernels — naive scalar reference + SIMD inner-loop version.
//!
//! Computes C = A @ B where:
//!   A: [M, K]
//!   B: [K, N]
//!   C: [M, N]
//!
//! Phase 1 scope:
//!   - f32 only
//!   - Row-major
//!   - Single-thread
//!   - No cache tiling
//!   - SIMD via @Vector(8, f32) along the K dimension
//!
//! What's NOT here (yet):
//!   - Multi-thread tiling — Phase 2
//!   - GEMM-style blocking (cache + register tiling) — Phase 2
//!   - Mixed-precision (f16 storage, f32 accumulate) — Phase 3
//!   - Fused GEMM+bias+act — Phase 2
//!
//! The naive reference exists primarily to validate the SIMD path
//! and serve as the correctness oracle for future optimization passes.

const std = @import("std");
const Matrix = @import("../tensor.zig").Matrix;

pub const Error = error{
    DimensionMismatch,
};

/// Naive O(M*N*K) scalar matmul. Reference implementation.
pub fn matmulScalar(a: *const Matrix, b: *const Matrix, c: *Matrix) Error!void {
    if (a.cols != b.rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
    const M = a.rows;
    const N = b.cols;
    const K = a.cols;

    var i: usize = 0;
    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < K) : (k += 1) {
                sum += a.data[i * K + k] * b.data[k * N + j];
            }
            c.data[i * N + j] = sum;
        }
    }
}

/// Multi-thread SIMD matmul. Each thread processes a contiguous chunk
/// of A's rows (and the corresponding chunk of C's rows). Inner loop
/// uses the same outer-product SIMD pattern as matmulSIMD.
///
/// For N threads, M rows split into N chunks of ceil(M/N) each.
/// T=1 falls through to single-thread matmulSIMD to avoid spawn overhead.
pub fn matmulSIMDMulti(a: *const Matrix, b: *const Matrix, c: *Matrix, n_threads: usize) Error!void {
    if (a.cols != b.rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
    const M = a.rows;
    if (n_threads <= 1 or M <= 1) return matmulSIMD(a, b, c);

    @memset(c.data, 0.0);

    const T = @min(n_threads, M);
    const chunk = (M + T - 1) / T;

    const Args = struct {
        a: *const Matrix,
        b: *const Matrix,
        c: *Matrix,
        row_start: usize,
        row_end: usize,
    };

    var threads_buf: [16]std.Thread = undefined;
    var args_buf: [16]Args = undefined;
    if (T > threads_buf.len) return matmulSIMD(a, b, c);

    var t: usize = 0;
    while (t < T) : (t += 1) {
        const start = t * chunk;
        const end = @min(start + chunk, M);
        args_buf[t] = .{
            .a = a,
            .b = b,
            .c = c,
            .row_start = start,
            .row_end = end,
        };
        threads_buf[t] = std.Thread.spawn(.{}, struct {
            fn run(arg: *Args) void {
                matmulSIMDRows(arg.a, arg.b, arg.c, arg.row_start, arg.row_end) catch unreachable;
            }
        }.run, .{&args_buf[t]}) catch return Error.DimensionMismatch;
    }
    t = 0;
    while (t < T) : (t += 1) threads_buf[t].join();
}

/// Smart multi-thread matmul: chooses row-split when M >= n_threads,
/// col-split when M < n_threads (typical for decode where M = seq = 1).
/// This is the v0.0.4 fix for the M=1 fall-through that prevented
/// multi-thread parallelism at decode in v0.0.3.
pub fn matmulSIMDAuto(a: *const Matrix, b: *const Matrix, c: *Matrix, n_threads: usize) Error!void {
    if (n_threads <= 1) return matmulSIMD(a, b, c);
    const M = a.rows;
    if (M >= n_threads) {
        return matmulSIMDMulti(a, b, c, n_threads);
    } else {
        return matmulSIMDColSplit(a, b, c, n_threads);
    }
}

/// Column-split multi-thread matmul: each thread handles a contiguous
/// slice of N (output columns) instead of M (output rows). Used when
/// M is too small to row-split (decode: M=seq=1).
pub fn matmulSIMDColSplit(a: *const Matrix, b: *const Matrix, c: *Matrix, n_threads: usize) Error!void {
    if (a.cols != b.rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
    if (n_threads <= 1) return matmulSIMD(a, b, c);

    @memset(c.data, 0.0);
    const N = b.cols;
    const T = @min(n_threads, N);
    if (T <= 1 or T > 16) return matmulSIMD(a, b, c);
    const chunk = (N + T - 1) / T;

    const Args = struct {
        a: *const Matrix,
        b: *const Matrix,
        c: *Matrix,
        col_start: usize,
        col_end: usize,
    };

    var threads_buf: [16]std.Thread = undefined;
    var args_buf: [16]Args = undefined;

    var t: usize = 0;
    while (t < T) : (t += 1) {
        const start = t * chunk;
        const end = @min(start + chunk, N);
        args_buf[t] = .{ .a = a, .b = b, .c = c, .col_start = start, .col_end = end };
        threads_buf[t] = std.Thread.spawn(.{}, struct {
            fn run(arg: *Args) void {
                matmulSIMDColRange(arg.a, arg.b, arg.c, arg.col_start, arg.col_end) catch unreachable;
            }
        }.run, .{&args_buf[t]}) catch return Error.DimensionMismatch;
    }
    t = 0;
    while (t < T) : (t += 1) threads_buf[t].join();
}

fn matmulSIMDColRange(a: *const Matrix, b: *const Matrix, c: *Matrix, col_start: usize, col_end: usize) Error!void {
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

/// Per-thread worker: matmulSIMD over a slice of A's rows [row_start, row_end).
fn matmulSIMDRows(a: *const Matrix, b: *const Matrix, c: *Matrix, row_start: usize, row_end: usize) Error!void {
    const N = b.cols;
    const K = a.cols;
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

/// SIMD matmul. The inner accumulator uses @Vector(8, f32) over the
/// K dimension. Tail K-elements (K mod 8) are handled in a scalar epilogue.
///
/// Important: this implementation transposes B implicitly by walking
/// b.row(k) in the inner loop, which gives us a contiguous-stride read
/// of B's k-th row. That's what makes the SIMD load aligned.
///
/// Actually since C[i,j] = sum_k A[i,k] * B[k,j], we walk B *down a column*
/// in the natural layout. To get SIMD on K, we need contiguous access
/// of B's column j. That requires either: (a) transposing B first, or
/// (b) accumulating into a row of C across many k-pairs simultaneously.
///
/// For v0.0.1 we use approach (b): for each (i, k) pair, broadcast
/// A[i,k] as a scalar and vector-multiply with B's k-th row, accumulating
/// into a row of C. This is the canonical "outer-product-on-the-right"
/// matmul pattern that produces clean SIMD over N.
pub fn matmulSIMD(a: *const Matrix, b: *const Matrix, c: *Matrix) Error!void {
    if (a.cols != b.rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b.cols) return Error.DimensionMismatch;
    const M = a.rows;
    const N = b.cols;
    const K = a.cols;
    const Vec = @Vector(8, f32);
    const LANE = 8;
    const n_tail = N - (N % LANE);

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
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
            // Tail
            while (j < N) : (j += 1) {
                c_row[j] += a_row[k] * b_row[j];
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "matmul scalar — 2x3 @ 3x4 known result" {
    var a = try Matrix.init(testing.allocator, 2, 3);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 3, 4);
    defer b.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 2, 4);
    defer c.deinit(testing.allocator);

    // a = [[1,2,3],[4,5,6]]
    const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    @memcpy(a.data, &a_vals);
    // b = identity-ish: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    b.set(0, 0, 1);
    b.set(1, 1, 1);
    b.set(2, 2, 1);
    // expected c = [[1,2,3,0],[4,5,6,0]]
    try matmulScalar(&a, &b, &c);
    try testing.expectEqual(@as(f32, 1), c.at(0, 0));
    try testing.expectEqual(@as(f32, 2), c.at(0, 1));
    try testing.expectEqual(@as(f32, 3), c.at(0, 2));
    try testing.expectEqual(@as(f32, 0), c.at(0, 3));
    try testing.expectEqual(@as(f32, 4), c.at(1, 0));
    try testing.expectEqual(@as(f32, 6), c.at(1, 2));
}

test "matmul scalar — dimension mismatch" {
    var a = try Matrix.init(testing.allocator, 2, 3);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 4, 5);
    defer b.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 2, 5);
    defer c.deinit(testing.allocator);
    try testing.expectError(Error.DimensionMismatch, matmulScalar(&a, &b, &c));
}

test "matmul SIMD agrees with scalar on (M, K, N) variants" {
    inline for ([_][3]usize{
        .{ 1, 1, 1 },
        .{ 8, 8, 8 },
        .{ 9, 9, 9 }, // tail
        .{ 16, 32, 16 },
        .{ 17, 33, 19 }, // all tail
        .{ 32, 64, 128 },
        .{ 128, 128, 128 },
    }) |dims| {
        const M = dims[0];
        const K = dims[1];
        const N = dims[2];

        var a = try Matrix.init(testing.allocator, M, K);
        defer a.deinit(testing.allocator);
        var b = try Matrix.init(testing.allocator, K, N);
        defer b.deinit(testing.allocator);
        var c_scalar = try Matrix.init(testing.allocator, M, N);
        defer c_scalar.deinit(testing.allocator);
        var c_simd = try Matrix.init(testing.allocator, M, N);
        defer c_simd.deinit(testing.allocator);

        a.fillRandom(M * 31 + K * 17 + N);
        b.fillRandom(M * 13 + K * 5 + N * 3);

        try matmulScalar(&a, &b, &c_scalar);
        try matmulSIMD(&a, &b, &c_simd);

        // FP reduction order differs between scalar (1 accumulator) and
        // SIMD (vector accumulator + later scalar tail). Tolerate
        // ~1e-4 relative for moderate K.
        try testing.expect(c_scalar.approxEq(&c_simd, 1e-4));
    }
}

test "matmulSIMDMulti — T=4 agrees with single-thread on 64x64x64" {
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;
    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, K, N);
    defer b.deinit(testing.allocator);
    var c1 = try Matrix.init(testing.allocator, M, N);
    defer c1.deinit(testing.allocator);
    var c4 = try Matrix.init(testing.allocator, M, N);
    defer c4.deinit(testing.allocator);

    a.fillRandom(7);
    b.fillRandom(11);

    try matmulSIMD(&a, &b, &c1);
    try matmulSIMDMulti(&a, &b, &c4, 4);

    try testing.expect(c1.approxEq(&c4, 1e-4));
}

test "matmulSIMDMulti — T=1 falls through to single-thread" {
    var a = try Matrix.init(testing.allocator, 8, 8);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 8, 8);
    defer b.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 8, 8);
    defer c.deinit(testing.allocator);
    a.fillRandom(3);
    b.fillRandom(5);
    try matmulSIMDMulti(&a, &b, &c, 1);
    // Just verify no errors + result is non-zero
    var any_nonzero = false;
    for (c.data) |v| if (v != 0) { any_nonzero = true; break; };
    try testing.expect(any_nonzero);
}

test "matmul identity — square matrix vs identity gives the original" {
    const N: usize = 16;
    var a = try Matrix.init(testing.allocator, N, N);
    defer a.deinit(testing.allocator);
    var ident = try Matrix.init(testing.allocator, N, N);
    defer ident.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, N, N);
    defer c.deinit(testing.allocator);

    a.fillRandom(7);
    var k: usize = 0;
    while (k < N) : (k += 1) ident.set(k, k, 1.0);

    try matmulSIMD(&a, &ident, &c);
    try testing.expect(a.approxEq(&c, 1e-6));
}
