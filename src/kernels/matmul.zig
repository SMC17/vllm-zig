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
