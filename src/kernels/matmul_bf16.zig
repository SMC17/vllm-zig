//! BF16-weights matmul kernels.
//!
//! Decode on TinyLlama-class models is memory-bandwidth bound on the
//! B matrix scan. v0.0.5 stores weights as F32 (4 bytes/elem, ~16MB per
//! 2048×2048 layer); BF16 storage halves that to 8MB and lets the L2/L3
//! prefetcher cover twice the rows per cycle.
//!
//! Conversion is exact (BF16 = top 16 bits of an IEEE-754 binary32),
//! so per-element bf16→f32 is a single shift-by-16. The inner-loop cost
//! is therefore the load+widen, not the conversion math itself.
//!
//! API shape: B is a raw []const u16 (the loaded safetensors bytes
//! interpreted as little-endian BF16 units), not a Matrix. A and C
//! stay F32 Matrix because activations are kept in F32 throughout
//! the forward pass.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("../tensor.zig").Matrix;

pub const Error = error{DimensionMismatch};

const CBLAS_ORDER = enum(c_int) { row_major = 101, col_major = 102 };
const CBLAS_TRANSPOSE = enum(c_int) { no_trans = 111, trans = 112 };

// Accelerate cblas_sgemm. Only invoked on Apple Silicon where the
// Accelerate framework is linked; declaration is harmless on other
// targets because we never call into it.
extern "c" fn cblas_sgemm(
    order: CBLAS_ORDER,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) void;

// Size threshold (in B elements) above which the Accelerate dispatch
// path beats the in-process SIMD path. Empirically the cblas_sgemm
// overhead amortizes around 1M B-elements on M2 Max; below that the
// hand-rolled @Vector(8, f32) BF16-widening kernel wins because no
// BF16→F32 scratch buffer is allocated.
const ACCELERATE_MIN_B_ELEMS: usize = 1_000_000;

/// Scalar BF16-weights matmul.
/// A: [M, K] F32, B: [K, N] BF16 (row-major u16), C: [M, N] F32.
pub fn matmulBF16Scalar(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const av = a_row[k];
            const b_row = b[k * N ..][0..N];
            var j: usize = 0;
            while (j < N) : (j += 1) {
                const bv: f32 = @bitCast(@as(u32, b_row[j]) << 16);
                c_row[j] += av * bv;
            }
        }
    }
}

/// Vectorised BF16-weights matmul. 8-lane @Vector(f32) accumulator;
/// widens 8 u16 → 8 u32 → bitcast to 8 f32 per lane group.
pub fn matmulBF16SIMD(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const Vec_u16 = @Vector(8, u16);
    const Vec_u32 = @Vector(8, u32);
    const Vec_f32 = @Vector(8, f32);
    const LANE = 8;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tail = N - (N % LANE);

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec_f32 = @splat(a_row[k]);
            const b_row = b[k * N ..][0..N];

            var j: usize = 0;
            while (j < n_tail) : (j += LANE) {
                const b_u16v: Vec_u16 = b_row[j..][0..LANE].*;
                const b_u32v: Vec_u32 = b_u16v; // implicit widen
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

/// Apple Silicon Accelerate (cblas_sgemm) BF16-weights matmul.
///
/// Allocates a scratch F32 buffer of size K×N, widens the BF16 weights
/// into it once, then calls cblas_sgemm. For lm_head-class shapes
/// (M=1, K=2048, N=32000) on M2 Max this measures **6.97× faster** than
/// matmulBF16SIMD in standalone benches, with max_abs_diff ≤ 1.0e-5
/// vs the SIMD path (within BF16 quantization noise).
///
/// Trade-off: per-call BF16→F32 conversion is O(K*N) and memory-bandwidth
/// bound; in serving contexts the conversion could be cached at load
/// time. This signature performs the conversion inline so the function
/// is a drop-in replacement for matmulBF16SIMD.
///
/// Build requirement: caller's build.zig must link Accelerate
/// framework on macOS targets (`linkFramework("Accelerate")`).
pub fn matmulBF16Accelerate(
    allocator: std.mem.Allocator,
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) (Error || std.mem.Allocator.Error)!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const b_f32 = try allocator.alloc(f32, b.len);
    defer allocator.free(b_f32);
    for (b, 0..) |w, idx| {
        b_f32[idx] = @bitCast(@as(u32, w) << 16);
    }

    const M: c_int = @intCast(a.rows);
    const N: c_int = @intCast(b_cols);
    const K: c_int = @intCast(a.cols);

    cblas_sgemm(
        .row_major,
        .no_trans,
        .no_trans,
        M,
        N,
        K,
        1.0,
        a.data.ptr,
        K,
        b_f32.ptr,
        N,
        0.0,
        c.data.ptr,
        N,
    );
}

/// Auto-dispatch BF16-weights matmul.
///
/// On macOS (Apple Silicon), routes B-matrices with ≥1M elements through
/// matmulBF16Accelerate (AMX dispatch path); smaller shapes use the
/// in-process matmulBF16SIMD because cblas dispatch overhead would
/// dominate.
///
/// On non-macOS targets, always calls matmulBF16SIMD (cblas_sgemm is
/// never reachable; the extern declaration above is dead).
pub fn matmulBF16Auto(
    allocator: std.mem.Allocator,
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) (Error || std.mem.Allocator.Error)!void {
    if (comptime builtin.os.tag == .macos) {
        if (b.len >= ACCELERATE_MIN_B_ELEMS) {
            return matmulBF16Accelerate(allocator, a, b, b_rows, b_cols, c);
        }
    }
    return matmulBF16SIMD(a, b, b_rows, b_cols, c);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const bf16_mod = @import("../bf16.zig");

fn matrixToBF16(allocator: std.mem.Allocator, m: *const Matrix) ![]u16 {
    const out = try allocator.alloc(u16, m.data.len);
    for (m.data, 0..) |x, i| out[i] = bf16_mod.f32ToBf16(x);
    return out;
}

test "matmulBF16Scalar — agrees with reference F32 matmul of BF16-quantised weights" {
    const N: usize = 16;
    var a = try Matrix.init(testing.allocator, N, N);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, N, N);
    defer b_f32.deinit(testing.allocator);

    a.fillRandom(101);
    b_f32.fillRandom(103);

    // Quantise B through BF16 first so the reference and the
    // BF16-kernel see the SAME numerical weights (eliminates the
    // F32→BF16 round-trip error from the comparison and isolates
    // the kernel's correctness).
    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);
    var b_requantised = try Matrix.init(testing.allocator, N, N);
    defer b_requantised.deinit(testing.allocator);
    for (b_bf16, 0..) |bits, idx| b_requantised.data[idx] = bf16_mod.bf16ToF32(bits);

    // Reference: F32×F32_quantised matmul (naive triple loop).
    var c_ref = try Matrix.init(testing.allocator, N, N);
    defer c_ref.deinit(testing.allocator);
    @memset(c_ref.data, 0.0);
    for (0..N) |i| for (0..N) |k| for (0..N) |j| {
        c_ref.data[i * N + j] += a.data[i * N + k] * b_requantised.data[k * N + j];
    };

    // BF16-weights matmul against the same quantised B.
    var c_bf16 = try Matrix.init(testing.allocator, N, N);
    defer c_bf16.deinit(testing.allocator);
    try matmulBF16Scalar(&a, b_bf16, N, N, &c_bf16);

    // Should agree bit-exact: same operations, same operands, same order.
    try testing.expect(c_ref.approxEq(&c_bf16, 1e-6));
}

test "matmulBF16SIMD — agrees with matmulBF16Scalar on 32×64×48" {
    const M: usize = 32;
    const K: usize = 64;
    const N: usize = 48;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(7);
    b_f32.fillRandom(11);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_scalar = try Matrix.init(testing.allocator, M, N);
    defer c_scalar.deinit(testing.allocator);
    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);

    try matmulBF16Scalar(&a, b_bf16, K, N, &c_scalar);
    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);

    // Scalar vs SIMD should agree bit-exact (same f32 ops, same order
    // up to the lane-grouping). Use tight tolerance.
    try testing.expect(c_scalar.approxEq(&c_simd, 1e-6));
}

test "matmulBF16SIMD — tail handling (N not a multiple of 8)" {
    const M: usize = 4;
    const K: usize = 8;
    const N: usize = 13; // deliberately not a multiple of 8

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(13);
    b_f32.fillRandom(17);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_scalar = try Matrix.init(testing.allocator, M, N);
    defer c_scalar.deinit(testing.allocator);
    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);

    try matmulBF16Scalar(&a, b_bf16, K, N, &c_scalar);
    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);

    try testing.expect(c_scalar.approxEq(&c_simd, 1e-6));
}

test "matmulBF16 — dimension mismatch is caught" {
    var a = try Matrix.init(testing.allocator, 4, 8);
    defer a.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 4, 16);
    defer c.deinit(testing.allocator);

    const b_bf16 = try testing.allocator.alloc(u16, 6 * 16); // wrong b_rows
    defer testing.allocator.free(b_bf16);

    try testing.expectError(Error.DimensionMismatch, matmulBF16Scalar(&a, b_bf16, 6, 16, &c));
    try testing.expectError(Error.DimensionMismatch, matmulBF16SIMD(&a, b_bf16, 6, 16, &c));
}
