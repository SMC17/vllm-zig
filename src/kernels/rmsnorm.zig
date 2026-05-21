//! RMSNorm — Root Mean Square Layer Normalization, as used by
//! Llama-family models.
//!
//! Given input x of dim D and weight w of dim D:
//!   y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
//!
//! Llama uses RMSNorm in place of LayerNorm. Mean is taken across the
//! feature dimension (last dim), with eps typically 1e-5 or 1e-6
//! depending on the architecture.

const std = @import("std");
const Matrix = @import("../tensor.zig").Matrix;

pub const Error = error{
    DimensionMismatch,
};

/// In-place RMSNorm on each row of `x` against the per-feature weight
/// vector `w` (length == x.cols). eps is added inside the sqrt.
pub fn rmsnormRow(x: []f32, w: []const f32, eps: f32) Error!void {
    if (x.len != w.len) return Error.DimensionMismatch;
    const D = x.len;

    // Compute sum-of-squares with @Vector(8, f32) where possible.
    const Vec = @Vector(8, f32);
    const LANE = 8;
    const tail = D - (D % LANE);

    var ss_vec: Vec = @splat(0.0);
    var i: usize = 0;
    while (i < tail) : (i += LANE) {
        const v: Vec = x[i..][0..LANE].*;
        ss_vec += v * v;
    }
    var ss: f32 = @reduce(.Add, ss_vec);
    while (i < D) : (i += 1) ss += x[i] * x[i];

    const rms = @sqrt(ss / @as(f32, @floatFromInt(D)) + eps);
    const scale = 1.0 / rms;

    const scale_vec: Vec = @splat(scale);
    i = 0;
    while (i < tail) : (i += LANE) {
        const xv: Vec = x[i..][0..LANE].*;
        const wv: Vec = w[i..][0..LANE].*;
        x[i..][0..LANE].* = xv * scale_vec * wv;
    }
    while (i < D) : (i += 1) x[i] = x[i] * scale * w[i];
}

/// Apply RMSNorm to every row of a [M, D] matrix using a shared
/// weight vector `w` of length D.
pub fn rmsnorm(m: *Matrix, w: []const f32, eps: f32) Error!void {
    if (m.cols != w.len) return Error.DimensionMismatch;
    var r: usize = 0;
    while (r < m.rows) : (r += 1) {
        try rmsnormRow(m.rowMut(r), w, eps);
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "rmsnorm — known result on small vector" {
    // x = [1, 2, 3, 4]; w = [1, 1, 1, 1]; eps = 0
    // ss = 1+4+9+16 = 30; rms = sqrt(30/4) ≈ 2.7386
    // y = x / 2.7386 ≈ [0.3651, 0.7303, 1.0954, 1.4606]
    var x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 1, 1, 1 };
    try rmsnormRow(&x, &w, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.3651), x[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.7303), x[1], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 1.0954), x[2], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 1.4606), x[3], 1e-3);
}

test "rmsnorm — uniform weight = 2 doubles output" {
    var x_a = [_]f32{ 1, 2, 3, 4 };
    var x_b = [_]f32{ 1, 2, 3, 4 };
    const w_unit = [_]f32{ 1, 1, 1, 1 };
    const w_two = [_]f32{ 2, 2, 2, 2 };
    try rmsnormRow(&x_a, &w_unit, 0.0);
    try rmsnormRow(&x_b, &w_two, 0.0);
    for (x_a, x_b) |a, b| try testing.expectApproxEqAbs(2.0 * a, b, 1e-5);
}

test "rmsnorm — zero input survives eps" {
    var x = [_]f32{ 0, 0, 0, 0 };
    const w = [_]f32{ 1, 1, 1, 1 };
    try rmsnormRow(&x, &w, 1e-6);
    // sqrt(0 + 1e-6) ≈ 1e-3; 0 / 1e-3 = 0
    for (x) |v| try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}

test "rmsnorm — dim mismatch" {
    var x = [_]f32{ 1, 2, 3 };
    const w = [_]f32{ 1, 1 };
    try testing.expectError(Error.DimensionMismatch, rmsnormRow(&x, &w, 0.0));
}

test "rmsnorm — matrix interface applies per-row" {
    var m = try Matrix.init(testing.allocator, 3, 8);
    defer m.deinit(testing.allocator);
    m.fillRandom(42);

    const w = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    try rmsnorm(&m, &w, 1e-6);

    // After RMSNorm with unit weights, each row should have RMS ≈ 1
    var r: usize = 0;
    while (r < m.rows) : (r += 1) {
        var ss: f32 = 0.0;
        for (m.row(r)) |v| ss += v * v;
        const rms = @sqrt(ss / 8.0);
        try testing.expectApproxEqAbs(@as(f32, 1.0), rms, 1e-3);
    }
}
