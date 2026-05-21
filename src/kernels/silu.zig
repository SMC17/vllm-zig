//! SiLU (Swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! Used in Llama-family MLPs. In Llama the MLP is:
//!   y = down(silu(gate(x)) * up(x))

const std = @import("std");

pub inline fn siluScalar(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn silu(x: []f32) void {
    var i: usize = 0;
    while (i < x.len) : (i += 1) {
        x[i] = siluScalar(x[i]);
    }
}

/// Multiplied-pair form used in Llama MLP:
///   out[i] = silu(gate[i]) * up[i]
pub fn siluMul(gate_silu: []f32, up: []const f32) error{DimensionMismatch}!void {
    if (gate_silu.len != up.len) return error.DimensionMismatch;
    var i: usize = 0;
    while (i < gate_silu.len) : (i += 1) {
        gate_silu[i] = siluScalar(gate_silu[i]) * up[i];
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "silu(0) == 0" {
    try testing.expectApproxEqAbs(@as(f32, 0.0), siluScalar(0.0), 1e-6);
}

test "silu(large positive) ~ x" {
    // sigmoid(10) ≈ 1, so silu(10) ≈ 10
    try testing.expectApproxEqAbs(@as(f32, 10.0), siluScalar(10.0), 1e-3);
}

test "silu(large negative) ~ 0" {
    try testing.expect(@abs(siluScalar(-10.0)) < 1e-3);
}

test "silu vec applies in place" {
    var x = [_]f32{ 0.0, 1.0, -1.0 };
    silu(&x);
    try testing.expectApproxEqAbs(@as(f32, 0.0), x[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.7310585), x[1], 1e-5);
    // silu(-1) ≈ -0.2689
    try testing.expectApproxEqAbs(@as(f32, -0.2689414), x[2], 1e-5);
}

test "siluMul dimension mismatch" {
    var g = [_]f32{ 1, 2 };
    const u = [_]f32{ 1, 2, 3 };
    try testing.expectError(error.DimensionMismatch, siluMul(&g, &u));
}

test "siluMul element-wise" {
    var g = [_]f32{ 0.0, 1.0, -1.0 };
    const u = [_]f32{ 2.0, 3.0, 4.0 };
    try siluMul(&g, &u);
    try testing.expectApproxEqAbs(@as(f32, 0.0), g[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.7310585 * 3.0), g[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.2689414 * 4.0), g[2], 1e-4);
}
