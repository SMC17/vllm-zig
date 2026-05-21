//! Softmax — numerically stable, in-place.
//!
//! Given input x of dim D:
//!   m = max(x)
//!   e[i] = exp(x[i] - m)
//!   y[i] = e[i] / sum(e)
//!
//! Subtracting the max before exp prevents overflow on large logit
//! magnitudes, which is the standard trick.

const std = @import("std");

pub fn softmaxRow(x: []f32) void {
    if (x.len == 0) return;

    // Find max
    var max_v: f32 = x[0];
    var i: usize = 1;
    while (i < x.len) : (i += 1) {
        if (x[i] > max_v) max_v = x[i];
    }

    // exp(x - max) + sum
    var sum: f32 = 0.0;
    i = 0;
    while (i < x.len) : (i += 1) {
        const e = @exp(x[i] - max_v);
        x[i] = e;
        sum += e;
    }

    // Normalize
    const inv = 1.0 / sum;
    i = 0;
    while (i < x.len) : (i += 1) x[i] *= inv;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "softmax — uniform input produces uniform output" {
    var x = [_]f32{ 0, 0, 0, 0 };
    softmaxRow(&x);
    for (x) |v| try testing.expectApproxEqAbs(@as(f32, 0.25), v, 1e-6);
}

test "softmax — output sums to 1" {
    var x = [_]f32{ 1, 2, 3, 4, 5 };
    softmaxRow(&x);
    var sum: f32 = 0.0;
    for (x) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "softmax — largest input gets highest probability" {
    var x = [_]f32{ 1.0, 2.0, 5.0, 0.5 };
    softmaxRow(&x);
    try testing.expect(x[2] > x[0]);
    try testing.expect(x[2] > x[1]);
    try testing.expect(x[2] > x[3]);
}

test "softmax — handles large magnitudes (numerical stability)" {
    var x = [_]f32{ 1000.0, 1001.0, 1002.0 };
    softmaxRow(&x);
    // Should not be NaN / Inf
    for (x) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(std.math.isFinite(v));
    }
    var sum: f32 = 0.0;
    for (x) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "softmax — known result on [-1, 0, 1]" {
    var x = [_]f32{ -1, 0, 1 };
    softmaxRow(&x);
    // expected ~ [0.0900, 0.2447, 0.6652]
    try testing.expectApproxEqAbs(@as(f32, 0.0900), x[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.2447), x[1], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.6652), x[2], 1e-3);
}

test "softmax — empty input is a no-op" {
    var x: [0]f32 = .{};
    softmaxRow(&x);
}
