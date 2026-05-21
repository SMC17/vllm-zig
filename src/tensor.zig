//! Minimal Tensor abstraction for Phase 1.
//!
//! v0.0.1 scope:
//!   - f32 storage, row-major layout
//!   - Owned slice ([]f32) backing — no mmap, no GPU memory, no
//!     borrowed views (yet)
//!   - 2D matrix view by default; ndim==2 matrices are the primary
//!     working unit (matmul, RoPE, attention all operate on 2D)
//!
//! This is the type that every kernel takes as input and produces as
//! output. Future GPU / dtype variants (f16, bf16, q8) will introduce
//! Tagged-union or comptime-dtype variants of this type without
//! changing the kernel signatures.

const std = @import("std");

pub const DType = enum {
    f32,
    // Reserved for future:
    // f16, bf16, f8_e4m3, f8_e5m2, q8_0, q4_k_m
};

/// A 2D matrix in row-major layout. Phase 1 default working unit.
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    /// Owned. Length must equal rows*cols.
    data: []f32,

    pub fn init(alloc: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try alloc.alloc(f32, rows * cols);
        @memset(data, 0.0);
        return .{ .rows = rows, .cols = cols, .data = data };
    }

    pub fn deinit(self: *Matrix, alloc: std.mem.Allocator) void {
        alloc.free(self.data);
    }

    pub inline fn at(self: *const Matrix, r: usize, c: usize) f32 {
        return self.data[r * self.cols + c];
    }

    pub inline fn set(self: *Matrix, r: usize, c: usize, v: f32) void {
        self.data[r * self.cols + c] = v;
    }

    pub inline fn row(self: *const Matrix, r: usize) []const f32 {
        return self.data[r * self.cols ..][0..self.cols];
    }

    pub inline fn rowMut(self: *Matrix, r: usize) []f32 {
        return self.data[r * self.cols ..][0..self.cols];
    }

    /// Fill all elements with a constant.
    pub fn fill(self: *Matrix, v: f32) void {
        for (self.data) |*x| x.* = v;
    }

    /// Fill from a deterministic PRNG seed. For tests + fixtures.
    pub fn fillRandom(self: *Matrix, seed: u64) void {
        var prng: std.Random.DefaultPrng = .init(seed);
        const rng = prng.random();
        for (self.data) |*x| x.* = rng.float(f32) - 0.5;
    }

    /// Approximate equality, useful for testing two implementations
    /// (scalar vs SIMD, naive vs blocked, etc.) of the same kernel.
    pub fn approxEq(self: *const Matrix, other: *const Matrix, rel_tol: f32) bool {
        if (self.rows != other.rows or self.cols != other.cols) return false;
        for (self.data, other.data) |a, b| {
            const diff = @abs(a - b);
            const m = @max(@abs(a), @abs(b));
            if (m > 1e-6 and diff / m > rel_tol) return false;
            if (m <= 1e-6 and diff > 1e-6) return false;
        }
        return true;
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Matrix init zeroes" {
    var m = try Matrix.init(testing.allocator, 4, 8);
    defer m.deinit(testing.allocator);
    for (m.data) |v| try testing.expectEqual(@as(f32, 0.0), v);
}

test "Matrix at + set" {
    var m = try Matrix.init(testing.allocator, 3, 5);
    defer m.deinit(testing.allocator);
    m.set(1, 2, 42.0);
    try testing.expectEqual(@as(f32, 42.0), m.at(1, 2));
    try testing.expectEqual(@as(f32, 0.0), m.at(0, 0));
}

test "Matrix row mutation" {
    var m = try Matrix.init(testing.allocator, 2, 4);
    defer m.deinit(testing.allocator);
    const r = m.rowMut(0);
    for (r) |*v| v.* = 7.0;
    try testing.expectEqual(@as(f32, 7.0), m.at(0, 0));
    try testing.expectEqual(@as(f32, 7.0), m.at(0, 3));
    try testing.expectEqual(@as(f32, 0.0), m.at(1, 0));
}

test "Matrix approxEq" {
    var a = try Matrix.init(testing.allocator, 2, 2);
    defer a.deinit(testing.allocator);
    var b = try Matrix.init(testing.allocator, 2, 2);
    defer b.deinit(testing.allocator);
    a.fillRandom(42);
    b.fillRandom(42);
    try testing.expect(a.approxEq(&b, 1e-6));
    b.set(0, 0, b.at(0, 0) + 1.0);
    try testing.expect(!a.approxEq(&b, 1e-3));
}
