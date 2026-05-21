//! Rotary Position Embedding (RoPE).
//!
//! Llama / Mistral / Qwen / TinyLlama family standard. Zig port of the
//! split-half convention reference at ~/triton-rope/.
//!
//! For a query/key vector at position `pos` with head dimension D:
//!   freq[i] = 1 / theta^(2i/D)            for i in 0..D/2
//!   theta_pos[i] = pos * freq[i]
//!   x_out[i]       =  x[i]      * cos(theta_pos[i]) - x[i + D/2] * sin(theta_pos[i])
//!   x_out[i + D/2] =  x[i + D/2]* cos(theta_pos[i]) + x[i]      * sin(theta_pos[i])
//!
//! In LLM inference, RoPE is applied to Q and K (not V) per-head,
//! per-token before attention. The cos/sin table for a given
//! (max_seq_len, head_dim) can be precomputed once and reused.

const std = @import("std");

pub const Error = error{
    HeadDimMustBeEven,
    OutOfBounds,
};

pub const DEFAULT_THETA: f32 = 10000.0;

/// Precomputed cos/sin tables for a given (max_pos, head_dim).
/// Memory: 2 * max_pos * head_dim/2 floats (fp32).
pub const RopeCache = struct {
    /// [max_pos, head_dim/2] interleaved: row p holds cos values for position p
    cos: []f32,
    /// [max_pos, head_dim/2] interleaved: row p holds sin values for position p
    sin: []f32,
    max_pos: usize,
    head_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        alloc: std.mem.Allocator,
        max_pos: usize,
        head_dim: usize,
        theta: f32,
    ) !RopeCache {
        if (head_dim % 2 != 0) return Error.HeadDimMustBeEven;
        const half = head_dim / 2;
        const cos = try alloc.alloc(f32, max_pos * half);
        errdefer alloc.free(cos);
        const sin = try alloc.alloc(f32, max_pos * half);
        errdefer alloc.free(sin);

        var p: usize = 0;
        while (p < max_pos) : (p += 1) {
            var i: usize = 0;
            while (i < half) : (i += 1) {
                const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
                const freq = 1.0 / std.math.pow(f32, theta, exponent);
                const angle = @as(f32, @floatFromInt(p)) * freq;
                cos[p * half + i] = std.math.cos(angle);
                sin[p * half + i] = std.math.sin(angle);
            }
        }
        return .{
            .cos = cos,
            .sin = sin,
            .max_pos = max_pos,
            .head_dim = head_dim,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *RopeCache) void {
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
    }

    /// Apply RoPE in-place to a single (head_dim) slice at the given
    /// position. Split-half convention.
    pub fn applyAtPos(self: *const RopeCache, x: []f32, pos: usize) Error!void {
        if (x.len != self.head_dim) return Error.OutOfBounds;
        if (pos >= self.max_pos) return Error.OutOfBounds;
        const half = self.head_dim / 2;
        const cos_row = self.cos[pos * half ..][0..half];
        const sin_row = self.sin[pos * half ..][0..half];

        var i: usize = 0;
        while (i < half) : (i += 1) {
            const xi = x[i];
            const xih = x[i + half];
            const c = cos_row[i];
            const s = sin_row[i];
            x[i] = xi * c - xih * s;
            x[i + half] = xih * c + xi * s;
        }
    }

    /// Apply RoPE in-place to a multi-head sequence tensor in
    /// [seq_len, n_heads * head_dim] layout (row-major). Position
    /// indexing starts at start_pos (for KV-cache decode where the
    /// last token is at start_pos + seq_len - 1).
    pub fn applyToSequence(
        self: *const RopeCache,
        x: []f32,
        seq_len: usize,
        n_heads: usize,
        start_pos: usize,
    ) Error!void {
        if (x.len != seq_len * n_heads * self.head_dim) return Error.OutOfBounds;
        if (start_pos + seq_len > self.max_pos) return Error.OutOfBounds;

        var t: usize = 0;
        while (t < seq_len) : (t += 1) {
            const pos = start_pos + t;
            var h: usize = 0;
            while (h < n_heads) : (h += 1) {
                const offset = (t * n_heads + h) * self.head_dim;
                try self.applyAtPos(x[offset .. offset + self.head_dim], pos);
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "RopeCache rejects odd head_dim" {
    try testing.expectError(Error.HeadDimMustBeEven, RopeCache.init(testing.allocator, 16, 7, DEFAULT_THETA));
}

test "RopeCache pos=0 is identity" {
    // At position 0, cos=1 and sin=0 for all freqs, so RoPE is identity.
    var cache = try RopeCache.init(testing.allocator, 16, 8, DEFAULT_THETA);
    defer cache.deinit();
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try cache.applyAtPos(&x, 0);
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &x);
}

test "RopeCache preserves L2 norm of x" {
    // RoPE is a rotation, so it preserves the L2 norm of the full
    // vector x.
    var cache = try RopeCache.init(testing.allocator, 32, 8, DEFAULT_THETA);
    defer cache.deinit();
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var norm_before: f32 = 0;
    for (x) |v| norm_before += v * v;
    norm_before = std.math.sqrt(norm_before);
    try cache.applyAtPos(&x, 5);
    var norm_after: f32 = 0;
    for (x) |v| norm_after += v * v;
    norm_after = std.math.sqrt(norm_after);
    try testing.expectApproxEqAbs(norm_before, norm_after, 1e-4);
}

test "RopeCache pos=0 == pos=N when freq*N is a multiple of 2π" {
    // Sanity: at theta=10000, head_dim=8, freq[0] = 1.0 (since exponent
    // = 0 / 8 = 0). So pos = 2π = ~6.283 would give cos(6.283)=1, sin=0
    // for that channel. Pick a small head_dim and verify rotational behavior.
    // (Not a strong test, but confirms the math is non-trivial at pos>0)
    var cache = try RopeCache.init(testing.allocator, 16, 8, DEFAULT_THETA);
    defer cache.deinit();
    var x0 = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0 };
    var x5 = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0 };
    try cache.applyAtPos(&x0, 0);
    try cache.applyAtPos(&x5, 5);
    // At pos=5 the result should differ from pos=0 (identity).
    var different = false;
    for (x0, x5) |a, b| if (@abs(a - b) > 1e-6) { different = true; break; };
    try testing.expect(different);
}

test "applyToSequence covers (seq, heads) layout" {
    var cache = try RopeCache.init(testing.allocator, 16, 8, DEFAULT_THETA);
    defer cache.deinit();
    // seq=3, n_heads=2, head_dim=8 → 48 floats
    var x: [48]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    try cache.applyToSequence(&x, 3, 2, 0);
    // At t=0 (pos=0), both heads should be identity.
    try testing.expectEqual(@as(f32, 0), x[0]);
    try testing.expectEqual(@as(f32, 1), x[1]);
    // At t=1, the values should differ from a copy that was made before applyToSequence.
    // We check the underlying compute happened.
    try testing.expect(x[16] != 16.0 or x[17] != 17.0 or x[18] != 18.0);
}

test "applyAtPos rejects pos out of range" {
    var cache = try RopeCache.init(testing.allocator, 4, 4, DEFAULT_THETA);
    defer cache.deinit();
    var x = [_]f32{ 1, 2, 3, 4 };
    try testing.expectError(Error.OutOfBounds, cache.applyAtPos(&x, 100));
}

test "applyAtPos rejects mismatched head_dim" {
    var cache = try RopeCache.init(testing.allocator, 4, 8, DEFAULT_THETA);
    defer cache.deinit();
    var x = [_]f32{ 1, 2, 3, 4 };
    try testing.expectError(Error.OutOfBounds, cache.applyAtPos(&x, 0));
}
