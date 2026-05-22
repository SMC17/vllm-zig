//! Per-layer K/V cache for autoregressive decode.
//!
//! Stores K and V values for the entire sequence-so-far at every
//! layer. During decode, the new query attends against the full
//! cached K/V history.
//!
//! v0.0.2: contiguous fixed-size allocation per layer (no PagedAttention).
//! Memory: 2 * n_layers * max_seq * n_kv_heads * head_dim * 4 bytes
//! For TinyLlama (22 layers, n_kv=4, head_dim=64) at max_seq=2048:
//!   2 * 22 * 2048 * 4 * 64 * 4 = 92 MB. Acceptable.
//! For Llama-3.2-1B (16 layers, n_kv=8, head_dim=64) at max_seq=8192:
//!   2 * 16 * 8192 * 8 * 64 * 4 = 537 MB. Acceptable.
//! Phase 1.5 work: PagedAttention block-table indirection per
//! ARCHITECTURE.md §4.

const std = @import("std");

pub const Error = error{
    SequenceOverflow,
    OutOfMemory,
};

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    n_layers: usize,
    max_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// per layer: [max_seq * n_kv_heads * head_dim]
    keys: [][]f32,
    values: [][]f32,
    /// per layer current sequence length (0 at startup, grows per decode step)
    seq_lens: []usize,

    pub fn init(
        alloc: std.mem.Allocator,
        n_layers: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) !KVCache {
        const per_layer_floats = max_seq * n_kv_heads * head_dim;
        const keys = try alloc.alloc([]f32, n_layers);
        errdefer alloc.free(keys);
        const values = try alloc.alloc([]f32, n_layers);
        errdefer alloc.free(values);
        const seq_lens = try alloc.alloc(usize, n_layers);
        errdefer alloc.free(seq_lens);

        var allocated: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < allocated) : (i += 1) {
                alloc.free(keys[i]);
                alloc.free(values[i]);
            }
        }
        while (allocated < n_layers) : (allocated += 1) {
            keys[allocated] = try alloc.alloc(f32, per_layer_floats);
            values[allocated] = try alloc.alloc(f32, per_layer_floats);
            @memset(keys[allocated], 0.0);
            @memset(values[allocated], 0.0);
            seq_lens[allocated] = 0;
        }

        return .{
            .allocator = alloc,
            .n_layers = n_layers,
            .max_seq = max_seq,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .keys = keys,
            .values = values,
            .seq_lens = seq_lens,
        };
    }

    pub fn deinit(self: *KVCache) void {
        var i: usize = 0;
        while (i < self.n_layers) : (i += 1) {
            self.allocator.free(self.keys[i]);
            self.allocator.free(self.values[i]);
        }
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.seq_lens);
    }

    /// Append `n` new tokens' K/V at layer `layer`.
    /// `k_new` / `v_new` shape: [n, n_kv_heads * head_dim]
    pub fn append(
        self: *KVCache,
        layer: usize,
        k_new: []const f32,
        v_new: []const f32,
        n: usize,
    ) Error!void {
        const row_size = self.n_kv_heads * self.head_dim;
        if (self.seq_lens[layer] + n > self.max_seq) return Error.SequenceOverflow;
        if (k_new.len < n * row_size) return Error.SequenceOverflow;
        if (v_new.len < n * row_size) return Error.SequenceOverflow;
        const start = self.seq_lens[layer] * row_size;
        const len = n * row_size;
        @memcpy(self.keys[layer][start .. start + len], k_new[0..len]);
        @memcpy(self.values[layer][start .. start + len], v_new[0..len]);
        self.seq_lens[layer] += n;
    }

    /// Return the K slice for layer covering all stored tokens so far.
    pub fn getK(self: *const KVCache, layer: usize) []const f32 {
        const row_size = self.n_kv_heads * self.head_dim;
        return self.keys[layer][0 .. self.seq_lens[layer] * row_size];
    }

    pub fn getV(self: *const KVCache, layer: usize) []const f32 {
        const row_size = self.n_kv_heads * self.head_dim;
        return self.values[layer][0 .. self.seq_lens[layer] * row_size];
    }

    pub fn currentLen(self: *const KVCache, layer: usize) usize {
        return self.seq_lens[layer];
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "KVCache init zeroes storage" {
    var c = try KVCache.init(testing.allocator, 2, 16, 4, 8);
    defer c.deinit();
    try testing.expectEqual(@as(usize, 0), c.currentLen(0));
    try testing.expectEqual(@as(usize, 0), c.currentLen(1));
    try testing.expectEqual(@as(usize, 0), c.getK(0).len);
}

test "KVCache append grows seq_len" {
    var c = try KVCache.init(testing.allocator, 1, 16, 2, 4);
    defer c.deinit();
    var k = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // 1 token × (2 heads × 4 dim)
    var v = [_]f32{ 9, 10, 11, 12, 13, 14, 15, 16 };
    try c.append(0, &k, &v, 1);
    try testing.expectEqual(@as(usize, 1), c.currentLen(0));
    try testing.expectEqualSlices(f32, &k, c.getK(0));
    try testing.expectEqualSlices(f32, &v, c.getV(0));

    // Append another
    try c.append(0, &k, &v, 1);
    try testing.expectEqual(@as(usize, 2), c.currentLen(0));
    try testing.expectEqual(@as(usize, 16), c.getK(0).len);
}

test "KVCache append rejects overflow" {
    var c = try KVCache.init(testing.allocator, 1, 2, 1, 4);
    defer c.deinit();
    const k = [_]f32{ 1, 2, 3, 4 };
    const v = [_]f32{ 5, 6, 7, 8 };
    try c.append(0, &k, &v, 1);
    try c.append(0, &k, &v, 1);
    // max_seq is 2; next append should overflow
    try testing.expectError(Error.SequenceOverflow, c.append(0, &k, &v, 1));
}

test "KVCache per-layer independent" {
    var c = try KVCache.init(testing.allocator, 3, 16, 1, 4);
    defer c.deinit();
    const k = [_]f32{ 1, 2, 3, 4 };
    const v = [_]f32{ 5, 6, 7, 8 };
    try c.append(1, &k, &v, 1);
    try testing.expectEqual(@as(usize, 0), c.currentLen(0));
    try testing.expectEqual(@as(usize, 1), c.currentLen(1));
    try testing.expectEqual(@as(usize, 0), c.currentLen(2));
}
