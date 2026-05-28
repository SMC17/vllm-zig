//! Physical Page Manager — backing store for PagedAttention KV cache.
//!
//! Owns a contiguous, 64-byte-aligned f32 buffer for K and V tensors,
//! laid out per-block × per-head × per-token. Two parallel regions:
//! a K plane and a V plane, each of size
//!   num_blocks × n_kv_heads × BLOCK_SIZE × head_dim
//!
//! Per-block per-head views are returned as `[]f32` of length
//! BLOCK_SIZE * head_dim, so callers can index by token-in-block * head_dim.
//!
//! Layout choice: K and V are split into two separate planes (rather than
//! interleaved per-token) so the attention kernel can stream K rows for the
//! score matmul and then V rows for the aggregation without crossing the
//! other tensor.

const std = @import("std");

pub const PageManager = struct {
    allocator: std.mem.Allocator,
    /// Backing storage. First half is K plane, second half is V plane.
    buffer: []align(64) f32,
    total_blocks: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Number of f32 elements per (block, head) slot.
    per_head_per_block: usize,
    /// Number of f32 elements per block (across all heads).
    per_block: usize,
    /// Number of f32 elements per plane (K plane = V plane size).
    per_plane: usize,

    /// Tokens per block. Keeping this fixed (and matching BlockAllocator's
    /// BLOCK_SIZE) lets attention iterate in constant strides.
    pub const BLOCK_SIZE: usize = 16;

    pub fn init(
        allocator: std.mem.Allocator,
        num_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) !PageManager {
        const per_head_per_block = BLOCK_SIZE * head_dim;
        const per_block = n_kv_heads * per_head_per_block;
        const per_plane = num_blocks * per_block;
        const total_elements = 2 * per_plane; // K plane + V plane

        const alignment = comptime @as(std.mem.Alignment, @enumFromInt(std.math.log2(@as(usize, 64))));
        const buf = try allocator.alignedAlloc(f32, alignment, total_elements);
        @memset(buf, 0);

        return .{
            .allocator = allocator,
            .buffer = buf,
            .total_blocks = num_blocks,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .per_head_per_block = per_head_per_block,
            .per_block = per_block,
            .per_plane = per_plane,
        };
    }

    pub fn deinit(self: *PageManager) void {
        self.allocator.free(self.buffer);
    }

    /// View of the K storage for a single (block, kv_head) pair.
    /// Length: BLOCK_SIZE * head_dim. Layout within: row-major [token, dim].
    pub fn getBlockK(self: *PageManager, block_id: usize, kv_head: usize) []f32 {
        const offset = block_id * self.per_block + kv_head * self.per_head_per_block;
        return self.buffer[offset .. offset + self.per_head_per_block];
    }

    /// View of the V storage for a single (block, kv_head) pair.
    pub fn getBlockV(self: *PageManager, block_id: usize, kv_head: usize) []f32 {
        const offset = self.per_plane + block_id * self.per_block + kv_head * self.per_head_per_block;
        return self.buffer[offset .. offset + self.per_head_per_block];
    }

    /// Raw view of a whole block (both K + V, all heads). Used by the engine
    /// scaffold to touch pages for page-in. Length: 2 * per_block.
    pub fn getBlock(self: *PageManager, block_id: usize) []f32 {
        // Concatenated K-block followed by V-block aren't contiguous in
        // memory (they live in separate planes), so this returns the K
        // portion only — sufficient for the existing engine scaffold,
        // which uses it as a page-touch primitive.
        const offset = block_id * self.per_block;
        return self.buffer[offset .. offset + self.per_block];
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "PageManager — aligned allocation + size correctness" {
    var pm = try PageManager.init(testing.allocator, 10, 4, 64);
    defer pm.deinit();

    const ptr_val = @intFromPtr(pm.buffer.ptr);
    try testing.expect(ptr_val % 64 == 0);

    // 10 blocks × 4 heads × 16 tokens × 64 dim × 2 (K+V) = 81,920 f32
    try testing.expectEqual(@as(usize, 10 * 4 * 16 * 64 * 2), pm.buffer.len);
}

test "PageManager — K and V are disjoint, independently addressable" {
    var pm = try PageManager.init(testing.allocator, 4, 2, 8);
    defer pm.deinit();

    const k0 = pm.getBlockK(0, 0);
    const v0 = pm.getBlockV(0, 0);
    try testing.expectEqual(@as(usize, 16 * 8), k0.len);
    try testing.expectEqual(@as(usize, 16 * 8), v0.len);

    k0[0] = 11.0;
    v0[0] = 22.0;
    try testing.expectEqual(@as(f32, 11.0), k0[0]);
    try testing.expectEqual(@as(f32, 22.0), v0[0]);

    // Different heads of same block must not alias each other
    const k0h1 = pm.getBlockK(0, 1);
    try testing.expectEqual(@as(f32, 0.0), k0h1[0]);
    k0h1[0] = 33.0;
    try testing.expectEqual(@as(f32, 11.0), k0[0]);
}
