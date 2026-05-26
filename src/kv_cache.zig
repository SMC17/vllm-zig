const std = @import("std");
const pm_mod = @import("page_manager.zig");
const ba_mod = @import("paged_attention.zig");

pub const Error = error{
    SequenceOverflow,
    OutOfMemory,
};

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    page_manager: *pm_mod.PageManager,
    block_allocator: *ba_mod.BlockAllocator,
    
    /// Block table per layer: maps sequence block index to physical block index.
    /// Shape: [n_layers] std.ArrayList(usize)
    block_tables: []std.ArrayList(usize),
    /// Current sequence length per layer.
    seq_lens: []usize,

    pub fn init(
        alloc: std.mem.Allocator,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        pm: *pm_mod.PageManager,
        ba: *ba_mod.BlockAllocator,
    ) !KVCache {
        const block_tables = try alloc.alloc(std.ArrayList(usize), n_layers);
        errdefer alloc.free(block_tables);
        const seq_lens = try alloc.alloc(usize, n_layers);
        errdefer alloc.free(seq_lens);

        for (0..n_layers) |i| {
            block_tables[i] = .empty;
            seq_lens[i] = 0;
        }

        return .{
            .allocator = alloc,
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .page_manager = pm,
            .block_allocator = ba,
            .block_tables = block_tables,
            .seq_lens = seq_lens,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.block_tables) |*bt| {
            // Free blocks back to allocator before deiniting the list
            for (bt.items) |bi| self.block_allocator.free(bi);
            bt.deinit(self.allocator);
        }
        self.allocator.free(self.block_tables);
        self.allocator.free(self.seq_lens);
    }

    /// Append `n` new tokens' K/V. This handles block allocation on-the-fly.
    pub fn append(
        self: *KVCache,
        layer: usize,
        k_new: []const f32,
        v_new: []const f32,
        n: usize,
    ) !void {
        const block_size = pm_mod.PageManager.BLOCK_SIZE;

        var i: usize = 0;
        while (i < n) : (i += 1) {
            const current_seq_idx = self.seq_lens[layer];
            const block_idx_in_seq = current_seq_idx / block_size;
            const token_in_block = current_seq_idx % block_size;

            // Allocate new block if needed
            if (block_idx_in_seq >= self.block_tables[layer].items.len) {
                const new_physical_idx = try self.block_allocator.allocate();
                try self.block_tables[layer].append(self.allocator, new_physical_idx);
            }

            const physical_idx = self.block_tables[layer].items[block_idx_in_seq];

            for (0..self.n_kv_heads) |h| {
                const head_offset = h * self.head_dim;
                const token_offset = i * (self.n_kv_heads * self.head_dim);
                const src_start = token_offset + head_offset;
                
                const k_src = k_new[src_start .. src_start + self.head_dim];
                const v_src = v_new[src_start .. src_start + self.head_dim];

                const k_dest = self.page_manager.getBlockK(physical_idx, h);
                const v_dest = self.page_manager.getBlockV(physical_idx, h);

                const dst_start = token_in_block * self.head_dim;
                @memcpy(k_dest[dst_start .. dst_start + self.head_dim], k_src);
                @memcpy(v_dest[dst_start .. dst_start + self.head_dim], v_src);
            }

            self.seq_lens[layer] += 1;
        }
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
    var pm = try pm_mod.PageManager.init(testing.allocator, 100, 4, 8);
    defer pm.deinit();
    var ba = try ba_mod.BlockAllocator.init(testing.allocator, 100);
    defer ba.deinit();

    var c = try KVCache.init(testing.allocator, 2, 4, 8, &pm, &ba);
    defer c.deinit();
    try testing.expectEqual(@as(usize, 0), c.currentLen(0));
    try testing.expectEqual(@as(usize, 0), c.currentLen(1));
}

test "KVCache append grows block table" {
    var pm = try pm_mod.PageManager.init(testing.allocator, 100, 2, 4);
    defer pm.deinit();
    var ba = try ba_mod.BlockAllocator.init(testing.allocator, 100);
    defer ba.deinit();

    var c = try KVCache.init(testing.allocator, 1, 2, 4, &pm, &ba);
    defer c.deinit();
    
    // 17 tokens max needed (1 + 16)
    var k = [_]f32{1.0} ** (17 * 8);
    var v = [_]f32{2.0} ** (17 * 8);
    
    // Append 1 token
    try c.append(0, &k, &v, 1);
    try testing.expectEqual(@as(usize, 1), c.currentLen(0));
    try testing.expectEqual(@as(usize, 1), c.block_tables[0].items.len);

    // Append 16 more tokens -> crosses block boundary
    try c.append(0, &k, &v, 16);
    try testing.expectEqual(@as(usize, 17), c.currentLen(0));
    try testing.expectEqual(@as(usize, 2), c.block_tables[0].items.len);
}
