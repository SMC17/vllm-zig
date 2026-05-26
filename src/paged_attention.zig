const std = @import("std");

/// PagedAttention Block Allocator
///
/// Manages a pool of fixed-size KV cache blocks.
/// Replaces contiguous tensor allocation with block-level indirection.
pub const BlockAllocator = struct {
    pub const BLOCK_SIZE = 16; // Number of tokens per block
    
    allocator: std.mem.Allocator,
    total_blocks: usize,
    free_list: std.DynamicBitSet,
    
    pub fn init(allocator: std.mem.Allocator, total_blocks: usize) !BlockAllocator {
        return BlockAllocator{
            .allocator = allocator,
            .total_blocks = total_blocks,
            .free_list = try std.DynamicBitSet.initFull(allocator, total_blocks),
        };
    }
    
    pub fn deinit(self: *BlockAllocator) void {
        self.free_list.deinit();
    }
    
    /// Allocate a block and return its physical index.
    pub fn allocate(self: *BlockAllocator) !usize {
        const index = self.free_list.findFirstSet() orelse return error.OutOfMemory;
        self.free_list.unset(index);
        return index;
    }
    
    /// Free a block index back to the pool.
    pub fn free(self: *BlockAllocator, index: usize) void {
        self.free_list.set(index);
    }
    
    /// Current occupancy rate (0.0 to 1.0)
    pub fn occupancy(self: BlockAllocator) f32 {
        const allocated_count = self.total_blocks - self.free_list.count();
        return @as(f32, @floatFromInt(allocated_count)) / @as(f32, @floatFromInt(self.total_blocks));
    }
};

test "BlockAllocator basic lifecycle" {
    var ba = try BlockAllocator.init(std.testing.allocator, 10);
    defer ba.deinit();
    
    const b1 = try ba.allocate();
    try std.testing.expectEqual(@as(usize, 0), b1);
    try std.testing.expectEqual(@as(f32, 0.1), ba.occupancy());
    
    const b2 = try ba.allocate();
    try std.testing.expectEqual(@as(usize, 1), b2);
    
    ba.free(b1);
    try std.testing.expectEqual(@as(f32, 0.1), ba.occupancy());
    
    const b3 = try ba.allocate();
    try std.testing.expectEqual(@as(usize, 0), b3); // Reused
}
