const std = @import("std");
const ba_mod = @import("paged_attention.zig");
const BlockAllocator = ba_mod.BlockAllocator;

/// Project Omega-Scalar: Continuous Batching Scheduler
///
/// Interleaves prefill and decode sequences in a single batch to maximize 
/// GPU utilization and minimize latency.
pub const ContinuousBatcher = struct {
    pub const Sequence = struct {
        id: u32,
        tokens: []const u32,
        is_prefill: bool,
        blocks: std.ArrayList(usize),
    };

    allocator: std.mem.Allocator,
    block_allocator: *BlockAllocator,
    running_sequences: std.ArrayList(Sequence),

    pub fn init(allocator: std.mem.Allocator, ba: *BlockAllocator) ContinuousBatcher {
        return .{
            .allocator = allocator,
            .block_allocator = ba,
            .running_sequences = .empty,
        };
    }

    pub fn deinit(self: *ContinuousBatcher) void {
        for (self.running_sequences.items) |*seq| {
            seq.blocks.deinit(self.allocator);
        }
        self.running_sequences.deinit(self.allocator);
    }

    /// Add a new sequence to the running pool.
    pub fn addSequence(self: *ContinuousBatcher, id: u32, tokens: []const u32) !void {
        var blocks: std.ArrayList(usize) = .empty;
        
        // Initial block allocation
        const n_blocks = (tokens.len + BlockAllocator.BLOCK_SIZE - 1) / BlockAllocator.BLOCK_SIZE;
        for (0..n_blocks) |_| {
            try blocks.append(self.allocator, try self.block_allocator.allocate());
        }

        try self.running_sequences.append(self.allocator, .{
            .id = id,
            .tokens = tokens,
            .is_prefill = true,
            .blocks = blocks,
        });
    }

    /// Build the next batch, prioritizing decode and fitting prefill in slack.
    pub fn buildBatch(self: *ContinuousBatcher, max_tokens: usize) ![]u32 {
        var batch_ids: std.ArrayList(u32) = .empty;
        var current_tokens: usize = 0;

        // 1. Prioritize Decode (Batch-1 tokens per sequence)
        for (self.running_sequences.items) |*seq| {
            if (!seq.is_prefill) {
                if (current_tokens + 1 <= max_tokens) {
                    try batch_ids.append(self.allocator, seq.id);
                    current_tokens += 1;
                }
            }
        }

        // 2. Fit Prefill in remaining slack
        for (self.running_sequences.items) |*seq| {
            if (seq.is_prefill) {
                if (current_tokens + seq.tokens.len <= max_tokens) {
                    try batch_ids.append(self.allocator, seq.id);
                    current_tokens += seq.tokens.len;
                    seq.is_prefill = false; // Transition to decode after first pass
                }
            }
        }

        return try batch_ids.toOwnedSlice(self.allocator);
    }
};

test "Continuous Batching - Interleaving" {
    var ba = try BlockAllocator.init(std.testing.allocator, 100);
    defer ba.deinit();

    var batcher = ContinuousBatcher.init(std.testing.allocator, &ba);
    defer batcher.deinit();

    // Seq 1: Prefill (8 tokens)
    try batcher.addSequence(1, &[_]u32{1} ** 8);
    // Seq 2: Prefill (4 tokens)
    try batcher.addSequence(2, &[_]u32{2} ** 4);

    // Build batch with limit 10
    const batch1 = try batcher.buildBatch(10);
    defer std.testing.allocator.free(batch1);
    
    // Should fit Seq 1 (8 tokens) but not Seq 2 (would be 12 total)
    try std.testing.expectEqual(@as(usize, 1), batch1.len);
    try std.testing.expectEqual(@as(u32, 1), batch1[0]);

    // Build next batch with limit 10
    const batch2 = try batcher.buildBatch(10);
    defer std.testing.allocator.free(batch2);
    
    // Now Seq 1 is in decode (1 token) and Seq 2 fits in prefill (4 tokens)
    // Total tokens = 5
    try std.testing.expectEqual(@as(usize, 2), batch2.len);
}
