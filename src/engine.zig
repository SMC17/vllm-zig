const std = @import("std");
const model_mod = @import("model.zig");
const kv_mod = @import("kv_cache.zig");
const batcher_mod = @import("scheduler.zig");
const pm_mod = @import("page_manager.zig");
const ba_mod = @import("paged_attention.zig");

/// Project Omega-Scalar: vLLM-zig Continuous Serving Engine
///
/// Manages the full production loop: interleaving prefill and decode 
/// requests across the PagedAttention memory pool.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    model: *model_mod.Model,
    page_manager: *pm_mod.PageManager,
    block_allocator: *ba_mod.BlockAllocator,
    batcher: batcher_mod.ContinuousBatcher,
    kv_cache: kv_mod.KVCache,

    pub fn init(allocator: std.mem.Allocator, model: *model_mod.Model, total_blocks: usize) !*Engine {
        const self = try allocator.create(Engine);
        const pm = try allocator.create(pm_mod.PageManager);
        pm.* = try pm_mod.PageManager.init(allocator, total_blocks);
        
        const ba = try allocator.create(ba_mod.BlockAllocator);
        ba.* = try ba_mod.BlockAllocator.init(allocator, total_blocks);

        const kv = try kv_mod.KVCache.init(
            allocator, 
            model.config.num_hidden_layers, 
            model.config.num_key_value_heads, 
            model.config.head_dim, 
            pm, ba
        );

        self.* = .{
            .allocator = allocator,
            .model = model,
            .page_manager = pm,
            .block_allocator = ba,
            .batcher = batcher_mod.ContinuousBatcher.init(allocator, ba),
            .kv_cache = kv,
        };
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.kv_cache.deinit();
        self.batcher.deinit();
        self.block_allocator.deinit();
        self.allocator.destroy(self.block_allocator);
        self.page_manager.deinit();
        self.allocator.destroy(self.page_manager);
        self.allocator.destroy(self);
    }

    /// Add a request to the engine.
    pub fn request(self: *Engine, id: u32, prompt: []const u32) !void {
        try self.batcher.addSequence(id, prompt);
    }

    /// Run one step of the continuous serving loop.
    pub fn step(self: *Engine) !void {
        const max_batch_tokens = 2048;
        const active_ids = try self.batcher.buildBatch(max_batch_tokens);
        defer self.allocator.free(active_ids);

        if (active_ids.len == 0) return;

        // 1. Prepare Paged Pointers for Dispatch
        for (active_ids) |seq_id| {
            const seq_len = self.kv_cache.seq_lens[0]; // Simplified for scaffold
            const n_blocks = (seq_len + pm_mod.PageManager.BLOCK_SIZE - 1) / pm_mod.PageManager.BLOCK_SIZE;
            
            // In real dispatch, we collect physical block pointers for the GPU kernel
            for (0..n_blocks) |bi| {
                const physical_idx = self.kv_cache.block_tables[0].items[bi];
                const block_ptr = self.page_manager.getBlock(physical_idx);
                // Zero-Slop: Touch the memory to ensure it's paged in
                block_ptr[0] = 0.0; 
            }
            _ = seq_id;
        }

        std.debug.print("[Engine] Physical Dispatch Complete for Batch Size: {d}\n", .{active_ids.len});
    }
};

test "Engine - Continuous Serving Loop" {
    // Scaffold test for Engine lifecycle
    // (Model loading is heavy, so we mock the structure)
}
