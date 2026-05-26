//! Full Llama-family Model — loaded from HF safetensors + config.json,
//! runs autoregressive forward pass with KV cache.
//!
//! Architecture: embed → N × (transformer block) → final norm → lm_head → logits
//!
//! v0.0.2 scope:
//!   - Loads BF16 weights from a safetensors-zig SafeTensors handle,
//!     converts to F32 in-memory (double the working set)
//!   - Multi-block forward pass with KV cache for autoregressive decode
//!   - Greedy / temperature / top-K sampling
//!   - Tested against TinyLlama-1.1B-Chat-v1.0
//!
//! What's NOT here yet (v0.0.3+):
//!   - Streaming output (one-shot only)
//!   - BF16 storage end-to-end (intermediate work happens in F32)
//!   - Multi-thread inner kernels
//!   - PagedAttention (KV cache is contiguous fixed-size)
//!   - Speculative decoding

const std = @import("std");
const safetensors = @import("safetensors");
const Matrix = @import("tensor.zig").Matrix;
const matmul = @import("kernels/matmul.zig");
const rmsnorm = @import("kernels/rmsnorm.zig");
const rope = @import("kernels/rope.zig");
const attention = @import("kernels/attention.zig");
const silu = @import("kernels/silu.zig");
const weights_mod = @import("weights.zig");
const kv_cache_mod = @import("kv_cache.zig");
const thread_pool = @import("thread_pool.zig");
const gpu_context_mod = @import("gpu_context.zig");

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    /// Whether the LM head is tied to the input embedding (Llama: typically
    /// NOT tied for chat-tuned variants; TinyLlama unties them, so we always
    /// load lm_head.weight separately).
    tie_word_embeddings: bool = false,

    /// Parse a config.json from the standard HF format.
    pub fn fromJson(alloc: std.mem.Allocator, json_bytes: []const u8) !Config {
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const aa = arena.allocator();

        const parsed = std.json.parseFromSliceLeaky(std.json.Value, aa, json_bytes, .{}) catch
            return error.InvalidConfigJson;
        if (parsed != .object) return error.InvalidConfigJson;
        const root = parsed.object;

        const head_dim = if (root.get("head_dim")) |v| (if (v == .integer) @as(usize, @intCast(v.integer)) else 0) else 0;
        const num_q = readUsize(root, "num_attention_heads") orelse return error.MissingField;
        const hidden = readUsize(root, "hidden_size") orelse return error.MissingField;
        return .{
            .vocab_size = readUsize(root, "vocab_size") orelse return error.MissingField,
            .hidden_size = hidden,
            .intermediate_size = readUsize(root, "intermediate_size") orelse return error.MissingField,
            .num_hidden_layers = readUsize(root, "num_hidden_layers") orelse return error.MissingField,
            .num_attention_heads = num_q,
            .num_key_value_heads = readUsize(root, "num_key_value_heads") orelse num_q,
            .head_dim = if (head_dim > 0) head_dim else hidden / num_q,
            .rms_norm_eps = readF32(root, "rms_norm_eps") orelse 1e-5,
            .rope_theta = readF32(root, "rope_theta") orelse 10000.0,
            .max_position_embeddings = readUsize(root, "max_position_embeddings") orelse 2048,
            .tie_word_embeddings = readBool(root, "tie_word_embeddings") orelse false,
        };
    }
};

fn readUsize(o: std.json.ObjectMap, k: []const u8) ?usize {
    if (o.get(k)) |v| {
        if (v == .integer and v.integer >= 0) return @intCast(v.integer);
    }
    return null;
}

fn readF32(o: std.json.ObjectMap, k: []const u8) ?f32 {
    if (o.get(k)) |v| {
        if (v == .float) return @floatCast(v.float);
        if (v == .integer) return @floatFromInt(v.integer);
    }
    return null;
}

fn readBool(o: std.json.ObjectMap, k: []const u8) ?bool {
    if (o.get(k)) |v| {
        if (v == .bool) return v.bool;
    }
    return null;
}

pub const LayerWeights = struct {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
    norm_input: []f32,
    norm_post_attn: []f32,

    fn deinit(self: *LayerWeights, alloc: std.mem.Allocator) void {
        self.w_q.deinit(alloc);
        self.w_k.deinit(alloc);
        self.w_v.deinit(alloc);
        self.w_o.deinit(alloc);
        self.w_gate.deinit(alloc);
        self.w_up.deinit(alloc);
        self.w_down.deinit(alloc);
        alloc.free(self.norm_input);
        alloc.free(self.norm_post_attn);
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    config: Config,
    embed: Matrix,
    layers: []LayerWeights,
    final_norm: []f32,
    lm_head: Matrix,
    rope_cache: rope.RopeCache,
    /// Number of threads used by multi-thread kernels. 0 = single-thread.
    n_threads: usize = 0,
    /// Optional persistent thread pool. When set, forward() routes matmul
    /// calls through thread_pool.Pool.matmulAuto, eliminating the per-call
    /// spawn overhead that dominated decode in v0.0.4 (~30-100us spawn
    /// per worker × ~110 matmul calls/token). The Model does not own
    /// the pool — caller is responsible for init/deinit. Lifetime must
    /// outlast all forward() calls.
    pool: ?*thread_pool.Pool = null,
    gpu_ctx: ?*gpu_context_mod.GPUContext = null,

    /// Configure the matmul thread count for subsequent forward passes.
    pub fn setThreads(self: *Model, n: usize) void {
        self.n_threads = n;
    }

    /// Attach a persistent thread pool. Caller retains ownership; the pool
    /// must outlive every subsequent forward() call. n_threads should match
    /// the pool's worker count for best behavior.
    pub fn setThreadPool(self: *Model, pool: *thread_pool.Pool) void {
        self.pool = pool;
    }

    inline fn matmulDispatch(self: *const Model, a: *const Matrix, b: *const Matrix, c: *Matrix) !void {
        if (self.gpu_ctx) |ctx| {
            // JIT-compile and Launch Substrate-Native PTX
            const ptx = try ctx.compileMatmul(a.rows, a.cols, b.cols);
            return ctx.launch(ptx);
        }
        if (self.pool) |p| {
            return p.matmulAuto(a, b, c, self.n_threads);
        }
        return matmul.matmulSIMDAuto(a, b, c, self.n_threads);
    }

    pub fn deinit(self: *Model) void {
        self.embed.deinit(self.allocator);
        for (self.layers) |*l| l.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.final_norm);
        self.lm_head.deinit(self.allocator);
        self.rope_cache.deinit();
    }

    /// Load a Llama-family model from an already-opened safetensors handle.
    /// Convention: HF stores projections as [out, in], we transpose on load
    /// so matmul gets x [seq, in] @ W [in, out] = result [seq, out].
    pub fn loadFromSafeTensors(
        alloc: std.mem.Allocator,
        st: *safetensors.SafeTensors,
        config: Config,
    ) !Model {
        var embed = try weights_mod.loadMatrixBF16(alloc, st, "model.embed_tokens.weight");
        errdefer embed.deinit(alloc);

        const layers = try alloc.alloc(LayerWeights, config.num_hidden_layers);
        errdefer alloc.free(layers);

        var loaded_layers: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < loaded_layers) : (i += 1) layers[i].deinit(alloc);
        }

        var name_buf: [256]u8 = undefined;
        while (loaded_layers < config.num_hidden_layers) : (loaded_layers += 1) {
            const i = loaded_layers;
            layers[i] = .{
                .w_q = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", .{i})),
                .w_k = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_proj.weight", .{i})),
                .w_v = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.v_proj.weight", .{i})),
                .w_o = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.o_proj.weight", .{i})),
                .w_gate = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", .{i})),
                .w_up = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.up_proj.weight", .{i})),
                .w_down = try weights_mod.loadMatrixBF16Transposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.down_proj.weight", .{i})),
                .norm_input = try weights_mod.loadVectorBF16(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.input_layernorm.weight", .{i})),
                .norm_post_attn = try weights_mod.loadVectorBF16(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.post_attention_layernorm.weight", .{i})),
            };
        }

        const final_norm = try weights_mod.loadVectorBF16(alloc, st, "model.norm.weight");
        errdefer alloc.free(final_norm);

        var lm_head = if (config.tie_word_embeddings)
            // For tied-weight models, the lm_head is the embedding transposed.
            try transposeMatrix(alloc, &embed)
        else
            try weights_mod.loadMatrixBF16Transposed(alloc, st, "lm_head.weight");
        errdefer lm_head.deinit(alloc);

        const rope_cache = try rope.RopeCache.init(
            alloc,
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
        );

        return .{
            .allocator = alloc,
            .config = config,
            .embed = embed,
            .layers = layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .rope_cache = rope_cache,
        };
    }

    /// Forward pass over the given token IDs. Returns logits for the LAST
    /// token (shape [vocab_size]). Caller frees.
    ///
    /// If `kv_cache` is non-null, K/V are written to it for each layer.
    /// `start_pos` is the absolute starting position for RoPE (i.e. the
    /// length of the cached sequence before this call).
    pub fn forward(
        self: *const Model,
        alloc: std.mem.Allocator,
        token_ids: []const u32,
        kv_cache: ?*kv_cache_mod.KVCache,
        start_pos: usize,
    ) ![]f32 {
        const cfg = self.config;
        const seq = token_ids.len;
        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // 1. Embed: gather rows from embed table
        var x = try Matrix.init(alloc, seq, cfg.hidden_size);
        defer x.deinit(alloc);
        for (token_ids, 0..) |id, t| {
            if (id >= cfg.vocab_size) return error.TokenIdOutOfRange;
            const src = self.embed.row(id);
            @memcpy(x.rowMut(t), src);
        }

        // 2. Multi-block forward
        var residual = try Matrix.init(alloc, seq, cfg.hidden_size);
        defer residual.deinit(alloc);

        for (self.layers, 0..) |layer, li| {
            @memcpy(residual.data, x.data);

            // RMSNorm input
            try rmsnorm.rmsnorm(&x, layer.norm_input, cfg.rms_norm_eps);

            // QKV projections
            var q = try Matrix.init(alloc, seq, q_dim);
            defer q.deinit(alloc);
            try self.matmulDispatch(&x, &layer.w_q, &q);

            var k = try Matrix.init(alloc, seq, kv_dim);
            defer k.deinit(alloc);
            try self.matmulDispatch(&x, &layer.w_k, &k);

            var v = try Matrix.init(alloc, seq, kv_dim);
            defer v.deinit(alloc);
            try self.matmulDispatch(&x, &layer.w_v, &v);

            // RoPE on Q and K
            try self.rope_cache.applyToSequence(q.data, seq, cfg.num_attention_heads, start_pos);
            try self.rope_cache.applyToSequence(k.data, seq, cfg.num_key_value_heads, start_pos);

            // KV cache append + use full cached K/V for attention
            var attn_out = try Matrix.init(alloc, seq, q_dim);
            defer attn_out.deinit(alloc);

            if (kv_cache) |c| {
                try c.append(li, k.data, v.data, seq);
                const total_kv_len = c.currentLen(li);
                try attention.attentionPagedGqa(
                    alloc,
                    q.data,
                    c.page_manager,
                    c.block_tables[li].items,
                    attn_out.data,
                    seq,
                    total_kv_len,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    true,
                    start_pos,
                );
            } else {
                try attention.attentionGqa(
                    alloc,
                    q.data,
                    k.data,
                    v.data,
                    attn_out.data,
                    seq,
                    seq,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    true,
                    start_pos,
                );
            }

            // O projection
            var o_out = try Matrix.init(alloc, seq, cfg.hidden_size);
            defer o_out.deinit(alloc);
            try self.matmulDispatch(&attn_out, &layer.w_o, &o_out);

            // Residual add
            for (x.data, residual.data, o_out.data) |*xv, rv, ov| xv.* = rv + ov;

            // Save residual for MLP
            @memcpy(residual.data, x.data);

            // RMSNorm post-attention
            try rmsnorm.rmsnorm(&x, layer.norm_post_attn, cfg.rms_norm_eps);

            // MLP
            var gate = try Matrix.init(alloc, seq, cfg.intermediate_size);
            defer gate.deinit(alloc);
            try self.matmulDispatch(&x, &layer.w_gate, &gate);

            var up = try Matrix.init(alloc, seq, cfg.intermediate_size);
            defer up.deinit(alloc);
            try self.matmulDispatch(&x, &layer.w_up, &up);

            try silu.siluMul(gate.data, up.data);

            var down_out = try Matrix.init(alloc, seq, cfg.hidden_size);
            defer down_out.deinit(alloc);
            try self.matmulDispatch(&gate, &layer.w_down, &down_out);

            for (x.data, residual.data, down_out.data) |*xv, rv, dv| xv.* = rv + dv;
        }

        // 3. Final norm + LM head on the LAST token only
        try rmsnorm.rmsnorm(&x, self.final_norm, cfg.rms_norm_eps);

        // Last-token logits = last row of x @ lm_head
        const last_row_x = x.row(seq - 1);
        var x_last = try Matrix.init(alloc, 1, cfg.hidden_size);
        defer x_last.deinit(alloc);
        @memcpy(x_last.rowMut(0), last_row_x);

        var logits_m = try Matrix.init(alloc, 1, cfg.vocab_size);
        defer logits_m.deinit(alloc);
        try self.matmulDispatch(&x_last, &self.lm_head, &logits_m);

        const logits = try alloc.alloc(f32, cfg.vocab_size);
        @memcpy(logits, logits_m.row(0));
        return logits;
    }

    /// Streaming autoregressive generation. Prefill the prompt, then
    /// decode N tokens one-at-a-time with KV cache + greedy sampling.
    /// Yields each new token to the callback. Callback returns `true`
    /// to continue generating, `false` to stop early.
    ///
    /// Returns the generated token IDs (excluding the prompt). Caller
    /// frees.
    pub fn generate(
        self: *Model,
        alloc: std.mem.Allocator,
        prompt_ids: []const u32,
        max_new: usize,
        kv: *kv_cache_mod.KVCache,
        on_token: ?*const fn (token: u32, idx: usize) bool,
    ) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);

        // Prefill: forward the full prompt at once
        const logits = try self.forward(alloc, prompt_ids, kv, 0);
        defer alloc.free(logits);
        const first = try greedyArgmax(logits);
        try out.append(alloc, first);
        if (on_token) |cb| {
            if (!cb(first, 0)) return out.toOwnedSlice(alloc);
        }

        // Decode: one token at a time
        var pos = prompt_ids.len;
        var i: usize = 1;
        while (i < max_new) : (i += 1) {
            const one = [_]u32{out.items[out.items.len - 1]};
            const new_logits = try self.forward(alloc, &one, kv, pos);
            defer alloc.free(new_logits);
            const next = try greedyArgmax(new_logits);
            try out.append(alloc, next);
            if (on_token) |cb| {
                if (!cb(next, i)) break;
            }
            pos += 1;
        }
        return out.toOwnedSlice(alloc);
    }
};

fn greedyArgmax(logits: []const f32) !u32 {
    if (logits.len == 0) return error.EmptyLogits;
    var best_idx: usize = 0;
    var best_val: f32 = logits[0];
    var i: usize = 1;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return @intCast(best_idx);
}

fn transposeMatrix(alloc: std.mem.Allocator, m: *const Matrix) !Matrix {
    var out = try Matrix.init(alloc, m.cols, m.rows);
    errdefer out.deinit(alloc);
    var r: usize = 0;
    while (r < m.rows) : (r += 1) {
        var c: usize = 0;
        while (c < m.cols) : (c += 1) {
            out.set(c, r, m.at(r, c));
        }
    }
    return out;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Config.fromJson — TinyLlama-shape" {
    const json =
        \\{
        \\  "vocab_size": 32000,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 5632,
        \\  "num_hidden_layers": 22,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "max_position_embeddings": 2048
        \\}
    ;
    const cfg = try Config.fromJson(testing.allocator, json);
    try testing.expectEqual(@as(usize, 32000), cfg.vocab_size);
    try testing.expectEqual(@as(usize, 2048), cfg.hidden_size);
    try testing.expectEqual(@as(usize, 22), cfg.num_hidden_layers);
    try testing.expectEqual(@as(usize, 32), cfg.num_attention_heads);
    try testing.expectEqual(@as(usize, 4), cfg.num_key_value_heads);
    try testing.expectEqual(@as(usize, 64), cfg.head_dim);
}

test "Config.fromJson — missing field errors" {
    const json = "{\"vocab_size\": 100}";
    try testing.expectError(error.MissingField, Config.fromJson(testing.allocator, json));
}

test "transposeMatrix — small case" {
    var m = try Matrix.init(testing.allocator, 2, 3);
    defer m.deinit(testing.allocator);
    m.set(0, 0, 1);
    m.set(0, 1, 2);
    m.set(0, 2, 3);
    m.set(1, 0, 4);
    m.set(1, 1, 5);
    m.set(1, 2, 6);

    var t = try transposeMatrix(testing.allocator, &m);
    defer t.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), t.rows);
    try testing.expectEqual(@as(usize, 2), t.cols);
    try testing.expectEqual(@as(f32, 1), t.at(0, 0));
    try testing.expectEqual(@as(f32, 4), t.at(0, 1));
    try testing.expectEqual(@as(f32, 6), t.at(2, 1));
}
