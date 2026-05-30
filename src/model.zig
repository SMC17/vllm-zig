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
const builtin = @import("builtin");
const safetensors = @import("safetensors");
const Matrix = @import("tensor.zig").Matrix;
const matmul = @import("kernels/matmul.zig");
const matmul_bf16 = @import("kernels/matmul_bf16.zig");

const rmsnorm = @import("kernels/rmsnorm.zig");
const rope = @import("kernels/rope.zig");
const attention = @import("kernels/attention.zig");
const silu = @import("kernels/silu.zig");
const weights_mod = @import("weights.zig");
const kv_cache_mod = @import("kv_cache.zig");
const thread_pool = @import("thread_pool.zig");
const gpu_context_mod = @import("gpu_context.zig");

// ============================================================================
// Per-region instrumentation
// ============================================================================
//
// The Instrumentation struct accumulates wall-time spent in each named
// region of Model.forward. The model never owns one; callers (benches,
// tests) construct an Instrumentation, attach it to the Model with
// setInstrumentation(), call forward() any number of times, then read
// totals back. Raw linux.clock_gettime(MONOTONIC) is used so the model
// path does not need an Io parameter; non-Linux platforms get a no-op
// timer (region time = 0) without breaking compilation.

pub const Region = enum(u8) {
    embed = 0,
    layernorm_input = 1,
    qkv_proj = 2,
    rope = 3,
    attention = 4,
    o_proj = 5,
    residual_add_ln = 6,
    mlp_gate_up = 7,
    silu_mul = 8,
    mlp_down = 9,
    final_norm = 10,
    lm_head = 11,
};

pub const region_names = [_][]const u8{
    "embed",
    "layernorm_input",
    "qkv_proj",
    "rope",
    "attention",
    "o_proj",
    "residual_add_ln",
    "mlp_gate_up",
    "silu_mul",
    "mlp_down",
    "final_norm",
    "lm_head",
};

pub const n_regions = region_names.len;

pub const Instrumentation = struct {
    // Total ns spent in each region across all forward() calls since
    // last reset(). Index by @intFromEnum(Region).
    totals_ns: [n_regions]u64 = [_]u64{0} ** n_regions,
    // Number of completed forward() calls instrumented (used for mean).
    n_calls: u64 = 0,

    pub fn reset(self: *Instrumentation) void {
        self.totals_ns = [_]u64{0} ** n_regions;
        self.n_calls = 0;
    }

    pub fn add(self: *Instrumentation, region: Region, ns: u64) void {
        self.totals_ns[@intFromEnum(region)] += ns;
    }

    pub fn totalNs(self: *const Instrumentation) u64 {
        var sum: u64 = 0;
        for (self.totals_ns) |v| sum += v;
        return sum;
    }
};

/// Returns nanoseconds since CLOCK_MONOTONIC origin. Linux uses
/// raw clock_gettime to avoid threading Io through Model.forward;
/// other platforms return 0 (region timings will all be zero -- the
/// model still runs).
inline fn nowNs() u64 {
    if (builtin.os.tag != .linux) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    const secs: u64 = @intCast(ts.sec);
    const nsecs: u64 = @intCast(ts.nsec);
    return secs * std.time.ns_per_s + nsecs;
}

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
    /// Optional raw BF16 copies of the MLP projections, used by the
    /// pool.matmulBF16Auto dispatch path when loadFromSafeTensorsOpts
    /// was called with load_mlp_bf16 = true. Storage layout matches
    /// the matmul_bf16 kernel: rows = K (input dim), cols = N (output
    /// dim). Null on default loads to keep memory at F32-only cost.
    w_gate_bf16: ?weights_mod.RawBF16 = null,
    w_up_bf16: ?weights_mod.RawBF16 = null,
    w_down_bf16: ?weights_mod.RawBF16 = null,
    /// Optional FUSED raw BF16 storage holding [w_gate | w_up] concatenated
    /// horizontally as a single [hidden, 2 * intermediate] matrix. When this
    /// field is populated (loadFromSafeTensorsOpts with fuse_mlp_gate_up = true),
    /// the forward path dispatches ONE pool.matmulBF16Auto call producing a
    /// fused [seq, 2N] result, then splits it into gate and up halves for
    /// silu_mul. Eliminates one pool wake/wait round-trip per layer and lets
    /// the cache prefetcher work across the contiguous concatenated B-matrix.
    w_gateup_bf16: ?weights_mod.RawBF16 = null,
    /// Optional FUSED raw BF16 storage holding [w_q | w_k | w_v] concatenated
    /// horizontally as a single [hidden, q_dim + 2 * kv_dim] matrix. When this
    /// field is populated (loadFromSafeTensorsOpts with fuse_qkv = true),
    /// forward dispatches ONE pool.matmulBF16Auto producing the fused
    /// [seq, q_dim + 2 * kv_dim] result, then splits it into Q / K / V
    /// before RoPE. Halves the attention BF16 pool dispatches per layer
    /// (3 -> 1) and lets the cache prefetcher work across the contiguous
    /// QKV B-matrix.
    w_qkv_bf16: ?weights_mod.RawBF16 = null,
    /// Optional raw BF16 copies of the attention projections, parallel
    /// to the MLP fields. Populated when LoadOpts.load_attn_bf16 = true.
    /// Used by the v0.0.8 lever-2 attention-pool dispatch path.
    w_q_bf16: ?weights_mod.RawBF16 = null,
    w_k_bf16: ?weights_mod.RawBF16 = null,
    w_v_bf16: ?weights_mod.RawBF16 = null,
    w_o_bf16: ?weights_mod.RawBF16 = null,

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
        if (self.w_gate_bf16) |*b| b.deinit(alloc);
        if (self.w_up_bf16) |*b| b.deinit(alloc);
        if (self.w_down_bf16) |*b| b.deinit(alloc);
        if (self.w_gateup_bf16) |*b| b.deinit(alloc);
        if (self.w_qkv_bf16) |*b| b.deinit(alloc);
        if (self.w_q_bf16) |*b| b.deinit(alloc);
        if (self.w_k_bf16) |*b| b.deinit(alloc);
        if (self.w_v_bf16) |*b| b.deinit(alloc);
        if (self.w_o_bf16) |*b| b.deinit(alloc);
    }
};

/// Options for loadFromSafeTensorsOpts. The default-constructed value
/// (`.{}`) reproduces the loadFromSafeTensors behavior bit-for-bit.
pub const LoadOpts = struct {
    /// When true, also load raw BF16 copies of the MLP projections
    /// into LayerWeights.w_gate_bf16 / w_up_bf16 / w_down_bf16. The
    /// pool.matmulBF16Auto dispatch reads these on the v0.0.8 MLP path.
    /// Adds roughly 380 MB to the TinyLlama load on top of F32 weights.
    load_mlp_bf16: bool = false,
    /// When true (and load_mlp_bf16 is also true), pack gate and up
    /// projections into a single fused w_gateup_bf16 [hidden, 2 * intermediate]
    /// matrix and SKIP allocating the separate w_gate_bf16 / w_up_bf16
    /// fields. forward() dispatches one matmul instead of two for the
    /// gate+up path. Same total memory as load_mlp_bf16 alone (the fused
    /// matrix replaces the two separates).
    fuse_mlp_gate_up: bool = false,
    /// When true (and load_attn_bf16 is also true), pack Q, K, V projections
    /// into a single fused w_qkv_bf16 [hidden, q_dim + 2 * kv_dim] matrix
    /// and SKIP allocating the separate w_q_bf16 / w_k_bf16 / w_v_bf16
    /// fields. forward() dispatches ONE matmul instead of three for the
    /// QKV path. Same total memory as load_attn_bf16 alone.
    fuse_qkv: bool = false,
    /// When true, also load raw BF16 copies of the attention projections
    /// (w_q / w_k / w_v / w_o) into the parallel _bf16 LayerWeights
    /// fields. The pool.matmulBF16Auto dispatch reads these on the
    /// v0.0.8 lever-2 attention path. Adds roughly 100 MB to the
    /// TinyLlama load on top of F32 attention weights.
    load_attn_bf16: bool = false,
    /// Lever-9: re-pack every BF16 weight matrix at load time into the
    /// N-tile-major layout consumed by matmulBF16SIMDNtileMajor /
    /// pool.matmulBF16NtileMajorAuto. Same total bytes, only the
    /// physical memory ordering changes. forward() must also have
    /// use_bf16_ntile_major=true for the new kernel path to engage.
    bf16_ntile_major: bool = false,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    config: Config,
    embed: Matrix,
    layers: []LayerWeights,
    final_norm: []f32,
    /// LM head stored as raw BF16 in the layout the matmulBF16 kernel
    /// consumes directly: rows = hidden_size (K), cols = vocab_size (N).
    /// At decode time M=1, so the LM-head matmul is the largest single
    /// memory-bandwidth sink (~131 MB BF16 vs ~262 MB F32 on TinyLlama).
    /// Keeping it in BF16 halves the bytes the scan has to pull through
    /// L2/L3 — see matmulBF16SIMD docstring.
    lm_head_bf16: weights_mod.RawBF16,
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
    /// Optional per-region wall-time accumulator. When set, forward()
    /// stamps CLOCK_MONOTONIC around each named region (layernorm, qkv,
    /// attention, lm_head, ...) and totals_ns[] grows by the region's
    /// elapsed ns. Caller retains ownership.
    instrument: ?*Instrumentation = null,
    /// When true, route lm_head matmul through pool.matmulBF16Auto
    /// instead of the single-thread matmul_bf16.matmulBF16SIMD. Wired
    /// in v0.0.7. Has no effect unless a pool is also attached.
    use_pooled_bf16_lm_head: bool = false,
    /// When true, route MLP gate / up / down matmuls through
    /// pool.matmulBF16Auto using the raw BF16 weight copies. Requires
    /// the model to have been loaded via loadFromSafeTensorsOpts with
    /// load_mlp_bf16 = true; otherwise forward() falls through to the
    /// F32 matmulDispatch path. Wired in v0.0.8.
    use_pooled_bf16_mlp: bool = false,
    /// When true, route attention Q / K / V / O projections through
    /// pool.matmulBF16Auto using the raw BF16 weight copies. Requires
    /// load_attn_bf16 = true at load time. v0.0.8 lever-2.
    use_pooled_bf16_attn: bool = false,
    /// Lever-9: when true, use pool.matmulBF16NtileMajorAuto (which calls
    /// the N-tile-major col-range kernel) for every pooled BF16 dispatch
    /// site. Requires LoadOpts.bf16_ntile_major = true at load time so
    /// the weight matrices are stored in the N-tile-major layout.
    use_bf16_ntile_major: bool = false,

    /// Configure the matmul thread count for subsequent forward passes.
    pub fn setThreads(self: *Model, n: usize) void {
        self.n_threads = n;
    }

    /// Attach a per-region instrumentation accumulator. Caller owns
    /// the struct and is responsible for reading it back after forward().
    pub fn setInstrumentation(self: *Model, inst: ?*Instrumentation) void {
        self.instrument = inst;
    }

    /// Toggle the pooled BF16 lm_head dispatch. When enabled AND a pool
    /// is attached, Model.forward routes the final logits matmul through
    /// pool.matmulBF16Auto (col-split at M=1) instead of the single-
    /// thread matmul_bf16.matmulBF16SIMD path.
    pub fn setPooledBF16LMHead(self: *Model, enable: bool) void {
        self.use_pooled_bf16_lm_head = enable;
    }

    /// Toggle the pooled BF16 MLP dispatch (v0.0.8 lever-1 retarget).
    /// When enabled AND a pool is attached AND the model was loaded
    /// with LoadOpts.load_mlp_bf16 = true, Model.forward routes the
    /// gate / up / down projections through pool.matmulBF16Auto using
    /// the raw BF16 weight copies. Otherwise forward() falls through
    /// to the existing F32 matmulDispatch path.
    pub fn setPooledBF16MLP(self: *Model, enable: bool) void {
        self.use_pooled_bf16_mlp = enable;
    }

    /// Toggle the pooled BF16 attention dispatch (v0.0.8 lever-2).
    /// When enabled AND a pool is attached AND the model was loaded
    /// with LoadOpts.load_attn_bf16 = true, Model.forward routes Q,K,V
    /// and O projections through pool.matmulBF16Auto.
    pub fn setPooledBF16Attn(self: *Model, enable: bool) void {
        self.use_pooled_bf16_attn = enable;
    }

    /// Lever-9: toggle the N-tile-major BF16 pool dispatch. Requires
    /// LoadOpts.bf16_ntile_major = true at model load — without that,
    /// the weight matrices are still in row-major layout and the new
    /// kernel will produce garbage.
    pub fn setBF16NtileMajor(self: *Model, enable: bool) void {
        self.use_bf16_ntile_major = enable;
    }

    /// Dispatch a pooled BF16 matmul through the layout-appropriate pool
    /// method. When ntile-major is enabled, prefill (M >= 4) routes
    /// through matmulBF16BlockedAuto (lever-12 GEMM substrate) and
    /// decode (M < 4) through matmulBF16NtileMajorAuto (lever-9).
    inline fn dispatchBF16(
        self: *const Model,
        p: *thread_pool.Pool,
        a: *const Matrix,
        b: []const u16,
        b_rows: usize,
        b_cols: usize,
        c: *Matrix,
    ) thread_pool.Error!void {
        if (self.use_bf16_ntile_major) {
            return p.matmulBF16BlockedAuto(a, b, b_rows, b_cols, c, self.n_threads);
        }
        return p.matmulBF16Auto(a, b, b_rows, b_cols, c, self.n_threads);
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
        self.lm_head_bf16.deinit(self.allocator);
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
        return loadFromSafeTensorsOpts(alloc, st, config, .{});
    }

    /// Variant of loadFromSafeTensors that accepts LoadOpts. The
    /// default-constructed LoadOpts reproduces loadFromSafeTensors
    /// bit-for-bit; LoadOpts.load_mlp_bf16 = true additionally loads
    /// raw BF16 copies of the MLP projections for the v0.0.8 lever-1
    /// pool dispatch path.
    pub fn loadFromSafeTensorsOpts(
        alloc: std.mem.Allocator,
        st: *safetensors.SafeTensors,
        config: Config,
        opts: LoadOpts,
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
            if (opts.load_mlp_bf16) {
                if (opts.fuse_mlp_gate_up) {
                    // Fused gate+up. Both gate and up share input dim hidden,
                    // output dim intermediate. Concatenate them horizontally
                    // into a single [hidden, 2 * intermediate] BF16 matrix
                    // so forward() can dispatch ONE pool.matmulBF16Auto call.
                    const gate_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", .{i});
                    var up_name_buf: [256]u8 = undefined;
                    const up_name = try std.fmt.bufPrint(&up_name_buf, "model.layers.{d}.mlp.up_proj.weight", .{i});
                    layers[i].w_gateup_bf16 = try weights_mod.loadMatrixBF16RawConcatTransposed(alloc, st, gate_name, up_name);
                } else {
                    layers[i].w_gate_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", .{i}));
                    layers[i].w_up_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.up_proj.weight", .{i}));
                }
                layers[i].w_down_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.down_proj.weight", .{i}));
            }
            if (opts.load_attn_bf16) {
                if (opts.fuse_qkv) {
                    // QKV fusion: concat Q | K | V columns into a single
                    // [hidden, q_dim + 2 * kv_dim] BF16 matrix. forward()
                    // dispatches one matmul instead of three; split before
                    // RoPE. Same lever as gate+up fusion (v0.0.8 lever-3).
                    const q_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", .{i});
                    var k_name_buf: [256]u8 = undefined;
                    const k_name = try std.fmt.bufPrint(&k_name_buf, "model.layers.{d}.self_attn.k_proj.weight", .{i});
                    var v_name_buf: [256]u8 = undefined;
                    const v_name = try std.fmt.bufPrint(&v_name_buf, "model.layers.{d}.self_attn.v_proj.weight", .{i});
                    layers[i].w_qkv_bf16 = try weights_mod.loadMatrixBF16RawConcat3Transposed(alloc, st, q_name, k_name, v_name);
                } else {
                    layers[i].w_q_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", .{i}));
                    layers[i].w_k_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_proj.weight", .{i}));
                    layers[i].w_v_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.v_proj.weight", .{i}));
                }
                layers[i].w_o_bf16 = try weights_mod.loadMatrixBF16RawTransposed(alloc, st, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.o_proj.weight", .{i}));
            }
        }

        // Lever-9 weight repack: convert every BF16 weight matrix from
        // row-major to N-tile-major layout in-place. Allocates a scratch
        // buffer per matrix, swaps the data pointer, frees the old data.
        if (opts.bf16_ntile_major) {
            for (0..config.num_hidden_layers) |i| {
                inline for (.{ "w_gateup_bf16", "w_gate_bf16", "w_up_bf16", "w_down_bf16", "w_qkv_bf16", "w_q_bf16", "w_k_bf16", "w_v_bf16", "w_o_bf16" }) |field_name| {
                    if (@field(layers[i], field_name)) |old_raw| {
                        const new_raw = try weights_mod.repackBF16ToNtileMajor(alloc, old_raw);
                        alloc.free(old_raw.data);
                        @field(layers[i], field_name) = new_raw;
                    }
                }
            }
        }

        const final_norm = try weights_mod.loadVectorBF16(alloc, st, "model.norm.weight");
        errdefer alloc.free(final_norm);

        // LM head: load as raw BF16 in [hidden_size, vocab_size] layout
        // so it slots straight into matmul_bf16.matmulBF16SIMD with no
        // F32 upcast. Tied-weights models re-pun the embed table into
        // BF16 (it was loaded as F32 above).
        var lm_head_bf16 = if (config.tie_word_embeddings)
            try transposedEmbedToBF16(alloc, &embed)
        else
            try weights_mod.loadMatrixBF16RawTransposed(alloc, st, "lm_head.weight");
        errdefer lm_head_bf16.deinit(alloc);

        // Lever-9: lm_head also repacked when flag set, so the single-thread
        // matmulBF16SIMD fallback in forward() will call the N-tile-major
        // variant. We do this after the errdefer to keep cleanup simple.
        if (opts.bf16_ntile_major) {
            const new_lm_head = try weights_mod.repackBF16ToNtileMajor(alloc, lm_head_bf16);
            alloc.free(lm_head_bf16.data);
            lm_head_bf16 = new_lm_head;
        }

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
            .lm_head_bf16 = lm_head_bf16,
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

        // Region-timing helpers. inst is null on the hot path; when set,
        // each region brackets its work with nowNs() and accumulates the
        // delta into self.instrument.totals_ns. The branch is predictable
        // (the same null-or-not for the lifetime of the call).
        const inst = self.instrument;

        // 1. Embed: gather rows from embed table
        const t_embed_start = if (inst != null) nowNs() else 0;
        var x = try Matrix.init(alloc, seq, cfg.hidden_size);
        defer x.deinit(alloc);
        for (token_ids, 0..) |id, t| {
            if (id >= cfg.vocab_size) return error.TokenIdOutOfRange;
            const src = self.embed.row(id);
            @memcpy(x.rowMut(t), src);
        }
        if (inst) |ip| ip.add(.embed, nowNs() - t_embed_start);

        // 2. Multi-block forward
        var residual = try Matrix.init(alloc, seq, cfg.hidden_size);
        defer residual.deinit(alloc);

        for (self.layers, 0..) |layer, li| {
            @memcpy(residual.data, x.data);

            // RMSNorm input
            const t_ln_in_start = if (inst != null) nowNs() else 0;
            try rmsnorm.rmsnorm(&x, layer.norm_input, cfg.rms_norm_eps);
            if (inst) |ip| ip.add(.layernorm_input, nowNs() - t_ln_in_start);

            // QKV projections. v0.0.8 lever-2: when use_pooled_bf16_attn is
            // set AND a pool is attached AND BF16 mirror weights are loaded,
            // dispatch through pool.matmulBF16Auto. Otherwise fall through
            // to the existing F32 matmulDispatch path.
            //
            // QKV fusion (v0.0.9 lever-4, LoadOpts.fuse_qkv): when
            // w_qkv_bf16 is populated, dispatch ONE matmul against the
            // concatenated [W_q | W_k | W_v] producing a [seq, q_dim + 2 * kv_dim]
            // output, then slice into Q / K / V before RoPE. Eliminates
            // two pool wake/wait round-trips per layer.
            const attn_bf16 = self.use_pooled_bf16_attn and self.pool != null and ((layer.w_qkv_bf16 != null) or (layer.w_q_bf16 != null and layer.w_k_bf16 != null and layer.w_v_bf16 != null)) and layer.w_o_bf16 != null;
            const attn_qkv_fused = self.use_pooled_bf16_attn and self.pool != null and layer.w_qkv_bf16 != null and layer.w_o_bf16 != null;
            const t_qkv_start = if (inst != null) nowNs() else 0;

            // Fused path uses one [seq, q_dim + 2 * kv_dim] matrix and
            // produces Q / K / V views onto it. Split paths allocate
            // q / k / v separately.
            const fused_qkv_cols: usize = q_dim + 2 * kv_dim;
            var qkv_buf: ?Matrix = if (attn_qkv_fused) try Matrix.init(alloc, seq, fused_qkv_cols) else null;
            defer if (qkv_buf) |*qb| qb.deinit(alloc);
            var q_buf: ?Matrix = if (!attn_qkv_fused) try Matrix.init(alloc, seq, q_dim) else null;
            defer if (q_buf) |*qb| qb.deinit(alloc);
            var k_buf: ?Matrix = if (!attn_qkv_fused) try Matrix.init(alloc, seq, kv_dim) else null;
            defer if (k_buf) |*kb| kb.deinit(alloc);
            var v_buf: ?Matrix = if (!attn_qkv_fused) try Matrix.init(alloc, seq, kv_dim) else null;
            defer if (v_buf) |*vb| vb.deinit(alloc);

            if (attn_qkv_fused) {
                const p = self.pool.?;
                const qkvw = layer.w_qkv_bf16.?;
                try self.dispatchBF16(p, &x, qkvw.data, qkvw.rows, qkvw.cols, &qkv_buf.?);
            } else if (attn_bf16) {
                const p = self.pool.?;
                try self.dispatchBF16(p, &x, layer.w_q_bf16.?.data, layer.w_q_bf16.?.rows, layer.w_q_bf16.?.cols, &q_buf.?);
                try self.dispatchBF16(p, &x, layer.w_k_bf16.?.data, layer.w_k_bf16.?.rows, layer.w_k_bf16.?.cols, &k_buf.?);
                try self.dispatchBF16(p, &x, layer.w_v_bf16.?.data, layer.w_v_bf16.?.rows, layer.w_v_bf16.?.cols, &v_buf.?);
            } else {
                try self.matmulDispatch(&x, &layer.w_q, &q_buf.?);
                try self.matmulDispatch(&x, &layer.w_k, &k_buf.?);
                try self.matmulDispatch(&x, &layer.w_v, &v_buf.?);
            }
            if (inst) |ip| ip.add(.qkv_proj, nowNs() - t_qkv_start);

            // RoPE needs []f32 slices over Q and K. In the fused path we
            // construct non-owning Matrix views onto qkv_buf so the rest
            // of the attention block can consume them identically.
            const q_slice_total = seq * q_dim;
            const k_slice_total = seq * kv_dim;
            const v_slice_total = seq * kv_dim;
            const q: Matrix = if (attn_qkv_fused) .{
                .rows = seq,
                .cols = q_dim,
                .data = qkv_buf.?.data[0..q_slice_total],
            } else q_buf.?;
            const k: Matrix = if (attn_qkv_fused) .{
                .rows = seq,
                .cols = kv_dim,
                .data = qkv_buf.?.data[q_slice_total .. q_slice_total + k_slice_total],
            } else k_buf.?;
            const v: Matrix = if (attn_qkv_fused) .{
                .rows = seq,
                .cols = kv_dim,
                .data = qkv_buf.?.data[q_slice_total + k_slice_total .. q_slice_total + k_slice_total + v_slice_total],
            } else v_buf.?;

            // RoPE on Q and K
            const t_rope_start = if (inst != null) nowNs() else 0;
            try self.rope_cache.applyToSequence(q.data, seq, cfg.num_attention_heads, start_pos);
            try self.rope_cache.applyToSequence(k.data, seq, cfg.num_key_value_heads, start_pos);
            if (inst) |ip| ip.add(.rope, nowNs() - t_rope_start);

            // KV cache append + use full cached K/V for attention
            const t_attn_start = if (inst != null) nowNs() else 0;
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
            if (inst) |ip| ip.add(.attention, nowNs() - t_attn_start);

            // O projection. v0.0.8 lever-2 BF16 dispatch when enabled
            // and BF16 mirror is loaded. Same gate as the QKV block above.
            const t_o_start = if (inst != null) nowNs() else 0;
            var o_out = try Matrix.init(alloc, seq, cfg.hidden_size);
            defer o_out.deinit(alloc);
            if (attn_bf16) {
                const p = self.pool.?;
                try self.dispatchBF16(p, &attn_out, layer.w_o_bf16.?.data, layer.w_o_bf16.?.rows, layer.w_o_bf16.?.cols, &o_out);
            } else {
                try self.matmulDispatch(&attn_out, &layer.w_o, &o_out);
            }
            if (inst) |ip| ip.add(.o_proj, nowNs() - t_o_start);

            // Residual add + post-attention norm bracketed together so
            // the timing matches the typical fused-residual+ln description.
            const t_rln_start = if (inst != null) nowNs() else 0;
            for (x.data, residual.data, o_out.data) |*xv, rv, ov| xv.* = rv + ov;

            // Save residual for MLP
            @memcpy(residual.data, x.data);

            // RMSNorm post-attention
            try rmsnorm.rmsnorm(&x, layer.norm_post_attn, cfg.rms_norm_eps);
            if (inst) |ip| ip.add(.residual_add_ln, nowNs() - t_rln_start);

            // MLP: gate + up projections. v0.0.8 lever-1 retarget: when
            // use_pooled_bf16_mlp is set AND a pool is attached AND the
            // BF16 weight copies are loaded, dispatch through
            // pool.matmulBF16Auto using the raw BF16 weights. Otherwise
            // fall back to the F32 matmulDispatch path.
            //
            // Gate+up fusion (LoadOpts.fuse_mlp_gate_up): when w_gateup_bf16
            // is populated, dispatch ONE matmul against the concatenated
            // [W_gate | W_up] producing a [seq, 2N] output, then slice
            // into gate / up halves for silu_mul. Eliminates one pool
            // wake/wait round-trip per layer.
            const mlp_bf16 = self.use_pooled_bf16_mlp and self.pool != null and ((layer.w_gateup_bf16 != null) or (layer.w_gate_bf16 != null and layer.w_up_bf16 != null)) and layer.w_down_bf16 != null;
            const mlp_fused = self.use_pooled_bf16_mlp and self.pool != null and layer.w_gateup_bf16 != null and layer.w_down_bf16 != null;
            const t_gu_start = if (inst != null) nowNs() else 0;

            // Fused path uses one [seq, 2N] matrix. Split paths each
            // allocate gate and up [seq, N] separately.
            const N: usize = cfg.intermediate_size;
            var gateup_buf: ?Matrix = if (mlp_fused) try Matrix.init(alloc, seq, 2 * N) else null;
            defer if (gateup_buf) |*gb| gb.deinit(alloc);
            var gate_buf: ?Matrix = if (!mlp_fused) try Matrix.init(alloc, seq, N) else null;
            defer if (gate_buf) |*gb| gb.deinit(alloc);
            var up_buf: ?Matrix = if (!mlp_fused) try Matrix.init(alloc, seq, N) else null;
            defer if (up_buf) |*ub| ub.deinit(alloc);

            if (mlp_fused) {
                const p = self.pool.?;
                const gu = layer.w_gateup_bf16.?;
                try self.dispatchBF16(p, &x, gu.data, gu.rows, gu.cols, &gateup_buf.?);
            } else if (mlp_bf16) {
                const p = self.pool.?;
                try self.dispatchBF16(p, &x, layer.w_gate_bf16.?.data, layer.w_gate_bf16.?.rows, layer.w_gate_bf16.?.cols, &gate_buf.?);
                try self.dispatchBF16(p, &x, layer.w_up_bf16.?.data, layer.w_up_bf16.?.rows, layer.w_up_bf16.?.cols, &up_buf.?);
            } else {
                try self.matmulDispatch(&x, &layer.w_gate, &gate_buf.?);
                try self.matmulDispatch(&x, &layer.w_up, &up_buf.?);
            }
            if (inst) |ip| ip.add(.mlp_gate_up, nowNs() - t_gu_start);

            const t_silu_start = if (inst != null) nowNs() else 0;
            // In the fused path silu_mul reads gateup[0..N] (gate) and
            // gateup[N..2N] (up) and writes the result into the gate half
            // so the down matmul can consume gateup[0..N] directly.
            if (mlp_fused) {
                const gb = &gateup_buf.?;
                const total_n = seq * N;
                try silu.siluMul(gb.data[0..total_n], gb.data[total_n .. total_n * 2]);
            } else {
                try silu.siluMul(gate_buf.?.data, up_buf.?.data);
            }
            if (inst) |ip| ip.add(.silu_mul, nowNs() - t_silu_start);

            const t_down_start = if (inst != null) nowNs() else 0;
            var down_out = try Matrix.init(alloc, seq, cfg.hidden_size);
            defer down_out.deinit(alloc);

            // Build a Matrix view of the silu_mul result for the down input.
            // In the fused path this is a non-owning view onto the first
            // half of gateup_buf (data slice ends inside gateup_buf.data;
            // we never deinit this synthetic Matrix). In the split path
            // gate_buf is already the right shape.
            var down_input: Matrix = if (mlp_fused) .{
                .rows = seq,
                .cols = N,
                .data = gateup_buf.?.data[0 .. seq * N],
            } else gate_buf.?;

            if (mlp_bf16) {
                const p = self.pool.?;
                try self.dispatchBF16(p, &down_input, layer.w_down_bf16.?.data, layer.w_down_bf16.?.rows, layer.w_down_bf16.?.cols, &down_out);
            } else {
                try self.matmulDispatch(&down_input, &layer.w_down, &down_out);
            }

            for (x.data, residual.data, down_out.data) |*xv, rv, dv| xv.* = rv + dv;
            if (inst) |ip| ip.add(.mlp_down, nowNs() - t_down_start);
        }

        // 3. Final norm + LM head on the LAST token only
        const t_fn_start = if (inst != null) nowNs() else 0;
        try rmsnorm.rmsnorm(&x, self.final_norm, cfg.rms_norm_eps);
        if (inst) |ip| ip.add(.final_norm, nowNs() - t_fn_start);

        // Last-token logits = last row of x @ lm_head, with the LM head
        // kept in BF16 so the inner-loop B-matrix scan halves its bandwidth
        // requirement. At TinyLlama scale this is the single largest matmul
        // in the decode step (M=1, K=2048, N=32000 = ~131 MB BF16 vs ~262 MB
        // F32). The SIMD kernel widens BF16→F32 in-register, no temp buffer.
        const last_row_x = x.row(seq - 1);
        var x_last = try Matrix.init(alloc, 1, cfg.hidden_size);
        defer x_last.deinit(alloc);
        @memcpy(x_last.rowMut(0), last_row_x);

        var logits_m = try Matrix.init(alloc, 1, cfg.vocab_size);
        defer logits_m.deinit(alloc);

        // lm_head: BF16 widening path. Two dispatch modes:
        //
        //   (a) single-thread matmulBF16SIMD (default, v0.0.6 path):
        //       the SIMD kernel widens BF16 -> F32 in-register with no
        //       scratch buffer. Compute-bound on widen-instructions at
        //       M=1 on Ice Lake (BENCH.md section 4.4).
        //
        //   (b) pooled matmulBF16Auto (v0.0.7 path, opt-in via
        //       setPooledBF16LMHead(true) plus an attached pool):
        //       col-split across pool workers. At M=1, N=32000 the
        //       work partitions cleanly. Closes BENCH.md section 6
        //       lever #1.
        //
        // An Accelerate cblas_sgemm variant was tried (see git log +
        // exp-1780016093-198112588): standalone bench measured 6.92x
        // faster on M2 Max, but end-to-end decode regressed 8% (104 to
        // 112 ms/tok, 5-run median). Kept available in the kernel
        // module for explicit callers but not wired here.
        const t_lm_start = if (inst != null) nowNs() else 0;
        if (self.use_pooled_bf16_lm_head) if (self.pool) |p| {
            try self.dispatchBF16(p, &x_last, self.lm_head_bf16.data, self.lm_head_bf16.rows, self.lm_head_bf16.cols, &logits_m);
        } else if (self.use_bf16_ntile_major) {
            try matmul_bf16.matmulBF16SIMDNtileMajor(&x_last, self.lm_head_bf16.data, self.lm_head_bf16.rows, self.lm_head_bf16.cols, &logits_m);
        } else {
            try matmul_bf16.matmulBF16SIMD(&x_last, self.lm_head_bf16.data, self.lm_head_bf16.rows, self.lm_head_bf16.cols, &logits_m);
        } else if (self.use_bf16_ntile_major) {
            try matmul_bf16.matmulBF16SIMDNtileMajor(&x_last, self.lm_head_bf16.data, self.lm_head_bf16.rows, self.lm_head_bf16.cols, &logits_m);
        } else {
            try matmul_bf16.matmulBF16SIMD(&x_last, self.lm_head_bf16.data, self.lm_head_bf16.rows, self.lm_head_bf16.cols, &logits_m);
        }
        if (inst) |ip| {
            ip.add(.lm_head, nowNs() - t_lm_start);
            ip.n_calls += 1;
        }

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

const bf16_mod = @import("bf16.zig");

/// Tied-weights fallback: requantise the F32 embed table back into BF16
/// in transposed layout (so it slots into matmul_bf16 the same way the
/// untied lm_head.weight does).
fn transposedEmbedToBF16(alloc: std.mem.Allocator, embed: *const Matrix) !weights_mod.RawBF16 {
    // embed is [vocab_size, hidden_size] (loadMatrixBF16 did NOT transpose).
    // We want lm_head [hidden_size, vocab_size].
    const rows: usize = embed.cols; // hidden_size
    const cols: usize = embed.rows; // vocab_size
    const data = try alloc.alloc(u16, rows * cols);
    errdefer alloc.free(data);
    var r: usize = 0;
    while (r < embed.rows) : (r += 1) {
        var c: usize = 0;
        while (c < embed.cols) : (c += 1) {
            // embed[r,c] = embed.data[r * embed.cols + c]; goes to data[c, r]
            data[c * rows + r] = bf16_mod.f32ToBf16(embed.data[r * embed.cols + c]);
        }
    }
    return .{ .data = data, .rows = rows, .cols = cols };
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

test "LoadOpts default is bit-for-bit equal to legacy loader path" {
    // Default LoadOpts must not change any field defaults that the
    // existing loadFromSafeTensors callers rely on. If a future field
    // is added it must default to the legacy behavior.
    const opts: LoadOpts = .{};
    try testing.expectEqual(false, opts.load_mlp_bf16);
}

test "LayerWeights mlp BF16 fields default to null" {
    // BF16 mirror fields must default to null so deinit() of a
    // legacy-loaded LayerWeights does not try to free unallocated
    // memory and so the forward() guard `layer.w_gate_bf16 != null`
    // correctly falls through to the F32 path.
    var w_q = try Matrix.init(testing.allocator, 1, 1);
    defer w_q.deinit(testing.allocator);
    var w_k = try Matrix.init(testing.allocator, 1, 1);
    defer w_k.deinit(testing.allocator);
    var w_v = try Matrix.init(testing.allocator, 1, 1);
    defer w_v.deinit(testing.allocator);
    var w_o = try Matrix.init(testing.allocator, 1, 1);
    defer w_o.deinit(testing.allocator);
    var w_gate = try Matrix.init(testing.allocator, 1, 1);
    defer w_gate.deinit(testing.allocator);
    var w_up = try Matrix.init(testing.allocator, 1, 1);
    defer w_up.deinit(testing.allocator);
    var w_down = try Matrix.init(testing.allocator, 1, 1);
    defer w_down.deinit(testing.allocator);
    const ni = try testing.allocator.alloc(f32, 1);
    defer testing.allocator.free(ni);
    const np = try testing.allocator.alloc(f32, 1);
    defer testing.allocator.free(np);
    const lw: LayerWeights = .{
        .w_q = w_q,
        .w_k = w_k,
        .w_v = w_v,
        .w_o = w_o,
        .w_gate = w_gate,
        .w_up = w_up,
        .w_down = w_down,
        .norm_input = ni,
        .norm_post_attn = np,
    };
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_gate_bf16);
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_up_bf16);
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_down_bf16);
}

test "setPooledBF16MLP toggles flag" {
    // Lightweight method-level test: setPooledBF16MLP must round-trip
    // through Model.use_pooled_bf16_mlp without side effects on the
    // sibling lm_head flag. Avoid constructing a full Model by using a
    // synthetic stub.
    var m: Model = undefined;
    m.use_pooled_bf16_lm_head = false;
    m.use_pooled_bf16_mlp = false;
    m.use_pooled_bf16_attn = false;
    m.setPooledBF16MLP(true);
    try testing.expectEqual(true, m.use_pooled_bf16_mlp);
    try testing.expectEqual(false, m.use_pooled_bf16_lm_head);
    try testing.expectEqual(false, m.use_pooled_bf16_attn);
    m.setPooledBF16MLP(false);
    try testing.expectEqual(false, m.use_pooled_bf16_mlp);
}

test "setPooledBF16Attn toggles flag, independent of MLP/lm_head" {
    var m: Model = undefined;
    m.use_pooled_bf16_lm_head = false;
    m.use_pooled_bf16_mlp = false;
    m.use_pooled_bf16_attn = false;
    m.setPooledBF16Attn(true);
    try testing.expectEqual(true, m.use_pooled_bf16_attn);
    try testing.expectEqual(false, m.use_pooled_bf16_mlp);
    try testing.expectEqual(false, m.use_pooled_bf16_lm_head);
    m.setPooledBF16Attn(false);
    try testing.expectEqual(false, m.use_pooled_bf16_attn);
}

test "LoadOpts default load_attn_bf16 = false" {
    const opts: LoadOpts = .{};
    try testing.expectEqual(false, opts.load_attn_bf16);
    try testing.expectEqual(false, opts.load_mlp_bf16);
}

test "LayerWeights attn BF16 fields default to null" {
    var w_q = try Matrix.init(testing.allocator, 1, 1);
    defer w_q.deinit(testing.allocator);
    var w_k = try Matrix.init(testing.allocator, 1, 1);
    defer w_k.deinit(testing.allocator);
    var w_v = try Matrix.init(testing.allocator, 1, 1);
    defer w_v.deinit(testing.allocator);
    var w_o = try Matrix.init(testing.allocator, 1, 1);
    defer w_o.deinit(testing.allocator);
    var w_gate = try Matrix.init(testing.allocator, 1, 1);
    defer w_gate.deinit(testing.allocator);
    var w_up = try Matrix.init(testing.allocator, 1, 1);
    defer w_up.deinit(testing.allocator);
    var w_down = try Matrix.init(testing.allocator, 1, 1);
    defer w_down.deinit(testing.allocator);
    const ni = try testing.allocator.alloc(f32, 1);
    defer testing.allocator.free(ni);
    const np = try testing.allocator.alloc(f32, 1);
    defer testing.allocator.free(np);
    const lw: LayerWeights = .{
        .w_q = w_q,
        .w_k = w_k,
        .w_v = w_v,
        .w_o = w_o,
        .w_gate = w_gate,
        .w_up = w_up,
        .w_down = w_down,
        .norm_input = ni,
        .norm_post_attn = np,
    };
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_q_bf16);
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_k_bf16);
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_v_bf16);
    try testing.expectEqual(@as(?weights_mod.RawBF16, null), lw.w_o_bf16);
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
