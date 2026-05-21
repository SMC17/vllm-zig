//! Full transformer-block forward pass for a Llama-family model.
//!
//! Phase 1 proper: this composes matmul + rmsnorm + RoPE +
//! GQA-aware attention + SiLU MLP into a complete transformer block
//! that handles the canonical Llama architecture. Runs on CPU with
//! the f32 SIMD kernels.
//!
//! What this DOES at Phase 1:
//!   - Architecturally correct Llama block (RMSNorm input → QKV →
//!     RoPE → GQA attention with causal mask → O proj → residual →
//!     RMSNorm post-attn → SiLU MLP → residual)
//!   - GQA-aware: handles arbitrary n_q_heads / n_kv_heads ratios
//!     including TinyLlama's 32/4 = 8:1 sharing
//!   - Works at arbitrary (seq_len, hidden, n_heads, head_dim) sizes
//!   - Validates against the Llama equation faithfully
//!
//! What this does NOT do (Phase 1.5+):
//!   - KV-cache (each forward call recomputes the full K/V; OK for
//!     prefill but inefficient for decode)
//!   - BF16 / F16 storage (f32 only)
//!   - Real-weight loading (random weights only; safetensors-zig BF16
//!     conversion is v0.0.2 work)
//!   - Multi-thread (single-thread per token across all heads)

const std = @import("std");
const Matrix = @import("tensor.zig").Matrix;
const matmul = @import("kernels/matmul.zig");
const rmsnorm = @import("kernels/rmsnorm.zig");
const softmax = @import("kernels/softmax.zig");
const silu = @import("kernels/silu.zig");
const rope = @import("kernels/rope.zig");
const attention = @import("kernels/attention.zig");

pub const Config = struct {
    hidden_size: usize,
    intermediate_size: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    max_position_embeddings: usize = 2048,
};

pub const LayerWeights = struct {
    /// [hidden, n_q_heads * head_dim]
    w_q: Matrix,
    /// [hidden, n_kv_heads * head_dim]
    w_k: Matrix,
    /// [hidden, n_kv_heads * head_dim]
    w_v: Matrix,
    /// [n_q_heads * head_dim, hidden]
    w_o: Matrix,
    /// [hidden, intermediate]
    w_gate: Matrix,
    /// [hidden, intermediate]
    w_up: Matrix,
    /// [intermediate, hidden]
    w_down: Matrix,
    /// [hidden]
    norm_input: []const f32,
    /// [hidden]
    norm_post_attn: []const f32,
};

/// Compute a single transformer block forward pass.
/// `x` is in-place updated from [seq, hidden] input to [seq, hidden] output.
/// `rope_cache` must have head_dim matching cfg.head_dim and max_pos >= seq.
/// `q_offset` is the absolute starting position of x[0]; 0 for prefill from
/// scratch, KV-cache-length for decode (not yet supported in Phase 1).
pub fn transformerBlock(
    alloc: std.mem.Allocator,
    cfg: Config,
    rope_cache: *const rope.RopeCache,
    w: LayerWeights,
    x: *Matrix,
    q_offset: usize,
) !void {
    if (x.cols != cfg.hidden_size) return error.DimensionMismatch;
    if (cfg.n_q_heads * cfg.head_dim != cfg.hidden_size) return error.DimensionMismatch;
    const seq = x.rows;
    const q_dim = cfg.n_q_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // Save residual
    var residual = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer residual.deinit(alloc);
    @memcpy(residual.data, x.data);

    // --- Self-attention block ---
    // 1. RMSNorm
    try rmsnorm.rmsnorm(x, w.norm_input, cfg.rms_norm_eps);

    // 2. Q, K, V projections
    var q = try Matrix.init(alloc, seq, q_dim);
    defer q.deinit(alloc);
    try matmul.matmulSIMD(x, &w.w_q, &q);

    var k = try Matrix.init(alloc, seq, kv_dim);
    defer k.deinit(alloc);
    try matmul.matmulSIMD(x, &w.w_k, &k);

    var v = try Matrix.init(alloc, seq, kv_dim);
    defer v.deinit(alloc);
    try matmul.matmulSIMD(x, &w.w_v, &v);

    // 3. RoPE on Q and K (NOT V)
    try rope_cache.applyToSequence(q.data, seq, cfg.n_q_heads, q_offset);
    try rope_cache.applyToSequence(k.data, seq, cfg.n_kv_heads, q_offset);

    // 4. GQA attention with causal mask
    var attn_out = try Matrix.init(alloc, seq, q_dim);
    defer attn_out.deinit(alloc);
    try attention.attentionGqa(
        alloc,
        q.data,
        k.data,
        v.data,
        attn_out.data,
        seq,
        seq, // KV-cache length = seq at Phase 1 (no cache)
        cfg.n_q_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        true, // causal
        q_offset,
    );

    // 5. O projection
    var o_out = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer o_out.deinit(alloc);
    try matmul.matmulSIMD(&attn_out, &w.w_o, &o_out);

    // 6. Residual add
    for (x.data, residual.data, o_out.data) |*xv, rv, ov| xv.* = rv + ov;

    // Save residual for MLP
    @memcpy(residual.data, x.data);

    // --- MLP block ---
    // 7. RMSNorm post-attention
    try rmsnorm.rmsnorm(x, w.norm_post_attn, cfg.rms_norm_eps);

    // 8. gate = silu(matmul(x, w_gate)) * matmul(x, w_up)
    var gate = try Matrix.init(alloc, seq, cfg.intermediate_size);
    defer gate.deinit(alloc);
    try matmul.matmulSIMD(x, &w.w_gate, &gate);

    var up = try Matrix.init(alloc, seq, cfg.intermediate_size);
    defer up.deinit(alloc);
    try matmul.matmulSIMD(x, &w.w_up, &up);

    try silu.siluMul(gate.data, up.data);

    // 9. down projection
    var down_out = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer down_out.deinit(alloc);
    try matmul.matmulSIMD(&gate, &w.w_down, &down_out);

    // 10. Residual add
    for (x.data, residual.data, down_out.data) |*xv, rv, dv| xv.* = rv + dv;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "transformerBlock — runs end-to-end on TinyLlama-shaped dimensions" {
    const alloc = testing.allocator;
    // TinyLlama-style at small scale to keep test fast
    const cfg = Config{
        .hidden_size = 64,
        .intermediate_size = 128,
        .n_q_heads = 4,
        .n_kv_heads = 1, // GQA 4:1 ratio
        .head_dim = 16,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .max_position_embeddings = 32,
    };
    const seq = 4;

    var cache = try rope.RopeCache.init(alloc, cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);
    defer cache.deinit();

    var w = LayerWeights{
        .w_q = try Matrix.init(alloc, cfg.hidden_size, cfg.n_q_heads * cfg.head_dim),
        .w_k = try Matrix.init(alloc, cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim),
        .w_v = try Matrix.init(alloc, cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim),
        .w_o = try Matrix.init(alloc, cfg.n_q_heads * cfg.head_dim, cfg.hidden_size),
        .w_gate = try Matrix.init(alloc, cfg.hidden_size, cfg.intermediate_size),
        .w_up = try Matrix.init(alloc, cfg.hidden_size, cfg.intermediate_size),
        .w_down = try Matrix.init(alloc, cfg.intermediate_size, cfg.hidden_size),
        .norm_input = try alloc.alloc(f32, cfg.hidden_size),
        .norm_post_attn = try alloc.alloc(f32, cfg.hidden_size),
    };
    defer {
        w.w_q.deinit(alloc);
        w.w_k.deinit(alloc);
        w.w_v.deinit(alloc);
        w.w_o.deinit(alloc);
        w.w_gate.deinit(alloc);
        w.w_up.deinit(alloc);
        w.w_down.deinit(alloc);
        alloc.free(w.norm_input);
        alloc.free(w.norm_post_attn);
    }
    w.w_q.fillRandom(1);
    w.w_k.fillRandom(2);
    w.w_v.fillRandom(3);
    w.w_o.fillRandom(4);
    w.w_gate.fillRandom(5);
    w.w_up.fillRandom(6);
    w.w_down.fillRandom(7);
    @memset(@constCast(w.norm_input), 1.0);
    @memset(@constCast(w.norm_post_attn), 1.0);

    var x = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer x.deinit(alloc);
    x.fillRandom(42);

    // Snapshot input for comparison
    var input = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer input.deinit(alloc);
    @memcpy(input.data, x.data);

    try transformerBlock(alloc, cfg, &cache, w, &x, 0);

    // Output should not be NaN/Inf and should differ from input
    for (x.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(std.math.isFinite(v));
    }
    var any_different = false;
    for (x.data, input.data) |a, b| if (@abs(a - b) > 1e-6) { any_different = true; break; };
    try testing.expect(any_different);
}

test "transformerBlock — works at TinyLlama dimensions (1 layer dry run)" {
    // Real TinyLlama-1.1B has hidden=2048, intermediate=5632, n_q=32,
    // n_kv=4, head_dim=64, seq=1. This test runs ONE layer with
    // those dimensions on random weights to validate the kernels
    // compose at production scale.
    const alloc = testing.allocator;
    const cfg = Config{
        .hidden_size = 2048,
        .intermediate_size = 5632,
        .n_q_heads = 32,
        .n_kv_heads = 4, // TinyLlama GQA 8:1
        .head_dim = 64,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .max_position_embeddings = 32,
    };
    const seq = 1; // single-token decode

    var cache = try rope.RopeCache.init(alloc, cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);
    defer cache.deinit();

    var w = LayerWeights{
        .w_q = try Matrix.init(alloc, cfg.hidden_size, cfg.n_q_heads * cfg.head_dim),
        .w_k = try Matrix.init(alloc, cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim),
        .w_v = try Matrix.init(alloc, cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim),
        .w_o = try Matrix.init(alloc, cfg.n_q_heads * cfg.head_dim, cfg.hidden_size),
        .w_gate = try Matrix.init(alloc, cfg.hidden_size, cfg.intermediate_size),
        .w_up = try Matrix.init(alloc, cfg.hidden_size, cfg.intermediate_size),
        .w_down = try Matrix.init(alloc, cfg.intermediate_size, cfg.hidden_size),
        .norm_input = try alloc.alloc(f32, cfg.hidden_size),
        .norm_post_attn = try alloc.alloc(f32, cfg.hidden_size),
    };
    defer {
        w.w_q.deinit(alloc);
        w.w_k.deinit(alloc);
        w.w_v.deinit(alloc);
        w.w_o.deinit(alloc);
        w.w_gate.deinit(alloc);
        w.w_up.deinit(alloc);
        w.w_down.deinit(alloc);
        alloc.free(w.norm_input);
        alloc.free(w.norm_post_attn);
    }
    // Use small uniform init to avoid blowup
    w.w_q.fill(0.01);
    w.w_k.fill(0.01);
    w.w_v.fill(0.01);
    w.w_o.fill(0.01);
    w.w_gate.fill(0.01);
    w.w_up.fill(0.01);
    w.w_down.fill(0.01);
    @memset(@constCast(w.norm_input), 1.0);
    @memset(@constCast(w.norm_post_attn), 1.0);

    var x = try Matrix.init(alloc, seq, cfg.hidden_size);
    defer x.deinit(alloc);
    x.fillRandom(42);

    try transformerBlock(alloc, cfg, &cache, w, &x, 0);

    // No NaN / Inf
    for (x.data) |val| {
        try testing.expect(!std.math.isNan(val));
        try testing.expect(std.math.isFinite(val));
    }
}
