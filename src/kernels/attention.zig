//! Multi-head + Grouped-Query Attention (GQA).
//!
//! GQA generalises multi-head attention by allowing fewer K/V heads
//! than Q heads. TinyLlama-1.1B uses 32 Q heads and 4 KV heads (so
//! each KV head is shared by 8 Q heads). Standard Llama-3.2-1B is
//! also 32 Q / 8 KV (4:1 sharing). The kernel handles all GQA ratios
//! including the trivial multi-head case (n_q == n_kv).
//!
//! Inputs (row-major):
//!   q : [seq_q, n_q_heads,  head_dim]
//!   k : [seq_kv, n_kv_heads, head_dim]
//!   v : [seq_kv, n_kv_heads, head_dim]
//! Output:
//!   out: [seq_q, n_q_heads, head_dim]
//!
//! For decode-with-KV-cache, the caller supplies the K and V tables
//! that include both prefilled history and the current step's
//! contribution. seq_q is typically 1 in decode, seq_q ~= prefill
//! length in prefill.
//!
//! Per-head scoring: scores[i, j] = (q[i] · k[j]) / sqrt(head_dim)
//! Causal masking: q-position p attends only to k-positions <= p.
//! Output: out[i] = sum_j softmax(scores[i, :])[j] * v[j]

const std = @import("std");
const softmax = @import("softmax.zig");
const pm_mod = @import("../page_manager.zig");

pub const Error = error{
    DimensionMismatch,
    InvalidGqaRatio,
    OutOfMemory,
};

/// Compute Q[i] @ K[j].T for one Q-head + one KV-head over all seq positions.
/// Returns scores: [seq_q, seq_kv] (caller-provided).
fn scoresOneHead(
    q: []const f32, // [seq_q, head_dim]
    k: []const f32, // [seq_kv, head_dim]
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    out_scores: []f32, // [seq_q * seq_kv]
) void {
    const Vec = @Vector(8, f32);
    const LANE = 8;
    const tail = head_dim - (head_dim % LANE);
    var i: usize = 0;
    while (i < seq_q) : (i += 1) {
        const q_row = q[i * head_dim ..][0..head_dim];
        var j: usize = 0;
        while (j < seq_kv) : (j += 1) {
            const k_row = k[j * head_dim ..][0..head_dim];
            // dot product
            var sum_vec: Vec = @splat(0.0);
            var d: usize = 0;
            while (d < tail) : (d += LANE) {
                const qv: Vec = q_row[d..][0..LANE].*;
                const kv: Vec = k_row[d..][0..LANE].*;
                sum_vec += qv * kv;
            }
            var dot: f32 = @reduce(.Add, sum_vec);
            while (d < head_dim) : (d += 1) dot += q_row[d] * k_row[d];
            out_scores[i * seq_kv + j] = dot * scale;
        }
    }
}

/// Apply causal mask in-place to a [seq_q, seq_kv] scores matrix.
/// q_offset is the absolute position of q[0] (for prefill it's 0;
/// for decode it's the KV-cache length before this step).
fn causalMask(scores: []f32, seq_q: usize, seq_kv: usize, q_offset: usize) void {
    var i: usize = 0;
    while (i < seq_q) : (i += 1) {
        const q_abs_pos = q_offset + i;
        var j: usize = 0;
        while (j < seq_kv) : (j += 1) {
            if (j > q_abs_pos) {
                scores[i * seq_kv + j] = -std.math.inf(f32);
            }
        }
    }
}

/// Compute out[i] = sum_j scores_softmax[i, j] * v[j] for one head.
/// scores_softmax: [seq_q, seq_kv], post-softmax (each row sums to 1).
fn aggregate(
    scores_softmax: []const f32,
    v: []const f32, // [seq_kv, head_dim]
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    out: []f32, // [seq_q, head_dim]
) void {
    @memset(out, 0.0);
    var i: usize = 0;
    while (i < seq_q) : (i += 1) {
        const out_row = out[i * head_dim ..][0..head_dim];
        var j: usize = 0;
        while (j < seq_kv) : (j += 1) {
            const w = scores_softmax[i * seq_kv + j];
            const v_row = v[j * head_dim ..][0..head_dim];
            var d: usize = 0;
            while (d < head_dim) : (d += 1) {
                out_row[d] += w * v_row[d];
            }
        }
    }
}

/// Full GQA attention. Allocates scratch internally; caller owns
/// the output buffer.
pub fn attentionGqa(
    alloc: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    seq_q: usize,
    seq_kv: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    q_offset: usize,
) Error!void {
    if (n_q_heads % n_kv_heads != 0) return Error.InvalidGqaRatio;
    const group = n_q_heads / n_kv_heads;
    if (q.len != seq_q * n_q_heads * head_dim) return Error.DimensionMismatch;
    if (k.len != seq_kv * n_kv_heads * head_dim) return Error.DimensionMismatch;
    if (v.len != seq_kv * n_kv_heads * head_dim) return Error.DimensionMismatch;
    if (out.len != seq_q * n_q_heads * head_dim) return Error.DimensionMismatch;

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));

    // Scratch: [seq_q, seq_kv] for scores per head
    const scores = try alloc.alloc(f32, seq_q * seq_kv);
    defer alloc.free(scores);

    // Per-head Q / K / V slices (reshape on the fly)
    // The naive layout is [seq, head, dim] row-major; for one head we
    // need to gather row q[t, h, :] for t in 0..seq_q. Walk linearly.
    const q_head_buf = try alloc.alloc(f32, seq_q * head_dim);
    defer alloc.free(q_head_buf);
    const k_head_buf = try alloc.alloc(f32, seq_kv * head_dim);
    defer alloc.free(k_head_buf);
    const v_head_buf = try alloc.alloc(f32, seq_kv * head_dim);
    defer alloc.free(v_head_buf);
    const out_head_buf = try alloc.alloc(f32, seq_q * head_dim);
    defer alloc.free(out_head_buf);

    var qh: usize = 0;
    while (qh < n_q_heads) : (qh += 1) {
        const kvh = qh / group; // GQA: map Q head → KV head

        // Gather q for this head
        var t: usize = 0;
        while (t < seq_q) : (t += 1) {
            const src = q[(t * n_q_heads + qh) * head_dim ..][0..head_dim];
            const dst = q_head_buf[t * head_dim ..][0..head_dim];
            @memcpy(dst, src);
        }
        // Gather K, V for this kvh
        t = 0;
        while (t < seq_kv) : (t += 1) {
            const k_src = k[(t * n_kv_heads + kvh) * head_dim ..][0..head_dim];
            const v_src = v[(t * n_kv_heads + kvh) * head_dim ..][0..head_dim];
            @memcpy(k_head_buf[t * head_dim ..][0..head_dim], k_src);
            @memcpy(v_head_buf[t * head_dim ..][0..head_dim], v_src);
        }

        // QK^T scaled
        scoresOneHead(q_head_buf, k_head_buf, seq_q, seq_kv, head_dim, scale, scores);

        // Causal mask
        if (causal) causalMask(scores, seq_q, seq_kv, q_offset);

        // Row-wise softmax
        var ri: usize = 0;
        while (ri < seq_q) : (ri += 1) {
            softmax.softmaxRow(scores[ri * seq_kv ..][0..seq_kv]);
        }

        // Aggregate scores * V
        aggregate(scores, v_head_buf, seq_q, seq_kv, head_dim, out_head_buf);

        // Scatter back
        t = 0;
        while (t < seq_q) : (t += 1) {
            const src = out_head_buf[t * head_dim ..][0..head_dim];
            const dst = out[(t * n_q_heads + qh) * head_dim ..][0..head_dim];
            @memcpy(dst, src);
        }
    }
}

/// Full GQA attention utilizing the PagedAttention Block Allocator
pub fn attentionPagedGqa(
    alloc: std.mem.Allocator,
    q: []const f32,
    pm: *pm_mod.PageManager,
    block_table: []const usize,
    out: []f32,
    seq_q: usize,
    seq_kv: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    q_offset: usize,
) Error!void {
    if (n_q_heads % n_kv_heads != 0) return Error.InvalidGqaRatio;
    const group = n_q_heads / n_kv_heads;
    if (q.len != seq_q * n_q_heads * head_dim) return Error.DimensionMismatch;
    if (out.len != seq_q * n_q_heads * head_dim) return Error.DimensionMismatch;

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));

    const scores = try alloc.alloc(f32, seq_q * seq_kv);
    defer alloc.free(scores);

    const q_head_buf = try alloc.alloc(f32, seq_q * head_dim);
    defer alloc.free(q_head_buf);
    const k_head_buf = try alloc.alloc(f32, seq_kv * head_dim);
    defer alloc.free(k_head_buf);
    const v_head_buf = try alloc.alloc(f32, seq_kv * head_dim);
    defer alloc.free(v_head_buf);
    const out_head_buf = try alloc.alloc(f32, seq_q * head_dim);
    defer alloc.free(out_head_buf);

    var qh: usize = 0;
    while (qh < n_q_heads) : (qh += 1) {
        const kvh = qh / group;

        var t: usize = 0;
        while (t < seq_q) : (t += 1) {
            const src = q[(t * n_q_heads + qh) * head_dim ..][0..head_dim];
            const dst = q_head_buf[t * head_dim ..][0..head_dim];
            @memcpy(dst, src);
        }

        const block_size = pm_mod.PageManager.BLOCK_SIZE;
        t = 0;
        while (t < seq_kv) : (t += 1) {
            const block_idx_in_seq = t / block_size;
            const token_in_block = t % block_size;
            const physical_idx = block_table[block_idx_in_seq];

            const k_block = pm.getBlockK(physical_idx, kvh);
            const v_block = pm.getBlockV(physical_idx, kvh);

            const k_src = k_block[token_in_block * head_dim ..][0..head_dim];
            const v_src = v_block[token_in_block * head_dim ..][0..head_dim];

            @memcpy(k_head_buf[t * head_dim ..][0..head_dim], k_src);
            @memcpy(v_head_buf[t * head_dim ..][0..head_dim], v_src);
        }

        scoresOneHead(q_head_buf, k_head_buf, seq_q, seq_kv, head_dim, scale, scores);
        if (causal) causalMask(scores, seq_q, seq_kv, q_offset);

        var ri: usize = 0;
        while (ri < seq_q) : (ri += 1) {
            softmax.softmaxRow(scores[ri * seq_kv ..][0..seq_kv]);
        }

        aggregate(scores, v_head_buf, seq_q, seq_kv, head_dim, out_head_buf);

        t = 0;
        while (t < seq_q) : (t += 1) {
            const src = out_head_buf[t * head_dim ..][0..head_dim];
            const dst = out[(t * n_q_heads + qh) * head_dim ..][0..head_dim];
            @memcpy(dst, src);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "attention — single head, single token, attends to itself" {
    // seq_q=1 seq_kv=1 n_q=1 n_kv=1 head_dim=4
    // Q=K=V=[1, 0, 0, 0]; result = V = [1, 0, 0, 0]
    const q = [_]f32{ 1, 0, 0, 0 };
    const k = [_]f32{ 1, 0, 0, 0 };
    const v = [_]f32{ 1, 0, 0, 0 };
    var out: [4]f32 = undefined;
    try attentionGqa(testing.allocator, &q, &k, &v, &out, 1, 1, 1, 1, 4, true, 0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
}

test "attention — multi-head equals stacked single-head" {
    // 2 heads, both independent: each head should produce the same
    // result as if it were single-head.
    const head_dim = 4;
    const n_heads = 2;
    var q: [n_heads * head_dim]f32 = undefined;
    var k: [n_heads * head_dim]f32 = undefined;
    var v: [n_heads * head_dim]f32 = undefined;
    @memcpy(q[0..head_dim], &[_]f32{ 1, 0, 0, 0 });
    @memcpy(q[head_dim..], &[_]f32{ 0, 1, 0, 0 });
    @memcpy(k[0..head_dim], &[_]f32{ 1, 0, 0, 0 });
    @memcpy(k[head_dim..], &[_]f32{ 0, 1, 0, 0 });
    @memcpy(v[0..head_dim], &[_]f32{ 5, 0, 0, 0 });
    @memcpy(v[head_dim..], &[_]f32{ 0, 7, 0, 0 });

    var out: [n_heads * head_dim]f32 = undefined;
    try attentionGqa(testing.allocator, &q, &k, &v, &out, 1, 1, n_heads, n_heads, head_dim, true, 0);
    try testing.expectApproxEqAbs(@as(f32, 5.0), out[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 7.0), out[head_dim + 1], 1e-5);
}

test "attention — GQA with n_q=2 n_kv=1 shares KV head" {
    // 2 Q heads, 1 KV head shared. Both Q heads attend to the same K/V.
    const head_dim = 4;
    const n_q = 2;
    const n_kv = 1;
    var q: [n_q * head_dim]f32 = undefined;
    @memcpy(q[0..head_dim], &[_]f32{ 1, 0, 0, 0 });
    @memcpy(q[head_dim..], &[_]f32{ 1, 0, 0, 0 });
    const k = [_]f32{ 1, 0, 0, 0 };
    const v = [_]f32{ 9, 9, 9, 9 };
    var out: [n_q * head_dim]f32 = undefined;
    try attentionGqa(testing.allocator, &q, &k, &v, &out, 1, 1, n_q, n_kv, head_dim, true, 0);
    // Both heads should produce v = [9, 9, 9, 9]
    for (out[0..head_dim]) |x| try testing.expectApproxEqAbs(@as(f32, 9.0), x, 1e-5);
    for (out[head_dim..]) |x| try testing.expectApproxEqAbs(@as(f32, 9.0), x, 1e-5);
}

test "attention — causal mask blocks future tokens" {
    // seq_q=2, seq_kv=2. With causal mask, position 0 cannot attend
    // to position 1. So out[0] = v[0] (after softmax over a single
    // unmasked position).
    const head_dim = 2;
    const n_heads = 1;
    var q: [2 * head_dim]f32 = undefined;
    @memcpy(q[0..head_dim], &[_]f32{ 1, 0 });
    @memcpy(q[head_dim..], &[_]f32{ 1, 0 });
    var k: [2 * head_dim]f32 = undefined;
    @memcpy(k[0..head_dim], &[_]f32{ 1, 0 });
    @memcpy(k[head_dim..], &[_]f32{ 1, 0 });
    var v: [2 * head_dim]f32 = undefined;
    @memcpy(v[0..head_dim], &[_]f32{ 10, 0 });
    @memcpy(v[head_dim..], &[_]f32{ 0, 100 });

    var out: [2 * head_dim]f32 = undefined;
    try attentionGqa(testing.allocator, &q, &k, &v, &out, 2, 2, n_heads, n_heads, head_dim, true, 0);
    // out[0] should be exactly v[0] = [10, 0]
    try testing.expectApproxEqAbs(@as(f32, 10.0), out[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5);
    // out[1] attends to both equally → softmax(0, 0) = (0.5, 0.5) → (5, 50)
    try testing.expectApproxEqAbs(@as(f32, 5.0), out[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 50.0), out[3], 1e-4);
}

test "attention — rejects invalid GQA ratio" {
    var q: [12]f32 = undefined;
    var k: [4]f32 = undefined;
    var v: [4]f32 = undefined;
    var out: [12]f32 = undefined;
    // 3 Q heads but n_kv=2 → 3 % 2 != 0
    try testing.expectError(
        Error.InvalidGqaRatio,
        attentionGqa(testing.allocator, &q, &k, &v, &out, 1, 1, 3, 2, 2, true, 0),
    );
}

test "attention — TinyLlama GQA ratios are valid" {
    // TinyLlama: 32 Q heads, 4 KV heads (8:1)
    // Llama-3.2: 32 Q, 8 KV (4:1)
    try testing.expectEqual(@as(usize, 0), @as(usize, 32) % @as(usize, 4));
    try testing.expectEqual(@as(usize, 0), @as(usize, 32) % @as(usize, 8));
}
