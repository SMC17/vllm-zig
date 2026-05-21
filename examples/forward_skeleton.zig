//! Forward-pass skeleton — demonstrates how the v0.0.1 kernels
//! compose into a Llama-style transformer block. NOT a runnable
//! inference path (no weights loaded, no real model).
//!
//! Intent: lock the kernel-orchestration pattern + show that the
//! API surface composes cleanly before any actual model integration
//! work begins.
//!
//! A single Llama-style transformer block is:
//!
//!   x = rmsnorm_input(x)
//!   q = matmul(x, w_q)
//!   k = matmul(x, w_k)
//!   v = matmul(x, w_v)
//!   # (apply RoPE to q, k -- not implemented in v0.0.1)
//!   attn = softmax(q @ k^T / sqrt(d)) @ v
//!   x = x + matmul(attn, w_o)
//!   x = rmsnorm_attn(x)
//!   gate = silu(matmul(x, w_gate))
//!   up = matmul(x, w_up)
//!   x = x + matmul(gate * up, w_down)
//!
//! What's missing for a real forward pass:
//!   - RoPE rotation (we have a Triton reference at ~/triton-rope;
//!     the Zig port lands in v0.1)
//!   - KV-cache management
//!   - Causal masking
//!   - Weight loading (via safetensors-zig)
//!   - Tokenization (via tokenizers-zig)

const std = @import("std");
const vllm = @import("vllm");

const Matrix = vllm.Matrix;

const Config = struct {
    hidden: usize, // model dim, e.g. 2048 for Llama-3.2-1B
    intermediate: usize, // MLP dim, e.g. 8192
    n_heads: usize,
    head_dim: usize,
    eps: f32,
};

const TransformerBlockWeights = struct {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
    norm_input: []f32, // [hidden]
    norm_attn: []f32, // [hidden]
};

/// Run one transformer block forward over a [seq, hidden] activation
/// matrix. In-place on `x`. This is a *skeleton* — it omits causal
/// masking, RoPE, and KV cache. Real inference cannot use this as-is.
fn forwardBlockSkeleton(
    alloc: std.mem.Allocator,
    cfg: Config,
    x: *Matrix,
    w: TransformerBlockWeights,
) !void {
    const seq = x.rows;

    // --- Self-attention ---
    // RMSNorm on input
    try vllm.rmsnorm.rmsnorm(x, w.norm_input, cfg.eps);

    // Q = x @ w_q
    var q = try Matrix.init(alloc, seq, cfg.hidden);
    defer q.deinit(alloc);
    try vllm.matmul.matmulSIMD(x, &w.w_q, &q);

    var k = try Matrix.init(alloc, seq, cfg.hidden);
    defer k.deinit(alloc);
    try vllm.matmul.matmulSIMD(x, &w.w_k, &k);

    var v = try Matrix.init(alloc, seq, cfg.hidden);
    defer v.deinit(alloc);
    try vllm.matmul.matmulSIMD(x, &w.w_v, &v);

    // (TODO: RoPE on q, k — v0.1)
    // (TODO: split into n_heads, per-head attention with causal mask — v0.1)

    // For the skeleton we collapse the multi-head structure to a
    // single hidden-by-hidden attention: scores = Q @ K^T then
    // softmax then @ V.
    var scores = try Matrix.init(alloc, seq, seq);
    defer scores.deinit(alloc);
    // Build K^T into k_t
    var k_t = try Matrix.init(alloc, cfg.hidden, seq);
    defer k_t.deinit(alloc);
    var r: usize = 0;
    while (r < seq) : (r += 1) {
        var c: usize = 0;
        while (c < cfg.hidden) : (c += 1) {
            k_t.set(c, r, k.at(r, c));
        }
    }
    try vllm.matmul.matmulSIMD(&q, &k_t, &scores);

    // Scale + softmax per row
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    r = 0;
    while (r < seq) : (r += 1) {
        const row = scores.rowMut(r);
        for (row) |*s| s.* *= scale;
        vllm.softmax.softmaxRow(row);
    }

    // attn = scores @ v
    var attn = try Matrix.init(alloc, seq, cfg.hidden);
    defer attn.deinit(alloc);
    try vllm.matmul.matmulSIMD(&scores, &v, &attn);

    // x = x + attn @ w_o
    var attn_out = try Matrix.init(alloc, seq, cfg.hidden);
    defer attn_out.deinit(alloc);
    try vllm.matmul.matmulSIMD(&attn, &w.w_o, &attn_out);
    for (x.data, attn_out.data) |*xv, av| xv.* += av;

    // --- MLP ---
    try vllm.rmsnorm.rmsnorm(x, w.norm_attn, cfg.eps);

    var gate = try Matrix.init(alloc, seq, cfg.intermediate);
    defer gate.deinit(alloc);
    try vllm.matmul.matmulSIMD(x, &w.w_gate, &gate);

    var up = try Matrix.init(alloc, seq, cfg.intermediate);
    defer up.deinit(alloc);
    try vllm.matmul.matmulSIMD(x, &w.w_up, &up);

    // gate = silu(gate) * up
    try vllm.silu.siluMul(gate.data, up.data);

    var down_out = try Matrix.init(alloc, seq, cfg.hidden);
    defer down_out.deinit(alloc);
    try vllm.matmul.matmulSIMD(&gate, &w.w_down, &down_out);
    for (x.data, down_out.data) |*xv, dv| xv.* += dv;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;

    // Tiny shapes for the skeleton run — proves the wiring works
    // without trying to be a real model.
    const cfg = Config{
        .hidden = 32,
        .intermediate = 64,
        .n_heads = 4,
        .head_dim = 8,
        .eps = 1e-5,
    };
    const seq = 8;

    // Allocate weights as random for the skeleton run.
    var w = TransformerBlockWeights{
        .w_q = try Matrix.init(alloc, cfg.hidden, cfg.hidden),
        .w_k = try Matrix.init(alloc, cfg.hidden, cfg.hidden),
        .w_v = try Matrix.init(alloc, cfg.hidden, cfg.hidden),
        .w_o = try Matrix.init(alloc, cfg.hidden, cfg.hidden),
        .w_gate = try Matrix.init(alloc, cfg.hidden, cfg.intermediate),
        .w_up = try Matrix.init(alloc, cfg.hidden, cfg.intermediate),
        .w_down = try Matrix.init(alloc, cfg.intermediate, cfg.hidden),
        .norm_input = try alloc.alloc(f32, cfg.hidden),
        .norm_attn = try alloc.alloc(f32, cfg.hidden),
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
        alloc.free(w.norm_attn);
    }
    w.w_q.fillRandom(1);
    w.w_k.fillRandom(2);
    w.w_v.fillRandom(3);
    w.w_o.fillRandom(4);
    w.w_gate.fillRandom(5);
    w.w_up.fillRandom(6);
    w.w_down.fillRandom(7);
    for (w.norm_input) |*v| v.* = 1.0;
    for (w.norm_attn) |*v| v.* = 1.0;

    var x = try Matrix.init(alloc, seq, cfg.hidden);
    defer x.deinit(alloc);
    x.fillRandom(42);

    std.debug.print("forward_skeleton: seq={d} hidden={d} intermediate={d}\n", .{ seq, cfg.hidden, cfg.intermediate });
    std.debug.print("input[0][0..4] = [{d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        x.at(0, 0), x.at(0, 1), x.at(0, 2), x.at(0, 3),
    });

    try forwardBlockSkeleton(alloc, cfg, &x, w);

    std.debug.print("output[0][0..4] = [{d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        x.at(0, 0), x.at(0, 1), x.at(0, 2), x.at(0, 3),
    });
    std.debug.print("(values are random-weight nonsense; this is a wiring smoke test)\n", .{});
}
