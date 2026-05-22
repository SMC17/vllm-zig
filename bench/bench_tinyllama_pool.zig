//! End-to-end TinyLlama-1.1B inference: pool vs no-pool head-to-head.
//!
//! Loads TinyLlama once, then runs the SAME prompt + generate config twice:
//!   (A) no pool — falls through to matmulSIMDAuto (v0.0.4 ad-hoc spawn)
//!   (B) with persistent thread pool (v0.0.5)
//!
//! Prints prefill, first-token-latency, decode tokens/sec, and total time
//! for each. Goal: validate the synthetic 2.47x decode improvement at the
//! end-to-end forward-pass level.

const std = @import("std");
const vllm = @import("vllm");
const safetensors = @import("safetensors");
const tokenizers = @import("tokenizers");

const MODEL_DIR = "../safetensors-zig/tests/fixtures/tinyllama-real";

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var arg_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer arg_iter.deinit();
    _ = arg_iter.next();
    const prompt = arg_iter.next() orelse "The capital of France is";
    const max_new_str = arg_iter.next() orelse "8";
    const max_new = try std.fmt.parseInt(usize, max_new_str, 10);

    const n_threads: usize = std.Thread.getCpuCount() catch 4;

    std.debug.print("=== vllm-zig — TinyLlama pool vs no-pool bench ===\n", .{});
    std.debug.print("prompt:     {s}\n", .{prompt});
    std.debug.print("max_new:    {d}\n", .{max_new});
    std.debug.print("n_threads:  {d}\n\n", .{n_threads});

    // --- Load config + tokenizer ---
    var cfg_path_buf: [256]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{MODEL_DIR});
    const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, alloc, std.Io.Limit.limited(1024 * 1024));
    defer alloc.free(cfg_bytes);
    const cfg = try vllm.model.Config.fromJson(alloc, cfg_bytes);

    var tok_path_buf: [256]u8 = undefined;
    const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{MODEL_DIR});
    const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tok_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(tok_bytes);
    var tok = try tokenizers.fromJson(alloc, tok_bytes);
    defer tok.deinit();

    var sf_path_buf: [256]u8 = undefined;
    const sf_path = try std.fmt.bufPrint(&sf_path_buf, "{s}/model.safetensors", .{MODEL_DIR});
    var st = try safetensors.SafeTensors.open(alloc, io, sf_path);
    defer st.deinit();

    // --- Load model ---
    const t_load_start = std.Io.Clock.awake.now(io);
    var model = try vllm.model.Model.loadFromSafeTensors(alloc, &st, cfg);
    defer model.deinit();
    model.setThreads(n_threads);
    const t_load_end = std.Io.Clock.awake.now(io);
    const load_ms = @as(u64, @intCast(t_load_start.durationTo(t_load_end).nanoseconds)) / 1_000_000;
    std.debug.print("model loaded in {d} ms\n\n", .{load_ms});

    // --- Tokenize prompt with BOS ---
    const ids = try tok.encode(alloc, prompt);
    defer alloc.free(ids);
    const ids_with_bos = try alloc.alloc(u32, ids.len + 1);
    defer alloc.free(ids_with_bos);
    ids_with_bos[0] = 1;
    @memcpy(ids_with_bos[1..], ids);

    std.debug.print("prompt tokens: {d}\n\n", .{ids_with_bos.len});

    const G = struct {
        var first_ns: u64 = 0;
        var count: usize = 0;
        var t_start: std.Io.Timestamp = undefined;
        var io_local: std.Io = undefined;

        fn cb(token: u32, idx: usize) bool {
            const now = std.Io.Clock.awake.now(io_local);
            const ns: u64 = @intCast(t_start.durationTo(now).nanoseconds);
            if (idx == 0) first_ns = ns;
            count = idx + 1;
            return token != 2;
        }

        fn reset() void {
            first_ns = 0;
            count = 0;
        }
    };

    // --- Run A: no pool (v0.0.4 ad-hoc spawn path) ---
    std.debug.print("--- (A) no pool — falls through to matmulSIMDAuto (v0.0.4) ---\n", .{});
    G.reset();
    G.io_local = io;
    G.t_start = std.Io.Clock.awake.now(io);
    var kv_a = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim);
    defer kv_a.deinit();
    const gen_a = try model.generate(alloc, ids_with_bos, max_new, &kv_a, &G.cb);
    defer alloc.free(gen_a);
    const t_a_end = std.Io.Clock.awake.now(io);
    const a_total_ns: u64 = @intCast(G.t_start.durationTo(t_a_end).nanoseconds);
    const a_first_ms = G.first_ns / 1_000_000;
    const a_total_ms = a_total_ns / 1_000_000;
    const a_decode_only_ns: u64 = if (gen_a.len > 1) a_total_ns - G.first_ns else 0;
    const a_decode_per_tok_ms: u64 = if (gen_a.len > 1) (a_decode_only_ns / (gen_a.len - 1)) / 1_000_000 else 0;
    std.debug.print("  tokens generated: {d}\n", .{gen_a.len});
    std.debug.print("  first-token (prefill+1): {d} ms\n", .{a_first_ms});
    std.debug.print("  total: {d} ms\n", .{a_total_ms});
    std.debug.print("  decode-only ms/tok: {d}\n\n", .{a_decode_per_tok_ms});

    // --- Run B: with persistent thread pool (v0.0.5) ---
    std.debug.print("--- (B) with persistent thread pool (v0.0.5) ---\n", .{});
    var pool = try vllm.Pool.init(alloc, n_threads);
    defer pool.deinit();
    model.setThreadPool(pool);

    G.reset();
    G.t_start = std.Io.Clock.awake.now(io);
    var kv_b = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim);
    defer kv_b.deinit();
    const gen_b = try model.generate(alloc, ids_with_bos, max_new, &kv_b, &G.cb);
    defer alloc.free(gen_b);
    const t_b_end = std.Io.Clock.awake.now(io);
    const b_total_ns: u64 = @intCast(G.t_start.durationTo(t_b_end).nanoseconds);
    const b_first_ms = G.first_ns / 1_000_000;
    const b_total_ms = b_total_ns / 1_000_000;
    const b_decode_only_ns: u64 = if (gen_b.len > 1) b_total_ns - G.first_ns else 0;
    const b_decode_per_tok_ms: u64 = if (gen_b.len > 1) (b_decode_only_ns / (gen_b.len - 1)) / 1_000_000 else 0;
    std.debug.print("  tokens generated: {d}\n", .{gen_b.len});
    std.debug.print("  first-token (prefill+1): {d} ms\n", .{b_first_ms});
    std.debug.print("  total: {d} ms\n", .{b_total_ms});
    std.debug.print("  decode-only ms/tok: {d}\n\n", .{b_decode_per_tok_ms});

    // --- Summary ---
    std.debug.print("=== summary ===\n", .{});
    std.debug.print("  metric          | (A) no pool | (B) pool    | speedup\n", .{});
    std.debug.print("  ----------------|-------------|-------------|--------\n", .{});
    std.debug.print("  first-token ms  | {d:>11} | {d:>11} | {d:.2}x\n", .{ a_first_ms, b_first_ms, if (b_first_ms > 0) @as(f64, @floatFromInt(a_first_ms)) / @as(f64, @floatFromInt(b_first_ms)) else 0.0 });
    std.debug.print("  total ms        | {d:>11} | {d:>11} | {d:.2}x\n", .{ a_total_ms, b_total_ms, if (b_total_ms > 0) @as(f64, @floatFromInt(a_total_ms)) / @as(f64, @floatFromInt(b_total_ms)) else 0.0 });
    std.debug.print("  decode ms/tok   | {d:>11} | {d:>11} | {d:.2}x\n", .{ a_decode_per_tok_ms, b_decode_per_tok_ms, if (b_decode_per_tok_ms > 0) @as(f64, @floatFromInt(a_decode_per_tok_ms)) / @as(f64, @floatFromInt(b_decode_per_tok_ms)) else 0.0 });
}
