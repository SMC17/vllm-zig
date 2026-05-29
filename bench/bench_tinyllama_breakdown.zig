//! TinyLlama decode region-by-region breakdown.
//!
//! Uses the Instrumentation infrastructure in vllm.model to attribute
//! each decode step's wall time to a named region (embed, qkv_proj,
//! attention, lm_head, etc.). Answers "what fraction of decode is
//! lm_head?" so the cblas attack can be re-tried with quantified
//! upside instead of vibes.
//!
//! Run: zig build bench-tinyllama-breakdown -Doptimize=ReleaseFast

const std = @import("std");
const vllm = @import("vllm");
const safetensors = @import("safetensors");
const tokenizers = @import("tokenizers");

const MODEL_DIR = "../safetensors-zig/tests/fixtures/tinyllama-real";
const MAX_NEW: usize = 8;
const WARMUP_TOKENS: usize = 2;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var cfg_path_buf: [256]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{MODEL_DIR});
    const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, alloc, std.Io.Limit.limited(1024 * 1024));
    defer alloc.free(cfg_bytes);
    const cfg = try vllm.model.Config.fromJson(alloc, cfg_bytes);

    var st_path_buf: [256]u8 = undefined;
    const st_path = try std.fmt.bufPrint(&st_path_buf, "{s}/model.safetensors", .{MODEL_DIR});
    var st = try safetensors.SafeTensors.open(alloc, io, st_path);
    defer st.deinit();

    var model = try vllm.model.Model.loadFromSafeTensors(alloc, &st, cfg);
    defer model.deinit();
    const n_threads: usize = std.Thread.getCpuCount() catch 4;
    model.setThreads(n_threads);
    var pool = try vllm.Pool.init(alloc, n_threads);
    defer pool.deinit();
    model.setThreadPool(pool);

    // Optional v0.0.7 lever: route lm_head through Pool.matmulBF16Auto
    // (col-split at M=1). Toggle via VLLM_POOLED_LM_HEAD=1.
    const env_pool_lm = init.environ_map.get("VLLM_POOLED_LM_HEAD");
    const pool_lm_on = env_pool_lm != null and !std.mem.eql(u8, env_pool_lm.?, "0");
    model.setPooledBF16LMHead(pool_lm_on);
    std.debug.print("pooled lm_head: {s}\n", .{if (pool_lm_on) "ON" else "off"});

    var tok_path_buf: [256]u8 = undefined;
    const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{MODEL_DIR});
    const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tok_path, alloc, std.Io.Limit.limited(16 * 1024 * 1024));
    defer alloc.free(tok_bytes);
    var tok = try tokenizers.fromJson(alloc, tok_bytes);
    defer tok.deinit();

    const prompt = "Hello, my name is";
    const ids = try tok.encode(alloc, prompt);
    defer alloc.free(ids);

    const bos: u32 = 1;
    const ids_with_bos = try alloc.alloc(u32, ids.len + 1);
    defer alloc.free(ids_with_bos);
    ids_with_bos[0] = bos;
    @memcpy(ids_with_bos[1..], ids);

    var pm = try vllm.page_manager.PageManager.init(alloc, 2000, cfg.num_key_value_heads, cfg.head_dim);
    defer pm.deinit();
    var ba = try vllm.paged_attention.BlockAllocator.init(alloc, 2000);
    defer ba.deinit();
    var kv = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, &pm, &ba);
    defer kv.deinit();

    // Warmup: prefill + a few tokens, without instrumentation
    const noop_cb: ?*const fn (u32, usize) bool = null;
    const warm = try model.generate(alloc, ids_with_bos, WARMUP_TOKENS, &kv, noop_cb);
    alloc.free(warm);

    // Reset state and attach instrumentation
    var inst: vllm.model.Instrumentation = .{};
    model.setInstrumentation(&inst);

    // Fresh KV cache for timed run so KV-cache size matches warmup state
    // doesn't bias one direction.
    var pm2 = try vllm.page_manager.PageManager.init(alloc, 2000, cfg.num_key_value_heads, cfg.head_dim);
    defer pm2.deinit();
    var ba2 = try vllm.paged_attention.BlockAllocator.init(alloc, 2000);
    defer ba2.deinit();
    var kv2 = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, &pm2, &ba2);
    defer kv2.deinit();

    inst.reset();
    const gen = try model.generate(alloc, ids_with_bos, MAX_NEW, &kv2, noop_cb);
    defer alloc.free(gen);

    model.setInstrumentation(null);

    const total_ns = inst.totalNs();
    std.debug.print("=== TinyLlama-1.1B decode breakdown ===\n", .{});
    std.debug.print("n_forward_calls   = {d}\n", .{inst.n_calls});
    std.debug.print("sum(regions) ns   = {d}\n", .{total_ns});
    std.debug.print("sum ms per token  = {d}\n", .{total_ns / inst.n_calls / 1_000_000});
    std.debug.print("\n", .{});
    std.debug.print("  region              |   total_ms |  per_call_us | pct_of_decode\n", .{});
    std.debug.print("  --------------------|-----------:|-------------:|--------------:\n", .{});

    for (vllm.model.region_names, 0..) |name, i| {
        const ns = inst.totals_ns[i];
        const total_ms = ns / 1_000_000;
        const per_call_us = ns / inst.n_calls / 1_000;
        const pct: f64 = @as(f64, @floatFromInt(ns)) * 100.0 / @as(f64, @floatFromInt(total_ns));
        std.debug.print("  {s:<19} | {d:>10} | {d:>12} | {d:>13.2}\n", .{ name, total_ms, per_call_us, pct });
    }
}
