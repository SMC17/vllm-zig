//! End-to-end TinyLlama-1.1B inference with KV cache + multi-thread +
//! streaming generate.
//!
//! v0.0.3 demonstration: prompt → multi-thread prefill → streaming
//! decode with KV cache, tokens-per-second reported.
//!
//! Usage:
//!   zig build infer-tinyllama
//!   zig build infer-tinyllama -- "your prompt"
//!   zig build infer-tinyllama -- "your prompt" 30   (generate 30 tokens)

const std = @import("std");
const vllm = @import("vllm");
const safetensors = @import("safetensors");
const tokenizers = @import("tokenizers");

const MODEL_DIR = "../safetensors-zig/tests/fixtures/tinyllama-real";

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const prompt = iter.next() orelse "The capital of France is";
    const max_new_str = iter.next() orelse "16";
    const max_new = try std.fmt.parseInt(usize, max_new_str, 10);

    const n_threads: usize = std.Thread.getCpuCount() catch 4;

    std.debug.print("=== vLLM-zig v0.0.3 — TinyLlama inference w/ KV cache + multi-thread ===\n", .{});
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

    // --- Open + load model ---
    var sf_path_buf: [256]u8 = undefined;
    const sf_path = try std.fmt.bufPrint(&sf_path_buf, "{s}/model.safetensors", .{MODEL_DIR});
    var st = try safetensors.SafeTensors.open(alloc, io, sf_path);
    defer st.deinit();

    const t_load_start = std.Io.Clock.awake.now(io);
    var model = try vllm.model.Model.loadFromSafeTensors(alloc, &st, cfg);
    defer model.deinit();
    model.setThreads(n_threads);
    var pool: ?*vllm.thread_pool.Pool = null;
    if (@import("builtin").os.tag == .linux) {
        pool = vllm.thread_pool.Pool.init(alloc, n_threads) catch null;
    }
    if (pool) |p| {
        model.setThreadPool(p);
    }
    defer if (pool) |p| p.deinit();
    const t_load_end = std.Io.Clock.awake.now(io);
    const load_ms = @as(u64, @intCast(t_load_start.durationTo(t_load_end).nanoseconds)) / 1_000_000;
    std.debug.print("model loaded in {d} ms (BF16→F32 parallel + {d} layers)\n", .{ load_ms, cfg.num_hidden_layers });

    // --- Tokenize prompt with BOS ---
    const ids = try tok.encode(alloc, prompt);
    defer alloc.free(ids);
    const ids_with_bos = try alloc.alloc(u32, ids.len + 1);
    defer alloc.free(ids_with_bos);
    ids_with_bos[0] = 1; // BOS
    @memcpy(ids_with_bos[1..], ids);

    std.debug.print("prompt tokens: {d}\n", .{ids_with_bos.len});

    // --- KV cache ---
    // Paged KV: enough blocks for max_position_embeddings tokens in every layer.
    const BLOCK_SIZE = vllm.page_manager.PageManager.BLOCK_SIZE;
    const kv_blocks = cfg.num_hidden_layers * ((cfg.max_position_embeddings + BLOCK_SIZE - 1) / BLOCK_SIZE);
    var pm = try vllm.page_manager.PageManager.init(alloc, kv_blocks, cfg.num_key_value_heads, cfg.head_dim);
    defer pm.deinit();
    var ba = try vllm.paged_attention.BlockAllocator.init(alloc, kv_blocks);
    defer ba.deinit();
    var kv = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, &pm, &ba);
    defer kv.deinit();

    // --- Generate ---
    std.debug.print("\n--- generating (prefill + decode) ---\n", .{});
    std.debug.print("{s}", .{prompt});

    const t_gen_start = std.Io.Clock.awake.now(io);
    var token_count: usize = 0;
    var first_token_ns: u64 = 0;

    const Stream = struct {
        var t_gen_start_inner: std.Io.Timestamp = undefined;
        var io_inner: std.Io = undefined;
        var tok_inner: *tokenizers.Tokenizer = undefined;
        var alloc_inner: std.mem.Allocator = undefined;
        var first_ns_inner: *u64 = undefined;
        var count_inner: *usize = undefined;

        fn cb(token: u32, idx: usize) bool {
            const now = std.Io.Clock.awake.now(io_inner);
            const ns: u64 = @intCast(t_gen_start_inner.durationTo(now).nanoseconds);
            if (idx == 0) first_ns_inner.* = ns;
            count_inner.* = idx + 1;
            // Decode the single token
            var one = [_]u32{token};
            const text = tok_inner.decode(alloc_inner, &one) catch return true;
            defer alloc_inner.free(text);
            std.debug.print("{s}", .{text});
            // Stop on EOS (id=2 in Llama)
            return token != 2;
        }
    };
    Stream.t_gen_start_inner = t_gen_start;
    Stream.io_inner = io;
    Stream.tok_inner = &tok;
    Stream.alloc_inner = alloc;
    Stream.first_ns_inner = &first_token_ns;
    Stream.count_inner = &token_count;

    const generated = try model.generate(alloc, ids_with_bos, max_new, &kv, &Stream.cb);
    defer alloc.free(generated);
    const t_gen_end = std.Io.Clock.awake.now(io);
    const gen_ns: u64 = @intCast(t_gen_start.durationTo(t_gen_end).nanoseconds);

    std.debug.print("\n--- end of generation ---\n", .{});
    std.debug.print("\ntokens generated:    {d}\n", .{generated.len});
    std.debug.print("first token (prefill+1):  {d} ms\n", .{first_token_ns / 1_000_000});
    std.debug.print("total generation time:    {d} ms\n", .{gen_ns / 1_000_000});
    if (generated.len > 1) {
        const decode_only_ns = gen_ns - first_token_ns;
        const tokens_after_first = generated.len - 1;
        std.debug.print("decode-only ms/token:     {d}\n", .{decode_only_ns / 1_000_000 / tokens_after_first});
        std.debug.print("decode-only tokens/sec:   {d}\n", .{tokens_after_first * std.time.ns_per_s / decode_only_ns});
    }
}
