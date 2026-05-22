//! End-to-end TinyLlama-1.1B inference smoke test.
//!
//! Loads the real model + tokenizer from
//! ../safetensors-zig/tests/fixtures/tinyllama-real/{model.safetensors,tokenizer.json,config.json},
//! tokenizes a prompt, runs ONE forward pass, prints the greedy
//! next-token. Times each phase.
//!
//! This is the v0.0.2 closing artifact: real weights through real
//! kernels through real tokenizer, end-to-end, on CPU.
//!
//! Usage:
//!   zig build infer-tinyllama
//!   (with prompt-override: --args "Your prompt here")

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
    const prompt = iter.next() orelse "Hello, world!";

    std.debug.print("=== vLLM-zig v0.0.2 — TinyLlama inference smoke ===\n", .{});
    std.debug.print("prompt: {s}\n\n", .{prompt});

    // --- Load config.json ---
    const t0 = std.Io.Clock.awake.now(io);
    var cfg_path_buf: [256]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{MODEL_DIR});
    const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, alloc, std.Io.Limit.limited(1024 * 1024));
    defer alloc.free(cfg_bytes);
    const cfg = try vllm.model.Config.fromJson(alloc, cfg_bytes);
    const t1 = std.Io.Clock.awake.now(io);
    std.debug.print("[1/5] loaded config.json in {d} us\n", .{
        @as(u64, @intCast(t0.durationTo(t1).nanoseconds)) / 1000,
    });
    std.debug.print("      vocab={d} hidden={d} layers={d} n_q={d} n_kv={d} head_dim={d}\n", .{
        cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers,
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
    });

    // --- Load tokenizer.json ---
    var tok_path_buf: [256]u8 = undefined;
    const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{MODEL_DIR});
    const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tok_path, alloc, std.Io.Limit.limited(64 * 1024 * 1024));
    defer alloc.free(tok_bytes);
    var tok = try tokenizers.fromJson(alloc, tok_bytes);
    defer tok.deinit();
    const t2 = std.Io.Clock.awake.now(io);
    std.debug.print("[2/5] loaded tokenizer.json in {d} ms\n", .{
        @as(u64, @intCast(t1.durationTo(t2).nanoseconds)) / 1_000_000,
    });

    // --- Open + load safetensors model ---
    var sf_path_buf: [256]u8 = undefined;
    const sf_path = try std.fmt.bufPrint(&sf_path_buf, "{s}/model.safetensors", .{MODEL_DIR});
    var st = try safetensors.SafeTensors.open(alloc, io, sf_path);
    defer st.deinit();

    std.debug.print("[3/5] safetensors handle opened (lazy header parse)\n", .{});
    std.debug.print("      header={d}B tensors={d}\n", .{ st.header_size, st.tensors.len });

    var model = try vllm.model.Model.loadFromSafeTensors(alloc, &st, cfg);
    defer model.deinit();
    const t3 = std.Io.Clock.awake.now(io);
    std.debug.print("[4/5] loaded {d} layers + embed + lm_head in {d} ms\n", .{
        cfg.num_hidden_layers,
        @as(u64, @intCast(t2.durationTo(t3).nanoseconds)) / 1_000_000,
    });

    // --- Tokenize prompt ---
    const ids = try tok.encode(alloc, prompt);
    defer alloc.free(ids);

    // Llama needs BOS token prepended. TinyLlama BOS = 1 (<s>).
    const ids_with_bos = try alloc.alloc(u32, ids.len + 1);
    defer alloc.free(ids_with_bos);
    ids_with_bos[0] = 1;
    @memcpy(ids_with_bos[1..], ids);

    std.debug.print("      tokens: [", .{});
    for (ids_with_bos, 0..) |id, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{id});
    }
    std.debug.print("] ({d} tokens incl. BOS)\n", .{ids_with_bos.len});

    // --- Forward pass + sample ---
    const logits = try model.forward(alloc, ids_with_bos, null, 0);
    defer alloc.free(logits);
    const t4 = std.Io.Clock.awake.now(io);
    std.debug.print("[5/5] forward pass over {d} tokens in {d} ms ({d} ms/token)\n", .{
        ids_with_bos.len,
        @as(u64, @intCast(t3.durationTo(t4).nanoseconds)) / 1_000_000,
        @as(u64, @intCast(t3.durationTo(t4).nanoseconds)) / 1_000_000 / ids_with_bos.len,
    });

    const next_id = try vllm.sampler.greedy(logits);
    std.debug.print("\nnext token (greedy): id={d}\n", .{next_id});

    // Decode just the next token
    var next_ids = [_]u32{next_id};
    const next_text = try tok.decode(alloc, &next_ids);
    defer alloc.free(next_text);
    std.debug.print("decoded:             \"{s}\"\n", .{next_text});

    // Top-5 alternatives
    std.debug.print("\ntop-5 logits (greedy + 4 runners-up):\n", .{});
    const SortItem = struct { id: u32, logit: f32 };
    const items = try alloc.alloc(SortItem, logits.len);
    defer alloc.free(items);
    for (logits, 0..) |l, i| items[i] = .{ .id = @intCast(i), .logit = l };
    std.sort.pdq(SortItem, items, {}, struct {
        fn lt(_: void, a: SortItem, b: SortItem) bool {
            return a.logit > b.logit;
        }
    }.lt);
    for (items[0..@min(5, items.len)]) |item| {
        var one = [_]u32{item.id};
        const txt = try tok.decode(alloc, &one);
        defer alloc.free(txt);
        std.debug.print("  id={d:>5} logit={d:>8.4}  \"{s}\"\n", .{ item.id, item.logit, txt });
    }
}
