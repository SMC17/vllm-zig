//! Per-region instrumentation bench: where does Model.forward time go?
//!
//! Loads TinyLlama-1.1B once, then runs 5 prefill+decode rounds with an
//! Instrumentation accumulator attached. Each round prefills the prompt
//! and decodes 16 tokens (so per-region totals sample 16 decode forward()
//! calls plus one prefill forward()). Per-region medians, p10/p90, and
//! shares of total wall time are reported.
//!
//! Wires the matmul thread pool (v0.0.5 substrate) so the numbers
//! reflect the dispatch path Model.forward actually uses in production
//! decode. lm_head dispatch can be toggled between the single-thread
//! BF16 SIMD path (baseline) and the pooled BF16 dispatch (v0.0.7 lever
//! #1) via the VLLM_POOLED_LM_HEAD=1 env var. When that is set, the
//! report adds a "pool_bf16" column for the lm_head row.
//!
//! Enabled by VLLM_INSTRUMENT=1; if the env var is unset the bench
//! prints a one-line banner and exits without running so the bench
//! step stays cheap to call in scripts.

const std = @import("std");
const vllm = @import("vllm");
const safetensors = @import("safetensors");
const tokenizers = @import("tokenizers");

const MODEL_DIR = "../safetensors-zig/tests/fixtures/tinyllama-real";
const ROUNDS: usize = 5;
const DECODE_TOKENS: usize = 16;

fn ltU64(_: void, a: u64, b: u64) bool {
    return a < b;
}

fn percentile(samples_sorted: []const u64, p: f64) u64 {
    if (samples_sorted.len == 0) return 0;
    const idx_f = p * @as(f64, @floatFromInt(samples_sorted.len - 1));
    const idx: usize = @intFromFloat(@floor(idx_f));
    return samples_sorted[idx];
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    // Env-gate so a default `zig build bench-forward-regions` (no env)
    // is cheap. Avoids accidental 5x model loads in CI.
    const env_instrument = init.environ_map.get("VLLM_INSTRUMENT");
    const env_pool_lm = init.environ_map.get("VLLM_POOLED_LM_HEAD");
    const instrument_on = env_instrument != null and !std.mem.eql(u8, env_instrument.?, "0");
    const pool_lm_on = env_pool_lm != null and !std.mem.eql(u8, env_pool_lm.?, "0");

    if (!instrument_on) {
        std.debug.print("bench-forward-regions: set VLLM_INSTRUMENT=1 to enable; exiting.\n", .{});
        return;
    }

    std.debug.print("=== vllm-zig -- Model.forward per-region instrumentation ===\n", .{});
    std.debug.print("rounds:         {d}\n", .{ROUNDS});
    std.debug.print("decode tokens:  {d}\n", .{DECODE_TOKENS});
    std.debug.print("pooled lm_head: {s}\n\n", .{if (pool_lm_on) "ON (VLLM_POOLED_LM_HEAD=1)" else "off (baseline single-thread BF16 SIMD)"});

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

    const t_load_start = std.Io.Clock.awake.now(io);
    var model = try vllm.model.Model.loadFromSafeTensors(alloc, &st, cfg);
    defer model.deinit();
    const n_threads: usize = std.Thread.getCpuCount() catch 4;
    model.setThreads(n_threads);
    const t_load_end = std.Io.Clock.awake.now(io);
    const load_ms = @as(u64, @intCast(t_load_start.durationTo(t_load_end).nanoseconds)) / 1_000_000;
    std.debug.print("model loaded in {d} ms (n_threads={d})\n\n", .{ load_ms, n_threads });

    var pool = try vllm.Pool.init(alloc, n_threads);
    defer pool.deinit();
    model.setThreadPool(pool);
    model.setPooledBF16LMHead(pool_lm_on);

    const prompt = "The capital of France is";
    const ids = try tok.encode(alloc, prompt);
    defer alloc.free(ids);
    const ids_with_bos = try alloc.alloc(u32, ids.len + 1);
    defer alloc.free(ids_with_bos);
    ids_with_bos[0] = 1;
    @memcpy(ids_with_bos[1..], ids);

    // Per-region per-round totals. round_totals[round][region] = ns
    // accumulated across all forward() calls in that round (1 prefill
    // + DECODE_TOKENS-1 decode steps).
    var round_totals: [ROUNDS][vllm.model.n_regions]u64 = undefined;
    var round_total_ns: [ROUNDS]u64 = undefined;
    var round_decode_ms_per_tok: [ROUNDS]u64 = undefined;

    var round: usize = 0;
    while (round < ROUNDS) : (round += 1) {
        var inst = vllm.model.Instrumentation{};
        model.setInstrumentation(&inst);

        var pm = try vllm.page_manager.PageManager.init(alloc, 2000, cfg.num_key_value_heads, cfg.head_dim);
        defer pm.deinit();
        var ba = try vllm.paged_attention.BlockAllocator.init(alloc, 2000);
        defer ba.deinit();
        var kv = try vllm.kv_cache.KVCache.init(alloc, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, &pm, &ba);
        defer kv.deinit();

        const t_round_start = std.Io.Clock.awake.now(io);

        // Prefill
        const t_prefill_start = std.Io.Clock.awake.now(io);
        const prefill_logits = try model.forward(alloc, ids_with_bos, &kv, 0);
        defer alloc.free(prefill_logits);
        var best_idx: usize = 0;
        var best_val: f32 = prefill_logits[0];
        for (prefill_logits, 0..) |v, idx| if (v > best_val) {
            best_val = v;
            best_idx = idx;
        };
        const t_prefill_end = std.Io.Clock.awake.now(io);
        const prefill_ns: u64 = @intCast(t_prefill_start.durationTo(t_prefill_end).nanoseconds);

        var next_tok: u32 = @intCast(best_idx);
        var pos = ids_with_bos.len;

        // Decode DECODE_TOKENS - 1 more tokens (prefill produced the first).
        var i: usize = 1;
        while (i < DECODE_TOKENS) : (i += 1) {
            const one = [_]u32{next_tok};
            const new_logits = try model.forward(alloc, &one, &kv, pos);
            defer alloc.free(new_logits);
            var bi: usize = 0;
            var bv: f32 = new_logits[0];
            for (new_logits, 0..) |v, idx| if (v > bv) {
                bv = v;
                bi = idx;
            };
            next_tok = @intCast(bi);
            pos += 1;
        }

        const t_round_end = std.Io.Clock.awake.now(io);
        const round_ns: u64 = @intCast(t_round_start.durationTo(t_round_end).nanoseconds);
        round_totals[round] = inst.totals_ns;
        round_total_ns[round] = round_ns;
        const decode_only_ns: u64 = round_ns - prefill_ns;
        round_decode_ms_per_tok[round] = (decode_only_ns / (DECODE_TOKENS - 1)) / 1_000_000;

        model.setInstrumentation(null);

        std.debug.print(
            "round {d}: prefill {d} ms, total {d} ms, decode {d} ms/tok\n",
            .{ round + 1, prefill_ns / 1_000_000, round_ns / 1_000_000, round_decode_ms_per_tok[round] },
        );
    }

    // Aggregate: per-region median + p10/p90 across rounds.
    std.debug.print("\n=== per-region totals (ns across prefill + {d} decode steps), per-round samples ===\n", .{DECODE_TOKENS - 1});
    std.debug.print("region              | med ms   | p10 ms   | p90 ms   | share of med-total\n", .{});
    std.debug.print("--------------------+----------+----------+----------+-------------------\n", .{});

    // total across all regions per round
    var med_total_ns: u64 = 0;
    {
        var samples_total: [ROUNDS]u64 = undefined;
        for (round_totals, 0..) |rt, r| {
            var s: u64 = 0;
            for (rt) |v| s += v;
            samples_total[r] = s;
        }
        std.mem.sort(u64, &samples_total, {}, ltU64);
        med_total_ns = samples_total[ROUNDS / 2];
    }

    var region_idx: usize = 0;
    while (region_idx < vllm.model.n_regions) : (region_idx += 1) {
        var samples: [ROUNDS]u64 = undefined;
        for (round_totals, 0..) |rt, r| samples[r] = rt[region_idx];
        std.mem.sort(u64, &samples, {}, ltU64);
        const med = samples[ROUNDS / 2];
        const p10 = percentile(&samples, 0.10);
        const p90 = percentile(&samples, 0.90);
        const share: f64 = if (med_total_ns == 0) 0 else @as(f64, @floatFromInt(med)) * 100.0 / @as(f64, @floatFromInt(med_total_ns));
        std.debug.print(
            "{s:<20}| {d:>8} | {d:>8} | {d:>8} | {d:5.1}%\n",
            .{
                vllm.model.region_names[region_idx],
                med / 1_000_000,
                p10 / 1_000_000,
                p90 / 1_000_000,
                share,
            },
        );
    }
    std.debug.print(
        "{s:<20}| {d:>8} | {d:>8} | {d:>8} | {d:5.1}%\n",
        .{ "TOTAL (sum)", med_total_ns / 1_000_000, @as(u64, 0), @as(u64, 0), 100.0 },
    );

    // Per-round decode ms/tok aggregate
    std.mem.sort(u64, &round_decode_ms_per_tok, {}, ltU64);
    const med_decode = round_decode_ms_per_tok[ROUNDS / 2];
    const p10_decode = percentile(&round_decode_ms_per_tok, 0.10);
    const p90_decode = percentile(&round_decode_ms_per_tok, 0.90);

    std.debug.print("\ndecode ms/tok across {d} rounds: med {d}, p10 {d}, p90 {d}\n", .{ ROUNDS, med_decode, p10_decode, p90_decode });
}
