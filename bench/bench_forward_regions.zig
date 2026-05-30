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
/// When VLLM_AB_TOGGLE is set, the bench runs 2*ROUNDS rounds with the
/// MLP pool flag flipped on alternating odd/even rounds so background
/// load drift hits both A and B halves equally. Reports two separate
/// medians for the off-rounds vs on-rounds.
const AB_ROUNDS: usize = ROUNDS * 2;

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
    const env_pool_mlp = init.environ_map.get("VLLM_POOLED_MLP");
    const env_pool_attn = init.environ_map.get("VLLM_POOLED_ATTN");
    const env_ab = init.environ_map.get("VLLM_AB_TOGGLE");
    const instrument_on = env_instrument != null and !std.mem.eql(u8, env_instrument.?, "0");
    const pool_lm_on = env_pool_lm != null and !std.mem.eql(u8, env_pool_lm.?, "0");
    const pool_mlp_on = env_pool_mlp != null and !std.mem.eql(u8, env_pool_mlp.?, "0");
    const pool_attn_on = env_pool_attn != null and !std.mem.eql(u8, env_pool_attn.?, "0");
    const ab_toggle = env_ab != null and !std.mem.eql(u8, env_ab.?, "0");

    if (!instrument_on) {
        std.debug.print("bench-forward-regions: set VLLM_INSTRUMENT=1 to enable; exiting.\n", .{});
        return;
    }

    std.debug.print("=== vllm-zig -- Model.forward per-region instrumentation ===\n", .{});
    if (ab_toggle) {
        std.debug.print("mode:           A/B toggle (round-by-round MLP off/on)\n", .{});
        std.debug.print("rounds:         {d} ({d} off + {d} on, interleaved)\n", .{ AB_ROUNDS, ROUNDS, ROUNDS });
    } else {
        std.debug.print("mode:           single-condition\n", .{});
        std.debug.print("rounds:         {d}\n", .{ROUNDS});
    }
    std.debug.print("decode tokens:  {d}\n", .{DECODE_TOKENS});
    std.debug.print("pooled lm_head: {s}\n", .{if (pool_lm_on) "ON (VLLM_POOLED_LM_HEAD=1)" else "off (baseline single-thread BF16 SIMD)"});
    if (ab_toggle) {
        std.debug.print("pooled MLP:     A/B toggled (VLLM_AB_TOGGLE=1)\n", .{});
    } else {
        std.debug.print("pooled MLP:     {s}\n", .{if (pool_mlp_on) "ON (VLLM_POOLED_MLP=1, lever-1 retarget)" else "off (baseline F32 pool.matmulAuto)"});
    }
    std.debug.print("pooled attn:    {s}\n\n", .{if (pool_attn_on) "ON (VLLM_POOLED_ATTN=1, lever-2)" else "off (baseline F32 pool.matmulAuto)"});

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
    var model = try vllm.model.Model.loadFromSafeTensorsOpts(alloc, &st, cfg, .{
        .load_mlp_bf16 = pool_mlp_on or ab_toggle,
        .load_attn_bf16 = pool_attn_on,
    });
    defer model.deinit();
    // VLLM_N_THREADS env override (default = logical CPU count).
    // For memory-bandwidth-bound MLP shapes, SMT siblings contend on
    // shared FP/load ports; setting n_threads to the physical-core count
    // (4 on i7-1065G7) often outperforms n_threads=8.
    const env_n_threads = init.environ_map.get("VLLM_N_THREADS");
    const n_threads: usize = if (env_n_threads) |s|
        std.fmt.parseInt(usize, s, 10) catch (std.Thread.getCpuCount() catch 4)
    else
        std.Thread.getCpuCount() catch 4;
    model.setThreads(n_threads);
    const t_load_end = std.Io.Clock.awake.now(io);
    const load_ms = @as(u64, @intCast(t_load_start.durationTo(t_load_end).nanoseconds)) / 1_000_000;
    std.debug.print("model loaded in {d} ms (n_threads={d})\n\n", .{ load_ms, n_threads });

    var pool = try vllm.Pool.init(alloc, n_threads);
    defer pool.deinit();
    model.setThreadPool(pool);
    model.setPooledBF16LMHead(pool_lm_on);
    // Initial MLP state. In A/B mode this gets flipped per round below.
    model.setPooledBF16MLP(pool_mlp_on);
    model.setPooledBF16Attn(pool_attn_on);

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
    var round_totals: [AB_ROUNDS][vllm.model.n_regions]u64 = undefined;
    var round_total_ns: [AB_ROUNDS]u64 = undefined;
    var round_decode_ms_per_tok: [AB_ROUNDS]u64 = undefined;
    var round_mlp_on: [AB_ROUNDS]bool = undefined;

    const n_rounds: usize = if (ab_toggle) AB_ROUNDS else ROUNDS;
    var round: usize = 0;
    while (round < n_rounds) : (round += 1) {
        // In A/B mode, alternate odd=on / even=off so background load
        // drift hits both halves with equal weight. When pool_attn_on
        // is also set the attn flag toggles alongside MLP so the
        // measured ratio reflects the FULL stacked lever, not just MLP.
        if (ab_toggle) {
            const on = (round % 2) == 1;
            model.setPooledBF16MLP(on);
            if (pool_attn_on) model.setPooledBF16Attn(on);
            round_mlp_on[round] = on;
        } else {
            round_mlp_on[round] = pool_mlp_on;
        }

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
            "round {d} [MLP {s}]: prefill {d} ms, total {d} ms, decode {d} ms/tok\n",
            .{ round + 1, if (round_mlp_on[round]) "ON " else "off", prefill_ns / 1_000_000, round_ns / 1_000_000, round_decode_ms_per_tok[round] },
        );
    }

    if (ab_toggle) {
        // Partition rounds into off-bucket and on-bucket, compute
        // per-region medians for each, then report the side-by-side
        // off vs on table. Load drift is shared between buckets so the
        // off-to-on ratio is the real lever signal.
        printAggregateAB(round_totals[0..n_rounds], round_decode_ms_per_tok[0..n_rounds], round_mlp_on[0..n_rounds]);
    } else {
        printAggregateSingle(round_totals[0..n_rounds], round_decode_ms_per_tok[0..n_rounds], n_rounds);
    }
}

fn printAggregateSingle(round_totals: []const [vllm.model.n_regions]u64, round_decode_ms: []u64, n_rounds: usize) void {
    std.debug.print("\n=== per-region totals (ns across prefill + {d} decode steps), per-round samples ===\n", .{DECODE_TOKENS - 1});
    std.debug.print("region              | med ms   | p10 ms   | p90 ms   | share of med-total\n", .{});
    std.debug.print("--------------------+----------+----------+----------+-------------------\n", .{});

    var samples_buf: [AB_ROUNDS]u64 = undefined;
    const samples_total = samples_buf[0..n_rounds];
    for (round_totals, 0..) |rt, r| {
        var s: u64 = 0;
        for (rt) |v| s += v;
        samples_total[r] = s;
    }
    std.mem.sort(u64, samples_total, {}, ltU64);
    const med_total_ns = samples_total[n_rounds / 2];

    var region_idx: usize = 0;
    while (region_idx < vllm.model.n_regions) : (region_idx += 1) {
        var sbuf: [AB_ROUNDS]u64 = undefined;
        const samples = sbuf[0..n_rounds];
        for (round_totals, 0..) |rt, r| samples[r] = rt[region_idx];
        std.mem.sort(u64, samples, {}, ltU64);
        const med = samples[n_rounds / 2];
        const p10 = percentile(samples, 0.10);
        const p90 = percentile(samples, 0.90);
        const share: f64 = if (med_total_ns == 0) 0 else @as(f64, @floatFromInt(med)) * 100.0 / @as(f64, @floatFromInt(med_total_ns));
        std.debug.print(
            "{s:<20}| {d:>8} | {d:>8} | {d:>8} | {d:5.1}%\n",
            .{ vllm.model.region_names[region_idx], med / 1_000_000, p10 / 1_000_000, p90 / 1_000_000, share },
        );
    }
    std.debug.print(
        "{s:<20}| {d:>8} | {d:>8} | {d:>8} | {d:5.1}%\n",
        .{ "TOTAL (sum)", med_total_ns / 1_000_000, @as(u64, 0), @as(u64, 0), 100.0 },
    );

    std.mem.sort(u64, round_decode_ms, {}, ltU64);
    const med_decode = round_decode_ms[n_rounds / 2];
    const p10_decode = percentile(round_decode_ms, 0.10);
    const p90_decode = percentile(round_decode_ms, 0.90);
    std.debug.print("\ndecode ms/tok across {d} rounds: med {d}, p10 {d}, p90 {d}\n", .{ n_rounds, med_decode, p10_decode, p90_decode });
}

fn printAggregateAB(round_totals: []const [vllm.model.n_regions]u64, round_decode_ms: []const u64, round_mlp_on: []const bool) void {
    // Build off/on indices.
    var off_idx_buf: [AB_ROUNDS]usize = undefined;
    var on_idx_buf: [AB_ROUNDS]usize = undefined;
    var n_off: usize = 0;
    var n_on: usize = 0;
    for (round_mlp_on, 0..) |on, r| {
        if (on) {
            on_idx_buf[n_on] = r;
            n_on += 1;
        } else {
            off_idx_buf[n_off] = r;
            n_off += 1;
        }
    }
    const off_idx = off_idx_buf[0..n_off];
    const on_idx = on_idx_buf[0..n_on];

    std.debug.print("\n=== A/B per-region totals (med ms over {d} off-rounds vs {d} on-rounds) ===\n", .{ n_off, n_on });
    std.debug.print("region              | off med  | on med   | ratio on/off | share off | share on\n", .{});
    std.debug.print("--------------------+----------+----------+--------------+-----------+----------\n", .{});

    var off_total_ns: u64 = 0;
    var on_total_ns: u64 = 0;
    {
        var off_sum_buf: [AB_ROUNDS]u64 = undefined;
        var on_sum_buf: [AB_ROUNDS]u64 = undefined;
        for (off_idx, 0..) |r, i| {
            var s: u64 = 0;
            for (round_totals[r]) |v| s += v;
            off_sum_buf[i] = s;
        }
        for (on_idx, 0..) |r, i| {
            var s: u64 = 0;
            for (round_totals[r]) |v| s += v;
            on_sum_buf[i] = s;
        }
        std.mem.sort(u64, off_sum_buf[0..n_off], {}, ltU64);
        std.mem.sort(u64, on_sum_buf[0..n_on], {}, ltU64);
        off_total_ns = off_sum_buf[n_off / 2];
        on_total_ns = on_sum_buf[n_on / 2];
    }

    var region_idx: usize = 0;
    while (region_idx < vllm.model.n_regions) : (region_idx += 1) {
        var off_buf: [AB_ROUNDS]u64 = undefined;
        var on_buf: [AB_ROUNDS]u64 = undefined;
        for (off_idx, 0..) |r, i| off_buf[i] = round_totals[r][region_idx];
        for (on_idx, 0..) |r, i| on_buf[i] = round_totals[r][region_idx];
        std.mem.sort(u64, off_buf[0..n_off], {}, ltU64);
        std.mem.sort(u64, on_buf[0..n_on], {}, ltU64);
        const off_med = off_buf[n_off / 2];
        const on_med = on_buf[n_on / 2];
        const ratio: f64 = if (off_med == 0) 0 else @as(f64, @floatFromInt(on_med)) / @as(f64, @floatFromInt(off_med));
        const share_off: f64 = if (off_total_ns == 0) 0 else @as(f64, @floatFromInt(off_med)) * 100.0 / @as(f64, @floatFromInt(off_total_ns));
        const share_on: f64 = if (on_total_ns == 0) 0 else @as(f64, @floatFromInt(on_med)) * 100.0 / @as(f64, @floatFromInt(on_total_ns));
        std.debug.print(
            "{s:<20}| {d:>8} | {d:>8} | {d:>11.3}x | {d:>8.1}% | {d:>7.1}%\n",
            .{ vllm.model.region_names[region_idx], off_med / 1_000_000, on_med / 1_000_000, ratio, share_off, share_on },
        );
    }
    const total_ratio: f64 = if (off_total_ns == 0) 0 else @as(f64, @floatFromInt(on_total_ns)) / @as(f64, @floatFromInt(off_total_ns));
    std.debug.print(
        "{s:<20}| {d:>8} | {d:>8} | {d:>11.3}x | {d:>8.1}% | {d:>7.1}%\n",
        .{ "TOTAL (sum)", off_total_ns / 1_000_000, on_total_ns / 1_000_000, total_ratio, 100.0, 100.0 },
    );

    // Decode ms/tok aggregate split.
    var off_dec_buf: [AB_ROUNDS]u64 = undefined;
    var on_dec_buf: [AB_ROUNDS]u64 = undefined;
    for (off_idx, 0..) |r, i| off_dec_buf[i] = round_decode_ms[r];
    for (on_idx, 0..) |r, i| on_dec_buf[i] = round_decode_ms[r];
    std.mem.sort(u64, off_dec_buf[0..n_off], {}, ltU64);
    std.mem.sort(u64, on_dec_buf[0..n_on], {}, ltU64);
    const off_med_dec = off_dec_buf[n_off / 2];
    const on_med_dec = on_dec_buf[n_on / 2];
    const dec_ratio: f64 = if (off_med_dec == 0) 0 else @as(f64, @floatFromInt(on_med_dec)) / @as(f64, @floatFromInt(off_med_dec));
    std.debug.print("\ndecode ms/tok: off med {d}, on med {d}, ratio on/off {d:.3}x\n", .{ off_med_dec, on_med_dec, dec_ratio });
}
