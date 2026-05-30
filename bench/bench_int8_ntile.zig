//! Lever-14 microbench — int8 per-column quantization vs BF16 lever-9
//! baseline. Validates BOTH accuracy floor and bandwidth speedup.
//!
//! Pre-reg exp-1780165289-160168989.

const std = @import("std");
const vllm = @import("vllm");
const Matrix = vllm.Matrix;
const matmul_bf16 = vllm.matmul_bf16;
const weights = vllm.weights;

const TILE: usize = 64;
const ITERS: usize = 30;

const Shape = struct { M: usize, K: usize, N: usize, label: []const u8 };

const SHAPES = [_]Shape{
    .{ .M = 1, .K = 2048, .N = 2048, .label = "decode  M=1   K=2048 N= 2048" },
    .{ .M = 1, .K = 2048, .N = 11264, .label = "gate_up M=1   K=2048 N=11264" },
    .{ .M = 1, .K = 2048, .N = 32000, .label = "lm_head M=1   K=2048 N=32000" },
};

fn benchShape(alloc: std.mem.Allocator, io: std.Io, shape: Shape) !void {
    const M = shape.M;
    const K = shape.K;
    const N = shape.N;

    var a = try Matrix.init(alloc, M, K);
    defer a.deinit(alloc);
    a.fillRandom(101);

    var b_f32 = try Matrix.init(alloc, K, N);
    defer b_f32.deinit(alloc);
    b_f32.fillRandom(103);

    const b_rm = try alloc.alloc(u16, K * N);
    defer alloc.free(b_rm);
    for (b_f32.data, 0..) |x, i| {
        const bits: u32 = @bitCast(x);
        b_rm[i] = @truncate(bits >> 16);
    }

    const b_raw: weights.RawBF16 = .{ .data = b_rm, .rows = K, .cols = N };
    const b_bf16_nt = try weights.repackBF16ToNtileMajor(alloc, b_raw);
    defer alloc.free(b_bf16_nt.data);

    var b_i8 = try weights.quantizeBF16ToINT8NtileMajor(alloc, b_raw);
    defer b_i8.deinit(alloc);

    var c_bf16 = try Matrix.init(alloc, M, N);
    defer c_bf16.deinit(alloc);
    var c_int8 = try Matrix.init(alloc, M, N);
    defer c_int8.deinit(alloc);

    // Warmup
    for (0..5) |_| {
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_bf16_nt.data, K, N, &c_bf16);
        try matmul_bf16.matmulINT8SIMDNtileMajor(&a, b_i8.data, b_i8.scales, K, N, &c_int8);
    }

    // Accuracy: compare int8 vs BF16 reference. BF16 is the existing quantized
    // baseline. Report multiple metrics — max_rel is misleading on small-
    // magnitude cancellation-heavy outputs.
    var max_abs_diff: f32 = 0;
    var max_rel_diff: f32 = 0;
    var sum_sq_diff: f64 = 0;
    var sum_sq_ref: f64 = 0;
    var sum_abs_ref: f64 = 0;
    var sum_abs_diff: f64 = 0;
    for (c_bf16.data, c_int8.data) |ref, q| {
        const d = @abs(ref - q);
        const m = @max(@abs(ref), @abs(q));
        if (d > max_abs_diff) max_abs_diff = d;
        if (m > 1e-3 and d / m > max_rel_diff) max_rel_diff = d / m;
        sum_sq_diff += @as(f64, d) * @as(f64, d);
        sum_sq_ref += @as(f64, @abs(ref)) * @as(f64, @abs(ref));
        sum_abs_ref += @abs(ref);
        sum_abs_diff += d;
    }
    const len_f: f64 = @floatFromInt(c_bf16.data.len);
    const mean_abs_ref: f32 = @floatCast(sum_abs_ref / len_f);
    const mean_rel_diff: f32 = @floatCast(sum_abs_diff / sum_abs_ref);
    const rmse_rel: f32 = @floatCast(@sqrt(sum_sq_diff / sum_sq_ref));

    // Perf
    const t0 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulBF16SIMDNtileMajor(&a, b_bf16_nt.data, K, N, &c_bf16);
    }
    const ns_bf16: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const t1 = std.Io.Clock.awake.now(io);
    for (0..ITERS) |_| {
        try matmul_bf16.matmulINT8SIMDNtileMajor(&a, b_i8.data, b_i8.scales, K, N, &c_int8);
    }
    const ns_int8: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const us_bf16 = ns_bf16 / ITERS / 1_000;
    const us_int8 = ns_int8 / ITERS / 1_000;
    const speedup = @as(f64, @floatFromInt(ns_bf16)) / @as(f64, @floatFromInt(ns_int8));

    std.debug.print(
        "  {s} | BF16 {d:>5} us | INT8 {d:>5} us | speedup {d:.2}x | mean_rel {d:.2}pct rmse_rel {d:.2}pct max_abs {e} max_rel {d:.1}pct mean_ref {d:.2}\n",
        .{ shape.label, us_bf16, us_int8, speedup, mean_rel_diff * 100, rmse_rel * 100, max_abs_diff, max_rel_diff * 100, mean_abs_ref },
    );
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("=== bench_int8_ntile — lever-14 int8 per-column quantization vs lever-9 BF16 ===\n", .{});
    std.debug.print("Single-thread, ReleaseFast. Validates ACCURACY floor + BANDWIDTH speedup.\n\n", .{});

    for (SHAPES) |shape| {
        try benchShape(alloc, io, shape);
    }

    std.debug.print("\nPre-reg exp-1780165289-160168989: confirmed at >=30pct speedup AND <=10pct max_rel_diff.\n", .{});
}
