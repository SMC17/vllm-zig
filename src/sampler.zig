//! Token sampler: greedy / temperature / top-K.
//!
//! Operates on logits (un-normalised scores). For temperature and top-K
//! the function divides by temperature, applies softmax, and samples
//! from the resulting distribution. For greedy it just argmax.

const std = @import("std");
const softmax = @import("kernels/softmax.zig");

pub const Error = error{
    EmptyLogits,
    InvalidTemperature,
    InvalidK,
};

pub fn greedy(logits: []const f32) Error!u32 {
    if (logits.len == 0) return Error.EmptyLogits;
    var best_idx: usize = 0;
    var best_val: f32 = logits[0];
    var i: usize = 1;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return @intCast(best_idx);
}

/// Temperature sampling: divide logits by temperature, softmax,
/// sample from the resulting distribution.
pub fn temperatureSample(
    alloc: std.mem.Allocator,
    rng: std.Random,
    logits: []const f32,
    temperature: f32,
) !u32 {
    if (logits.len == 0) return Error.EmptyLogits;
    if (temperature <= 0.0) return Error.InvalidTemperature;
    const scaled = try alloc.dupe(f32, logits);
    defer alloc.free(scaled);
    for (scaled) |*x| x.* /= temperature;
    softmax.softmaxRow(scaled);
    return sampleFromProbs(rng, scaled);
}

/// Top-K sampling: keep top K logits, mask the rest to -inf, then softmax + sample.
pub fn topKSample(
    alloc: std.mem.Allocator,
    rng: std.Random,
    logits: []const f32,
    temperature: f32,
    k: usize,
) !u32 {
    if (logits.len == 0) return Error.EmptyLogits;
    if (temperature <= 0.0) return Error.InvalidTemperature;
    if (k == 0) return Error.InvalidK;
    const effective_k = @min(k, logits.len);

    // Find the kth-largest value as the threshold.
    const sorted = try alloc.dupe(f32, logits);
    defer alloc.free(sorted);
    std.sort.pdq(f32, sorted, {}, std.sort.desc(f32));
    const threshold = sorted[effective_k - 1];

    const scaled = try alloc.alloc(f32, logits.len);
    defer alloc.free(scaled);
    for (logits, scaled) |l, *s| {
        if (l >= threshold) {
            s.* = l / temperature;
        } else {
            s.* = -std.math.inf(f32);
        }
    }
    softmax.softmaxRow(scaled);
    return sampleFromProbs(rng, scaled);
}

fn sampleFromProbs(rng: std.Random, probs: []const f32) u32 {
    const u = rng.float(f32);
    var cum: f32 = 0.0;
    for (probs, 0..) |p, i| {
        cum += p;
        if (u < cum) return @intCast(i);
    }
    // Fallback (should not occur except for FP edge cases)
    return @intCast(probs.len - 1);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "greedy picks argmax" {
    const logits = [_]f32{ 0.1, 0.7, 0.3, 0.5 };
    try testing.expectEqual(@as(u32, 1), try greedy(&logits));
}

test "greedy on single element returns 0" {
    const logits = [_]f32{42};
    try testing.expectEqual(@as(u32, 0), try greedy(&logits));
}

test "greedy on empty errors" {
    const logits = [_]f32{};
    try testing.expectError(Error.EmptyLogits, greedy(&logits));
}

test "temperatureSample with temp=very-high tends toward uniform" {
    // Stochastic but with seed
    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();
    const logits = [_]f32{ 10, 0, 0, 0 };
    // At temperature 1000.0 the distribution flattens. Single sample may
    // still pick index 0, but over many samples the distribution should
    // be roughly uniform. Just verify the function returns a valid index.
    const id = try temperatureSample(testing.allocator, rng, &logits, 1000.0);
    try testing.expect(id < 4);
}

test "topKSample with k=1 equals greedy" {
    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();
    const logits = [_]f32{ 0.1, 0.9, 0.5, 0.3 };
    const id = try topKSample(testing.allocator, rng, &logits, 1.0, 1);
    try testing.expectEqual(@as(u32, 1), id);
}

test "topKSample with k=0 errors" {
    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();
    const logits = [_]f32{ 0.1, 0.9 };
    try testing.expectError(Error.InvalidK, topKSample(testing.allocator, rng, &logits, 1.0, 0));
}

test "temperatureSample with temp=0 errors" {
    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();
    const logits = [_]f32{ 0.1, 0.9 };
    try testing.expectError(Error.InvalidTemperature, temperatureSample(testing.allocator, rng, &logits, 0.0));
}
