//! BFloat16 ↔ Float32 conversion.
//!
//! BFloat16 is just the upper 16 bits of an IEEE-754 binary32:
//!   sign (1) + exponent (8) + truncated mantissa (7)
//! So conversion is bit-level shift; no rounding ambiguity for
//! BF16 → F32 (it's exact). F32 → BF16 truncates the lower 16 bits.
//!
//! All BF16 storage in safetensors is little-endian on x86_64.

const std = @import("std");

pub inline fn bf16ToF32(bf16_bits: u16) f32 {
    const x: u32 = @as(u32, bf16_bits) << 16;
    return @bitCast(x);
}

pub inline fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    // Round-to-nearest-even (banker's rounding). The standard
    // implementation. For the BF16→F32 path we don't need this,
    // but include it for completeness.
    const rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    return @truncate(rounded);
}

/// Convert a packed BF16 byte buffer (little-endian) into an F32
/// slice. Caller owns the returned slice.
pub fn bf16BytesToF32(alloc: std.mem.Allocator, bf16_bytes: []const u8) ![]f32 {
    if (bf16_bytes.len % 2 != 0) return error.OddByteCount;
    const numel = bf16_bytes.len / 2;
    const out = try alloc.alloc(f32, numel);
    errdefer alloc.free(out);
    var i: usize = 0;
    while (i < numel) : (i += 1) {
        const lo: u16 = bf16_bytes[i * 2];
        const hi: u16 = bf16_bytes[i * 2 + 1];
        const bits: u16 = (hi << 8) | lo;
        out[i] = bf16ToF32(bits);
    }
    return out;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "bf16ToF32 — 1.0 round-trip" {
    // 1.0 in BF16 is 0x3F80 (same as F32 high bits)
    try testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
}

test "bf16ToF32 — 0.0 round-trip" {
    try testing.expectEqual(@as(f32, 0.0), bf16ToF32(0x0000));
}

test "bf16ToF32 — -1.0 round-trip" {
    try testing.expectEqual(@as(f32, -1.0), bf16ToF32(0xBF80));
}

test "bf16ToF32 — 2.0" {
    // BF16(2.0) = 0x4000
    try testing.expectEqual(@as(f32, 2.0), bf16ToF32(0x4000));
}

test "f32ToBf16 round-trip for exact-representable values" {
    inline for ([_]f32{ 0.0, 1.0, 2.0, -1.0, 0.5, 0.25 }) |v| {
        const back = bf16ToF32(f32ToBf16(v));
        try testing.expectEqual(v, back);
    }
}

test "f32ToBf16 truncation tolerance for arbitrary values" {
    // 0.1 isn't exact in either; check that round-trip is within
    // BF16 quantization tolerance (~1% relative)
    const x: f32 = 0.1;
    const back = bf16ToF32(f32ToBf16(x));
    try testing.expectApproxEqAbs(x, back, 0.01);
}

test "bf16BytesToF32 — small slice" {
    const alloc = testing.allocator;
    // 4 BF16 values: 1.0, 2.0, 0.0, -1.0 (LE: 80 3F  00 40  00 00  80 BF)
    const bytes = [_]u8{ 0x80, 0x3F, 0x00, 0x40, 0x00, 0x00, 0x80, 0xBF };
    const out = try bf16BytesToF32(alloc, &bytes);
    defer alloc.free(out);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 0.0, -1.0 }, out);
}

test "bf16BytesToF32 — odd byte count rejects" {
    const alloc = testing.allocator;
    const bytes = [_]u8{ 0x80, 0x3F, 0x00 };
    try testing.expectError(error.OddByteCount, bf16BytesToF32(alloc, &bytes));
}
