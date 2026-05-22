//! Weight loader — converts BF16 tensors from a safetensors-zig
//! SafeTensors handle into F32 Matrix / vector for the vLLM-zig
//! kernel substrate.
//!
//! Memory: doubles the tensor footprint (BF16 = 2 bytes, F32 = 4 bytes).
//! For TinyLlama-1.1B that's 2.2 GB BF16 → ~4.4 GB F32. Acceptable on
//! consumer Linux with 8+ GB RAM. v0.0.3 will add an in-place BF16
//! kernel path (matmul that reads BF16 weights + F32 activations) to
//! halve the working set.

const std = @import("std");
const safetensors = @import("safetensors");
const bf16_mod = @import("bf16.zig");
const tensor = @import("tensor.zig");
const Matrix = tensor.Matrix;

pub const Error = error{
    TensorNotFound,
    UnexpectedDtype,
    ExpectedMatrix,
    ExpectedVector,
    OutOfMemory,
};

/// Load a [rows, cols] BF16 matrix and convert to F32. Caller owns the
/// returned Matrix.
pub fn loadMatrixBF16(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    name: []const u8,
) !Matrix {
    const t = st.get(name) orelse return Error.TensorNotFound;
    if (t.dtype != .bf16) return Error.UnexpectedDtype;
    if (t.shape.len != 2) return Error.ExpectedMatrix;

    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    var m = try Matrix.init(alloc, rows, cols);
    errdefer m.deinit(alloc);

    const numel: usize = @intCast(t.numel());
    const bytes = try alloc.alloc(u8, numel * 2);
    defer alloc.free(bytes);
    try st.readTensorBytes(t, bytes);

    var i: usize = 0;
    while (i < numel) : (i += 1) {
        const lo: u16 = bytes[i * 2];
        const hi: u16 = bytes[i * 2 + 1];
        const bits: u16 = (hi << 8) | lo;
        m.data[i] = bf16_mod.bf16ToF32(bits);
    }
    return m;
}

/// Load a [D] BF16 vector (e.g., layer norm weights) and convert to F32.
pub fn loadVectorBF16(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    name: []const u8,
) ![]f32 {
    const t = st.get(name) orelse return Error.TensorNotFound;
    if (t.dtype != .bf16) return Error.UnexpectedDtype;
    if (t.shape.len != 1) return Error.ExpectedVector;

    const numel: usize = @intCast(t.numel());
    const bytes = try alloc.alloc(u8, numel * 2);
    defer alloc.free(bytes);
    try st.readTensorBytes(t, bytes);

    const out = try alloc.alloc(f32, numel);
    errdefer alloc.free(out);
    var i: usize = 0;
    while (i < numel) : (i += 1) {
        const lo: u16 = bytes[i * 2];
        const hi: u16 = bytes[i * 2 + 1];
        const bits: u16 = (hi << 8) | lo;
        out[i] = bf16_mod.bf16ToF32(bits);
    }
    return out;
}

/// Convenience: load a matrix and transpose it. HF stores attention
/// projection weights in [out_features, in_features] layout, but our
/// matmul expects [in_features, out_features] (so x @ W = result where
/// W.shape[0] == x.shape[1]). For LLama-style weights this is
/// equivalent to transposing on load.
pub fn loadMatrixBF16Transposed(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    name: []const u8,
) !Matrix {
    var m = try loadMatrixBF16(alloc, st, name);
    errdefer m.deinit(alloc);
    var out = try Matrix.init(alloc, m.cols, m.rows);
    errdefer out.deinit(alloc);
    var r: usize = 0;
    while (r < m.rows) : (r += 1) {
        var c: usize = 0;
        while (c < m.cols) : (c += 1) {
            out.set(c, r, m.at(r, c));
        }
    }
    m.deinit(alloc);
    return out;
}
