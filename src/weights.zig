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
    DimensionMismatch,
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

    bf16ToF32Parallel(bytes, m.data, 0);
    return m;
}

/// Parallel BF16 LE bytes → F32 conversion. Splits the work across
/// std.Thread.getCpuCount() threads (or n_threads if non-zero).
/// Bandwidth-bound on the byte-shuffle so multi-thread gives near-
/// linear scaling on modern CPUs.
pub fn bf16ToF32Parallel(bytes: []const u8, out: []f32, n_threads: usize) void {
    const numel = out.len;
    if (numel < 1024) {
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const lo: u16 = bytes[i * 2];
            const hi: u16 = bytes[i * 2 + 1];
            const bits: u16 = (hi << 8) | lo;
            out[i] = bf16_mod.bf16ToF32(bits);
        }
        return;
    }
    const T = if (n_threads == 0) (std.Thread.getCpuCount() catch 4) else n_threads;
    const T_cap = @min(T, 16);
    const chunk = (numel + T_cap - 1) / T_cap;

    const Args = struct {
        bytes: []const u8,
        out: []f32,
        start: usize,
        end: usize,
    };

    var threads: [16]std.Thread = undefined;
    var args: [16]Args = undefined;
    var spawned: usize = 0;
    var t: usize = 0;
    while (t < T_cap) : (t += 1) {
        const start = t * chunk;
        const end = @min(start + chunk, numel);
        if (start >= numel) break;
        args[t] = .{ .bytes = bytes, .out = out, .start = start, .end = end };
        threads[t] = std.Thread.spawn(.{}, struct {
            fn run(a: *Args) void {
                var i = a.start;
                while (i < a.end) : (i += 1) {
                    const lo: u16 = a.bytes[i * 2];
                    const hi: u16 = a.bytes[i * 2 + 1];
                    const bits: u16 = (hi << 8) | lo;
                    a.out[i] = bf16_mod.bf16ToF32(bits);
                }
            }
        }.run, .{&args[t]}) catch {
            // Fall back: do this chunk on the calling thread.
            var i = args[t].start;
            while (i < args[t].end) : (i += 1) {
                const lo: u16 = args[t].bytes[i * 2];
                const hi: u16 = args[t].bytes[i * 2 + 1];
                const bits: u16 = (hi << 8) | lo;
                out[i] = bf16_mod.bf16ToF32(bits);
            }
            continue;
        };
        spawned += 1;
    }
    var j: usize = 0;
    while (j < spawned) : (j += 1) threads[j].join();
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
    // Vectors are typically small (a few thousand for norm weights);
    // single-thread is fine.
    var i: usize = 0;
    while (i < numel) : (i += 1) {
        const lo: u16 = bytes[i * 2];
        const hi: u16 = bytes[i * 2 + 1];
        const bits: u16 = (hi << 8) | lo;
        out[i] = bf16_mod.bf16ToF32(bits);
    }
    return out;
}

/// Raw BF16 matrix view + dimensions. The bytes from safetensors are
/// already little-endian u16, so we just pun them. Caller owns the
/// returned slice.
///
/// Layout follows HF convention: [rows, cols] = [out_features, in_features]
/// — the SAME shape as the on-disk safetensors tensor (no transpose).
/// For Llama projection weights you typically want the transposed form
/// (so x @ W gives [seq, out]); see loadMatrixBF16RawTransposed.
pub const RawBF16 = struct {
    data: []u16,
    rows: usize,
    cols: usize,

    pub fn deinit(self: *RawBF16, alloc: std.mem.Allocator) void {
        alloc.free(self.data);
    }
};

pub fn loadMatrixBF16Raw(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    name: []const u8,
) !RawBF16 {
    const t = st.get(name) orelse return Error.TensorNotFound;
    if (t.dtype != .bf16) return Error.UnexpectedDtype;
    if (t.shape.len != 2) return Error.ExpectedMatrix;

    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    const numel: usize = rows * cols;

    const bytes = try alloc.alloc(u8, numel * 2);
    defer alloc.free(bytes);
    try st.readTensorBytes(t, bytes);

    const out = try alloc.alloc(u16, numel);
    errdefer alloc.free(out);
    var i: usize = 0;
    while (i < numel) : (i += 1) {
        const lo: u16 = bytes[i * 2];
        const hi: u16 = bytes[i * 2 + 1];
        out[i] = (hi << 8) | lo;
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

/// Load a BF16 matrix as raw u16, then transpose it. Result is
/// [cols, rows] in row-major, so a downstream matmul A[seq, cols] @
/// B[cols, rows] = C[seq, rows] reproduces x @ W.T for HF-layout weights.
pub fn loadMatrixBF16RawTransposed(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    name: []const u8,
) !RawBF16 {
    const raw = try loadMatrixBF16Raw(alloc, st, name);
    defer alloc.free(raw.data);

    const out = try alloc.alloc(u16, raw.rows * raw.cols);
    errdefer alloc.free(out);
    var r: usize = 0;
    while (r < raw.rows) : (r += 1) {
        var c: usize = 0;
        while (c < raw.cols) : (c += 1) {
            out[c * raw.rows + r] = raw.data[r * raw.cols + c];
        }
    }
    return .{ .data = out, .rows = raw.cols, .cols = raw.rows };
}

/// Load two BF16 matrices (gate_proj and up_proj in Llama MLP) and
/// produce a single fused [in_features, 2 * out_features] raw BF16
/// matrix where each input-feature row holds [gate_row | up_row]
/// concatenated horizontally. Used by the v0.0.8 gate+up fusion path
/// so forward() can dispatch one matmul instead of two against the
/// same input x. Both source matrices must share their input-feature
/// dim. The output layout matches matmul_bf16.matmulBF16SIMD's
/// expectations: rows = K (input dim), cols = 2N (concatenated outputs).
pub fn loadMatrixBF16RawConcatTransposed(
    alloc: std.mem.Allocator,
    st: *safetensors.SafeTensors,
    gate_name: []const u8,
    up_name: []const u8,
) !RawBF16 {
    var gate = try loadMatrixBF16RawTransposed(alloc, st, gate_name);
    defer alloc.free(gate.data);
    var up = try loadMatrixBF16RawTransposed(alloc, st, up_name);
    defer alloc.free(up.data);

    if (gate.rows != up.rows or gate.cols != up.cols) return Error.DimensionMismatch;

    const K = gate.rows;
    const N = gate.cols;
    const out = try alloc.alloc(u16, K * 2 * N);
    errdefer alloc.free(out);
    var k: usize = 0;
    while (k < K) : (k += 1) {
        const gate_src = gate.data[k * N ..][0..N];
        const up_src = up.data[k * N ..][0..N];
        @memcpy(out[k * 2 * N ..][0..N], gate_src);
        @memcpy(out[k * 2 * N + N ..][0..N], up_src);
    }
    return .{ .data = out, .rows = K, .cols = 2 * N };
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
