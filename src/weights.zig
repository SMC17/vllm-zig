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
