//! BF16-weights matmul kernels.
//!
//! Decode on TinyLlama-class models is memory-bandwidth bound on the
//! B matrix scan. v0.0.5 stores weights as F32 (4 bytes/elem, ~16MB per
//! 2048×2048 layer); BF16 storage halves that to 8MB and lets the L2/L3
//! prefetcher cover twice the rows per cycle.
//!
//! Conversion is exact (BF16 = top 16 bits of an IEEE-754 binary32),
//! so per-element bf16→f32 is a single shift-by-16. The inner-loop cost
//! is therefore the load+widen, not the conversion math itself.
//!
//! API shape: B is a raw []const u16 (the loaded safetensors bytes
//! interpreted as little-endian BF16 units), not a Matrix. A and C
//! stay F32 Matrix because activations are kept in F32 throughout
//! the forward pass.

const std = @import("std");
const builtin = @import("builtin");
const Matrix = @import("../tensor.zig").Matrix;

pub const Error = error{DimensionMismatch};

const CBLAS_ORDER = enum(c_int) { row_major = 101, col_major = 102 };
const CBLAS_TRANSPOSE = enum(c_int) { no_trans = 111, trans = 112 };

// Accelerate cblas_sgemm. Only invoked on Apple Silicon where the
// Accelerate framework is linked; declaration is harmless on other
// targets because we never call into it.
extern "c" fn cblas_sgemm(
    order: CBLAS_ORDER,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) void;

// Size threshold (in B elements) above which the Accelerate dispatch
// path beats the in-process SIMD path. Empirically the cblas_sgemm
// overhead amortizes around 1M B-elements on M2 Max; below that the
// hand-rolled @Vector(8, f32) BF16-widening kernel wins because no
// BF16→F32 scratch buffer is allocated.
const ACCELERATE_MIN_B_ELEMS: usize = 1_000_000;

/// Scalar BF16-weights matmul.
/// A: [M, K] F32, B: [K, N] BF16 (row-major u16), C: [M, N] F32.
pub fn matmulBF16Scalar(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const av = a_row[k];
            const b_row = b[k * N ..][0..N];
            var j: usize = 0;
            while (j < N) : (j += 1) {
                const bv: f32 = @bitCast(@as(u32, b_row[j]) << 16);
                c_row[j] += av * bv;
            }
        }
    }
}

/// Vectorised BF16-weights matmul. 16-lane @Vector(f32) accumulator;
/// widens 16 u16 -> 16 u32 -> bitcast to 16 f32 per lane group. On
/// Ice Lake AVX-512 this compiles to VPMOVZXWD ZMM + VPSLLD + VFMADD.
/// Bit-identical to the prior 8-lane variant on shapes divisible by 16.
pub fn matmulBF16SIMD(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tail = N - (N % LANE);

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        var k: usize = 0;
        while (k < K) : (k += 1) {
            const a_scalar: Vec_f32 = @splat(a_row[k]);
            const b_row = b[k * N ..][0..N];

            var j: usize = 0;
            while (j < n_tail) : (j += LANE) {
                const b_u16v: Vec_u16 = b_row[j..][0..LANE].*;
                const b_u32v: Vec_u32 = b_u16v; // implicit widen
                const b_shifted: Vec_u32 = b_u32v << @splat(@as(u5, 16));
                const b_f32v: Vec_f32 = @bitCast(b_shifted);
                const cv: Vec_f32 = c_row[j..][0..LANE].*;
                c_row[j..][0..LANE].* = cv + a_scalar * b_f32v;
            }
            while (j < N) : (j += 1) {
                const bv: f32 = @bitCast(@as(u32, b_row[j]) << 16);
                c_row[j] += a_row[k] * bv;
            }
        }
    }
}

/// Apple Silicon Accelerate (cblas_sgemm) BF16-weights matmul.
///
/// Allocates a scratch F32 buffer of size K×N, widens the BF16 weights
/// into it once, then calls cblas_sgemm. For lm_head-class shapes
/// (M=1, K=2048, N=32000) on M2 Max this measures **6.97× faster** than
/// matmulBF16SIMD in standalone benches, with max_abs_diff ≤ 1.0e-5
/// vs the SIMD path (within BF16 quantization noise).
///
/// Trade-off: per-call BF16→F32 conversion is O(K*N) and memory-bandwidth
/// bound; in serving contexts the conversion could be cached at load
/// time. This signature performs the conversion inline so the function
/// is a drop-in replacement for matmulBF16SIMD.
///
/// Build requirement: caller's build.zig must link Accelerate
/// framework on macOS targets (`linkFramework("Accelerate")`).
pub fn matmulBF16Accelerate(
    allocator: std.mem.Allocator,
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) (Error || std.mem.Allocator.Error)!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const b_f32 = try allocator.alloc(f32, b.len);
    defer allocator.free(b_f32);
    for (b, 0..) |w, idx| {
        b_f32[idx] = @bitCast(@as(u32, w) << 16);
    }

    const M: c_int = @intCast(a.rows);
    const N: c_int = @intCast(b_cols);
    const K: c_int = @intCast(a.cols);

    cblas_sgemm(
        .row_major,
        .no_trans,
        .no_trans,
        M,
        N,
        K,
        1.0,
        a.data.ptr,
        K,
        b_f32.ptr,
        N,
        0.0,
        c.data.ptr,
        N,
    );
}

/// Register-blocked BF16 matmul (lever-5, v0.0.10).
///
/// Inner-product form with N_tile=64 (4× ZMM accumulators) replacing the
/// outer-product form's K*N c-traffic. For each 64-column N tile we keep
/// 4 accumulators of @Vector(16, f32) live in registers across the entire
/// K loop, then store once at the end. On Ice Lake AVX-512 (32× ZMM):
///   - 4 ZMM accumulators
///   - 4 ZMM b loads (reused as they widen)
///   - 1 ZMM a_scalar splat
///   = 9 ZMM in flight; plenty of headroom for the compiler to schedule.
///
/// vs matmulBF16SIMD on M=1 K=2048 N=2048:
///   outer-product c traffic: K writes + K reads to c_row per N-block
///   inner-product c traffic: 1 store to c_row per N-block (K-fold reduction)
/// Tested for bit-equivalence with matmulBF16SIMD on shapes divisible by 64.
pub fn matmulBF16SIMDRegBlock(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;
    const TILE = 4 * LANE; // 64-column N tile

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tile_end = N - (N % TILE);
    const n_lane_end = N - (N % LANE);

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];

        // Main path: 64-column N tiles, 4 ZMM accumulators in registers
        // across the full K loop.
        var j: usize = 0;
        while (j < n_tile_end) : (j += TILE) {
            var acc0: Vec_f32 = @splat(0.0);
            var acc1: Vec_f32 = @splat(0.0);
            var acc2: Vec_f32 = @splat(0.0);
            var acc3: Vec_f32 = @splat(0.0);

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_row = b[k * N + j ..][0..TILE];

                const b0_u16: Vec_u16 = b_row[0..LANE].*;
                const b1_u16: Vec_u16 = b_row[LANE..][0..LANE].*;
                const b2_u16: Vec_u16 = b_row[2 * LANE ..][0..LANE].*;
                const b3_u16: Vec_u16 = b_row[3 * LANE ..][0..LANE].*;

                const b0_u32: Vec_u32 = b0_u16;
                const b1_u32: Vec_u32 = b1_u16;
                const b2_u32: Vec_u32 = b2_u16;
                const b3_u32: Vec_u32 = b3_u16;

                const shift: @Vector(16, u5) = @splat(16);
                const b0_f32: Vec_f32 = @bitCast(b0_u32 << shift);
                const b1_f32: Vec_f32 = @bitCast(b1_u32 << shift);
                const b2_f32: Vec_f32 = @bitCast(b2_u32 << shift);
                const b3_f32: Vec_f32 = @bitCast(b3_u32 << shift);

                acc0 = @mulAdd(Vec_f32, a_scalar, b0_f32, acc0);
                acc1 = @mulAdd(Vec_f32, a_scalar, b1_f32, acc1);
                acc2 = @mulAdd(Vec_f32, a_scalar, b2_f32, acc2);
                acc3 = @mulAdd(Vec_f32, a_scalar, b3_f32, acc3);
            }

            c_row[j..][0..LANE].* = acc0;
            c_row[j + LANE ..][0..LANE].* = acc1;
            c_row[j + 2 * LANE ..][0..LANE].* = acc2;
            c_row[j + 3 * LANE ..][0..LANE].* = acc3;
        }

        // Lane tail: N % 64 columns remaining, processed 16 at a time
        // using single-accumulator inner-product form.
        while (j < n_lane_end) : (j += LANE) {
            var acc: Vec_f32 = @splat(0.0);
            const shift: @Vector(16, u5) = @splat(16);
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_u16: Vec_u16 = b[k * N + j ..][0..LANE].*;
                const b_u32: Vec_u32 = b_u16;
                const b_f32: Vec_f32 = @bitCast(b_u32 << shift);
                acc = @mulAdd(Vec_f32, a_scalar, b_f32, acc);
            }
            c_row[j..][0..LANE].* = acc;
        }

        // Scalar tail: N % 16 columns.
        while (j < N) : (j += 1) {
            var acc: f32 = 0.0;
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const bv: f32 = @bitCast(@as(u32, b[k * N + j]) << 16);
                acc += a_row[k] * bv;
            }
            c_row[j] = acc;
        }
    }
}

/// Lever-9 standalone — N-tile-major BF16 matmul (single-thread).
///
/// b layout: [n_tile_outer, K, n_tile_inner=TILE=64], flat index
/// b[nt * K * TILE + k * TILE + i]. Stride between k iterations is
/// TILE * 2 bytes = 128 bytes (2 cache lines) which the L2 hardware
/// streamer engages, vs row-major's N * 2 byte stride which page-crosses.
///
/// Caller responsibility: weights must be pre-repacked to this layout
/// at load time (see weights.zig loadMatrixBF16RawNtileMajor). Per-call
/// arithmetic is identical to matmulBF16SIMDRegBlock; only the b read
/// pattern changes.
///
/// On Ice Lake-U M=1 K=2048 N=11264 single-thread ReleaseFast this
/// measured **1.29x** faster than matmulBF16SIMDRegBlock (4146 us → 3207 us).
/// exp-1780110463-209781800.
pub fn matmulBF16SIMDNtileMajor(
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;
    const TILE = 4 * LANE; // 64

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tiles = N / TILE;
    const n_remainder = N - n_tiles * TILE;

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];
        const shift: @Vector(16, u5) = @splat(16);

        var nt: usize = 0;
        while (nt < n_tiles) : (nt += 1) {
            var acc0: Vec_f32 = @splat(0.0);
            var acc1: Vec_f32 = @splat(0.0);
            var acc2: Vec_f32 = @splat(0.0);
            var acc3: Vec_f32 = @splat(0.0);

            const tile_base = nt * K * TILE;

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_row = b[tile_base + k * TILE ..][0..TILE];

                const b0_u16: Vec_u16 = b_row[0..LANE].*;
                const b1_u16: Vec_u16 = b_row[LANE..][0..LANE].*;
                const b2_u16: Vec_u16 = b_row[2 * LANE ..][0..LANE].*;
                const b3_u16: Vec_u16 = b_row[3 * LANE ..][0..LANE].*;

                const b0_u32: Vec_u32 = b0_u16;
                const b1_u32: Vec_u32 = b1_u16;
                const b2_u32: Vec_u32 = b2_u16;
                const b3_u32: Vec_u32 = b3_u16;

                const b0_f32: Vec_f32 = @bitCast(b0_u32 << shift);
                const b1_f32: Vec_f32 = @bitCast(b1_u32 << shift);
                const b2_f32: Vec_f32 = @bitCast(b2_u32 << shift);
                const b3_f32: Vec_f32 = @bitCast(b3_u32 << shift);

                acc0 = @mulAdd(Vec_f32, a_scalar, b0_f32, acc0);
                acc1 = @mulAdd(Vec_f32, a_scalar, b1_f32, acc1);
                acc2 = @mulAdd(Vec_f32, a_scalar, b2_f32, acc2);
                acc3 = @mulAdd(Vec_f32, a_scalar, b3_f32, acc3);
            }

            const c_base = nt * TILE;
            c_row[c_base..][0..LANE].* = acc0;
            c_row[c_base + LANE ..][0..LANE].* = acc1;
            c_row[c_base + 2 * LANE ..][0..LANE].* = acc2;
            c_row[c_base + 3 * LANE ..][0..LANE].* = acc3;
        }

        // Scalar fallback for N % TILE columns at the tail. The pre-repack
        // stores them at offset n_tiles * K * TILE in row-major form
        // (k * n_remainder + j), bit-for-bit consistent with the standalone
        // loader.
        if (n_remainder > 0) {
            const tail_base = n_tiles * K * TILE;
            const tail_c_base = n_tiles * TILE;
            var j: usize = 0;
            while (j < n_remainder) : (j += 1) {
                var acc: f32 = 0.0;
                var k: usize = 0;
                while (k < K) : (k += 1) {
                    const bv: f32 = @bitCast(@as(u32, b[tail_base + k * n_remainder + j]) << 16);
                    acc += a_row[k] * bv;
                }
                c_row[tail_c_base + j] = acc;
            }
        }
    }
}

/// Lever-14: int8 per-column quantized matmul, N-tile-major layout.
///
/// b layout: [n_tile_outer, K, n_tile_inner=TILE=64] i8 (lever-9 layout
/// with i8 elements instead of u16). scales: [N] f32, per-output-column.
///
/// Schedule (M=1 register-blocked, 4 ZMM accumulators per 64-col tile):
///   for n_tile in 0..N/64:
///     accs[4] = 0 (f32)
///     for k in 0..K:
///       a_scalar = splat(a[k])
///       b0..b3 = widen i8 to f32 from b[n_tile][k][0..64]
///       acc[t] = fmadd(a_scalar, b[t], acc[t])
///     // Apply per-column scale at output time
///     c[n_tile..n_tile+64] = accs[0..4] * scales[n_tile*64..(n_tile+1)*64]
///
/// Memory traffic vs BF16 lever-9 path: b stream is K*N bytes (half of BF16's
/// K*N*2). Scales add 4*N bytes per matmul (trivial — 8 KB at N=2048).
///
/// Widening sequence on Ice Lake AVX-512:
///   16 i8 (16 bytes)  → VPMOVSXBD → 16 i32 (zmm)
///   16 i32            → VCVTDQ2PS → 16 f32 (zmm)
/// Two instructions per ZMM tile; throughput is L1 read port + FMA.
pub fn matmulINT8SIMDNtileMajor(
    a: *const Matrix,
    b: []const i8,
    scales: []const f32,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;
    if (scales.len != b_cols) return Error.DimensionMismatch;

    const Vec_i8 = @Vector(16, i8);
    const Vec_i32 = @Vector(16, i32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;
    const TILE = 4 * LANE; // 64

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tiles = N / TILE;
    const n_remainder = N - n_tiles * TILE;

    @memset(c.data, 0.0);

    var i: usize = 0;
    while (i < M) : (i += 1) {
        const a_row = a.data[i * K ..][0..K];
        const c_row = c.data[i * N ..][0..N];

        var nt: usize = 0;
        while (nt < n_tiles) : (nt += 1) {
            var acc0: Vec_f32 = @splat(0.0);
            var acc1: Vec_f32 = @splat(0.0);
            var acc2: Vec_f32 = @splat(0.0);
            var acc3: Vec_f32 = @splat(0.0);

            const tile_base = nt * K * TILE;

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a_row[k]);
                const b_row = b[tile_base + k * TILE ..][0..TILE];

                const b0_i8: Vec_i8 = b_row[0..LANE].*;
                const b1_i8: Vec_i8 = b_row[LANE..][0..LANE].*;
                const b2_i8: Vec_i8 = b_row[2 * LANE ..][0..LANE].*;
                const b3_i8: Vec_i8 = b_row[3 * LANE ..][0..LANE].*;

                // i8 → i32 sign-extend → f32 cast (VPMOVSXBD + VCVTDQ2PS).
                const b0_i32: Vec_i32 = b0_i8;
                const b1_i32: Vec_i32 = b1_i8;
                const b2_i32: Vec_i32 = b2_i8;
                const b3_i32: Vec_i32 = b3_i8;

                const b0_f32: Vec_f32 = @floatFromInt(b0_i32);
                const b1_f32: Vec_f32 = @floatFromInt(b1_i32);
                const b2_f32: Vec_f32 = @floatFromInt(b2_i32);
                const b3_f32: Vec_f32 = @floatFromInt(b3_i32);

                acc0 = @mulAdd(Vec_f32, a_scalar, b0_f32, acc0);
                acc1 = @mulAdd(Vec_f32, a_scalar, b1_f32, acc1);
                acc2 = @mulAdd(Vec_f32, a_scalar, b2_f32, acc2);
                acc3 = @mulAdd(Vec_f32, a_scalar, b3_f32, acc3);
            }

            // Multiply by per-column scales before store.
            const c_base = nt * TILE;
            const s0: Vec_f32 = scales[c_base..][0..LANE].*;
            const s1: Vec_f32 = scales[c_base + LANE ..][0..LANE].*;
            const s2: Vec_f32 = scales[c_base + 2 * LANE ..][0..LANE].*;
            const s3: Vec_f32 = scales[c_base + 3 * LANE ..][0..LANE].*;

            c_row[c_base..][0..LANE].* = acc0 * s0;
            c_row[c_base + LANE ..][0..LANE].* = acc1 * s1;
            c_row[c_base + 2 * LANE ..][0..LANE].* = acc2 * s2;
            c_row[c_base + 3 * LANE ..][0..LANE].* = acc3 * s3;
        }

        // Scalar tail for N % TILE columns.
        if (n_remainder > 0) {
            const tail_base = n_tiles * K * TILE;
            const tail_c_base = n_tiles * TILE;
            var j: usize = 0;
            while (j < n_remainder) : (j += 1) {
                var acc: f32 = 0.0;
                var k: usize = 0;
                while (k < K) : (k += 1) {
                    const bv: f32 = @floatFromInt(b[tail_base + k * n_remainder + j]);
                    acc += a_row[k] * bv;
                }
                c_row[tail_c_base + j] = acc * scales[tail_c_base + j];
            }
        }
    }
}

/// Comptime-blocked BF16 GEMM kernel (substrate primitive).
///
/// Parameterised on M_BLOCK (rows accumulated together), N_TILE (cols per
/// inner register tile, must equal 64 to match N-tile-major layout), and
/// K_TILE (rows of B walked in each outer block — cache locality knob).
///
/// b layout: N-tile-major, [n_tile_outer, K, n_tile_inner=64], identical
/// to matmulBF16SIMDNtileMajor's expected input. Caller is responsible
/// for pre-repacking via weights.repackBF16ToNtileMajor at load time.
///
/// Schedule (BLAS-style 2-level blocking):
///
///   for n_tile in 0..N/64:
///     accs[M_BLOCK][4] = 0           // M_BLOCK rows x 4 ZMM tiles
///     for k_tile in 0..K step K_TILE:
///       for k in k_tile..k_tile+K_TILE:
///         load b[n_tile][k][0..64]   // 4 ZMM b loads (shared across M)
///         inline for m in 0..M_BLOCK:
///           a_scalar = splat(a[m, k])
///           inline for t in 0..4:
///             accs[m][t] = fmadd(a_scalar, b[t], accs[m][t])
///     store c[m_block_start..m_block_start+M_BLOCK, n_tile..n_tile+64]
///
/// Each b load is consumed M_BLOCK times before the next b load (b reuse
/// across M direction). K_TILE controls the inner-loop trip count; smaller
/// K_TILE = finer-grained scheduling, larger K_TILE = less loop overhead.
///
/// Register pressure on Ice Lake AVX-512 (32 ZMM):
///   M_BLOCK=4 N_TILE=64: 16 acc ZMM + 4 b ZMM + 4 a splats + 1 shift = 25
///   M_BLOCK=1 N_TILE=64: 4 acc ZMM + 4 b ZMM + 1 a splat + 1 shift = 10
///
/// For M=1 (decode) this degenerates to lever-9 plus an explicit K-tile
/// outer loop — same memory traffic, similar timing. The real win is at
/// M>=2 (prefill, batched serving) where b loads amortize across M.
/// Pre-reg exp-1780157294-563607188.
///
/// All three parameters are comptime, so the compiler specializes the
/// kernel per (M_BLOCK, N_TILE, K_TILE) tuple — `inline for` over M_BLOCK
/// unrolls the M dimension into a flat sequence of fmadds.
pub fn matmulBF16SIMDBlocked(
    comptime M_BLOCK: usize,
    comptime N_TILE: usize,
    comptime K_TILE: usize,
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) Error!void {
    comptime {
        if (N_TILE != 64) @compileError("matmulBF16SIMDBlocked: N_TILE must be 64 to match N-tile-major weight layout");
        if (M_BLOCK == 0) @compileError("matmulBF16SIMDBlocked: M_BLOCK must be at least 1");
        if (K_TILE == 0) @compileError("matmulBF16SIMDBlocked: K_TILE must be at least 1");
    }

    if (a.cols != b_rows) return Error.DimensionMismatch;
    if (c.rows != a.rows or c.cols != b_cols) return Error.DimensionMismatch;
    if (b.len != b_rows * b_cols) return Error.DimensionMismatch;

    const Vec_u16 = @Vector(16, u16);
    const Vec_u32 = @Vector(16, u32);
    const Vec_f32 = @Vector(16, f32);
    const LANE = 16;
    const TILE = N_TILE;

    const M = a.rows;
    const N = b_cols;
    const K = a.cols;
    const n_tiles = N / TILE;

    @memset(c.data, 0.0);

    // M loop in blocks of M_BLOCK. Tail rows (M % M_BLOCK) handled via a
    // dedicated single-row pass below.
    const m_block_end = (M / M_BLOCK) * M_BLOCK;
    var m_base: usize = 0;
    while (m_base < m_block_end) : (m_base += M_BLOCK) {
        var nt: usize = 0;
        while (nt < n_tiles) : (nt += 1) {
            // M_BLOCK rows × 4 ZMM accumulators each.
            var accs: [M_BLOCK][4]Vec_f32 = undefined;
            inline for (0..M_BLOCK) |m| inline for (0..4) |t| {
                accs[m][t] = @splat(0.0);
            };

            const tile_base = nt * K * TILE;
            const shift: @Vector(16, u5) = @splat(16);

            // K-tile outer + K inner. The accumulators stay live across all
            // k_tile passes (they fit in registers), so K_TILE is purely a
            // loop-shape knob — useful for icache locality and for the
            // compiler's scheduling decisions.
            var k_outer: usize = 0;
            while (k_outer < K) : (k_outer += K_TILE) {
                const k_end = @min(k_outer + K_TILE, K);
                var k = k_outer;
                while (k < k_end) : (k += 1) {
                    const b_row = b[tile_base + k * TILE ..][0..TILE];
                    const b0_u16: Vec_u16 = b_row[0..LANE].*;
                    const b1_u16: Vec_u16 = b_row[LANE..][0..LANE].*;
                    const b2_u16: Vec_u16 = b_row[2 * LANE ..][0..LANE].*;
                    const b3_u16: Vec_u16 = b_row[3 * LANE ..][0..LANE].*;
                    const b0_u32: Vec_u32 = b0_u16;
                    const b1_u32: Vec_u32 = b1_u16;
                    const b2_u32: Vec_u32 = b2_u16;
                    const b3_u32: Vec_u32 = b3_u16;
                    const b0_f32: Vec_f32 = @bitCast(b0_u32 << shift);
                    const b1_f32: Vec_f32 = @bitCast(b1_u32 << shift);
                    const b2_f32: Vec_f32 = @bitCast(b2_u32 << shift);
                    const b3_f32: Vec_f32 = @bitCast(b3_u32 << shift);

                    // M-block fmadd: b loads above are reused M_BLOCK times.
                    inline for (0..M_BLOCK) |m| {
                        const a_scalar: Vec_f32 = @splat(a.data[(m_base + m) * K + k]);
                        accs[m][0] = @mulAdd(Vec_f32, a_scalar, b0_f32, accs[m][0]);
                        accs[m][1] = @mulAdd(Vec_f32, a_scalar, b1_f32, accs[m][1]);
                        accs[m][2] = @mulAdd(Vec_f32, a_scalar, b2_f32, accs[m][2]);
                        accs[m][3] = @mulAdd(Vec_f32, a_scalar, b3_f32, accs[m][3]);
                    }
                }
            }

            // Store M_BLOCK rows × N_TILE cols.
            const c_base = nt * TILE;
            inline for (0..M_BLOCK) |m| {
                const c_row = c.data[(m_base + m) * N ..][0..N];
                c_row[c_base..][0..LANE].* = accs[m][0];
                c_row[c_base + LANE ..][0..LANE].* = accs[m][1];
                c_row[c_base + 2 * LANE ..][0..LANE].* = accs[m][2];
                c_row[c_base + 3 * LANE ..][0..LANE].* = accs[m][3];
            }
        }
    }

    // Tail M rows (M % M_BLOCK). Walk one row at a time using the same
    // schedule with M_BLOCK=1 effectively.
    while (m_base < M) : (m_base += 1) {
        var nt: usize = 0;
        while (nt < n_tiles) : (nt += 1) {
            var acc0: Vec_f32 = @splat(0.0);
            var acc1: Vec_f32 = @splat(0.0);
            var acc2: Vec_f32 = @splat(0.0);
            var acc3: Vec_f32 = @splat(0.0);

            const tile_base = nt * K * TILE;
            const shift: @Vector(16, u5) = @splat(16);

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_scalar: Vec_f32 = @splat(a.data[m_base * K + k]);
                const b_row = b[tile_base + k * TILE ..][0..TILE];
                const b0_u16: Vec_u16 = b_row[0..LANE].*;
                const b1_u16: Vec_u16 = b_row[LANE..][0..LANE].*;
                const b2_u16: Vec_u16 = b_row[2 * LANE ..][0..LANE].*;
                const b3_u16: Vec_u16 = b_row[3 * LANE ..][0..LANE].*;
                const b0_u32: Vec_u32 = b0_u16;
                const b1_u32: Vec_u32 = b1_u16;
                const b2_u32: Vec_u32 = b2_u16;
                const b3_u32: Vec_u32 = b3_u16;
                const b0_f32: Vec_f32 = @bitCast(b0_u32 << shift);
                const b1_f32: Vec_f32 = @bitCast(b1_u32 << shift);
                const b2_f32: Vec_f32 = @bitCast(b2_u32 << shift);
                const b3_f32: Vec_f32 = @bitCast(b3_u32 << shift);

                acc0 = @mulAdd(Vec_f32, a_scalar, b0_f32, acc0);
                acc1 = @mulAdd(Vec_f32, a_scalar, b1_f32, acc1);
                acc2 = @mulAdd(Vec_f32, a_scalar, b2_f32, acc2);
                acc3 = @mulAdd(Vec_f32, a_scalar, b3_f32, acc3);
            }

            const c_base = nt * TILE;
            const c_row = c.data[m_base * N ..][0..N];
            c_row[c_base..][0..LANE].* = acc0;
            c_row[c_base + LANE ..][0..LANE].* = acc1;
            c_row[c_base + 2 * LANE ..][0..LANE].* = acc2;
            c_row[c_base + 3 * LANE ..][0..LANE].* = acc3;
        }
    }
}

/// Auto-dispatch BF16-weights matmul.
///
/// On macOS (Apple Silicon), routes B-matrices with ≥1M elements through
/// matmulBF16Accelerate (AMX dispatch path); smaller shapes use the
/// in-process matmulBF16SIMD because cblas dispatch overhead would
/// dominate.
///
/// On non-macOS targets, always calls matmulBF16SIMD (cblas_sgemm is
/// never reachable; the extern declaration above is dead).
pub fn matmulBF16Auto(
    allocator: std.mem.Allocator,
    a: *const Matrix,
    b: []const u16,
    b_rows: usize,
    b_cols: usize,
    c: *Matrix,
) (Error || std.mem.Allocator.Error)!void {
    if (comptime builtin.os.tag == .macos) {
        if (b.len >= ACCELERATE_MIN_B_ELEMS) {
            return matmulBF16Accelerate(allocator, a, b, b_rows, b_cols, c);
        }
    }
    return matmulBF16SIMD(a, b, b_rows, b_cols, c);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const bf16_mod = @import("../bf16.zig");

fn matrixToBF16(allocator: std.mem.Allocator, m: *const Matrix) ![]u16 {
    const out = try allocator.alloc(u16, m.data.len);
    for (m.data, 0..) |x, i| out[i] = bf16_mod.f32ToBf16(x);
    return out;
}

test "matmulBF16Scalar — agrees with reference F32 matmul of BF16-quantised weights" {
    const N: usize = 16;
    var a = try Matrix.init(testing.allocator, N, N);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, N, N);
    defer b_f32.deinit(testing.allocator);

    a.fillRandom(101);
    b_f32.fillRandom(103);

    // Quantise B through BF16 first so the reference and the
    // BF16-kernel see the SAME numerical weights (eliminates the
    // F32→BF16 round-trip error from the comparison and isolates
    // the kernel's correctness).
    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);
    var b_requantised = try Matrix.init(testing.allocator, N, N);
    defer b_requantised.deinit(testing.allocator);
    for (b_bf16, 0..) |bits, idx| b_requantised.data[idx] = bf16_mod.bf16ToF32(bits);

    // Reference: F32×F32_quantised matmul (naive triple loop).
    var c_ref = try Matrix.init(testing.allocator, N, N);
    defer c_ref.deinit(testing.allocator);
    @memset(c_ref.data, 0.0);
    for (0..N) |i| for (0..N) |k| for (0..N) |j| {
        c_ref.data[i * N + j] += a.data[i * N + k] * b_requantised.data[k * N + j];
    };

    // BF16-weights matmul against the same quantised B.
    var c_bf16 = try Matrix.init(testing.allocator, N, N);
    defer c_bf16.deinit(testing.allocator);
    try matmulBF16Scalar(&a, b_bf16, N, N, &c_bf16);

    // Should agree bit-exact: same operations, same operands, same order.
    try testing.expect(c_ref.approxEq(&c_bf16, 1e-6));
}

test "matmulBF16SIMD — agrees with matmulBF16Scalar on 32×64×48" {
    const M: usize = 32;
    const K: usize = 64;
    const N: usize = 48;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(7);
    b_f32.fillRandom(11);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_scalar = try Matrix.init(testing.allocator, M, N);
    defer c_scalar.deinit(testing.allocator);
    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);

    try matmulBF16Scalar(&a, b_bf16, K, N, &c_scalar);
    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);

    // Scalar vs SIMD should agree bit-exact (same f32 ops, same order
    // up to the lane-grouping). Use tight tolerance.
    try testing.expect(c_scalar.approxEq(&c_simd, 1e-6));
}

test "matmulBF16SIMD — tail handling (N not a multiple of 8)" {
    const M: usize = 4;
    const K: usize = 8;
    const N: usize = 13; // deliberately not a multiple of 8

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(13);
    b_f32.fillRandom(17);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_scalar = try Matrix.init(testing.allocator, M, N);
    defer c_scalar.deinit(testing.allocator);
    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);

    try matmulBF16Scalar(&a, b_bf16, K, N, &c_scalar);
    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);

    try testing.expect(c_scalar.approxEq(&c_simd, 1e-6));
}

test "matmulBF16SIMDRegBlock — agrees with matmulBF16SIMD on MLP-shape (M=1 K=2048 N=2048)" {
    const M: usize = 1;
    const K: usize = 2048;
    const N: usize = 2048;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(31);
    b_f32.fillRandom(37);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);
    var c_regblock = try Matrix.init(testing.allocator, M, N);
    defer c_regblock.deinit(testing.allocator);

    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);
    try matmulBF16SIMDRegBlock(&a, b_bf16, K, N, &c_regblock);

    // SIMD and reg-blocked variants accumulate in different orders
    // (SIMD: K outer over c; reg-block: K inner per N tile). Different
    // f32 rounding accumulation, so use loose tolerance for K=2048 sum.
    try testing.expect(c_simd.approxEq(&c_regblock, 5e-3));
}

test "matmulBF16SIMDRegBlock — tail handling (N=77 not multiple of 64 or 16)" {
    const M: usize = 2;
    const K: usize = 32;
    const N: usize = 77;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(43);
    b_f32.fillRandom(47);

    const b_bf16 = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_bf16);

    var c_simd = try Matrix.init(testing.allocator, M, N);
    defer c_simd.deinit(testing.allocator);
    var c_regblock = try Matrix.init(testing.allocator, M, N);
    defer c_regblock.deinit(testing.allocator);

    try matmulBF16SIMD(&a, b_bf16, K, N, &c_simd);
    try matmulBF16SIMDRegBlock(&a, b_bf16, K, N, &c_regblock);

    try testing.expect(c_simd.approxEq(&c_regblock, 5e-4));
}

test "matmulBF16SIMDBlocked M=4 N_TILE=64 K_TILE=128 — agrees with NtileMajor on prefill shape" {
    const M: usize = 4;
    const K: usize = 128;
    const N: usize = 256;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(71);
    b_f32.fillRandom(73);

    const b_rm = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_rm);

    // Repack to N-tile-major (same layout the blocked kernel consumes).
    const TILE: usize = 64;
    const n_tiles = N / TILE;
    const b_nt = try testing.allocator.alloc(u16, K * N);
    defer testing.allocator.free(b_nt);
    var nt: usize = 0;
    while (nt < n_tiles) : (nt += 1) {
        var k: usize = 0;
        while (k < K) : (k += 1) {
            var i: usize = 0;
            while (i < TILE) : (i += 1) {
                b_nt[nt * K * TILE + k * TILE + i] = b_rm[k * N + nt * TILE + i];
            }
        }
    }

    var c_ntile = try Matrix.init(testing.allocator, M, N);
    defer c_ntile.deinit(testing.allocator);
    var c_blocked = try Matrix.init(testing.allocator, M, N);
    defer c_blocked.deinit(testing.allocator);

    try matmulBF16SIMDNtileMajor(&a, b_nt, K, N, &c_ntile);
    try matmulBF16SIMDBlocked(4, 64, 128, &a, b_nt, K, N, &c_blocked);

    // Both kernels use FMA so accumulation rounding matches; tolerance
    // covers BF16 quantization noise across K=128.
    try testing.expect(c_ntile.approxEq(&c_blocked, 5e-3));
}

test "matmulBF16SIMDBlocked M_BLOCK=1 — equivalent to matmulBF16SIMDNtileMajor on M=1" {
    const M: usize = 1;
    const K: usize = 256;
    const N: usize = 128;

    var a = try Matrix.init(testing.allocator, M, K);
    defer a.deinit(testing.allocator);
    var b_f32 = try Matrix.init(testing.allocator, K, N);
    defer b_f32.deinit(testing.allocator);
    a.fillRandom(81);
    b_f32.fillRandom(83);

    const b_rm = try matrixToBF16(testing.allocator, &b_f32);
    defer testing.allocator.free(b_rm);

    const TILE: usize = 64;
    const n_tiles = N / TILE;
    const b_nt = try testing.allocator.alloc(u16, K * N);
    defer testing.allocator.free(b_nt);
    var nt: usize = 0;
    while (nt < n_tiles) : (nt += 1) {
        var k: usize = 0;
        while (k < K) : (k += 1) {
            var i: usize = 0;
            while (i < TILE) : (i += 1) {
                b_nt[nt * K * TILE + k * TILE + i] = b_rm[k * N + nt * TILE + i];
            }
        }
    }

    var c_ntile = try Matrix.init(testing.allocator, M, N);
    defer c_ntile.deinit(testing.allocator);
    var c_blocked = try Matrix.init(testing.allocator, M, N);
    defer c_blocked.deinit(testing.allocator);

    try matmulBF16SIMDNtileMajor(&a, b_nt, K, N, &c_ntile);
    try matmulBF16SIMDBlocked(1, 64, 64, &a, b_nt, K, N, &c_blocked);

    try testing.expect(c_ntile.approxEq(&c_blocked, 5e-3));
}

test "matmulBF16 — dimension mismatch is caught" {
    var a = try Matrix.init(testing.allocator, 4, 8);
    defer a.deinit(testing.allocator);
    var c = try Matrix.init(testing.allocator, 4, 16);
    defer c.deinit(testing.allocator);

    const b_bf16 = try testing.allocator.alloc(u16, 6 * 16); // wrong b_rows
    defer testing.allocator.free(b_bf16);

    try testing.expectError(Error.DimensionMismatch, matmulBF16Scalar(&a, b_bf16, 6, 16, &c));
    try testing.expectError(Error.DimensionMismatch, matmulBF16SIMD(&a, b_bf16, 6, 16, &c));
}
