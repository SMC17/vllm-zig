//! vllm-zig — Zig kernels + (future) BEAM/OTP scheduler for
//! production LLM serving.
//!
//! Portfolio artifact #1 (headline) of the 8-artifact jaw-drop
//! portfolio. See ARCHITECTURE.md for the full design.
//!
//! v0.0.1 scope: Phase 1 prelude — CPU-only scalar/SIMD kernel
//! substrate. The kernels here (matmul, rmsnorm, softmax, silu) are
//! the foundation that the BEAM scheduler will eventually dispatch to
//! Triton/CUDA implementations on GPU. The CPU baseline ships first
//! so the API + kernel-launcher pattern is locked before hardware
//! work begins.
//!
//! What's here in v0.0.1:
//!   - tensor.zig — Matrix abstraction (f32, row-major, owned)
//!   - kernels/matmul.zig — scalar + SIMD matmul kernels, tested
//!     against each other on multiple shape variants
//!   - kernels/rmsnorm.zig — Llama-style RMSNorm
//!   - kernels/softmax.zig — numerically stable softmax
//!   - kernels/silu.zig — SiLU activation + Llama MLP-style siluMul
//!
//! What's NOT here yet:
//!   - The actual forward pass (Phase 1 proper)
//!   - PagedAttention KV-cache (Phase 1.5)
//!   - BEAM scheduler (Phase 2)
//!   - HTTP front-end (Phase 2)
//!   - CUDA / Triton kernels (Phase 1.5+, hardware-gated on Ampere+)

const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const Matrix = tensor.Matrix;

pub const matmul = @import("kernels/matmul.zig");
pub const rmsnorm = @import("kernels/rmsnorm.zig");
pub const softmax = @import("kernels/softmax.zig");
pub const silu = @import("kernels/silu.zig");

// ============================================================================
// Re-export tests — pulls tests from every submodule by referencing them.
// ============================================================================

comptime {
    _ = tensor;
    _ = matmul;
    _ = rmsnorm;
    _ = softmax;
    _ = silu;
}
