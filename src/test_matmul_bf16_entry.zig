// Isolated test entry for the BF16 matmul kernel.
// Bypasses kv_cache test signature drift in v0.0.5 engine refactor.

const std = @import("std");
const tensor = @import("tensor.zig");
const bf16 = @import("bf16.zig");
const matmul_bf16 = @import("kernels/matmul_bf16.zig");

comptime {
    _ = tensor;
    _ = bf16;
    _ = matmul_bf16;
}
