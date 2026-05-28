//! GPU context — placeholder for the future PTX/SASS JIT path.
//!
//! v0.0.6 ships CPU-only kernels; this module is kept as a hook so the
//! Model.matmulDispatch() path can early-return through a non-null
//! `gpu_ctx`. The earlier wiring depended on an external `ark_bridge`
//! module that lives outside this repo and is gated behind a much
//! larger toolchain decision; until that lands, GPUContext.init() is
//! a stub that returns a no-op handle.
//!
//! When the PTX/SASS lane is ready, the methods below can be filled in
//! and the build.zig flipped to add the bridge import.

const std = @import("std");

pub const GPUContext = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*GPUContext {
        const self = try allocator.create(GPUContext);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *GPUContext) void {
        self.allocator.destroy(self);
    }

    /// Stub: returns an empty PTX blob. The real implementation will
    /// JIT-compile a shape-specialised matmul kernel.
    pub fn compileMatmul(self: *GPUContext, m: usize, k: usize, n: usize) ![]const u8 {
        _ = self;
        _ = m;
        _ = k;
        _ = n;
        return error.GpuNotConfigured;
    }

    /// Stub: would cuModuleLoadData + cuLaunchKernel.
    pub fn launch(self: *GPUContext, kernel_ptx: []const u8) !void {
        _ = self;
        _ = kernel_ptx;
        return error.GpuNotConfigured;
    }
};
