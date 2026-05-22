const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const safetensors_dep = b.dependency("safetensors", .{
        .target = target,
        .optimize = optimize,
    });
    const tokenizers_dep = b.dependency("tokenizers", .{
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("vllm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("safetensors", safetensors_dep.module("safetensors"));
    mod.addImport("tokenizers", tokenizers_dep.module("tokenizers"));

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Matmul bench
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_matmul.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("vllm", mod);
    const bench_exe = b.addExecutable(.{
        .name = "bench-matmul",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.has_side_effects = true;
    const bench_step = b.step("bench-matmul", "Matmul kernel bench");
    bench_step.dependOn(&run_bench.step);

    // Forward-pass skeleton example
    const ex_mod = b.createModule(.{
        .root_source_file = b.path("examples/forward_skeleton.zig"),
        .target = target,
        .optimize = optimize,
    });
    ex_mod.addImport("vllm", mod);
    const ex_exe = b.addExecutable(.{
        .name = "forward-skeleton",
        .root_module = ex_mod,
    });
    b.installArtifact(ex_exe);
    const run_ex = b.addRunArtifact(ex_exe);
    const ex_step = b.step("forward-skeleton", "Run the transformer-block skeleton smoke test");
    ex_step.dependOn(&run_ex.step);

    // End-to-end TinyLlama inference smoke test
    const infer_mod = b.createModule(.{
        .root_source_file = b.path("examples/infer_tinyllama.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    infer_mod.addImport("vllm", mod);
    infer_mod.addImport("safetensors", safetensors_dep.module("safetensors"));
    infer_mod.addImport("tokenizers", tokenizers_dep.module("tokenizers"));
    const infer_exe = b.addExecutable(.{
        .name = "infer-tinyllama",
        .root_module = infer_mod,
    });
    b.installArtifact(infer_exe);
    const run_infer = b.addRunArtifact(infer_exe);
    if (b.args) |args| run_infer.addArgs(args);
    const infer_step = b.step("infer-tinyllama", "End-to-end TinyLlama inference smoke test");
    infer_step.dependOn(&run_infer.step);
}
