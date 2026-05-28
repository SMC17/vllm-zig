const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Two flavors of the safetensors / tokenizers / vllm modules:
    //   * `mod` — built at the user-selected -Doptimize (default Debug).
    //     Used by `zig build test` so failing assertions get good traces.
    //   * `mod_fast` — ReleaseFast. Used by every bench/example step so
    //     the recorded numbers reflect optimized codegen, not Debug
    //     overhead (which the historical 581 ms/tok baseline did NOT
    //     have factored in, and which inflated decode times by ~3x on
    //     this hardware).
    const safetensors_dep = b.dependency("safetensors", .{
        .target = target,
        .optimize = optimize,
    });
    const tokenizers_dep = b.dependency("tokenizers", .{
        .target = target,
        .optimize = optimize,
    });
    const safetensors_fast = b.dependency("safetensors", .{
        .target = target,
        .optimize = .ReleaseFast,
    });
    const tokenizers_fast = b.dependency("tokenizers", .{
        .target = target,
        .optimize = .ReleaseFast,
    });

    const mod = b.addModule("vllm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("safetensors", safetensors_dep.module("safetensors"));
    mod.addImport("tokenizers", tokenizers_dep.module("tokenizers"));

    const mod_fast = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    mod_fast.addImport("safetensors", safetensors_fast.module("safetensors"));
    mod_fast.addImport("tokenizers", tokenizers_fast.module("tokenizers"));

    // ------------------------------------------------------------------
    // test (Debug)
    // ------------------------------------------------------------------
    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests (re-exported from src/root.zig)");
    test_step.dependOn(&run_tests.step);

    // ------------------------------------------------------------------
    // bench-matmul — scalar vs SIMD f32 matmul, several shapes
    // ------------------------------------------------------------------
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_matmul.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("vllm", mod_fast);
    const bench_exe = b.addExecutable(.{
        .name = "bench-matmul",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.has_side_effects = true;
    const bench_step = b.step("bench-matmul", "Matmul kernel bench");
    bench_step.dependOn(&run_bench.step);

    // ------------------------------------------------------------------
    // bench-pool — persistent pool vs ad-hoc spawn
    // ------------------------------------------------------------------
    const bench_pool_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_pool.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_pool_mod.addImport("vllm", mod_fast);
    const bench_pool_exe = b.addExecutable(.{
        .name = "bench-pool",
        .root_module = bench_pool_mod,
    });
    const run_bench_pool = b.addRunArtifact(bench_pool_exe);
    run_bench_pool.has_side_effects = true;
    const bench_pool_step = b.step("bench-pool", "Persistent pool vs ad-hoc spawn bench");
    bench_pool_step.dependOn(&run_bench_pool.step);

    // ------------------------------------------------------------------
    // bench-tiled — tiled vs non-tiled matmul scan
    // ------------------------------------------------------------------
    const bench_tiled_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_tiled.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tiled_mod.addImport("vllm", mod_fast);
    const bench_tiled_exe = b.addExecutable(.{
        .name = "bench-tiled",
        .root_module = bench_tiled_mod,
    });
    const run_bench_tiled = b.addRunArtifact(bench_tiled_exe);
    run_bench_tiled.has_side_effects = true;
    const bench_tiled_step = b.step("bench-tiled", "Cache-tiled vs non-tiled matmul bench");
    bench_tiled_step.dependOn(&run_bench_tiled.step);

    // ------------------------------------------------------------------
    // forward-skeleton — transformer-block smoke test
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // infer-tinyllama — end-to-end inference smoke test
    // ------------------------------------------------------------------
    const infer_mod = b.createModule(.{
        .root_source_file = b.path("examples/infer_tinyllama.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    infer_mod.addImport("vllm", mod_fast);
    infer_mod.addImport("safetensors", safetensors_fast.module("safetensors"));
    infer_mod.addImport("tokenizers", tokenizers_fast.module("tokenizers"));
    const infer_exe = b.addExecutable(.{
        .name = "infer-tinyllama",
        .root_module = infer_mod,
    });
    b.installArtifact(infer_exe);
    const run_infer = b.addRunArtifact(infer_exe);
    if (b.args) |args| run_infer.addArgs(args);
    const infer_step = b.step("infer-tinyllama", "End-to-end TinyLlama inference smoke test");
    infer_step.dependOn(&run_infer.step);

    // ------------------------------------------------------------------
    // bench-tinyllama-pool — end-to-end pool-vs-no-pool head-to-head
    // ------------------------------------------------------------------
    const bench_tll_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_tinyllama_pool.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tll_mod.addImport("vllm", mod_fast);
    bench_tll_mod.addImport("safetensors", safetensors_fast.module("safetensors"));
    bench_tll_mod.addImport("tokenizers", tokenizers_fast.module("tokenizers"));
    const bench_tll_exe = b.addExecutable(.{
        .name = "bench-tinyllama-pool",
        .root_module = bench_tll_mod,
    });
    const run_bench_tll = b.addRunArtifact(bench_tll_exe);
    if (b.args) |args| run_bench_tll.addArgs(args);
    run_bench_tll.has_side_effects = true;
    const bench_tll_step = b.step("bench-tinyllama-pool", "TinyLlama end-to-end pool vs no-pool head-to-head");
    bench_tll_step.dependOn(&run_bench_tll.step);
}
