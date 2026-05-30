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
    // bench-bf16-ntile-major — lever-9 layout vs lever-5 microbench
    // ------------------------------------------------------------------
    const bench_ntile_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_bf16_ntile_major.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ntile_mod.addImport("vllm", mod_fast);
    const bench_ntile_exe = b.addExecutable(.{
        .name = "bench-bf16-ntile-major",
        .root_module = bench_ntile_mod,
    });
    const run_bench_ntile = b.addRunArtifact(bench_ntile_exe);
    run_bench_ntile.has_side_effects = true;
    const bench_ntile_step = b.step("bench-bf16-ntile-major", "Lever-9 N-tile-major BF16 layout microbench");
    bench_ntile_step.dependOn(&run_bench_ntile.step);

    // ------------------------------------------------------------------
    // bench-bf16-ntile-shapes — lever-9 win across all TinyLlama shapes
    // ------------------------------------------------------------------
    const bench_ntile_shapes_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_bf16_ntile_shapes.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ntile_shapes_mod.addImport("vllm", mod_fast);
    const bench_ntile_shapes_exe = b.addExecutable(.{
        .name = "bench-bf16-ntile-shapes",
        .root_module = bench_ntile_shapes_mod,
    });
    const run_bench_ntile_shapes = b.addRunArtifact(bench_ntile_shapes_exe);
    run_bench_ntile_shapes.has_side_effects = true;
    const bench_ntile_shapes_step = b.step("bench-bf16-ntile-shapes", "Lever-9 win across all five TinyLlama BF16 matmul shapes");
    bench_ntile_shapes_step.dependOn(&run_bench_ntile_shapes.step);

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
    // bench-lmhead-bf16-vs-f32 — isolates the BF16 lm_head bandwidth claim
    // (BENCH.md section 3.2 / 4.2). Single-thread, no pool, no tiling.
    // Compares matmul.matmulSIMD (F32 weights) against
    // matmul_bf16.matmulBF16SIMD (BF16 weights) at three TinyLlama-class
    // shapes with M=1 (single-token decode). Reports median + p10/p90
    // over 30 per-call samples.
    // ------------------------------------------------------------------
    const bench_bf16_vs_f32_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_lmhead_bf16_vs_f32.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_bf16_vs_f32_mod.addImport("vllm", mod_fast);
    const bench_bf16_vs_f32_exe = b.addExecutable(.{
        .name = "bench-lmhead-bf16-vs-f32",
        .root_module = bench_bf16_vs_f32_mod,
    });
    const run_bench_bf16_vs_f32 = b.addRunArtifact(bench_bf16_vs_f32_exe);
    run_bench_bf16_vs_f32.has_side_effects = true;
    const bench_bf16_vs_f32_step = b.step("bench-lmhead-bf16-vs-f32", "BF16 vs F32 lm_head matmul microbench");
    bench_bf16_vs_f32_step.dependOn(&run_bench_bf16_vs_f32.step);

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
    // Link Accelerate on macOS for the cblas_sgemm lm_head fast path
    // in Model.forward. Harmless on other targets — the framework
    // step is only added when the host target is macOS.
    if (target.result.os.tag == .macos) {
        bench_tll_mod.linkFramework("Accelerate", .{});
        bench_tll_mod.link_libc = true;
    }
    const run_bench_tll = b.addRunArtifact(bench_tll_exe);
    if (b.args) |args| run_bench_tll.addArgs(args);
    run_bench_tll.has_side_effects = true;
    const bench_tll_step = b.step("bench-tinyllama-pool", "TinyLlama end-to-end pool vs no-pool head-to-head");
    bench_tll_step.dependOn(&run_bench_tll.step);

    // ------------------------------------------------------------------
    // bench-tinyllama-breakdown — per-region decode time via Instrumentation
    // ------------------------------------------------------------------
    const bench_bd_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_tinyllama_breakdown.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_bd_mod.addImport("vllm", mod_fast);
    bench_bd_mod.addImport("safetensors", safetensors_fast.module("safetensors"));
    bench_bd_mod.addImport("tokenizers", tokenizers_fast.module("tokenizers"));
    const bench_bd_exe = b.addExecutable(.{
        .name = "bench-tinyllama-breakdown",
        .root_module = bench_bd_mod,
    });
    const run_bench_bd = b.addRunArtifact(bench_bd_exe);
    run_bench_bd.has_side_effects = true;
    const bench_bd_step = b.step("bench-tinyllama-breakdown", "Per-region TinyLlama decode breakdown via Instrumentation");
    bench_bd_step.dependOn(&run_bench_bd.step);

    // ------------------------------------------------------------------
    // bench-forward-regions -- multi-round per-region timing (med + p10/p90)
    // Env-gated: set VLLM_INSTRUMENT=1 to enable instrumentation, and
    // VLLM_POOLED_LM_HEAD=1 to route the lm_head matmul through
    // Pool.matmulBF16Auto (v0.0.7 lever #1). Same model, same prompt,
    // 5 rounds of prefill + 16 decode tokens; reports median + p10/p90
    // per region under load-violated conditions.
    // ------------------------------------------------------------------
    const bench_regions_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_forward_regions.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_regions_mod.addImport("vllm", mod_fast);
    bench_regions_mod.addImport("safetensors", safetensors_fast.module("safetensors"));
    bench_regions_mod.addImport("tokenizers", tokenizers_fast.module("tokenizers"));
    const bench_regions_exe = b.addExecutable(.{
        .name = "bench-forward-regions",
        .root_module = bench_regions_mod,
    });
    const run_bench_regions = b.addRunArtifact(bench_regions_exe);
    if (b.args) |args| run_bench_regions.addArgs(args);
    run_bench_regions.has_side_effects = true;
    const bench_regions_step = b.step("bench-forward-regions", "Per-region Model.forward timing (set VLLM_INSTRUMENT=1)");
    bench_regions_step.dependOn(&run_bench_regions.step);

    // ------------------------------------------------------------------
    // bench-mlp-cblas — MLP-shape F32 SIMD vs cblas_sgemm head-to-head
    // ------------------------------------------------------------------
    const bench_mc_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_mlp_cblas.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mc_mod.addImport("vllm", mod_fast);
    if (target.result.os.tag == .macos) {
        bench_mc_mod.linkFramework("Accelerate", .{});
        bench_mc_mod.link_libc = true;
    } else if (target.result.os.tag == .linux) {
        bench_mc_mod.link_libc = true;
        bench_mc_mod.addLibraryPath(.{ .cwd_relative = "/nix/store/z11jjs7wr129hvnrfdgmf0qar7zgfd9i-openblas-0.3.32/lib" });
        bench_mc_mod.linkSystemLibrary("openblas", .{});
    }
    const bench_mc_exe = b.addExecutable(.{
        .name = "bench-mlp-cblas",
        .root_module = bench_mc_mod,
    });
    const run_bench_mc = b.addRunArtifact(bench_mc_exe);
    run_bench_mc.has_side_effects = true;
    const bench_mc_step = b.step("bench-mlp-cblas", "MLP-shape vllm matmulSIMD vs cblas_sgemm");
    bench_mc_step.dependOn(&run_bench_mc.step);
}
