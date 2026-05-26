const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. THE DNA: Categorical Wire Protocol
    const ribosome = b.createModule(.{ .root_source_file = b.path("../crpc-core/src/crpc.zig") });

    // 2. THE CALCULUS: Symbolic PTX/SASS Ark
    const ark = b.createModule(.{ .root_source_file = b.path("../Work/lab/zig-canonical/src/gpu/ptx_ark.zig") });

    // 3. THE INGEST: Tokenizers & SafeTensors & FAISS
    const tokenizer = b.createModule(.{ .root_source_file = b.path("../tokenizers-zig/src/root.zig") });
    const safetensors = b.createModule(.{ .root_source_file = b.path("../safetensors-zig/src/root.zig") });
    const faiss = b.createModule(.{ .root_source_file = b.path("../faiss-zig/src/root.zig") });

    // 4. THE FINANCE: Rippled & Orderbook & TigerBeetle & Water
    const rippled = b.createModule(.{ .root_source_file = b.path("../rippled-zig/src/ledger.zig") });
    const orderbook = b.createModule(.{ .root_source_file = b.path("../orderbook/zig/src/hft_core.zig") });
    const tigerbeetle = b.createModule(.{ .root_source_file = b.path("../tigerbeetle/src/tigerbeetle.zig") });
    const water = b.createModule(.{ .root_source_file = b.path("../water-ledger/src/root.zig") });

    // 5. THE PHYSICS: Qualitative Process Engine (QPE)
    const qpe = b.createModule(.{ .root_source_file = b.path("../qpe-zig/src/qpe.zig") });

    // 6. THE BUILD: Sentinel SBOM
    const sentinel = b.createModule(.{ .root_source_file = b.path("../sentinel-sbom/src/lib.zig") });

    // 7. THE SOLVE: ARC-AGI & Market OS & Zeth
    const h3 = b.createModule(.{ .root_source_file = b.path("../zig-h3/src/pure_proj.zig") });
    const smos = b.createModule(.{ .root_source_file = b.path("../smos-zig/src/root.zig") });
    const arc = b.createModule(.{ .root_source_file = b.path("../arc-agi-3-attack/src/root.zig") });
    const zeth = b.createModule(.{ .root_source_file = b.path("../zeth/src/main.zig") });

    // THE UNITARY SOVEREIGN ENGINE
    const engine = b.addExecutable(.{
        .name = "sovereign_engine",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    engine.root_module.addImport("ribosome", ribosome);
    engine.root_module.addImport("ark", ark);
    engine.root_module.addImport("tokenizer", tokenizer);
    engine.root_module.addImport("safetensors", safetensors);
    engine.root_module.addImport("faiss", faiss);
    engine.root_module.addImport("rippled", rippled);
    engine.root_module.addImport("orderbook", orderbook);
    engine.root_module.addImport("tigerbeetle", tigerbeetle);
    engine.root_module.addImport("water", water);
    engine.root_module.addImport("qpe", qpe);
    engine.root_module.addImport("sentinel", sentinel);
    engine.root_module.addImport("h3", h3);
    engine.root_module.addImport("smos", smos);
    engine.root_module.addImport("arc", arc);
    engine.root_module.addImport("zeth", zeth);
    
    b.installArtifact(engine);
}
