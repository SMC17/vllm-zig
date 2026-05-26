const std = @import("std");

/// Project Omega-Scalar: Sparse Autoencoder (SAE) Hook
///
/// Projects high-dimensional residual stream activations onto a 
/// Sparse Feature Dictionary. This is the foundation for actual 
/// mechanistic interpretability (monosemantic feature finding).
pub const SAEHook = struct {
    pub const HIDDEN_DIM = 2048; // Residual stream width
    pub const DICT_DIM = HIDDEN_DIM * 32; // Overcomplete dictionary expansion

    /// The SAE Dictionary (Encoder)
    pub const Dictionary = struct {
        encoder_weights: []f32,
        encoder_bias: []f32,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Dictionary {
            const weights = try allocator.alloc(f32, HIDDEN_DIM * DICT_DIM);
            const bias = try allocator.alloc(f32, DICT_DIM);
            
            // Random initialization for scaffold
            var prng = std.Random.DefaultPrng.init(0xDEAF);
            for (weights) |*w| w.* = prng.random().float(f32) * 0.01;
            for (bias) |*b| b.* = 0.0;
            
            return Dictionary{
                .encoder_weights = weights,
                .encoder_bias = bias,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Dictionary) void {
            self.allocator.free(self.encoder_weights);
            self.allocator.free(self.encoder_bias);
        }

        /// Project hidden state to sparse features: TopK or ReLU activation.
        /// Features = ReLU( (X - pre_bias) * W_enc + b_enc )
        pub fn encode(self: Dictionary, hidden: []const f32, features: []f32) void {
            std.debug.assert(hidden.len == HIDDEN_DIM);
            std.debug.assert(features.len == DICT_DIM);

            for (0..DICT_DIM) |j| {
                var sum: f32 = self.encoder_bias[j];
                for (0..HIDDEN_DIM) |i| {
                    sum += hidden[i] * self.encoder_weights[j * HIDDEN_DIM + i];
                }
                // ReLU activation for sparsity
                features[j] = @max(0.0, sum);
            }
        }
    };
};

test "SAE Hook Encoding" {
    var dict = try SAEHook.Dictionary.init(std.testing.allocator);
    defer dict.deinit();

    const hidden = [_]f32{1.0} ** SAEHook.HIDDEN_DIM;
    const features = try std.testing.allocator.alloc(f32, SAEHook.DICT_DIM);
    defer std.testing.allocator.free(features);

    dict.encode(&hidden, features);

    // Verify non-zero activation
    var active_count: usize = 0;
    for (features) |f| {
        if (f > 0.0) active_count += 1;
    }
    try std.testing.expect(active_count > 0);
}
