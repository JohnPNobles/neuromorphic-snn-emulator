// src/poisson_encoder.rs
//
// Host-side (off-chip) spike encoding. This is the "DAC" of the emulated
// chip: it converts continuous-valued receptive-field densities into binary
// spike events the integer-only crossbar can consume. Uses f64/f32 for the
// random draw itself, but its *output* is strictly 0/1 -- everything
// downstream of this module operates on integers only.

use rand::RngExt; // Crucial trait for rand 0.10+ methods

/// Namespace struct for the Poisson (Bernoulli-per-timestep) spike encoder.
/// Holds no state -- randomness is drawn fresh from the thread-local
/// generator on each call.
pub struct PoissonEncoder;

impl PoissonEncoder {
    pub fn new() -> Self {
        PoissonEncoder
    }

    /// Translates a vector of continuous densities in [0.0, 1.0] into a
    /// binary spike vector for a single timestep. Each entry independently
    /// "fires" (1) with probability equal to its density value, and stays
    /// silent (0) otherwise -- this is a per-timestep Bernoulli trial, which
    /// approximates a Poisson process over many timesteps and gives higher
    /// input values a proportionally higher firing rate.
    pub fn encode_features_to_spikes(&self, features: &[f32]) -> Vec<i32> {
        let mut rng = rand::rng();

        features
            .iter()
            .map(|&value| if rng.random_bool(value as f64) { 1 } else { 0 })
            .collect()
    }
}
