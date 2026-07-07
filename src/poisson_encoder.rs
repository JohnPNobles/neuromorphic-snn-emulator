use rand::RngExt; // Crucial trait for rand 0.10+ methods

pub struct PoissonEncoder;

impl PoissonEncoder {
    pub fn new() -> Self {
        PoissonEncoder
    }

    /// Translates continuous normalized feature metrics [0.0, 1.0] to a binary spike vector
    pub fn encode_features_to_spikes(&self, features: &[f32]) -> Vec<i32> {
        let mut rng = rand::rng();

        features
            .iter()
            .map(|&value| if rng.random_bool(value as f64) { 1 } else { 0 })
            .collect()
    }
}
