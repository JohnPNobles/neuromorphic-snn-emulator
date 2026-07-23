// src/receptive_field.rs
//
// Host-side (off-chip) population coding. Expands each of the 13 raw
// features into 5 overlapping Gaussian "receptive fields", turning one
// scalar value per feature into a small population of tuning-curve
// responses -- much like biological sensory neurons each respond most
// strongly to a preferred stimulus value. This is a standard technique for
// converting continuous tabular data into a richer, more separable spike
// pattern before Poisson encoding. Uses f32/exp() as a data-preparation
// step only; the emulated chip itself never touches floating point.

/// Holds the fixed Gaussian tuning-curve parameters (centers and shared
/// sigma) used to expand each raw feature into multiple receptive fields.
pub struct ReceptiveFieldEncoder {
    pub num_features: usize,
    pub fields_per_feature: usize,
    centers: Vec<f32>, // Preferred (peak-response) value for each field
    sigma: f32,        // Shared width of each Gaussian tuning curve
}

impl ReceptiveFieldEncoder {
    /// Builds `fields_per_feature` evenly-spaced Gaussian tuning curves
    /// covering the [0.0, 1.0] normalized feature range, with sigma scaled
    /// to give neighboring fields sensible overlap.
    pub fn new(num_features: usize, fields_per_feature: usize) -> Self {
        let mut centers = Vec::with_capacity(fields_per_feature);
        for i in 0..fields_per_feature {
            let center = (i as f32 + 0.5) / (fields_per_feature as f32);
            centers.push(center);
        }

        // Dynamically scale sigma based on field density for optimal overlap
        // between neighboring receptive fields.
        let sigma = 1.0 / (fields_per_feature as f32 * 1.5);

        Self {
            num_features,
            fields_per_feature,
            centers,
            sigma,
        }
    }

    /// Expands a raw 13-value feature vector into a 65-value (13 x 5)
    /// density vector, where each output entry is how strongly one
    /// receptive field responds to one raw feature value. These densities
    /// are what the Poisson encoder later converts into spikes.
    pub fn encode_to_densities(&self, features: &[f32]) -> Vec<f32> {
        let mut encoded = Vec::with_capacity(self.num_features * self.fields_per_feature);

        // For every raw feature, compute its Gaussian response under each
        // of this feature's receptive fields.
        for &val in features {
            for &center in &self.centers {
                let diff = val - center;
                let response = (-(diff * diff) / (2.0 * self.sigma * self.sigma)).exp();
                encoded.push(response.clamp(0.0, 1.0));
            }
        }

        encoded
    }
}
