pub struct ReceptiveFieldEncoder {
    pub num_features: usize,
    pub fields_per_feature: usize,
    centers: Vec<f32>,
    sigma: f32,
}

impl ReceptiveFieldEncoder {
    pub fn new(num_features: usize, fields_per_feature: usize) -> Self {
        let mut centers = Vec::with_capacity(fields_per_feature);
        for i in 0..fields_per_feature {
            let center = (i as f32 + 0.5) / (fields_per_feature as f32);
            centers.push(center);
        }

        // Dynamically scale sigma based on field density for optimal overlap
        let sigma = 1.0 / (fields_per_feature as f32 * 1.5);

        Self {
            num_features,
            fields_per_feature,
            centers,
            sigma,
        }
    }

    pub fn encode_to_densities(&self, features: &[f32]) -> Vec<f32> {
        let mut encoded = Vec::with_capacity(self.num_features * self.fields_per_feature);

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
