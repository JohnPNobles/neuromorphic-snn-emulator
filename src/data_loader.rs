// src/data_loader.rs
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct WineRecord {
    pub label: usize,       // Mapped to 0, 1, or 2 for our 3 output neurons
    pub features: Vec<f32>, // 13 chemical features scaled [0.0, 1.0]
}

pub struct WineDatasetLoader;

impl WineDatasetLoader {
    /// Reads a raw 'wine.data' CSV string from disk, normalizes features, and returns records
    pub fn load_from_file(file_path: &str) -> Result<Vec<WineRecord>, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        // Hardcoded min/max thresholds for min-max normalization across the 13 attributes
        // (Calculated from known boundaries of the UCI dataset distribution to prevent floating overflows)
        let mins = [
            11.03, 0.74, 1.36, 10.60, 70.0, 0.98, 0.34, 0.13, 0.41, 1.28, 0.48, 1.27, 278.0,
        ];
        let maxs = [
            14.83, 5.80, 3.23, 30.00, 162.0, 3.88, 5.08, 0.66, 3.58, 13.00, 1.71, 4.00, 1680.0,
        ];

        for line in reader.lines() {
            let line_str = line?;
            if line_str.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line_str.split(',').collect();
            if parts.len() < 14 {
                continue;
            } // Expecting label + 13 features

            // 1. Extract raw label (UCI uses 1, 2, 3 -> map to 0, 1, 2 index layout)
            let raw_label: usize = parts[0].parse().unwrap_or(1);
            let label = raw_label.saturating_sub(1);

            // 2. Extract and normalize the 13 continuous elements
            let mut features = Vec::with_capacity(13);
            for i in 0..13 {
                let val: f32 = parts[i + 1].parse().unwrap_or(0.0);

                // Min-Max normalizer equation to safely output scales strictly bounded [0.0, 1.0]
                let normalized = (val - mins[i]) / (maxs[i] - mins[i]);
                features.push(normalized.clamp(0.0, 1.0));
            }

            records.push(WineRecord { label, features });
        }

        Ok(records)
    }
}
