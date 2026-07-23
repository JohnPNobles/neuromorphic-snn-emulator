// src/data_loader.rs
//
// Host-side (off-chip) data preparation. Reads the raw UCI "wine.data" CSV,
// parses each row into a label + 13 chemical feature values, and min-max
// normalizes the features into [0.0, 1.0]. This normalization happens once,
// before any data ever reaches the emulated chip, and uses f32 purely as a
// data-preparation convenience -- it is NOT part of the integer-only on-chip
// pipeline (crossbar, LIF neurons, learning rules).

use std::fs::File;
use std::io::{BufRead, BufReader};

/// A single parsed wine sample: a class label (0, 1, or 2) and its 13
/// normalized chemical feature values.
pub struct WineRecord {
    pub label: usize,       // Mapped to 0, 1, or 2 for our 3 output neurons
    pub features: Vec<f32>, // 13 chemical features scaled to [0.0, 1.0]
}

/// Namespace struct for the dataset-loading routine. Holds no state -- it
/// exists purely to group the loader function under a clear type name.
pub struct WineDatasetLoader;

impl WineDatasetLoader {
    /// Reads a raw 'wine.data' CSV file from disk, normalizes features, and
    /// returns one WineRecord per valid row.
    pub fn load_from_file(file_path: &str) -> Result<Vec<WineRecord>, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        // Hardcoded min/max bounds for min-max normalization across the 13
        // attributes (taken from the known value ranges of the UCI Wine
        // dataset, rather than computed at runtime, so normalization is
        // deterministic and independent of which rows end up in the train
        // vs. test split).
        let mins = [
            11.03, 0.74, 1.36, 10.60, 70.0, 0.98, 0.34, 0.13, 0.41, 1.28, 0.48, 1.27, 278.0,
        ];
        let maxs = [
            14.83, 5.80, 3.23, 30.00, 162.0, 3.88, 5.08, 0.66, 3.58, 13.00, 1.71, 4.00, 1680.0,
        ];

        // Parse the file line by line -- one wine sample per line.
        for line in reader.lines() {
            let line_str = line?;
            if line_str.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line_str.split(',').collect();
            if parts.len() < 14 {
                continue; // Expecting label + 13 features; skip malformed rows.
            }

            // 1. Extract raw label. UCI Wine uses classes 1, 2, 3 -- shift
            //    down by one so labels line up with 0-indexed output neurons.
            let raw_label: usize = parts[0].parse().unwrap_or(1);
            let label = raw_label.saturating_sub(1);

            // 2. Extract and min-max normalize each of the 13 continuous
            //    features into [0.0, 1.0], so they're all on a comparable
            //    scale before being handed to the receptive field encoder.
            let mut features = Vec::with_capacity(13);
            for i in 0..13 {
                let val: f32 = parts[i + 1].parse().unwrap_or(0.0);

                let normalized = (val - mins[i]) / (maxs[i] - mins[i]);
                features.push(normalized.clamp(0.0, 1.0));
            }

            records.push(WineRecord { label, features });
        }

        Ok(records)
    }
}
