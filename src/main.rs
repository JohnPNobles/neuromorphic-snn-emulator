// src/main.rs
//
// STDP training + evaluation pipeline for the emulated neuromorphic chip.
// Pipeline shape: raw wine data -> min-max normalize -> receptive field
// population coding -> Poisson spike encoding -> integer-only crossbar +
// LIF neurons -> STDP plasticity during training -> frozen-weight,
// held-out test evaluation.

mod data_loader;
mod learning;
mod poisson_encoder;
mod receptive_field;
mod snn;

use crate::data_loader::WineDatasetLoader;
use crate::learning::StdpLearning;
use crate::poisson_encoder::PoissonEncoder;
use crate::receptive_field::ReceptiveFieldEncoder;
use crate::snn::NeuromorphicCore;
use rand::RngExt;

fn main() {
    println!("Loading High-Resolution Gaussian Neuromorphic Pipeline...");

    // --- Chip / encoding dimensions ---
    let raw_features = 13;
    let fields_per_feature = 5; // 5 receptive fields per feature
    let num_axons = raw_features * fields_per_feature; // 65 input axons
    let num_neurons = 3; // One output neuron per wine cultivar

    // Jittered initial weights (rather than identical constants) break the
    // symmetry that would otherwise let one neuron win purely by chance in
    // the earliest training samples.
    let mut rng = rand::rng();
    let mut initial_weights = vec![vec![0_i32; num_neurons]; num_axons];
    for axon in initial_weights.iter_mut() {
        for w in axon.iter_mut() {
            *w = 10 + (rng.random::<u32>() % 5) as i32; // range [10, 14]
        }
    }

    let mut chip_core = NeuromorphicCore::new(num_axons, num_neurons, initial_weights);

    let pop_encoder = ReceptiveFieldEncoder::new(raw_features, fields_per_feature);
    let poisson = PoissonEncoder::new();

    // Unified timestep count used consistently across training, label
    // assignment, and test evaluation, so spike-count statistics are
    // directly comparable across all three phases.
    let timesteps = 25;

    // Homeostatic threshold-tuning parameters: instead of an unbounded
    // ratchet, each neuron's threshold is pulled toward a target firing
    // rate (`target_spikes` spikes per `timesteps`-cycle window) and
    // clamped so no neuron can freeze out permanently or dominate forever.
    let base_threshold = 30_i32;
    let max_threshold = 120_i32;
    let target_spikes = 4_i32;
    let homeostasis_gain = 2_i32;
    let mut thresholds = vec![40_i32; num_neurons];

    match WineDatasetLoader::load_from_file("wine.data") {
        Ok(mut dataset) => {
            println!(
                "Successfully parsed {} samples from wine.data",
                dataset.len()
            );

            // 1. Deterministic LCG shuffle -- fixed seed so STDP and GA
            //    runs are trained/tested on identical splits for a fair
            //    cross-comparison.
            let mut seed = 42_usize;
            for i in (1..dataset.len()).rev() {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
                let j = seed % (i + 1);
                dataset.swap(i, j);
            }

            // 2. 80/20 train/test split.
            let train_size = (dataset.len() as f32 * 0.8) as usize;
            let train_set = &dataset[0..train_size];
            let test_set = &dataset[train_size..];

            // =========================================================================
            // PHASE 1: MULTI-EPOCH TRAINING (STDP ON | 65 AXONS)
            // =========================================================================
            let epochs = 3;
            println!(
                "\n--- Phase 1: Training Phase ({} Epochs | {} Input Axons) ---",
                epochs, num_axons
            );

            for epoch in 1..=epochs {
                // Anneal the LTP bonus down slightly each epoch, so learning
                // is more aggressive early and more fine-grained later.
                let pos_stdp = (2 - (epoch - 1)) as i32;
                let stdp_rule = StdpLearning::new(pos_stdp.max(1), 2, 0, 45, num_axons as i32 * 12);

                // Train on every sample in the training set once per epoch.
                for sample in train_set.iter() {
                    // Push the current homeostatic thresholds onto the chip
                    // before running this sample.
                    for i in 0..num_neurons {
                        chip_core.neurons[i].threshold = thresholds[i];
                    }

                    let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                    let mut sample_spikes = vec![0; num_neurons];

                    // Present this one sample for `timesteps` clock cycles,
                    // re-drawing a fresh Poisson spike pattern each cycle,
                    // with STDP active so the crossbar learns from it.
                    for _step in 0..timesteps {
                        let active_pins = poisson.encode_features_to_spikes(&pop_densities);
                        let output_spikes =
                            chip_core.forward_clock_cycle(&active_pins, Some(&stdp_rule));

                        for i in 0..num_neurons {
                            if output_spikes[i] == 1 {
                                sample_spikes[i] += 1;
                            }
                        }
                    }

                    // Homeostatic threshold update: pull each neuron's
                    // threshold toward the target spike count, clamped to a
                    // safe range. This adjusts the *local* `thresholds`
                    // array; it is pushed back onto `chip_core.neurons` at
                    // the top of the next sample's loop iteration -- or,
                    // for the final converged values, right after the
                    // training loop ends below.
                    for i in 0..num_neurons {
                        let delta = (sample_spikes[i] as i32 - target_spikes) * homeostasis_gain;
                        thresholds[i] =
                            (thresholds[i] + delta).clamp(base_threshold, max_threshold);
                    }

                    // Reset membrane voltages between samples so one
                    // sample's leftover potential can't bleed into the next.
                    for neuron in chip_core.neurons.iter_mut() {
                        neuron.voltage = 0;
                    }
                }
                println!(" -> Epoch {}/{} Complete.", epoch, epochs);
            }

            // FIX: write the final converged homeostatic thresholds back to
            // the live chip once training is done. Without this, Phase 2/3
            // would run against whatever thresholds happened to be loaded
            // during the very last training sample (one update behind the
            // fully converged values computed above).
            for i in 0..num_neurons {
                chip_core.neurons[i].threshold = thresholds[i];
            }

            println!(
                "\nTraining complete! Crossbar Array Dimensions: {}x{}",
                num_axons, num_neurons
            );

            // =========================================================================
            // PHASE 2: 1-TO-1 OPTIMAL LABEL MAPPING
            // =========================================================================
            // Weights are now frozen (no learning rule passed below). We
            // run every training sample through the chip in pure inference
            // mode to discover which neuron tends to win for which class,
            // then resolve a clean 1-to-1 neuron -> class mapping.
            println!("\n--- Phase 2: Generating Neuromorphic Label Maps ---");

            let mut assignment_matrix = vec![vec![0_usize; 3]; num_neurons];

            for sample in train_set.iter() {
                let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                let mut spike_counts = vec![0; num_neurons];

                for _step in 0..timesteps {
                    let active_pins = poisson.encode_features_to_spikes(&pop_densities);
                    let output_spikes =
                        chip_core.forward_clock_cycle(&active_pins, None::<&StdpLearning>);

                    for i in 0..num_neurons {
                        if output_spikes[i] == 1 {
                            spike_counts[i] += 1;
                        }
                    }
                }

                // Whichever neuron fired most for this sample "votes" for
                // this sample's true class.
                let max_spikes = spike_counts.iter().max().copied().unwrap_or(0);
                if max_spikes > 0 {
                    let winning_neuron = spike_counts
                        .iter()
                        .position(|&x| x == max_spikes)
                        .unwrap_or(0);
                    assignment_matrix[winning_neuron][sample.label] += 1;
                }

                for neuron in chip_core.neurons.iter_mut() {
                    neuron.voltage = 0;
                }
            }

            // Resolve a 1-to-1 neuron -> class assignment via greedy
            // highest-affinity-first matching (a simple Hungarian-style
            // matching, since we only have 3 neurons and 3 classes).
            let mut neuron_assignments = vec![0_usize; num_neurons];
            let mut claimed_classes = vec![false; 3];

            let mut candidates = Vec::new();
            for n in 0..num_neurons {
                for c in 0..3 {
                    candidates.push((assignment_matrix[n][c], n, c));
                }
            }
            candidates.sort_by(|a, b| b.0.cmp(&a.0));

            let mut assigned_neurons = vec![false; num_neurons];

            for (_score, n, c) in candidates {
                if !assigned_neurons[n] && !claimed_classes[c] {
                    neuron_assignments[n] = c;
                    assigned_neurons[n] = true;
                    claimed_classes[c] = true;
                }
            }

            // Fallback: assign any leftover neuron to any leftover
            // unclaimed class (shouldn't normally trigger with 3 vs 3).
            for n in 0..num_neurons {
                if !assigned_neurons[n] {
                    if let Some(free_class) = claimed_classes.iter().position(|&claimed| !claimed) {
                        neuron_assignments[n] = free_class;
                        claimed_classes[free_class] = true;
                    }
                }
                println!(
                    " -> Spatial Mapping: Neuron Index {} => Assigned Cultivar {} (Affinity Matrix: {:?})",
                    n,
                    neuron_assignments[n] + 1,
                    assignment_matrix[n]
                );
            }

            // =========================================================================
            // PHASE 3: EVALUATION TESTING PHASE
            // =========================================================================
            // Held-out test set: weights, thresholds, and neuron->class
            // assignments are all frozen (learned entirely from train_set
            // above). No sample here was ever used for weight updates or
            // label-mapping -- this is the only number reported as final
            // accuracy.
            println!(
                "\n--- Phase 3: Beginning Testing Phase (SAMPLES: {}) ---",
                test_set.len()
            );
            let mut correct_predictions = 0;
            let total_tests = test_set.len();

            for sample in test_set.iter() {
                let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                let mut spike_counts = vec![0; num_neurons];

                for _step in 0..timesteps {
                    let active_pins = poisson.encode_features_to_spikes(&pop_densities);
                    let output_spikes =
                        chip_core.forward_clock_cycle(&active_pins, None::<&StdpLearning>);

                    for i in 0..num_neurons {
                        if output_spikes[i] == 1 {
                            spike_counts[i] += 1;
                        }
                    }
                }

                let max_spikes = spike_counts.iter().max().copied().unwrap_or(0);
                // A sample with zero spikes across the whole window counts
                // as a genuine miss (included in the denominator), rather
                // than being silently excluded from accuracy accounting.
                if max_spikes > 0 {
                    let winning_neuron = spike_counts
                        .iter()
                        .position(|&x| x == max_spikes)
                        .unwrap_or(0);
                    let predicted_class = neuron_assignments[winning_neuron];
                    if predicted_class == sample.label {
                        correct_predictions += 1;
                    }
                }

                for neuron in chip_core.neurons.iter_mut() {
                    neuron.voltage = 0;
                }
            }

            let accuracy = (correct_predictions as f32 / total_tests as f32) * 100.0;

            println!("\n=============================================");
            println!(
                " High-Res PopSAN (65-Pin) Accuracy: {:.2}% ({}/{})",
                accuracy, correct_predictions, total_tests
            );
            println!("=============================================");
        }
        Err(e) => println!("Execution Failure: {}", e),
    }
}
