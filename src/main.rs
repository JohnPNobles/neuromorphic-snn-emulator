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

fn main() {
    println!("Loading High-Resolution Gaussian Neuromorphic Pipeline...");

    let raw_features = 13;
    let fields_per_feature = 5; // Upgraded: 5 Receptive Fields per feature
    let num_axons = raw_features * fields_per_feature; // 65 Input Axons!
    let num_neurons = 3;

    // Expand crossbar array to 65x3
    let initial_weights = vec![vec![12_i32; num_neurons]; num_axons];
    let mut chip_core = NeuromorphicCore::new(num_axons, num_neurons, initial_weights);

    let pop_encoder = ReceptiveFieldEncoder::new(raw_features, fields_per_feature);
    let poisson = PoissonEncoder::new();

    let mut thresholds = vec![40_i32; num_neurons];
    let base_threshold = 30_i32;

    match WineDatasetLoader::load_from_file("wine.data") {
        Ok(mut dataset) => {
            println!(
                "Successfully parsed {} samples from wine.data",
                dataset.len()
            );

            // 1. Deterministic LCG Shuffle
            let mut seed = 42_usize;
            for i in (1..dataset.len()).rev() {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
                let j = seed % (i + 1);
                dataset.swap(i, j);
            }

            // 2. 80/20 Split
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
                // Anneal STDP rate slightly each epoch
                let pos_stdp = (2 - (epoch - 1)) as i32;
                let stdp_rule = StdpLearning::new(pos_stdp.max(1), 2, 0, 45);

                for sample in train_set.iter() {
                    for i in 0..num_neurons {
                        chip_core.neurons[i].threshold = thresholds[i];
                    }

                    let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                    let mut sample_spikes = vec![0; num_neurons];

                    for _step in 0..20 {
                        let active_pins = poisson.encode_features_to_spikes(&pop_densities);
                        let output_spikes =
                            chip_core.forward_clock_cycle(&active_pins, Some(&stdp_rule));

                        for i in 0..num_neurons {
                            if output_spikes[i] == 1 {
                                sample_spikes[i] += 1;
                            }
                        }
                    }

                    // Homeostasis threshold tuning
                    for i in 0..num_neurons {
                        if sample_spikes[i] > 0 {
                            thresholds[i] += 4;
                        } else {
                            thresholds[i] = (thresholds[i] - 1).max(base_threshold);
                        }
                    }

                    for neuron in chip_core.neurons.iter_mut() {
                        neuron.voltage = 0;
                    }
                }
                println!(" -> Epoch {}/{} Complete.", epoch, epochs);
            }

            println!(
                "\nTraining complete! Crossbar Array Dimensions: {}x{}",
                num_axons, num_neurons
            );

            // =========================================================================
            // PHASE 2: 1-TO-1 OPTIMAL LABEL MAPPING
            // =========================================================================
            println!("\n--- Phase 2: Generating Neuromorphic Label Maps ---");

            let mut assignment_matrix = vec![vec![0_usize; 3]; num_neurons];

            for sample in train_set.iter() {
                let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                let mut spike_counts = vec![0; num_neurons];

                for _step in 0..20 {
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

            // Global 1-to-1 Hungarian Matching
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
            // PHASE 3: EVALUATION TESTING PHASE (EXTENDED TIMESTEPS)
            // =========================================================================
            println!(
                "\n--- Phase 3: Beginning Testing Phase (SAMPLES: {}) ---",
                test_set.len()
            );
            let mut correct_predictions = 0;
            let mut valid_tests = 0;

            for sample in test_set.iter() {
                let pop_densities = pop_encoder.encode_to_densities(&sample.features);
                let mut spike_counts = vec![0; num_neurons];

                // 30 timesteps during test phase to eliminate Poisson edge noise
                for _step in 0..30 {
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
                if max_spikes > 0 {
                    let winning_neuron = spike_counts
                        .iter()
                        .position(|&x| x == max_spikes)
                        .unwrap_or(0);
                    let predicted_class = neuron_assignments[winning_neuron];
                    if predicted_class == sample.label {
                        correct_predictions += 1;
                    }
                    valid_tests += 1;
                }

                for neuron in chip_core.neurons.iter_mut() {
                    neuron.voltage = 0;
                }
            }

            let accuracy = if valid_tests > 0 {
                (correct_predictions as f32 / valid_tests as f32) * 100.0
            } else {
                0.0
            };

            println!("\n=============================================");
            println!(
                " High-Res PopSAN (65-Pin) Accuracy: {:.2}% ({}/{})",
                accuracy,
                correct_predictions,
                test_set.len()
            );
            println!("=============================================");
        }
        Err(e) => println!("Execution Failure: {}", e),
    }
}
