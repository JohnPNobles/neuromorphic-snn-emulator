// src/main.rs
mod data_loader;
mod learning;
mod poisson_encoder;
mod snn;

use crate::data_loader::WineDatasetLoader;
use crate::learning::StdpLearning;
use crate::poisson_encoder::PoissonEncoder;
use crate::snn::NeuromorphicCore;

fn main() {
    println!("Loading Neuromorphic Hardware Simulator Pipeline...");

    let num_axons = 13;
    let num_neurons = 3;

    // Initialize baseline starting matrix weights
    let initial_weights = vec![vec![15_i32; num_neurons]; num_axons];
    let mut chip_core = NeuromorphicCore::new(num_axons, num_neurons, initial_weights);
    let encoder = PoissonEncoder::new();

    // Balanced STDP parameters: Strong LTD ensures clear feature pruning
    let stdp_rule = StdpLearning::new(1, 2, 0, 45);

    let mut thresholds = vec![45_i32; num_neurons];
    let base_threshold = 35_i32;

    match WineDatasetLoader::load_from_file("wine.data") {
        Ok(mut dataset) => {
            println!(
                "Successfully parsed {} samples from wine.data",
                dataset.len()
            );

            // 1. Pseudo-random LCG Shuffle
            let mut seed = 42_usize;
            for i in (1..dataset.len()).rev() {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
                let j = seed % (i + 1);
                dataset.swap(i, j);
            }

            // 2. 80/20 Train/Test Data Split
            let train_size = (dataset.len() as f32 * 0.8) as usize;
            let train_set = &dataset[0..train_size];
            let test_set = &dataset[train_size..];

            // =========================================================================
            // PHASE 1: UNSUPERVISED TRAINING (STDP ON)
            // =========================================================================
            println!(
                "\n--- Phase 1: Beginning Training Phase (SAMPLES: {}) ---",
                train_set.len()
            );

            for (idx, sample) in train_set.iter().enumerate() {
                for i in 0..num_neurons {
                    chip_core.neurons[i].threshold = thresholds[i];
                }

                let mut sample_spikes = vec![0; num_neurons];

                for _step in 0..20 {
                    let active_pins = encoder.encode_features_to_spikes(&sample.features);
                    let output_spikes =
                        chip_core.forward_clock_cycle(&active_pins, Some(&stdp_rule));

                    for i in 0..num_neurons {
                        if output_spikes[i] == 1 {
                            sample_spikes[i] += 1;
                        }
                    }
                }

                // Homeostasis threshold adjustments
                for i in 0..num_neurons {
                    if sample_spikes[i] > 0 {
                        thresholds[i] += 3;
                    } else {
                        thresholds[i] = (thresholds[i] - 1).max(base_threshold);
                    }
                }

                for neuron in chip_core.neurons.iter_mut() {
                    neuron.voltage = 0;
                }

                if idx % 40 == 0 {
                    println!(
                        " -> Training sample {}/{} | Firing distribution: {:?}",
                        idx,
                        train_set.len(),
                        sample_spikes
                    );
                }
            }

            println!("\nTraining complete! Finalized Synaptic Weights Matrix:");
            println!("{:#?}", chip_core.synaptic_weights);

            // =========================================================================
            // PHASE 2: POST-TRAINING LABEL ASSIGNMENT (STDP OFF)
            // =========================================================================
            println!("\n--- Phase 2: Generating Neuromorphic Label Maps ---");

            // Grid matrix: [neuron_index][target_class_counts]
            let mut assignment_matrix = vec![vec![0_usize; 3]; num_neurons];

            for sample in train_set.iter() {
                let mut spike_counts = vec![0; num_neurons];

                for _step in 0..20 {
                    let active_pins = encoder.encode_features_to_spikes(&sample.features);
                    // STDP is disabled (None passed) to evaluate static cluster affinity
                    let output_spikes =
                        chip_core.forward_clock_cycle(&active_pins, None::<&StdpLearning>);

                    for i in 0..num_neurons {
                        if output_spikes[i] == 1 {
                            spike_counts[i] += 1;
                        }
                    }
                }

                // Match highest spiking output to target class
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

            // Map each neuron index to whichever class it claimed most frequently
            let mut neuron_assignments = vec![0_usize; num_neurons];
            for i in 0..num_neurons {
                let max_class_count = assignment_matrix[i].iter().max().copied().unwrap_or(0);
                let assigned_class = assignment_matrix[i]
                    .iter()
                    .position(|&x| x == max_class_count)
                    .unwrap_or(0);
                neuron_assignments[i] = assigned_class;
                println!(
                    " -> Structural Mapping Identity: Neuron Index {} => Assigned Class Cultivar {}",
                    i,
                    assigned_class + 1
                );
            }

            // =========================================================================
            // PHASE 3: EVALUATION TESTING PHASE (STDP OFF)
            // =========================================================================
            println!(
                "\n--- Phase 3: Beginning Testing Phase (SAMPLES: {}) ---",
                test_set.len()
            );
            let mut correct_predictions = 0;
            let mut valid_tests = 0;

            for sample in test_set.iter() {
                let mut spike_counts = vec![0; num_neurons];

                for _step in 0..20 {
                    let active_pins = encoder.encode_features_to_spikes(&sample.features);
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

                    // FIXED: Resolve predictions using our dynamically mapped identity vector!
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
                " Mapped SNN Evaluation Accuracy: {:.2}% ({}/{})",
                accuracy,
                correct_predictions,
                test_set.len()
            );
            println!("=============================================");
        }
        Err(e) => println!("Execution Failure: {}", e),
    }
}
