// src/bin/ga_main.rs
//
// Separate entry point: trains the same 65-axon x 3-neuron emulated chip
// using a genetic algorithm instead of STDP, on the same integer-only
// crossbar representation. Run with:
//   cargo run --release --bin ga_main
// (release mode strongly recommended - the GA does many forward-pass
// evaluations per generation)
//
// Each `mod` below re-declares the shared chip/encoding modules by
// explicit path so this binary can reuse them without needing a separate
// library crate or Cargo.toml changes.

#[path = "../data_loader.rs"]
mod data_loader;
#[path = "../ga.rs"]
mod ga;
#[path = "../learning.rs"]
mod learning;
#[path = "../poisson_encoder.rs"]
mod poisson_encoder;
#[path = "../receptive_field.rs"]
mod receptive_field;
#[path = "../snn.rs"]
mod snn;

use crate::data_loader::WineDatasetLoader;
use crate::ga::{GaConfig, run_ga};
use crate::learning::StdpLearning;
use crate::poisson_encoder::PoissonEncoder;
use crate::receptive_field::ReceptiveFieldEncoder;

fn main() {
    println!("Loading Genetic-Algorithm Neuromorphic Pipeline...");

    // --- Chip / encoding dimensions (identical to the STDP pipeline, so
    // the two approaches are directly cross-comparable). ---
    let raw_features = 13;
    let fields_per_feature = 5;
    let num_axons = raw_features * fields_per_feature; // 65
    let num_neurons = 3;
    let num_classes = 3;

    // "Balanced" GA budget: moderate population/generation counts, with
    // thresholds fixed (not evolved) per the project's design choice.
    let cfg = GaConfig {
        num_axons,
        num_neurons,
        num_classes,
        min_weight: 0,
        max_weight: 45,
        fixed_threshold: 40,
        timesteps: 25,      // Same timestep count as the STDP pipeline
        fitness_repeats: 2, // Average over 2 Poisson draws per sample to reduce fitness noise
        population_size: 50,
        generations: 50,
        tournament_size: 4,
        elitism: 3,
        mutation_rate: 0.08,
        mutation_span: 5,
    };

    let encoder = ReceptiveFieldEncoder::new(raw_features, fields_per_feature);
    let poisson = PoissonEncoder::new();

    match WineDatasetLoader::load_from_file("wine.data") {
        Ok(mut dataset) => {
            println!(
                "Successfully parsed {} samples from wine.data",
                dataset.len()
            );

            // Same deterministic shuffle + 80/20 split as the STDP script,
            // so both approaches are compared on identical train/test sets.
            let mut seed = 42_usize;
            for i in (1..dataset.len()).rev() {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
                let j = seed % (i + 1);
                dataset.swap(i, j);
            }

            let train_size = (dataset.len() as f32 * 0.8) as usize;
            let train_set = &dataset[0..train_size];
            let test_set = &dataset[train_size..];

            println!(
                "\n--- GA Training Phase ({} generations | population {}) ---",
                cfg.generations, cfg.population_size
            );

            // Evolve the population; fitness is measured only on
            // `train_set` (analogous to STDP's weight updates -- this is
            // the optimization signal, not the reported result).
            let (best_genome, train_fitness, neuron_assignments) =
                run_ga(&cfg, &encoder, &poisson, train_set);

            println!(
                "\nGA training complete. Best training-set (resubstitution) fitness: {:.2}%",
                train_fitness * 100.0
            );
            for (n, class) in neuron_assignments.iter().enumerate() {
                println!(" -> Neuron {} => Assigned Cultivar {}", n, class + 1);
            }

            // Held-out evaluation: re-run the best genome (frozen weights)
            // on the test set, using the neuron->class assignment learned
            // on the training set. No test sample is used for evolution,
            // fitness scoring, or label assignment.
            println!(
                "\n--- Held-Out Test Evaluation (SAMPLES: {}) ---",
                test_set.len()
            );

            let (test_accuracy, correct) = evaluate_on_fixed_assignment(
                &best_genome,
                &cfg,
                &encoder,
                &poisson,
                test_set,
                &neuron_assignments,
            );

            println!("\n=============================================");
            println!(
                " GA-Trained SNN Accuracy: {:.2}% ({}/{})",
                test_accuracy * 100.0,
                correct,
                test_set.len()
            );
            println!("=============================================");
        }
        Err(e) => println!("Execution Failure: {}", e),
    }
}

/// Like `ga::evaluate_genome`, but scores against a *given* neuron->class
/// assignment (learned on the training set) instead of deriving a new one
/// from the evaluation set itself. This is what makes the test-set number
/// a genuine held-out accuracy rather than a resubstitution accuracy --
/// the direct GA-side equivalent of the STDP script's Phase 3.
fn evaluate_on_fixed_assignment(
    genome: &ga::Genome,
    cfg: &GaConfig,
    encoder: &ReceptiveFieldEncoder,
    poisson: &PoissonEncoder,
    samples: &[data_loader::WineRecord],
    neuron_assignments: &[usize],
) -> (f32, usize) {
    let mut core =
        snn::NeuromorphicCore::new(cfg.num_axons, cfg.num_neurons, genome.weights.clone());
    for neuron in core.neurons.iter_mut() {
        neuron.threshold = cfg.fixed_threshold;
    }

    let mut correct = 0;
    for sample in samples.iter() {
        let densities = encoder.encode_to_densities(&sample.features);
        let mut spike_counts = vec![0_i32; cfg.num_neurons];

        for neuron in core.neurons.iter_mut() {
            neuron.voltage = 0;
        }
        for _step in 0..cfg.timesteps {
            let active_pins = poisson.encode_features_to_spikes(&densities);
            let output_spikes = core.forward_clock_cycle(&active_pins, None::<&StdpLearning>);
            for i in 0..cfg.num_neurons {
                if output_spikes[i] == 1 {
                    spike_counts[i] += 1;
                }
            }
        }

        let max_spikes = spike_counts.iter().max().copied().unwrap_or(0);
        if max_spikes > 0 {
            let winner = spike_counts
                .iter()
                .position(|&x| x == max_spikes)
                .unwrap_or(0);
            if neuron_assignments[winner] == sample.label {
                correct += 1;
            }
        }
    }

    (correct as f32 / samples.len() as f32, correct)
}
