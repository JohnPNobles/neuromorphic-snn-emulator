// src/bin/ga_main.rs
//
// GA-based training + repeated deterministic evaluation pipeline.
//
// Behavior:
// - Train GA model once
// - Run 50 evaluations on random subsets of test data (n = 36)
// - Use a fixed RNG seed for reproducibility
// - Export accuracies to an Excel file

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

use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand_chacha::ChaCha8Rng;

use rust_xlsxwriter::{Workbook, XlsxError};

const NUM_RUNS: usize = 50;
const SAMPLE_SIZE: usize = 36;
const RNG_SEED: u64 = 1337;

fn main() {
    println!("Loading Genetic-Algorithm Neuromorphic Pipeline...");

    let raw_features = 13;
    let fields_per_feature = 5;
    let num_axons = raw_features * fields_per_feature;
    let num_neurons = 3;
    let num_classes = 3;

    let cfg = GaConfig {
        num_axons,
        num_neurons,
        num_classes,
        min_weight: 0,
        max_weight: 45,
        fixed_threshold: 40,
        timesteps: 25,
        fitness_repeats: 2,
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

            // Deterministic shuffle
            let mut seed = 42_usize;
            for i in (1..dataset.len()).rev() {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
                let j = seed % (i + 1);
                dataset.swap(i, j);
            }

            // Train/test split
            let train_size = (dataset.len() as f32 * 0.8) as usize;
            let train_set = &dataset[0..train_size];
            let test_set = &dataset[train_size..];

            println!(
                "\n--- GA Training Phase ({} generations | population {}) ---",
                cfg.generations, cfg.population_size
            );

            let (best_genome, train_fitness, neuron_assignments) =
                run_ga(&cfg, &encoder, &poisson, train_set);

            println!(
                "\nGA training complete. Best training accuracy: {:.2}%",
                train_fitness * 100.0
            );

            // Seeded RNG for reproducibility
            let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut accuracies = Vec::with_capacity(NUM_RUNS);

            println!(
                "\n--- Repeated Test Evaluation ({} runs | sample size {}) ---",
                NUM_RUNS, SAMPLE_SIZE
            );

            // Create index pool once (0..test_set.len())
            let indices: Vec<usize> = (0..test_set.len()).collect();

            for run in 0..NUM_RUNS {
                let sample: Vec<&data_loader::WineRecord> = indices
                    .sample(&mut rng, SAMPLE_SIZE)
                    .map(|&i| &test_set[i])
                    .collect();

                let (acc, _) = evaluate_on_fixed_assignment_refs(
                    &best_genome,
                    &cfg,
                    &encoder,
                    &poisson,
                    &sample,
                    &neuron_assignments,
                );

                println!("Run {:02} -> {:.4}", run + 1, acc);
                accuracies.push(acc);
            }

            write_to_excel(&accuracies).expect("Excel write failed");

            println!("\nSaved results to ga_accuracy_runs.xlsx");
        }
        Err(e) => println!("Execution Failure: {}", e),
    }
}

/// Excel writer
fn write_to_excel(data: &[f32]) -> Result<(), XlsxError> {
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();

    worksheet.write_string(0, 0, "Run")?;
    worksheet.write_string(0, 1, "Accuracy")?;

    for (i, acc) in data.iter().enumerate() {
        worksheet.write_number((i + 1) as u32, 0, (i + 1) as f64)?;
        worksheet.write_number((i + 1) as u32, 1, *acc as f64)?;
    }

    workbook.save("ga_accuracy_runs.xlsx")?;
    Ok(())
}

/// Evaluation using references (no cloning required)
fn evaluate_on_fixed_assignment_refs(
    genome: &ga::Genome,
    cfg: &GaConfig,
    encoder: &ReceptiveFieldEncoder,
    poisson: &PoissonEncoder,
    samples: &[&data_loader::WineRecord],
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
