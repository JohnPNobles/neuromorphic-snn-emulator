// src/ga.rs
//
// Genetic-algorithm training for the emulated chip. Instead of local
// spike-timing updates (STDP), each individual in the population *is* a
// full crossbar weight matrix. Fitness is measured by running the chip
// forward (no learning, fixed thresholds) over the training set and
// scoring classification accuracy after a best-effort neuron -> class
// assignment (the same 1-to-1 matching logic used in the STDP script's
// Phase 2). Weight values (the genome itself) are i32 throughout -- the GA
// operates entirely within the same integer-only crossbar representation
// as the STDP pipeline.

use crate::data_loader::WineRecord;
use crate::learning::StdpLearning;
use crate::poisson_encoder::PoissonEncoder;
use crate::receptive_field::ReceptiveFieldEncoder;
use crate::snn::NeuromorphicCore;
use rand::RngExt;

/// One candidate solution: a full [axon][neuron] integer weight matrix,
/// exactly the same shape and value range as the STDP crossbar.
#[derive(Clone, Debug)]
pub struct Genome {
    pub weights: Vec<Vec<i32>>, // [num_axons][num_neurons]
}

/// All tunable parameters for the GA run, gathered in one place so
/// `ga_main.rs` can configure a full experiment without touching this file.
pub struct GaConfig {
    pub num_axons: usize,
    pub num_neurons: usize,
    pub num_classes: usize,
    pub min_weight: i32,
    pub max_weight: i32,
    pub fixed_threshold: i32, // Thresholds are NOT evolved -- fixed for every neuron
    pub timesteps: usize,
    pub fitness_repeats: usize, // Repeats per sample to reduce Poisson noise in fitness
    pub population_size: usize,
    pub generations: usize,
    pub tournament_size: usize,
    pub elitism: usize,
    pub mutation_rate: f64, // Probability per-gene of mutation
    pub mutation_span: i32, // Max +/- delta applied on mutation
}

impl Genome {
    /// Creates one genome with every weight drawn uniformly at random
    /// within the allowed weight bounds -- used to seed generation zero
    /// of the population with maximum diversity.
    pub fn random(cfg: &GaConfig) -> Self {
        let mut rng = rand::rng();
        let mut weights = vec![vec![0_i32; cfg.num_neurons]; cfg.num_axons];
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                let span = (cfg.max_weight - cfg.min_weight + 1) as u32;
                *w = cfg.min_weight + (rng.random::<u32>() % span) as i32;
            }
        }
        Genome { weights }
    }
}

/// Builds a chip from this genome's weights, runs every training sample
/// through it in pure inference mode (no plasticity), and derives both a
/// neuron->class assignment and a resubstitution accuracy on that same
/// training set. This accuracy is what the GA optimizes as fitness -- it is
/// NOT the final reported result (see `ga_main.rs::evaluate_on_fixed_assignment`
/// for the genuine held-out test evaluation).
pub fn evaluate_genome(
    genome: &Genome,
    cfg: &GaConfig,
    encoder: &ReceptiveFieldEncoder,
    poisson: &PoissonEncoder,
    samples: &[WineRecord],
) -> (f32, Vec<usize>) {
    let mut core = NeuromorphicCore::new(cfg.num_axons, cfg.num_neurons, genome.weights.clone());
    for neuron in core.neurons.iter_mut() {
        neuron.threshold = cfg.fixed_threshold;
    }

    let mut assignment_matrix = vec![vec![0_usize; cfg.num_classes]; cfg.num_neurons];
    let mut per_sample_winner: Vec<Option<usize>> = Vec::with_capacity(samples.len());

    // Run every sample through the chip, tallying which neuron wins for
    // which true class (repeated `fitness_repeats` times per sample and
    // summed, to average out some of the Poisson encoding's inherent
    // trial-to-trial noise before scoring fitness).
    for sample in samples.iter() {
        let densities = encoder.encode_to_densities(&sample.features);
        let mut spike_counts = vec![0_i32; cfg.num_neurons];

        for _rep in 0..cfg.fitness_repeats {
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
        }

        let max_spikes = spike_counts.iter().max().copied().unwrap_or(0);
        if max_spikes > 0 {
            let winner = spike_counts
                .iter()
                .position(|&x| x == max_spikes)
                .unwrap_or(0);
            assignment_matrix[winner][sample.label] += 1;
            per_sample_winner.push(Some(winner));
        } else {
            per_sample_winner.push(None);
        }
    }

    let neuron_assignments =
        resolve_assignment(&assignment_matrix, cfg.num_neurons, cfg.num_classes);

    // Score accuracy on the training set using the assignment just derived
    // from that same set -- this is the fitness value, analogous to a loss
    // function guiding STDP's weight updates.
    let mut correct = 0;
    for (sample, winner) in samples.iter().zip(per_sample_winner.iter()) {
        if let Some(w) = winner {
            if neuron_assignments[*w] == sample.label {
                correct += 1;
            }
        }
    }

    let accuracy = correct as f32 / samples.len() as f32;
    (accuracy, neuron_assignments)
}

/// 1-to-1 greedy matching: identical logic to the STDP script's Phase 2,
/// extracted here so both training approaches share one implementation of
/// "given vote counts, resolve the best neuron -> class mapping."
pub fn resolve_assignment(
    assignment_matrix: &[Vec<usize>],
    num_neurons: usize,
    num_classes: usize,
) -> Vec<usize> {
    let mut neuron_assignments = vec![0_usize; num_neurons];
    let mut claimed_classes = vec![false; num_classes];

    let mut candidates = Vec::new();
    for n in 0..num_neurons {
        for c in 0..num_classes {
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
    }
    neuron_assignments
}

/// Randomly samples `tournament_size` genomes from the population and
/// returns the fittest of that sample -- selection pressure without
/// needing to normalize or scale fitness values.
fn tournament_select<'a>(
    population: &'a [Genome],
    fitnesses: &[f32],
    tournament_size: usize,
) -> &'a Genome {
    let mut rng = rand::rng();
    let mut best_idx = (rng.random::<u32>() as usize) % population.len();
    let mut best_fit = fitnesses[best_idx];

    for _ in 1..tournament_size {
        let idx = (rng.random::<u32>() as usize) % population.len();
        if fitnesses[idx] > best_fit {
            best_fit = fitnesses[idx];
            best_idx = idx;
        }
    }
    &population[best_idx]
}

/// Produces one child genome by independently choosing, for every single
/// weight cell, which of the two parents to inherit it from.
fn uniform_crossover(a: &Genome, b: &Genome) -> Genome {
    let mut rng = rand::rng();
    let mut weights = a.weights.clone();
    for (row_idx, row) in weights.iter_mut().enumerate() {
        for (col_idx, w) in row.iter_mut().enumerate() {
            if rng.random_bool(0.5) {
                *w = b.weights[row_idx][col_idx];
            }
        }
    }
    Genome { weights }
}

/// Applies random integer perturbations to a genome in place: each weight
/// cell independently has `mutation_rate` probability of being nudged by a
/// random delta in [-mutation_span, +mutation_span], clamped back into the
/// legal weight range.
fn mutate(genome: &mut Genome, cfg: &GaConfig) {
    let mut rng = rand::rng();
    let span = (2 * cfg.mutation_span + 1) as u32;
    for row in genome.weights.iter_mut() {
        for w in row.iter_mut() {
            if rng.random_bool(cfg.mutation_rate) {
                let delta = (rng.random::<u32>() % span) as i32 - cfg.mutation_span;
                *w = (*w + delta).clamp(cfg.min_weight, cfg.max_weight);
            }
        }
    }
}

/// Runs the full generational GA loop: evaluate every genome's fitness,
/// track the best genome ever seen, carry the top `elitism` genomes forward
/// unchanged, and fill the rest of the next generation via tournament
/// selection + uniform crossover + mutation. Returns the best genome found
/// across all generations, its training-set fitness, and its neuron->class
/// assignment.
pub fn run_ga(
    cfg: &GaConfig,
    encoder: &ReceptiveFieldEncoder,
    poisson: &PoissonEncoder,
    train_set: &[WineRecord],
) -> (Genome, f32, Vec<usize>) {
    // Generation 0: fully random population for maximum initial diversity.
    let mut population: Vec<Genome> = (0..cfg.population_size)
        .map(|_| Genome::random(cfg))
        .collect();

    let mut best_genome: Option<(Genome, f32, Vec<usize>)> = None;

    for genr in 1..=cfg.generations {
        // Evaluate every genome in the current population against the
        // training set.
        let evaluations: Vec<(f32, Vec<usize>)> = population
            .iter()
            .map(|g| evaluate_genome(g, cfg, encoder, poisson, train_set))
            .collect();
        let fitnesses: Vec<f32> = evaluations.iter().map(|(f, _)| *f).collect();

        // Track this generation's best individual.
        let mut gen_best_idx = 0;
        for i in 1..fitnesses.len() {
            if fitnesses[i] > fitnesses[gen_best_idx] {
                gen_best_idx = i;
            }
        }
        let gen_best_fit = fitnesses[gen_best_idx];

        // Update the all-time best genome if this generation beat it.
        if best_genome
            .as_ref()
            .map_or(true, |(_, f, _)| gen_best_fit > *f)
        {
            best_genome = Some((
                population[gen_best_idx].clone(),
                gen_best_fit,
                evaluations[gen_best_idx].1.clone(),
            ));
        }

        println!(
            " -> Generation {}/{}: best train fitness = {:.2}%  (running best = {:.2}%)",
            genr,
            cfg.generations,
            gen_best_fit * 100.0,
            best_genome.as_ref().unwrap().1 * 100.0
        );

        // Rank the population by fitness so the top `elitism` genomes can
        // be carried forward unchanged.
        let mut ranked_indices: Vec<usize> = (0..population.len()).collect();
        ranked_indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

        let mut next_population = Vec::with_capacity(cfg.population_size);
        for &idx in ranked_indices.iter().take(cfg.elitism) {
            next_population.push(population[idx].clone());
        }

        // Fill the remainder of the next generation via selection,
        // crossover, and mutation.
        while next_population.len() < cfg.population_size {
            let parent_a = tournament_select(&population, &fitnesses, cfg.tournament_size);
            let parent_b = tournament_select(&population, &fitnesses, cfg.tournament_size);
            let mut child = uniform_crossover(parent_a, parent_b);
            mutate(&mut child, cfg);
            next_population.push(child);
        }

        population = next_population;
    }

    let (genome, fitness, assignments) = best_genome.expect("at least one generation must run");
    (genome, fitness, assignments)
}
