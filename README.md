# Neuromorphic Wine Classifier (Rust)

An integer-only, cycle-accurate **emulation** of a digital neuromorphic chip in Rust, trained two different ways — **Spike-Timing-Dependent Plasticity (STDP)** and a **Genetic Algorithm (GA)** — to classify the UCI Wine dataset from Poisson-encoded spike trains.

This project's goal is to *emulate*, not simulate: the chip core (crossbar array + LIF neurons + learning rules) is built entirely on fixed-point integer arithmetic, structured as discrete clock cycles, the same way real digital neuromorphic hardware (e.g. TrueNorth, Loihi, SpiNNaker) operates — rather than approximating continuous-time neuron dynamics with floating-point ODE solvers.

## Table of Contents

- [Core Design Principles](#core-design-principles)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Running It](#running-it)
- [Results](#results)
- [Methodology Notes](#methodology-notes)
- [Dataset](#dataset)
- [License](#license)

## Core Design Principles

- **Emulation, not simulation.** The chip advances one discrete clock cycle at a time (`forward_clock_cycle`), with all state (voltages, weights, traces) persisted as integer registers between cycles.
- **Leaky Integrate-and-Fire (LIF) neurons.** Each neuron has a membrane voltage, firing threshold, integer leak factor, and reset voltage — implemented with pure `i32` arithmetic (no floating-point decay curves).
- **Constant current synapses (CUBA).** Each incoming spike injects a fixed current (the synaptic weight) directly into a neuron's membrane potential, rather than modulating conductance.
- **Digital crossbar array.** Synaptic weights are stored as an explicit `[axon][neuron]` integer grid — a faithful digital-crossbar representation, as opposed to a continuous-current analog memristor crossbar (which would require floating point by nature).
- **Integer-only on-chip processing.** Every component inside the emulated chip boundary — the crossbar, the LIF neurons, and both learning rules (STDP and GA) — operates exclusively on `i32` (with `i64` used only as a transient overflow-safe accumulator, never as a fractional value). Floating point is used *only* on the host side: normalizing the raw dataset, Gaussian receptive-field encoding, Poisson spike-probability draws, and final accuracy reporting. None of this touches the chip's internal state.
- **No train/test leakage.** Both training approaches learn (weights, and for STDP, thresholds) exclusively from the training split. The held-out test split is only ever used for final, frozen-weight inference — it never contributes to a single weight update, fitness score, or label assignment.

## Architecture

```
13 raw features
      ↓  min-max normalize → [0.0, 1.0]
      ↓  Gaussian receptive-field encoding (5 fields/feature)
     65 population-coded densities
      ↓  Poisson (per-timestep Bernoulli) spike encoding
     65 input axons ──────────────┐
                                  ▼
                         [65 x 3 integer crossbar]
                                  ▼
                    3 LIF neurons (one per cultivar)
                                  ▼
                  soft winner-take-all + spike output
```

Everything left of the input axons is host-side (off-chip) data preparation. Everything from the crossbar onward is the emulated chip, executed one clock cycle at a time.

## Pipeline

1. **Load & normalize** — parse `wine.data`, min-max normalize each of the 13 chemical features to `[0.0, 1.0]`.
2. **Receptive field encoding** — expand each feature into 5 overlapping Gaussian tuning-curve responses (65 total inputs), a standard population-coding technique for converting scalar values into a richer, more separable representation.
3. **Poisson encoding** — on every clock cycle, each of the 65 inputs independently fires (1) with probability equal to its receptive-field response, or stays silent (0).
4. **Chip execution** — spikes are routed through the integer crossbar into 3 LIF neurons; a soft winner-take-all rule resolves at most one winning spike per cycle.
5. **Training** (STDP *or* GA — see below) — adjusts crossbar weights (and, for STDP only, per-neuron thresholds) using only the training split.
6. **Label assignment** — a 1-to-1 neuron→cultivar mapping is derived from training-set voting behavior.
7. **Held-out evaluation** — frozen weights, frozen thresholds, frozen label mapping — scored only against the test split.

### Approach A: STDP

Pair-based spike-timing-dependent plasticity with:
- An axon "trace" that marks recent activity and decays each cycle, defining a timing window for potentiation vs. depression.
- **Weight normalization** on each winning neuron's incoming weight column, which is what prevents one neuron from permanently starving the others — the standard failure mode in unnormalized competitive Hebbian learning.
- **Homeostatic threshold tuning**, pulling each neuron's firing threshold toward a target spike rate every training sample, bounded so no neuron can freeze out or dominate indefinitely.

### Approach B: Genetic Algorithm

Metaheuristic search over the same crossbar weight space, with thresholds held fixed (not evolved) so the comparison isolates the *search strategy* (local, timing-based STDP vs. global, population-based GA) rather than adding extra adaptive degrees of freedom to one side.

- **Genome** = one full crossbar weight matrix.
- **Fitness** = resubstitution accuracy on the training set (frozen weights, no learning, derived label mapping) — analogous to a loss function.
- **Selection**: tournament selection.
- **Crossover**: uniform, per-weight-cell.
- **Mutation**: per-cell random integer perturbation, clamped to weight bounds.
- **Elitism**: top genomes carried forward unchanged each generation.

## Project Structure

```
src/
├── main.rs              STDP training + evaluation pipeline (entry point)
├── snn.rs                LIF neuron + crossbar core (the emulated chip)
├── learning.rs           STDP plasticity rule
├── data_loader.rs         Wine dataset parsing + normalization
├── receptive_field.rs    Gaussian population-coding encoder
├── poisson_encoder.rs   Poisson (Bernoulli-per-timestep) spike encoder
├── ga.rs                 Genetic algorithm (genome, selection, crossover, mutation)
└── bin/
    └── ga_main.rs        GA training + evaluation pipeline (separate entry point)
```

`ga_main.rs` reuses `snn.rs`, `learning.rs`, `data_loader.rs`, `receptive_field.rs`, and `poisson_encoder.rs` directly via explicit `#[path]` module declarations, so both training approaches run against the exact same chip core, encoder, and dataset split — no code duplication of the underlying hardware model.

## Running It

Requires `wine.data` (UCI Wine dataset) in the project root.

```bash
# STDP-trained model
cargo run --release

# GA-trained model
cargo run --release --bin ga_main
```

Release mode is recommended for both, and especially for the GA binary, which evaluates many genomes per generation.

## Results

Both approaches are trained and evaluated on an identical 80/20 split (fixed-seed deterministic shuffle), with identical crossbar dimensions (65×3 axons/neurons), identical weight bounds, and identical timestep counts — making the two directly cross-comparable.

| Approach | Runs | Average Accuracy | Range |
|---|---|---|---|
| STDP | 50 | 90.89% | 83.33% – 100% |
| Genetic Algorithm | 50 | 97.94% | 94.44% - 100% |

*(All figures are held-out test-set accuracy; no test sample ever contributes to a weight update, fitness score, or label assignment in either approach.)*

## Methodology Notes

A few points worth stating explicitly for anyone reviewing this as research:

- **Digital vs. analog crossbar**: this project emulates a *digital* crossbar (discrete integer cells, explicit per-cycle summation), not an *analog* memristor crossbar (which would require continuous current summation and floating point by nature). Integer-only arithmetic is valid specifically because the target is digital neuromorphic hardware.
- **STDP vs. GA is not purely an "algorithm vs. algorithm" comparison.** STDP has one extra adaptive degree of freedom (per-neuron threshold homeostasis) that the GA does not use, by design, to keep the GA's search space limited to weights only. Any accuracy gap between the two should be read with that difference in mind.
- **Floating point boundary**: floats appear only in host-side dataset normalization, Gaussian receptive-field computation, Poisson probability draws, and final printed accuracy percentages — never inside neuron state, crossbar weights, or learning-rule updates.

## Dataset

[UCI Machine Learning Repository — Wine Data Set](https://archive.ics.uci.edu/dataset/109/wine). 178 samples, 13 continuous chemical features, 3 cultivar classes.

## License

MIT — see [LICENSE](LICENSE). You retain copyright; this just makes clear that others may use, modify, and redistribute the code (with attribution), which is the standard choice for research code you may want to reference, cite, or build on later.
