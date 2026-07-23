// src/snn.rs
//
// The emulated neuromorphic chip core. Everything in this file operates on
// i32 only -- there is no floating point anywhere in the neuron dynamics,
// synaptic crossbar, or spike routing logic. This is the "on-chip" boundary
// of the project: data enters here already spike-encoded (see
// poisson_encoder.rs / receptive_field.rs), and everything from this point
// forward is integer register state, updated one discrete clock cycle at a
// time via `forward_clock_cycle` -- i.e. an emulation of real digital
// neuromorphic hardware, not a continuous-time simulation.

use crate::learning::LearningRule;

/// A single Leaky Integrate-and-Fire (LIF) neuron, implemented entirely
/// with fixed-point integer arithmetic. `decay_numerator` / `decay_denominator`
/// together act as an integer-only leak factor (e.g. 9/10 = "keep 90% of
/// voltage per cycle") standing in for the continuous exponential decay a
/// real LIF neuron would use.
#[derive(Debug, Clone)]
pub struct LifNeuron {
    pub voltage: i32,           // Current membrane potential
    pub threshold: i32,         // Voltage level that triggers a spike
    pub decay_numerator: i32,   // Integer leak factor numerator
    pub decay_denominator: i32, // Integer leak factor denominator
    pub reset_voltage: i32,     // Voltage the neuron snaps back to after firing
}

impl LifNeuron {
    pub fn new(threshold: i32, decay_num: i32, decay_den: i32, reset_voltage: i32) -> Self {
        Self {
            voltage: reset_voltage,
            threshold,
            decay_numerator: decay_num,
            decay_denominator: decay_den,
            reset_voltage,
        }
    }

    /// Applies one clock cycle of leaky integration: decays the existing
    /// membrane voltage by the integer leak factor, then adds the incoming
    /// synaptic current (a "constant current" injection -- current-based,
    /// not conductance-based, synapse model). Voltage is floored at 0 since
    /// real membrane potential in this model can't go negative.
    pub fn integrate_and_leak(&mut self, incoming_current: i32) {
        let leaked_voltage =
            (self.voltage.saturating_mul(self.decay_numerator)) / self.decay_denominator;
        let new_voltage = leaked_voltage.saturating_add(incoming_current);
        self.voltage = std::cmp::max(0, new_voltage);
    }
}

/// The emulated crossbar array + neuron population. `synaptic_weights` is a
/// [num_axons][num_neurons] integer grid -- a direct digital analogue of a
/// physical crossbar, where each cell is one synapse's weight. Axon traces
/// are integer "eligibility" values used by STDP to know how recently each
/// input axon fired, standing in for the biological synaptic trace used in
/// spike-timing-dependent plasticity.
pub struct NeuromorphicCore {
    pub neurons: Vec<LifNeuron>,
    pub synaptic_weights: Vec<Vec<i32>>, // Crossbar cells: [axon][neuron]
    pub axon_traces: Vec<i32>,           // Per-axon recent-activity trace
    pub neuron_traces: Vec<i32>,         // Per-neuron trace (reserved for future use)
    pub trace_decay_rate: i32,           // Fixed per-cycle trace decay amount
}

impl NeuromorphicCore {
    /// Builds a crossbar of the given size with the provided initial
    /// weights, and a bank of identically-configured LIF neurons (default
    /// threshold 40, 90% per-cycle leak, reset to 0).
    pub fn new(num_axons: usize, num_neurons: usize, initial_weights: Vec<Vec<i32>>) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);
        for _ in 0..num_neurons {
            neurons.push(LifNeuron {
                voltage: 0,
                threshold: 40,
                decay_numerator: 9,
                decay_denominator: 10,
                reset_voltage: 0,
            });
        }

        Self {
            neurons,
            synaptic_weights: initial_weights,
            axon_traces: vec![0; num_axons],
            neuron_traces: vec![0; num_neurons],
            trace_decay_rate: 3,
        }
    }

    /// Advances the entire chip by exactly one discrete clock cycle:
    /// decays traces, routes active input spikes through the crossbar into
    /// each neuron's membrane potential, resolves which neuron (if any)
    /// crosses threshold, applies winner-take-all inhibition, and -- if a
    /// learning rule was supplied -- commits a plasticity update to the
    /// crossbar. Returns which neuron(s) spiked this cycle.
    pub fn forward_clock_cycle(
        &mut self,
        active_pins: &[i32],
        learning_rule: Option<&impl LearningRule>,
    ) -> Vec<i32> {
        let mut output_spikes = vec![0; self.neurons.len()];

        // 1. Leak step: every axon trace decays by a fixed integer amount
        //    each cycle, modeling the fading "recently active" eligibility
        //    signal STDP needs to judge spike timing.
        for trace in self.axon_traces.iter_mut() {
            *trace = trace.saturating_sub(self.trace_decay_rate);
        }

        // 2. Route active input spikes through the crossbar: for every
        //    axon that fired this cycle, refresh its trace to the ceiling
        //    value and inject its weighted "constant current" into every
        //    neuron's membrane potential via leaky integration.
        for axon_idx in 0..active_pins.len() {
            if active_pins[axon_idx] == 1 {
                self.axon_traces[axon_idx] = 20; // Mark axon as just-fired

                for neuron_idx in 0..self.neurons.len() {
                    let current = self.synaptic_weights[axon_idx][neuron_idx];
                    self.neurons[neuron_idx].integrate_and_leak(current);
                }
            }
        }

        // 3. Threshold check: find the neuron with the highest membrane
        //    voltage among those that crossed their firing threshold this
        //    cycle (there can be at most one winner).
        let mut winner_idx: Option<usize> = None;
        let mut max_voltage = -1;

        for i in 0..self.neurons.len() {
            if self.neurons[i].voltage >= self.neurons[i].threshold {
                if self.neurons[i].voltage > max_voltage {
                    max_voltage = self.neurons[i].voltage;
                    winner_idx = Some(i);
                }
            }
        }

        // 4. Soft winner-take-all lateral inhibition: the winning neuron
        //    spikes and resets to its reset voltage; every other neuron has
        //    its potential reduced (but not wiped out), so a strong runner-up
        //    isn't fully erased and can still compete on the next cycle.
        if let Some(winner) = winner_idx {
            output_spikes[winner] = 1;

            for i in 0..self.neurons.len() {
                if i == winner {
                    self.neurons[i].voltage = self.neurons[i].reset_voltage;
                } else {
                    self.neurons[i].voltage = (self.neurons[i].voltage - 20).max(0);
                }
            }
        }

        // 5. Learning step: if a learning rule (e.g. STDP) is active, let
        //    it commit weight updates to the crossbar based on which axons
        //    were recently active and which neuron won this cycle. Passing
        //    `None` here (as the training scripts do during evaluation)
        //    runs the chip in pure inference mode with no plasticity.
        if let Some(rule) = learning_rule {
            if let Some(winner) = winner_idx {
                rule.adjust_weights(&mut self.synaptic_weights, &self.axon_traces, winner);
            }
        }

        output_spikes
    }
}
