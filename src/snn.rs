/*
    Fixed-Point Neuromorphic Crossbar Array Emulator (Safe Stack)
*/

use crate::learning::LearningRule;

#[derive(Debug, Clone)]
pub struct LifNeuron {
    pub voltage: i32,
    pub threshold: i32,
    pub decay_numerator: i32,
    pub decay_denominator: i32,
    pub reset_voltage: i32,
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

    /// Evaluates integration and decay physics safely using saturating operations.
    pub fn integrate_and_leak(&mut self, incoming_current: i32) {
        // V_leaked = (V_old * num) / den
        let leaked_voltage =
            (self.voltage.saturating_mul(self.decay_numerator)) / self.decay_denominator;

        // Overflow-safe integration bound down at 0V floor minimum
        let new_voltage = leaked_voltage.saturating_add(incoming_current);
        self.voltage = std::cmp::max(0, new_voltage);
    }
}

pub struct NeuromorphicCore {
    pub neurons: Vec<LifNeuron>,
    /// 2D Crossbar Matrix: weights[input_axon][neuron_index]
    pub synaptic_weights: Vec<Vec<i32>>,
    /// STDP Timing Traces: Parallel history vectors tracking relative intervals
    pub axon_traces: Vec<i32>,
    pub neuron_traces: Vec<i32>,
    pub trace_decay_rate: i32, // Structural leak of the memory trace
}

impl NeuromorphicCore {
    pub fn new(num_neurons: usize, num_axons: usize, initial_weights: Vec<Vec<i32>>) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..num_neurons {
            // Threshold = 200 (2.0V), Decay = 8/10, Reset = 0
            neurons.push(LifNeuron::new(200, 8, 10, 0));
        }

        Self {
            neurons,
            synaptic_weights: initial_weights,
            axon_traces: vec![0; num_axons],
            neuron_traces: vec![0; num_neurons],
            trace_decay_rate: 9, // E.g., decays by 9/10 each cycle
        }
    }

    pub fn forward_clock_cycle(
        &mut self,
        axon_spikes: &[i32],
        learning_rule: Option<&impl LearningRule>,
    ) -> Vec<i32> {
        let num_neurons = self.neurons.len();
        let mut outward_spikes = vec![0; num_neurons];

        // 1. Decay Traces
        for trace in self.axon_traces.iter_mut() {
            if *trace > 0 {
                *trace -= 1;
            }
        }
        for trace in self.neuron_traces.iter_mut() {
            if *trace > 0 {
                *trace -= 1;
            }
        }

        // 2. Capture incoming spikes into traces
        for axon_idx in 0..axon_spikes.len() {
            if axon_spikes[axon_idx] == 1 {
                self.axon_traces[axon_idx] = 100; // Reset trace to max strength
            }
        }

        // 3. Accumulate currents
        let mut accumulated_currents = vec![0_i32; num_neurons];
        for axon_idx in 0..axon_spikes.len() {
            if axon_spikes[axon_idx] == 1 {
                for neuron_idx in 0..num_neurons {
                    accumulated_currents[neuron_idx] = accumulated_currents[neuron_idx]
                        .saturating_add(self.synaptic_weights[axon_idx][neuron_idx]);
                }
            }
        }

        // 4. Integrate voltages and check for spikes
        let mut winner_idx = 0;
        let mut max_voltage = -1;
        let mut someone_spiked = false;

        for neuron_idx in 0..num_neurons {
            let n = &mut self.neurons[neuron_idx];
            n.voltage = n.voltage.saturating_add(accumulated_currents[neuron_idx]);

            // Leaky integration decay
            if n.voltage > 0 {
                n.voltage -= 1;
            }

            if n.voltage >= n.threshold {
                someone_spiked = true;
                outward_spikes[neuron_idx] = 1;
                if n.voltage > max_voltage {
                    max_voltage = n.voltage;
                    winner_idx = neuron_idx;
                }
            }
        }

        // 5. Lateral Inhibition (WTA) and STDP Weight Adjustment
        if someone_spiked {
            for neuron_idx in 0..num_neurons {
                if neuron_idx == winner_idx {
                    self.neurons[neuron_idx].voltage = self.neurons[winner_idx].reset_voltage;
                    self.neuron_traces[neuron_idx] = 100;
                } else {
                    self.neurons[neuron_idx].voltage = 0; // Inhibited
                }
            }

            // If a learning engine is provided, adjust weights locally based on current axon traces
            if let Some(rule) = learning_rule {
                rule.adjust_weights(&mut self.synaptic_weights, &self.axon_traces, winner_idx);
            }
        }

        outward_spikes
    }
}
