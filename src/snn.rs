// src/snn.rs
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

    pub fn integrate_and_leak(&mut self, incoming_current: i32) {
        let leaked_voltage =
            (self.voltage.saturating_mul(self.decay_numerator)) / self.decay_denominator;
        let new_voltage = leaked_voltage.saturating_add(incoming_current);
        self.voltage = std::cmp::max(0, new_voltage);
    }
}

pub struct NeuromorphicCore {
    pub neurons: Vec<LifNeuron>,
    pub synaptic_weights: Vec<Vec<i32>>,
    pub axon_traces: Vec<i32>,
    pub neuron_traces: Vec<i32>,
    pub trace_decay_rate: i32,
}

impl NeuromorphicCore {
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

    pub fn forward_clock_cycle(
        &mut self,
        active_pins: &[i32],
        learning_rule: Option<&impl LearningRule>,
    ) -> Vec<i32> {
        let mut output_spikes = vec![0; self.neurons.len()];

        // 1. Structural Leak Step for existing dynamic memory traces
        for trace in self.axon_traces.iter_mut() {
            *trace = trace.saturating_sub(self.trace_decay_rate);
        }

        // 2. Accumulate incoming currents through Leak Physics
        for axon_idx in 0..active_pins.len() {
            if active_pins[axon_idx] == 1 {
                // FIXED: Mark the historical trace as active (e.g., hit a high ceiling value)
                self.axon_traces[axon_idx] = 20;

                for neuron_idx in 0..self.neurons.len() {
                    let current = self.synaptic_weights[axon_idx][neuron_idx];
                    // FIXED: Route via the proper physics model instead of direct addition
                    self.neurons[neuron_idx].integrate_and_leak(current);
                }
            }
        }

        // 3. Evaluate firing threshold metrics
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

        // 4. SOFT Winner-Take-All lateral inhibition (Enables Ensemble Co-firing)
        if let Some(winner) = winner_idx {
            output_spikes[winner] = 1;

            for i in 0..self.neurons.len() {
                if i == winner {
                    self.neurons[i].voltage = self.neurons[i].reset_voltage; // Reset winner to 0
                } else {
                    // SOFT INHIBITION: Reduce competitor potentials by 20 mV instead of wiping to 0
                    self.neurons[i].voltage = (self.neurons[i].voltage - 20).max(0);
                }
            }
        }

        // 5. Commit STDP adjustments back to the structural crossbar matrix cells
        if let Some(rule) = learning_rule {
            if let Some(winner) = winner_idx {
                rule.adjust_weights(&mut self.synaptic_weights, &self.axon_traces, winner);
            }
        }

        output_spikes
    }
}
