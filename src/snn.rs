/*
    Fixed-Point Neuromorphic Crossbar Array Emulator (Safe Stack)
*/

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

    pub fn forward_clock_cycle(&mut self, axon_spikes: &[i32]) -> Vec<bool> {
        let num_neurons = self.neurons.len();
        let mut core_outputs = vec![false; num_neurons];
        let mut accumulated_currents = vec![0_i32; num_neurons];

        // 1. Decaying Pre- and Post-Synaptic STDP Timing Traces
        for trace in self.axon_traces.iter_mut() {
            *trace = (*trace * self.trace_decay_rate) / 10;
        }
        for trace in self.neuron_traces.iter_mut() {
            *trace = (*trace * self.trace_decay_rate) / 10;
        }

        // 2. Crossbar Matrix Multiplication & Axon Trace Forcing
        for (axon_idx, &spiked) in axon_spikes.iter().enumerate() {
            if spiked == 1 {
                self.axon_traces[axon_idx] = 100; // Force trace ceiling upon impulse
                for neuron_idx in 0..num_neurons {
                    accumulated_currents[neuron_idx] = accumulated_currents[neuron_idx]
                        .saturating_add(self.synaptic_weights[axon_idx][neuron_idx]);
                }
            }
        }

        // 3. Parallel Integration & Leak Step
        for neuron_idx in 0..num_neurons {
            self.neurons[neuron_idx].integrate_and_leak(accumulated_currents[neuron_idx]);
        }

        // 4. Threshold Evaluation with Immediate Winner-Take-All (WTA) Inhibition
        let mut someone_spiked = false;
        let mut winner_idx = 0;

        for neuron_idx in 0..num_neurons {
            if self.neurons[neuron_idx].voltage >= self.neurons[neuron_idx].threshold {
                core_outputs[neuron_idx] = true;
                someone_spiked = true;
                winner_idx = neuron_idx;
                break; // Hard WTA selection: First index matching condition suppresses others
            }
        }

        // 5. Apply Resets and Update Post-Synaptic Traces
        if someone_spiked {
            for neuron_idx in 0..num_neurons {
                if neuron_idx == winner_idx {
                    self.neurons[neuron_idx].voltage = self.neurons[winner_idx].reset_voltage;
                    self.neuron_traces[neuron_idx] = 100; // Impulse spike trace for learning matching
                } else {
                    // Lateral Inhibition: Suppress losers back to base potential instantly
                    self.neurons[neuron_idx].voltage = 0;
                }
            }
        }

        core_outputs
    }
}
