/*
    Fixed-Point Neuromorphic Crossbar Array Emulator
*/

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

    pub fn step(&mut self, incoming_current: i32) -> bool {
        // Apply exponential decay: V_leaked = (V_old * num) / den
        let leaked_voltage = (self.voltage * self.decay_numerator) / self.decay_denominator;
        self.voltage = leaked_voltage + incoming_current;

        if self.voltage >= self.threshold {
            self.voltage = self.reset_voltage;
            true // Spike fired!
        } else {
            false
        }
    }
}

pub struct NeuromorphicCore {
    pub neurons: Vec<LifNeuron>,
    /// 2D Matrix: weights[input_axon][neuron_index]
    pub synaptic_weights: Vec<Vec<i32>>,
}

impl NeuromorphicCore {
    pub fn new(num_neurons: usize, _num_axons: usize, initial_weights: Vec<Vec<i32>>) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..num_neurons {
            // Threshold = 200 (2.0V), Decay = 8/10 (20% loss per step), Reset = 0
            neurons.push(LifNeuron::new(200, 8, 10, 0));
        }

        Self {
            neurons,
            synaptic_weights: initial_weights,
        }
    }

    pub fn forward_clock_cycle(&mut self, axon_spikes: &[i32]) -> Vec<bool> {
        let num_neurons = self.neurons.len();
        let mut core_outputs = vec![false; num_neurons];
        let mut accumulated_currents = vec![0; num_neurons];

        // 1. Crossbar Matrix Multiplication
        for (axon_idx, &spiked) in axon_spikes.iter().enumerate() {
            if spiked == 1 {
                for neuron_idx in 0..num_neurons {
                    accumulated_currents[neuron_idx] += self.synaptic_weights[axon_idx][neuron_idx];
                }
            }
        }

        // 2. Parallel State Machine Evaluation
        for neuron_idx in 0..num_neurons {
            let spiked = self.neurons[neuron_idx].step(accumulated_currents[neuron_idx]);
            core_outputs[neuron_idx] = spiked;
        }

        core_outputs
    }
}
