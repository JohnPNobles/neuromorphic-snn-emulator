// src/learning.rs

pub trait LearningRule {
    fn adjust_weights(
        &self,
        weights: &mut Vec<Vec<i32>>,
        axon_traces: &[i32],
        winner_neuron_idx: usize,
    );
}

pub struct StdpLearning {
    pub ltp_bonus: i32,
    pub ltd_penalty: i32,
    pub max_weight: i32,
    pub min_weight: i32,
}

impl StdpLearning {
    pub fn new(ltp_bonus: i32, ltd_penalty: i32, min_weight: i32, max_weight: i32) -> Self {
        Self {
            ltp_bonus,
            ltd_penalty,
            max_weight,
            min_weight,
        }
    }
}

impl LearningRule for StdpLearning {
    fn adjust_weights(
        &self,
        weights: &mut Vec<Vec<i32>>,
        axon_traces: &[i32],
        winner_neuron_idx: usize,
    ) {
        for axon_idx in 0..weights.len() {
            let trace_val = axon_traces[axon_idx];
            let current_weight = weights[axon_idx][winner_neuron_idx];

            if trace_val > 15 {
                let new_weight = current_weight.saturating_add(self.ltp_bonus);
                weights[axon_idx][winner_neuron_idx] = std::cmp::min(self.max_weight, new_weight);
            } else {
                let new_weight = current_weight.saturating_sub(self.ltd_penalty);
                weights[axon_idx][winner_neuron_idx] = std::cmp::max(self.min_weight, new_weight);
            }
        }
    }
}
