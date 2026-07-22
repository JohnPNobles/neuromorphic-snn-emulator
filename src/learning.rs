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
    pub target_weight_sum: i32,
}

impl StdpLearning {
    pub fn new(
        ltp_bonus: i32,
        ltd_penalty: i32,
        min_weight: i32,
        max_weight: i32,
        target_weight_sum: i32,
    ) -> Self {
        Self {
            ltp_bonus,
            ltd_penalty,
            max_weight,
            min_weight,
            target_weight_sum,
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
        if weights.is_empty() {
            return;
        }

        let ltp_cutoff = 15;
        let ltd_cutoff = 5;

        // FIX: only the winner is updated. The previous "competitor suppression"
        // term only ever decreased losing neurons' weights with no path back up,
        // which permanently killed them (weights floored at min_weight -> zero
        // current forever -> that neuron can never win again). Removed.
        for axon_idx in 0..weights.len() {
            let trace_val = axon_traces[axon_idx];
            let current_weight = weights[axon_idx][winner_neuron_idx];

            if trace_val > ltp_cutoff {
                let new_weight = current_weight.saturating_add(self.ltp_bonus);
                weights[axon_idx][winner_neuron_idx] = std::cmp::min(self.max_weight, new_weight);
            } else if trace_val < ltd_cutoff {
                let new_weight = current_weight.saturating_sub(self.ltd_penalty);
                weights[axon_idx][winner_neuron_idx] = std::cmp::max(self.min_weight, new_weight);
            }
            // neutral zone: leave unchanged
        }

        // FIX: synaptic weight normalization. Keeps the winner's *total* incoming
        // weight roughly constant, so strengthening some axons (LTP) forces a
        // proportional tradeoff against its own other axons — not against other
        // neurons. This is what actually prevents one neuron from monopolizing
        // every sample: its total synaptic "budget" can't grow without bound.
        let column_sum: i64 = weights
            .iter()
            .map(|row| row[winner_neuron_idx] as i64)
            .sum();

        if column_sum > 0 {
            let target = self.target_weight_sum as i64;
            for row in weights.iter_mut() {
                let w = row[winner_neuron_idx] as i64;
                let scaled = (w * target) / column_sum;
                row[winner_neuron_idx] =
                    scaled.clamp(self.min_weight as i64, self.max_weight as i64) as i32;
            }
        }
    }
}
