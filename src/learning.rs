// src/learning.rs
//
// On-chip spike-timing-dependent plasticity (STDP). Entirely integer
// arithmetic (i64 is used only as a temporary overflow-safe accumulator
// during weight normalization, never as a fractional value) -- this file
// is part of the emulated chip's plasticity circuitry, not a host-side
// convenience.

/// Generic interface for a plasticity rule the crossbar can invoke after
/// each clock cycle. Allows `NeuromorphicCore::forward_clock_cycle` to stay
/// agnostic to which specific learning algorithm is plugged in (only STDP
/// is implemented here, but this trait is what lets inference-only runs
/// pass `None` instead).
pub trait LearningRule {
    fn adjust_weights(
        &self,
        weights: &mut Vec<Vec<i32>>,
        axon_traces: &[i32],
        winner_neuron_idx: usize,
    );
}

/// Pair-based STDP rule with bounded weights and synaptic normalization.
/// `target_weight_sum` is the total incoming weight a neuron's column is
/// normalized toward after each update, which is what prevents runaway
/// winner-take-all weight growth (see the normalization step below).
pub struct StdpLearning {
    pub ltp_bonus: i32,         // Weight increase on long-term potentiation
    pub ltd_penalty: i32,       // Weight decrease on long-term depression
    pub max_weight: i32,        // Upper clamp for any single synapse
    pub min_weight: i32,        // Lower clamp for any single synapse
    pub target_weight_sum: i32, // Target total weight per neuron's input column
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
    /// Updates only the winning neuron's incoming weight column, then
    /// renormalizes that column back toward a fixed total. Losing neurons'
    /// weights are left untouched by this rule -- weight normalization
    /// (not competitor suppression) is what keeps a single neuron from
    /// permanently starving the others.
    fn adjust_weights(
        &self,
        weights: &mut Vec<Vec<i32>>,
        axon_traces: &[i32],
        winner_neuron_idx: usize,
    ) {
        if weights.is_empty() {
            return;
        }

        // Trace thresholds define the STDP timing window: axons that fired
        // recently enough (trace still high) get potentiated; axons that
        // fired too long ago (trace decayed low) get depressed; anything
        // in between is left alone as a neutral zone to avoid flip-flopping
        // on small timing jitter.
        let ltp_cutoff = 15;
        let ltd_cutoff = 5;

        // 1. Apply LTP/LTD only to the winning neuron's incoming weights.
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
            // else: neutral zone, weight unchanged this cycle.
        }

        // 2. Synaptic weight normalization: rescale the winner's entire
        //    incoming weight column so its total stays near
        //    `target_weight_sum`. This is what forces a tradeoff *within*
        //    a neuron's own synapses (strengthen some axons, weaken
        //    others) instead of letting one neuron's total input strength
        //    grow without bound relative to the others.
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
