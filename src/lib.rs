//! Quantum-inspired probabilistic decision making.
//!
//! This crate provides tools for decision making inspired by quantum mechanics,
//! including superposition of states, measurement/collapse, entanglement between
//! agents, constructive/destructive interference, and quantum tunneling.


// ============================================================================
// superposition module
// ============================================================================

pub mod superposition {
    

    /// A quantum state representing superposition of multiple values.
    #[derive(Debug, Clone)]
    pub struct Superposition<T: Clone> {
        states: Vec<(T, f64)>, // (value, probability_amplitude)
    }

    impl<T: Clone> Superposition<T> {
        pub fn new() -> Self {
            Self { states: Vec::new() }
        }

        pub fn with_states(states: Vec<(T, f64)>) -> Self {
            let mut s = Self { states };
            s.normalize();
            s
        }

        pub fn add_state(&mut self, value: T, amplitude: f64) {
            self.states.push((value, amplitude));
            self.normalize();
        }

        fn normalize(&mut self) {
            let total: f64 = self.states.iter().map(|(_, a)| a * a).sum::<f64>().sqrt();
            if total > 0.0 {
                for (_, a) in &mut self.states {
                    *a /= total;
                }
            }
        }

        pub fn probabilities(&self) -> Vec<(&T, f64)> {
            let total: f64 = self.states.iter().map(|(_, a)| a * a).sum::<f64>();
            if total == 0.0 { return vec![]; }
            self.states.iter().map(|(v, a)| (v, (a * a) / total)).collect()
        }

        pub fn state_count(&self) -> usize {
            self.states.len()
        }

        pub fn is_empty(&self) -> bool {
            self.states.is_empty()
        }

        pub fn states(&self) -> &[(T, f64)] {
            &self.states
        }

        pub fn amplitude_sum(&self) -> f64 {
            self.states.iter().map(|(_, a)| a.abs()).sum()
        }

        pub fn probability_of<F>(&self, predicate: F) -> f64
        where F: Fn(&T) -> bool {
            let probs = self.probabilities();
            probs.iter().filter(|(v, _)| predicate(v)).map(|(_, p)| p).sum()
        }

        /// Scale all amplitudes by a factor.
        pub fn scale(&mut self, factor: f64) {
            for (_, a) in &mut self.states {
                *a *= factor;
            }
        }

        /// Combine two superpositions (tensor product simplified).
        pub fn combine<U: Clone>(&self, other: &Superposition<U>) -> Superposition<(T, U)> {
            let mut combined = Vec::new();
            for (v1, a1) in &self.states {
                for (v2, a2) in &other.states {
                    combined.push(((v1.clone(), v2.clone()), a1 * a2));
                }
            }
            Superposition::with_states(combined)
        }
    }

    impl<T: Clone> Default for Superposition<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Create a uniform superposition over values.
    pub fn uniform<T: Clone>(values: Vec<T>) -> Superposition<T> {
        let n = values.len() as f64;
        let amplitude = 1.0 / n.sqrt();
        Superposition::with_states(values.into_iter().map(|v| (v, amplitude)).collect())
    }

    /// Create a superposition with Gaussian-like weighting around an index.
    pub fn gaussian<T: Clone>(values: Vec<T>, center: usize, sigma: f64) -> Superposition<T> {
        let states: Vec<(T, f64)> = values.into_iter().enumerate().map(|(i, v)| {
            let dist = (i as f64 - center as f64).abs();
            let amp = (-dist * dist / (2.0 * sigma * sigma)).exp();
            (v, amp)
        }).collect();
        Superposition::with_states(states)
    }
}

// ============================================================================
// measurement module
// ============================================================================

pub mod measurement {
    use std::collections::HashMap;
    use super::superposition::Superposition;

    /// Result of measuring (collapsing) a superposition.
    #[derive(Debug, Clone)]
    pub struct MeasurementResult<T: Clone> {
        pub value: T,
        pub probability: f64,
        pub collapsed_from: usize,
    }

    impl<T: Clone> MeasurementResult<T> {
        pub fn new(value: T, probability: f64, collapsed_from: usize) -> Self {
            Self { value, probability, collapsed_from }
        }
    }

    /// Deterministic measurement using a seed (for reproducibility).
    pub fn measure_deterministic<T: Clone + PartialEq>(sup: &Superposition<T>, seed: f64) -> MeasurementResult<T> {
        let probs = sup.probabilities();
        if probs.is_empty() {
            panic!("Cannot measure empty superposition");
        }
        let mut cumulative = 0.0;
        let seed_clamped = seed.clamp(0.0, 1.0);
        for (idx, (value, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if seed_clamped <= cumulative || idx == probs.len() - 1 {
                return MeasurementResult::new((*value).clone(), *prob, idx);
            }
        }
        MeasurementResult::new(probs[0].0.clone(), probs[0].1, 0)
    }

    /// Measure using a simple pseudo-random approach based on input.
    pub fn measure<T: Clone + PartialEq>(sup: &Superposition<T>, entropy: f64) -> MeasurementResult<T> {
        let frac = entropy.fract().abs();
        measure_deterministic(sup, frac)
    }

    /// Get the most probable outcome without collapsing.
    pub fn most_probable<T: Clone>(sup: &Superposition<T>) -> Option<MeasurementResult<T>> {
        let probs = sup.probabilities();
        probs.into_iter().enumerate().max_by(|a, b| (a.1).1.partial_cmp(&(b.1).1).unwrap())
            .map(|(idx, (v, p))| MeasurementResult::new((*v).clone(), p, idx))
    }

    /// Get the least probable outcome.
    pub fn least_probable<T: Clone>(sup: &Superposition<T>) -> Option<MeasurementResult<T>> {
        let probs = sup.probabilities();
        probs.into_iter().enumerate().min_by(|a, b| (a.1).1.partial_cmp(&(b.1).1).unwrap())
            .map(|(idx, (v, p))| MeasurementResult::new((*v).clone(), p, idx))
    }

    /// Perform N measurements and return distribution.
    pub fn sample_distribution<T: Clone + PartialEq + std::hash::Hash + Eq>(sup: &Superposition<T>, n: usize) -> HashMap<T, usize> {
        let mut counts: HashMap<T, usize> = HashMap::new();
        for i in 0..n {
            let seed = (i as f64 * 0.618033988749895).fract();
            let result = measure_deterministic(sup, seed);
            *counts.entry(result.value).or_insert(0) += 1;
        }
        counts
    }

    /// Expectation value for numeric superpositions.
    pub fn expectation(sup: &Superposition<f64>) -> f64 {
        let probs = sup.probabilities();
        probs.iter().map(|(v, p)| *v * p).sum()
    }

    /// Variance of numeric superposition.
    pub fn variance(sup: &Superposition<f64>) -> f64 {
        let exp = expectation(sup);
        let probs = sup.probabilities();
        probs.iter().map(|(v, p)| (*v - exp).powi(2) * p).sum()
    }

}

// ============================================================================
// entanglement module
// ============================================================================

pub mod entanglement {
    use std::collections::HashMap;

    /// An entangled pair of agents with correlated decisions.
    #[derive(Debug, Clone)]
    pub struct EntangledPair {
        pub id_a: String,
        pub id_b: String,
        pub correlation: f64, // -1.0 to 1.0
        pub strength: f64,    // 0.0 to 1.0
    }

    impl EntangledPair {
        pub fn new(id_a: &str, id_b: &str, correlation: f64, strength: f64) -> Self {
            Self {
                id_a: id_a.to_string(),
                id_b: id_b.to_string(),
                correlation: correlation.clamp(-1.0, 1.0),
                strength: strength.clamp(0.0, 1.0),
            }
        }

        pub fn correlate_decision(&self, decision_a: f64) -> f64 {
            let noise = 1.0 - self.strength;
            let pseudo_noise = (decision_a * 7.3).sin() * noise;
            decision_a * self.correlation + pseudo_noise
        }

        pub fn is_strong(&self) -> bool {
            self.strength > 0.7
        }

        pub fn is_anticorrelated(&self) -> bool {
            self.correlation < -0.5
        }

        pub fn decay(&mut self, factor: f64) {
            self.strength *= factor;
        }
    }

    /// Manages entanglement relationships between agents.
    #[derive(Debug, Clone)]
    pub struct EntanglementRegistry {
        pairs: Vec<EntangledPair>,
        agent_states: HashMap<String, f64>,
    }

    impl EntanglementRegistry {
        pub fn new() -> Self {
            Self { pairs: Vec::new(), agent_states: HashMap::new() }
        }

        pub fn entangle(&mut self, id_a: &str, id_b: &str, correlation: f64, strength: f64) {
            self.pairs.push(EntangledPair::new(id_a, id_b, correlation, strength));
        }

        pub fn set_state(&mut self, agent_id: &str, value: f64) {
            self.agent_states.insert(agent_id.to_string(), value);
            // Propagate through entanglements
            for pair in &self.pairs {
                if pair.id_a == agent_id {
                    let correlated = pair.correlate_decision(value);
                    self.agent_states.entry(pair.id_b.clone())
                        .and_modify(|v| *v = *v * (1.0 - pair.strength) + correlated * pair.strength);
                } else if pair.id_b == agent_id {
                    let correlated = pair.correlate_decision(value);
                    self.agent_states.entry(pair.id_a.clone())
                        .and_modify(|v| *v = *v * (1.0 - pair.strength) + correlated * pair.strength);
                }
            }
        }

        pub fn get_state(&self, agent_id: &str) -> Option<f64> {
            self.agent_states.get(agent_id).copied()
        }

        pub fn get_pair(&self, id_a: &str, id_b: &str) -> Option<&EntangledPair> {
            self.pairs.iter().find(|p|
                (p.id_a == id_a && p.id_b == id_b) ||
                (p.id_a == id_b && p.id_b == id_a)
            )
        }

        pub fn pairs_for(&self, agent_id: &str) -> Vec<&EntangledPair> {
            self.pairs.iter().filter(|p| p.id_a == agent_id || p.id_b == agent_id).collect()
        }

        pub fn pair_count(&self) -> usize {
            self.pairs.len()
        }

        pub fn decay_all(&mut self, factor: f64) {
            for pair in &mut self.pairs {
                pair.decay(factor);
            }
            self.pairs.retain(|p| p.strength > 0.01);
        }

        pub fn agent_count(&self) -> usize {
            self.agent_states.len()
        }

        pub fn all_pairs(&self) -> &[EntangledPair] {
            &self.pairs
        }
    }

    impl Default for EntanglementRegistry {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ============================================================================
// interference module
// ============================================================================

pub mod interference {
    /// Constructive or destructive interference between probability amplitudes.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum InterferenceType {
        Constructive,
        Destructive,
        Neutral,
    }

    /// Result of interference calculation.
    #[derive(Debug, Clone)]
    pub struct InterferenceResult {
        pub amplitude: f64,
        pub interference_type: InterferenceType,
        pub components: usize,
    }

    impl InterferenceResult {
        pub fn new(amplitude: f64, components: usize) -> Self {
            let itype = if amplitude > 1.0 {
                InterferenceType::Constructive
            } else if amplitude < 0.0 {
                InterferenceType::Destructive
            } else {
                InterferenceType::Neutral
            };
            Self { amplitude, interference_type: itype, components }
        }
    }

    /// Compute interference between two amplitudes with phases.
    pub fn interfere(amp1: f64, phase1: f64, amp2: f64, phase2: f64) -> InterferenceResult {
        let real1 = amp1 * phase1.cos();
        let imag1 = amp1 * phase1.sin();
        let real2 = amp2 * phase2.cos();
        let imag2 = amp2 * phase2.sin();
        let real_sum = real1 + real2;
        let imag_sum = imag1 + imag2;
        let resultant = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        InterferenceResult::new(resultant, 2)
    }

    /// Multi-amplitude interference.
    pub fn multi_interfere(amplitudes: &[(f64, f64)]) -> InterferenceResult {
        let real_sum: f64 = amplitudes.iter().map(|(a, p)| a * p.cos()).sum();
        let imag_sum: f64 = amplitudes.iter().map(|(a, p)| a * p.sin()).sum();
        let resultant = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        InterferenceResult::new(resultant, amplitudes.len())
    }

    /// Create constructive interference (phases aligned).
    pub fn constructive(amplitudes: &[f64]) -> InterferenceResult {
        let sum: f64 = amplitudes.iter().sum();
        InterferenceResult::new(sum, amplitudes.len())
    }

    /// Create destructive interference (phases opposed).
    pub fn destructive(amplitudes: &[f64]) -> InterferenceResult {
        let sum: f64 = amplitudes.iter().map(|a| -a).sum::<f64>() + amplitudes.first().unwrap_or(&0.0) * 2.0;
        InterferenceResult::new(sum, amplitudes.len())
    }

    /// Compute interference pattern from N equally-spaced sources.
    pub fn pattern(n_sources: usize, amplitude: f64, phase_step: f64) -> Vec<InterferenceResult> {
        let points = n_sources * 4;
        (0..points).map(|p| {
            let t = p as f64 * std::f64::consts::PI * 2.0 / points as f64;
            let components: Vec<(f64, f64)> = (0..n_sources)
                .map(|s| {
                    let phase = s as f64 * phase_step + t;
                    (amplitude, phase)
                })
                .collect();
            multi_interfere(&components)
        }).collect()
    }

    /// Check if interference is constructive.
    pub fn is_constructive(result: &InterferenceResult) -> bool {
        result.interference_type == InterferenceType::Constructive
    }

    /// Compute probability from interference result.
    pub fn to_probability(result: &InterferenceResult) -> f64 {
        result.amplitude * result.amplitude
    }

    /// Mach-Zehnder-like interference: split, phase shift, recombine.
    pub fn mach_zehnder(input_amplitude: f64, phase_shift: f64) -> (InterferenceResult, InterferenceResult) {
        // Two paths: direct and phase-shifted
        let a1 = input_amplitude / 2.0_f64.sqrt();
        let a2 = input_amplitude / 2.0_f64.sqrt();
        // Output port 1: constructive
        let out1 = interfere(a1, 0.0, a2, phase_shift);
        // Output port 2: complementary
        let out2 = interfere(a1, 0.0, a2, phase_shift + std::f64::consts::PI);
        (out1, out2)
    }
}

// ============================================================================
// tunneling module
// ============================================================================

pub mod tunneling {
    /// A potential barrier in decision space.
    #[derive(Debug, Clone)]
    pub struct Barrier {
        pub height: f64,
        pub width: f64,
        pub position: f64,
    }

    impl Barrier {
        pub fn new(height: f64, width: f64, position: f64) -> Self {
            Self { height, width, position }
        }

        /// Calculate transmission probability through this barrier.
        pub fn transmission_probability(&self, energy: f64) -> f64 {
            if energy >= self.height {
                return 1.0;
            }
            if energy <= 0.0 || self.height <= 0.0 || self.width <= 0.0 {
                return 0.0;
            }
            let kappa = (2.0 * (self.height - energy)).sqrt();
            let exponent = -2.0 * kappa * self.width;
            exponent.exp().min(1.0)
        }

        pub fn is_impenetrable(&self) -> bool {
            self.height > 1000.0 && self.width > 100.0
        }
    }

    /// A landscape of barriers representing local optima.
    #[derive(Debug, Clone)]
    pub struct BarrierLandscape {
        pub barriers: Vec<Barrier>,
        pub current_position: f64,
        pub current_energy: f64,
    }

    impl BarrierLandscape {
        pub fn new(current_position: f64, current_energy: f64) -> Self {
            Self { barriers: Vec::new(), current_position, current_energy }
        }

        pub fn add_barrier(&mut self, barrier: Barrier) {
            self.barriers.push(barrier);
        }

        /// Find the nearest barrier in a direction.
        pub fn nearest_barrier(&self, direction: f64) -> Option<&Barrier> {
            self.barriers.iter()
                .filter(|b| {
                    if direction > 0.0 { b.position > self.current_position }
                    else { b.position < self.current_position }
                })
                .min_by(|a, b| {
                    let da = (a.position - self.current_position).abs();
                    let db = (b.position - self.current_position).abs();
                    da.partial_cmp(&db).unwrap()
                })
        }

        /// Attempt to tunnel through barriers to reach a target.
        pub fn attempt_tunnel(&self, target: f64, direction: f64) -> TunnelResult {
            let mut position = self.current_position;
            let mut total_prob = 1.0;
            let mut barriers_crossed = 0;

            let sorted_barriers: Vec<&Barrier> = self.barriers.iter()
                .filter(|b| {
                    if direction > 0.0 {
                        b.position >= self.current_position && b.position <= target
                    } else {
                        b.position <= self.current_position && b.position >= target
                    }
                })
                .collect();

            for barrier in &sorted_barriers {
                let trans = barrier.transmission_probability(self.current_energy);
                total_prob *= trans;
                barriers_crossed += 1;
                position = barrier.position + barrier.width * direction.signum();
            }

            TunnelResult {
                success_probability: total_prob,
                final_position: if total_prob > 0.01 { target } else { position },
                barriers_crossed,
                reached_target: total_prob > 0.5,
            }
        }

        /// Add energy (e.g., from annealing) to increase tunneling probability.
        pub fn add_energy(&mut self, amount: f64) {
            self.current_energy += amount;
        }

        pub fn barrier_count(&self) -> usize {
            self.barriers.len()
        }
    }

    /// Result of a tunneling attempt.
    #[derive(Debug, Clone)]
    pub struct TunnelResult {
        pub success_probability: f64,
        pub final_position: f64,
        pub barriers_crossed: usize,
        pub reached_target: bool,
    }

    impl TunnelResult {
        pub fn is_successful(&self) -> bool {
            self.reached_target
        }
    }

    /// Simulated annealing-inspired energy boost for tunneling.
    pub fn annealing_energy(temperature: f64, base_energy: f64) -> f64 {
        base_energy + temperature * (temperature * 2.0).sin().abs()
    }

    /// Optimize tunneling path by choosing best direction.
    pub fn optimize_direction(landscape: &BarrierLandscape, candidates: &[f64]) -> (f64, TunnelResult) {
        candidates.iter()
            .map(|&target| {
                let direction = target - landscape.current_position;
                let result = landscape.attempt_tunnel(target, direction.signum());
                (target, result)
            })
            .max_by(|a, b| a.1.success_probability.partial_cmp(&b.1.success_probability).unwrap())
            .unwrap_or((landscape.current_position, TunnelResult {
                success_probability: 0.0,
                final_position: landscape.current_position,
                barriers_crossed: 0,
                reached_target: false,
            }))
    }
}

// Re-exports
pub use superposition::{Superposition, uniform, gaussian};
pub use measurement::{MeasurementResult, measure, measure_deterministic, most_probable, least_probable, expectation, variance};
pub use entanglement::{EntangledPair, EntanglementRegistry};
pub use interference::{InterferenceResult, InterferenceType, interfere, multi_interfere, constructive, destructive};
pub use tunneling::{Barrier, BarrierLandscape, TunnelResult, annealing_energy, optimize_direction};

#[cfg(test)]
mod tests {
    use super::*;

    // ---- superposition tests (12) ----

    #[test]
    fn test_superposition_new() {
        let sup: superposition::Superposition<i32> = superposition::Superposition::new();
        assert!(sup.is_empty());
        assert_eq!(sup.state_count(), 0);
    }

    #[test]
    fn test_superposition_add_state() {
        let mut sup = superposition::Superposition::new();
        sup.add_state("a", 1.0);
        sup.add_state("b", 1.0);
        assert_eq!(sup.state_count(), 2);
    }

    #[test]
    fn test_superposition_probabilities_sum_to_one() {
        let sup = superposition::uniform(vec![10, 20, 30, 40]);
        let probs = sup.probabilities();
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_uniform_superposition() {
        let sup = superposition::uniform(vec![1, 2, 3]);
        assert_eq!(sup.state_count(), 3);
        let probs = sup.probabilities();
        for (_, p) in probs {
            assert!((p - 1.0/3.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_gaussian_superposition() {
        let sup = superposition::gaussian(vec![0, 1, 2, 3, 4], 2, 1.0);
        assert_eq!(sup.state_count(), 5);
        // Center should have highest probability
        let probs = sup.probabilities();
        let center_prob = probs[2].1;
        for (_, p) in &probs {
            assert!(*p <= center_prob + 0.01);
        }
    }

    #[test]
    fn test_probability_of() {
        let sup = superposition::uniform(vec![1, 2, 3, 4]);
        let prob_even = sup.probability_of(|v| v % 2 == 0);
        assert!((prob_even - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_superposition_scale() {
        let mut sup = superposition::Superposition::with_states(vec![("a", 1.0), ("b", 1.0)]);
        sup.scale(2.0);
        // After scaling and normalization, should still be 50/50
        let probs = sup.probabilities();
        assert!((probs[0].1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_superposition_combine() {
        let s1 = superposition::uniform(vec!["a", "b"]);
        let s2 = superposition::uniform(vec![1, 2]);
        let combined = s1.combine(&s2);
        assert_eq!(combined.state_count(), 4);
    }

    #[test]
    fn test_superposition_amplitude_sum() {
        let sup = superposition::uniform(vec![1, 2, 3]);
        let sum = sup.amplitude_sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_superposition_default() {
        let sup: superposition::Superposition<f64> = superposition::Superposition::default();
        assert!(sup.is_empty());
    }

    #[test]
    fn test_superposition_single_state() {
        let mut sup = superposition::Superposition::new();
        sup.add_state(42, 1.0);
        let probs = sup.probabilities();
        assert!((probs[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_superposition_states_access() {
        let sup = superposition::Superposition::with_states(vec![("x", 0.6), ("y", 0.8)]);
        assert_eq!(sup.states().len(), 2);
    }

    // ---- measurement tests (12) ----

    #[test]
    fn test_measure_deterministic_first() {
        let sup = superposition::uniform(vec!["a", "b"]);
        let result = measurement::measure_deterministic(&sup, 0.0);
        assert_eq!(result.value, "a");
    }

    #[test]
    fn test_measure_deterministic_last() {
        let sup = superposition::uniform(vec!["a", "b"]);
        let result = measurement::measure_deterministic(&sup, 0.99);
        assert_eq!(result.value, "b");
    }

    #[test]
    fn test_measure_entropy() {
        let sup = superposition::uniform(vec![1, 2, 3, 4]);
        let result = measurement::measure(&sup, 7.77);
        assert!([1, 2, 3, 4].contains(&result.value));
    }

    #[test]
    fn test_most_probable() {
        let sup = superposition::Superposition::with_states(vec![("a", 0.9), ("b", 0.1)]);
        let result = measurement::most_probable(&sup).unwrap();
        assert_eq!(result.value, "a");
    }

    #[test]
    fn test_least_probable() {
        let sup = superposition::Superposition::with_states(vec![("a", 0.9), ("b", 0.1)]);
        let result = measurement::least_probable(&sup).unwrap();
        assert_eq!(result.value, "b");
    }

    #[test]
    fn test_expectation() {
        let sup = superposition::uniform(vec![2.0, 4.0]);
        let exp = measurement::expectation(&sup);
        assert!((exp - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_variance() {
        let sup = superposition::uniform(vec![2.0, 4.0]);
        let var = measurement::variance(&sup);
        assert!(var > 0.0);
    }

    #[test]
    fn test_sample_distribution() {
        let sup = superposition::uniform(vec![0, 1]);
        let dist = measurement::sample_distribution(&sup, 1000);
        assert_eq!(dist.len(), 2);
        // Should be roughly equal
        let count_0 = dist.get(&0).copied().unwrap_or(0);
        assert!(count_0 > 300 && count_0 < 700);
    }

    #[test]
    fn test_measurement_result_new() {
        let mr = measurement::MeasurementResult::new(42, 0.75, 1);
        assert_eq!(mr.value, 42);
        assert!((mr.probability - 0.75).abs() < 0.01);
        assert_eq!(mr.collapsed_from, 1);
    }

    #[test]
    fn test_most_probable_empty() {
        let sup: superposition::Superposition<i32> = superposition::Superposition::new();
        assert!(measurement::most_probable(&sup).is_none());
    }

    #[test]
    fn test_expectation_single() {
        let mut sup = superposition::Superposition::new();
        sup.add_state(5.0, 1.0);
        let exp = measurement::expectation(&sup);
        assert!((exp - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_zero() {
        let mut sup = superposition::Superposition::new();
        sup.add_state(3.0, 1.0);
        let var = measurement::variance(&sup);
        assert!(var < 0.01);
    }

    // ---- entanglement tests (12) ----

    #[test]
    fn test_entangled_pair_new() {
        let pair = entanglement::EntangledPair::new("a", "b", 0.8, 0.9);
        assert_eq!(pair.id_a, "a");
        assert_eq!(pair.id_b, "b");
        assert!((pair.correlation - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_entangled_pair_clamped() {
        let pair = entanglement::EntangledPair::new("a", "b", 5.0, 2.0);
        assert!(pair.correlation <= 1.0);
        assert!(pair.strength <= 1.0);
    }

    #[test]
    fn test_correlate_decision_positive() {
        let pair = entanglement::EntangledPair::new("a", "b", 1.0, 1.0);
        let correlated = pair.correlate_decision(0.5);
        assert!((correlated - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_is_strong() {
        let pair = entanglement::EntangledPair::new("a", "b", 0.5, 0.8);
        assert!(pair.is_strong());
    }

    #[test]
    fn test_is_anticorrelated() {
        let pair = entanglement::EntangledPair::new("a", "b", -0.7, 0.5);
        assert!(pair.is_anticorrelated());
    }

    #[test]
    fn test_decay() {
        let mut pair = entanglement::EntangledPair::new("a", "b", 0.5, 1.0);
        pair.decay(0.5);
        assert!((pair.strength - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_registry_entangle() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.entangle("x", "y", 0.9, 0.8);
        assert_eq!(reg.pair_count(), 1);
    }

    #[test]
    fn test_registry_set_state() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.entangle("a", "b", 1.0, 1.0);
        reg.set_state("a", 0.5);
        assert_eq!(reg.get_state("a"), Some(0.5));
    }

    #[test]
    fn test_registry_get_pair() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.entangle("a", "b", 0.5, 0.5);
        let pair = reg.get_pair("a", "b");
        assert!(pair.is_some());
        let pair_rev = reg.get_pair("b", "a");
        assert!(pair_rev.is_some());
    }

    #[test]
    fn test_registry_pairs_for() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.entangle("a", "b", 0.5, 0.5);
        reg.entangle("a", "c", 0.3, 0.3);
        assert_eq!(reg.pairs_for("a").len(), 2);
        assert_eq!(reg.pairs_for("b").len(), 1);
    }

    #[test]
    fn test_registry_decay_all() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.entangle("a", "b", 0.5, 0.5);
        reg.entangle("c", "d", 0.5, 0.01);
        reg.decay_all(0.5);
        assert_eq!(reg.pair_count(), 1); // c-d removed
    }

    #[test]
    fn test_registry_agent_count() {
        let mut reg = entanglement::EntanglementRegistry::new();
        reg.set_state("a", 1.0);
        reg.set_state("b", 2.0);
        assert_eq!(reg.agent_count(), 2);
    }

    // ---- interference tests (12) ----

    #[test]
    fn test_interfere_constructive() {
        let result = interference::interfere(1.0, 0.0, 1.0, 0.0);
        assert!((result.amplitude - 2.0).abs() < 0.01);
        assert_eq!(result.interference_type, interference::InterferenceType::Constructive);
    }

    #[test]
    fn test_interfere_destructive() {
        let result = interference::interfere(1.0, 0.0, 1.0, std::f64::consts::PI);
        assert!(result.amplitude < 0.01);
    }

    #[test]
    fn test_multi_interfere() {
        let components = vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)];
        let result = interference::multi_interfere(&components);
        assert!((result.amplitude - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_constructive() {
        let result = interference::constructive(&[1.0, 2.0, 3.0]);
        assert!((result.amplitude - 6.0).abs() < 0.01);
        assert!(interference::is_constructive(&result));
    }

    #[test]
    fn test_destructive() {
        let result = interference::destructive(&[1.0, 2.0]);
        assert!(result.amplitude < 5.0);
    }

    #[test]
    fn test_pattern() {
        let results = interference::pattern(3, 1.0, 0.5);
        assert_eq!(results.len(), 12);
    }

    #[test]
    fn test_to_probability() {
        let result = interference::InterferenceResult::new(2.0, 2);
        let prob = interference::to_probability(&result);
        assert!((prob - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_mach_zehnder_half() {
        let (out1, _out2) = interference::mach_zehnder(1.0, 0.0);
        // At phase 0, port 1 should be constructive
        assert!(out1.amplitude > 0.5);
    }

    #[test]
    fn test_mach_zehnder_quarter() {
        let (out1, out2) = interference::mach_zehnder(1.0, std::f64::consts::PI / 2.0);
        // At quarter phase, outputs should be equal
        assert!((out1.amplitude - out2.amplitude).abs() < 0.1);
    }

    #[test]
    fn test_interference_result_new() {
        let result = interference::InterferenceResult::new(1.5, 3);
        assert_eq!(result.components, 3);
    }

    #[test]
    fn test_interfere_partial() {
        let result = interference::interfere(1.0, 0.0, 0.5, std::f64::consts::PI / 4.0);
        assert!(result.amplitude > 0.0);
        assert!(result.amplitude < 2.0);
    }

    #[test]
    fn test_multi_interfere_empty() {
        let result = interference::multi_interfere(&[]);
        assert!((result.amplitude).abs() < 0.01);
    }

    // ---- tunneling tests (12) ----

    #[test]
    fn test_barrier_new() {
        let b = tunneling::Barrier::new(1.0, 0.5, 2.0);
        assert_eq!(b.height, 1.0);
        assert_eq!(b.width, 0.5);
        assert_eq!(b.position, 2.0);
    }

    #[test]
    fn test_barrier_high_energy() {
        let b = tunneling::Barrier::new(1.0, 0.5, 0.0);
        let prob = b.transmission_probability(2.0);
        assert!((prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_barrier_low_energy() {
        let b = tunneling::Barrier::new(10.0, 5.0, 0.0);
        let prob = b.transmission_probability(0.1);
        assert!(prob < 0.01);
    }

    #[test]
    fn test_barrier_medium_energy() {
        let b = tunneling::Barrier::new(2.0, 1.0, 0.0);
        let prob = b.transmission_probability(1.0);
        assert!(prob > 0.0 && prob < 1.0);
    }

    #[test]
    fn test_barrier_impenetrable() {
        let b = tunneling::Barrier::new(10000.0, 1000.0, 0.0);
        assert!(b.is_impenetrable());
    }

    #[test]
    fn test_landscape_new() {
        let ls = tunneling::BarrierLandscape::new(0.0, 1.0);
        assert_eq!(ls.barrier_count(), 0);
    }

    #[test]
    fn test_landscape_add_barrier() {
        let mut ls = tunneling::BarrierLandscape::new(0.0, 1.0);
        ls.add_barrier(tunneling::Barrier::new(2.0, 1.0, 3.0));
        assert_eq!(ls.barrier_count(), 1);
    }

    #[test]
    fn test_landscape_nearest_barrier() {
        let mut ls = tunneling::BarrierLandscape::new(0.0, 1.0);
        ls.add_barrier(tunneling::Barrier::new(2.0, 1.0, 5.0));
        ls.add_barrier(tunneling::Barrier::new(1.0, 1.0, 3.0));
        let nearest = ls.nearest_barrier(1.0);
        assert!(nearest.is_some());
        assert!((nearest.unwrap().position - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_attempt_tunnel_no_barriers() {
        let ls = tunneling::BarrierLandscape::new(0.0, 1.0);
        let result = ls.attempt_tunnel(5.0, 1.0);
        assert!(result.reached_target);
        assert!((result.success_probability - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_attempt_tunnel_through_barrier() {
        let mut ls = tunneling::BarrierLandscape::new(0.0, 0.5);
        ls.add_barrier(tunneling::Barrier::new(2.0, 0.1, 2.0));
        let result = ls.attempt_tunnel(5.0, 1.0);
        assert!(result.barriers_crossed > 0);
    }

    #[test]
    fn test_annealing_energy() {
        let energy = tunneling::annealing_energy(1.0, 0.5);
        assert!(energy > 0.5);
    }

    #[test]
    fn test_optimize_direction() {
        let mut ls = tunneling::BarrierLandscape::new(0.0, 0.5);
        ls.add_barrier(tunneling::Barrier::new(10.0, 5.0, 2.0)); // high barrier
        ls.add_barrier(tunneling::Barrier::new(0.5, 0.1, -2.0)); // low barrier
        let (best_target, result) = tunneling::optimize_direction(&ls, &[-3.0, 3.0]);
        // Should prefer the direction with lower barrier
        assert!(result.success_probability > 0.0);
    }
}
