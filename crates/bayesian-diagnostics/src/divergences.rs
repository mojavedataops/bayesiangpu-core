//! Divergence tracking for MCMC sampling
//!
//! Divergences occur when the Hamiltonian dynamics become numerically unstable,
//! typically due to regions of high curvature in the posterior. High divergence
//! rates indicate problems with the model or sampler configuration.
//!
//! # Interpretation
//! - 0% divergences: Ideal
//! - < 1% divergences: Generally acceptable
//! - 1-5% divergences: May need to increase target_accept or adjust model
//! - > 5% divergences: Serious issues, results may be unreliable
//!
//! # References
//! - Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.

use serde::{Deserialize, Serialize};

/// Summary of divergences from MCMC sampling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DivergenceInfo {
    /// Total number of divergent transitions
    pub total: usize,

    /// Divergences per chain
    pub per_chain: Vec<usize>,

    /// Fraction of divergent transitions (divergences / total_samples)
    pub fraction: f64,

    /// Total number of samples (including warmup if applicable)
    pub total_samples: usize,

    /// Number of chains
    pub num_chains: usize,

    /// Threshold used for divergence detection (max energy error)
    pub threshold: Option<f64>,
}

impl DivergenceInfo {
    /// Create a new DivergenceInfo from per-chain divergence counts.
    ///
    /// # Arguments
    /// * `per_chain` - Number of divergences in each chain
    /// * `samples_per_chain` - Number of samples drawn per chain
    pub fn new(per_chain: Vec<usize>, samples_per_chain: usize) -> Self {
        let total: usize = per_chain.iter().sum();
        let num_chains = per_chain.len();
        let total_samples = num_chains * samples_per_chain;
        let fraction = if total_samples > 0 {
            total as f64 / total_samples as f64
        } else {
            0.0
        };

        Self {
            total,
            per_chain,
            fraction,
            total_samples,
            num_chains,
            threshold: None,
        }
    }

    /// Create from total counts and chain count.
    pub fn from_total(total: usize, num_chains: usize, samples_per_chain: usize) -> Self {
        let per_chain = vec![total / num_chains.max(1); num_chains];
        let total_samples = num_chains * samples_per_chain;
        let fraction = if total_samples > 0 {
            total as f64 / total_samples as f64
        } else {
            0.0
        };

        Self {
            total,
            per_chain,
            fraction,
            total_samples,
            num_chains,
            threshold: None,
        }
    }

    /// Create with no divergences.
    pub fn none(num_chains: usize, samples_per_chain: usize) -> Self {
        Self::new(vec![0; num_chains], samples_per_chain)
    }

    /// Set the divergence detection threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Check if divergence rate is acceptable.
    ///
    /// Default threshold is 1% (0.01).
    pub fn is_acceptable(&self) -> bool {
        is_acceptable(self)
    }

    /// Check if divergence rate is acceptable with custom threshold.
    pub fn is_acceptable_with_threshold(&self, threshold: f64) -> bool {
        self.fraction <= threshold
    }

    /// Get diagnostic message based on divergence rate.
    pub fn diagnostic_message(&self) -> String {
        if self.total == 0 {
            "No divergences detected.".to_string()
        } else if self.fraction < 0.01 {
            format!(
                "Low divergence rate ({:.2}%). Results should be reliable.",
                self.fraction * 100.0
            )
        } else if self.fraction < 0.05 {
            format!(
                "Moderate divergence rate ({:.2}%). Consider increasing target_accept or \
                 reparameterizing the model.",
                self.fraction * 100.0
            )
        } else {
            format!(
                "High divergence rate ({:.2}%). Results may be unreliable. \
                 Consider: 1) increasing target_accept, 2) reparameterizing the model, \
                 3) using a non-centered parameterization.",
                self.fraction * 100.0
            )
        }
    }

    /// Get the chain with the most divergences.
    pub fn worst_chain(&self) -> Option<(usize, usize)> {
        self.per_chain
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(idx, &count)| (idx, count))
    }
}

/// Check if divergence rate is acceptable (< 1% is typical threshold).
///
/// # Arguments
/// * `info` - Divergence information
///
/// # Returns
/// true if divergence fraction is below 1%
pub fn is_acceptable(info: &DivergenceInfo) -> bool {
    info.fraction < 0.01
}

/// Check if divergence rate is acceptable with custom threshold.
pub fn is_acceptable_with_threshold(info: &DivergenceInfo, threshold: f64) -> bool {
    info.fraction <= threshold
}

/// Builder for tracking divergences during sampling.
#[derive(Debug, Clone)]
pub struct DivergenceTracker {
    /// Divergences per chain
    per_chain: Vec<usize>,
    /// Samples per chain
    samples_per_chain: usize,
    /// Whether we're in warmup phase
    in_warmup: bool,
    /// Warmup divergences (tracked separately)
    warmup_divergences: usize,
}

impl DivergenceTracker {
    /// Create a new divergence tracker.
    pub fn new(num_chains: usize) -> Self {
        Self {
            per_chain: vec![0; num_chains],
            samples_per_chain: 0,
            in_warmup: true,
            warmup_divergences: 0,
        }
    }

    /// Mark the end of warmup phase.
    pub fn end_warmup(&mut self) {
        self.in_warmup = false;
    }

    /// Record a divergence in a specific chain.
    pub fn record_divergence(&mut self, chain: usize) {
        if self.in_warmup {
            self.warmup_divergences += 1;
        } else if chain < self.per_chain.len() {
            self.per_chain[chain] += 1;
        }
    }

    /// Record a sample (to track total samples per chain).
    pub fn record_sample(&mut self) {
        if !self.in_warmup {
            self.samples_per_chain += 1;
        }
    }

    /// Record multiple divergences at once (useful for batch operations).
    pub fn record_divergences(&mut self, chain: usize, count: usize) {
        if self.in_warmup {
            self.warmup_divergences += count;
        } else if chain < self.per_chain.len() {
            self.per_chain[chain] += count;
        }
    }

    /// Get the divergence info summary.
    pub fn info(&self) -> DivergenceInfo {
        DivergenceInfo::new(self.per_chain.clone(), self.samples_per_chain)
    }

    /// Get warmup divergences (for diagnostics).
    pub fn warmup_divergences(&self) -> usize {
        self.warmup_divergences
    }

    /// Get total divergences (including warmup).
    pub fn total_divergences(&self) -> usize {
        self.per_chain.iter().sum::<usize>() + self.warmup_divergences
    }
}

/// Analyze divergence patterns across chains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceAnalysis {
    /// Basic divergence info
    pub info: DivergenceInfo,

    /// Whether divergences are clustered in specific chains
    pub chain_clustering: bool,

    /// Coefficient of variation of divergences across chains
    pub chain_cv: f64,

    /// Recommendation based on analysis
    pub recommendation: String,
}

/// Analyze divergence patterns for additional diagnostics.
pub fn analyze_divergences(info: &DivergenceInfo) -> DivergenceAnalysis {
    let num_chains = info.per_chain.len();

    // Compute coefficient of variation across chains
    let chain_cv = if num_chains > 1 && info.total > 0 {
        let mean = info.total as f64 / num_chains as f64;
        let variance: f64 = info
            .per_chain
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / (num_chains - 1) as f64;
        if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Check for clustering (high CV indicates uneven distribution)
    let chain_clustering = chain_cv > 1.0;

    let recommendation = if info.total == 0 {
        "Sampling completed without divergences.".to_string()
    } else if info.fraction < 0.01 && !chain_clustering {
        "Low divergence rate; results are likely reliable.".to_string()
    } else if chain_clustering {
        format!(
            "Divergences are clustered in specific chains (CV={:.2}). \
             This may indicate initialization issues or multimodal posterior.",
            chain_cv
        )
    } else if info.fraction < 0.05 {
        "Moderate divergence rate. Try increasing adapt_delta (target_accept).".to_string()
    } else {
        "High divergence rate. Consider reparameterization or non-centered parameterization."
            .to_string()
    };

    DivergenceAnalysis {
        info: info.clone(),
        chain_clustering,
        chain_cv,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_info_new() {
        let info = DivergenceInfo::new(vec![5, 3, 2, 4], 100);
        assert_eq!(info.total, 14);
        assert_eq!(info.num_chains, 4);
        assert_eq!(info.total_samples, 400);
        assert!((info.fraction - 0.035).abs() < 1e-6);
    }

    #[test]
    fn test_divergence_info_none() {
        let info = DivergenceInfo::none(4, 100);
        assert_eq!(info.total, 0);
        assert_eq!(info.fraction, 0.0);
        assert!(info.is_acceptable());
    }

    #[test]
    fn test_is_acceptable() {
        let acceptable = DivergenceInfo::new(vec![0, 1, 0, 0], 100);
        assert!(is_acceptable(&acceptable));

        let not_acceptable = DivergenceInfo::new(vec![5, 3, 4, 6], 100);
        assert!(!is_acceptable(&not_acceptable));
    }

    #[test]
    fn test_divergence_tracker() {
        let mut tracker = DivergenceTracker::new(4);

        // Record warmup divergences
        tracker.record_divergence(0);
        tracker.record_divergence(1);
        assert_eq!(tracker.warmup_divergences(), 2);

        // End warmup
        tracker.end_warmup();

        // Record sampling divergences
        for _ in 0..100 {
            tracker.record_sample();
        }
        tracker.record_divergence(0);
        tracker.record_divergence(2);

        let info = tracker.info();
        assert_eq!(info.total, 2);
        assert_eq!(info.total_samples, 400); // 4 chains * 100 samples
    }

    #[test]
    fn test_worst_chain() {
        let info = DivergenceInfo::new(vec![1, 5, 2, 3], 100);
        let (chain, count) = info.worst_chain().unwrap();
        assert_eq!(chain, 1);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_diagnostic_message() {
        let none = DivergenceInfo::none(4, 100);
        assert!(none.diagnostic_message().contains("No divergences"));

        let low = DivergenceInfo::new(vec![1, 1, 0, 0], 500);
        assert!(low.diagnostic_message().contains("Low"));

        let high = DivergenceInfo::new(vec![10, 10, 10, 10], 100);
        assert!(high.diagnostic_message().contains("High"));
    }

    #[test]
    fn test_analyze_divergences() {
        let info = DivergenceInfo::new(vec![0, 10, 0, 0], 100);
        let analysis = analyze_divergences(&info);
        assert!(analysis.chain_clustering);
        assert!(analysis.recommendation.contains("clustered"));
    }

    #[test]
    fn test_analyze_divergences_uniform() {
        let info = DivergenceInfo::new(vec![2, 2, 2, 2], 100);
        let analysis = analyze_divergences(&info);
        assert!(!analysis.chain_clustering);
        assert!(analysis.chain_cv < 0.1);
    }

    #[test]
    fn test_from_total() {
        let info = DivergenceInfo::from_total(12, 4, 100);
        assert_eq!(info.total, 12);
        assert_eq!(info.num_chains, 4);
        assert_eq!(info.per_chain, vec![3, 3, 3, 3]);
    }
}
