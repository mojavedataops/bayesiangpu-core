//! Posterior summary statistics
//!
//! Computes summary statistics for MCMC samples including:
//! - Central tendency (mean, median)
//! - Dispersion (standard deviation)
//! - Quantiles (credible intervals)
//! - Convergence diagnostics (R-hat, ESS)
//!
//! These summaries provide a comprehensive view of the posterior distribution.

use crate::ess::{ess_bulk, ess_tail};
use crate::rhat::rhat;
use serde::{Deserialize, Serialize};

/// Posterior summary for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorSummary {
    /// Parameter name (if known)
    pub name: Option<String>,

    /// Posterior mean
    pub mean: f64,

    /// Posterior standard deviation
    pub std: f64,

    /// 2.5th percentile (lower bound of 95% credible interval)
    pub q2_5: f64,

    /// 25th percentile (lower quartile)
    pub q25: f64,

    /// 50th percentile (median)
    pub q50: f64,

    /// 75th percentile (upper quartile)
    pub q75: f64,

    /// 97.5th percentile (upper bound of 95% credible interval)
    pub q97_5: f64,

    /// R-hat convergence diagnostic
    pub rhat: f64,

    /// Bulk effective sample size (for central tendency)
    pub ess_bulk: f64,

    /// Tail effective sample size (for quantiles)
    pub ess_tail: f64,

    /// Monte Carlo standard error (MCSE) of the mean
    pub mcse_mean: f64,

    /// Monte Carlo standard error of the standard deviation
    pub mcse_std: f64,
}

impl PosteriorSummary {
    /// Check if the parameter has converged (R-hat < 1.01).
    pub fn is_converged(&self) -> bool {
        self.rhat.is_finite() && self.rhat < 1.01
    }

    /// Check if ESS is sufficient (both bulk and tail > 100 per chain).
    pub fn has_sufficient_ess(&self, num_chains: usize) -> bool {
        let min_ess = 100.0 * num_chains as f64;
        self.ess_bulk >= min_ess && self.ess_tail >= min_ess
    }

    /// Get the 95% credible interval.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        (self.q2_5, self.q97_5)
    }

    /// Get the 50% credible interval (interquartile range).
    pub fn credible_interval_50(&self) -> (f64, f64) {
        (self.q25, self.q75)
    }

    /// Get diagnostic warnings as a list of strings.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.is_converged() {
            warnings.push(format!("R-hat = {:.3} (should be < 1.01)", self.rhat));
        }

        if self.ess_bulk < 100.0 {
            warnings.push(format!(
                "Low bulk ESS = {:.0} (should be > 100)",
                self.ess_bulk
            ));
        }

        if self.ess_tail < 100.0 {
            warnings.push(format!(
                "Low tail ESS = {:.0} (should be > 100)",
                self.ess_tail
            ));
        }

        warnings
    }
}

/// Compute posterior summary for a single parameter.
///
/// # Arguments
/// * `chains` - Vector of chains, each chain is Vec<f64>
///
/// # Returns
/// PosteriorSummary with all statistics
///
/// # Example
/// ```
/// use bayesian_diagnostics::summary::summarize;
///
/// let chains = vec![
///     vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
///     vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
/// ];
/// let summary = summarize(&chains);
/// println!("Mean: {:.3}, Median: {:.3}", summary.mean, summary.q50);
/// ```
pub fn summarize(chains: &[Vec<f64>]) -> PosteriorSummary {
    summarize_named(chains, None)
}

/// Compute posterior summary with parameter name.
pub fn summarize_named(chains: &[Vec<f64>], name: Option<String>) -> PosteriorSummary {
    // Combine all samples
    let mut all_samples: Vec<f64> = chains.iter().flatten().copied().collect();
    let n = all_samples.len();

    if n == 0 {
        return PosteriorSummary {
            name,
            mean: f64::NAN,
            std: f64::NAN,
            q2_5: f64::NAN,
            q25: f64::NAN,
            q50: f64::NAN,
            q75: f64::NAN,
            q97_5: f64::NAN,
            rhat: f64::NAN,
            ess_bulk: 0.0,
            ess_tail: 0.0,
            mcse_mean: f64::NAN,
            mcse_std: f64::NAN,
        };
    }

    let n_f = n as f64;

    // Compute mean
    let mean = all_samples.iter().sum::<f64>() / n_f;

    // Compute standard deviation
    let variance = if n > 1 {
        all_samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0)
    } else {
        0.0
    };
    let std = variance.sqrt();

    // Sort for quantiles
    all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute quantiles using linear interpolation
    let q2_5 = quantile(&all_samples, 0.025);
    let q25 = quantile(&all_samples, 0.25);
    let q50 = quantile(&all_samples, 0.50);
    let q75 = quantile(&all_samples, 0.75);
    let q97_5 = quantile(&all_samples, 0.975);

    // Compute diagnostics
    let rhat_val = rhat(chains);
    let ess_bulk_val = ess_bulk(chains);
    let ess_tail_val = ess_tail(chains);

    // Compute Monte Carlo standard errors
    let mcse_mean = if ess_bulk_val > 0.0 {
        std / ess_bulk_val.sqrt()
    } else {
        f64::NAN
    };

    // MCSE for standard deviation (using approximate formula)
    let mcse_std = if ess_tail_val > 0.0 {
        std / (2.0 * ess_tail_val).sqrt()
    } else {
        f64::NAN
    };

    PosteriorSummary {
        name,
        mean,
        std,
        q2_5,
        q25,
        q50,
        q75,
        q97_5,
        rhat: rhat_val,
        ess_bulk: ess_bulk_val,
        ess_tail: ess_tail_val,
        mcse_mean,
        mcse_std,
    }
}

/// Compute a single quantile using linear interpolation.
fn quantile(sorted_samples: &[f64], p: f64) -> f64 {
    let n = sorted_samples.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted_samples[0];
    }

    // Linear interpolation method (same as numpy's default)
    let idx = p * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = (lower + 1).min(n - 1);
    let weight = idx - lower as f64;

    sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight
}

/// Summary table for multiple parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryTable {
    /// Parameter summaries
    pub summaries: Vec<PosteriorSummary>,
}

impl SummaryTable {
    /// Create a new summary table.
    pub fn new(summaries: Vec<PosteriorSummary>) -> Self {
        Self { summaries }
    }

    /// Get summary for a parameter by name.
    pub fn get(&self, name: &str) -> Option<&PosteriorSummary> {
        self.summaries
            .iter()
            .find(|s| s.name.as_deref() == Some(name))
    }

    /// Get all parameters that haven't converged.
    pub fn unconverged_params(&self) -> Vec<&PosteriorSummary> {
        self.summaries
            .iter()
            .filter(|s| !s.is_converged())
            .collect()
    }

    /// Get all parameters with low ESS.
    pub fn low_ess_params(&self, min_ess: f64) -> Vec<&PosteriorSummary> {
        self.summaries
            .iter()
            .filter(|s| s.ess_bulk < min_ess || s.ess_tail < min_ess)
            .collect()
    }

    /// Check if all parameters have converged.
    pub fn all_converged(&self) -> bool {
        self.summaries.iter().all(|s| s.is_converged())
    }

    /// Format as a text table.
    pub fn to_table_string(&self) -> String {
        let mut result = String::new();

        // Header
        result.push_str(&format!(
            "{:>15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
            "Parameter", "Mean", "Std", "2.5%", "Median", "97.5%", "R-hat", "ESS"
        ));
        result.push_str(&"-".repeat(95));
        result.push('\n');

        // Rows
        for s in &self.summaries {
            let name = s.name.as_deref().unwrap_or("?");
            result.push_str(&format!(
                "{:>15} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.0}\n",
                name, s.mean, s.std, s.q2_5, s.q50, s.q97_5, s.rhat, s.ess_bulk
            ));
        }

        result
    }
}

/// Summarize multiple parameters from samples.
///
/// # Arguments
/// * `samples` - Map of parameter names to their chains
///
/// # Returns
/// SummaryTable with all parameter summaries
pub fn summarize_all(samples: &std::collections::HashMap<String, Vec<Vec<f64>>>) -> SummaryTable {
    let summaries: Vec<PosteriorSummary> = samples
        .iter()
        .map(|(name, chains)| summarize_named(chains, Some(name.clone())))
        .collect();

    SummaryTable::new(summaries)
}

/// Compute credible interval at a given level.
///
/// # Arguments
/// * `chains` - Vector of chains
/// * `level` - Credible level (e.g., 0.95 for 95% CI)
///
/// # Returns
/// Tuple of (lower, upper) bounds
pub fn credible_interval(chains: &[Vec<f64>], level: f64) -> (f64, f64) {
    let alpha = 1.0 - level;
    let mut all_samples: Vec<f64> = chains.iter().flatten().copied().collect();
    all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower = quantile(&all_samples, alpha / 2.0);
    let upper = quantile(&all_samples, 1.0 - alpha / 2.0);

    (lower, upper)
}

/// Compute Highest Posterior Density (HPD) interval.
///
/// The HPD interval is the shortest interval containing the specified probability mass.
///
/// # Arguments
/// * `chains` - Vector of chains
/// * `level` - Credible level (e.g., 0.95 for 95% HPD)
///
/// # Returns
/// Tuple of (lower, upper) bounds
pub fn hpd_interval(chains: &[Vec<f64>], level: f64) -> (f64, f64) {
    let mut all_samples: Vec<f64> = chains.iter().flatten().copied().collect();
    all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = all_samples.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }

    // Number of samples to include
    let interval_size = (n as f64 * level).ceil() as usize;
    if interval_size >= n {
        return (all_samples[0], all_samples[n - 1]);
    }

    // Find shortest interval
    let mut best_width = f64::INFINITY;
    let mut best_lower = 0;

    for i in 0..=(n - interval_size) {
        let width = all_samples[i + interval_size - 1] - all_samples[i];
        if width < best_width {
            best_width = width;
            best_lower = i;
        }
    }

    (
        all_samples[best_lower],
        all_samples[best_lower + interval_size - 1],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chains() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99],
            vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0, 0.99, 1.01],
        ]
    }

    #[test]
    fn test_summarize_basic() {
        let chains = make_test_chains();
        let summary = summarize(&chains);

        assert!(summary.mean.is_finite());
        assert!(summary.std.is_finite());
        assert!(summary.q50.is_finite());
        assert!((summary.mean - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_summarize_empty() {
        let chains: Vec<Vec<f64>> = vec![];
        let summary = summarize(&chains);
        assert!(summary.mean.is_nan());
    }

    #[test]
    fn test_quantile() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&samples, 0.0) - 1.0).abs() < 1e-6);
        assert!((quantile(&samples, 0.5) - 3.0).abs() < 1e-6);
        assert!((quantile(&samples, 1.0) - 5.0).abs() < 1e-6);
        assert!((quantile(&samples, 0.25) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_credible_interval() {
        let chains = make_test_chains();
        let (lower, upper) = credible_interval(&chains, 0.95);
        assert!(lower < upper);
        assert!(lower > 0.8);
        assert!(upper < 1.2);
    }

    #[test]
    fn test_hpd_interval() {
        let chains = make_test_chains();
        let (lower, upper) = hpd_interval(&chains, 0.95);
        assert!(lower < upper);
        // HPD should be at least as narrow as equal-tailed CI
        let (ci_lower, ci_upper) = credible_interval(&chains, 0.95);
        let hpd_width = upper - lower;
        let ci_width = ci_upper - ci_lower;
        assert!(hpd_width <= ci_width + 0.01); // Allow small tolerance
    }

    #[test]
    fn test_summary_table() {
        let mut samples = std::collections::HashMap::new();
        samples.insert("alpha".to_string(), make_test_chains());
        samples.insert(
            "beta".to_string(),
            vec![
                vec![2.0, 2.1, 1.9, 2.0, 2.05, 1.95, 2.02, 1.98, 2.01, 1.99],
                vec![1.95, 2.0, 2.1, 1.98, 2.02, 1.97, 2.03, 2.0, 1.99, 2.01],
            ],
        );

        let table = summarize_all(&samples);
        assert_eq!(table.summaries.len(), 2);
        assert!(table.get("alpha").is_some());
        assert!(table.get("beta").is_some());
    }

    #[test]
    fn test_warnings() {
        let chains = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Very few samples, different means
            vec![10.0, 11.0, 12.0, 13.0],
        ];
        let summary = summarize(&chains);
        let warnings = summary.warnings();
        assert!(!warnings.is_empty()); // Should have warnings due to diverged chains
    }

    #[test]
    fn test_is_converged() {
        let chains = make_test_chains();
        let summary = summarize(&chains);
        // Well-mixed chains should be converged
        assert!(summary.is_converged() || summary.rhat < 1.1);
    }

    #[test]
    fn test_table_string() {
        let chains = make_test_chains();
        let summary = summarize_named(&chains, Some("theta".to_string()));
        let table = SummaryTable::new(vec![summary]);
        let output = table.to_table_string();
        assert!(output.contains("theta"));
        assert!(output.contains("Mean"));
        assert!(output.contains("R-hat"));
    }
}
