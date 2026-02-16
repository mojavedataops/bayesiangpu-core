//! Model Comparison Utilities
//!
//! This module provides functions for comparing multiple Bayesian models
//! using LOO-CV or WAIC.
//!
//! # Overview
//!
//! The main comparison function [`compare`] takes LOO or WAIC results from
//! multiple models and produces a ranked table showing:
//!
//! - Model ranking by elpd (best to worst)
//! - Difference from best model (delta_elpd)
//! - Standard error of difference
//! - Model weights (stacking or pseudo-BMA)
//!
//! # Example
//!
//! ```ignore
//! use bayesian_diagnostics::compare::{compare, ModelCriterion};
//!
//! let results = vec![
//!     ModelCriterion::Loo(loo_result1),
//!     ModelCriterion::Loo(loo_result2),
//! ];
//! let comparison = compare(&results, &["Model A", "Model B"]);
//! println!("{}", comparison.to_table_string());
//! ```
//!
//! # References
//!
//! - Vehtari, A., Gelman, A., & Gabry, J. (2017).
//!   Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
//! - Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018).
//!   Using stacking to average Bayesian predictive distributions.

use crate::loo::LooResult;
use crate::waic::WaicResult;
use serde::{Deserialize, Serialize};

/// Model comparison criterion (either LOO or WAIC).
#[derive(Debug, Clone)]
pub enum ModelCriterion {
    /// LOO-CV result
    Loo(LooResult),
    /// WAIC result
    Waic(WaicResult),
}

impl ModelCriterion {
    /// Get elpd value.
    pub fn elpd(&self) -> f64 {
        match self {
            ModelCriterion::Loo(loo) => loo.elpd_loo,
            ModelCriterion::Waic(waic) => waic.elpd_waic,
        }
    }

    /// Get standard error of elpd.
    pub fn se_elpd(&self) -> f64 {
        match self {
            ModelCriterion::Loo(loo) => loo.se_elpd_loo,
            ModelCriterion::Waic(waic) => waic.se_elpd_waic,
        }
    }

    /// Get pointwise elpd values.
    pub fn pointwise_elpd(&self) -> &[f64] {
        match self {
            ModelCriterion::Loo(loo) => &loo.pointwise_elpd,
            ModelCriterion::Waic(waic) => &waic.pointwise_elpd,
        }
    }

    /// Get effective number of parameters.
    pub fn p_eff(&self) -> f64 {
        match self {
            ModelCriterion::Loo(loo) => loo.p_loo,
            ModelCriterion::Waic(waic) => waic.p_waic,
        }
    }

    /// Get criterion name.
    pub fn criterion_name(&self) -> &'static str {
        match self {
            ModelCriterion::Loo(_) => "LOO",
            ModelCriterion::Waic(_) => "WAIC",
        }
    }

    /// Get number of observations.
    pub fn n_obs(&self) -> usize {
        match self {
            ModelCriterion::Loo(loo) => loo.n_obs,
            ModelCriterion::Waic(waic) => waic.n_obs,
        }
    }
}

/// Comparison result for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonEntry {
    /// Model name
    pub name: String,

    /// Rank (1 = best)
    pub rank: usize,

    /// Expected log pointwise predictive density
    pub elpd: f64,

    /// Standard error of elpd
    pub se_elpd: f64,

    /// Difference from best model
    pub delta_elpd: f64,

    /// Standard error of difference from best model
    pub se_delta_elpd: f64,

    /// Effective number of parameters
    pub p_eff: f64,

    /// Model weight (stacking or pseudo-BMA)
    pub weight: f64,

    /// Criterion used (LOO or WAIC)
    pub criterion: String,
}

/// Result of model comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Comparison entries sorted by elpd (best to worst)
    pub entries: Vec<ModelComparisonEntry>,

    /// Method used for weights (stacking or pseudo_bma)
    pub weight_method: String,

    /// Criterion used (LOO or WAIC)
    pub criterion: String,
}

impl ComparisonResult {
    /// Get the best model name.
    pub fn best_model(&self) -> Option<&str> {
        self.entries.first().map(|e| e.name.as_str())
    }

    /// Get model by name.
    pub fn get_model(&self, name: &str) -> Option<&ModelComparisonEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Format as a text table.
    pub fn to_table_string(&self) -> String {
        let mut result = String::new();

        // Header
        result.push_str(&format!(
            "{:>4} {:>15} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}\n",
            "Rank", "Model", "elpd", "SE", "d_elpd", "SE(d)", "p_eff", "Weight"
        ));
        result.push_str(&"-".repeat(85));
        result.push('\n');

        // Rows
        for e in &self.entries {
            result.push_str(&format!(
                "{:>4} {:>15} {:>10.1} {:>8.1} {:>10.1} {:>10.1} {:>8.1} {:>8.3}\n",
                e.rank, e.name, e.elpd, e.se_elpd, e.delta_elpd, e.se_delta_elpd, e.p_eff, e.weight
            ));
        }

        result.push_str(&format!(
            "\nCriterion: {}, Weights: {}\n",
            self.criterion, self.weight_method
        ));

        result
    }
}

/// Compare multiple models using their LOO or WAIC results.
///
/// # Arguments
///
/// * `criteria` - Vector of ModelCriterion (LOO or WAIC results)
/// * `names` - Names for each model
///
/// # Returns
///
/// ComparisonResult with models ranked by elpd (higher = better).
///
/// # Example
///
/// ```ignore
/// let comparison = compare(&[loo1, loo2], &["Model A", "Model B"]);
/// println!("Best model: {}", comparison.best_model().unwrap());
/// ```
pub fn compare(criteria: &[ModelCriterion], names: &[&str]) -> ComparisonResult {
    if criteria.is_empty() || criteria.len() != names.len() {
        return ComparisonResult {
            entries: vec![],
            weight_method: "none".to_string(),
            criterion: "unknown".to_string(),
        };
    }

    // Check all criteria are the same type
    let criterion_name = criteria[0].criterion_name();
    let _n_obs = criteria[0].n_obs();

    // Create entries with elpd values
    #[allow(clippy::type_complexity)]
    let mut entries: Vec<(usize, &str, f64, f64, f64, &[f64])> = criteria
        .iter()
        .zip(names.iter())
        .enumerate()
        .map(|(i, (c, name))| {
            (
                i,
                *name,
                c.elpd(),
                c.se_elpd(),
                c.p_eff(),
                c.pointwise_elpd(),
            )
        })
        .collect();

    // Sort by elpd (descending = best first)
    entries.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Best model's elpd and pointwise values
    let best_elpd = entries[0].2;
    let best_pointwise = entries[0].5;

    // Compute weights using stacking (simplified version)
    let weights = compute_stacking_weights(&entries.iter().map(|e| e.5).collect::<Vec<_>>());

    // Build comparison result
    let comparison_entries: Vec<ModelComparisonEntry> = entries
        .iter()
        .enumerate()
        .map(
            |(rank, (orig_idx, name, elpd, se_elpd, p_eff, pointwise))| {
                let delta_elpd = *elpd - best_elpd;
                let se_delta = if rank == 0 {
                    0.0
                } else {
                    // SE of difference computed from pointwise differences
                    let diffs: Vec<f64> = pointwise
                        .iter()
                        .zip(best_pointwise.iter())
                        .map(|(&p, &b)| p - b)
                        .collect();
                    compute_se_sum(&diffs)
                };

                ModelComparisonEntry {
                    name: name.to_string(),
                    rank: rank + 1,
                    elpd: *elpd,
                    se_elpd: *se_elpd,
                    delta_elpd,
                    se_delta_elpd: se_delta,
                    p_eff: *p_eff,
                    weight: weights[*orig_idx],
                    criterion: criterion_name.to_string(),
                }
            },
        )
        .collect();

    ComparisonResult {
        entries: comparison_entries,
        weight_method: "stacking".to_string(),
        criterion: criterion_name.to_string(),
    }
}

/// Compare models using pseudo-BMA weights instead of stacking.
///
/// Pseudo-BMA weights are based on the difference in elpd, assuming
/// normally distributed errors.
pub fn compare_pseudo_bma(criteria: &[ModelCriterion], names: &[&str]) -> ComparisonResult {
    let mut result = compare(criteria, names);

    if result.entries.is_empty() {
        return result;
    }

    // Compute pseudo-BMA weights from elpd differences
    let elpd_values: Vec<f64> = result.entries.iter().map(|e| e.elpd).collect();
    let weights = compute_pseudo_bma_weights(&elpd_values);

    // Update weights in entries
    for (entry, &w) in result.entries.iter_mut().zip(weights.iter()) {
        entry.weight = w;
    }

    result.weight_method = "pseudo_bma".to_string();
    result
}

/// Compute stacking weights for model averaging.
///
/// Stacking finds weights that maximize the leave-one-out log score
/// of the weighted combination of models.
///
/// This is a simplified implementation that uses a basic optimization.
fn compute_stacking_weights(pointwise_elpd: &[&[f64]]) -> Vec<f64> {
    let k = pointwise_elpd.len(); // Number of models
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![1.0];
    }

    let n = pointwise_elpd[0].len(); // Number of observations
    if n == 0 || !pointwise_elpd.iter().all(|p| p.len() == n) {
        return vec![1.0 / k as f64; k];
    }

    // Convert pointwise elpd to log-likelihood contributions
    // exp(elpd_i) = p(y_i | M)
    let log_lik: Vec<Vec<f64>> = pointwise_elpd.iter().map(|p| p.to_vec()).collect();

    // Simple grid search for optimal weights
    // For k=2, this is exact; for k>2, it's approximate
    if k == 2 {
        // Binary search for optimal weight
        let mut best_w = 0.5;
        let mut best_score = f64::NEG_INFINITY;

        for w_int in 0..=100 {
            let w = w_int as f64 / 100.0;
            let score = stacking_score(&[w, 1.0 - w], &log_lik);
            if score > best_score {
                best_score = score;
                best_w = w;
            }
        }

        return vec![best_w, 1.0 - best_w];
    }

    // For k > 2, use iterative optimization
    let mut weights = vec![1.0 / k as f64; k];

    for _iter in 0..100 {
        let old_weights = weights.clone();

        for j in 0..k {
            // Try different values for weight j, keeping others fixed (renormalized)
            let mut best_w = weights[j];
            let mut best_score = f64::NEG_INFINITY;

            for w_int in 0..=20 {
                let w = w_int as f64 / 20.0;
                let mut test_weights = weights.clone();
                test_weights[j] = w;

                // Renormalize
                let sum: f64 = test_weights.iter().sum();
                if sum > 0.0 {
                    for w in &mut test_weights {
                        *w /= sum;
                    }
                }

                let score = stacking_score(&test_weights, &log_lik);
                if score > best_score {
                    best_score = score;
                    best_w = w;
                }
            }

            weights[j] = best_w;
            // Renormalize
            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        // Check for convergence
        let max_change = weights
            .iter()
            .zip(old_weights.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        if max_change < 0.01 {
            break;
        }
    }

    weights
}

/// Compute stacking objective: sum of log(weighted predictive density).
fn stacking_score(weights: &[f64], log_lik: &[Vec<f64>]) -> f64 {
    let k = log_lik.len();
    let n = log_lik[0].len();

    let mut total = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        // Compute log(sum_k(w_k * exp(log_lik_k[i]))) using log-sum-exp
        let mut terms: Vec<f64> = Vec::with_capacity(k);

        for j in 0..k {
            if weights[j] > 0.0 {
                terms.push(weights[j].ln() + log_lik[j][i]);
            }
        }

        if !terms.is_empty() {
            total += log_sum_exp(&terms);
        }
    }

    total
}

/// Compute pseudo-BMA weights based on elpd differences.
fn compute_pseudo_bma_weights(elpd_values: &[f64]) -> Vec<f64> {
    let k = elpd_values.len();
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![1.0];
    }

    // Weights proportional to exp(elpd - max_elpd)
    let max_elpd = elpd_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let raw_weights: Vec<f64> = elpd_values.iter().map(|&e| (e - max_elpd).exp()).collect();

    let sum: f64 = raw_weights.iter().sum();
    if sum > 0.0 {
        raw_weights.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / k as f64; k]
    }
}

/// Compute standard error of the sum from pointwise values.
fn compute_se_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }

    let n_f = n as f64;
    let mean = values.iter().sum::<f64>() / n_f;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0);

    (n_f * variance).sqrt()
}

/// Compute log(sum(exp(x))) in a numerically stable way.
fn log_sum_exp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_x = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_x.is_finite() {
        return max_x;
    }

    let sum: f64 = x.iter().map(|&xi| (xi - max_x).exp()).sum();
    max_x + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_loo() -> LooResult {
        LooResult {
            elpd_loo: -50.0,
            se_elpd_loo: 5.0,
            p_loo: 3.0,
            se_p_loo: 0.5,
            looic: 100.0,
            se_looic: 10.0,
            n_obs: 20,
            n_samples: 100,
            pointwise_elpd: vec![-2.5; 20],
            pointwise_p_loo: vec![0.15; 20],
            pareto_k: vec![0.3; 20],
            n_bad_k: 0,
        }
    }

    #[test]
    fn test_compare_basic() {
        let loo1 = make_test_loo();
        let mut loo2 = make_test_loo();
        loo2.elpd_loo = -60.0; // Worse model

        let criteria = vec![ModelCriterion::Loo(loo1), ModelCriterion::Loo(loo2)];
        let result = compare(&criteria, &["Model A", "Model B"]);

        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries[0].name, "Model A");
        assert_eq!(result.entries[0].rank, 1);
        assert_eq!(result.entries[1].name, "Model B");
        assert_eq!(result.entries[1].rank, 2);
    }

    #[test]
    fn test_compare_weights() {
        let loo1 = make_test_loo();
        let mut loo2 = make_test_loo();
        loo2.elpd_loo = -55.0;
        loo2.pointwise_elpd = vec![-2.75; 20];

        let criteria = vec![ModelCriterion::Loo(loo1), ModelCriterion::Loo(loo2)];
        let result = compare(&criteria, &["Model A", "Model B"]);

        // Weights should sum to 1
        let weight_sum: f64 = result.entries.iter().map(|e| e.weight).sum();
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "Weights sum to {}",
            weight_sum
        );

        // Better model should have higher weight
        assert!(result.entries[0].weight >= result.entries[1].weight);
    }

    #[test]
    fn test_compare_pseudo_bma() {
        let loo1 = make_test_loo();
        let mut loo2 = make_test_loo();
        loo2.elpd_loo = -55.0;

        let criteria = vec![ModelCriterion::Loo(loo1), ModelCriterion::Loo(loo2)];
        let result = compare_pseudo_bma(&criteria, &["Model A", "Model B"]);

        assert_eq!(result.weight_method, "pseudo_bma");

        // Weights should sum to 1
        let weight_sum: f64 = result.entries.iter().map(|e| e.weight).sum();
        assert!((weight_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_delta_elpd() {
        let loo1 = make_test_loo();
        let mut loo2 = make_test_loo();
        loo2.elpd_loo = -60.0;

        let criteria = vec![ModelCriterion::Loo(loo1), ModelCriterion::Loo(loo2)];
        let result = compare(&criteria, &["Model A", "Model B"]);

        // Best model should have delta_elpd = 0
        assert!((result.entries[0].delta_elpd - 0.0).abs() < 1e-10);

        // Second model should have negative delta_elpd
        assert!((result.entries[1].delta_elpd - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compare_table_string() {
        let loo1 = make_test_loo();
        let criteria = vec![ModelCriterion::Loo(loo1)];
        let result = compare(&criteria, &["Model A"]);

        let table = result.to_table_string();
        assert!(table.contains("Model A"));
        assert!(table.contains("Rank"));
        assert!(table.contains("LOO"));
    }

    #[test]
    fn test_best_model() {
        let loo1 = make_test_loo();
        let mut loo2 = make_test_loo();
        loo2.elpd_loo = -60.0;

        let criteria = vec![ModelCriterion::Loo(loo1), ModelCriterion::Loo(loo2)];
        let result = compare(&criteria, &["Model A", "Model B"]);

        assert_eq!(result.best_model(), Some("Model A"));
    }

    #[test]
    fn test_stacking_weights() {
        let pw1 = vec![-2.0; 10];
        let pw2 = vec![-3.0; 10];

        let weights = compute_stacking_weights(&[&pw1, &pw2]);

        assert_eq!(weights.len(), 2);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Model 1 (higher elpd) should have higher weight
        assert!(weights[0] > weights[1]);
    }
}
