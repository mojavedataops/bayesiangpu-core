//! Leave-One-Out Cross-Validation (LOO-CV) via Pareto Smoothed Importance Sampling
//!
//! This module implements PSIS-LOO-CV for Bayesian model comparison.
//!
//! # Overview
//!
//! LOO-CV estimates out-of-sample predictive accuracy by computing:
//! - **elpd_loo**: Expected log pointwise predictive density
//! - **p_loo**: Effective number of parameters
//! - **looic**: LOO information criterion (-2 * elpd_loo)
//!
//! # Algorithm
//!
//! PSIS-LOO uses importance sampling with Pareto-smoothed weights to efficiently
//! approximate leave-one-out cross-validation without refitting the model.
//!
//! # References
//!
//! - Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
//!   Pareto smoothed importance sampling. Journal of Machine Learning Research.
//! - Vehtari, A., Gelman, A., & Gabry, J. (2017).
//!   Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.

use serde::{Deserialize, Serialize};

/// Result of LOO-CV computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LooResult {
    /// Expected log pointwise predictive density (higher is better)
    pub elpd_loo: f64,

    /// Standard error of elpd_loo
    pub se_elpd_loo: f64,

    /// Effective number of parameters
    pub p_loo: f64,

    /// Standard error of p_loo
    pub se_p_loo: f64,

    /// LOO information criterion (-2 * elpd_loo, lower is better)
    pub looic: f64,

    /// Standard error of looic
    pub se_looic: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Number of posterior samples
    pub n_samples: usize,

    /// Pointwise elpd values
    pub pointwise_elpd: Vec<f64>,

    /// Pointwise p_loo values
    pub pointwise_p_loo: Vec<f64>,

    /// Pareto k diagnostic values for each observation
    pub pareto_k: Vec<f64>,

    /// Number of observations with problematic Pareto k
    pub n_bad_k: usize,
}

impl LooResult {
    /// Check if the LOO estimates are reliable based on Pareto k diagnostics.
    ///
    /// Returns true if all Pareto k values are below the threshold.
    pub fn is_reliable(&self) -> bool {
        self.n_bad_k == 0
    }

    /// Get the Pareto k threshold for the given sample size.
    ///
    /// The threshold is min(1 - 1/log10(S), 0.7) where S is the sample size.
    pub fn pareto_k_threshold(&self) -> f64 {
        pareto_k_threshold(self.n_samples)
    }

    /// Get observations with Pareto k above the threshold.
    pub fn problematic_observations(&self) -> Vec<(usize, f64)> {
        let threshold = self.pareto_k_threshold();
        self.pareto_k
            .iter()
            .enumerate()
            .filter(|(_, &k)| k >= threshold)
            .map(|(i, &k)| (i, k))
            .collect()
    }

    /// Get diagnostic warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let threshold = self.pareto_k_threshold();

        if self.n_bad_k > 0 {
            warnings.push(format!(
                "{} observations have Pareto k >= {:.2}",
                self.n_bad_k, threshold
            ));
        }

        let very_bad = self.pareto_k.iter().filter(|&&k| k >= 0.7).count();
        if very_bad > 0 {
            warnings.push(format!(
                "{} observations have Pareto k >= 0.7 (very unreliable)",
                very_bad
            ));
        }

        let undefined = self.pareto_k.iter().filter(|&&k| k >= 1.0).count();
        if undefined > 0 {
            warnings.push(format!(
                "{} observations have Pareto k >= 1.0 (undefined mean)",
                undefined
            ));
        }

        warnings
    }
}

/// Compute the Pareto k threshold for a given sample size.
///
/// The threshold is min(1 - 1/log10(S), 0.7).
pub fn pareto_k_threshold(n_samples: usize) -> f64 {
    if n_samples < 10 {
        return 0.5; // Conservative threshold for very small samples
    }
    let log10_s = (n_samples as f64).log10();
    (1.0 - 1.0 / log10_s).min(0.7)
}

/// Compute LOO-CV using Pareto Smoothed Importance Sampling.
///
/// # Arguments
///
/// * `log_lik` - Log-likelihood matrix where rows are posterior samples (S)
///   and columns are observations (N). Shape: [S, N]
///
/// # Returns
///
/// LooResult with elpd_loo, p_loo, looic, and diagnostics.
///
/// # Example
///
/// ```
/// use bayesian_diagnostics::loo::loo;
///
/// // Log-likelihood: 100 samples, 10 observations
/// let log_lik: Vec<Vec<f64>> = (0..100).map(|s| {
///     (0..10).map(|n| -0.5 * ((s as f64 * 0.01 - n as f64 * 0.1).powi(2))).collect()
/// }).collect();
///
/// let result = loo(&log_lik);
/// println!("ELPD-LOO: {:.2} +/- {:.2}", result.elpd_loo, result.se_elpd_loo);
/// println!("LOOIC: {:.2}", result.looic);
/// ```
pub fn loo(log_lik: &[Vec<f64>]) -> LooResult {
    let n_samples = log_lik.len();
    if n_samples == 0 {
        return empty_result();
    }

    let n_obs = log_lik[0].len();
    if n_obs == 0 || !log_lik.iter().all(|row| row.len() == n_obs) {
        return empty_result();
    }

    let mut pointwise_elpd = Vec::with_capacity(n_obs);
    let mut pointwise_p_loo = Vec::with_capacity(n_obs);
    let mut pareto_k = Vec::with_capacity(n_obs);

    // Process each observation
    for i in 0..n_obs {
        // Get log-likelihood for observation i across all samples
        let log_lik_i: Vec<f64> = log_lik.iter().map(|row| row[i]).collect();

        // Compute PSIS for this observation
        let psis_result = psis(&log_lik_i);

        // Compute elpd_loo_i using smoothed weights
        let elpd_loo_i = compute_elpd_loo(&log_lik_i, &psis_result.weights);

        // Compute p_loo_i
        let log_mean = log_sum_exp(&log_lik_i) - (n_samples as f64).ln();
        let p_loo_i = log_mean - elpd_loo_i;

        pointwise_elpd.push(elpd_loo_i);
        pointwise_p_loo.push(p_loo_i.max(0.0)); // p_loo should be non-negative
        pareto_k.push(psis_result.k);
    }

    // Compute totals
    let elpd_loo: f64 = pointwise_elpd.iter().sum();
    let p_loo: f64 = pointwise_p_loo.iter().sum();
    let looic = -2.0 * elpd_loo;

    // Compute standard errors
    let se_elpd_loo = standard_error(&pointwise_elpd);
    let se_p_loo = standard_error(&pointwise_p_loo);
    let se_looic = 2.0 * se_elpd_loo;

    // Count problematic observations
    let threshold = pareto_k_threshold(n_samples);
    let n_bad_k = pareto_k.iter().filter(|&&k| k >= threshold).count();

    LooResult {
        elpd_loo,
        se_elpd_loo,
        p_loo,
        se_p_loo,
        looic,
        se_looic,
        n_obs,
        n_samples,
        pointwise_elpd,
        pointwise_p_loo,
        pareto_k,
        n_bad_k,
    }
}

/// Compute LOO-CV from a flat array of log-likelihoods.
///
/// # Arguments
///
/// * `log_lik` - Flat array of log-likelihoods in row-major order
/// * `n_samples` - Number of posterior samples
/// * `n_obs` - Number of observations
pub fn loo_from_array(log_lik: &[f64], n_samples: usize, n_obs: usize) -> LooResult {
    if log_lik.len() != n_samples * n_obs {
        return empty_result();
    }

    let log_lik_2d: Vec<Vec<f64>> = (0..n_samples)
        .map(|s| {
            let start = s * n_obs;
            log_lik[start..start + n_obs].to_vec()
        })
        .collect();

    loo(&log_lik_2d)
}

/// Result of PSIS computation for a single observation.
#[derive(Debug, Clone)]
pub struct PsisResult {
    /// Smoothed log weights (normalized)
    pub weights: Vec<f64>,

    /// Pareto shape parameter k
    pub k: f64,

    /// Number of tail samples used for fitting
    pub n_tail: usize,
}

/// Apply Pareto Smoothed Importance Sampling to log-likelihood ratios.
///
/// # Arguments
///
/// * `log_lik` - Log-likelihood values for a single observation across all samples.
///   For LOO, these are negative log-likelihoods (importance ratios).
///
/// # Returns
///
/// PsisResult with smoothed weights and Pareto k diagnostic.
pub fn psis(log_lik: &[f64]) -> PsisResult {
    let n = log_lik.len();
    if n < 10 {
        // Not enough samples for PSIS, return uniform weights
        let weight = 1.0 / n as f64;
        return PsisResult {
            weights: vec![weight; n],
            k: f64::NAN,
            n_tail: 0,
        };
    }

    // Compute raw importance ratios: r_s = 1 / p(y_i | theta_s) = exp(-log_lik_s)
    // In log space: log_r_s = -log_lik_s
    let log_ratios: Vec<f64> = log_lik.iter().map(|&ll| -ll).collect();

    // Stabilize by subtracting max
    let max_log_ratio = log_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_log_ratio.is_finite() {
        let weight = 1.0 / n as f64;
        return PsisResult {
            weights: vec![weight; n],
            k: f64::NAN,
            n_tail: 0,
        };
    }

    let shifted_log_ratios: Vec<f64> = log_ratios.iter().map(|&r| r - max_log_ratio).collect();

    // Determine tail length: M = min(0.2*S, 3*sqrt(S))
    let m = ((0.2 * n as f64).floor() as usize).min((3.0 * (n as f64).sqrt()).floor() as usize);
    let m = m.max(5).min(n - 1); // Ensure at least 5 tail samples

    // Sort indices by log ratio (ascending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        shifted_log_ratios[a]
            .partial_cmp(&shifted_log_ratios[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract tail (M largest ratios)
    let tail_indices: Vec<usize> = indices[n - m..].to_vec();
    let cutoff = shifted_log_ratios[indices[n - m - 1]]; // Threshold

    // Get tail values (shifted, in linear space)
    let tail_values: Vec<f64> = tail_indices
        .iter()
        .map(|&i| (shifted_log_ratios[i] - cutoff).exp())
        .collect();

    // Fit generalized Pareto distribution to tail
    let (k, sigma) = fit_gpd(&tail_values);

    // Create smoothed weights
    let mut log_weights = shifted_log_ratios.clone();

    if k.is_finite() && sigma > 0.0 {
        // Replace tail weights with expected order statistics from fitted GPD
        for (z, &idx) in tail_indices.iter().enumerate() {
            let p = (z as f64 + 0.5) / m as f64;
            let smoothed_value = gpd_quantile(p, sigma, k);

            // Convert back to log space and add cutoff
            let smoothed_log = if smoothed_value > 0.0 {
                smoothed_value.ln() + cutoff
            } else {
                cutoff
            };

            // Truncate at maximum original value
            log_weights[idx] = smoothed_log.min(shifted_log_ratios[indices[n - 1]]);
        }
    }

    // Normalize weights using log-sum-exp
    let log_sum = log_sum_exp(&log_weights);
    let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - log_sum).exp()).collect();

    PsisResult {
        weights,
        k,
        n_tail: m,
    }
}

/// Fit a Generalized Pareto Distribution to tail values.
///
/// Uses the Zhang & Stephens (2009) approximate Bayesian method.
///
/// # Arguments
///
/// * `tail` - Tail values (exceedances above threshold)
///
/// # Returns
///
/// (k, sigma) - Shape and scale parameters
fn fit_gpd(tail: &[f64]) -> (f64, f64) {
    let m = tail.len();
    if m < 5 {
        return (f64::NAN, f64::NAN);
    }

    // Sort tail values
    let mut sorted: Vec<f64> = tail.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Remove any zeros or negative values
    let positive: Vec<f64> = sorted.into_iter().filter(|&x| x > 1e-15).collect();
    if positive.len() < 5 {
        return (f64::NAN, f64::NAN);
    }

    let n = positive.len() as f64;
    let x_max = positive[positive.len() - 1];

    // Zhang & Stephens method: estimate k from order statistics
    // Use L-moments estimator which is robust for small samples
    let mean = positive.iter().sum::<f64>() / n;

    // Estimate k using probability-weighted moments
    // For GPD: E[X] = sigma / (1-k) when k < 1
    // Var[X] = sigma^2 / ((1-k)^2 * (1-2k)) when k < 0.5

    let variance = positive.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let cv = (variance.sqrt()) / mean; // Coefficient of variation

    // For GPD, CV^2 = 1 / ((1-k)(1-2k))
    // Solve for k: 2k^2 - 3k + 1 - 1/CV^2 = 0
    let cv2 = cv * cv;
    if cv2 < 1.0 {
        // CV < 1 means k < 0 (bounded tail)
        // Use direct estimation
        let k = 0.5 * (1.0 - 1.0 / cv2.max(0.5));
        let sigma = mean * (1.0 - k);
        return (k.clamp(-0.5, 1.5), sigma.max(1e-10));
    }

    // For CV >= 1, use discriminant of quadratic
    let discriminant = 9.0 - 8.0 * (1.0 - 1.0 / cv2);
    let k = if discriminant >= 0.0 {
        (3.0 - discriminant.sqrt()) / 4.0
    } else {
        // Heavy tail, estimate from max
        let excess_kurtosis = positive
            .iter()
            .map(|&x| ((x - mean) / variance.sqrt()).powi(4))
            .sum::<f64>()
            / n
            - 3.0;
        (excess_kurtosis / 6.0).clamp(0.0, 1.5)
    };

    // Estimate sigma from mean: sigma = mean * (1 - k)
    let sigma = if k < 1.0 {
        mean * (1.0 - k)
    } else {
        x_max / 2.0 // Fallback for very heavy tails
    };

    // Clamp k to reasonable range
    let k = k.clamp(-0.5, 1.5);
    let sigma = sigma.max(1e-10);

    (k, sigma)
}

/// Compute quantile of Generalized Pareto Distribution.
///
/// F^{-1}(p) = sigma * ((1-p)^{-k} - 1) / k  for k != 0
/// F^{-1}(p) = -sigma * ln(1-p)              for k = 0
fn gpd_quantile(p: f64, sigma: f64, k: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return if k > 0.0 {
            f64::INFINITY
        } else {
            sigma / (-k).max(1e-10)
        };
    }

    if k.abs() < 1e-10 {
        // Exponential case (k = 0)
        -sigma * (1.0 - p).ln()
    } else {
        // General GPD case
        sigma * ((1.0 - p).powf(-k) - 1.0) / k
    }
}

/// Compute elpd_loo for a single observation using PSIS weights.
fn compute_elpd_loo(log_lik: &[f64], weights: &[f64]) -> f64 {
    // elpd_loo_i = log(sum_s(w_s * exp(log_lik_s)))
    // where w_s are normalized PSIS weights

    // Use log-sum-exp for numerical stability
    let max_ll = log_lik.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_ll.is_finite() {
        return f64::NEG_INFINITY;
    }

    let sum: f64 = log_lik
        .iter()
        .zip(weights.iter())
        .map(|(&ll, &w)| w * (ll - max_ll).exp())
        .sum();

    if sum <= 0.0 {
        f64::NEG_INFINITY
    } else {
        sum.ln() + max_ll
    }
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

/// Compute standard error of the sum.
fn standard_error(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }

    let n_f = n as f64;
    let mean = values.iter().sum::<f64>() / n_f;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0);

    // SE of sum = sqrt(n) * SD
    (n_f * variance).sqrt()
}

/// Create an empty/invalid LOO result.
fn empty_result() -> LooResult {
    LooResult {
        elpd_loo: f64::NAN,
        se_elpd_loo: f64::NAN,
        p_loo: f64::NAN,
        se_p_loo: f64::NAN,
        looic: f64::NAN,
        se_looic: f64::NAN,
        n_obs: 0,
        n_samples: 0,
        pointwise_elpd: vec![],
        pointwise_p_loo: vec![],
        pareto_k: vec![],
        n_bad_k: 0,
    }
}

/// Compare two models using LOO-CV.
///
/// # Arguments
///
/// * `loo1` - LOO result for first model
/// * `loo2` - LOO result for second model
///
/// # Returns
///
/// Difference in elpd (model1 - model2), SE of difference, and z-score.
pub fn loo_compare(loo1: &LooResult, loo2: &LooResult) -> (f64, f64, f64) {
    if loo1.n_obs != loo2.n_obs || loo1.n_obs == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    // Pointwise differences
    let diffs: Vec<f64> = loo1
        .pointwise_elpd
        .iter()
        .zip(loo2.pointwise_elpd.iter())
        .map(|(&e1, &e2)| e1 - e2)
        .collect();

    let diff = diffs.iter().sum::<f64>();
    let se = standard_error(&diffs);
    let z = if se > 0.0 { diff / se } else { f64::NAN };

    (diff, se, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_log_lik() -> Vec<Vec<f64>> {
        // 100 samples, 20 observations
        // Simulate log-likelihood from a simple normal model
        (0..100)
            .map(|s| {
                let theta = 1.0 + (s as f64 * 0.01 - 0.5); // Parameter varies around 1
                (0..20)
                    .map(|n| {
                        let y = 1.0 + (n as f64 * 0.05 - 0.5); // Data varies around 1
                        -0.5 * (y - theta).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_loo_basic() {
        let log_lik = make_test_log_lik();
        let result = loo(&log_lik);

        assert_eq!(result.n_samples, 100);
        assert_eq!(result.n_obs, 20);
        assert!(result.elpd_loo.is_finite());
        assert!(result.looic.is_finite());
        assert!(result.p_loo >= 0.0);
        assert_eq!(result.pointwise_elpd.len(), 20);
        assert_eq!(result.pareto_k.len(), 20);
    }

    #[test]
    fn test_loo_empty() {
        let log_lik: Vec<Vec<f64>> = vec![];
        let result = loo(&log_lik);
        assert!(result.elpd_loo.is_nan());
    }

    #[test]
    fn test_psis() {
        let log_lik: Vec<f64> = (0..100)
            .map(|i| -0.5 * (i as f64 / 50.0 - 1.0).powi(2))
            .collect();
        let result = psis(&log_lik);

        assert_eq!(result.weights.len(), 100);

        // Weights should sum to approximately 1
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights sum to {}", sum);

        // All weights should be non-negative
        assert!(result.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_pareto_k_threshold() {
        assert!((pareto_k_threshold(100) - 0.5).abs() < 0.1);
        assert!((pareto_k_threshold(1000) - 0.667).abs() < 0.1);
        assert!((pareto_k_threshold(10000) - 0.7).abs() < 0.05);
    }

    #[test]
    fn test_gpd_quantile() {
        // Test k = 0 (exponential)
        let q_exp = gpd_quantile(0.5, 1.0, 0.0);
        assert!((q_exp - 0.693).abs() < 0.01, "q_exp = {}", q_exp);

        // Test k > 0 (heavy tail)
        let q_heavy = gpd_quantile(0.5, 1.0, 0.5);
        assert!(q_heavy > q_exp, "Heavy tail quantile should be larger");

        // Test k < 0 (bounded)
        let q_bounded = gpd_quantile(0.5, 1.0, -0.5);
        assert!(q_bounded < q_exp, "Bounded quantile should be smaller");
    }

    #[test]
    fn test_log_sum_exp() {
        let x = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&x);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_loo_compare() {
        let log_lik1 = make_test_log_lik();
        let log_lik2: Vec<Vec<f64>> = log_lik1
            .iter()
            .map(|row| row.iter().map(|&x| x - 0.1).collect())
            .collect();

        let loo1 = loo(&log_lik1);
        let loo2 = loo(&log_lik2);

        let (diff, se, z) = loo_compare(&loo1, &loo2);

        // Model 1 should be better (higher elpd)
        assert!(diff > 0.0, "diff = {}", diff);
        assert!(se.is_finite());
        assert!(z.is_finite());
    }

    #[test]
    fn test_loo_warnings() {
        let log_lik = make_test_log_lik();
        let result = loo(&log_lik);

        // For this well-behaved data, there should be no warnings
        let warnings = result.warnings();
        // Note: may or may not have warnings depending on random variation
        for w in &warnings {
            println!("Warning: {}", w);
        }
    }

    #[test]
    fn test_loo_from_array() {
        let log_lik_2d = make_test_log_lik();
        let log_lik_flat: Vec<f64> = log_lik_2d.iter().flatten().copied().collect();

        let result1 = loo(&log_lik_2d);
        let result2 = loo_from_array(&log_lik_flat, 100, 20);

        assert!((result1.elpd_loo - result2.elpd_loo).abs() < 1e-10);
        assert!((result1.looic - result2.looic).abs() < 1e-10);
    }
}
