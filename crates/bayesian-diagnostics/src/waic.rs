//! Widely Applicable Information Criterion (WAIC)
//!
//! This module implements WAIC for Bayesian model comparison.
//!
//! # Overview
//!
//! WAIC is a fully Bayesian information criterion that estimates out-of-sample
//! predictive accuracy. It is computed as:
//!
//! ```text
//! WAIC = -2 * (lppd - p_waic)
//! ```
//!
//! Where:
//! - **lppd**: Log pointwise predictive density
//! - **p_waic**: Effective number of parameters (penalty term)
//!
//! # Comparison with LOO-CV
//!
//! WAIC and PSIS-LOO-CV are asymptotically equivalent, but LOO-CV provides
//! better diagnostics (Pareto k) for detecting when estimates are unreliable.
//! We recommend using LOO-CV via [`crate::loo::loo`] when possible.
//!
//! # References
//!
//! - Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
//!   widely applicable information criterion in singular learning theory.
//!   Journal of Machine Learning Research.
//! - Gelman, A., Hwang, J., & Vehtari, A. (2014). Understanding predictive
//!   information criteria for Bayesian models.

use serde::{Deserialize, Serialize};

/// Result of WAIC computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaicResult {
    /// Expected log pointwise predictive density (higher is better)
    pub elpd_waic: f64,

    /// Standard error of elpd_waic
    pub se_elpd_waic: f64,

    /// Effective number of parameters
    pub p_waic: f64,

    /// Standard error of p_waic
    pub se_p_waic: f64,

    /// WAIC value (-2 * elpd_waic, lower is better)
    pub waic: f64,

    /// Standard error of WAIC
    pub se_waic: f64,

    /// Log pointwise predictive density (sum of pointwise lppd)
    pub lppd: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Number of posterior samples
    pub n_samples: usize,

    /// Pointwise elpd_waic values
    pub pointwise_elpd: Vec<f64>,

    /// Pointwise p_waic values
    pub pointwise_p_waic: Vec<f64>,
}

impl WaicResult {
    /// Check if any pointwise p_waic exceeds the threshold (0.4).
    ///
    /// High pointwise p_waic values suggest the WAIC approximation may be
    /// unreliable. In such cases, LOO-CV is recommended.
    pub fn has_high_p_waic(&self) -> bool {
        self.pointwise_p_waic.iter().any(|&p| p > 0.4)
    }

    /// Get observations with high p_waic (> 0.4).
    pub fn high_p_waic_observations(&self) -> Vec<(usize, f64)> {
        self.pointwise_p_waic
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.4)
            .map(|(i, &p)| (i, p))
            .collect()
    }

    /// Get diagnostic warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        let n_high = self.high_p_waic_observations().len();
        if n_high > 0 {
            warnings.push(format!(
                "{} observations have p_waic > 0.4. Consider using LOO-CV instead.",
                n_high
            ));
        }

        if self.p_waic > self.n_obs as f64 / 2.0 {
            warnings.push(format!(
                "p_waic ({:.1}) is more than half the number of observations ({}). \
                 Model may be overfitting or WAIC may be unreliable.",
                self.p_waic, self.n_obs
            ));
        }

        warnings
    }
}

/// Compute WAIC from pointwise log-likelihoods.
///
/// # Arguments
///
/// * `log_lik` - Log-likelihood matrix where rows are posterior samples (S)
///   and columns are observations (N). Shape: [S, N]
///
/// # Returns
///
/// WaicResult with elpd_waic, p_waic, waic, and diagnostics.
///
/// # Example
///
/// ```
/// use bayesian_diagnostics::waic::waic;
///
/// // Log-likelihood: 100 samples, 10 observations
/// let log_lik: Vec<Vec<f64>> = (0..100).map(|s| {
///     (0..10).map(|n| -0.5 * ((s as f64 * 0.01 - n as f64 * 0.1).powi(2))).collect()
/// }).collect();
///
/// let result = waic(&log_lik);
/// println!("WAIC: {:.2} +/- {:.2}", result.waic, result.se_waic);
/// ```
pub fn waic(log_lik: &[Vec<f64>]) -> WaicResult {
    let n_samples = log_lik.len();
    if n_samples == 0 {
        return empty_result();
    }

    let n_obs = log_lik[0].len();
    if n_obs == 0 || !log_lik.iter().all(|row| row.len() == n_obs) {
        return empty_result();
    }

    let s = n_samples as f64;

    let mut pointwise_lppd = Vec::with_capacity(n_obs);
    let mut pointwise_p_waic = Vec::with_capacity(n_obs);

    // Process each observation
    for i in 0..n_obs {
        // Get log-likelihood for observation i across all samples
        let log_lik_i: Vec<f64> = log_lik.iter().map(|row| row[i]).collect();

        // Compute lppd_i = log(mean(exp(log_lik)))
        // = log(sum(exp(log_lik)) / S)
        // = log_sum_exp(log_lik) - log(S)
        let lppd_i = log_sum_exp(&log_lik_i) - s.ln();

        // Compute p_waic_i = Var_s(log_lik_i)
        // This is the sample variance of the log-likelihoods
        let mean_ll = log_lik_i.iter().sum::<f64>() / s;
        let var_ll = log_lik_i
            .iter()
            .map(|&ll| (ll - mean_ll).powi(2))
            .sum::<f64>()
            / (s - 1.0);

        pointwise_lppd.push(lppd_i);
        pointwise_p_waic.push(var_ll);
    }

    // Compute totals
    let lppd: f64 = pointwise_lppd.iter().sum();
    let p_waic: f64 = pointwise_p_waic.iter().sum();

    // elpd_waic = lppd - p_waic
    let elpd_waic = lppd - p_waic;

    // WAIC = -2 * elpd_waic (on deviance scale)
    let waic_val = -2.0 * elpd_waic;

    // Pointwise elpd for SE calculation
    let pointwise_elpd: Vec<f64> = pointwise_lppd
        .iter()
        .zip(pointwise_p_waic.iter())
        .map(|(&lppd_i, &p_i)| lppd_i - p_i)
        .collect();

    // Compute standard errors
    let se_elpd = standard_error(&pointwise_elpd);
    let se_p_waic = standard_error(&pointwise_p_waic);
    let se_waic = 2.0 * se_elpd;

    WaicResult {
        elpd_waic,
        se_elpd_waic: se_elpd,
        p_waic,
        se_p_waic,
        waic: waic_val,
        se_waic,
        lppd,
        n_obs,
        n_samples,
        pointwise_elpd,
        pointwise_p_waic,
    }
}

/// Compute WAIC from a flat array of log-likelihoods.
///
/// # Arguments
///
/// * `log_lik` - Flat array of log-likelihoods in row-major order
/// * `n_samples` - Number of posterior samples
/// * `n_obs` - Number of observations
pub fn waic_from_array(log_lik: &[f64], n_samples: usize, n_obs: usize) -> WaicResult {
    if log_lik.len() != n_samples * n_obs {
        return empty_result();
    }

    let log_lik_2d: Vec<Vec<f64>> = (0..n_samples)
        .map(|s| {
            let start = s * n_obs;
            log_lik[start..start + n_obs].to_vec()
        })
        .collect();

    waic(&log_lik_2d)
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

/// Create an empty/invalid WAIC result.
fn empty_result() -> WaicResult {
    WaicResult {
        elpd_waic: f64::NAN,
        se_elpd_waic: f64::NAN,
        p_waic: f64::NAN,
        se_p_waic: f64::NAN,
        waic: f64::NAN,
        se_waic: f64::NAN,
        lppd: f64::NAN,
        n_obs: 0,
        n_samples: 0,
        pointwise_elpd: vec![],
        pointwise_p_waic: vec![],
    }
}

/// Compare two models using WAIC.
///
/// # Arguments
///
/// * `waic1` - WAIC result for first model
/// * `waic2` - WAIC result for second model
///
/// # Returns
///
/// Difference in elpd (model1 - model2), SE of difference, and z-score.
pub fn waic_compare(waic1: &WaicResult, waic2: &WaicResult) -> (f64, f64, f64) {
    if waic1.n_obs != waic2.n_obs || waic1.n_obs == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    // Pointwise differences
    let diffs: Vec<f64> = waic1
        .pointwise_elpd
        .iter()
        .zip(waic2.pointwise_elpd.iter())
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
    fn test_waic_basic() {
        let log_lik = make_test_log_lik();
        let result = waic(&log_lik);

        assert_eq!(result.n_samples, 100);
        assert_eq!(result.n_obs, 20);
        assert!(result.elpd_waic.is_finite());
        assert!(result.waic.is_finite());
        assert!(result.p_waic >= 0.0);
        assert!(result.lppd.is_finite());
        assert_eq!(result.pointwise_elpd.len(), 20);
        assert_eq!(result.pointwise_p_waic.len(), 20);
    }

    #[test]
    fn test_waic_empty() {
        let log_lik: Vec<Vec<f64>> = vec![];
        let result = waic(&log_lik);
        assert!(result.elpd_waic.is_nan());
    }

    #[test]
    fn test_waic_formula() {
        let log_lik = make_test_log_lik();
        let result = waic(&log_lik);

        // Verify WAIC = -2 * elpd_waic
        assert!((result.waic - (-2.0 * result.elpd_waic)).abs() < 1e-10);

        // Verify elpd_waic = lppd - p_waic
        assert!((result.elpd_waic - (result.lppd - result.p_waic)).abs() < 1e-10);
    }

    #[test]
    fn test_waic_compare() {
        let log_lik1 = make_test_log_lik();
        let log_lik2: Vec<Vec<f64>> = log_lik1
            .iter()
            .map(|row| row.iter().map(|&x| x - 0.1).collect())
            .collect();

        let waic1 = waic(&log_lik1);
        let waic2 = waic(&log_lik2);

        let (diff, se, z) = waic_compare(&waic1, &waic2);

        // Model 1 should be better (higher elpd)
        assert!(diff > 0.0, "diff = {}", diff);
        assert!(se.is_finite());
        assert!(z.is_finite());
    }

    #[test]
    fn test_waic_warnings() {
        let log_lik = make_test_log_lik();
        let result = waic(&log_lik);

        // Check warnings method works
        let warnings = result.warnings();
        for w in &warnings {
            println!("Warning: {}", w);
        }
    }

    #[test]
    fn test_log_sum_exp() {
        let x = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&x);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_waic_from_array() {
        let log_lik_2d = make_test_log_lik();
        let log_lik_flat: Vec<f64> = log_lik_2d.iter().flatten().copied().collect();

        let result1 = waic(&log_lik_2d);
        let result2 = waic_from_array(&log_lik_flat, 100, 20);

        assert!((result1.elpd_waic - result2.elpd_waic).abs() < 1e-10);
        assert!((result1.waic - result2.waic).abs() < 1e-10);
    }

    #[test]
    fn test_high_p_waic_detection() {
        // Create data with high variance in log-likelihood
        let log_lik: Vec<Vec<f64>> = (0..100)
            .map(|s| {
                (0..5)
                    .map(|_| {
                        // Large variance across samples
                        (s as f64 - 50.0) * 0.5
                    })
                    .collect()
            })
            .collect();

        let result = waic(&log_lik);

        // With high variance, p_waic should be substantial
        assert!(result.p_waic > 0.0);
    }
}
