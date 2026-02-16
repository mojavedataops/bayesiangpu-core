//! R-hat (Gelman-Rubin) convergence diagnostic
//!
//! The R-hat statistic measures convergence by comparing between-chain
//! and within-chain variance. Values close to 1.0 indicate convergence.
//!
//! This implements the "split R-hat" variant which splits each chain in half
//! for more robust estimation.
//!
//! # References
//! - Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences.
//! - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
//!   Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC.

use serde::{Deserialize, Serialize};

/// Result of R-hat computation with additional diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhatResult {
    /// The R-hat value (should be < 1.01 for convergence)
    pub rhat: f64,
    /// Number of chains used (after splitting)
    pub num_chains: usize,
    /// Number of samples per chain (after splitting)
    pub samples_per_chain: usize,
    /// Between-chain variance (B)
    pub between_chain_var: f64,
    /// Within-chain variance (W)
    pub within_chain_var: f64,
}

/// Compute R-hat (Gelman-Rubin diagnostic) using split chains.
///
/// Values close to 1.0 indicate convergence. A common threshold is R-hat < 1.01.
///
/// # Arguments
/// * `chains` - Vector of chains, each chain is Vec<f64>
///
/// # Returns
/// The R-hat statistic, or NaN if computation is not possible.
///
/// # Example
/// ```
/// use bayesian_diagnostics::rhat::rhat;
///
/// let chains = vec![
///     vec![1.0, 1.1, 0.9, 1.0, 1.05],
///     vec![0.95, 1.0, 1.1, 0.98, 1.02],
/// ];
/// let r = rhat(&chains);
/// assert!(r < 1.1); // Should be close to 1.0 for converged chains
/// ```
pub fn rhat(chains: &[Vec<f64>]) -> f64 {
    rhat_detailed(chains).rhat
}

/// Compute R-hat with detailed results including intermediate values.
///
/// This is useful for debugging and understanding the convergence behavior.
pub fn rhat_detailed(chains: &[Vec<f64>]) -> RhatResult {
    let num_chains = chains.len();

    // Validate inputs
    if num_chains < 2 {
        return RhatResult {
            rhat: f64::NAN,
            num_chains: 0,
            samples_per_chain: 0,
            between_chain_var: f64::NAN,
            within_chain_var: f64::NAN,
        };
    }

    let min_chain_length = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    if min_chain_length < 4 {
        return RhatResult {
            rhat: f64::NAN,
            num_chains,
            samples_per_chain: min_chain_length,
            between_chain_var: f64::NAN,
            within_chain_var: f64::NAN,
        };
    }

    // Split each chain in half for more robust estimate
    let split_chains: Vec<Vec<f64>> = chains
        .iter()
        .flat_map(|c| {
            let mid = c.len() / 2;
            vec![c[..mid].to_vec(), c[mid..2 * mid].to_vec()]
        })
        .collect();

    let m = split_chains.len() as f64;
    let n = split_chains[0].len() as f64;

    // Check for degenerate case (all same values)
    let all_same = split_chains.iter().all(|c| {
        let first = c.first().unwrap_or(&0.0);
        c.iter().all(|x| (x - first).abs() < 1e-15)
    });

    if all_same {
        return RhatResult {
            rhat: 1.0,
            num_chains: split_chains.len(),
            samples_per_chain: split_chains[0].len(),
            between_chain_var: 0.0,
            within_chain_var: 0.0,
        };
    }

    // Chain means
    let chain_means: Vec<f64> = split_chains
        .iter()
        .map(|c| c.iter().sum::<f64>() / n)
        .collect();

    // Overall mean
    let overall_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance: B = (n / (m - 1)) * sum((chain_mean - overall_mean)^2)
    let b = (n / (m - 1.0))
        * chain_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>();

    // Within-chain variance: W = (1/m) * sum(s_j^2)
    // where s_j^2 is the variance of chain j
    let w = split_chains
        .iter()
        .zip(chain_means.iter())
        .map(|(chain, &mean)| chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / m;

    // Handle edge case where W is essentially zero
    if w < 1e-15 {
        return RhatResult {
            rhat: if b < 1e-15 { 1.0 } else { f64::INFINITY },
            num_chains: split_chains.len(),
            samples_per_chain: split_chains[0].len(),
            between_chain_var: b,
            within_chain_var: w,
        };
    }

    // Estimate of marginal posterior variance
    let var_hat = ((n - 1.0) / n) * w + (1.0 / n) * b;

    // R-hat = sqrt(var_hat / W)
    let rhat = (var_hat / w).sqrt();

    RhatResult {
        rhat,
        num_chains: split_chains.len(),
        samples_per_chain: split_chains[0].len(),
        between_chain_var: b,
        within_chain_var: w,
    }
}

/// Compute R-hat for a 2D array of samples.
///
/// # Arguments
/// * `samples` - 2D slice where each row is a chain
/// * `num_chains` - Number of chains
/// * `samples_per_chain` - Number of samples per chain
pub fn rhat_from_array(samples: &[f64], num_chains: usize, samples_per_chain: usize) -> f64 {
    if samples.len() != num_chains * samples_per_chain {
        return f64::NAN;
    }

    let chains: Vec<Vec<f64>> = (0..num_chains)
        .map(|i| {
            let start = i * samples_per_chain;
            let end = start + samples_per_chain;
            samples[start..end].to_vec()
        })
        .collect();

    rhat(&chains)
}

/// Compute rank-normalized R-hat (more robust for non-normal distributions).
///
/// This replaces each sample with its rank, then applies normal quantile transform.
/// Recommended for distributions with heavy tails or multiple modes.
pub fn rhat_rank_normalized(chains: &[Vec<f64>]) -> f64 {
    // Flatten and compute ranks
    let mut all_values: Vec<(f64, usize, usize)> = chains
        .iter()
        .enumerate()
        .flat_map(|(chain_idx, chain)| {
            chain
                .iter()
                .enumerate()
                .map(move |(sample_idx, &value)| (value, chain_idx, sample_idx))
        })
        .collect();

    let n_total = all_values.len();
    if n_total == 0 {
        return f64::NAN;
    }

    // Sort by value to get ranks
    all_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute z-scores from ranks using inverse normal CDF
    let mut ranked_chains: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();

    for (rank, (_, chain_idx, sample_idx)) in all_values.iter().enumerate() {
        // Fractional rank (avoiding 0 and 1)
        let p = (rank as f64 + 0.5) / n_total as f64;
        // Inverse normal CDF approximation
        let z = inverse_normal_cdf(p);
        ranked_chains[*chain_idx][*sample_idx] = z;
    }

    rhat(&ranked_chains)
}

/// Approximation of inverse normal CDF (probit function).
/// Uses Abramowitz & Stegun approximation.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Constants for approximation
    const A1: f64 = -3.969683028665376e+01;
    const A2: f64 = 2.209460984245205e+02;
    const A3: f64 = -2.759285104469687e+02;
    const A4: f64 = 1.383_577_518_672_69e2;
    const A5: f64 = -3.066479806614716e+01;
    const A6: f64 = 2.506628277459239e+00;

    const B1: f64 = -5.447609879822406e+01;
    const B2: f64 = 1.615858368580409e+02;
    const B3: f64 = -1.556989798598866e+02;
    const B4: f64 = 6.680131188771972e+01;
    const B5: f64 = -1.328068155288572e+01;

    const C1: f64 = -7.784894002430293e-03;
    const C2: f64 = -3.223964580411365e-01;
    const C3: f64 = -2.400758277161838e+00;
    const C4: f64 = -2.549732539343734e+00;
    const C5: f64 = 4.374664141464968e+00;
    const C6: f64 = 2.938163982698783e+00;

    const D1: f64 = 7.784695709041462e-03;
    const D2: f64 = 3.224671290700398e-01;
    const D3: f64 = 2.445134137142996e+00;
    const D4: f64 = 3.754408661907416e+00;

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let q;
    let r;

    if p < P_LOW {
        q = (-2.0 * p.ln()).sqrt();
        (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
            / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
    } else if p <= P_HIGH {
        q = p - 0.5;
        r = q * q;
        (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q
            / (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0)
    } else {
        q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
            / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
    }
}

/// Check if R-hat indicates convergence.
///
/// Common thresholds:
/// - R-hat < 1.01: Good convergence (recommended)
/// - R-hat < 1.05: Acceptable convergence
/// - R-hat < 1.1: May need more samples
pub fn is_converged(rhat: f64, threshold: f64) -> bool {
    rhat.is_finite() && rhat < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhat_identical_chains() {
        // Identical chains should have R-hat = 1.0
        let chains = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let r = rhat(&chains);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rhat_converged_chains() {
        // Well-mixed chains from same distribution
        let chains = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.0],
            vec![0.98, 1.02, 0.96, 1.04, 1.0, 0.99, 1.01, 0.97],
            vec![1.01, 0.99, 1.03, 0.97, 1.0, 1.02, 0.98, 1.01],
            vec![0.99, 1.01, 0.98, 1.02, 1.0, 0.99, 1.01, 1.0],
        ];
        let r = rhat(&chains);
        assert!(r < 1.1, "R-hat should be close to 1.0, got {}", r);
    }

    #[test]
    fn test_rhat_diverged_chains() {
        // Chains at different locations should have high R-hat
        let chains = vec![
            vec![0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
            vec![10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1],
        ];
        let r = rhat(&chains);
        assert!(
            r > 1.5,
            "R-hat should be high for diverged chains, got {}",
            r
        );
    }

    #[test]
    fn test_rhat_insufficient_chains() {
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let r = rhat(&chains);
        assert!(r.is_nan());
    }

    #[test]
    fn test_rhat_insufficient_samples() {
        let chains = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let r = rhat(&chains);
        assert!(r.is_nan());
    }

    #[test]
    fn test_rhat_detailed() {
        let chains = vec![
            vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
            vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
        ];
        let result = rhat_detailed(&chains);
        assert!(result.rhat.is_finite());
        assert_eq!(result.num_chains, 4); // 2 original * 2 (split)
        assert_eq!(result.samples_per_chain, 4); // 8/2
    }

    #[test]
    fn test_rhat_from_array() {
        let samples = vec![
            1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, // chain 0
            0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0, // chain 1
        ];
        let r = rhat_from_array(&samples, 2, 8);
        assert!(r.is_finite());
    }

    #[test]
    fn test_is_converged() {
        assert!(is_converged(1.001, 1.01));
        assert!(!is_converged(1.02, 1.01));
        assert!(!is_converged(f64::NAN, 1.01));
        assert!(!is_converged(f64::INFINITY, 1.01));
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test known values
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.841344746) - 1.0).abs() < 1e-3);
        assert!((inverse_normal_cdf(0.158655254) - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_rhat_rank_normalized() {
        let chains = vec![
            vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
            vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
        ];
        let r = rhat_rank_normalized(&chains);
        assert!(r.is_finite());
        assert!(r > 0.0);
    }
}
