//! Effective Sample Size (ESS) computation
//!
//! ESS estimates the number of independent samples in an autocorrelated chain.
//! Low ESS indicates high autocorrelation and suggests more samples are needed.
//!
//! This module implements:
//! - Basic ESS using autocorrelation
//! - Bulk ESS (for central tendency estimates)
//! - Tail ESS (for quantile estimates)
//!
//! # References
//! - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
//!   Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC.

use serde::{Deserialize, Serialize};

/// Result of ESS computation with additional diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssResult {
    /// Effective sample size
    pub ess: f64,
    /// Total number of samples across all chains
    pub total_samples: usize,
    /// ESS per second (if timing info provided)
    pub ess_per_sec: Option<f64>,
    /// Maximum lag used in autocorrelation computation
    pub max_lag: usize,
}

/// Compute effective sample size using autocorrelation.
///
/// Uses the initial monotone sequence estimator (IMSE) which is more robust
/// than the naive sum of autocorrelations.
///
/// # Arguments
/// * `chains` - Vector of chains, each chain is Vec<f64>
///
/// # Returns
/// The effective sample size, or the total sample size if computation fails.
///
/// # Example
/// ```
/// use bayesian_diagnostics::ess::ess;
///
/// let chains = vec![
///     vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
///     vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
/// ];
/// let effective_samples = ess(&chains);
/// assert!(effective_samples > 0.0);
/// ```
pub fn ess(chains: &[Vec<f64>]) -> f64 {
    ess_detailed(chains).ess
}

/// Compute ESS with detailed results using proper multi-chain estimation.
///
/// Implements the method from Vehtari et al. (2021) which correctly handles
/// multiple chains by computing within-chain and between-chain variance,
/// then pooling autocorrelations across chains.
pub fn ess_detailed(chains: &[Vec<f64>]) -> EssResult {
    let num_chains = chains.len();
    if num_chains == 0 {
        return EssResult {
            ess: 0.0,
            total_samples: 0,
            ess_per_sec: None,
            max_lag: 0,
        };
    }

    // Check all chains have the same length
    let n_per_chain = chains[0].len();
    if n_per_chain < 4 || !chains.iter().all(|c| c.len() == n_per_chain) {
        // Fall back to simple concatenation for irregular chains
        let total: usize = chains.iter().map(|c| c.len()).sum();
        return EssResult {
            ess: total as f64,
            total_samples: total,
            ess_per_sec: None,
            max_lag: 0,
        };
    }

    let m = num_chains as f64; // number of chains
    let n = n_per_chain as f64; // samples per chain
    let total_samples = num_chains * n_per_chain;

    // Compute chain means
    let chain_means: Vec<f64> = chains
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / n)
        .collect();

    // Compute global mean
    let global_mean = chain_means.iter().sum::<f64>() / m;

    // Compute within-chain variance (W)
    let chain_variances: Vec<f64> = chains
        .iter()
        .zip(chain_means.iter())
        .map(|(chain, &mean)| chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
        .collect();
    let w = chain_variances.iter().sum::<f64>() / m;

    // Compute between-chain variance (B)
    let b = n * chain_means
        .iter()
        .map(|&mean| (mean - global_mean).powi(2))
        .sum::<f64>()
        / (m - 1.0);

    // Combined variance estimate (var_hat+)
    let var_plus = ((n - 1.0) / n) * w + b / n;

    // Handle degenerate case (zero variance)
    if var_plus < 1e-15 {
        return EssResult {
            ess: total_samples as f64,
            total_samples,
            ess_per_sec: None,
            max_lag: 0,
        };
    }

    // Compute max lag for autocorrelation
    let max_lag = ((n / 2.0) as usize).min(1000);

    // Pool autocorrelations across chains using Vehtari et al. formula:
    // rho_t = 1 - (W - mean(variogram_t)) / var_plus
    // where variogram_t = mean of squared differences at lag t
    let pooled_autocorr: Vec<f64> = (0..max_lag)
        .map(|lag| {
            if lag == 0 {
                1.0
            } else {
                // Compute mean variogram at this lag across all chains
                let variogram_sum: f64 = chains
                    .iter()
                    .map(|chain| {
                        let count = chain.len() - lag;
                        if count == 0 {
                            0.0
                        } else {
                            chain
                                .iter()
                                .zip(chain.iter().skip(lag))
                                .map(|(&a, &b)| (a - b).powi(2))
                                .sum::<f64>()
                                / (2.0 * count as f64)
                        }
                    })
                    .sum();
                let mean_variogram = variogram_sum / m;

                // rho_t = 1 - V_t / var_plus
                1.0 - mean_variogram / var_plus
            }
        })
        .collect();

    // Use initial monotone sequence estimator (Geyer, 1992)
    // Sum pairs of consecutive autocorrelations until the sum becomes negative
    let mut rho_sum = 0.0;
    let mut t = 0;
    let mut prev_pair_sum = f64::INFINITY;

    while t + 1 < max_lag {
        let rho_t = pooled_autocorr[t];
        let rho_t1 = pooled_autocorr[t + 1];
        let pair_sum = rho_t + rho_t1;

        // Initial monotone sequence: stop if pair sum becomes negative
        // or if it increases (not monotone decreasing)
        if pair_sum < 0.0 {
            break;
        }
        if pair_sum > prev_pair_sum {
            break;
        }

        rho_sum += pair_sum;
        prev_pair_sum = pair_sum;
        t += 2;
    }

    // ESS = (M * N) / tau, where tau = 1 + 2 * sum(rho_t)
    // Since we're summing pairs, rho_sum = sum of pairs starting from rho_0
    // tau = -1 + 2 * rho_sum (since rho_0 = 1 is included in first pair)
    let tau = -1.0 + 2.0 * rho_sum;
    let tau = tau.max(1.0); // tau must be at least 1

    let ess = (m * n) / tau;

    // Ensure ESS is at least 1 and at most total samples
    let ess = ess.max(1.0).min(total_samples as f64);

    EssResult {
        ess,
        total_samples,
        ess_per_sec: None,
        max_lag: t,
    }
}

/// Compute bulk ESS (for central tendency estimates).
///
/// Bulk ESS uses rank-normalized samples to better capture mixing
/// in the center of the distribution.
///
/// # Arguments
/// * `chains` - Vector of chains
///
/// # Returns
/// Bulk ESS value
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    let ranked_chains = rank_normalize(chains);
    ess(&ranked_chains)
}

/// Compute tail ESS (for quantile estimates).
///
/// Tail ESS uses folded and rank-normalized samples to assess
/// mixing in the tails of the distribution.
///
/// # Arguments
/// * `chains` - Vector of chains
///
/// # Returns
/// Tail ESS value
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    // Fold at the median to focus on tails
    let folded_chains = fold_at_median(chains);
    let ranked_chains = rank_normalize(&folded_chains);
    ess(&ranked_chains)
}

/// Rank-normalize samples across all chains.
fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
        return chains.to_vec();
    }

    // Sort by value
    all_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks and transform to z-scores
    let mut ranked_chains: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();

    for (rank, (_, chain_idx, sample_idx)) in all_values.iter().enumerate() {
        let p = (rank as f64 + 0.5) / n_total as f64;
        let z = inverse_normal_cdf(p);
        ranked_chains[*chain_idx][*sample_idx] = z;
    }

    ranked_chains
}

/// Fold samples at the median to focus on tails.
fn fold_at_median(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Compute global median
    let mut all_values: Vec<f64> = chains.iter().flatten().copied().collect();
    if all_values.is_empty() {
        return chains.to_vec();
    }

    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = all_values[all_values.len() / 2];

    // Fold at median: |x - median|
    chains
        .iter()
        .map(|chain| chain.iter().map(|&x| (x - median).abs()).collect())
        .collect()
}

/// Inverse normal CDF approximation (same as in rhat.rs).
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

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

/// Compute ESS from a 2D array of samples.
///
/// # Arguments
/// * `samples` - Flat array of samples
/// * `num_chains` - Number of chains
/// * `samples_per_chain` - Number of samples per chain
pub fn ess_from_array(samples: &[f64], num_chains: usize, samples_per_chain: usize) -> f64 {
    if samples.len() != num_chains * samples_per_chain {
        return 0.0;
    }

    let chains: Vec<Vec<f64>> = (0..num_chains)
        .map(|i| {
            let start = i * samples_per_chain;
            let end = start + samples_per_chain;
            samples[start..end].to_vec()
        })
        .collect();

    ess(&chains)
}

/// Check if ESS is sufficient.
///
/// Rule of thumb: ESS should be at least 100 per chain for reliable estimates,
/// or at least 400 total for good precision.
pub fn is_sufficient(ess_value: f64, min_ess: f64) -> bool {
    ess_value.is_finite() && ess_value >= min_ess
}

/// Compute ESS efficiency ratio (ESS / total samples).
///
/// Higher values (closer to 1.0) indicate better sampling efficiency.
pub fn ess_efficiency(chains: &[Vec<f64>]) -> f64 {
    let total_samples: usize = chains.iter().map(|c| c.len()).sum();
    if total_samples == 0 {
        return 0.0;
    }
    ess(chains) / total_samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ess_independent_samples() {
        // Independent samples should have ESS close to n
        // Using random-looking but deterministic values
        let chains = vec![
            vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.15, 0.85],
            vec![0.5, 0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.38, 0.62, 0.48],
        ];
        let ess_val = ess(&chains);
        assert!(ess_val > 5.0, "ESS should be significant, got {}", ess_val);
    }

    #[test]
    fn test_ess_highly_correlated() {
        // Highly autocorrelated samples should have low ESS
        let chains = vec![
            vec![1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
            vec![2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09],
        ];
        let ess_val = ess(&chains);
        // ESS should be much lower than total samples (20)
        assert!(
            ess_val < 20.0,
            "ESS should be less than n for correlated samples"
        );
    }

    #[test]
    fn test_ess_constant() {
        // Constant values should have ESS = n
        let chains = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let ess_val = ess(&chains);
        assert!((ess_val - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_ess_empty() {
        let chains: Vec<Vec<f64>> = vec![];
        let ess_val = ess(&chains);
        assert!((ess_val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ess_single_sample() {
        let chains = vec![vec![1.0]];
        let ess_val = ess(&chains);
        assert!((ess_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ess_bulk() {
        let chains = vec![
            vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
            vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
        ];
        let bulk = ess_bulk(&chains);
        assert!(bulk > 0.0);
        assert!(bulk.is_finite());
    }

    #[test]
    fn test_ess_tail() {
        let chains = vec![
            vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98],
            vec![0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0],
        ];
        let tail = ess_tail(&chains);
        assert!(tail > 0.0);
        assert!(tail.is_finite());
    }

    #[test]
    fn test_ess_from_array() {
        let samples = vec![
            1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, // chain 0
            0.95, 1.0, 1.1, 0.98, 1.02, 0.97, 1.03, 1.0, // chain 1
        ];
        let ess_val = ess_from_array(&samples, 2, 8);
        assert!(ess_val > 0.0);
    }

    #[test]
    fn test_is_sufficient() {
        assert!(is_sufficient(500.0, 100.0));
        assert!(!is_sufficient(50.0, 100.0));
        assert!(!is_sufficient(f64::NAN, 100.0));
    }

    #[test]
    fn test_ess_efficiency() {
        let chains = vec![
            vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6],
            vec![0.5, 0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.38],
        ];
        let efficiency = ess_efficiency(&chains);
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);
    }

    #[test]
    fn test_rank_normalize() {
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let ranked = rank_normalize(&chains);

        // Check that values are now z-scores (centered around 0)
        let all: Vec<f64> = ranked.iter().flatten().copied().collect();
        let mean: f64 = all.iter().sum::<f64>() / all.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "Rank-normalized mean should be ~0, got {}",
            mean
        );
    }

    #[test]
    fn test_fold_at_median() {
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let folded = fold_at_median(&chains);

        // All values should be non-negative after folding
        assert!(folded[0].iter().all(|&x| x >= 0.0));
    }
}
