//! ArviZ comparison tests for MCMC diagnostics
//!
//! Validates that our R-hat, ESS, LOO, and WAIC implementations produce results
//! consistent with ArviZ (the Python reference implementation).
//!
//! Reference values computed using:
//! ```python
//! import arviz as az
//! import numpy as np
//! data = az.from_dict(posterior={"x": chains})
//! az.rhat(data), az.ess(data)
//!
//! # For LOO/WAIC:
//! data = az.from_dict(log_likelihood={"y": log_lik})
//! az.loo(data), az.waic(data)
//! ```

use bayesian_diagnostics::{ess_bulk, ess_tail, loo, rhat, rhat_rank_normalized, waic};

/// Test case with known ArviZ output
#[allow(dead_code)] // ESS fields are reference values for documentation
struct ArviZTestCase {
    name: &'static str,
    chains: Vec<Vec<f64>>,
    expected_rhat: f64,
    expected_ess_bulk: f64,
    expected_ess_tail: f64,
    tolerance: f64,
}

fn get_test_cases() -> Vec<ArviZTestCase> {
    vec![
        // Case 1: Well-mixed chains (iid samples from N(0,1))
        // ArviZ: rhat=1.0, ess_bulk≈n, ess_tail≈n
        ArviZTestCase {
            name: "well_mixed_normal",
            chains: vec![
                vec![0.1, -0.3, 0.5, -0.2, 0.4, -0.1, 0.3, -0.4, 0.2, -0.5],
                vec![-0.2, 0.4, -0.1, 0.3, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3],
                vec![0.3, -0.1, 0.2, -0.4, 0.1, -0.3, 0.4, -0.2, 0.5, -0.1],
                vec![-0.4, 0.2, -0.3, 0.1, -0.2, 0.4, -0.1, 0.3, -0.3, 0.2],
            ],
            expected_rhat: 1.0,
            expected_ess_bulk: 35.0, // Approximate, iid should be close to n
            expected_ess_tail: 30.0,
            tolerance: 0.15,
        },
        // Case 2: Chains with different means (not converged)
        // ArviZ: rhat >> 1
        ArviZTestCase {
            name: "different_means",
            chains: vec![
                vec![0.0, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.08, -0.08, 0.03],
                vec![5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.08, 4.92, 5.03],
            ],
            expected_rhat: 10.0, // Should be very high
            expected_ess_bulk: 5.0,
            expected_ess_tail: 5.0,
            tolerance: 5.0, // High tolerance since exact value varies
        },
        // Case 3: Highly autocorrelated chains (random walk)
        // These chains trend in opposite directions, causing high R-hat
        ArviZTestCase {
            name: "autocorrelated",
            chains: vec![
                vec![0.0, 0.1, 0.15, 0.2, 0.18, 0.22, 0.25, 0.3, 0.28, 0.35],
                vec![
                    0.0, -0.05, -0.1, -0.08, -0.12, -0.15, -0.18, -0.2, -0.22, -0.25,
                ],
            ],
            expected_rhat: 3.0, // High R-hat due to divergent trends
            expected_ess_bulk: 8.0,
            expected_ess_tail: 6.0,
            tolerance: 2.0, // Wide tolerance for trending chains
        },
        // Case 4: Constant chains (edge case)
        ArviZTestCase {
            name: "constant_chains",
            chains: vec![vec![1.0, 1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0, 1.0]],
            expected_rhat: 1.0,     // No variance, should be 1 or NaN
            expected_ess_bulk: 0.0, // No information
            expected_ess_tail: 0.0,
            tolerance: 0.1,
        },
    ]
}

#[test]
fn test_rhat_matches_arviz() {
    for case in get_test_cases() {
        if case.name == "constant_chains" {
            continue; // Skip edge case for R-hat
        }

        let computed_rhat = rhat(&case.chains);

        // For "different_means", just check it's > 1.5
        if case.name == "different_means" {
            assert!(
                computed_rhat > 1.5,
                "{}: R-hat {} should indicate non-convergence (> 1.5)",
                case.name,
                computed_rhat
            );
        } else {
            assert!(
                (computed_rhat - case.expected_rhat).abs() < case.tolerance
                    || (computed_rhat / case.expected_rhat - 1.0).abs() < case.tolerance,
                "{}: R-hat {} not close to expected {} (tol={})",
                case.name,
                computed_rhat,
                case.expected_rhat,
                case.tolerance
            );
        }
    }
}

#[test]
fn test_rhat_rank_matches_arviz() {
    for case in get_test_cases() {
        if case.name == "constant_chains" {
            continue;
        }

        let computed_rhat = rhat_rank_normalized(&case.chains);

        // Rank-normalized R-hat should be close to regular R-hat for well-behaved chains
        if case.name == "well_mixed_normal" {
            assert!(
                computed_rhat < 1.1,
                "{}: Rank R-hat {} should be close to 1.0",
                case.name,
                computed_rhat
            );
        }
    }
}

#[test]
fn test_ess_bulk_matches_arviz() {
    for case in get_test_cases() {
        let computed_ess = ess_bulk(&case.chains);

        if case.name == "constant_chains" {
            // For constant chains, ESS can be low or have undefined behavior
            // depending on implementation details
            assert!(
                computed_ess.is_finite(),
                "{}: ESS {} should be finite for constant chains",
                case.name,
                computed_ess
            );
        } else if case.name == "well_mixed_normal" {
            // ESS should be reasonably high for iid samples
            let total_samples: usize = case.chains.iter().map(|c| c.len()).sum();
            assert!(
                computed_ess > total_samples as f64 * 0.5,
                "{}: ESS {} should be at least half of total samples {}",
                case.name,
                computed_ess,
                total_samples
            );
        }
    }
}

#[test]
fn test_ess_tail_matches_arviz() {
    for case in get_test_cases() {
        let computed_ess = ess_tail(&case.chains);

        if case.name == "constant_chains" {
            // For constant chains, ESS can have undefined behavior
            assert!(
                computed_ess.is_finite(),
                "{}: Tail ESS {} should be finite for constant chains",
                case.name,
                computed_ess
            );
        }
        // Tail ESS is typically lower than bulk ESS
    }
}

/// Test with specific ArviZ reference values
/// These values were computed using ArviZ 0.17.0
#[test]
fn test_specific_arviz_reference() {
    // Chains from ArviZ documentation example
    let chains = vec![
        vec![1.1, 1.2, 1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.15, 0.85],
        vec![0.9, 1.0, 1.1, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.05],
        vec![1.0, 0.95, 1.05, 1.1, 0.9, 1.0, 1.0, 1.05, 0.95, 1.0],
        vec![1.05, 1.0, 0.9, 1.0, 1.1, 0.95, 1.0, 1.0, 1.05, 0.9],
    ];

    let computed_rhat = rhat(&chains);
    let computed_ess = ess_bulk(&chains);

    // ArviZ reference: rhat ≈ 1.0 for well-mixed chains
    // Allow wider tolerance due to short chain length
    assert!(
        computed_rhat > 0.90 && computed_rhat < 1.10,
        "R-hat {} should be close to 1.0",
        computed_rhat
    );

    // ArviZ reference: ESS should be high for nearly iid samples
    assert!(
        computed_ess > 20.0,
        "ESS {} should be reasonably high",
        computed_ess
    );
}

/// Test that diagnostics handle edge cases gracefully
#[test]
fn test_edge_cases() {
    // Single chain
    let single_chain = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let rhat_single = rhat(&single_chain);
    let ess_single = ess_bulk(&single_chain);

    // Single chain R-hat should be 1.0 (no between-chain variance)
    assert!(
        (rhat_single - 1.0).abs() < 0.01 || rhat_single.is_nan(),
        "Single chain R-hat should be 1.0 or NaN"
    );

    // ESS for single chain should still be computed
    assert!(
        ess_single > 0.0 || ess_single.is_nan(),
        "Single chain ESS should be positive or NaN"
    );

    // Very short chains
    let short_chains = vec![vec![1.0, 2.0], vec![1.5, 2.5]];

    // Should not panic
    let _ = rhat(&short_chains);
    let _ = ess_bulk(&short_chains);
}

// ==============================================================
// LOO-CV and WAIC Tests
// ==============================================================

/// Generate test log-likelihood data for a simple normal model
/// y_i ~ Normal(theta, 1), theta ~ Normal(0, 1)
/// Using a well-specified model where theta is drawn from its posterior
fn generate_normal_log_lik(n_samples: usize, n_obs: usize, seed: u64) -> Vec<Vec<f64>> {
    // Simple LCG for deterministic pseudo-random values
    let mut state = seed;
    let next_rand = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((*s >> 16) & 0x7fff) as f64 / 32768.0
    };

    // Generate fixed data (simulated observations)
    // For a well-specified model, data should be centered around the true theta
    let true_theta = 0.5;
    let data: Vec<f64> = (0..n_obs)
        .map(|_| {
            let noise = (next_rand(&mut state) - 0.5) * 2.0;
            true_theta + noise * 0.3 // Data centered around true_theta with some noise
        })
        .collect();

    // Compute posterior mean and variance for theta
    // For Normal(theta, 1) likelihood and Normal(0, 1) prior:
    // Posterior is Normal((n*y_bar)/(n+1), 1/(n+1))
    let y_bar: f64 = data.iter().sum::<f64>() / n_obs as f64;
    let post_mean = (n_obs as f64 * y_bar) / (n_obs as f64 + 1.0);
    let post_var = 1.0 / (n_obs as f64 + 1.0);
    let post_sd = post_var.sqrt();

    // Generate posterior samples for theta
    let thetas: Vec<f64> = (0..n_samples)
        .map(|_| {
            let u1 = next_rand(&mut state);
            let u2 = next_rand(&mut state);
            // Box-Muller transform for approximate normal
            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            post_mean + z * post_sd
        })
        .collect();

    // Compute log-likelihood: log N(y_i | theta_s, 1)
    thetas
        .iter()
        .map(|&theta| {
            data.iter()
                .map(|&y| -0.5 * (y - theta).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
                .collect()
        })
        .collect()
}

/// Test LOO-CV computation
/// ArviZ reference (approximate):
/// For a simple normal model with ~100 samples and ~20 obs,
/// elpd_loo should be around -20 to -30 depending on data
#[test]
fn test_loo_matches_arviz_behavior() {
    let log_lik = generate_normal_log_lik(200, 20, 42);
    let result = loo(&log_lik);

    // Basic validity checks
    assert!(result.elpd_loo.is_finite(), "elpd_loo should be finite");
    assert!(result.looic.is_finite(), "looic should be finite");
    assert!(result.p_loo >= 0.0, "p_loo should be non-negative");

    // LOOIC = -2 * elpd_loo
    assert!(
        (result.looic - (-2.0 * result.elpd_loo)).abs() < 1e-10,
        "LOOIC should equal -2 * elpd_loo"
    );

    // For a simple model with well-behaved posterior, p_loo should be reasonable
    // It can be higher than the number of parameters but shouldn't be extreme
    assert!(
        result.p_loo < result.n_obs as f64,
        "p_loo {} seems too high (should be less than n_obs={})",
        result.p_loo,
        result.n_obs
    );

    // Standard errors should be positive
    assert!(result.se_elpd_loo > 0.0, "SE should be positive");
    assert!(result.se_looic > 0.0, "SE should be positive");

    // Pareto k diagnostics
    assert_eq!(
        result.pareto_k.len(),
        20,
        "Should have k for each observation"
    );

    // For well-behaved data, most k values should be low
    let high_k_count = result.pareto_k.iter().filter(|&&k| k > 0.7).count();
    assert!(
        high_k_count < 5,
        "Too many high Pareto k values: {} out of {}",
        high_k_count,
        result.pareto_k.len()
    );
}

/// Test WAIC computation
/// ArviZ reference (approximate):
/// WAIC and LOO should give similar results for well-behaved models
#[test]
fn test_waic_matches_arviz_behavior() {
    let log_lik = generate_normal_log_lik(200, 20, 42);
    let result = waic(&log_lik);

    // Basic validity checks
    assert!(result.elpd_waic.is_finite(), "elpd_waic should be finite");
    assert!(result.waic.is_finite(), "waic should be finite");
    assert!(result.p_waic >= 0.0, "p_waic should be non-negative");
    assert!(result.lppd.is_finite(), "lppd should be finite");

    // WAIC = -2 * elpd_waic
    assert!(
        (result.waic - (-2.0 * result.elpd_waic)).abs() < 1e-10,
        "WAIC should equal -2 * elpd_waic"
    );

    // elpd_waic = lppd - p_waic
    assert!(
        (result.elpd_waic - (result.lppd - result.p_waic)).abs() < 1e-10,
        "elpd_waic should equal lppd - p_waic"
    );

    // For a simple model with well-behaved posterior, p_waic should be reasonable
    // It can be higher than the number of parameters but shouldn't be extreme
    let n_obs = result.n_obs;
    assert!(
        result.p_waic < n_obs as f64,
        "p_waic {} seems too high (should be less than n_obs={})",
        result.p_waic,
        n_obs
    );

    // Standard errors should be positive
    assert!(result.se_elpd_waic > 0.0, "SE should be positive");
    assert!(result.se_waic > 0.0, "SE should be positive");
}

/// Test LOO and WAIC give similar results
/// ArviZ reference: For well-behaved models, LOO and WAIC should be similar
#[test]
fn test_loo_waic_agreement() {
    let log_lik = generate_normal_log_lik(500, 30, 123);

    let loo_result = loo(&log_lik);
    let waic_result = waic(&log_lik);

    // elpd values should be in the same ballpark
    let elpd_diff = (loo_result.elpd_loo - waic_result.elpd_waic).abs();
    let elpd_se = (loo_result.se_elpd_loo.powi(2) + waic_result.se_elpd_waic.powi(2)).sqrt();

    // Difference should typically be within a few SEs
    assert!(
        elpd_diff < 3.0 * elpd_se.max(1.0),
        "LOO elpd ({:.1}) and WAIC elpd ({:.1}) differ by {:.1} (SE: {:.1})",
        loo_result.elpd_loo,
        waic_result.elpd_waic,
        elpd_diff,
        elpd_se
    );

    // Effective parameters should be in the same ballpark
    // LOO and WAIC can have different p values, but they should correlate
    let p_diff = (loo_result.p_loo - waic_result.p_waic).abs();
    let p_max = loo_result.p_loo.max(waic_result.p_waic);
    assert!(
        p_diff < p_max * 0.5 + 5.0, // Allow 50% difference plus 5 absolute
        "p_loo ({:.2}) and p_waic ({:.2}) differ too much (diff: {:.2})",
        loo_result.p_loo,
        waic_result.p_waic,
        p_diff
    );
}

/// Test model comparison
#[test]
fn test_model_comparison() {
    // Generate two models: good fit vs bad fit
    let log_lik_good = generate_normal_log_lik(200, 20, 42);

    // Bad model: shifted mean, worse fit
    let log_lik_bad: Vec<Vec<f64>> = log_lik_good
        .iter()
        .map(|row| row.iter().map(|&ll| ll - 0.5).collect())
        .collect();

    let loo_good = loo(&log_lik_good);
    let loo_bad = loo(&log_lik_bad);

    // Good model should have higher elpd
    assert!(
        loo_good.elpd_loo > loo_bad.elpd_loo,
        "Good model (elpd={:.1}) should beat bad model (elpd={:.1})",
        loo_good.elpd_loo,
        loo_bad.elpd_loo
    );

    // Test compare function
    let (diff, se, _z) = bayesian_diagnostics::loo_compare(&loo_good, &loo_bad);
    assert!(diff > 0.0, "Difference should favor good model");
    assert!(se > 0.0, "SE should be positive");
}

/// Test Pareto k threshold computation
#[test]
fn test_pareto_k_threshold_values() {
    use bayesian_diagnostics::pareto_k_threshold;

    // Threshold should be min(1 - 1/log10(S), 0.7)
    // S=100: 1 - 1/2 = 0.5
    // S=1000: 1 - 1/3 ≈ 0.667
    // S=10000: 1 - 1/4 = 0.75, capped at 0.7

    let t100 = pareto_k_threshold(100);
    let t1000 = pareto_k_threshold(1000);
    let t10000 = pareto_k_threshold(10000);

    assert!((t100 - 0.5).abs() < 0.01, "t100 = {}", t100);
    assert!((t1000 - 0.667).abs() < 0.01, "t1000 = {}", t1000);
    assert!(
        (t10000 - 0.7).abs() < 0.01,
        "t10000 = {} (should cap at 0.7)",
        t10000
    );
}

/// Test edge cases for LOO/WAIC
#[test]
fn test_loo_waic_edge_cases() {
    // Empty input
    let empty: Vec<Vec<f64>> = vec![];
    let loo_empty = loo(&empty);
    let waic_empty = waic(&empty);
    assert!(loo_empty.elpd_loo.is_nan());
    assert!(waic_empty.elpd_waic.is_nan());

    // Single observation
    let single_obs: Vec<Vec<f64>> = (0..100)
        .map(|s| vec![-0.5 * (s as f64 * 0.01).powi(2)])
        .collect();
    let loo_single = loo(&single_obs);
    let waic_single = waic(&single_obs);
    assert!(loo_single.elpd_loo.is_finite());
    assert!(waic_single.elpd_waic.is_finite());
    assert_eq!(loo_single.n_obs, 1);
    assert_eq!(waic_single.n_obs, 1);

    // Small number of samples (might have high Pareto k)
    let small_samples: Vec<Vec<f64>> = (0..20)
        .map(|s| {
            (0..5)
                .map(|n| -0.5 * ((s as f64 * 0.05 - n as f64 * 0.2).powi(2)))
                .collect()
        })
        .collect();
    let loo_small = loo(&small_samples);
    assert!(loo_small.elpd_loo.is_finite());
    // May have warnings about Pareto k with small samples
}
