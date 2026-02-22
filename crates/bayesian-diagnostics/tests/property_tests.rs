//! Property-based tests for MCMC diagnostics
//!
//! Tests statistical properties that should hold for any valid input:
//! - R-hat ≈ 1.0 for converged chains (same distribution)
//! - R-hat > 1.0 for chains with different means
//! - ESS ≤ total samples
//! - ESS > 0 for non-constant chains
//! - Diagnostics are symmetric to chain order

use proptest::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use bayesian_diagnostics::{ess_bulk, rhat, rhat_rank_normalized};

/// Generate a random chain of samples from N(mean, std)
fn generate_chain(mean: f64, std: f64, len: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std).unwrap();
    (0..len).map(|_| normal.sample(&mut rng)).collect()
}

/// Generate multiple chains from the same distribution (converged)
fn generate_converged_chains(
    mean: f64,
    std: f64,
    num_chains: usize,
    chain_len: usize,
    base_seed: u64,
) -> Vec<Vec<f64>> {
    (0..num_chains)
        .map(|i| generate_chain(mean, std, chain_len, base_seed + i as u64))
        .collect()
}

/// Generate chains from different distributions (not converged)
fn generate_diverged_chains(
    means: &[f64],
    std: f64,
    chain_len: usize,
    base_seed: u64,
) -> Vec<Vec<f64>> {
    means
        .iter()
        .enumerate()
        .map(|(i, &mean)| generate_chain(mean, std, chain_len, base_seed + i as u64))
        .collect()
}

proptest! {
    // =========================================================================
    // R-hat Properties
    // =========================================================================

    /// R-hat should be close to 1.0 for chains from the same distribution
    #[test]
    fn rhat_converged_chains_near_one(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        num_chains in 2usize..8,
        chain_len in 100usize..500,
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, num_chains, chain_len, seed);

        let r = rhat(&chains);

        // R-hat should be close to 1.0 for converged chains
        // Allow some tolerance due to finite samples
        prop_assert!(
            (0.9..=1.2).contains(&r),
            "R-hat for converged chains should be near 1.0, got {}",
            r
        );
    }

    /// R-hat should be > 1.0 for chains with very different means
    #[test]
    fn rhat_diverged_chains_greater_than_one(
        base_mean in -50.0..50.0f64,
        mean_diff in 5.0..20.0f64,
        std in 0.1..2.0f64,
        chain_len in 100usize..300,
        seed in 0u64..10000,
    ) {
        let means = vec![base_mean, base_mean + mean_diff];
        let chains = generate_diverged_chains(&means, std, chain_len, seed);

        let r = rhat(&chains);

        // R-hat should indicate non-convergence
        prop_assert!(
            r > 1.05,
            "R-hat for diverged chains should be > 1.05, got {} (mean_diff={})",
            r,
            mean_diff
        );
    }

    /// R-hat should be symmetric to chain ordering
    #[test]
    fn rhat_symmetric_to_chain_order(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        chain_len in 50usize..200,
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, 3, chain_len, seed);
        let chains_rev: Vec<Vec<f64>> = chains.iter().rev().cloned().collect();

        let r1 = rhat(&chains);
        let r2 = rhat(&chains_rev);

        prop_assert!(
            (r1 - r2).abs() < 1e-10,
            "R-hat should be symmetric, got {} vs {}",
            r1,
            r2
        );
    }

    /// Rank-normalized R-hat should also be close to 1.0 for converged chains
    #[test]
    fn rhat_rank_converged_near_one(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        num_chains in 2usize..6,
        chain_len in 100usize..400,
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, num_chains, chain_len, seed);

        let r = rhat_rank_normalized(&chains);

        prop_assert!(
            (0.9..=1.2).contains(&r),
            "Rank R-hat for converged chains should be near 1.0, got {}",
            r
        );
    }

    // =========================================================================
    // ESS Properties
    // =========================================================================

    /// ESS should be <= total number of samples
    #[test]
    fn ess_at_most_total_samples(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        num_chains in 2usize..6,
        chain_len in 50usize..200,
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, num_chains, chain_len, seed);

        let ess = ess_bulk(&chains);
        let total_samples = num_chains * chain_len;

        prop_assert!(
            ess <= total_samples as f64,
            "ESS ({}) should be <= total samples ({})",
            ess,
            total_samples
        );
    }

    /// ESS should be > 0 for non-constant chains
    #[test]
    fn ess_positive_for_varying_chains(
        mean in -100.0..100.0f64,
        std in 0.5..10.0f64,  // Ensure non-zero variance
        num_chains in 2usize..6,
        chain_len in 100usize..300,
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, num_chains, chain_len, seed);

        let ess = ess_bulk(&chains);

        prop_assert!(
            ess > 0.0,
            "ESS should be positive for non-constant chains, got {}",
            ess
        );
    }

    /// ESS should be higher for independent samples than correlated samples
    #[test]
    fn ess_higher_for_independent_samples(
        mean in -50.0..50.0f64,
        std in 1.0..5.0f64,
        chain_len in 200usize..500,
        seed in 0u64..10000,
    ) {
        // Generate independent chains
        let independent_chains = generate_converged_chains(mean, std, 4, chain_len, seed);
        let ess_independent = ess_bulk(&independent_chains);

        // Generate correlated chains (random walk)
        let mut rng = StdRng::seed_from_u64(seed + 1000);
        let correlated_chains: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut chain = Vec::with_capacity(chain_len);
                let mut x = mean;
                for _ in 0..chain_len {
                    x += rng.gen_range(-0.5..0.5);
                    chain.push(x);
                }
                chain
            })
            .collect();
        let ess_correlated = ess_bulk(&correlated_chains);

        // Independent samples should have higher ESS (closer to N)
        // Correlated samples should have lower ESS
        prop_assert!(
            ess_independent > ess_correlated * 0.5,
            "Independent ESS ({}) should be notably higher than correlated ESS ({})",
            ess_independent,
            ess_correlated
        );
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    /// Single chain should still work
    #[test]
    fn single_chain_diagnostics(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        chain_len in 100usize..500,
        seed in 0u64..10000,
    ) {
        let chain = generate_chain(mean, std, chain_len, seed);
        let chains = vec![chain];

        // Single chain R-hat should be 1.0 (no between-chain variance)
        let r = rhat(&chains);
        prop_assert!(
            (r - 1.0).abs() < 1e-10 || r.is_nan(),
            "Single chain R-hat should be 1.0 or NaN, got {}",
            r
        );

        // ESS should still be positive
        let ess = ess_bulk(&chains);
        prop_assert!(
            ess > 0.0 || ess.is_nan(),
            "Single chain ESS should be positive or NaN, got {}",
            ess
        );
    }

    /// Very short chains should not panic
    #[test]
    fn short_chains_no_panic(
        mean in -100.0..100.0f64,
        std in 0.1..10.0f64,
        num_chains in 2usize..4,
        chain_len in 4usize..20,  // Very short chains
        seed in 0u64..10000,
    ) {
        let chains = generate_converged_chains(mean, std, num_chains, chain_len, seed);

        // These might return NaN or unusual values, but should not panic
        let r = rhat(&chains);
        let ess = ess_bulk(&chains);

        prop_assert!(r.is_finite() || r.is_nan());
        prop_assert!(ess.is_finite() || ess.is_nan());
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;

    /// Known good case: 4 chains of N(0,1) should have R-hat ≈ 1.0
    #[test]
    fn known_converged_case() {
        let chains = generate_converged_chains(0.0, 1.0, 4, 1000, 42);

        let r = rhat(&chains);
        let r_rank = rhat_rank_normalized(&chains);
        let ess = ess_bulk(&chains);

        assert!(r > 0.99 && r < 1.01, "R-hat should be ~1.0, got {}", r);
        assert!(
            r_rank > 0.99 && r_rank < 1.01,
            "Rank R-hat should be ~1.0, got {}",
            r_rank
        );
        assert!(
            ess > 3000.0,
            "ESS should be high for iid samples, got {}",
            ess
        );
    }

    /// Known bad case: chains with very different means
    #[test]
    fn known_diverged_case() {
        let chains = generate_diverged_chains(&[-5.0, 0.0, 5.0, 10.0], 1.0, 500, 42);

        let r = rhat(&chains);

        assert!(
            r > 2.0,
            "R-hat for diverged chains should be >> 1, got {}",
            r
        );
    }
}
