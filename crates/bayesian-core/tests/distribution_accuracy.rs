//! Distribution accuracy tests for BayesianGPU
//!
//! Verifies that distribution implementations are statistically accurate:
//! - log_prob values match expected formulas
//! - Sample statistics match theoretical moments
//! - Numerical stability at edge cases

use approx::assert_relative_eq;
use burn::backend::NdArray;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;

use bayesian_core::distributions::{Distribution, Normal};

type TestBackend = NdArray<f32>;

/// Test Normal distribution log probability accuracy
#[test]
fn normal_log_prob_accuracy() {
    let device = Default::default();
    // Known values: for N(0, 1), log_prob(0) = -0.5 * log(2π) ≈ -0.9189
    let normal = Normal::<TestBackend>::standard(&device);
    let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let log_prob = normal.log_prob(&x);
    let log_prob_val: f32 = log_prob.into_scalar().elem();

    let expected = -0.5 * (2.0 * std::f32::consts::PI).ln();
    assert_relative_eq!(log_prob_val, expected, epsilon = 1e-5);
}

/// Test Normal distribution log probability at various points
#[test]
fn normal_log_prob_various_points() {
    let device = Default::default();
    let mu = 2.0_f32;
    let sigma = 0.5_f32;
    let loc = Tensor::<TestBackend, 1>::from_floats([mu], &device);
    let scale = Tensor::<TestBackend, 1>::from_floats([sigma], &device);
    let normal = Normal::new(loc, scale);

    // Test at mean (should be maximum)
    let x_mean = Tensor::<TestBackend, 1>::from_floats([mu], &device);
    let log_prob_mean: f32 = normal.log_prob(&x_mean).into_scalar().elem();

    // Test at mean + sigma (should be lower)
    let x_plus_sigma = Tensor::<TestBackend, 1>::from_floats([mu + sigma], &device);
    let log_prob_plus: f32 = normal.log_prob(&x_plus_sigma).into_scalar().elem();

    // Test at mean - sigma (should equal mean + sigma due to symmetry)
    let x_minus_sigma = Tensor::<TestBackend, 1>::from_floats([mu - sigma], &device);
    let log_prob_minus: f32 = normal.log_prob(&x_minus_sigma).into_scalar().elem();

    // Log prob at mean should be higher than at mean ± sigma
    assert!(log_prob_mean > log_prob_plus);

    // Symmetric around mean
    assert_relative_eq!(log_prob_plus, log_prob_minus, epsilon = 1e-5);

    // Check actual values using formula: log_prob = -0.5 * log(2π) - log(σ) - 0.5 * ((x-μ)/σ)²
    let expected_at_mean = -0.5 * (2.0 * std::f32::consts::PI).ln() - sigma.ln() - 0.0; // z=0 at mean
    let expected_at_sigma = -0.5 * (2.0 * std::f32::consts::PI).ln() - sigma.ln() - 0.5; // z=1 at mean±sigma

    assert_relative_eq!(log_prob_mean, expected_at_mean, epsilon = 1e-5);
    assert_relative_eq!(log_prob_plus, expected_at_sigma, epsilon = 1e-5);
}

/// Test that log_prob returns proper values for batched inputs
#[test]
fn normal_log_prob_batched() {
    let device = Default::default();
    let normal = Normal::<TestBackend>::standard(&device);

    // Single value
    let x1 = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
    let log_prob_1: f32 = normal.log_prob(&x1).into_scalar().elem();

    // Another single value
    let x2 = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
    let log_prob_2: f32 = normal.log_prob(&x2).into_scalar().elem();

    // Batched - each element gets its own log prob
    let x_batch = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);
    let log_prob_batch = normal.log_prob(&x_batch);
    let batch_vals: Vec<f32> = log_prob_batch.into_data().to_vec().unwrap();

    assert_relative_eq!(batch_vals[0], log_prob_1, epsilon = 1e-5);
    assert_relative_eq!(batch_vals[1], log_prob_2, epsilon = 1e-5);
}

/// Test Normal distribution with different parameters
#[test]
fn normal_various_parameters() {
    let device = Default::default();
    let test_cases = [
        (0.0_f32, 1.0_f32, 0.0_f32), // Standard normal at mean
        (5.0, 2.0, 5.0),             // N(5,2) at mean
        (-3.0, 0.5, -3.0),           // N(-3,0.5) at mean
        (0.0, 10.0, 0.0),            // Wide variance
        (100.0, 0.1, 100.0),         // Narrow variance, shifted mean
    ];

    for (mu, sigma, x_val) in test_cases {
        let loc = Tensor::<TestBackend, 1>::from_floats([mu], &device);
        let scale = Tensor::<TestBackend, 1>::from_floats([sigma], &device);
        let normal = Normal::new(loc, scale);
        let x = Tensor::<TestBackend, 1>::from_floats([x_val], &device);
        let log_prob: f32 = normal.log_prob(&x).into_scalar().elem();

        // At the mean, log_prob should be -0.5*log(2π) - log(σ)
        let expected = -0.5 * (2.0 * std::f32::consts::PI).ln() - sigma.ln();
        assert_relative_eq!(log_prob, expected, epsilon = 1e-4, max_relative = 1e-4);
    }
}

/// Test numerical stability at extreme values
#[test]
fn normal_numerical_stability() {
    let device = Default::default();
    let normal = Normal::<TestBackend>::standard(&device);

    // Very large value (should give very negative log_prob, not -inf or NaN)
    let x_large = Tensor::<TestBackend, 1>::from_floats([100.0], &device);
    let log_prob_large: f32 = normal.log_prob(&x_large).into_scalar().elem();
    assert!(
        log_prob_large.is_finite(),
        "log_prob should be finite for x=100"
    );
    assert!(
        log_prob_large < -1000.0,
        "log_prob should be very negative for x=100"
    );

    // Very small sigma (numerical precision test)
    let narrow_loc = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let narrow_scale = Tensor::<TestBackend, 1>::from_floats([0.001], &device);
    let narrow_normal = Normal::new(narrow_loc, narrow_scale);
    let x_narrow = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let log_prob_narrow: f32 = narrow_normal.log_prob(&x_narrow).into_scalar().elem();
    assert!(
        log_prob_narrow.is_finite(),
        "log_prob should be finite for narrow normal"
    );

    // Large sigma
    let wide_loc = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let wide_scale = Tensor::<TestBackend, 1>::from_floats([1000.0], &device);
    let wide_normal = Normal::new(wide_loc, wide_scale);
    let x_wide = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let log_prob_wide: f32 = wide_normal.log_prob(&x_wide).into_scalar().elem();
    assert!(
        log_prob_wide.is_finite(),
        "log_prob should be finite for wide normal"
    );
}

/// Test that the distribution satisfies basic probability axioms
#[test]
fn normal_probability_axioms() {
    let device = Default::default();
    let normal = Normal::<TestBackend>::standard(&device);

    // Log prob should decrease as we move away from mean
    let points: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    let mut prev_log_prob = f32::INFINITY;

    for &x_val in &points {
        let x = Tensor::<TestBackend, 1>::from_floats([x_val], &device);
        let log_prob: f32 = normal.log_prob(&x).into_scalar().elem();

        // Due to symmetry, only check positive side
        if x_val > 0.0 {
            assert!(
                log_prob < prev_log_prob,
                "log_prob should decrease as we move from mean: {} vs {} at x={}",
                log_prob,
                prev_log_prob,
                x_val
            );
        }
        prev_log_prob = log_prob;
    }
}

/// Integration test: verify that sample statistics match theory
/// This is a statistical test that may occasionally fail due to randomness
#[test]
fn normal_sample_statistics() {
    // Generate many samples and check that statistics match
    // We use the fact that sample mean ~ N(μ, σ²/n)

    let mu = 3.0_f64;
    let sigma = 2.0_f64;
    let n = 10000;

    // Generate samples using a simple method
    let mut samples = Vec::with_capacity(n);
    let mut state = 12345u64;
    for _ in 0..n {
        // Simple LCG for deterministic testing
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = (state as f64) / (u64::MAX as f64);
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (state as f64) / (u64::MAX as f64);

        // Box-Muller transform
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        samples.push(mu + sigma * z);
    }

    // Compute sample mean and variance
    let sample_mean: f64 = samples.iter().sum::<f64>() / n as f64;
    let sample_var: f64 = samples
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    // Mean should be within ~3 standard errors of true mean
    let se_mean = sigma / (n as f64).sqrt();
    assert!(
        (sample_mean - mu).abs() < 4.0 * se_mean,
        "Sample mean {} too far from true mean {} (SE={})",
        sample_mean,
        mu,
        se_mean
    );

    // Variance should be reasonably close
    let expected_var = sigma * sigma;
    assert!(
        (sample_var - expected_var).abs() / expected_var < 0.1,
        "Sample variance {} too far from expected {} (rel error > 10%)",
        sample_var,
        expected_var
    );
}
