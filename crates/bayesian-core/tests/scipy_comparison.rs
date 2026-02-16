//! Tests comparing Dirichlet and Multinomial against SciPy values.
//!
//! These expected values were computed using scipy.stats.dirichlet and
//! scipy.stats.multinomial in Python.

use burn::backend::NdArray;
use burn::prelude::*;

use bayesian_core::distributions::dirichlet::Dirichlet;
use bayesian_core::distributions::multinomial::Multinomial;

type TestBackend = NdArray<f32>;

/// Compare against mathematically computed values.
///
/// The log PDF of the Dirichlet distribution is:
/// log p(x | alpha) = log(Gamma(sum(alpha))) - sum(log(Gamma(alpha_k))) + sum((alpha_k - 1) * log(x_k))
#[test]
fn test_dirichlet_against_scipy() {
    let device = Default::default();

    // Test case 1: Uniform Dirichlet([1, 1, 1])
    // log_norm = log(Gamma(3)) - 3*log(Gamma(1)) = log(2) - 0 = 0.693...
    // For uniform, (alpha - 1) = 0, so the x terms vanish
    {
        let alpha = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 1.0], &device);
        let dirichlet = Dirichlet::new(alpha);
        let x = Tensor::from_floats([0.2, 0.3, 0.5], &device);
        let log_prob: f32 = dirichlet.log_prob(&x).into_scalar().elem();
        let expected = 0.693147180559949_f32;
        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Uniform Dirichlet: expected {}, got {}",
            expected,
            log_prob
        );
    }

    // Test case 2: Concentrated Dirichlet([5, 5, 5])
    {
        let alpha = Tensor::<TestBackend, 1>::from_floats([5.0, 5.0, 5.0], &device);
        let dirichlet = Dirichlet::new(alpha);
        let x = Tensor::from_floats([0.3, 0.35, 0.35], &device);
        let log_prob: f32 = dirichlet.log_prob(&x).into_scalar().elem();
        let expected = 2.4425914784016847_f32;
        assert!(
            (log_prob - expected).abs() < 1e-3,
            "Concentrated Dirichlet: expected ~{}, got {}",
            expected,
            log_prob
        );
    }

    // Test case 3: Asymmetric Dirichlet([1, 2, 3])
    {
        let alpha = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let dirichlet = Dirichlet::new(alpha);
        let x = Tensor::from_floats([0.1, 0.3, 0.6], &device);
        let log_prob: f32 = dirichlet.log_prob(&x).into_scalar().elem();
        let expected = 1.8687205103641853_f32;
        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Asymmetric Dirichlet: expected {}, got {}",
            expected,
            log_prob
        );
    }
}

/// Compare against mathematically computed values.
///
/// The log PMF of the Multinomial distribution is:
/// log P(x | n, p) = log(n!) - sum(log(x_k!)) + sum(x_k * log(p_k))
#[test]
fn test_multinomial_against_scipy() {
    let device = Default::default();

    // Test case 1: Simple multinomial
    {
        let probs = Tensor::<TestBackend, 1>::from_floats([0.2, 0.3, 0.5], &device);
        let multinomial = Multinomial::new(10, probs);
        let x = Tensor::from_floats([2.0, 3.0, 5.0], &device);
        let log_prob: f32 = multinomial.log_prob(&x).into_scalar().elem();
        let expected = -2.464515960140268_f32;
        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Simple Multinomial: expected {}, got {}",
            expected,
            log_prob
        );
    }

    // Test case 2: Uniform probabilities
    {
        let probs =
            Tensor::<TestBackend, 1>::from_floats([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &device);
        let multinomial = Multinomial::new(6, probs);
        let x = Tensor::from_floats([2.0, 2.0, 2.0], &device);
        let log_prob: f32 = multinomial.log_prob(&x).into_scalar().elem();
        let expected = -2.091864061678394_f32;
        assert!(
            (log_prob - expected).abs() < 1e-3,
            "Uniform Multinomial: expected ~{}, got {}",
            expected,
            log_prob
        );
    }

    // Test case 3: All counts in one category
    {
        let probs = Tensor::<TestBackend, 1>::from_floats([0.8, 0.1, 0.1], &device);
        let multinomial = Multinomial::new(5, probs);
        let x = Tensor::from_floats([5.0, 0.0, 0.0], &device);
        let log_prob: f32 = multinomial.log_prob(&x).into_scalar().elem();
        // log(0.8^5) = 5 * log(0.8) = -1.1157...
        let expected = -1.1157177565710468_f32;
        assert!(
            (log_prob - expected).abs() < 1e-3,
            "All one category: expected ~{}, got {}",
            expected,
            log_prob
        );
    }
}

/// Test that Dirichlet([a, b]) matches Beta(a, b) for 2D case
#[test]
fn test_dirichlet_beta_consistency() {
    let device = Default::default();

    // Test several parameter combinations
    let test_cases = vec![
        ([1.0, 1.0], [0.3, 0.7]),
        ([2.0, 5.0], [0.2, 0.8]),
        ([0.5, 0.5], [0.1, 0.9]),
        ([3.0, 2.0], [0.6, 0.4]),
    ];

    for (alpha, x) in test_cases {
        let alpha_tensor = Tensor::<TestBackend, 1>::from_floats(alpha.as_slice(), &device);
        let dirichlet = Dirichlet::new(alpha_tensor);
        let x_tensor = Tensor::from_floats(x.as_slice(), &device);
        let log_prob_dirichlet: f32 = dirichlet.log_prob(&x_tensor).into_scalar().elem();

        // Compute Beta(a, b) at x[0] using formula
        // log_prob = (a-1)*log(x) + (b-1)*log(1-x) - log(B(a,b))
        let a = alpha[0] as f64;
        let b = alpha[1] as f64;
        let x_val = x[0] as f64;
        let ln_beta = bayesian_core::ln_beta(a, b);
        let log_prob_beta = (a - 1.0) * x_val.ln() + (b - 1.0) * (1.0 - x_val).ln() - ln_beta;

        assert!(
            (log_prob_dirichlet as f64 - log_prob_beta).abs() < 1e-4,
            "Dirichlet({:?}) != Beta at {:?}: {} vs {}",
            alpha,
            x,
            log_prob_dirichlet,
            log_prob_beta
        );
    }
}

/// Test that Multinomial(n, [p, 1-p]) matches Binomial(n, p)
#[test]
fn test_multinomial_binomial_consistency() {
    let device = Default::default();

    // Test several (n, p, k) combinations
    let test_cases: Vec<(usize, f64, usize)> = vec![
        (10, 0.3, 3),  // n=10, p=0.3, k=3 successes
        (20, 0.5, 10), // n=20, p=0.5, k=10
        (5, 0.8, 4),   // n=5, p=0.8, k=4
        (15, 0.2, 2),  // n=15, p=0.2, k=2
    ];

    for (n, p, k) in test_cases {
        let probs = Tensor::<TestBackend, 1>::from_floats([p as f32, (1.0 - p) as f32], &device);
        let multinomial = Multinomial::new(n, probs);
        let x = Tensor::from_floats([k as f32, (n - k) as f32], &device);
        let log_prob_mult: f32 = multinomial.log_prob(&x).into_scalar().elem();

        // Binomial log_prob: log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
        let log_comb = bayesian_core::ln_gamma((n + 1) as f64)
            - bayesian_core::ln_gamma((k + 1) as f64)
            - bayesian_core::ln_gamma((n - k + 1) as f64);
        let log_prob_binom = log_comb + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln();

        assert!(
            (log_prob_mult as f64 - log_prob_binom).abs() < 1e-4,
            "Multinomial(n={}, p=[{}, {}]) != Binomial at k={}: {} vs {}",
            n,
            p,
            1.0 - p,
            k,
            log_prob_mult,
            log_prob_binom
        );
    }
}
