//! Multinomial distribution
//!
//! The multinomial distribution is a generalization of the binomial distribution
//! to K categories. It describes the probability of counts for each of K categories
//! in n independent trials.

use super::Support;
use crate::math::ln_gamma;
use burn::prelude::*;

/// Multinomial distribution
///
/// The multinomial distribution models the number of occurrences of K outcomes
/// in n independent trials, where each trial has probabilities p_1, ..., p_K.
///
/// P(x | n, p) = n! / (∏_k x_k!) * ∏_k p_k^x_k
///
/// # Parameters
/// - `n`: Total number of trials
/// - `probs`: Probability vector (must sum to 1)
///
/// # Support
/// Non-negative integer vectors x where ∑_k x_k = n
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::multinomial::Multinomial;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // 10 trials with probabilities [0.2, 0.3, 0.5]
/// let probs = Tensor::<B, 1>::from_floats([0.2, 0.3, 0.5], &device);
/// let dist = Multinomial::new(10, probs);
///
/// // Observed counts
/// let x = Tensor::<B, 1>::from_floats([2.0, 3.0, 5.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Multinomial<B: Backend> {
    /// Total number of trials
    pub n: usize,
    /// Probability vector (sums to 1)
    pub probs: Tensor<B, 1>,
    /// Number of categories K
    dim: usize,
    /// Pre-computed: log(n!)
    log_n_factorial: f32,
    /// Pre-computed: log(probs)
    log_probs: Tensor<B, 1>,
}

impl<B: Backend> Multinomial<B> {
    /// Create a new Multinomial distribution.
    ///
    /// # Arguments
    /// * `n` - Total number of trials
    /// * `probs` - Probability vector (must sum to 1, all elements >= 0)
    ///
    /// # Panics
    /// Panics if probs has fewer than 2 elements.
    pub fn new(n: usize, probs: Tensor<B, 1>) -> Self {
        let [k] = probs.dims();
        assert!(k >= 2, "Multinomial requires at least 2 categories");

        // Pre-compute log(n!)
        let log_n_factorial = ln_gamma((n + 1) as f64) as f32;

        // Pre-compute log(probs)
        let log_probs = probs.clone().log();

        Self {
            n,
            probs,
            dim: k,
            log_n_factorial,
            log_probs,
        }
    }

    /// Create a Multinomial with uniform probabilities.
    ///
    /// # Arguments
    /// * `n` - Total number of trials
    /// * `k` - Number of categories
    /// * `device` - Device to create tensors on
    pub fn uniform(n: usize, k: usize, device: &B::Device) -> Self {
        let prob = 1.0 / k as f32;
        let probs = Tensor::from_floats(vec![prob; k].as_slice(), device);
        Self::new(n, probs)
    }

    /// Get the number of categories.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the number of trials.
    pub fn num_trials(&self) -> usize {
        self.n
    }

    /// Compute the log probability of an observation.
    ///
    /// # Arguments
    /// * `x` - Count vector (non-negative integers summing to n, as floats)
    ///
    /// # Returns
    /// Scalar tensor containing log P(x | n, p)
    pub fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Multinomial distribution:
        // log P(x | n, p) = log(n!) - ∑_k log(x_k!) + ∑_k x_k * log(p_k)
        //
        // Note: x_k are treated as floats but should be non-negative integers

        let device = x.device();

        // Get counts as a vector
        let counts: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        // Compute ∑ log(x_k!)
        let sum_log_factorials: f32 = counts
            .iter()
            .map(|&c| ln_gamma((c as f64) + 1.0) as f32)
            .sum();

        // Compute ∑ x_k * log(p_k)
        let weighted_log_probs = x.clone() * self.log_probs.clone();
        let sum_weighted: f32 = weighted_log_probs.sum().into_scalar().elem();

        // log P(x | n, p) = log(n!) - ∑ log(x_k!) + ∑ x_k * log(p_k)
        let log_prob = self.log_n_factorial - sum_log_factorials + sum_weighted;

        Tensor::from_floats([log_prob], &device)
    }

    /// Get the support of the distribution.
    ///
    /// Note: Returns NonNegativeInteger as an approximation.
    /// The true support is the set of K-dimensional non-negative integer vectors
    /// that sum to n.
    pub fn support(&self) -> Support {
        Support::NonNegativeInteger
    }
}

/// Compute the log multinomial coefficient: log(n! / (∏_k x_k!))
///
/// This is a utility function that can be used independently.
pub fn log_multinomial_coefficient(n: usize, counts: &[usize]) -> f64 {
    let log_n_factorial = ln_gamma((n + 1) as f64);
    let sum_log_factorials: f64 = counts.iter().map(|&c| ln_gamma((c + 1) as f64)).sum();
    log_n_factorial - sum_log_factorials
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_multinomial_binomial_equivalence() {
        let device = Default::default();

        // Multinomial with K=2 should be equivalent to Binomial
        // Binomial(n=10, p=0.3) at k=3 successes
        let probs = Tensor::<TestBackend, 1>::from_floats([0.3, 0.7], &device);
        let multinomial = Multinomial::new(10, probs);

        let x = Tensor::from_floats([3.0, 7.0], &device);
        let log_prob: f32 = multinomial.log_prob(&x).into_scalar().elem();

        // Binomial log_prob: log(C(10,3)) + 3*log(0.3) + 7*log(0.7)
        // C(10,3) = 120
        let expected = 120.0_f32.ln() + 3.0 * 0.3_f32.ln() + 7.0 * 0.7_f32.ln();

        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Multinomial([0.3, 0.7]) should match Binomial(0.3), expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_multinomial_uniform() {
        let device = Default::default();

        // Uniform probabilities [1/3, 1/3, 1/3]
        let multinomial = Multinomial::<TestBackend>::uniform(6, 3, &device);

        let x = Tensor::from_floats([2.0, 2.0, 2.0], &device);
        let log_prob: f32 = multinomial.log_prob(&x).into_scalar().elem();

        // log P = log(6!/(2!2!2!)) + 2*log(1/3) + 2*log(1/3) + 2*log(1/3)
        //       = log(90) + 6*log(1/3)
        //       = log(90) - 6*log(3)
        let expected = 90.0_f32.ln() - 6.0 * 3.0_f32.ln();

        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_multinomial_mode() {
        let device = Default::default();

        // Skewed probabilities [0.6, 0.3, 0.1]
        let probs = Tensor::<TestBackend, 1>::from_floats([0.6, 0.3, 0.1], &device);
        let multinomial = Multinomial::new(10, probs);

        // Most likely outcome should have more counts in category 1
        let x_expected = Tensor::from_floats([6.0, 3.0, 1.0], &device);
        let x_unlikely = Tensor::from_floats([1.0, 3.0, 6.0], &device);

        let log_prob_expected: f32 = multinomial.log_prob(&x_expected).into_scalar().elem();
        let log_prob_unlikely: f32 = multinomial.log_prob(&x_unlikely).into_scalar().elem();

        assert!(
            log_prob_expected > log_prob_unlikely,
            "Outcome matching probabilities should be more likely"
        );
    }

    #[test]
    fn test_multinomial_dimension() {
        let device = Default::default();
        let multinomial = Multinomial::<TestBackend>::uniform(10, 5, &device);
        assert_eq!(multinomial.dim(), 5);
    }

    #[test]
    fn test_multinomial_num_trials() {
        let device = Default::default();
        let multinomial = Multinomial::<TestBackend>::uniform(15, 3, &device);
        assert_eq!(multinomial.num_trials(), 15);
    }

    #[test]
    fn test_log_multinomial_coefficient() {
        // C(6, [2,2,2]) = 6!/(2!2!2!) = 720/8 = 90
        let log_coef = log_multinomial_coefficient(6, &[2, 2, 2]);
        let expected = 90.0_f64.ln();
        assert!(
            (log_coef - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            log_coef
        );

        // C(5, [2,3]) = 5!/(2!3!) = 120/12 = 10
        let log_coef2 = log_multinomial_coefficient(5, &[2, 3]);
        let expected2 = 10.0_f64.ln();
        assert!(
            (log_coef2 - expected2).abs() < 1e-10,
            "Expected {}, got {}",
            expected2,
            log_coef2
        );
    }

    #[test]
    fn test_multinomial_support() {
        let device = Default::default();
        let multinomial = Multinomial::<TestBackend>::uniform(10, 3, &device);
        assert_eq!(multinomial.support(), Support::NonNegativeInteger);
    }

    #[test]
    #[should_panic(expected = "Multinomial requires at least 2 categories")]
    fn test_multinomial_requires_at_least_2_categories() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let probs = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let _ = Multinomial::new(10, probs);
    }
}
