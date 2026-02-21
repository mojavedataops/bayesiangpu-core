//! Zero-Inflated Poisson distribution
//!
//! The Zero-Inflated Poisson distribution models count data with excess zeros.
//! It is a mixture of a point mass at zero and a Poisson distribution.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Zero-Inflated Poisson distribution
///
/// # Parameters
/// - `rate`: Poisson rate parameter (lambda > 0)
/// - `zero_prob`: Probability of structural zero (0 <= pi < 1)
///
/// # Mathematical Definition
/// ```text
/// For k = 0: log f(0) = log(pi + (1-pi)*exp(-rate))
/// For k > 0: log f(k) = log(1-pi) + k*log(rate) - rate - ln_gamma(k+1)
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::zero_inflated_poisson::ZeroInflatedPoisson;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let rate = Tensor::<B, 1>::from_floats([3.0], &device);
/// let zero_prob = Tensor::<B, 1>::from_floats([0.3], &device);
/// let dist = ZeroInflatedPoisson::new(rate, zero_prob);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct ZeroInflatedPoisson<B: Backend> {
    /// Poisson rate parameter (lambda)
    pub rate: Tensor<B, 1>,
    /// Probability of structural zero (pi)
    pub zero_prob: Tensor<B, 1>,
    /// Pre-computed: rate as f64
    #[allow(dead_code)]
    rate_f64: f64,
    /// Pre-computed: zero_prob as f64
    #[allow(dead_code)]
    zero_prob_f64: f64,
    /// Pre-computed: log(1 - pi)
    log_1_minus_pi: f64,
    /// Pre-computed: log(rate)
    log_rate: f64,
    /// Pre-computed: -rate
    neg_rate: f64,
    /// Pre-computed: log(pi + (1-pi)*exp(-rate)) for k=0 case
    log_prob_zero: f64,
}

impl<B: Backend> ZeroInflatedPoisson<B> {
    /// Create a new Zero-Inflated Poisson distribution
    ///
    /// # Arguments
    /// * `rate` - Poisson rate parameter (lambda > 0)
    /// * `zero_prob` - Probability of structural zero (0 <= pi < 1)
    pub fn new(rate: Tensor<B, 1>, zero_prob: Tensor<B, 1>) -> Self {
        let rate_f64: f64 = rate.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let zero_prob_f64: f64 = zero_prob.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;

        let log_1_minus_pi = (1.0 - zero_prob_f64).ln();
        let log_rate = rate_f64.ln();
        let neg_rate = -rate_f64;

        // For k=0: log(pi + (1-pi)*exp(-rate))
        let log_prob_zero = (zero_prob_f64 + (1.0 - zero_prob_f64) * (-rate_f64).exp()).ln();

        Self {
            rate,
            zero_prob,
            rate_f64,
            zero_prob_f64,
            log_1_minus_pi,
            log_rate,
            neg_rate,
            log_prob_zero,
        }
    }
}

impl<B: Backend> Distribution<B> for ZeroInflatedPoisson<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();
        let k_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = k_values
            .iter()
            .map(|&k| {
                let k_f64 = k as f64;
                if k_f64 == 0.0 {
                    // log(pi + (1-pi)*exp(-rate))
                    self.log_prob_zero as f32
                } else {
                    // log(1-pi) + k*log(rate) - rate - ln_gamma(k+1)
                    (self.log_1_minus_pi + k_f64 * self.log_rate + self.neg_rate
                        - ln_gamma(k_f64 + 1.0)) as f32
                }
            })
            .collect();

        Tensor::from_floats(log_probs.as_slice(), &device)
    }

    fn support(&self) -> Support {
        Support::NonNegativeInteger
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_zip_zero_prob_zero_reduces_to_poisson() {
        let device = Default::default();

        // When zero_prob = 0, ZIP reduces to standard Poisson
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let dist = ZeroInflatedPoisson::new(rate, zero_prob);

        // Check k=0: should be exp(-3) = Poisson(0|3)
        let x0 = Tensor::from_floats([0.0f32], &device);
        let lp0: f32 = dist.log_prob(&x0).into_scalar().elem();
        let expected0 = -3.0_f64; // log(exp(-3)) = -3
        assert!(
            (lp0 as f64 - expected0).abs() < 1e-5,
            "Expected {}, got {}",
            expected0,
            lp0
        );

        // Check k=2: log(3^2 * exp(-3) / 2!) = 2*ln(3) - 3 - ln(2)
        let x2 = Tensor::from_floats([2.0f32], &device);
        let lp2: f32 = dist.log_prob(&x2).into_scalar().elem();
        let expected2 = 2.0 * 3.0_f64.ln() - 3.0 - ln_gamma(3.0);
        assert!(
            (lp2 as f64 - expected2).abs() < 1e-5,
            "Expected {}, got {}",
            expected2,
            lp2
        );
    }

    #[test]
    fn test_zip_high_zero_prob() {
        let device = Default::default();

        // With high zero_prob, most mass should be at 0
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.9], &device);
        let dist = ZeroInflatedPoisson::new(rate, zero_prob);

        let x = Tensor::from_floats([0.0f32, 1.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // k=0 should have much higher probability than k=1 or k=5
        assert!(
            log_probs[0] > log_probs[1],
            "k=0 ({}) should be more likely than k=1 ({})",
            log_probs[0],
            log_probs[1]
        );
        assert!(
            log_probs[0] > log_probs[2],
            "k=0 ({}) should be more likely than k=5 ({})",
            log_probs[0],
            log_probs[2]
        );
    }

    #[test]
    fn test_zip_known_values() {
        let device = Default::default();

        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.3], &device);
        let dist = ZeroInflatedPoisson::new(rate, zero_prob);

        // k=0: log(0.3 + 0.7 * exp(-2))
        let x0 = Tensor::from_floats([0.0f32], &device);
        let lp0: f32 = dist.log_prob(&x0).into_scalar().elem();
        let expected0 = (0.3 + 0.7 * (-2.0_f64).exp()).ln();
        assert!(
            (lp0 as f64 - expected0).abs() < 1e-5,
            "k=0: Expected {}, got {}",
            expected0,
            lp0
        );

        // k=3: log(0.7) + 3*log(2) - 2 - ln_gamma(4)
        let x3 = Tensor::from_floats([3.0f32], &device);
        let lp3: f32 = dist.log_prob(&x3).into_scalar().elem();
        let expected3 = 0.7_f64.ln() + 3.0 * 2.0_f64.ln() - 2.0 - ln_gamma(4.0);
        assert!(
            (lp3 as f64 - expected3).abs() < 1e-5,
            "k=3: Expected {}, got {}",
            expected3,
            lp3
        );
    }

    #[test]
    fn test_zip_vectorized() {
        let device = Default::default();

        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([2.5], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.2], &device);
        let dist = ZeroInflatedPoisson::new(rate, zero_prob);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);
    }

    #[test]
    fn test_zip_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.2], &device);
        let dist = ZeroInflatedPoisson::new(rate, zero_prob);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
