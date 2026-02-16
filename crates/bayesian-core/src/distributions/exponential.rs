//! Exponential distribution
//!
//! The exponential distribution with rate parameter lambda.
//! Commonly used for modeling waiting times.

use super::{Distribution, Support};
use burn::prelude::*;

/// Exponential distribution
///
/// The exponential distribution is parameterized by the rate (lambda).
/// It has support on (0, infinity).
///
/// # Parameters
/// - `rate`: Rate parameter lambda (must be positive)
///
/// # Mathematical Definition
/// ```text
/// log p(x | λ) = log(λ) - λx   for x > 0
/// ```
///
/// The mean is 1/lambda and the variance is 1/lambda^2.
#[derive(Debug, Clone)]
pub struct Exponential<B: Backend> {
    /// Rate parameter (lambda)
    pub rate: Tensor<B, 1>,
    /// Pre-computed log(rate)
    log_rate: Tensor<B, 1>,
}

impl<B: Backend> Exponential<B> {
    /// Create a new Exponential distribution
    ///
    /// # Arguments
    /// * `rate` - Rate parameter (lambda, must be positive)
    pub fn new(rate: Tensor<B, 1>) -> Self {
        let log_rate = rate.clone().log();
        Self { rate, log_rate }
    }

    /// Create an exponential distribution with rate=1 (standard exponential)
    pub fn standard(device: &B::Device) -> Self {
        let rate = Tensor::ones([1], device);
        Self::new(rate)
    }

    /// Create from scale (mean) parameter instead of rate
    ///
    /// # Arguments
    /// * `scale` - Scale parameter (mean = 1/rate)
    pub fn from_scale(scale: Tensor<B, 1>) -> Self {
        // rate = 1 / scale
        let rate = scale.recip();
        Self::new(rate)
    }
}

impl<B: Backend> Distribution<B> for Exponential<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log p(x | λ) = log(λ) - λx
        //
        // For x <= 0, the log probability should be -infinity, but we don't
        // explicitly handle this as the sampler should only propose positive values.

        // -λx term
        let linear = (x.clone() * self.rate.clone()).neg();

        // log(λ) - λx
        self.log_rate.clone() + linear
    }

    fn support(&self) -> Support {
        Support::Positive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_exponential_at_zero() {
        let device = Default::default();
        let rate = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = Exponential::new(rate);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=0, lambda=1: log(1) - 0 = 0
        assert!((log_prob as f64).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_at_one() {
        let device = Default::default();
        let rate = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = Exponential::new(rate);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, lambda=1: log(1) - 1 = -1
        assert!((log_prob as f64 - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_different_rates() {
        let device = Default::default();

        // rate = 2 means faster decay
        let rate = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
        let dist = Exponential::new(rate);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, lambda=2: log(2) - 2 ≈ -1.307
        let expected = 2.0_f64.ln() - 2.0;
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_from_scale() {
        let device = Default::default();

        // scale=0.5 means rate=2
        let scale = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let dist = Exponential::from_scale(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // Should be same as rate=2
        let expected = 2.0_f64.ln() - 2.0;
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_support() {
        let device = Default::default();
        let rate = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = Exponential::new(rate);

        assert_eq!(dist.support(), Support::Positive);
    }

    #[test]
    fn test_exponential_batched() {
        let device = Default::default();
        let rate = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = Exponential::new(rate);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 2.0], &device);
        let log_probs = dist.log_prob(&x);

        // Check shape
        assert_eq!(log_probs.dims(), [3]);

        // Log probs should be 0, -1, -2
        let data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        assert!((data[0] as f64).abs() < 1e-5);
        assert!((data[1] as f64 - (-1.0)).abs() < 1e-5);
        assert!((data[2] as f64 - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_decreasing() {
        let device = Default::default();
        let rate = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = Exponential::new(rate);

        // Exponential density is decreasing for x > 0
        let x = Tensor::<TestBackend, 1>::from_floats([0.1, 0.5, 1.0, 2.0], &device);
        let log_probs = dist.log_prob(&x);

        let data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        for i in 0..data.len() - 1 {
            assert!(data[i] > data[i + 1], "Exponential should be decreasing");
        }
    }
}
