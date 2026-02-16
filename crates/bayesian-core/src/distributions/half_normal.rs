//! Half-Normal distribution
//!
//! The half-normal distribution is a truncated normal distribution with support on (0, inf).
//! It is commonly used as a prior for scale parameters.

use super::{Distribution, Support};
use burn::prelude::*;

/// Half-Normal distribution
///
/// The half-normal is a normal distribution truncated to positive values.
/// It has support on (0, infinity) and is useful as a prior for scale parameters.
///
/// # Parameters
/// - `scale`: Scale parameter (sigma, must be positive)
///
/// # Mathematical Definition
/// ```text
/// log p(x | σ) = log(2) - 0.5*log(2πσ²) - x²/(2σ²)   for x > 0
///              = log(√(2/π)) - log(σ) - x²/(2σ²)
/// ```
#[derive(Debug, Clone)]
pub struct HalfNormal<B: Backend> {
    /// Scale parameter (sigma)
    pub scale: Tensor<B, 1>,
    /// Pre-computed log normalizer: log(sqrt(2/pi)) - log(sigma)
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> HalfNormal<B> {
    /// Create a new HalfNormal distribution
    ///
    /// # Arguments
    /// * `scale` - Scale parameter (standard deviation of the underlying normal)
    pub fn new(scale: Tensor<B, 1>) -> Self {
        // log(sqrt(2/pi)) = 0.5 * (log(2) - log(pi))
        let log_norm_const = 0.5 * (2.0_f64.ln() - std::f64::consts::PI.ln());
        let log_normalizer = scale.clone().log().neg().add_scalar(log_norm_const);

        Self {
            scale,
            log_normalizer,
        }
    }

    /// Create a standard half-normal distribution (scale=1)
    pub fn standard(device: &B::Device) -> Self {
        let scale = Tensor::ones([1], device);
        Self::new(scale)
    }
}

impl<B: Backend> Distribution<B> for HalfNormal<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log p(x | σ) = log(√(2/π)) - log(σ) - x²/(2σ²)
        //
        // For x <= 0, the log probability should be -infinity, but we don't
        // explicitly handle this as the sampler should only propose positive values.
        // The constraint is handled by the Support::Positive type.

        // Compute -x²/(2σ²)
        let x_over_sigma = x.clone() / self.scale.clone();
        let quadratic = x_over_sigma.powf_scalar(2.0).mul_scalar(-0.5);

        self.log_normalizer.clone() + quadratic
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
    fn test_half_normal_at_zero() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfNormal::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=0: log(sqrt(2/pi)) ≈ -0.2257
        let expected = 0.5 * (2.0_f64.ln() - std::f64::consts::PI.ln());
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_normal_at_one() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfNormal::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, sigma=1: log(sqrt(2/pi)) - 0.5 ≈ -0.7257
        let expected = 0.5 * (2.0_f64.ln() - std::f64::consts::PI.ln()) - 0.5;
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_normal_scaling() {
        let device = Default::default();

        // With scale=2, the distribution is wider
        let scale = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
        let dist = HalfNormal::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, sigma=2: log(sqrt(2/pi)) - log(2) - 0.125
        let expected = 0.5 * (2.0_f64.ln() - std::f64::consts::PI.ln()) - 2.0_f64.ln() - 0.125;
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_normal_support() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfNormal::new(scale);

        assert_eq!(dist.support(), Support::Positive);
    }

    #[test]
    fn test_half_normal_batched() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfNormal::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 2.0], &device);
        let log_probs = dist.log_prob(&x);

        // Check shape
        assert_eq!(log_probs.dims(), [3]);

        // Check that log_prob decreases as x increases from 0
        let data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        assert!(data[0] > data[1]);
        assert!(data[1] > data[2]);
    }
}
