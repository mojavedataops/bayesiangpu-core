//! Half-Cauchy distribution
//!
//! The half-Cauchy distribution is a Cauchy distribution truncated to positive values.
//! It is commonly used as a prior for scale parameters, especially in hierarchical models.

use super::{Distribution, Support};
use burn::prelude::*;

/// Half-Cauchy distribution
///
/// The half-Cauchy is a Cauchy distribution restricted to x > 0.
/// It has heavier tails than the half-normal and is widely recommended
/// as a prior for scale parameters in hierarchical models.
///
/// # Parameters
/// - `scale`: Scale parameter (must be positive)
///
/// # Mathematical Definition
/// ```text
/// log p(x | scale) = log(2) - log(pi) - log(scale) - log(1 + (x/scale)^2)   for x > 0
/// ```
#[derive(Debug, Clone)]
pub struct HalfCauchy<B: Backend> {
    /// Scale parameter
    pub scale: Tensor<B, 1>,
    /// Pre-computed log normalizer: log(2) - log(pi) - log(scale)
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> HalfCauchy<B> {
    /// Create a new HalfCauchy distribution
    ///
    /// # Arguments
    /// * `scale` - Scale parameter (must be positive)
    pub fn new(scale: Tensor<B, 1>) -> Self {
        let log_norm_const = 2.0_f64.ln() - std::f64::consts::PI.ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(log_norm_const);

        Self {
            scale,
            log_normalizer,
        }
    }

    /// Create a standard half-Cauchy distribution (scale=1)
    pub fn standard(device: &B::Device) -> Self {
        let scale = Tensor::ones([1], device);
        Self::new(scale)
    }
}

impl<B: Backend> Distribution<B> for HalfCauchy<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log p(x | scale) = log(2) - log(pi) - log(scale) - log(1 + (x/scale)^2)
        //
        // For x <= 0, the log probability should be -infinity, but we don't
        // explicitly handle this as the sampler should only propose positive values.
        // The constraint is handled by the Support::Positive type.

        let z = x.clone() / self.scale.clone();
        let log_kernel = z.powf_scalar(2.0).add_scalar(1.0).log().neg();

        self.log_normalizer.clone() + log_kernel
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
    fn test_half_cauchy_at_zero() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfCauchy::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=0: log(2) - log(pi) - log(1) - log(1 + 0) = log(2/pi)
        let expected = (2.0_f64 / std::f64::consts::PI).ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_cauchy_at_one() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfCauchy::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, scale=1: log(2) - log(pi) - log(1 + 1) = log(2/pi) - log(2) = -log(pi)
        let expected = 2.0_f64.ln() - std::f64::consts::PI.ln() - 2.0_f64.ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_cauchy_scaling() {
        let device = Default::default();

        // With scale=5, the distribution is wider
        let scale = Tensor::<TestBackend, 1>::from_floats([5.0], &device);
        let dist = HalfCauchy::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        // At x=1, scale=5: log(2) - log(pi) - log(5) - log(1 + (1/5)^2)
        let expected =
            2.0_f64.ln() - std::f64::consts::PI.ln() - 5.0_f64.ln() - (1.0 + 0.04_f64).ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_half_cauchy_support() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfCauchy::new(scale);

        assert_eq!(dist.support(), Support::Positive);
    }

    #[test]
    fn test_half_cauchy_batched() {
        let device = Default::default();
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let dist = HalfCauchy::new(scale);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 2.0], &device);
        let log_probs = dist.log_prob(&x);

        // Check shape
        assert_eq!(log_probs.dims(), [3]);

        // Check that log_prob decreases as x increases from 0
        let data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        assert!(data[0] > data[1]);
        assert!(data[1] > data[2]);
    }

    #[test]
    fn test_half_cauchy_standard() {
        let device = Default::default();
        let dist = HalfCauchy::<TestBackend>::standard(&device);

        let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        let expected = (2.0_f64 / std::f64::consts::PI).ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }
}
