//! Weibull distribution
//!
//! The Weibull distribution is parameterized by shape (k) and scale (lambda).
//! It is widely used in reliability engineering and survival analysis.

use super::{Distribution, Support};
use burn::prelude::*;

/// Weibull distribution
///
/// # Parameters
/// - `shape`: Shape parameter (k > 0)
/// - `scale`: Scale parameter (lambda > 0)
///
/// # Mathematical Definition
/// ```text
/// log f(x | k, lambda) = ln(k/lambda) + (k-1)*ln(x/lambda) - (x/lambda)^k   for x > 0
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::weibull::Weibull;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let shape = Tensor::<B, 1>::from_floats([2.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Weibull::new(shape, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.5, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Weibull<B: Backend> {
    /// Shape parameter (k)
    pub shape: Tensor<B, 1>,
    /// Scale parameter (lambda)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: ln(k) - ln(lambda)
    log_shape_over_scale: Tensor<B, 1>,
    /// Pre-computed: k - 1
    shape_minus_one: Tensor<B, 1>,
}

impl<B: Backend> Weibull<B> {
    /// Create a new Weibull distribution
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (k > 0)
    /// * `scale` - Scale parameter (lambda > 0)
    pub fn new(shape: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: ln(k) - ln(lambda)
        let log_shape_over_scale = shape.clone().log() - scale.clone().log();
        let shape_minus_one = shape.clone().sub_scalar(1.0);

        Self {
            shape,
            scale,
            log_shape_over_scale,
            shape_minus_one,
        }
    }
}

impl<B: Backend> Distribution<B> for Weibull<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(x | k, lambda) = ln(k/lambda) + (k-1)*ln(x/lambda) - (x/lambda)^k
        //
        // Breaking it down:
        // 1. log_shape_over_scale = ln(k) - ln(lambda)  [pre-computed]
        // 2. (k-1) * ln(x/lambda)
        // 3. -(x/lambda)^k

        let x_over_scale = x.clone() / self.scale.clone();
        let log_x_over_scale = x_over_scale.clone().log();

        // (k-1) * ln(x/lambda)
        let power_term = self.shape_minus_one.clone() * log_x_over_scale;

        // -(x/lambda)^k
        let exp_term = x_over_scale.powf(self.shape.clone()).neg();

        self.log_shape_over_scale.clone() + power_term + exp_term
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
    fn test_weibull_known_value() {
        let device = Default::default();

        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Weibull::new(shape, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At x=1, k=2, lambda=1:
        // ln(2/1) + (2-1)*ln(1/1) - (1/1)^2 = ln(2) + 0 - 1 = ln(2) - 1
        let expected = 2.0_f64.ln() - 1.0;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_weibull_shape_one_equals_exponential() {
        // Weibull(shape=1, scale=lambda) should equal Exponential(rate=1/lambda)
        let device = Default::default();

        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = Weibull::new(shape, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // Exponential(rate=1/2) at x=1: ln(0.5) - 0.5*1 = -ln(2) - 0.5
        let expected = -(2.0_f64).ln() - 0.5;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Weibull(1, lambda) should equal Exponential(1/lambda). Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_weibull_monotonicity() {
        let device = Default::default();

        // With shape > 1, Weibull is unimodal with mode at lambda*((k-1)/k)^(1/k)
        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Weibull::new(shape, scale);

        // Mode for k=3, lambda=1: ((3-1)/3)^(1/3) ~ 0.874
        // Values after the mode should decrease
        let x = Tensor::from_floats([1.0f32, 2.0, 3.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // After the mode, density should decrease
        assert!(
            log_probs[0] > log_probs[1],
            "log_prob(1) > log_prob(2) for Weibull(3,1)"
        );
        assert!(
            log_probs[1] > log_probs[2],
            "log_prob(2) > log_prob(3) for Weibull(3,1)"
        );
    }

    #[test]
    fn test_weibull_with_scale() {
        let device = Default::default();

        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Weibull::new(shape, scale);

        let x = Tensor::from_floats([1.5f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // ln(2/3) + (2-1)*ln(1.5/3) - (1.5/3)^2
        // = ln(2/3) + ln(0.5) - 0.25
        let expected = (2.0_f64 / 3.0).ln() + 0.5_f64.ln() - 0.25;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_weibull_vectorized() {
        let device = Default::default();

        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Weibull::new(shape, scale);

        let x = Tensor::from_floats([0.5f32, 1.0, 1.5, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 4);
    }

    #[test]
    fn test_weibull_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Weibull::new(shape, scale);
        assert_eq!(dist.support(), Support::Positive);
    }
}
