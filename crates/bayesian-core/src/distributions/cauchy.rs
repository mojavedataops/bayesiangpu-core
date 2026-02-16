//! Cauchy distribution
//!
//! The Cauchy distribution is a heavy-tailed distribution often used as a
//! robust prior when outliers are expected.

use super::{Distribution, Support};
use burn::prelude::*;

/// Cauchy distribution
///
/// Also known as the Lorentz distribution. Has undefined mean and variance
/// due to heavy tails.
///
/// # Parameters
/// - `loc`: Location parameter (median)
/// - `scale`: Scale parameter (half-width at half-maximum)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::cauchy::Cauchy;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Cauchy::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Cauchy<B: Backend> {
    /// Location parameter (median)
    pub loc: Tensor<B, 1>,
    /// Scale parameter
    pub scale: Tensor<B, 1>,
    /// Pre-computed: -log(pi) - log(scale)
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> Cauchy<B> {
    /// Create a new Cauchy distribution
    ///
    /// # Arguments
    /// * `loc` - Location parameter (median)
    /// * `scale` - Scale parameter (must be positive)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        let log_pi = std::f64::consts::PI.ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(-log_pi);

        Self {
            loc,
            scale,
            log_normalizer,
        }
    }

    /// Create a standard Cauchy distribution (loc=0, scale=1)
    pub fn standard(device: &B::Device) -> Self {
        let loc = Tensor::zeros([1], device);
        let scale = Tensor::ones([1], device);
        Self::new(loc, scale)
    }
}

impl<B: Backend> Distribution<B> for Cauchy<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Cauchy distribution:
        // log p(x | loc, scale) = -log(pi) - log(scale) - log(1 + ((x - loc)/scale)^2)
        //
        // Breaking it down:
        // 1. log_normalizer = -log(pi) - log(scale)  [pre-computed]
        // 2. z = (x - loc) / scale
        // 3. log_prob = log_normalizer - log(1 + z^2)

        let z = (x.clone() - self.loc.clone()) / self.scale.clone();

        // log(1 + z^2)
        let log_kernel = z.powf_scalar(2.0).add_scalar(1.0).log();

        self.log_normalizer.clone() - log_kernel
    }

    fn support(&self) -> Support {
        Support::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_cauchy_at_zero() {
        let device = Default::default();
        let dist = Cauchy::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // log(1/pi) = -log(pi) ≈ -1.1447
        let result: f32 = log_prob.into_scalar().elem();
        let expected = -std::f64::consts::PI.ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_cauchy_at_one() {
        let device = Default::default();
        let dist = Cauchy::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // log(1/(pi * (1 + 1))) = -log(2*pi) ≈ -1.8379
        let result: f32 = log_prob.into_scalar().elem();
        let expected = -(2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_cauchy_with_params() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0f32], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([3.0f32], &device);
        let dist = Cauchy::new(loc, scale);

        // At x=loc, z=0, so density is 1/(pi*scale)
        let x = Tensor::from_floats([2.0f32], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_scalar().elem();
        // log(1/(3*pi))
        let expected = -(3.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_cauchy_symmetry() {
        let device = Default::default();
        let dist = Cauchy::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([-2.0f32, 0.0, 2.0], &device);
        let log_prob = dist.log_prob(&x);

        let results: Vec<f32> = log_prob.into_data().to_vec().unwrap();

        // Symmetric around 0
        assert!(
            (results[0] - results[2]).abs() < 1e-6,
            "log_prob(-2) should equal log_prob(2)"
        );

        // Maximum at median
        assert!(
            results[1] > results[0],
            "log_prob(0) should be greater than log_prob(-2)"
        );
    }

    #[test]
    fn test_cauchy_heavy_tails() {
        let device = Default::default();
        let dist = Cauchy::<TestBackend>::standard(&device);

        // Compare tail behavior with normal
        // Cauchy has heavier tails (slower decay)
        let x_near = Tensor::from_floats([1.0f32], &device);
        let x_far = Tensor::from_floats([10.0f32], &device);

        let log_prob_near = dist.log_prob(&x_near).into_scalar().elem::<f32>();
        let log_prob_far = dist.log_prob(&x_far).into_scalar().elem::<f32>();

        // The ratio should decrease slowly (heavy tails)
        // For Cauchy: log_prob(10) - log_prob(1) ≈ -log(101) + log(2) ≈ -3.92
        // For Normal: would be much more negative (exponential decay)
        let log_ratio = log_prob_far - log_prob_near;
        assert!(
            log_ratio > -5.0,
            "Cauchy should have heavy tails, got log_ratio: {}",
            log_ratio
        );
    }

    #[test]
    fn test_cauchy_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = Cauchy::<TestBackend>::standard(&device);
        assert_eq!(dist.support(), Support::Real);
    }
}
