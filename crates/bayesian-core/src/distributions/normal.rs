//! Normal (Gaussian) distribution
//!
//! The normal distribution is parameterized by location (mean) and scale (standard deviation).

use super::{Distribution, Support};
use burn::prelude::*;

/// Normal (Gaussian) distribution
///
/// # Parameters
/// - `loc`: Mean of the distribution
/// - `scale`: Standard deviation of the distribution (must be positive)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::normal::Normal;
/// use burn::prelude::*;
/// use burn::backend::Wgpu;
///
/// type B = Wgpu;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Normal::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Normal<B: Backend> {
    /// Location (mean) parameter
    pub loc: Tensor<B, 1>,
    /// Scale (standard deviation) parameter
    pub scale: Tensor<B, 1>,
    /// Pre-computed log normalizer: -0.5 * log(2 * pi) - log(sigma)
    /// Caching this avoids recomputing log(scale) on every log_prob call
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> Normal<B> {
    /// Create a new Normal distribution
    ///
    /// # Arguments
    /// * `loc` - Mean of the distribution
    /// * `scale` - Standard deviation of the distribution
    ///
    /// # Panics
    /// Panics if scale contains non-positive values (in debug mode)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute the log normalizer: -0.5 * log(2 * pi) - log(sigma)
        // This avoids recomputing log(scale) on every log_prob call
        let log_norm_const = -0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(log_norm_const);

        Self {
            loc,
            scale,
            log_normalizer,
        }
    }

    /// Create a standard normal distribution (mean=0, std=1)
    pub fn standard(device: &B::Device) -> Self {
        let loc = Tensor::zeros([1], device);
        let scale = Tensor::ones([1], device);
        Self::new(loc, scale)
    }
}

impl<B: Backend> Distribution<B> for Normal<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Normal distribution:
        // log p(x | mu, sigma) = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
        //
        // Breaking it down:
        // 1. log_normalizer = -0.5 * log(2*pi) - log(sigma)  [pre-computed]
        // 2. z = (x - mu) / sigma  (standardized value)
        // 3. log_prob = log_normalizer - 0.5 * z^2

        // Compute the standardized values: z = (x - mu) / sigma
        // Only clone x and loc once, then chain operations to avoid extra clones
        let z = (x.clone() - self.loc.clone()) / self.scale.clone();

        // Quadratic term: -0.5 * z^2
        // Use powf_scalar to avoid cloning z for multiply
        let quadratic = z.powf_scalar(2.0).mul_scalar(-0.5);

        // Total log probability using pre-computed log_normalizer
        self.log_normalizer.clone() + quadratic
    }

    fn support(&self) -> Support {
        Support::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_normal_standard_at_zero() {
        let device = Default::default();
        let dist = Normal::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // log(1/sqrt(2*pi)) = -0.5 * log(2*pi) ≈ -0.9189
        let result: f32 = log_prob.into_scalar().elem();
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_normal_standard_at_one() {
        let device = Default::default();
        let dist = Normal::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // log(exp(-0.5) / sqrt(2*pi)) = -0.5 - 0.5*log(2*pi) ≈ -1.4189
        let result: f32 = log_prob.into_scalar().elem();
        let expected = -0.5 - 0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_normal_with_params() {
        let device = NdArrayDevice::default();

        // Normal(2, 3) at x=2 should give same as standard normal at x=0
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0f32], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([3.0f32], &device);
        let dist = Normal::new(loc, scale);

        let x: Tensor<TestBackend, 1> = Tensor::from_floats([2.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // log(1/(3*sqrt(2*pi))) = -log(3) - 0.5*log(2*pi) ≈ -2.0171
        let result: f32 = log_prob.into_data().to_vec().unwrap()[0];
        let expected = -(3.0_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_normal_vectorized() {
        let device = Default::default();
        let dist = Normal::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([-1.0f32, 0.0, 1.0], &device);
        let log_prob = dist.log_prob(&x);

        let results: Vec<f32> = log_prob.into_data().to_vec().unwrap();

        // Symmetric around 0
        assert!(
            (results[0] - results[2]).abs() < 1e-6,
            "log_prob(-1) should equal log_prob(1)"
        );

        // Maximum at mean
        assert!(
            results[1] > results[0],
            "log_prob(0) should be greater than log_prob(-1)"
        );
    }

    #[test]
    fn test_normal_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = Normal::<TestBackend>::standard(&device);
        assert_eq!(dist.support(), Support::Real);
    }
}
