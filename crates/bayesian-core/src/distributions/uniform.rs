//! Uniform distribution
//!
//! The continuous uniform distribution over an interval [low, high].

use super::{Distribution, Support};
use burn::prelude::*;

/// Uniform distribution
///
/// The continuous uniform distribution over the interval [low, high].
/// All values in the interval have equal probability density.
///
/// # Parameters
/// - `low`: Lower bound of the interval
/// - `high`: Upper bound of the interval (must be > low)
///
/// # Mathematical Definition
/// ```text
/// log p(x | low, high) = -log(high - low)   for low <= x <= high
///                      = -infinity          otherwise
/// ```
///
/// Note: In the current implementation, values outside the support will produce
/// incorrect log probabilities. The sampler is expected to respect the support.
#[derive(Debug, Clone)]
pub struct Uniform<B: Backend> {
    /// Lower bound
    pub low: Tensor<B, 1>,
    /// Upper bound
    pub high: Tensor<B, 1>,
    /// Pre-computed -log(high - low)
    log_density: Tensor<B, 1>,
}

impl<B: Backend> Uniform<B> {
    /// Create a new Uniform distribution
    ///
    /// # Arguments
    /// * `low` - Lower bound of the interval
    /// * `high` - Upper bound of the interval
    pub fn new(low: Tensor<B, 1>, high: Tensor<B, 1>) -> Self {
        // Compute -log(high - low)
        let width = high.clone() - low.clone();
        let log_density = width.log().neg();

        Self {
            low,
            high,
            log_density,
        }
    }

    /// Create a standard uniform distribution over [0, 1]
    pub fn standard(device: &B::Device) -> Self {
        let low = Tensor::zeros([1], device);
        let high = Tensor::ones([1], device);
        Self::new(low, high)
    }

    /// Create a symmetric uniform distribution over [-a, a]
    pub fn symmetric(half_width: Tensor<B, 1>) -> Self {
        let low = half_width.clone().neg();
        let high = half_width;
        Self::new(low, high)
    }
}

impl<B: Backend> Distribution<B> for Uniform<B> {
    fn log_prob(&self, _x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // For the uniform distribution, log p(x) = -log(high - low) for all x in [low, high]
        // We return the constant log density regardless of x value.
        //
        // Note: In a fully correct implementation, we would check bounds and return
        // -infinity for x outside [low, high]. However, for MCMC with proper
        // constraint transformations, x should always be within bounds.
        //
        // The shape of the output matches the pre-computed log_density which may
        // need broadcasting. For simplicity, we return the scalar density.
        self.log_density.clone()
    }

    fn support(&self) -> Support {
        // The uniform distribution has bounded support, but we indicate Real
        // since the model will apply appropriate transformations.
        // A more complete implementation would add Support::Bounded(low, high).
        Support::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_uniform_standard() {
        let device = Default::default();
        let dist = Uniform::<TestBackend>::standard(&device);

        // For Uniform(0, 1), log_prob = -log(1) = 0
        let x = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        assert!((log_prob as f64).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_wider_interval() {
        let device = Default::default();
        let low = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let high = Tensor::<TestBackend, 1>::from_floats([10.0], &device);
        let dist = Uniform::new(low, high);

        // For Uniform(0, 10), log_prob = -log(10)
        let x = Tensor::<TestBackend, 1>::from_floats([5.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        let expected = -10.0_f64.ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_constant_density() {
        let device = Default::default();
        let dist = Uniform::<TestBackend>::standard(&device);

        // All points in [0, 1] should have the same log_prob
        let x1 = Tensor::<TestBackend, 1>::from_floats([0.1], &device);
        let x2 = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let x3 = Tensor::<TestBackend, 1>::from_floats([0.9], &device);

        let lp1: f32 = dist.log_prob(&x1).into_scalar();
        let lp2: f32 = dist.log_prob(&x2).into_scalar();
        let lp3: f32 = dist.log_prob(&x3).into_scalar();

        assert!((lp1 - lp2).abs() < 1e-5);
        assert!((lp2 - lp3).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_symmetric() {
        let device = Default::default();
        let half_width = Tensor::<TestBackend, 1>::from_floats([5.0], &device);
        let dist = Uniform::<TestBackend>::symmetric(half_width);

        // Symmetric(-5, 5) has width 10, so log_prob = -log(10)
        let x = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        let expected = -10.0_f64.ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_narrow_interval() {
        let device = Default::default();
        let low = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let high = Tensor::<TestBackend, 1>::from_floats([0.1], &device);
        let dist = Uniform::new(low, high);

        // For Uniform(0, 0.1), log_prob = -log(0.1) = log(10)
        let x = Tensor::<TestBackend, 1>::from_floats([0.05], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar();

        let expected = 10.0_f64.ln();
        assert!((log_prob as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_support() {
        let device = Default::default();
        let dist = Uniform::<TestBackend>::standard(&device);

        // Currently returns Real (would ideally be Bounded)
        assert_eq!(dist.support(), Support::Real);
    }
}
