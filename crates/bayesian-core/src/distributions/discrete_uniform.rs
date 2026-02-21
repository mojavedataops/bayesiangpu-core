//! Discrete Uniform distribution
//!
//! The Discrete Uniform distribution assigns equal probability to each integer
//! in the range [low, high].

use super::{Distribution, Support};
use burn::prelude::*;

/// Discrete Uniform distribution
///
/// # Parameters
/// - `low`: Lower bound (integer)
/// - `high`: Upper bound (integer), low <= high
///
/// # Mathematical Definition
/// ```text
/// log f(k | low, high) = -ln(high - low + 1)   for low <= k <= high
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::discrete_uniform::DiscreteUniform;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let dist = DiscreteUniform::<B>::new(1, 6, &device);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 3.0, 6.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct DiscreteUniform<B: Backend> {
    /// Lower bound
    pub low: i64,
    /// Upper bound
    pub high: i64,
    /// Pre-computed: -ln(high - low + 1) as a tensor
    log_prob_value: Tensor<B, 1>,
}

impl<B: Backend> DiscreteUniform<B> {
    /// Create a new Discrete Uniform distribution
    ///
    /// # Arguments
    /// * `low` - Lower bound (integer)
    /// * `high` - Upper bound (integer), must be >= low
    /// * `device` - Device to create tensors on
    pub fn new(low: i64, high: i64, device: &B::Device) -> Self {
        assert!(high >= low, "high ({}) must be >= low ({})", high, low);

        let n = (high - low + 1) as f32;
        let log_prob_val = -n.ln();
        let log_prob_value = Tensor::from_floats([log_prob_val], device);

        Self {
            low,
            high,
            log_prob_value,
        }
    }
}

impl<B: Backend> Distribution<B> for DiscreteUniform<B> {
    fn log_prob(&self, _x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // All values in support have equal probability: -ln(high - low + 1)
        // We return a constant tensor (broadcast to match x shape if needed)
        self.log_prob_value.clone()
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
    fn test_discrete_uniform_die() {
        let device = Default::default();

        // Fair 6-sided die: DiscreteUniform(1, 6)
        let dist = DiscreteUniform::<TestBackend>::new(1, 6, &device);

        let x = Tensor::from_floats([3.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // -ln(6)
        let expected = -(6.0_f64).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_discrete_uniform_single_value() {
        let device = Default::default();

        // DiscreteUniform(5, 5) has all mass at 5
        let dist = DiscreteUniform::<TestBackend>::new(5, 5, &device);

        let x = Tensor::from_floats([5.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // -ln(1) = 0
        assert!(
            (log_prob as f64).abs() < 1e-5,
            "Expected 0.0, got {}",
            log_prob
        );
    }

    #[test]
    fn test_discrete_uniform_all_equal() {
        let device = Default::default();

        let dist = DiscreteUniform::<TestBackend>::new(0, 9, &device);

        // All values in support should have equal log probability
        let x1 = Tensor::from_floats([0.0f32], &device);
        let x2 = Tensor::from_floats([5.0f32], &device);
        let x3 = Tensor::from_floats([9.0f32], &device);

        let lp1: f32 = dist.log_prob(&x1).into_scalar().elem();
        let lp2: f32 = dist.log_prob(&x2).into_scalar().elem();
        let lp3: f32 = dist.log_prob(&x3).into_scalar().elem();

        assert!(
            (lp1 - lp2).abs() < 1e-6,
            "All values should have equal log_prob"
        );
        assert!(
            (lp2 - lp3).abs() < 1e-6,
            "All values should have equal log_prob"
        );
    }

    #[test]
    fn test_discrete_uniform_known_value() {
        let device = Default::default();

        // DiscreteUniform(0, 3): 4 values, each with probability 1/4
        let dist = DiscreteUniform::<TestBackend>::new(0, 3, &device);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = -(4.0_f64).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_discrete_uniform_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = DiscreteUniform::<TestBackend>::new(1, 6, &device);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }

    #[test]
    #[should_panic(expected = "high (0) must be >= low (5)")]
    fn test_discrete_uniform_invalid_bounds() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = DiscreteUniform::<TestBackend>::new(5, 0, &device);
    }
}
