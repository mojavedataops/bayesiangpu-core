//! Gumbel distribution (Type-I extreme value distribution)
//!
//! The Gumbel distribution is parameterized by location (loc) and scale.
//! It is used to model the distribution of the maximum (or minimum) of
//! a number of samples of various distributions.

use super::{Distribution, Support};
use burn::prelude::*;

/// Gumbel distribution
///
/// # Parameters
/// - `loc`: Location parameter (mu, real)
/// - `scale`: Scale parameter (beta > 0)
///
/// # Mathematical Definition
/// ```text
/// log f(x | mu, beta) = -z - exp(-z) - ln(beta)   where z = (x - mu) / beta
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::gumbel::Gumbel;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Gumbel::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Gumbel<B: Backend> {
    /// Location parameter (mu)
    pub loc: Tensor<B, 1>,
    /// Scale parameter (beta)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: -ln(beta)
    neg_log_scale: Tensor<B, 1>,
}

impl<B: Backend> Gumbel<B> {
    /// Create a new Gumbel distribution
    ///
    /// # Arguments
    /// * `loc` - Location parameter (mu)
    /// * `scale` - Scale parameter (beta > 0)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: -ln(beta)
        let neg_log_scale = scale.clone().log().neg();

        Self {
            loc,
            scale,
            neg_log_scale,
        }
    }
}

impl<B: Backend> Distribution<B> for Gumbel<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(x | mu, beta) = -z - exp(-z) - ln(beta)
        // where z = (x - mu) / beta
        //
        // Breaking it down:
        // 1. neg_log_scale = -ln(beta)  [pre-computed]
        // 2. z = (x - mu) / beta
        // 3. -z - exp(-z)

        let z = (x.clone() - self.loc.clone()) / self.scale.clone();

        // -z - exp(-z)
        let neg_z = z.clone().neg();
        let exp_neg_z = neg_z.clone().exp();

        self.neg_log_scale.clone() + neg_z - exp_neg_z
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
    fn test_gumbel_at_mode() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Gumbel::new(loc, scale);

        // The mode of Gumbel(0, 1) is at x=0
        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At x=0, z=0: -0 - exp(0) - ln(1) = -1
        let expected = -1.0_f64;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_gumbel_known_value() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Gumbel::new(loc, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At x=1, z=1: -1 - exp(-1) - 0 = -1 - 0.3679 = -1.3679
        let expected = -1.0 - (-1.0_f64).exp();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_gumbel_with_loc_scale() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Gumbel::new(loc, scale);

        // At mode x=loc=2, z=0: -0 - exp(0) - ln(3) = -1 - ln(3)
        let x = Tensor::from_floats([2.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = -1.0 - 3.0_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_gumbel_vectorized() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Gumbel::new(loc, scale);

        let x = Tensor::from_floats([-2.0f32, -1.0, 0.0, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);

        // Mode at 0 should have highest density
        assert!(log_probs[2] > log_probs[0], "log_prob(0) > log_prob(-2)");
        assert!(log_probs[2] > log_probs[4], "log_prob(0) > log_prob(2)");
    }

    #[test]
    fn test_gumbel_asymmetry() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Gumbel::new(loc, scale);

        // Gumbel is right-skewed (NOT symmetric)
        let x = Tensor::from_floats([-1.0f32, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // The Gumbel distribution has a heavier right tail
        // log_prob(-1) != log_prob(1) in general
        assert!(
            (log_probs[0] - log_probs[1]).abs() > 0.01,
            "Gumbel should be asymmetric"
        );
    }

    #[test]
    fn test_gumbel_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Gumbel::new(loc, scale);
        assert_eq!(dist.support(), Support::Real);
    }
}
