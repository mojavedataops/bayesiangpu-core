//! Laplace (Double Exponential) distribution
//!
//! The Laplace distribution is parameterized by location (mu) and scale (b).

use super::{Distribution, Support};
use burn::prelude::*;

/// Laplace (Double Exponential) distribution
///
/// # Parameters
/// - `loc`: Location parameter (mu)
/// - `scale`: Scale parameter (b > 0)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::laplace::Laplace;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Laplace::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Laplace<B: Backend> {
    /// Location parameter (mu)
    pub loc: Tensor<B, 1>,
    /// Scale parameter (b)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: -ln(2) - ln(b)
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> Laplace<B> {
    /// Create a new Laplace distribution
    ///
    /// # Arguments
    /// * `loc` - Location parameter (mu)
    /// * `scale` - Scale parameter (b > 0)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: -ln(2) - ln(b)
        let neg_ln2 = -(2.0_f64).ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(neg_ln2);

        Self {
            loc,
            scale,
            log_normalizer,
        }
    }
}

impl<B: Backend> Distribution<B> for Laplace<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Laplace distribution:
        // log p(x | mu, b) = -ln(2b) - |x - mu| / b
        //
        // Breaking it down:
        // 1. log_normalizer = -ln(2) - ln(b)  [pre-computed]
        // 2. -|x - mu| / b

        let abs_dev = (x.clone() - self.loc.clone()).abs();
        let scaled_dev = abs_dev / self.scale.clone();

        self.log_normalizer.clone() - scaled_dev
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
    fn test_laplace_at_mode() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Laplace::new(loc, scale);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At mode x=0: log(1/(2*1)) = -ln(2) ~ -0.6931
        let expected = -(2.0_f64).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_laplace_at_known_value() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Laplace::new(loc, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // log(1/(2*1) * exp(-|1|/1)) = -ln(2) - 1 ~ -1.6931
        let expected = -(2.0_f64).ln() - 1.0;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_laplace_vectorized() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Laplace::new(loc, scale);

        let x = Tensor::from_floats([-2.0f32, -1.0, 0.0, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Should be 5 results
        assert_eq!(log_probs.len(), 5);

        // Mode at 0 should have highest density
        assert!(log_probs[2] > log_probs[1], "log_prob(0) > log_prob(-1)");
        assert!(log_probs[1] > log_probs[0], "log_prob(-1) > log_prob(-2)");
    }

    #[test]
    fn test_laplace_symmetry() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = Laplace::new(loc, scale);

        let x = Tensor::from_floats([1.0f32, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Symmetric around loc=3: log_prob(1) should equal log_prob(5)
        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-6,
            "log_prob(1) should equal log_prob(5) for Laplace(3, 2)"
        );
    }

    #[test]
    fn test_laplace_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Laplace::new(loc, scale);
        assert_eq!(dist.support(), Support::Real);
    }
}
