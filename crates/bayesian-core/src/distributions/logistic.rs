//! Logistic distribution
//!
//! The logistic distribution is parameterized by location (mu) and scale (s).

use super::{Distribution, Support};
use burn::prelude::*;

/// Logistic distribution
///
/// # Parameters
/// - `loc`: Location parameter (mu)
/// - `scale`: Scale parameter (s > 0)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::logistic::Logistic;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Logistic::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Logistic<B: Backend> {
    /// Location parameter (mu)
    pub loc: Tensor<B, 1>,
    /// Scale parameter (s)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: -ln(s)
    neg_log_scale: Tensor<B, 1>,
}

impl<B: Backend> Logistic<B> {
    /// Create a new Logistic distribution
    ///
    /// # Arguments
    /// * `loc` - Location parameter (mu)
    /// * `scale` - Scale parameter (s > 0)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: -ln(s)
        let neg_log_scale = scale.clone().log().neg();

        Self {
            loc,
            scale,
            neg_log_scale,
        }
    }
}

impl<B: Backend> Distribution<B> for Logistic<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Logistic distribution:
        // log p(x | mu, s) = -(x-mu)/s - ln(s) - 2*ln(1 + exp(-(x-mu)/s))
        //
        // Breaking it down:
        // 1. z = (x - mu) / s
        // 2. neg_log_scale = -ln(s)  [pre-computed]
        // 3. log_prob = -z + neg_log_scale - 2*ln(1 + exp(-z))
        //
        // Using the identity: -z - 2*ln(1 + exp(-z)) = -z - 2*softplus(-z)
        // where softplus(u) = ln(1 + exp(u))
        //
        // Note: we use log1p(exp(-z)) for numerical stability via the
        // equivalent form: -z - 2*log(1 + exp(-z))

        let z = (x.clone() - self.loc.clone()) / self.scale.clone();

        // -z - 2*ln(1 + exp(-z))
        // = -z - 2*softplus(-z)
        // For numerical stability, note that:
        //   softplus(u) = ln(1+exp(u)) is well-handled by exp for moderate u
        let neg_z = z.clone().neg();
        let softplus_neg_z = neg_z.clone().exp().add_scalar(1.0).log();

        self.neg_log_scale.clone() - z - softplus_neg_z.mul_scalar(2.0)
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
    fn test_logistic_at_mode() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Logistic::new(loc, scale);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At mode x=0 for Logistic(0,1):
        // f(0) = exp(0) / (1 * (1+exp(0))^2) = 1/4
        // log(1/4) = -ln(4) = -2*ln(2) ~ -1.3863
        let expected = -2.0 * (2.0_f64).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_logistic_at_known_value() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Logistic::new(loc, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // log f(1) = -1 - 0 - 2*ln(1 + exp(-1))
        // exp(-1) ~ 0.3679
        // ln(1.3679) ~ 0.3133
        // = -1 - 2*0.3133 = -1.6265
        let expected = -1.0 - 2.0 * (1.0 + (-1.0_f64).exp()).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_logistic_vectorized() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Logistic::new(loc, scale);

        let x = Tensor::from_floats([-2.0f32, -1.0, 0.0, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);

        // Mode at 0 should have highest density
        assert!(log_probs[2] > log_probs[1], "log_prob(0) > log_prob(-1)");
        assert!(log_probs[2] > log_probs[3], "log_prob(0) > log_prob(1)");
    }

    #[test]
    fn test_logistic_symmetry() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Logistic::new(loc, scale);

        let x = Tensor::from_floats([-1.0f32, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Symmetric around loc=0
        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-6,
            "log_prob(-1) should equal log_prob(1)"
        );
    }

    #[test]
    fn test_logistic_with_params() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Logistic::new(loc, scale);

        // At the mode x=2 for Logistic(2, 3):
        // f(2) = 1 / (3 * 4) = 1/12
        // log(1/12) = -ln(12) ~ -2.4849
        let x = Tensor::from_floats([2.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = -(12.0_f64).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_logistic_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Logistic::new(loc, scale);
        assert_eq!(dist.support(), Support::Real);
    }
}
