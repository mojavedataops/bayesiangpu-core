//! Inverse Gamma distribution
//!
//! The inverse gamma distribution is parameterized by concentration (alpha) and scale (beta).
//! If X ~ Gamma(alpha, beta), then 1/X ~ InverseGamma(alpha, 1/beta).

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Inverse Gamma distribution
///
/// # Parameters
/// - `concentration` (alpha): Shape parameter (must be positive)
/// - `scale` (beta): Scale parameter (must be positive)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::inverse_gamma::InverseGamma;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let concentration = Tensor::<B, 1>::from_floats([2.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = InverseGamma::new(concentration, scale);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct InverseGamma<B: Backend> {
    /// Shape (concentration) parameter alpha
    pub concentration: Tensor<B, 1>,
    /// Scale parameter beta
    pub scale: Tensor<B, 1>,
    /// Pre-computed: alpha*ln(beta) - ln(Gamma(alpha))
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: -(alpha + 1)
    neg_alpha_minus_one: Tensor<B, 1>,
}

impl<B: Backend> InverseGamma<B> {
    /// Create a new Inverse Gamma distribution
    ///
    /// # Arguments
    /// * `concentration` - Shape parameter (alpha > 0)
    /// * `scale` - Scale parameter (beta > 0)
    pub fn new(concentration: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: alpha*ln(beta) - ln(Gamma(alpha))
        let alpha_data: Vec<f32> = concentration.clone().into_data().to_vec().unwrap();
        let ln_gamma_alpha: Vec<f32> = alpha_data
            .iter()
            .map(|&a| ln_gamma(a as f64) as f32)
            .collect();
        let device = concentration.device();

        let ln_gamma_tensor = Tensor::from_floats(ln_gamma_alpha.as_slice(), &device);
        let log_normalizer = concentration.clone() * scale.clone().log() - ln_gamma_tensor;

        // Pre-compute: -(alpha + 1)
        let neg_alpha_minus_one = concentration.clone().add_scalar(1.0).neg();

        Self {
            concentration,
            scale,
            log_normalizer,
            neg_alpha_minus_one,
        }
    }
}

impl<B: Backend> Distribution<B> for InverseGamma<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Inverse Gamma distribution:
        // log p(x | alpha, beta) = alpha*ln(beta) - ln(Gamma(alpha))
        //                          + (-(alpha+1))*ln(x) - beta/x
        //
        // Breaking it down:
        // 1. log_normalizer = alpha*ln(beta) - ln(Gamma(alpha))  [pre-computed]
        // 2. neg_alpha_minus_one * ln(x)
        // 3. -beta / x

        let log_x = x.clone().log();
        let shape_term = self.neg_alpha_minus_one.clone() * log_x;
        let rate_term = self.scale.clone() / x.clone();

        self.log_normalizer.clone() + shape_term - rate_term
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
    fn test_inverse_gamma_at_mode() {
        let device = Default::default();

        // InverseGamma(3, 2): mode is at beta/(alpha+1) = 2/4 = 0.5
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = InverseGamma::new(conc, scale);

        let x = Tensor::from_floats([0.5f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // log p(0.5 | 3, 2) = 3*ln(2) - ln(Gamma(3)) + (-4)*ln(0.5) - 2/0.5
        // = 3*ln(2) - ln(2) + 4*ln(2) - 4
        // = 6*ln(2) - 4
        let expected = 6.0 * (2.0_f64).ln() - 4.0;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_inverse_gamma_at_one() {
        let device = Default::default();

        // InverseGamma(2, 1) at x=1
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = InverseGamma::new(conc, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // log p(1 | 2, 1) = 2*ln(1) - ln(Gamma(2)) + (-3)*ln(1) - 1/1
        // = 0 - 0 + 0 - 1 = -1
        let expected = -1.0_f64;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_inverse_gamma_vectorized() {
        let device = Default::default();

        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = InverseGamma::new(conc, scale);

        // Mode at beta/(alpha+1) = 0.5
        let x = Tensor::from_floats([0.25f32, 0.5, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 4);

        // Mode at 0.5 should have highest density
        assert!(
            log_probs[1] > log_probs[0],
            "log_prob(0.5) > log_prob(0.25)"
        );
        assert!(log_probs[1] > log_probs[2], "log_prob(0.5) > log_prob(1)");
        assert!(log_probs[2] > log_probs[3], "log_prob(1) > log_prob(2)");
    }

    #[test]
    fn test_inverse_gamma_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = InverseGamma::new(conc, scale);
        assert_eq!(dist.support(), Support::Positive);
    }

    #[test]
    fn test_inverse_gamma_monotonicity() {
        let device = Default::default();

        // InverseGamma(1, 1) has mode at 1/2 = 0.5
        // For x > mode, density should decrease
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = InverseGamma::new(conc, scale);

        let x = Tensor::from_floats([1.0f32, 2.0, 5.0, 10.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // For x > mode, should be monotonically decreasing
        assert!(log_probs[0] > log_probs[1], "log_prob(1) > log_prob(2)");
        assert!(log_probs[1] > log_probs[2], "log_prob(2) > log_prob(5)");
        assert!(log_probs[2] > log_probs[3], "log_prob(5) > log_prob(10)");
    }
}
