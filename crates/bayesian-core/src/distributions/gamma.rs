//! Gamma distribution
//!
//! The gamma distribution is parameterized by shape (alpha/concentration) and rate (beta).

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Gamma distribution
///
/// Uses the rate parameterization: f(x) = β^α / Γ(α) * x^(α-1) * e^(-βx)
///
/// # Parameters
/// - `concentration` (α): Shape parameter (must be positive)
/// - `rate` (β): Rate parameter (must be positive)
///
/// # Alternative Parameterization
/// Some software uses scale (θ = 1/β) instead of rate. Use `from_shape_scale()` for that.
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::gamma::Gamma;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // Exponential(1) is same as Gamma(1, 1)
/// let concentration = Tensor::<B, 1>::from_floats([1.0], &device);
/// let rate = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Gamma::new(concentration, rate);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Gamma<B: Backend> {
    /// Shape (concentration) parameter α
    pub concentration: Tensor<B, 1>,
    /// Rate parameter β
    pub rate: Tensor<B, 1>,
    /// Pre-computed: α*log(β) - log(Γ(α))
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: α - 1 (for log_prob)
    alpha_minus_one: Tensor<B, 1>,
}

impl<B: Backend> Gamma<B> {
    /// Create a new Gamma distribution with shape and rate parameterization.
    ///
    /// # Arguments
    /// * `concentration` - Shape parameter (α > 0)
    /// * `rate` - Rate parameter (β > 0)
    pub fn new(concentration: Tensor<B, 1>, rate: Tensor<B, 1>) -> Self {
        // Pre-compute: α*log(β) - log(Γ(α))
        let alpha_data: Vec<f32> = concentration.clone().into_data().to_vec().unwrap();
        let ln_gamma_alpha: Vec<f32> = alpha_data
            .iter()
            .map(|&a| ln_gamma(a as f64) as f32)
            .collect();
        let device = concentration.device();

        let ln_gamma_tensor = Tensor::from_floats(ln_gamma_alpha.as_slice(), &device);
        let log_normalizer = concentration.clone() * rate.clone().log() - ln_gamma_tensor;

        let alpha_minus_one = concentration.clone().sub_scalar(1.0);

        Self {
            concentration,
            rate,
            log_normalizer,
            alpha_minus_one,
        }
    }

    /// Create a Gamma distribution from shape (α) and scale (θ = 1/β) parameters.
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (α > 0)
    /// * `scale` - Scale parameter (θ > 0, where rate β = 1/θ)
    pub fn from_shape_scale(shape: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // rate = 1 / scale
        let one = Tensor::ones_like(&scale);
        let rate = one / scale;
        Self::new(shape, rate)
    }

    /// Create an exponential distribution as a special case of Gamma.
    ///
    /// Exponential(λ) = Gamma(1, λ)
    pub fn exponential(rate: Tensor<B, 1>) -> Self {
        let device = rate.device();
        let concentration = Tensor::ones([1], &device);
        Self::new(concentration, rate)
    }
}

impl<B: Backend> Distribution<B> for Gamma<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Gamma distribution:
        // log p(x | α, β) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - β*x
        //
        // Breaking it down:
        // 1. log_normalizer = α*log(β) - log(Γ(α))  [pre-computed]
        // 2. (α-1)*log(x)
        // 3. -β*x

        let log_x = x.clone().log();
        let shape_term = self.alpha_minus_one.clone() * log_x;
        let rate_term = self.rate.clone() * x.clone();

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
    fn test_gamma_exponential_equivalence() {
        let device = Default::default();

        // Gamma(1, 1) should equal Exponential(1)
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let gamma = Gamma::new(conc, rate);

        let x = Tensor::from_floats([1.0], &device);
        let log_prob: f32 = gamma.log_prob(&x).into_scalar().elem();

        // Exponential(1) at x=1: log(exp(-1)) = -1
        let expected = -1.0_f32;
        assert!(
            (log_prob - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_gamma_at_mode() {
        let device = Default::default();

        // Gamma(3, 2): mode is at (α-1)/β = 2/2 = 1
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let gamma = Gamma::new(conc, rate);

        // At mode, x=1
        let x = Tensor::from_floats([1.0], &device);
        let log_prob: f32 = gamma.log_prob(&x).into_scalar().elem();

        // log_prob at mode for Gamma(3,2) at x=1:
        // 3*log(2) - log(Γ(3)) + 2*log(1) - 2*1
        // = 3*ln(2) - ln(2) + 0 - 2
        // = 2*ln(2) - 2 ≈ -0.614
        let expected = 2.0 * 2.0_f32.ln() - 2.0;
        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_gamma_from_shape_scale() {
        let device = Default::default();

        // Gamma with shape=2, scale=0.5 equals Gamma with shape=2, rate=2
        let shape: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let gamma_ss = Gamma::from_shape_scale(shape.clone(), scale);

        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let gamma_sr = Gamma::new(shape, rate);

        let x = Tensor::from_floats([1.0], &device);
        let log_prob_ss: f32 = gamma_ss.log_prob(&x).into_scalar().elem();
        let log_prob_sr: f32 = gamma_sr.log_prob(&x).into_scalar().elem();

        assert!(
            (log_prob_ss - log_prob_sr).abs() < 1e-5,
            "Shape-scale and shape-rate should give same result"
        );
    }

    #[test]
    fn test_gamma_decreasing_from_mode() {
        let device = Default::default();

        // Gamma(3, 1): mode is at α-1 = 2
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let gamma = Gamma::new(conc, rate);

        let x = Tensor::from_floats([1.0, 2.0, 3.0, 5.0], &device);
        let log_probs: Vec<f32> = gamma.log_prob(&x).into_data().to_vec().unwrap();

        // Mode at x=2 should have highest density
        assert!(log_probs[1] > log_probs[0], "log_prob(2) > log_prob(1)");
        assert!(log_probs[1] > log_probs[2], "log_prob(2) > log_prob(3)");
        assert!(log_probs[2] > log_probs[3], "log_prob(3) > log_prob(5)");
    }

    #[test]
    fn test_gamma_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let gamma = Gamma::new(conc, rate);
        assert_eq!(gamma.support(), Support::Positive);
    }
}
