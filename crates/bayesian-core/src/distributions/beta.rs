//! Beta distribution
//!
//! The beta distribution is parameterized by two shape parameters (α, β) and
//! is defined on the unit interval [0, 1].

use super::{Distribution, Support};
use crate::math::ln_beta;
use burn::prelude::*;

/// Beta distribution
///
/// The beta distribution is commonly used as a prior for probabilities.
/// f(x) = x^(α-1) * (1-x)^(β-1) / B(α, β)
///
/// # Parameters
/// - `concentration1` (α): First shape parameter (must be positive)
/// - `concentration0` (β): Second shape parameter (must be positive)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::beta::Beta;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // Uniform on [0, 1] is Beta(1, 1)
/// let alpha = Tensor::<B, 1>::from_floats([1.0], &device);
/// let beta = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Beta::new(alpha, beta);
///
/// let x = Tensor::<B, 1>::from_floats([0.3, 0.5, 0.7], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Beta<B: Backend> {
    /// First shape parameter (α)
    pub concentration1: Tensor<B, 1>,
    /// Second shape parameter (β)
    pub concentration0: Tensor<B, 1>,
    /// Pre-computed: -log(B(α, β))
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: α - 1
    alpha_minus_one: Tensor<B, 1>,
    /// Pre-computed: β - 1
    beta_minus_one: Tensor<B, 1>,
}

impl<B: Backend> Beta<B> {
    /// Create a new Beta distribution.
    ///
    /// # Arguments
    /// * `concentration1` - First shape parameter (α > 0)
    /// * `concentration0` - Second shape parameter (β > 0)
    pub fn new(concentration1: Tensor<B, 1>, concentration0: Tensor<B, 1>) -> Self {
        // Pre-compute: -log(B(α, β)) = log(Γ(α+β)) - log(Γ(α)) - log(Γ(β))
        let alpha_data: Vec<f32> = concentration1.clone().into_data().to_vec().unwrap();
        let beta_data: Vec<f32> = concentration0.clone().into_data().to_vec().unwrap();

        let ln_beta_vals: Vec<f32> = alpha_data
            .iter()
            .zip(beta_data.iter())
            .map(|(&a, &b)| ln_beta(a as f64, b as f64) as f32)
            .collect();

        let device = concentration1.device();
        let ln_beta_tensor = Tensor::from_floats(ln_beta_vals.as_slice(), &device);
        let log_normalizer = ln_beta_tensor.neg();

        let alpha_minus_one = concentration1.clone().sub_scalar(1.0);
        let beta_minus_one = concentration0.clone().sub_scalar(1.0);

        Self {
            concentration1,
            concentration0,
            log_normalizer,
            alpha_minus_one,
            beta_minus_one,
        }
    }

    /// Create a uniform distribution on [0, 1], which is Beta(1, 1).
    pub fn uniform(device: &B::Device) -> Self {
        let alpha = Tensor::ones([1], device);
        let beta = Tensor::ones([1], device);
        Self::new(alpha, beta)
    }

    /// Create a symmetric beta distribution Beta(α, α).
    pub fn symmetric(concentration: Tensor<B, 1>) -> Self {
        let beta = concentration.clone();
        Self::new(concentration, beta)
    }
}

impl<B: Backend> Distribution<B> for Beta<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Beta distribution:
        // log p(x | α, β) = -log(B(α, β)) + (α-1)*log(x) + (β-1)*log(1-x)
        //
        // Breaking it down:
        // 1. log_normalizer = -log(B(α, β))  [pre-computed]
        // 2. (α-1)*log(x)
        // 3. (β-1)*log(1-x)

        let log_x = x.clone().log();
        let one = Tensor::ones_like(x);
        let log_one_minus_x = (one - x.clone()).log();

        let alpha_term = self.alpha_minus_one.clone() * log_x;
        let beta_term = self.beta_minus_one.clone() * log_one_minus_x;

        self.log_normalizer.clone() + alpha_term + beta_term
    }

    fn support(&self) -> Support {
        Support::UnitInterval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_beta_uniform() {
        let device = Default::default();
        let beta = Beta::<TestBackend>::uniform(&device);

        // Beta(1, 1) is uniform, so all x ∈ (0, 1) have log_prob = 0
        let x = Tensor::from_floats([0.3], &device);
        let log_prob: f32 = beta.log_prob(&x).into_scalar().elem();

        assert!(
            (log_prob - 0.0).abs() < 1e-5,
            "Beta(1,1) should have constant log_prob = 0, got {}",
            log_prob
        );
    }

    #[test]
    fn test_beta_at_half() {
        let device = Default::default();

        // Beta(2, 2) is symmetric with mode at 0.5
        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta_param: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta = Beta::new(alpha, beta_param);

        let x = Tensor::from_floats([0.5], &device);
        let log_prob: f32 = beta.log_prob(&x).into_scalar().elem();

        // Beta(2,2) at x=0.5: 6 * (0.5)^1 * (0.5)^1 = 6 * 0.25 = 1.5
        // log(1.5) ≈ 0.405
        let expected = 1.5_f32.ln();
        assert!(
            (log_prob - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_beta_symmetry() {
        let device = Default::default();

        // Beta(2, 2) is symmetric around 0.5
        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta_param: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta = Beta::new(alpha, beta_param);

        let x = Tensor::from_floats([0.3, 0.7], &device);
        let log_probs: Vec<f32> = beta.log_prob(&x).into_data().to_vec().unwrap();

        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-5,
            "Symmetric Beta should have equal density at 0.3 and 0.7"
        );
    }

    #[test]
    fn test_beta_mode() {
        let device = Default::default();

        // Beta(3, 2) has mode at (α-1)/(α+β-2) = 2/3
        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let beta_param: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta = Beta::new(alpha, beta_param);

        let x = Tensor::from_floats([0.5, 0.6667, 0.8], &device);
        let log_probs: Vec<f32> = beta.log_prob(&x).into_data().to_vec().unwrap();

        // Mode (x ≈ 0.667) should have highest density
        assert!(
            log_probs[1] > log_probs[0],
            "Mode should have higher density than 0.5"
        );
        assert!(
            log_probs[1] > log_probs[2],
            "Mode should have higher density than 0.8"
        );
    }

    #[test]
    fn test_beta_symmetric_constructor() {
        let device = Default::default();

        // Beta.symmetric(2) should equal Beta(2, 2)
        let conc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let symmetric = Beta::symmetric(conc);

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let beta_param: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let explicit = Beta::new(alpha, beta_param);

        let x = Tensor::from_floats([0.4], &device);
        let log_prob_sym: f32 = symmetric.log_prob(&x).into_scalar().elem();
        let log_prob_exp: f32 = explicit.log_prob(&x).into_scalar().elem();

        assert!(
            (log_prob_sym - log_prob_exp).abs() < 1e-5,
            "Symmetric constructor should match explicit"
        );
    }

    #[test]
    fn test_beta_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let beta = Beta::<TestBackend>::uniform(&device);
        assert_eq!(beta.support(), Support::UnitInterval);
    }
}
