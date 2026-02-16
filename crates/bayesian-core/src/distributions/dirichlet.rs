//! Dirichlet distribution
//!
//! The Dirichlet distribution is a multivariate generalization of the Beta distribution.
//! It is parameterized by a concentration vector α and is defined on the simplex
//! (vectors that sum to 1).

use super::Support;
use crate::math::ln_gamma;
use burn::prelude::*;

/// Dirichlet distribution
///
/// The Dirichlet distribution is the conjugate prior for the Multinomial distribution
/// and is commonly used in topic modeling (LDA) and Bayesian categorical models.
///
/// f(x | α) = (1/B(α)) * ∏_k x_k^(α_k - 1)
///
/// where B(α) = ∏_k Γ(α_k) / Γ(∑_k α_k)
///
/// # Parameters
/// - `concentration` (α): Concentration vector (all elements must be positive)
///
/// # Support
/// The simplex: x_k ∈ (0, 1) and ∑_k x_k = 1
///
/// # Special Cases
/// - α = [1, 1, ..., 1]: Uniform distribution over the simplex
/// - α = [a, a, ..., a] with a > 1: Concentrated toward the center
/// - α = [a, a, ..., a] with a < 1: Concentrated toward the corners
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::dirichlet::Dirichlet;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // Uniform on 3-simplex: Dirichlet([1, 1, 1])
/// let alpha = Tensor::<B, 1>::from_floats([1.0, 1.0, 1.0], &device);
/// let dist = Dirichlet::new(alpha);
///
/// // A point on the simplex
/// let x = Tensor::<B, 1>::from_floats([0.2, 0.3, 0.5], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Dirichlet<B: Backend> {
    /// Concentration parameter α (vector of positive values)
    pub concentration: Tensor<B, 1>,
    /// Dimension K (number of categories)
    dim: usize,
    /// Pre-computed: log(Γ(∑_k α_k)) - ∑_k log(Γ(α_k))
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: α - 1 (for log_prob computation)
    alpha_minus_one: Tensor<B, 1>,
}

impl<B: Backend> Dirichlet<B> {
    /// Create a new Dirichlet distribution.
    ///
    /// # Arguments
    /// * `concentration` - Concentration parameter vector (all elements > 0)
    ///
    /// # Panics
    /// Panics if concentration has fewer than 2 elements.
    pub fn new(concentration: Tensor<B, 1>) -> Self {
        let [k] = concentration.dims();
        assert!(k >= 2, "Dirichlet requires at least 2 categories");

        let device = concentration.device();
        let alpha_data: Vec<f32> = concentration.clone().into_data().to_vec().unwrap();

        // Compute log normalizer: log(Γ(∑α)) - ∑log(Γ(α_k))
        let alpha_sum: f64 = alpha_data.iter().map(|&a| a as f64).sum();
        let ln_gamma_sum = ln_gamma(alpha_sum);
        let ln_gamma_parts: f64 = alpha_data.iter().map(|&a| ln_gamma(a as f64)).sum();
        let log_norm = (ln_gamma_sum - ln_gamma_parts) as f32;

        let log_normalizer = Tensor::<B, 1>::from_floats([log_norm], &device);
        let alpha_minus_one = concentration.clone().sub_scalar(1.0);

        Self {
            concentration,
            dim: k,
            log_normalizer,
            alpha_minus_one,
        }
    }

    /// Create a symmetric Dirichlet distribution with all concentrations equal.
    ///
    /// # Arguments
    /// * `k` - Number of categories
    /// * `alpha` - Concentration value (same for all categories)
    /// * `device` - Device to create tensors on
    pub fn symmetric(k: usize, alpha: f32, device: &B::Device) -> Self {
        let concentration = Tensor::from_floats(vec![alpha; k].as_slice(), device);
        Self::new(concentration)
    }

    /// Create a uniform Dirichlet distribution (all α = 1).
    ///
    /// # Arguments
    /// * `k` - Number of categories
    /// * `device` - Device to create tensors on
    pub fn uniform(k: usize, device: &B::Device) -> Self {
        Self::symmetric(k, 1.0, device)
    }

    /// Get the dimension (number of categories).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute the log probability of an observation.
    ///
    /// # Arguments
    /// * `x` - A point on the simplex (vector of shape [K] with elements summing to 1)
    ///
    /// # Returns
    /// Scalar tensor containing log p(x | α)
    pub fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Dirichlet distribution:
        // log p(x | α) = log(Γ(∑α)) - ∑log(Γ(α_k)) + ∑(α_k - 1)*log(x_k)
        //
        // = log_normalizer + ∑(α_k - 1)*log(x_k)

        let log_x = x.clone().log();

        // Element-wise (α - 1) * log(x), then sum
        let weighted = self.alpha_minus_one.clone() * log_x;
        let sum_term = weighted.sum().reshape([1]);

        self.log_normalizer.clone() + sum_term
    }

    /// Get the support of the distribution.
    pub fn support(&self) -> Support {
        Support::Simplex(self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dirichlet_uniform() {
        let device = Default::default();
        let dirichlet = Dirichlet::<TestBackend>::uniform(3, &device);

        // For uniform Dirichlet([1,1,1]) over 3-simplex,
        // log_prob should be log(2!) = log(2) ≈ 0.693 (constant)
        // because the normalizing constant is B([1,1,1]) = Γ(1)^3 / Γ(3) = 1/2
        let x = Tensor::from_floats([0.2, 0.3, 0.5], &device);
        let log_prob: f32 = dirichlet.log_prob(&x).into_scalar().elem();

        // Expected: log(Γ(3)) - 3*log(Γ(1)) = log(2) - 0 = log(2)
        let expected = 2.0_f32.ln();
        assert!(
            (log_prob - expected).abs() < 1e-5,
            "Uniform Dirichlet should have constant log_prob = log(2), got {}",
            log_prob
        );
    }

    #[test]
    fn test_dirichlet_at_center() {
        let device = Default::default();

        // Symmetric Dirichlet([2, 2, 2])
        let dirichlet = Dirichlet::<TestBackend>::symmetric(3, 2.0, &device);

        // At center (1/3, 1/3, 1/3)
        let x_center = Tensor::from_floats([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &device);
        let log_prob_center: f32 = dirichlet.log_prob(&x_center).into_scalar().elem();

        // At a corner-ish point (0.8, 0.1, 0.1)
        let x_corner = Tensor::from_floats([0.8, 0.1, 0.1], &device);
        let log_prob_corner: f32 = dirichlet.log_prob(&x_corner).into_scalar().elem();

        // For α > 1, center should have higher density than corners
        assert!(
            log_prob_center > log_prob_corner,
            "Symmetric Dirichlet with α > 1 should have mode at center"
        );
    }

    #[test]
    fn test_dirichlet_concentrated_at_corners() {
        let device = Default::default();

        // Symmetric Dirichlet([0.5, 0.5, 0.5])
        let dirichlet = Dirichlet::<TestBackend>::symmetric(3, 0.5, &device);

        // At center (1/3, 1/3, 1/3)
        let x_center = Tensor::from_floats([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &device);
        let log_prob_center: f32 = dirichlet.log_prob(&x_center).into_scalar().elem();

        // At a corner-ish point (0.9, 0.05, 0.05)
        let x_corner = Tensor::from_floats([0.9, 0.05, 0.05], &device);
        let log_prob_corner: f32 = dirichlet.log_prob(&x_corner).into_scalar().elem();

        // For α < 1, corners should have higher density than center
        assert!(
            log_prob_corner > log_prob_center,
            "Symmetric Dirichlet with α < 1 should concentrate at corners"
        );
    }

    #[test]
    fn test_dirichlet_beta_equivalence() {
        let device = Default::default();

        // Dirichlet([2, 3]) should be equivalent to Beta(2, 3)
        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 3.0], &device);
        let dirichlet = Dirichlet::new(alpha);

        // Test at x = 0.4 (so [0.4, 0.6] on the simplex)
        let x = Tensor::from_floats([0.4, 0.6], &device);
        let log_prob_dirichlet: f32 = dirichlet.log_prob(&x).into_scalar().elem();

        // Compare with Beta(2, 3) at 0.4
        // Beta log_prob = (α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))
        // = 1*log(0.4) + 2*log(0.6) - log(B(2,3))
        // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1*2/24 = 1/12
        let expected = 1.0 * 0.4_f32.ln() + 2.0 * 0.6_f32.ln() - (1.0 / 12.0_f32).ln();
        assert!(
            (log_prob_dirichlet - expected).abs() < 1e-4,
            "Dirichlet([2,3]) should match Beta(2,3), expected {}, got {}",
            expected,
            log_prob_dirichlet
        );
    }

    #[test]
    fn test_dirichlet_dimension() {
        let device = Default::default();
        let dirichlet = Dirichlet::<TestBackend>::uniform(5, &device);
        assert_eq!(dirichlet.dim(), 5);
    }

    #[test]
    fn test_dirichlet_support() {
        let device = Default::default();
        let dirichlet = Dirichlet::<TestBackend>::uniform(4, &device);
        assert_eq!(dirichlet.support(), Support::Simplex(4));
    }

    #[test]
    fn test_dirichlet_symmetric_constructor() {
        let device = Default::default();

        let sym = Dirichlet::<TestBackend>::symmetric(3, 2.0, &device);

        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0, 2.0], &device);
        let explicit = Dirichlet::new(alpha);

        let x = Tensor::from_floats([0.3, 0.3, 0.4], &device);
        let log_prob_sym: f32 = sym.log_prob(&x).into_scalar().elem();
        let log_prob_exp: f32 = explicit.log_prob(&x).into_scalar().elem();

        assert!(
            (log_prob_sym - log_prob_exp).abs() < 1e-5,
            "Symmetric constructor should match explicit"
        );
    }

    #[test]
    #[should_panic(expected = "Dirichlet requires at least 2 categories")]
    fn test_dirichlet_requires_at_least_2_categories() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let alpha = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let _ = Dirichlet::new(alpha);
    }
}
