//! Dirichlet-Multinomial distribution (Compound distribution)
//!
//! Also known as the Polya distribution or multivariate Polya distribution.
//! It models overdispersed count data where the category probabilities
//! themselves are uncertain and follow a Dirichlet distribution.

use super::Support;
use crate::math::ln_gamma;
use burn::prelude::*;

/// Dirichlet-Multinomial distribution
///
/// The Dirichlet-Multinomial is a compound distribution where:
/// 1. p ~ Dirichlet(α)
/// 2. x | p ~ Multinomial(n, p)
///
/// Marginalizing over p gives the Dirichlet-Multinomial distribution:
///
/// P(x | n, α) = n! / (∏_k x_k!) × (Γ(α_0) / Γ(n + α_0)) × ∏_k (Γ(x_k + α_k) / Γ(α_k))
///
/// where α_0 = ∑_k α_k is the concentration parameter.
///
/// # Parameters
/// - `n`: Total number of trials
/// - `concentration` (α): Dirichlet concentration vector (all elements > 0)
///
/// # Use Cases
/// - Overdispersed count data (more variance than Multinomial)
/// - Topic modeling (LDA)
/// - Population genetics
/// - When category probabilities are uncertain
///
/// # Special Cases
/// - As α → ∞, approaches Multinomial with p = α / α_0
/// - As α → 0, approaches a distribution concentrated on vertices
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::dirichlet_multinomial::DirichletMultinomial;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // 10 trials with uncertain probabilities (Dirichlet([2, 3, 5]))
/// let alpha = Tensor::<B, 1>::from_floats([2.0, 3.0, 5.0], &device);
/// let dist = DirichletMultinomial::new(10, alpha);
///
/// // Observed counts
/// let x = Tensor::<B, 1>::from_floats([2.0, 3.0, 5.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct DirichletMultinomial<B: Backend> {
    /// Total number of trials
    pub n: usize,
    /// Dirichlet concentration parameter α
    pub concentration: Tensor<B, 1>,
    /// Number of categories K
    dim: usize,
    /// Pre-computed: α_0 = ∑_k α_k (total concentration)
    alpha_sum: f64,
    /// Pre-computed: log(n!)
    log_n_factorial: f64,
    /// Pre-computed: log(Γ(α_0)) - log(Γ(n + α_0))
    log_normalizer_n: f64,
    /// Pre-computed: log(Γ(α_k)) for each k
    log_gamma_alpha: Vec<f64>,
}

impl<B: Backend> DirichletMultinomial<B> {
    /// Create a new Dirichlet-Multinomial distribution.
    ///
    /// # Arguments
    /// * `n` - Total number of trials
    /// * `concentration` - Dirichlet concentration parameter (all elements > 0)
    ///
    /// # Panics
    /// Panics if concentration has fewer than 2 elements.
    pub fn new(n: usize, concentration: Tensor<B, 1>) -> Self {
        let [k] = concentration.dims();
        assert!(
            k >= 2,
            "DirichletMultinomial requires at least 2 categories"
        );

        let alpha_data: Vec<f32> = concentration.clone().into_data().to_vec().unwrap();

        // Compute α_0 = ∑_k α_k
        let alpha_sum: f64 = alpha_data.iter().map(|&a| a as f64).sum();

        // Pre-compute log(n!)
        let log_n_factorial = ln_gamma((n + 1) as f64);

        // Pre-compute log(Γ(α_0)) - log(Γ(n + α_0))
        let log_normalizer_n = ln_gamma(alpha_sum) - ln_gamma(n as f64 + alpha_sum);

        // Pre-compute log(Γ(α_k)) for each k
        let log_gamma_alpha: Vec<f64> = alpha_data.iter().map(|&a| ln_gamma(a as f64)).collect();

        Self {
            n,
            concentration,
            dim: k,
            alpha_sum,
            log_n_factorial,
            log_normalizer_n,
            log_gamma_alpha,
        }
    }

    /// Create a symmetric Dirichlet-Multinomial with all concentrations equal.
    ///
    /// # Arguments
    /// * `n` - Total number of trials
    /// * `k` - Number of categories
    /// * `alpha` - Common concentration value
    /// * `device` - Device to create tensors on
    pub fn symmetric(n: usize, k: usize, alpha: f32, device: &B::Device) -> Self {
        let concentration = Tensor::from_floats(vec![alpha; k].as_slice(), device);
        Self::new(n, concentration)
    }

    /// Get the number of categories.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the number of trials.
    pub fn num_trials(&self) -> usize {
        self.n
    }

    /// Get the total concentration α_0.
    pub fn total_concentration(&self) -> f64 {
        self.alpha_sum
    }

    /// Compute the log probability of an observation.
    ///
    /// # Arguments
    /// * `x` - Count vector (non-negative integers summing to n, as floats)
    ///
    /// # Returns
    /// Scalar tensor containing log P(x | n, α)
    ///
    /// # Formula
    /// log P(x | n, α) = log(n!) - ∑_k log(x_k!)
    ///                   + log(Γ(α_0)) - log(Γ(n + α_0))
    ///                   + ∑_k [log(Γ(x_k + α_k)) - log(Γ(α_k))]
    pub fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();

        // Get counts as a vector
        let counts: Vec<f32> = x.clone().into_data().to_vec().unwrap();
        let alpha_data: Vec<f32> = self.concentration.clone().into_data().to_vec().unwrap();

        // Compute ∑_k log(x_k!)
        let sum_log_x_factorial: f64 = counts.iter().map(|&c| ln_gamma(c as f64 + 1.0)).sum();

        // Compute ∑_k [log(Γ(x_k + α_k)) - log(Γ(α_k))]
        let sum_gamma_terms: f64 = counts
            .iter()
            .zip(alpha_data.iter())
            .zip(self.log_gamma_alpha.iter())
            .map(|((&x_k, &alpha_k), &log_gamma_alpha_k)| {
                ln_gamma(x_k as f64 + alpha_k as f64) - log_gamma_alpha_k
            })
            .sum();

        // log P(x | n, α) = log(n!) - ∑ log(x_k!) + log_normalizer_n + sum_gamma_terms
        let log_prob =
            self.log_n_factorial - sum_log_x_factorial + self.log_normalizer_n + sum_gamma_terms;

        Tensor::from_floats([log_prob as f32], &device)
    }

    /// Compute the expected probability for each category.
    ///
    /// E[p_k] = α_k / α_0
    pub fn expected_probs(&self) -> Tensor<B, 1> {
        self.concentration.clone().div_scalar(self.alpha_sum as f32)
    }

    /// Get the support of the distribution.
    pub fn support(&self) -> Support {
        Support::NonNegativeInteger
    }

    /// Compute the variance inflation factor compared to Multinomial.
    ///
    /// The Dirichlet-Multinomial has higher variance than Multinomial:
    /// Var(x_k) = n × p_k × (1 - p_k) × (n + α_0) / (1 + α_0)
    ///
    /// Returns the factor (n + α_0) / (1 + α_0) by which variance is inflated.
    pub fn variance_inflation(&self) -> f64 {
        (self.n as f64 + self.alpha_sum) / (1.0 + self.alpha_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dirichlet_multinomial_basic() {
        let device = Default::default();

        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 3.0, 5.0], &device);
        let dm = DirichletMultinomial::new(10, alpha);

        let x = Tensor::from_floats([2.0, 3.0, 5.0], &device);
        let log_prob: f32 = dm.log_prob(&x).into_scalar().elem();

        // The log probability should be finite and negative
        assert!(log_prob.is_finite(), "log_prob should be finite");
        assert!(log_prob < 0.0, "log_prob should be negative");
    }

    #[test]
    fn test_dirichlet_multinomial_symmetric() {
        let device = Default::default();

        // Symmetric α = [2, 2, 2] with n=6 trials
        let dm = DirichletMultinomial::<TestBackend>::symmetric(6, 3, 2.0, &device);

        // Equal counts should have higher probability than unequal
        let x_equal = Tensor::from_floats([2.0, 2.0, 2.0], &device);
        let x_unequal = Tensor::from_floats([5.0, 1.0, 0.0], &device);

        let log_prob_equal: f32 = dm.log_prob(&x_equal).into_scalar().elem();
        let log_prob_unequal: f32 = dm.log_prob(&x_unequal).into_scalar().elem();

        assert!(
            log_prob_equal > log_prob_unequal,
            "Symmetric DirMult should prefer equal counts: {} vs {}",
            log_prob_equal,
            log_prob_unequal
        );
    }

    #[test]
    fn test_dirichlet_multinomial_concentrated() {
        let device = Default::default();

        // Concentrated α = [10, 2, 2] should prefer counts matching expected probs
        let alpha = Tensor::<TestBackend, 1>::from_floats([10.0, 2.0, 2.0], &device);
        let dm = DirichletMultinomial::new(14, alpha);

        // Expected probs: [10/14, 2/14, 2/14] ≈ [0.71, 0.14, 0.14]
        // For n=14: expected counts ≈ [10, 2, 2]
        let x_expected = Tensor::from_floats([10.0, 2.0, 2.0], &device);
        let x_opposite = Tensor::from_floats([2.0, 2.0, 10.0], &device);

        let log_prob_expected: f32 = dm.log_prob(&x_expected).into_scalar().elem();
        let log_prob_opposite: f32 = dm.log_prob(&x_opposite).into_scalar().elem();

        assert!(
            log_prob_expected > log_prob_opposite,
            "Concentrated DirMult should prefer counts matching expected: {} vs {}",
            log_prob_expected,
            log_prob_opposite
        );
    }

    #[test]
    fn test_expected_probs() {
        let device = Default::default();

        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 3.0, 5.0], &device);
        let dm = DirichletMultinomial::new(10, alpha);

        let expected = dm.expected_probs();
        let probs: Vec<f32> = expected.into_data().to_vec().unwrap();

        // α_0 = 10, so expected probs are [0.2, 0.3, 0.5]
        assert!((probs[0] - 0.2).abs() < 1e-5);
        assert!((probs[1] - 0.3).abs() < 1e-5);
        assert!((probs[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_variance_inflation() {
        let device = Default::default();

        // With α_0 = 10 and n = 20
        let dm = DirichletMultinomial::<TestBackend>::symmetric(20, 3, 10.0 / 3.0, &device);

        // Variance inflation = (20 + 10) / (1 + 10) = 30/11 ≈ 2.73
        let inflation = dm.variance_inflation();
        let expected = 30.0 / 11.0;
        assert!(
            (inflation - expected).abs() < 1e-5,
            "Variance inflation {} should be ~{}",
            inflation,
            expected
        );
    }

    #[test]
    fn test_large_alpha_approaches_multinomial() {
        let device = Default::default();

        // With very large α, DirMult should be close to Multinomial
        // α = [100, 200, 300] → expected probs = [1/6, 2/6, 3/6]
        let alpha_large = Tensor::<TestBackend, 1>::from_floats([100.0, 200.0, 300.0], &device);
        let dm_large = DirichletMultinomial::new(6, alpha_large);

        // Small α = [1, 2, 3] → same expected probs but more variance
        let alpha_small = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let dm_small = DirichletMultinomial::new(6, alpha_small);

        // Both have same expected probs
        let probs_large: Vec<f32> = dm_large.expected_probs().into_data().to_vec().unwrap();
        let probs_small: Vec<f32> = dm_small.expected_probs().into_data().to_vec().unwrap();

        for (pl, ps) in probs_large.iter().zip(probs_small.iter()) {
            assert!(
                (pl - ps).abs() < 1e-5,
                "Expected probs should match: {} vs {}",
                pl,
                ps
            );
        }

        // But variance inflation should be lower for large α
        assert!(
            dm_large.variance_inflation() < dm_small.variance_inflation(),
            "Large α should have lower variance inflation"
        );
    }

    #[test]
    fn test_dimension_accessors() {
        let device = Default::default();
        let dm = DirichletMultinomial::<TestBackend>::symmetric(20, 5, 2.0, &device);

        assert_eq!(dm.dim(), 5);
        assert_eq!(dm.num_trials(), 20);
        assert!((dm.total_concentration() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_support() {
        let device = Default::default();
        let dm = DirichletMultinomial::<TestBackend>::symmetric(10, 3, 1.0, &device);
        assert_eq!(dm.support(), Support::NonNegativeInteger);
    }

    #[test]
    #[should_panic(expected = "DirichletMultinomial requires at least 2 categories")]
    fn test_requires_at_least_2_categories() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let alpha = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let _ = DirichletMultinomial::new(10, alpha);
    }

    #[test]
    fn test_probabilities_sum_correctly() {
        // For small n, we can verify that probabilities sum to 1
        // by enumerating all possible outcomes
        let device = Default::default();

        let dm = DirichletMultinomial::<TestBackend>::symmetric(3, 2, 1.0, &device);

        // For n=3, K=2, possible outcomes are:
        // [3,0], [2,1], [1,2], [0,3]
        let outcomes = [[3.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 3.0]];

        let mut total_prob = 0.0_f64;
        for outcome in &outcomes {
            let x = Tensor::<TestBackend, 1>::from_floats(outcome.as_slice(), &device);
            let log_prob: f32 = dm.log_prob(&x).into_scalar().elem();
            total_prob += (log_prob as f64).exp();
        }

        assert!(
            (total_prob - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {}",
            total_prob
        );
    }
}
