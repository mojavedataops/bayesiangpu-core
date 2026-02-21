//! Chi-Squared distribution
//!
//! The chi-squared distribution with k degrees of freedom is a special case
//! of the Gamma distribution: ChiSquared(k) = Gamma(k/2, 1/2).

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Chi-Squared distribution
///
/// # Parameters
/// - `df` (k): Degrees of freedom (must be positive)
///
/// This is a special case of Gamma(alpha=k/2, rate=1/2).
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::chi_squared::ChiSquared;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let df = Tensor::<B, 1>::from_floats([3.0], &device);
/// let dist = ChiSquared::new(df);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct ChiSquared<B: Backend> {
    /// Degrees of freedom (k)
    pub df: Tensor<B, 1>,
    /// Pre-computed: -(k/2)*ln(2) - ln(Gamma(k/2))
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: k/2 - 1
    half_df_minus_one: Tensor<B, 1>,
}

impl<B: Backend> ChiSquared<B> {
    /// Create a new Chi-Squared distribution
    ///
    /// # Arguments
    /// * `df` - Degrees of freedom (k > 0)
    pub fn new(df: Tensor<B, 1>) -> Self {
        let df_data: Vec<f32> = df.clone().into_data().to_vec().unwrap();
        let device = df.device();

        // half_df = k/2
        let half_df: Vec<f32> = df_data.iter().map(|&k| k / 2.0).collect();

        // ln_gamma(k/2)
        let ln_gamma_half_df: Vec<f32> = half_df
            .iter()
            .map(|&hd| ln_gamma(hd as f64) as f32)
            .collect();

        // log_normalizer = -(k/2)*ln(2) - ln(Gamma(k/2))
        let log_norm: Vec<f32> = half_df
            .iter()
            .zip(ln_gamma_half_df.iter())
            .map(|(&hd, &lg)| -hd * (2.0_f32).ln() - lg)
            .collect();

        let log_normalizer = Tensor::from_floats(log_norm.as_slice(), &device);

        // half_df_minus_one = k/2 - 1
        let hdfm1: Vec<f32> = half_df.iter().map(|&hd| hd - 1.0).collect();
        let half_df_minus_one = Tensor::from_floats(hdfm1.as_slice(), &device);

        Self {
            df,
            log_normalizer,
            half_df_minus_one,
        }
    }
}

impl<B: Backend> Distribution<B> for ChiSquared<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Chi-Squared distribution:
        // log p(x | k) = -(k/2)*ln(2) - ln(Gamma(k/2)) + (k/2-1)*ln(x) - x/2
        //
        // Breaking it down:
        // 1. log_normalizer = -(k/2)*ln(2) - ln(Gamma(k/2))  [pre-computed]
        // 2. (k/2-1)*ln(x)
        // 3. -x/2

        let log_x = x.clone().log();
        let shape_term = self.half_df_minus_one.clone() * log_x;
        let exp_term = x.clone().mul_scalar(-0.5);

        self.log_normalizer.clone() + shape_term + exp_term
    }

    fn support(&self) -> Support {
        Support::Positive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::gamma::Gamma;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_chi_squared_at_known_value() {
        let device = Default::default();

        // Chi-squared(2) at x=1
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = ChiSquared::new(df);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // ChiSq(2) = Exp(0.5), so log p(1) = ln(0.5) - 0.5*1 = -ln(2) - 0.5
        let expected = -(2.0_f64).ln() - 0.5;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_chi_squared_gamma_equivalence() {
        let device = Default::default();

        // ChiSquared(k) = Gamma(k/2, 1/2)
        let k = 5.0_f32;

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([k], &device);
        let chi2 = ChiSquared::new(df);

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([k / 2.0], &device);
        let rate: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let gamma = Gamma::new(alpha, rate);

        let x = Tensor::from_floats([2.0f32, 4.0, 6.0], &device);

        let chi2_lp: Vec<f32> = chi2.log_prob(&x).into_data().to_vec().unwrap();
        let gamma_lp: Vec<f32> = gamma.log_prob(&x).into_data().to_vec().unwrap();

        for i in 0..3 {
            assert!(
                (chi2_lp[i] - gamma_lp[i]).abs() < 1e-4,
                "ChiSquared and Gamma should agree at x={}: chi2={}, gamma={}",
                [2.0, 4.0, 6.0][i],
                chi2_lp[i],
                gamma_lp[i]
            );
        }
    }

    #[test]
    fn test_chi_squared_vectorized() {
        let device = Default::default();

        // Chi-squared(4): mode at k-2 = 2
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([4.0], &device);
        let dist = ChiSquared::new(df);

        let x = Tensor::from_floats([0.5f32, 1.0, 2.0, 4.0, 8.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);

        // Mode at 2 should have highest density
        assert!(log_probs[2] > log_probs[0], "log_prob(2) > log_prob(0.5)");
        assert!(log_probs[2] > log_probs[1], "log_prob(2) > log_prob(1)");
        assert!(log_probs[2] > log_probs[3], "log_prob(2) > log_prob(4)");
    }

    #[test]
    fn test_chi_squared_monotone_decrease_from_mode() {
        let device = Default::default();

        // Chi-squared(6): mode at k-2 = 4
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([6.0], &device);
        let dist = ChiSquared::new(df);

        let x = Tensor::from_floats([4.0f32, 6.0, 8.0, 12.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Monotonically decreasing from mode
        assert!(log_probs[0] > log_probs[1], "log_prob(4) > log_prob(6)");
        assert!(log_probs[1] > log_probs[2], "log_prob(6) > log_prob(8)");
        assert!(log_probs[2] > log_probs[3], "log_prob(8) > log_prob(12)");
    }

    #[test]
    fn test_chi_squared_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = ChiSquared::new(df);
        assert_eq!(dist.support(), Support::Positive);
    }
}
