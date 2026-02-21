//! Pareto distribution
//!
//! The Pareto distribution is parameterized by shape (alpha) and scale/minimum (x_m).
//! It is commonly used to model heavy-tailed phenomena such as wealth distribution.

use super::{Distribution, Support};
use burn::prelude::*;

/// Pareto distribution
///
/// # Parameters
/// - `alpha`: Shape parameter (alpha > 0)
/// - `x_m`: Scale/minimum parameter (x_m > 0)
///
/// # Mathematical Definition
/// ```text
/// log f(x | alpha, x_m) = ln(alpha) + alpha*ln(x_m) - (alpha+1)*ln(x)   for x >= x_m
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::pareto::Pareto;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let alpha = Tensor::<B, 1>::from_floats([2.0], &device);
/// let x_m = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = Pareto::new(alpha, x_m);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Pareto<B: Backend> {
    /// Shape parameter (alpha)
    pub alpha: Tensor<B, 1>,
    /// Scale/minimum parameter (x_m)
    pub x_m: Tensor<B, 1>,
    /// Pre-computed: ln(alpha) + alpha*ln(x_m)
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: alpha + 1
    alpha_plus_one: Tensor<B, 1>,
}

impl<B: Backend> Pareto<B> {
    /// Create a new Pareto distribution
    ///
    /// # Arguments
    /// * `alpha` - Shape parameter (alpha > 0)
    /// * `x_m` - Scale/minimum parameter (x_m > 0)
    pub fn new(alpha: Tensor<B, 1>, x_m: Tensor<B, 1>) -> Self {
        // Pre-compute: ln(alpha) + alpha*ln(x_m)
        let log_normalizer = alpha.clone().log() + alpha.clone() * x_m.clone().log();
        let alpha_plus_one = alpha.clone().add_scalar(1.0);

        Self {
            alpha,
            x_m,
            log_normalizer,
            alpha_plus_one,
        }
    }
}

impl<B: Backend> Distribution<B> for Pareto<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(x | alpha, x_m) = ln(alpha) + alpha*ln(x_m) - (alpha+1)*ln(x)
        //
        // Breaking it down:
        // 1. log_normalizer = ln(alpha) + alpha*ln(x_m)  [pre-computed]
        // 2. -(alpha+1) * ln(x)

        let log_x = x.clone().log();
        let tail_term = self.alpha_plus_one.clone() * log_x;

        self.log_normalizer.clone() - tail_term
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
    fn test_pareto_at_x_m() {
        let device = Default::default();

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Pareto::new(alpha, x_m);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At x=1, alpha=2, x_m=1:
        // ln(2) + 2*ln(1) - 3*ln(1) = ln(2)
        let expected = 2.0_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_pareto_known_value() {
        let device = Default::default();

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = Pareto::new(alpha, x_m);

        let x = Tensor::from_floats([4.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // ln(3) + 3*ln(2) - 4*ln(4)
        let expected = 3.0_f64.ln() + 3.0 * 2.0_f64.ln() - 4.0 * 4.0_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_pareto_decreasing() {
        let device = Default::default();

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Pareto::new(alpha, x_m);

        // Pareto density is always decreasing for x >= x_m
        let x = Tensor::from_floats([1.0f32, 2.0, 3.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        for i in 0..log_probs.len() - 1 {
            assert!(
                log_probs[i] > log_probs[i + 1],
                "Pareto density should decrease: log_prob({}) > log_prob({})",
                i,
                i + 1
            );
        }
    }

    #[test]
    fn test_pareto_normalization() {
        // For Pareto(alpha, x_m), the density integrates to 1 over [x_m, inf)
        // We verify by checking that at x_m, f(x_m) = alpha / x_m
        let device = Default::default();

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Pareto::new(alpha, x_m);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // f(x_m) = alpha/x_m when x_m=1: f(1) = 5
        // log(5) ~ 1.6094
        let expected = 5.0_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_pareto_vectorized() {
        let device = Default::default();

        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Pareto::new(alpha, x_m);

        let x = Tensor::from_floats([1.0f32, 2.0, 3.0, 4.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 4);
    }

    #[test]
    fn test_pareto_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let alpha: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let x_m: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = Pareto::new(alpha, x_m);
        assert_eq!(dist.support(), Support::Positive);
    }
}
