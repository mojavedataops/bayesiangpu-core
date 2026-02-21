//! Geometric distribution
//!
//! The Geometric distribution models the number of failures before the first
//! success in a sequence of independent Bernoulli trials.

use super::{Distribution, Support};
use burn::prelude::*;

/// Geometric distribution
///
/// Models the number of failures before the first success.
///
/// # Parameters
/// - `p`: Success probability (0 < p <= 1)
///
/// # Mathematical Definition
/// ```text
/// log f(k | p) = ln(p) + k * ln(1 - p)    for k = 0, 1, 2, ...
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::geometric::Geometric;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let p = Tensor::<B, 1>::from_floats([0.3], &device);
/// let dist = Geometric::new(p);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Geometric<B: Backend> {
    /// Success probability
    pub p: Tensor<B, 1>,
    /// Pre-computed: ln(p)
    log_p: Tensor<B, 1>,
    /// Pre-computed: ln(1 - p)
    log_1_minus_p: Tensor<B, 1>,
}

impl<B: Backend> Geometric<B> {
    /// Create a new Geometric distribution
    ///
    /// # Arguments
    /// * `p` - Success probability (0 < p <= 1)
    pub fn new(p: Tensor<B, 1>) -> Self {
        let log_p = p.clone().log();
        let log_1_minus_p = p.clone().neg().add_scalar(1.0).log();

        Self {
            p,
            log_p,
            log_1_minus_p,
        }
    }
}

impl<B: Backend> Distribution<B> for Geometric<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(k | p) = ln(p) + k * ln(1 - p)
        self.log_p.clone() + x.clone() * self.log_1_minus_p.clone()
    }

    fn support(&self) -> Support {
        Support::NonNegativeInteger
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_geometric_at_zero() {
        let device = Default::default();

        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.3], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At k=0: ln(0.3) + 0*ln(0.7) = ln(0.3)
        let expected = 0.3_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_geometric_known_value() {
        let device = Default::default();

        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.3], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([2.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At k=2: ln(0.3) + 2*ln(0.7)
        let expected = 0.3_f64.ln() + 2.0 * 0.7_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_geometric_high_p() {
        let device = Default::default();

        // When p is very high, almost all mass is at k=0
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.99], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At k=0: ln(0.99) + 0*ln(0.01) = ln(0.99) ~ -0.01005
        let expected = 0.99_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );

        // k=0 should be much more likely than k=5
        let x5 = Tensor::from_floats([5.0f32], &device);
        let log_prob_5: f32 = dist.log_prob(&x5).into_scalar().elem();
        assert!(
            log_prob > log_prob_5,
            "k=0 should be much more likely than k=5 when p is high"
        );
    }

    #[test]
    fn test_geometric_monotonically_decreasing() {
        let device = Default::default();

        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Geometric distribution is monotonically decreasing
        assert!(log_probs[0] > log_probs[1], "log_prob(0) > log_prob(1)");
        assert!(log_probs[1] > log_probs[2], "log_prob(1) > log_prob(2)");
        assert!(log_probs[2] > log_probs[3], "log_prob(2) > log_prob(3)");
    }

    #[test]
    fn test_geometric_p_half_symmetry() {
        let device = Default::default();

        // With p=0.5, P(k) = 0.5^(k+1)
        // So ratio P(k+1)/P(k) = 0.5 for all k
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Differences should all be ln(0.5)
        let diff1 = log_probs[1] - log_probs[0];
        let diff2 = log_probs[2] - log_probs[1];
        let expected_diff = 0.5_f32.ln();

        assert!(
            (diff1 - expected_diff).abs() < 1e-5,
            "Expected constant ratio ln(0.5), got {}",
            diff1
        );
        assert!(
            (diff2 - expected_diff).abs() < 1e-5,
            "Expected constant ratio ln(0.5), got {}",
            diff2
        );
    }

    #[test]
    fn test_geometric_vectorized() {
        let device = Default::default();

        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.4], &device);
        let dist = Geometric::new(p);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0, 4.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);
    }

    #[test]
    fn test_geometric_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = Geometric::new(p);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
