//! Negative Binomial distribution
//!
//! The Negative Binomial distribution models the number of failures before
//! achieving a specified number of successes in independent Bernoulli trials.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Negative Binomial distribution
///
/// # Parameters
/// - `r`: Number of successes (r > 0, can be non-integer)
/// - `p`: Success probability (0 < p < 1)
///
/// # Mathematical Definition
/// ```text
/// log f(k | r, p) = ln_gamma(k + r) - ln_gamma(k + 1) - ln_gamma(r) + r*ln(p) + k*ln(1-p)
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::negative_binomial::NegativeBinomial;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let r = Tensor::<B, 1>::from_floats([5.0], &device);
/// let p = Tensor::<B, 1>::from_floats([0.5], &device);
/// let dist = NegativeBinomial::new(r, p);
///
/// let x = Tensor::<B, 1>::from_floats([3.0, 5.0, 10.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct NegativeBinomial<B: Backend> {
    /// Number of successes parameter (r)
    pub r: Tensor<B, 1>,
    /// Success probability
    pub p: Tensor<B, 1>,
    /// Pre-computed: r as f64 for gamma computations
    r_f64: f64,
    /// Pre-computed: r * ln(p)
    r_log_p: Tensor<B, 1>,
    /// Pre-computed: ln(1 - p)
    log_1_minus_p: Tensor<B, 1>,
    /// Pre-computed: -ln_gamma(r)
    neg_ln_gamma_r: f32,
}

impl<B: Backend> NegativeBinomial<B> {
    /// Create a new Negative Binomial distribution
    ///
    /// # Arguments
    /// * `r` - Number of successes (r > 0)
    /// * `p` - Success probability (0 < p < 1)
    pub fn new(r: Tensor<B, 1>, p: Tensor<B, 1>) -> Self {
        let r_f64: f64 = r.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;

        // Pre-compute r * ln(p)
        let r_log_p = r.clone() * p.clone().log();

        // Pre-compute ln(1 - p)
        let log_1_minus_p = p.clone().neg().add_scalar(1.0).log();

        // Pre-compute -ln_gamma(r)
        let neg_ln_gamma_r = -(ln_gamma(r_f64) as f32);

        Self {
            r,
            p,
            r_f64,
            r_log_p,
            log_1_minus_p,
            neg_ln_gamma_r,
        }
    }
}

impl<B: Backend> Distribution<B> for NegativeBinomial<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(k | r, p) = ln_gamma(k + r) - ln_gamma(k + 1) - ln_gamma(r) + r*ln(p) + k*ln(1-p)
        //
        // The ln_gamma terms depend on k and must be computed per-element.
        // We extract values, compute gamma terms, then combine with tensor ops.

        let device = x.device();
        let k_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let gamma_terms: Vec<f32> = k_values
            .iter()
            .map(|&k| {
                let k_f64 = k as f64;
                // ln_gamma(k + r) - ln_gamma(k + 1)
                let lg_k_plus_r = ln_gamma(k_f64 + self.r_f64);
                let lg_k_plus_1 = ln_gamma(k_f64 + 1.0);
                (lg_k_plus_r - lg_k_plus_1) as f32
            })
            .collect();

        let gamma_tensor = Tensor::from_floats(gamma_terms.as_slice(), &device);

        // gamma_terms + neg_ln_gamma_r + r*ln(p) + k*ln(1-p)
        gamma_tensor.add_scalar(self.neg_ln_gamma_r)
            + self.r_log_p.clone()
            + x.clone() * self.log_1_minus_p.clone()
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
    fn test_negative_binomial_known_value() {
        let device = Default::default();

        // NB(r=1, p=0.5) at k=0
        // log f(0 | 1, 0.5) = ln_gamma(0+1) - ln_gamma(0+1) - ln_gamma(1) + 1*ln(0.5) + 0*ln(0.5)
        // = 0 - 0 - 0 + ln(0.5) = ln(0.5)
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = NegativeBinomial::new(r, p);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = 0.5_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_negative_binomial_r1_is_geometric() {
        let device = Default::default();

        // NB(r=1, p) is the geometric distribution
        // P(k) = p * (1-p)^k
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.3], &device);
        let dist = NegativeBinomial::new(r, p);

        let x = Tensor::from_floats([2.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // Geometric: ln(0.3) + 2*ln(0.7)
        let expected = 0.3_f64.ln() + 2.0 * 0.7_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "NB(1, p) should equal Geometric(p). Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_negative_binomial_p_half_symmetry() {
        let device = Default::default();

        // With r=1, p=0.5: P(k) = 0.5^(k+1)
        // Ratio P(k+1)/P(k) = 0.5
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = NegativeBinomial::new(r, p);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        let diff1 = log_probs[1] - log_probs[0];
        let diff2 = log_probs[2] - log_probs[1];
        let expected_diff = 0.5_f32.ln();

        assert!(
            (diff1 - expected_diff).abs() < 1e-4,
            "Expected constant ratio ln(0.5)={}, got {}",
            expected_diff,
            diff1
        );
        assert!(
            (diff2 - expected_diff).abs() < 1e-4,
            "Expected constant ratio ln(0.5)={}, got {}",
            expected_diff,
            diff2
        );
    }

    #[test]
    fn test_negative_binomial_larger_r() {
        let device = Default::default();

        // NB(r=5, p=0.4) at k=3
        // ln_gamma(3+5) - ln_gamma(3+1) - ln_gamma(5) + 5*ln(0.4) + 3*ln(0.6)
        // = ln_gamma(8) - ln_gamma(4) - ln_gamma(5) + 5*ln(0.4) + 3*ln(0.6)
        // = ln(7!) - ln(3!) - ln(4!) + 5*ln(0.4) + 3*ln(0.6)
        // = ln(5040) - ln(6) - ln(24) + 5*ln(0.4) + 3*ln(0.6)
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.4], &device);
        let dist = NegativeBinomial::new(r, p);

        let x = Tensor::from_floats([3.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected =
            ln_gamma(8.0) - ln_gamma(4.0) - ln_gamma(5.0) + 5.0 * 0.4_f64.ln() + 3.0 * 0.6_f64.ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_negative_binomial_vectorized() {
        let device = Default::default();

        let r: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = NegativeBinomial::new(r, p);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 5.0, 10.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);
    }

    #[test]
    fn test_negative_binomial_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let dist = NegativeBinomial::new(r, p);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
