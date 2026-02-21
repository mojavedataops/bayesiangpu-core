//! Zero-Inflated Negative Binomial distribution
//!
//! The Zero-Inflated Negative Binomial distribution models count data with
//! excess zeros and overdispersion. It is a mixture of a point mass at zero
//! and a Negative Binomial distribution.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Zero-Inflated Negative Binomial distribution
///
/// # Parameters
/// - `r`: Number of successes (r > 0, can be non-integer)
/// - `p`: Success probability (0 < p < 1)
/// - `zero_prob`: Probability of structural zero (0 <= pi < 1)
///
/// # Mathematical Definition
/// ```text
/// For k = 0: log f(0) = log(pi + (1-pi)*p^r)
/// For k > 0: log f(k) = log(1-pi) + ln_gamma(k+r) - ln_gamma(k+1) - ln_gamma(r)
///                        + r*log(p) + k*log(1-p)
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::zero_inflated_neg_binomial::ZeroInflatedNegativeBinomial;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let r = Tensor::<B, 1>::from_floats([5.0], &device);
/// let p = Tensor::<B, 1>::from_floats([0.5], &device);
/// let zero_prob = Tensor::<B, 1>::from_floats([0.3], &device);
/// let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 5.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct ZeroInflatedNegativeBinomial<B: Backend> {
    /// Number of successes parameter (r)
    pub r: Tensor<B, 1>,
    /// Success probability
    pub p: Tensor<B, 1>,
    /// Probability of structural zero (pi)
    pub zero_prob: Tensor<B, 1>,
    /// Pre-computed: r as f64
    r_f64: f64,
    /// Pre-computed: p as f64
    #[allow(dead_code)]
    p_f64: f64,
    /// Pre-computed: zero_prob as f64
    #[allow(dead_code)]
    zero_prob_f64: f64,
    /// Pre-computed: log(1 - pi)
    log_1_minus_pi: f64,
    /// Pre-computed: r * log(p)
    r_log_p: f64,
    /// Pre-computed: log(1 - p)
    log_1_minus_p: f64,
    /// Pre-computed: -ln_gamma(r)
    neg_ln_gamma_r: f64,
    /// Pre-computed: log(pi + (1-pi)*p^r) for k=0 case
    log_prob_zero: f64,
}

impl<B: Backend> ZeroInflatedNegativeBinomial<B> {
    /// Create a new Zero-Inflated Negative Binomial distribution
    ///
    /// # Arguments
    /// * `r` - Number of successes (r > 0)
    /// * `p` - Success probability (0 < p < 1)
    /// * `zero_prob` - Probability of structural zero (0 <= pi < 1)
    pub fn new(r: Tensor<B, 1>, p: Tensor<B, 1>, zero_prob: Tensor<B, 1>) -> Self {
        let r_f64: f64 = r.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let p_f64: f64 = p.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let zero_prob_f64: f64 = zero_prob.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;

        let log_1_minus_pi = (1.0 - zero_prob_f64).ln();
        let r_log_p = r_f64 * p_f64.ln();
        let log_1_minus_p = (1.0 - p_f64).ln();
        let neg_ln_gamma_r = -ln_gamma(r_f64);

        // For k=0: log(pi + (1-pi)*p^r)
        let log_prob_zero = (zero_prob_f64 + (1.0 - zero_prob_f64) * p_f64.powf(r_f64)).ln();

        Self {
            r,
            p,
            zero_prob,
            r_f64,
            p_f64,
            zero_prob_f64,
            log_1_minus_pi,
            r_log_p,
            log_1_minus_p,
            neg_ln_gamma_r,
            log_prob_zero,
        }
    }
}

impl<B: Backend> Distribution<B> for ZeroInflatedNegativeBinomial<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();
        let k_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = k_values
            .iter()
            .map(|&k| {
                let k_f64 = k as f64;
                if k_f64 == 0.0 {
                    self.log_prob_zero as f32
                } else {
                    // log(1-pi) + ln_gamma(k+r) - ln_gamma(k+1) - ln_gamma(r) + r*log(p) + k*log(1-p)
                    let lg_k_plus_r = ln_gamma(k_f64 + self.r_f64);
                    let lg_k_plus_1 = ln_gamma(k_f64 + 1.0);
                    (self.log_1_minus_pi + lg_k_plus_r - lg_k_plus_1
                        + self.neg_ln_gamma_r
                        + self.r_log_p
                        + k_f64 * self.log_1_minus_p) as f32
                }
            })
            .collect();

        Tensor::from_floats(log_probs.as_slice(), &device)
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
    fn test_zinb_zero_prob_zero_reduces_to_neg_binomial() {
        let device = Default::default();

        // When zero_prob = 0, ZINB reduces to standard NegativeBinomial
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.4], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);

        // Check k=0: should match NB(r=5, p=0.4) at k=0
        // NB at k=0: ln_gamma(0+5) - ln_gamma(0+1) - ln_gamma(5) + 5*ln(0.4) + 0*ln(0.6)
        //          = ln_gamma(5) - 0 - ln_gamma(5) + 5*ln(0.4) = 5*ln(0.4)
        let x0 = Tensor::from_floats([0.0f32], &device);
        let lp0: f32 = dist.log_prob(&x0).into_scalar().elem();
        let expected0 = 5.0 * 0.4_f64.ln(); // = r * ln(p) when k=0
        assert!(
            (lp0 as f64 - expected0).abs() < 1e-5,
            "k=0: Expected {}, got {}",
            expected0,
            lp0
        );

        // Check k=3: should match NB at k=3
        let x3 = Tensor::from_floats([3.0f32], &device);
        let lp3: f32 = dist.log_prob(&x3).into_scalar().elem();
        let expected3 =
            ln_gamma(8.0) - ln_gamma(4.0) - ln_gamma(5.0) + 5.0 * 0.4_f64.ln() + 3.0 * 0.6_f64.ln();
        assert!(
            (lp3 as f64 - expected3).abs() < 1e-4,
            "k=3: Expected {}, got {}",
            expected3,
            lp3
        );
    }

    #[test]
    fn test_zinb_high_zero_prob() {
        let device = Default::default();

        let r: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.9], &device);
        let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);

        let x = Tensor::from_floats([0.0f32, 1.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // k=0 should have much higher probability
        assert!(
            log_probs[0] > log_probs[1],
            "k=0 ({}) should be more likely than k=1 ({})",
            log_probs[0],
            log_probs[1]
        );
        assert!(
            log_probs[0] > log_probs[2],
            "k=0 ({}) should be more likely than k=5 ({})",
            log_probs[0],
            log_probs[2]
        );
    }

    #[test]
    fn test_zinb_known_values() {
        let device = Default::default();

        let r: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.6], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.25], &device);
        let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);

        // k=0: log(0.25 + 0.75 * 0.6^2)
        let x0 = Tensor::from_floats([0.0f32], &device);
        let lp0: f32 = dist.log_prob(&x0).into_scalar().elem();
        let expected0 = (0.25 + 0.75 * 0.6_f64.powf(2.0)).ln();
        assert!(
            (lp0 as f64 - expected0).abs() < 1e-5,
            "k=0: Expected {}, got {}",
            expected0,
            lp0
        );

        // k=2: log(0.75) + ln_gamma(4) - ln_gamma(3) - ln_gamma(2) + 2*log(0.6) + 2*log(0.4)
        let x2 = Tensor::from_floats([2.0f32], &device);
        let lp2: f32 = dist.log_prob(&x2).into_scalar().elem();
        let expected2 = 0.75_f64.ln() + ln_gamma(4.0) - ln_gamma(3.0) - ln_gamma(2.0)
            + 2.0 * 0.6_f64.ln()
            + 2.0 * 0.4_f64.ln();
        assert!(
            (lp2 as f64 - expected2).abs() < 1e-4,
            "k=2: Expected {}, got {}",
            expected2,
            lp2
        );
    }

    #[test]
    fn test_zinb_vectorized() {
        let device = Default::default();

        let r: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.2], &device);
        let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 5.0, 10.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);
    }

    #[test]
    fn test_zinb_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let r: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let p: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let zero_prob: Tensor<TestBackend, 1> = Tensor::from_floats([0.2], &device);
        let dist = ZeroInflatedNegativeBinomial::new(r, p, zero_prob);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
