//! Beta-Binomial distribution
//!
//! The Beta-Binomial distribution is a compound distribution where the success
//! probability follows a Beta distribution. It models overdispersed binomial data.

use super::{Distribution, Support};
use crate::math::{ln_beta, ln_gamma};
use burn::prelude::*;

/// Beta-Binomial distribution
///
/// # Parameters
/// - `n`: Number of trials (non-negative integer)
/// - `alpha`: First shape parameter of the Beta prior (alpha > 0)
/// - `beta`: Second shape parameter of the Beta prior (beta > 0)
///
/// # Mathematical Definition
/// ```text
/// log f(k | n, alpha, beta) = ln_choose(n, k) + ln_beta(alpha + k, beta + n - k) - ln_beta(alpha, beta)
/// ```
///
/// where:
/// - `ln_choose(n, k) = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1)`
/// - `ln_beta(a, b) = ln_gamma(a) + ln_gamma(b) - ln_gamma(a+b)`
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::beta_binomial::BetaBinomial;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let dist = BetaBinomial::<B>::new(10, 2.0, 3.0, &device);
///
/// let x = Tensor::<B, 1>::from_floats([3.0, 5.0, 7.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct BetaBinomial<B: Backend> {
    /// Number of trials
    pub n: usize,
    /// Alpha parameter (first Beta shape)
    pub alpha: f64,
    /// Beta parameter (second Beta shape)
    pub beta_param: f64,
    /// Pre-computed: ln_gamma(n + 1)
    ln_gamma_n_plus_1: f64,
    /// Pre-computed: -ln_beta(alpha, beta)
    neg_ln_beta_ab: f64,
    /// Device
    _device: B::Device,
}

impl<B: Backend> BetaBinomial<B> {
    /// Create a new Beta-Binomial distribution
    ///
    /// # Arguments
    /// * `n` - Number of trials (non-negative integer)
    /// * `alpha` - First Beta shape parameter (alpha > 0)
    /// * `beta_param` - Second Beta shape parameter (beta > 0)
    /// * `device` - Device to create tensors on
    pub fn new(n: usize, alpha: f64, beta_param: f64, device: &B::Device) -> Self {
        let ln_gamma_n_plus_1 = ln_gamma((n + 1) as f64);
        let neg_ln_beta_ab = -ln_beta(alpha, beta_param);

        Self {
            n,
            alpha,
            beta_param,
            ln_gamma_n_plus_1,
            neg_ln_beta_ab,
            _device: device.clone(),
        }
    }
}

impl<B: Backend> Distribution<B> for BetaBinomial<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(k | n, alpha, beta) = ln_choose(n, k) + ln_beta(alpha+k, beta+n-k) - ln_beta(alpha, beta)
        //
        // ln_choose(n, k) = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1)
        // ln_beta(a, b) = ln_gamma(a) + ln_gamma(b) - ln_gamma(a+b)

        let device = x.device();
        let k_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = k_values
            .iter()
            .map(|&k| {
                let k_f64 = k as f64;
                let n_f64 = self.n as f64;

                // ln_choose(n, k)
                let ln_choose =
                    self.ln_gamma_n_plus_1 - ln_gamma(k_f64 + 1.0) - ln_gamma(n_f64 - k_f64 + 1.0);

                // ln_beta(alpha + k, beta + n - k)
                let ln_beta_posterior =
                    ln_beta(self.alpha + k_f64, self.beta_param + n_f64 - k_f64);

                (ln_choose + ln_beta_posterior + self.neg_ln_beta_ab) as f32
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
    fn test_beta_binomial_alpha_beta_one_is_discrete_uniform() {
        let device = Default::default();

        // BetaBinomial(n, 1, 1) reduces to DiscreteUniform on {0, ..., n}
        // P(k) = 1/(n+1) for all k
        let n = 5;
        let dist = BetaBinomial::<TestBackend>::new(n, 1.0, 1.0, &device);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        let expected = -((n + 1) as f64).ln();

        for (i, &lp) in log_probs.iter().enumerate() {
            assert!(
                (lp as f64 - expected).abs() < 1e-4,
                "BetaBinomial(n, 1, 1) should be uniform. k={}: expected {}, got {}",
                i,
                expected,
                lp
            );
        }
    }

    #[test]
    fn test_beta_binomial_known_value() {
        let device = Default::default();

        // BetaBinomial(n=3, alpha=2, beta=3) at k=1
        let dist = BetaBinomial::<TestBackend>::new(3, 2.0, 3.0, &device);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // Manual calculation:
        // ln_choose(3, 1) = ln(3) = ln_gamma(4) - ln_gamma(2) - ln_gamma(3)
        //                 = ln(6) - 0 - ln(2) = ln(3)
        // ln_beta(2+1, 3+3-1) = ln_beta(3, 5) = ln_gamma(3)+ln_gamma(5)-ln_gamma(8)
        //                      = ln(2)+ln(24)-ln(5040) = ln(48/5040) = ln(1/105)
        // ln_beta(2, 3) = ln_gamma(2)+ln_gamma(3)-ln_gamma(5) = 0+ln(2)-ln(24) = ln(2/24) = ln(1/12)
        // log_prob = ln(3) + ln(1/105) - ln(1/12) = ln(3) + ln(12/105) = ln(3*12/105) = ln(36/105)
        let expected =
            ln_gamma(4.0) - ln_gamma(2.0) - ln_gamma(3.0) + ln_beta(3.0, 5.0) - ln_beta(2.0, 3.0);
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_beta_binomial_symmetric() {
        let device = Default::default();

        // With alpha == beta, the distribution should be symmetric around n/2
        let n = 6;
        let dist = BetaBinomial::<TestBackend>::new(n, 3.0, 3.0, &device);

        let x = Tensor::from_floats([1.0f32, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // P(k=1) should equal P(k=5) when alpha == beta
        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-5,
            "BetaBinomial with alpha=beta should be symmetric. P(1)={}, P(5)={}",
            log_probs[0],
            log_probs[1]
        );
    }

    #[test]
    fn test_beta_binomial_endpoints() {
        let device = Default::default();

        // BetaBinomial(n=4, alpha=2, beta=2) at k=0 and k=4
        let dist = BetaBinomial::<TestBackend>::new(4, 2.0, 2.0, &device);

        let x = Tensor::from_floats([0.0f32, 4.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // With alpha=beta, P(0) == P(n)
        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-5,
            "P(0) should equal P(n) when alpha=beta. P(0)={}, P(4)={}",
            log_probs[0],
            log_probs[1]
        );
    }

    #[test]
    fn test_beta_binomial_vectorized() {
        let device = Default::default();

        let dist = BetaBinomial::<TestBackend>::new(10, 2.0, 5.0, &device);

        let x = Tensor::from_floats([0.0f32, 2.0, 5.0, 8.0, 10.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 5);
    }

    #[test]
    fn test_beta_binomial_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = BetaBinomial::<TestBackend>::new(10, 2.0, 3.0, &device);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }

    #[test]
    fn test_beta_binomial_probabilities_sum_to_one() {
        let device = Default::default();

        let n = 5;
        let dist = BetaBinomial::<TestBackend>::new(n, 2.0, 3.0, &device);

        // Sum exp(log_prob) for all k in {0, ..., n}
        let k_vals: Vec<f32> = (0..=n).map(|k| k as f32).collect();
        let x = Tensor::<TestBackend, 1>::from_floats(k_vals.as_slice(), &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        let total: f64 = log_probs.iter().map(|&lp| (lp as f64).exp()).sum();

        assert!(
            (total - 1.0).abs() < 1e-4,
            "Probabilities should sum to 1, got {}",
            total
        );
    }
}
