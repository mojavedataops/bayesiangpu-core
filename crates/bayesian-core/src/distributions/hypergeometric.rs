//! Hypergeometric distribution
//!
//! The Hypergeometric distribution models the number of successes in draws
//! from a finite population without replacement.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Compute ln(C(a, b)) = ln_gamma(a+1) - ln_gamma(b+1) - ln_gamma(a-b+1)
fn ln_choose(a: f64, b: f64) -> f64 {
    ln_gamma(a + 1.0) - ln_gamma(b + 1.0) - ln_gamma(a - b + 1.0)
}

/// Hypergeometric distribution
///
/// # Parameters
/// - `big_n`: Population size (N >= 0)
/// - `big_k`: Number of success states in population (0 <= K <= N)
/// - `n`: Number of draws (0 <= n <= N)
///
/// # Mathematical Definition
/// ```text
/// log f(k | N, K, n) = ln_choose(K, k) + ln_choose(N-K, n-k) - ln_choose(N, n)
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::hypergeometric::Hypergeometric;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let big_n = Tensor::<B, 1>::from_floats([52.0], &device);
/// let big_k = Tensor::<B, 1>::from_floats([13.0], &device);
/// let n = Tensor::<B, 1>::from_floats([5.0], &device);
/// let dist = Hypergeometric::new(big_n, big_k, n);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Hypergeometric<B: Backend> {
    /// Population size (N)
    pub big_n: Tensor<B, 1>,
    /// Success states in population (K)
    pub big_k: Tensor<B, 1>,
    /// Number of draws (n)
    pub n: Tensor<B, 1>,
    /// Pre-computed: N as f64
    big_n_f64: f64,
    /// Pre-computed: K as f64
    big_k_f64: f64,
    /// Pre-computed: n as f64
    n_f64: f64,
    /// Pre-computed: ln_choose(N, n) — the normalizing constant
    ln_choose_n_n: f64,
}

impl<B: Backend> Hypergeometric<B> {
    /// Create a new Hypergeometric distribution
    ///
    /// # Arguments
    /// * `big_n` - Population size (N >= 0)
    /// * `big_k` - Number of success states in population (0 <= K <= N)
    /// * `n` - Number of draws (0 <= n <= N)
    pub fn new(big_n: Tensor<B, 1>, big_k: Tensor<B, 1>, n: Tensor<B, 1>) -> Self {
        let big_n_f64: f64 = big_n.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let big_k_f64: f64 = big_k.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let n_f64: f64 = n.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;

        let ln_choose_n_n = ln_choose(big_n_f64, n_f64);

        Self {
            big_n,
            big_k,
            n,
            big_n_f64,
            big_k_f64,
            n_f64,
            ln_choose_n_n,
        }
    }
}

impl<B: Backend> Distribution<B> for Hypergeometric<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();
        let k_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = k_values
            .iter()
            .map(|&k| {
                let k_f64 = k as f64;
                // Check support: k must be in [max(0, n-(N-K)), min(K, n)]
                let k_min = (self.n_f64 - (self.big_n_f64 - self.big_k_f64)).max(0.0);
                let k_max = self.big_k_f64.min(self.n_f64);
                if k_f64 < k_min || k_f64 > k_max || k_f64.fract() != 0.0 {
                    return f32::NEG_INFINITY;
                }
                // ln_choose(K, k) + ln_choose(N-K, n-k) - ln_choose(N, n)
                let lc1 = ln_choose(self.big_k_f64, k_f64);
                let lc2 = ln_choose(self.big_n_f64 - self.big_k_f64, self.n_f64 - k_f64);
                (lc1 + lc2 - self.ln_choose_n_n) as f32
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
    fn test_hypergeometric_deck_of_cards() {
        let device = Default::default();

        // Drawing 5 cards from a 52-card deck, 13 hearts
        // P(exactly 1 heart) = C(13,1)*C(39,4)/C(52,5)
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([52.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([13.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([1.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        // Expected: ln(C(13,1)) + ln(C(39,4)) - ln(C(52,5))
        let expected = ln_choose(13.0, 1.0) + ln_choose(39.0, 4.0) - ln_choose(52.0, 5.0);
        assert!(
            (lp as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            lp
        );
    }

    #[test]
    fn test_hypergeometric_small_example() {
        let device = Default::default();

        // N=10, K=4, n=3
        // P(k=2) = C(4,2)*C(6,1)/C(10,3)
        //        = 6*6/120 = 36/120 = 0.3
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([10.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([4.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([2.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = (0.3_f64).ln();
        assert!(
            (lp as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            lp
        );
    }

    #[test]
    fn test_hypergeometric_boundary_k_zero() {
        let device = Default::default();

        // N=10, K=4, n=3: P(k=0) = C(4,0)*C(6,3)/C(10,3) = 1*20/120 = 1/6
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([10.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([4.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([0.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = (1.0_f64 / 6.0).ln();
        assert!(
            (lp as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            lp
        );
    }

    #[test]
    fn test_hypergeometric_boundary_k_max() {
        let device = Default::default();

        // N=10, K=4, n=3: P(k=3) = C(4,3)*C(6,0)/C(10,3) = 4*1/120 = 1/30
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([10.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([4.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([3.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        let expected = (1.0_f64 / 30.0).ln();
        assert!(
            (lp as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            lp
        );
    }

    #[test]
    fn test_hypergeometric_probabilities_sum_to_one() {
        let device = Default::default();

        // N=10, K=4, n=3: valid k = 0, 1, 2, 3
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([10.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([4.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        let total: f64 = log_probs.iter().map(|&lp| (lp as f64).exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "Probabilities should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_hypergeometric_vectorized() {
        let device = Default::default();

        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([20.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([10.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 6);
    }

    #[test]
    fn test_hypergeometric_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let big_n: Tensor<TestBackend, 1> = Tensor::from_floats([52.0], &device);
        let big_k: Tensor<TestBackend, 1> = Tensor::from_floats([13.0], &device);
        let n: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let dist = Hypergeometric::new(big_n, big_k, n);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
