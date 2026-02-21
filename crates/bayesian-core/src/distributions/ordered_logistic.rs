//! Ordered Logistic distribution
//!
//! The Ordered Logistic (also called cumulative logit) distribution models
//! ordinal outcomes. It is widely used in ordinal regression models.

use super::{Distribution, Support};
use burn::prelude::*;

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Ordered Logistic distribution
///
/// # Parameters
/// - `eta`: Linear predictor (scalar)
/// - `cutpoints`: Ordered vector of K-1 thresholds
///
/// # Mathematical Definition
/// For K categories (0-indexed), with K-1 cutpoints c_0 < c_1 < ... < c_{K-2}:
/// ```text
/// j = 0:     log f(0) = log(sigmoid(c_0 - eta))
/// j = K-1:   log f(K-1) = log(1 - sigmoid(c_{K-2} - eta))
/// otherwise: log f(j) = log(sigmoid(c_j - eta) - sigmoid(c_{j-1} - eta))
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::ordered_logistic::OrderedLogistic;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let eta = Tensor::<B, 1>::from_floats([0.0], &device);
/// let cutpoints = vec![-1.0, 0.0, 1.0];
/// let dist = OrderedLogistic::new(eta, cutpoints);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct OrderedLogistic<B: Backend> {
    /// Linear predictor (eta)
    pub eta: Tensor<B, 1>,
    /// Cutpoints (ordered thresholds)
    pub cutpoints: Vec<f32>,
    /// Pre-computed: eta as f64
    #[allow(dead_code)]
    eta_f64: f64,
    /// Number of categories (K = cutpoints.len() + 1)
    k: usize,
    /// Pre-computed: cumulative probabilities P(Y <= j) for j = 0..K-1
    /// cum_probs[j] = sigmoid(cutpoints[j] - eta) for j < K-1
    /// cum_probs[K-1] = 1.0 (implicit)
    #[allow(dead_code)]
    cum_probs: Vec<f64>,
    /// Pre-computed: log category probabilities
    log_cat_probs: Vec<f64>,
}

impl<B: Backend> OrderedLogistic<B> {
    /// Create a new Ordered Logistic distribution
    ///
    /// # Arguments
    /// * `eta` - Linear predictor (scalar tensor)
    /// * `cutpoints` - Ordered vector of K-1 thresholds (as f32)
    pub fn new(eta: Tensor<B, 1>, cutpoints: Vec<f32>) -> Self {
        let eta_f64: f64 = eta.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let k = cutpoints.len() + 1;

        // Compute cumulative probabilities: P(Y <= j) = sigmoid(c_j - eta)
        let cum_probs: Vec<f64> = cutpoints
            .iter()
            .map(|&c| sigmoid_f64(c as f64 - eta_f64))
            .collect();

        // Compute category probabilities
        let mut log_cat_probs = Vec::with_capacity(k);
        for j in 0..k {
            let prob = if j == 0 {
                // P(Y = 0) = sigmoid(c_0 - eta)
                cum_probs[0]
            } else if j == k - 1 {
                // P(Y = K-1) = 1 - sigmoid(c_{K-2} - eta)
                1.0 - cum_probs[k - 2]
            } else {
                // P(Y = j) = sigmoid(c_j - eta) - sigmoid(c_{j-1} - eta)
                cum_probs[j] - cum_probs[j - 1]
            };
            // Clamp to avoid log(0)
            log_cat_probs.push(prob.max(1e-20).ln());
        }

        Self {
            eta,
            cutpoints,
            eta_f64,
            k,
            cum_probs,
            log_cat_probs,
        }
    }

    /// Get the number of categories.
    pub fn num_categories(&self) -> usize {
        self.k
    }
}

impl<B: Backend> Distribution<B> for OrderedLogistic<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();
        let j_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = j_values
            .iter()
            .map(|&j| {
                let idx = j.round() as usize;
                if idx < self.k {
                    self.log_cat_probs[idx] as f32
                } else {
                    f32::NEG_INFINITY
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
    fn test_ordered_logistic_two_categories_is_logistic_regression() {
        let device = Default::default();

        // With 1 cutpoint (2 categories), OrderedLogistic is logistic regression
        // P(Y=0) = sigmoid(c - eta), P(Y=1) = 1 - sigmoid(c - eta)
        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([1.5], &device);
        let cutpoints = vec![0.0f32]; // Single cutpoint
        let dist = OrderedLogistic::new(eta, cutpoints);

        let x = Tensor::from_floats([0.0f32, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // P(Y=0) = sigmoid(0.0 - 1.5) = sigmoid(-1.5)
        let sig_neg = sigmoid_f64(-1.5);
        let expected_0 = sig_neg.ln();
        let expected_1 = (1.0 - sig_neg).ln();

        assert!(
            (log_probs[0] as f64 - expected_0).abs() < 1e-5,
            "P(Y=0): Expected {}, got {}",
            expected_0,
            log_probs[0]
        );
        assert!(
            (log_probs[1] as f64 - expected_1).abs() < 1e-5,
            "P(Y=1): Expected {}, got {}",
            expected_1,
            log_probs[1]
        );
    }

    #[test]
    fn test_ordered_logistic_symmetric_cutpoints() {
        let device = Default::default();

        // Symmetric cutpoints with eta=0 should give symmetric probabilities
        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let cutpoints = vec![-1.0f32, 1.0]; // Symmetric around 0
        let dist = OrderedLogistic::new(eta, cutpoints);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // P(Y=0) should equal P(Y=2) by symmetry
        assert!(
            (log_probs[0] - log_probs[2]).abs() < 1e-5,
            "P(Y=0)={} should equal P(Y=2)={} by symmetry",
            log_probs[0],
            log_probs[2]
        );
    }

    #[test]
    fn test_ordered_logistic_known_values() {
        let device = Default::default();

        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let cutpoints = vec![-1.0f32, 0.0, 1.0]; // 4 categories
        let dist = OrderedLogistic::new(eta, cutpoints);

        // Compute expected values manually
        let sig0 = sigmoid_f64(-1.0 - 0.5); // sigmoid(-1.5)
        let sig1 = sigmoid_f64(0.0 - 0.5); // sigmoid(-0.5)
        let sig2 = sigmoid_f64(1.0 - 0.5); // sigmoid(0.5)

        let p0 = sig0;
        let p1 = sig1 - sig0;
        let p2 = sig2 - sig1;
        let p3 = 1.0 - sig2;

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert!(
            (log_probs[0] as f64 - p0.ln()).abs() < 1e-5,
            "P(Y=0): Expected {}, got {}",
            p0.ln(),
            log_probs[0]
        );
        assert!(
            (log_probs[1] as f64 - p1.ln()).abs() < 1e-5,
            "P(Y=1): Expected {}, got {}",
            p1.ln(),
            log_probs[1]
        );
        assert!(
            (log_probs[2] as f64 - p2.ln()).abs() < 1e-5,
            "P(Y=2): Expected {}, got {}",
            p2.ln(),
            log_probs[2]
        );
        assert!(
            (log_probs[3] as f64 - p3.ln()).abs() < 1e-5,
            "P(Y=3): Expected {}, got {}",
            p3.ln(),
            log_probs[3]
        );
    }

    #[test]
    fn test_ordered_logistic_probabilities_sum_to_one() {
        let device = Default::default();

        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let cutpoints = vec![-1.0f32, 0.0, 1.0];
        let dist = OrderedLogistic::new(eta, cutpoints);

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
    fn test_ordered_logistic_large_eta_shifts_mass_right() {
        let device = Default::default();

        let cutpoints = vec![-1.0f32, 0.0, 1.0];

        // Large positive eta shifts mass to higher categories
        let eta_high: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let dist_high = OrderedLogistic::new(eta_high, cutpoints.clone());

        let x = Tensor::from_floats([0.0f32, 3.0], &device);
        let lp_high: Vec<f32> = dist_high.log_prob(&x).into_data().to_vec().unwrap();

        // P(Y=3) should be much larger than P(Y=0) for large eta
        assert!(
            lp_high[1] > lp_high[0],
            "With large eta, P(Y=3)={} should be > P(Y=0)={}",
            lp_high[1],
            lp_high[0]
        );
    }

    #[test]
    fn test_ordered_logistic_out_of_range() {
        let device = Default::default();

        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let cutpoints = vec![0.0f32]; // 2 categories: 0 and 1
        let dist = OrderedLogistic::new(eta, cutpoints);

        // k=2 is out of range for 2 categories
        let x = Tensor::from_floats([2.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();
        assert!(
            lp == f32::NEG_INFINITY,
            "Out-of-range should give -inf, got {}",
            lp
        );
    }

    #[test]
    fn test_ordered_logistic_num_categories() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let cutpoints = vec![-1.0f32, 0.0, 1.0];
        let dist = OrderedLogistic::new(eta, cutpoints);
        assert_eq!(dist.num_categories(), 4);
    }

    #[test]
    fn test_ordered_logistic_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let eta: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let cutpoints = vec![-1.0f32, 0.0, 1.0];
        let dist = OrderedLogistic::new(eta, cutpoints);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }
}
