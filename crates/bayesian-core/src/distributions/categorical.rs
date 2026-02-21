//! Categorical distribution
//!
//! The Categorical distribution models a single draw from K categories,
//! where each category has a specified probability.

use super::{Distribution, Support};
use burn::prelude::*;

/// Categorical distribution
///
/// # Parameters
/// - `probs`: Probability vector (must sum to 1, all elements >= 0)
///
/// # Mathematical Definition
/// ```text
/// log f(k | probs) = ln(probs[k])   for k = 0, 1, ..., K-1
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::categorical::Categorical;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let probs = Tensor::<B, 1>::from_floats([0.2, 0.3, 0.5], &device);
/// let dist = Categorical::new(probs);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct Categorical<B: Backend> {
    /// Probability vector
    pub probs: Tensor<B, 1>,
    /// Pre-computed: log(probs) as f32 values for indexing
    log_probs_vec: Vec<f32>,
    /// Number of categories
    dim: usize,
}

impl<B: Backend> Categorical<B> {
    /// Create a new Categorical distribution
    ///
    /// # Arguments
    /// * `probs` - Probability vector (must sum to 1, all elements >= 0)
    ///
    /// # Panics
    /// Panics if probs has fewer than 2 elements.
    pub fn new(probs: Tensor<B, 1>) -> Self {
        let [k] = probs.dims();
        assert!(k >= 2, "Categorical requires at least 2 categories");

        // Pre-compute log(probs) as f32 values for lookup
        let probs_vec: Vec<f32> = probs.clone().into_data().to_vec().unwrap();
        let log_probs_vec: Vec<f32> = probs_vec.iter().map(|&p| p.ln()).collect();

        Self {
            probs,
            log_probs_vec,
            dim: k,
        }
    }

    /// Create a Categorical with uniform probabilities.
    ///
    /// # Arguments
    /// * `k` - Number of categories
    /// * `device` - Device to create tensors on
    pub fn uniform(k: usize, device: &B::Device) -> Self {
        let prob = 1.0 / k as f32;
        let probs = Tensor::from_floats(vec![prob; k].as_slice(), device);
        Self::new(probs)
    }

    /// Get the number of categories.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<B: Backend> Distribution<B> for Categorical<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // For each element in x, look up log_probs[round(x)]
        let device = x.device();
        let x_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let log_probs: Vec<f32> = x_values
            .iter()
            .map(|&v| {
                let idx = v.round() as usize;
                if idx < self.dim {
                    self.log_probs_vec[idx]
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
    fn test_categorical_uniform() {
        let device = Default::default();

        let dist = Categorical::<TestBackend>::uniform(4, &device);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        let expected = (0.25_f64).ln();
        for (i, &lp) in log_probs.iter().enumerate() {
            assert!(
                (lp as f64 - expected).abs() < 1e-5,
                "Category {}: expected {}, got {}",
                i,
                expected,
                lp
            );
        }
    }

    #[test]
    fn test_categorical_skewed() {
        let device = Default::default();

        let probs = Tensor::<TestBackend, 1>::from_floats([0.1, 0.2, 0.7], &device);
        let dist = Categorical::new(probs);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert!(
            (log_probs[0] as f64 - 0.1_f64.ln()).abs() < 1e-5,
            "Expected ln(0.1), got {}",
            log_probs[0]
        );
        assert!(
            (log_probs[1] as f64 - 0.2_f64.ln()).abs() < 1e-5,
            "Expected ln(0.2), got {}",
            log_probs[1]
        );
        assert!(
            (log_probs[2] as f64 - 0.7_f64.ln()).abs() < 1e-5,
            "Expected ln(0.7), got {}",
            log_probs[2]
        );
    }

    #[test]
    fn test_categorical_most_likely_category() {
        let device = Default::default();

        let probs = Tensor::<TestBackend, 1>::from_floats([0.1, 0.6, 0.3], &device);
        let dist = Categorical::new(probs);

        let x = Tensor::from_floats([0.0f32, 1.0, 2.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Category 1 should have the highest log probability
        assert!(
            log_probs[1] > log_probs[0],
            "Category 1 should be more likely than category 0"
        );
        assert!(
            log_probs[1] > log_probs[2],
            "Category 1 should be more likely than category 2"
        );
    }

    #[test]
    fn test_categorical_dimension() {
        let device = Default::default();
        let dist = Categorical::<TestBackend>::uniform(5, &device);
        assert_eq!(dist.dim(), 5);
    }

    #[test]
    fn test_categorical_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let probs = Tensor::<TestBackend, 1>::from_floats([0.3, 0.7], &device);
        let dist = Categorical::new(probs);
        assert_eq!(dist.support(), Support::NonNegativeInteger);
    }

    #[test]
    #[should_panic(expected = "Categorical requires at least 2 categories")]
    fn test_categorical_requires_at_least_2_categories() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let probs = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let _ = Categorical::new(probs);
    }

    #[test]
    fn test_categorical_binary() {
        let device = Default::default();

        // Binary categorical: coin flip
        let probs = Tensor::<TestBackend, 1>::from_floats([0.3, 0.7], &device);
        let dist = Categorical::new(probs);

        let x = Tensor::from_floats([0.0f32, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert!(
            (log_probs[0] as f64 - 0.3_f64.ln()).abs() < 1e-5,
            "Expected ln(0.3), got {}",
            log_probs[0]
        );
        assert!(
            (log_probs[1] as f64 - 0.7_f64.ln()).abs() < 1e-5,
            "Expected ln(0.7), got {}",
            log_probs[1]
        );
    }
}
