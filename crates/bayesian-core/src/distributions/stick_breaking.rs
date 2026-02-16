//! Stick-breaking construction for Dirichlet processes
//!
//! The stick-breaking process provides a constructive definition of the
//! Dirichlet process and related infinite-dimensional distributions.
//! It generates an infinite sequence of weights that sum to 1.

use burn::prelude::*;

/// Stick-breaking process representation
///
/// The stick-breaking construction generates weights w_k via:
/// 1. Draw v_k ~ Beta(1, α) independently for k = 1, 2, ...
/// 2. Set w_k = v_k × ∏_{j<k} (1 - v_j)
///
/// This gives weights that sum to 1: ∑_k w_k = 1.
///
/// # Parameters
/// - `concentration` (α): Controls how quickly weights decay
///   - Larger α → more uniform weights (more categories needed)
///   - Smaller α → weights concentrated on first few categories
///
/// # Truncation
/// In practice, we truncate at K components, with the remaining mass
/// assigned to category K.
///
/// # Use Cases
/// - Dirichlet Process Mixtures (DPM)
/// - Infinite-dimensional topic models
/// - Nonparametric Bayesian clustering
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::stick_breaking::StickBreaking;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // Truncated stick-breaking with α=1, K=10 components
/// let sb = StickBreaking::<B>::new(1.0, 10, &device);
///
/// // Get the truncated weights
/// let weights = sb.weights();
/// ```
#[derive(Debug, Clone)]
pub struct StickBreaking<B: Backend> {
    /// Concentration parameter α > 0
    pub concentration: f32,
    /// Number of truncated components K
    pub truncation: usize,
    /// Pre-computed Beta(1, α) parameters
    /// All beta_a = 1.0 (first parameter)
    _beta_a: f32,
    /// Beta second parameter = α
    _beta_b: f32,
    /// Device for tensor operations
    device: B::Device,
}

impl<B: Backend> StickBreaking<B> {
    /// Create a new truncated stick-breaking process.
    ///
    /// # Arguments
    /// * `concentration` - Concentration parameter α > 0
    /// * `truncation` - Number of components K to truncate at
    /// * `device` - Device to create tensors on
    ///
    /// # Panics
    /// Panics if concentration <= 0 or truncation < 2.
    pub fn new(concentration: f32, truncation: usize, device: &B::Device) -> Self {
        assert!(concentration > 0.0, "Concentration must be positive");
        assert!(truncation >= 2, "Truncation must be at least 2");

        Self {
            concentration,
            truncation,
            _beta_a: 1.0,
            _beta_b: concentration,
            device: device.clone(),
        }
    }

    /// Get the truncation level.
    pub fn truncation(&self) -> usize {
        self.truncation
    }

    /// Compute stick-breaking weights from Beta samples v_1, ..., v_{K-1}.
    ///
    /// Given a tensor of K-1 values from Beta(1, α), computes the weights:
    /// w_k = v_k × ∏_{j<k} (1 - v_j)
    ///
    /// The last weight w_K gets all remaining probability.
    ///
    /// # Arguments
    /// * `v` - Tensor of shape [K-1] with Beta(1, α) samples in (0, 1)
    ///
    /// # Returns
    /// Tensor of shape [K] with weights that sum to 1
    pub fn weights_from_beta(&self, v: &Tensor<B, 1>) -> Tensor<B, 1> {
        let [k_minus_1] = v.dims();
        let k = k_minus_1 + 1;
        assert_eq!(k, self.truncation, "v must have truncation-1 elements");

        // Get v values
        let v_data: Vec<f32> = v.clone().into_data().to_vec().unwrap();

        // Compute weights via stick-breaking
        let mut weights = Vec::with_capacity(k);
        let mut remaining = 1.0_f32;

        for &v_i in &v_data {
            let w_i = remaining * v_i;
            weights.push(w_i);
            remaining *= 1.0 - v_i;
        }

        // Last weight gets remaining probability
        weights.push(remaining);

        Tensor::from_floats(weights.as_slice(), &self.device)
    }

    /// Compute the log Jacobian for the stick-breaking transformation.
    ///
    /// When transforming from Beta samples v to weights w, the Jacobian is:
    /// log |∂w/∂v| = ∑_{k=1}^{K-1} (K-1-k) × log(1 - v_k)
    ///
    /// This is needed for computing log probabilities when parameterizing
    /// a model in terms of weights instead of Beta samples.
    ///
    /// # Arguments
    /// * `v` - Tensor of shape [K-1] with Beta(1, α) samples
    ///
    /// # Returns
    /// Scalar tensor containing the log Jacobian
    pub fn log_jacobian(&self, v: &Tensor<B, 1>) -> Tensor<B, 1> {
        let [k_minus_1] = v.dims();

        // log |J| = ∑_{k=1}^{K-1} (K-1-k) × log(1 - v_k)
        let v_data: Vec<f32> = v.clone().into_data().to_vec().unwrap();

        let mut log_jac = 0.0_f32;
        for (k, &v_k) in v_data.iter().enumerate() {
            let multiplier = (k_minus_1 - k) as f32;
            log_jac += multiplier * (1.0 - v_k).ln();
        }

        Tensor::from_floats([log_jac], &self.device)
    }

    /// Compute the log probability of a set of weights under the stick-breaking prior.
    ///
    /// This computes:
    /// log p(w | α) = log p(v | α) + log |∂v/∂w|
    ///
    /// where v are the Beta samples corresponding to weights w, and
    /// log p(v | α) = ∑_k log Beta(v_k; 1, α).
    ///
    /// # Arguments
    /// * `weights` - Tensor of shape [K] with weights summing to 1
    ///
    /// # Returns
    /// Scalar tensor containing the log probability
    pub fn log_prob(&self, weights: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Convert weights back to v values
        let v = self.weights_to_beta(weights);

        // Log probability of Beta(1, α) for each v_k
        // log Beta(v_k; 1, α) = (α-1) × log(1 - v_k) - log(B(1, α))
        // B(1, α) = 1/α, so log(B(1,α)) = -log(α)
        let v_data: Vec<f32> = v.clone().into_data().to_vec().unwrap();

        let log_normalizer = self.concentration.ln();
        let alpha_minus_1 = self.concentration - 1.0;

        let mut log_prob = 0.0_f32;
        for &v_k in &v_data {
            // For v ~ Beta(1, α): log p(v) = (α-1)×log(1-v) + log(α)
            log_prob += alpha_minus_1 * (1.0 - v_k).ln() + log_normalizer;
        }

        // Add inverse Jacobian (converting from w to v)
        // This is negative of the forward Jacobian
        let log_jac: f32 = self.log_jacobian(&v).into_scalar().elem();
        log_prob -= log_jac;

        Tensor::from_floats([log_prob], &self.device)
    }

    /// Convert weights back to Beta samples (inverse stick-breaking).
    ///
    /// Given weights w_1, ..., w_K, recover v_1, ..., v_{K-1} where:
    /// v_k = w_k / (1 - ∑_{j<k} w_j)
    ///
    /// # Arguments
    /// * `weights` - Tensor of shape [K] with weights summing to 1
    ///
    /// # Returns
    /// Tensor of shape [K-1] with recovered Beta samples
    pub fn weights_to_beta(&self, weights: &Tensor<B, 1>) -> Tensor<B, 1> {
        let w_data: Vec<f32> = weights.clone().into_data().to_vec().unwrap();
        let k = w_data.len();

        let mut v = Vec::with_capacity(k - 1);
        let mut remaining = 1.0_f32;

        for w_i in w_data.iter().take(k - 1) {
            // v_k = w_k / remaining
            let v_i = if remaining > 1e-10 {
                (w_i / remaining).clamp(1e-10, 1.0 - 1e-10)
            } else {
                0.5 // Fallback for numerical stability
            };
            v.push(v_i);
            remaining -= w_i;
        }

        Tensor::from_floats(v.as_slice(), &self.device)
    }

    /// Get the expected weight for the k-th component.
    ///
    /// E[w_k] = (α / (1 + α))^(k-1) × (1 / (1 + α))
    ///
    /// # Arguments
    /// * `k` - Component index (1-indexed)
    ///
    /// # Returns
    /// Expected weight for component k
    pub fn expected_weight(&self, k: usize) -> f32 {
        assert!(k >= 1 && k <= self.truncation, "k must be in [1, K]");

        let ratio = self.concentration / (1.0 + self.concentration);

        if k < self.truncation {
            ratio.powi((k - 1) as i32) / (1.0 + self.concentration)
        } else {
            // Last component gets all remaining expected mass
            ratio.powi((k - 1) as i32)
        }
    }

    /// Get the expected number of components with significant weight.
    ///
    /// This is approximately α × log(n / α) for n observations,
    /// but for the truncated case we use a simpler estimate.
    ///
    /// # Arguments
    /// * `threshold` - Minimum weight to consider "significant"
    ///
    /// # Returns
    /// Approximate number of components expected above threshold
    pub fn expected_num_components(&self, threshold: f32) -> usize {
        // Number of components needed to capture 1 - threshold of mass
        // Approximately log(threshold) / log(α / (1 + α))
        if self.concentration >= 1.0 {
            let ratio = self.concentration / (1.0 + self.concentration);
            let num = (threshold.ln() / ratio.ln()).ceil() as usize;
            num.min(self.truncation)
        } else {
            // Small α concentrates mass in first few components
            (((1.0 / threshold).ln() / (1.0 / self.concentration).ln()).ceil() as usize)
                .min(self.truncation)
        }
    }
}

/// GEM distribution (Griffiths-Engen-McCloskey)
///
/// The GEM distribution is the probability distribution over infinite
/// probability vectors induced by the stick-breaking construction with
/// Beta(1, α) draws. It's the marginal distribution of Dirichlet process
/// weights.
///
/// GEM(α) is equivalent to StickBreaking with K → ∞.
pub type GEM<B> = StickBreaking<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_stick_breaking_weights_sum_to_one() {
        let device = Default::default();
        let sb = StickBreaking::<TestBackend>::new(1.0, 5, &device);

        // Create some Beta(1,1) samples (uniform on [0,1])
        let v = Tensor::from_floats([0.3, 0.4, 0.5, 0.6], &device);
        let weights = sb.weights_from_beta(&v);

        let sum: f32 = weights.clone().sum().into_scalar().elem();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Stick-breaking weights should sum to 1, got {}",
            sum
        );

        // All weights should be positive
        let w_data: Vec<f32> = weights.into_data().to_vec().unwrap();
        for &w in &w_data {
            assert!(w > 0.0, "All weights should be positive, got {}", w);
        }
    }

    #[test]
    fn test_stick_breaking_roundtrip() {
        let device = Default::default();
        let sb = StickBreaking::<TestBackend>::new(2.0, 4, &device);

        // Original Beta samples
        let v_original = Tensor::from_floats([0.3, 0.5, 0.7], &device);

        // Convert to weights
        let weights = sb.weights_from_beta(&v_original);

        // Convert back to Beta samples
        let v_recovered = sb.weights_to_beta(&weights);

        let v_orig_data: Vec<f32> = v_original.into_data().to_vec().unwrap();
        let v_rec_data: Vec<f32> = v_recovered.into_data().to_vec().unwrap();

        for (orig, rec) in v_orig_data.iter().zip(v_rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "Roundtrip should preserve v: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_large_alpha_uniform_weights() {
        let device = Default::default();

        // Large α should give more uniform weights
        let sb = StickBreaking::<TestBackend>::new(10.0, 5, &device);

        // Beta(1, 10) has mean 1/(1+10) = 0.091
        // So with v close to 0.1, we should get roughly equal weights
        let v = Tensor::from_floats([0.2, 0.25, 0.33, 0.5], &device);
        let weights = sb.weights_from_beta(&v);
        let w_data: Vec<f32> = weights.into_data().to_vec().unwrap();

        // With these specific v values, first few weights should be relatively similar
        // (not testing exact uniformity, just that it's not extremely skewed)
        let max_w = w_data.iter().cloned().fold(0.0, f32::max);
        let min_w = w_data.iter().cloned().fold(1.0, f32::min);
        let ratio = max_w / min_w;

        assert!(
            ratio < 10.0,
            "Large α should give more uniform weights, ratio {}",
            ratio
        );
    }

    #[test]
    fn test_small_alpha_concentrated() {
        let device = Default::default();

        // Small α should concentrate on first components
        let sb = StickBreaking::<TestBackend>::new(0.1, 5, &device);

        // Beta(1, 0.1) has mean 1/(1+0.1) ≈ 0.91
        // So first weight should be large
        let v = Tensor::from_floats([0.9, 0.9, 0.9, 0.9], &device);
        let weights = sb.weights_from_beta(&v);
        let w_data: Vec<f32> = weights.into_data().to_vec().unwrap();

        // First weight should be dominant
        assert!(
            w_data[0] > 0.8,
            "Small α with large v should concentrate on first component: {}",
            w_data[0]
        );
    }

    #[test]
    fn test_expected_weights() {
        let device = Default::default();
        let alpha = 2.0;
        let sb = StickBreaking::<TestBackend>::new(alpha, 5, &device);

        // E[w_1] = 1 / (1 + α) = 1/3
        let expected_1 = sb.expected_weight(1);
        assert!(
            (expected_1 - 1.0 / 3.0).abs() < 1e-5,
            "E[w_1] should be 1/3, got {}",
            expected_1
        );

        // E[w_2] = α / (1+α)^2 = 2/9
        let expected_2 = sb.expected_weight(2);
        assert!(
            (expected_2 - 2.0 / 9.0).abs() < 1e-5,
            "E[w_2] should be 2/9, got {}",
            expected_2
        );
    }

    #[test]
    fn test_log_prob_finite() {
        let device = Default::default();
        let sb = StickBreaking::<TestBackend>::new(1.0, 4, &device);

        let weights = Tensor::from_floats([0.4, 0.3, 0.2, 0.1], &device);
        let log_prob: f32 = sb.log_prob(&weights).into_scalar().elem();

        assert!(
            log_prob.is_finite(),
            "Log probability should be finite, got {}",
            log_prob
        );
    }

    #[test]
    fn test_log_jacobian() {
        let device = Default::default();
        let sb = StickBreaking::<TestBackend>::new(1.0, 4, &device);

        // For v at midpoints, Jacobian should be finite
        let v = Tensor::from_floats([0.5, 0.5, 0.5], &device);
        let log_jac: f32 = sb.log_jacobian(&v).into_scalar().elem();

        assert!(
            log_jac.is_finite(),
            "Log Jacobian should be finite, got {}",
            log_jac
        );
    }

    #[test]
    #[should_panic(expected = "Concentration must be positive")]
    fn test_invalid_concentration() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = StickBreaking::<TestBackend>::new(0.0, 5, &device);
    }

    #[test]
    #[should_panic(expected = "Truncation must be at least 2")]
    fn test_invalid_truncation() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = StickBreaking::<TestBackend>::new(1.0, 1, &device);
    }
}
