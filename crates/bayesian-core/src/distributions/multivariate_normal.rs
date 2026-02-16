//! Multivariate Normal (Gaussian) distribution
//!
//! The multivariate normal distribution is parameterized by a mean vector
//! and a covariance matrix (via its Cholesky factor).

use burn::prelude::*;

/// Multivariate Normal distribution parameterized by Cholesky factor.
///
/// The distribution is N(mu, Sigma) where Sigma = L @ L^T and L is
/// lower triangular (Cholesky factor).
///
/// # Parameters
/// - `mu`: Mean vector of shape [D]
/// - `scale_tril`: Lower triangular Cholesky factor L of shape [D, D]
///
/// # Log probability
/// ```text
/// log p(x | mu, L) = -0.5 * D * log(2*pi)
///                    - sum(log(diag(L)))
///                    - 0.5 * ||L^{-1}(x - mu)||^2
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::multivariate_normal::MultivariateNormal;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // 2D MVN with identity covariance
/// let mu = Tensor::<B, 1>::from_floats([0.0, 0.0], &device);
/// let scale_tril = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
/// let dist = MultivariateNormal::new(mu, scale_tril);
///
/// let x = Tensor::<B, 1>::from_floats([0.5, -0.5], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateNormal<B: Backend> {
    /// Mean vector [D]
    pub mu: Tensor<B, 1>,
    /// Dimension
    dim: usize,
    /// Pre-computed log normalizer: -0.5 * D * log(2*pi) - sum(log(diag(L)))
    log_normalizer: Tensor<B, 1>,
    /// Inverse of Cholesky factor, flattened [D*D] (row-major)
    /// Pre-computed for efficient log_prob
    scale_tril_inv_flat: Tensor<B, 1>,
}

impl<B: Backend> MultivariateNormal<B> {
    /// Create a new Multivariate Normal distribution.
    ///
    /// # Arguments
    /// * `mu` - Mean vector of shape [D]
    /// * `scale_tril` - Lower triangular Cholesky factor of shape [D, D]
    ///
    /// # Panics
    /// Panics if dimensions don't match or matrix is not square.
    pub fn new(mu: Tensor<B, 1>, scale_tril: Tensor<B, 2>) -> Self {
        let [d] = mu.dims();
        let [n, m] = scale_tril.dims();
        assert_eq!(n, m, "Cholesky factor must be square");
        assert_eq!(n, d, "Cholesky factor dimension must match mean");

        let device = mu.device();

        // Extract Cholesky factor to compute inverse and log determinant
        let l_data: Vec<f32> = scale_tril.clone().into_data().to_vec().unwrap();

        // Compute log determinant: sum of log of diagonal elements
        let mut log_det = 0.0f32;
        for i in 0..d {
            log_det += l_data[i * d + i].ln();
        }

        // Compute inverse of L using forward substitution
        // L^{-1} is also lower triangular
        let mut l_inv = vec![0.0f32; d * d];
        for i in 0..d {
            // Solve L * x = e_i for each column i
            for j in 0..=i {
                if i == j {
                    l_inv[i * d + j] = 1.0 / l_data[i * d + i];
                } else {
                    let mut sum = 0.0f32;
                    for k in j..i {
                        sum += l_data[i * d + k] * l_inv[k * d + j];
                    }
                    l_inv[i * d + j] = -sum / l_data[i * d + i];
                }
            }
        }

        // Pre-compute log normalizer
        let log_2pi = (2.0 * std::f64::consts::PI).ln() as f32;
        let log_norm = -0.5 * (d as f32) * log_2pi - log_det;
        let log_normalizer = Tensor::<B, 1>::from_floats([log_norm], &device);

        let scale_tril_inv_flat =
            Tensor::<B, 1>::from_floats(TensorData::new(l_inv, [d * d]), &device);

        Self {
            mu,
            dim: d,
            log_normalizer,
            scale_tril_inv_flat,
        }
    }

    /// Create a standard multivariate normal (mean=0, covariance=I).
    ///
    /// # Arguments
    /// * `dim` - Dimension of the distribution
    /// * `device` - Device to create tensors on
    pub fn standard(dim: usize, device: &B::Device) -> Self {
        let mu = Tensor::zeros([dim], device);

        // Identity Cholesky factor
        let mut l_data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            l_data[i * dim + i] = 1.0;
        }
        let scale_tril = Tensor::<B, 2>::from_floats(TensorData::new(l_data, [dim, dim]), device);

        Self::new(mu, scale_tril)
    }

    /// Get the dimension of the distribution.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute the log probability of a single observation.
    ///
    /// # Arguments
    /// * `x` - Observation vector of shape [D]
    ///
    /// # Returns
    /// Scalar tensor containing log p(x)
    pub fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let d = self.dim;
        // Compute z = x - mu
        let z = x.clone() - self.mu.clone();

        // Compute L^{-1} @ z using matmul
        // Reshape L^{-1} from [D*D] to [D, D]
        let l_inv = self.scale_tril_inv_flat.clone().reshape([d, d]);

        // Reshape z from [D] to [D, 1] for matmul
        let z_col = z.reshape([d, 1]);

        // y = L^{-1} @ z -> [D, 1]
        let y = l_inv.matmul(z_col);

        // Compute ||y||^2 = y^T @ y
        let y_flat = y.reshape([d]);
        let y_sq = y_flat.clone() * y_flat;

        // Sum to get scalar
        let mahalanobis = y_sq.sum();

        // log_prob = log_normalizer - 0.5 * ||L^{-1}(x-mu)||^2
        self.log_normalizer.clone() - mahalanobis.reshape([1]).mul_scalar(0.5)
    }

    /// Compute log probability for a batch of observations.
    ///
    /// # Arguments
    /// * `x` - Batch of observations of shape [N, D]
    ///
    /// # Returns
    /// Tensor of shape [N] containing log p(x_i) for each observation
    pub fn log_prob_batch(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let [n, d] = x.dims();
        assert_eq!(d, self.dim, "Observation dimension must match");

        let _device = x.device();

        // Broadcast mu: [D] -> [N, D]
        let mu_batch = self.mu.clone().reshape([1, d]).repeat_dim(0, n);

        // z = x - mu, shape [N, D]
        let z = x - mu_batch;

        // L^{-1} shape [D, D]
        let l_inv = self
            .scale_tril_inv_flat
            .clone()
            .reshape([self.dim, self.dim]);

        // y = z @ L^{-1}^T = (L^{-1} @ z^T)^T
        // z is [N, D], we want [N, D] @ [D, D]^T -> but matmul needs compatible dims
        // Alternative: y = z @ L^{-1}^T where L^{-1}^T is [D, D]
        let l_inv_t = l_inv.transpose();
        let y = z.matmul(l_inv_t); // [N, D]

        // ||y_i||^2 for each row
        let y_sq = y.clone() * y;
        // sum_dim(1) gives [N, 1], reshape to [N]
        let mahalanobis = y_sq.sum_dim(1).reshape([n]); // [N]

        // log_prob = log_normalizer - 0.5 * mahalanobis
        let log_norm_scalar: f32 = self
            .log_normalizer
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        mahalanobis.mul_scalar(-0.5).add_scalar(log_norm_scalar)
    }
}

/// Create a Multivariate Normal from a covariance matrix.
///
/// Computes the Cholesky decomposition internally.
///
/// # Arguments
/// * `mu` - Mean vector of shape [D]
/// * `cov` - Covariance matrix of shape [D, D] (must be positive definite)
/// * `device` - Device for tensor allocation
///
/// # Returns
/// MultivariateNormal distribution
pub fn mvn_from_covariance<B: Backend>(
    mu: Tensor<B, 1>,
    cov: Tensor<B, 2>,
) -> MultivariateNormal<B> {
    let [d, _] = cov.dims();
    let device = mu.device();

    // Compute Cholesky decomposition of covariance
    let cov_data: Vec<f32> = cov.into_data().to_vec().unwrap();

    let mut l = vec![0.0f32; d * d];

    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;

            if i == j {
                for k in 0..j {
                    sum += l[j * d + k] * l[j * d + k];
                }
                l[j * d + j] = (cov_data[j * d + j] - sum).sqrt();
            } else {
                for k in 0..j {
                    sum += l[i * d + k] * l[j * d + k];
                }
                l[i * d + j] = (cov_data[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    let scale_tril = Tensor::<B, 2>::from_floats(TensorData::new(l, [d, d]), &device);

    MultivariateNormal::new(mu, scale_tril)
}

/// Create a Multivariate Normal from a precision matrix.
///
/// # Arguments
/// * `mu` - Mean vector of shape [D]
/// * `precision` - Precision matrix (inverse covariance) of shape [D, D]
///
/// # Returns
/// MultivariateNormal distribution
pub fn mvn_from_precision<B: Backend>(
    mu: Tensor<B, 1>,
    precision: Tensor<B, 2>,
) -> MultivariateNormal<B> {
    let [d, _] = precision.dims();
    let device = mu.device();

    // Compute Cholesky of precision: Lambda = L_p @ L_p^T
    // Then Sigma = Lambda^{-1}, and Sigma = L @ L^T where L = L_p^{-T}
    let prec_data: Vec<f32> = precision.into_data().to_vec().unwrap();

    // Cholesky of precision
    let mut l_p = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;
            if i == j {
                for k in 0..j {
                    sum += l_p[j * d + k] * l_p[j * d + k];
                }
                l_p[j * d + j] = (prec_data[j * d + j] - sum).sqrt();
            } else {
                for k in 0..j {
                    sum += l_p[i * d + k] * l_p[j * d + k];
                }
                l_p[i * d + j] = (prec_data[i * d + j] - sum) / l_p[j * d + j];
            }
        }
    }

    // Invert L_p to get L_p^{-1}
    let mut l_p_inv = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            if i == j {
                l_p_inv[i * d + j] = 1.0 / l_p[i * d + i];
            } else {
                let mut sum = 0.0f32;
                for k in j..i {
                    sum += l_p[i * d + k] * l_p_inv[k * d + j];
                }
                l_p_inv[i * d + j] = -sum / l_p[i * d + i];
            }
        }
    }

    // L = L_p^{-T} (transpose of inverse)
    let mut l = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            l[i * d + j] = l_p_inv[j * d + i]; // Transpose
        }
    }

    // L is upper triangular after transpose, we need lower triangular
    // Actually L_p^{-T} where L_p is lower triangular gives upper triangular
    // For MVN we need lower triangular, so use L = (L_p^{-1})^T properly

    // Simpler: just invert precision to get covariance, then Cholesky
    // This is less efficient but correct
    let mut cov = vec![0.0f32; d * d];

    // cov = L_p^{-T} @ L_p^{-1}
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += l_p_inv[k * d + i] * l_p_inv[k * d + j];
            }
            cov[i * d + j] = sum;
        }
    }

    // Now Cholesky of cov
    let mut l_cov = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;
            if i == j {
                for k in 0..j {
                    sum += l_cov[j * d + k] * l_cov[j * d + k];
                }
                l_cov[j * d + j] = (cov[j * d + j] - sum).sqrt();
            } else {
                for k in 0..j {
                    sum += l_cov[i * d + k] * l_cov[j * d + k];
                }
                l_cov[i * d + j] = (cov[i * d + j] - sum) / l_cov[j * d + j];
            }
        }
    }

    let scale_tril = Tensor::<B, 2>::from_floats(TensorData::new(l_cov, [d, d]), &device);

    MultivariateNormal::new(mu, scale_tril)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mvn_standard_2d() {
        let device = NdArrayDevice::default();
        let dist = MultivariateNormal::<TestBackend>::standard(2, &device);

        // log p(0, 0) for standard MVN = -0.5 * 2 * log(2*pi) = -1.8379
        let x = Tensor::from_floats([0.0f32, 0.0], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_data().to_vec::<f32>().unwrap()[0];
        let expected = -0.5 * 2.0 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_mvn_standard_at_one() {
        let device = NdArrayDevice::default();
        let dist = MultivariateNormal::<TestBackend>::standard(2, &device);

        // log p(1, 0) = -0.5 * 2 * log(2*pi) - 0.5 * 1 = -2.3379
        let x = Tensor::from_floats([1.0f32, 0.0], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_data().to_vec::<f32>().unwrap()[0];
        let expected = -0.5 * 2.0 * (2.0 * std::f64::consts::PI).ln() - 0.5;

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_mvn_with_covariance() {
        let device = NdArrayDevice::default();

        // Covariance [[4, 0], [0, 1]] means sigma_1 = 2, sigma_2 = 1
        let mu: Tensor<TestBackend, 1> = Tensor::from_floats([0.0f32, 0.0], &device);
        let cov: Tensor<TestBackend, 2> = Tensor::from_floats([[4.0f32, 0.0], [0.0, 1.0]], &device);

        let dist = mvn_from_covariance(mu, cov);

        // At x = (0, 0): log p = -0.5*2*log(2pi) - 0.5*log(4) - 0.5*log(1)
        //                      = -0.5*2*log(2pi) - log(2)
        let x = Tensor::from_floats([0.0f32, 0.0], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_data().to_vec::<f32>().unwrap()[0];
        let expected = -0.5 * 2.0 * (2.0 * std::f64::consts::PI).ln() - (2.0_f64).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_mvn_with_correlation() {
        let device = NdArrayDevice::default();

        // Covariance [[1, 0.5], [0.5, 1]] (correlation 0.5)
        let mu: Tensor<TestBackend, 1> = Tensor::from_floats([0.0f32, 0.0], &device);
        let cov: Tensor<TestBackend, 2> = Tensor::from_floats([[1.0f32, 0.5], [0.5, 1.0]], &device);

        let dist = mvn_from_covariance(mu, cov);

        // Just verify it runs and gives reasonable output
        let x = Tensor::from_floats([0.0f32, 0.0], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_data().to_vec::<f32>().unwrap()[0];

        // Should be negative and finite
        assert!(result.is_finite(), "log_prob should be finite");
        assert!(result < 0.0, "log_prob at mean should be negative");
    }

    #[test]
    fn test_mvn_batch() {
        let device = NdArrayDevice::default();
        let dist = MultivariateNormal::<TestBackend>::standard(2, &device);

        // Batch of 3 observations
        let x = Tensor::from_floats([[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]], &device);

        let log_probs = dist.log_prob_batch(x);
        let results: Vec<f32> = log_probs.into_data().to_vec().unwrap();

        assert_eq!(results.len(), 3);

        // First should be highest (at mean)
        assert!(results[0] > results[1]);
        assert!(results[0] > results[2]);

        // Second and third should be equal (symmetric)
        assert!((results[1] - results[2]).abs() < 1e-5);
    }

    #[test]
    fn test_mvn_3d() {
        let device = NdArrayDevice::default();
        let dist = MultivariateNormal::<TestBackend>::standard(3, &device);

        let x = Tensor::from_floats([0.0f32, 0.0, 0.0], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_data().to_vec::<f32>().unwrap()[0];
        let expected = -0.5 * 3.0 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_mvn_dimension() {
        let device = NdArrayDevice::default();
        let dist = MultivariateNormal::<TestBackend>::standard(5, &device);

        assert_eq!(dist.dim(), 5);
    }
}
