//! Bayesian model trait for HMC/NUTS sampling
//!
//! This module defines the [`BayesianModel`] trait that models must implement
//! to be used with the HMC and NUTS samplers.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// A Bayesian model that can compute log-probability and gradients
///
/// This trait defines the interface for models that can be sampled using
/// gradient-based MCMC methods like HMC and NUTS.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend for gradient computation
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::BayesianModel;
/// use burn::prelude::*;
/// use burn::tensor::backend::AutodiffBackend;
///
/// struct LinearRegression<B: Backend> {
///     x: Tensor<B, 2>,
///     y: Tensor<B, 1>,
/// }
///
/// impl<B: AutodiffBackend> BayesianModel<B> for LinearRegression<B> {
///     fn dim(&self) -> usize {
///         self.x.dims()[1] + 1  // coefficients + sigma
///     }
///
///     fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 0> {
///         // Compute joint log probability of priors and likelihood
///         // ...
///     }
///
///     fn param_names(&self) -> Vec<String> {
///         // Return parameter names for diagnostics
///         // ...
///     }
/// }
/// ```
pub trait BayesianModel<B: AutodiffBackend>: Send + Sync {
    /// Number of parameters (flat dimension of parameter space)
    ///
    /// This is the total number of scalar parameters in the model.
    fn dim(&self) -> usize;

    /// Compute log-probability given flattened parameters
    ///
    /// The parameters should be in unconstrained space (real-valued).
    /// The model is responsible for any necessary transformations
    /// (e.g., exp for positive parameters) and including the appropriate
    /// Jacobian correction.
    ///
    /// # Arguments
    ///
    /// * `params` - Flattened parameter vector in unconstrained space
    ///
    /// # Returns
    ///
    /// A 1-D tensor with shape [1] containing the log probability (unnormalized posterior)
    fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 1>;

    /// Parameter names for diagnostics and reporting
    ///
    /// Returns a vector of strings naming each parameter. The order should
    /// match the flattened parameter vector used in `log_prob`.
    fn param_names(&self) -> Vec<String>;

    /// Transform from unconstrained to constrained space (optional)
    ///
    /// Default implementation returns parameters unchanged (identity transform).
    /// Override this if your model has constrained parameters.
    ///
    /// # Arguments
    ///
    /// * `unconstrained` - Parameters in unconstrained space
    ///
    /// # Returns
    ///
    /// Parameters in constrained space
    fn transform(&self, unconstrained: &Tensor<B, 1>) -> Tensor<B, 1> {
        unconstrained.clone()
    }

    /// Log determinant of Jacobian for the transformation
    ///
    /// Default implementation returns 0 (identity transform has unit Jacobian).
    /// Override this if you override `transform`.
    ///
    /// # Arguments
    ///
    /// * `unconstrained` - Parameters in unconstrained space
    ///
    /// # Returns
    ///
    /// 1-D tensor with shape [1] containing log |det(J)|
    fn log_det_jacobian(&self, unconstrained: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = unconstrained.device();
        Tensor::<B, 1>::from_floats([0.0f32], &device)
    }

    /// Compute log probability and gradient directly without autodiff
    ///
    /// Models that have an optimized (e.g., GPU-accelerated) path for computing
    /// log_prob and its gradient can override this method. When this returns
    /// `Some((logp, grad))`, the sampler will use these values directly instead
    /// of building an autodiff computation graph.
    ///
    /// This is the primary integration point for GPU REDUCE kernels: the model
    /// computes priors analytically on CPU and dispatches likelihood sums to
    /// GPU compute shaders, bypassing Burn's autodiff entirely.
    ///
    /// # Arguments
    ///
    /// * `params` - Flattened parameter vector as f32 values
    ///
    /// # Returns
    ///
    /// `Some((log_prob, gradient_vector))` if the model has an optimized path,
    /// `None` to fall back to autodiff via `log_prob()` + `.backward()`.
    fn logp_and_grad_direct(&self, _params: &[f32]) -> Option<(f64, Vec<f64>)> {
        None
    }
}

/// Compute log probability with Jacobian adjustment for transformations
///
/// This function computes the log probability in constrained space with
/// the appropriate Jacobian correction for any parameter transformations.
///
/// # Arguments
///
/// * `model` - The Bayesian model
/// * `unconstrained` - Parameters in unconstrained space
///
/// # Returns
///
/// Scalar tensor containing adjusted log probability
pub fn log_prob_transformed<B: AutodiffBackend, M: BayesianModel<B>>(
    model: &M,
    unconstrained: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let log_prob = model.log_prob(unconstrained);
    let log_det_jac = model.log_det_jacobian(unconstrained);
    log_prob + log_det_jac
}

/// Compute log probability and gradient in one pass
///
/// This function efficiently computes both the log probability and its
/// gradient with respect to parameters using autodiff.
///
/// # Arguments
///
/// * `model` - The Bayesian model
/// * `params` - Parameter tensor (with gradient tracking enabled)
///
/// # Returns
///
/// A tuple of (log_prob_value, gradient_vector)
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::model::logp_and_grad;
///
/// let params = Tensor::zeros([model.dim()], &device).require_grad();
/// let (logp, grad) = logp_and_grad(&model, params);
/// ```
pub fn logp_and_grad<B, M>(model: &M, params: Tensor<B, 1>) -> (f64, Vec<f64>)
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    // Try the direct (GPU-accelerated) path first.
    // This bypasses Burn's autodiff entirely when the model provides
    // an optimized implementation (e.g., GPU REDUCE kernels for likelihood).
    let params_f32: Vec<f32> = params.clone().into_data().to_vec().unwrap();
    if let Some(result) = model.logp_and_grad_direct(&params_f32) {
        return result;
    }

    // Fall back to autodiff path
    let params = params.require_grad();

    // Forward pass - compute log probability
    let log_prob = model.log_prob(&params);

    // Backward pass - compute gradients
    let grads = log_prob.backward();
    let grad_tensor = params.grad(&grads).expect("Gradient should be available");

    // Extract scalar values
    let logp_data: Vec<f32> = log_prob.into_data().to_vec().unwrap();
    let logp_val: f64 = logp_data[0] as f64;
    let grad_data = grad_tensor.into_data();
    let grad_vec: Vec<f64> = grad_data.iter::<f32>().map(|x| x as f64).collect();

    (logp_val, grad_vec)
}

/// Compute batched log probability and gradients for multiple chains
///
/// This function efficiently computes log probabilities and gradients for
/// multiple parameter vectors simultaneously. This is useful for multi-chain
/// sampling where we want to leverage GPU parallelism.
///
/// # Arguments
///
/// * `model` - The Bayesian model
/// * `params` - Batched parameter tensor of shape `[num_chains, dim]`
///
/// # Returns
///
/// A tuple of:
/// - Log probabilities: `Vec<f64>` of length `num_chains`
/// - Gradients: `Vec<Vec<f64>>` where outer vec is per-chain, inner vec is per-param
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::model::batched_logp_and_grad;
///
/// // Parameters for 4 chains, each with 3 dimensions
/// let params = Tensor::zeros([4, 3], &device);
/// let (logps, grads) = batched_logp_and_grad(&model, params);
///
/// assert_eq!(logps.len(), 4);
/// assert_eq!(grads.len(), 4);
/// assert_eq!(grads[0].len(), 3);
/// ```
///
/// # Note
///
/// Currently this function processes each chain sequentially. Future optimizations
/// could leverage batched operations on the GPU for improved performance.
pub fn batched_logp_and_grad<B, M>(model: &M, params: Tensor<B, 2>) -> (Vec<f64>, Vec<Vec<f64>>)
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    let num_chains = params.dims()[0];
    let dim = params.dims()[1];

    let mut log_probs = Vec::with_capacity(num_chains);
    let mut gradients = Vec::with_capacity(num_chains);

    // Process each chain
    // NOTE: This could be optimized with true batched operations
    for i in 0..num_chains {
        // Extract parameters for this chain
        let chain_params = params.clone().slice([i..i + 1, 0..dim]).reshape([dim]);

        // Compute log prob and gradient
        let (logp, grad) = logp_and_grad(model, chain_params);

        log_probs.push(logp);
        gradients.push(grad);
    }

    (log_probs, gradients)
}

/// Compute batched log probability and gradients, returning tensors
///
/// Similar to `batched_logp_and_grad` but returns results as tensors
/// for efficient GPU computation.
///
/// # Arguments
///
/// * `model` - The Bayesian model
/// * `params` - Batched parameter tensor of shape `[num_chains, dim]`
///
/// # Returns
///
/// A tuple of:
/// - Log probabilities tensor: shape `[num_chains]`
/// - Gradients tensor: shape `[num_chains, dim]`
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::model::batched_logp_and_grad_tensor;
///
/// let params = Tensor::zeros([4, 3], &device);
/// let (logp_tensor, grad_tensor) = batched_logp_and_grad_tensor(&model, params);
///
/// assert_eq!(logp_tensor.dims(), [4]);
/// assert_eq!(grad_tensor.dims(), [4, 3]);
/// ```
pub fn batched_logp_and_grad_tensor<B, M>(
    model: &M,
    params: Tensor<B, 2>,
) -> (Tensor<B, 1>, Tensor<B, 2>)
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    let device = params.device();
    let num_chains = params.dims()[0];
    let dim = params.dims()[1];

    let (log_probs, gradients) = batched_logp_and_grad(model, params);

    // Convert to tensors
    let logp_data: Vec<f32> = log_probs.iter().map(|&x| x as f32).collect();
    let logp_tensor = Tensor::<B, 1>::from_floats(logp_data.as_slice(), &device);

    let mut grad_data: Vec<f32> = Vec::with_capacity(num_chains * dim);
    for grad in gradients {
        for &g in &grad {
            grad_data.push(g as f32);
        }
    }
    let grad_tensor =
        Tensor::<B, 1>::from_floats(grad_data.as_slice(), &device).reshape([num_chains, dim]);

    (logp_tensor, grad_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    /// Simple quadratic model for testing: log p(x) = -0.5 * x^2
    /// This is a standard normal distribution
    struct QuadraticModel {
        dim: usize,
    }

    impl QuadraticModel {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl BayesianModel<TestBackend> for QuadraticModel {
        fn dim(&self) -> usize {
            self.dim
        }

        fn log_prob(&self, params: &Tensor<TestBackend, 1>) -> Tensor<TestBackend, 1> {
            // log p(x) = -0.5 * sum(x^2) = -0.5 * ||x||^2
            let squared = params.clone().powf_scalar(2.0);
            squared.mul_scalar(-0.5).sum().reshape([1])
        }

        fn param_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{}]", i)).collect()
        }
    }

    #[test]
    fn test_quadratic_model_dim() {
        let model = QuadraticModel::new(5);
        assert_eq!(model.dim(), 5);
    }

    #[test]
    fn test_quadratic_model_param_names() {
        let model = QuadraticModel::new(3);
        let names = model.param_names();
        assert_eq!(names, vec!["x[0]", "x[1]", "x[2]"]);
    }

    #[test]
    fn test_quadratic_model_log_prob_at_zero() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        let params = Tensor::<TestBackend, 1>::zeros([3], &device);
        let log_prob = model.log_prob(&params);

        let result_data: Vec<f32> = log_prob.into_data().to_vec().unwrap();
        let result: f32 = result_data[0];
        // At x=0, log p(x) = -0.5 * 0 = 0
        assert!((result - 0.0).abs() < 1e-6, "Expected 0, got {}", result);
    }

    #[test]
    fn test_quadratic_model_log_prob_at_one() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(1);

        let params = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let log_prob = model.log_prob(&params);

        let result_data: Vec<f32> = log_prob.into_data().to_vec().unwrap();
        let result: f32 = result_data[0];
        // At x=1, log p(x) = -0.5 * 1 = -0.5
        assert!(
            (result - (-0.5)).abs() < 1e-6,
            "Expected -0.5, got {}",
            result
        );
    }

    #[test]
    fn test_logp_and_grad() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        // Test at x = [1, 2, 3]
        let params = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let (logp, grad) = logp_and_grad(&model, params);

        // log p(x) = -0.5 * (1 + 4 + 9) = -7.0
        assert!((logp - (-7.0)).abs() < 1e-4, "Expected -7.0, got {}", logp);

        // grad log p(x) = -x = [-1, -2, -3]
        assert_eq!(grad.len(), 3);
        assert!(
            (grad[0] - (-1.0)).abs() < 1e-4,
            "Expected -1.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - (-2.0)).abs() < 1e-4,
            "Expected -2.0, got {}",
            grad[1]
        );
        assert!(
            (grad[2] - (-3.0)).abs() < 1e-4,
            "Expected -3.0, got {}",
            grad[2]
        );
    }

    #[test]
    fn test_default_transform() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        let params = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let transformed = model.transform(&params);

        // Default transform is identity
        let original: Vec<f32> = params.into_data().to_vec().unwrap();
        let result: Vec<f32> = transformed.into_data().to_vec().unwrap();

        for (a, b) in original.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_default_log_det_jacobian() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        let params = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let log_det_jac = model.log_det_jacobian(&params);

        // Default log det Jacobian is 0 (identity transform)
        let result_data: Vec<f32> = log_det_jac.into_data().to_vec().unwrap();
        let result: f32 = result_data[0];
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_prob_transformed() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        let params = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let result = log_prob_transformed(&model, &params);

        // With identity transform, should equal regular log_prob
        let expected: f32 = -7.0; // -0.5 * (1 + 4 + 9)
        let result_data: Vec<f32> = result.into_data().to_vec().unwrap();
        let actual: f32 = result_data[0];

        assert!((actual - expected).abs() < 1e-4);
    }

    #[test]
    fn test_batched_logp_and_grad() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(2);

        // Create batched params for 3 chains, each with 2 parameters
        // Chain 0: [1, 0] -> logp = -0.5 * 1 = -0.5
        // Chain 1: [0, 2] -> logp = -0.5 * 4 = -2.0
        // Chain 2: [1, 1] -> logp = -0.5 * 2 = -1.0
        let params =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], &device);

        let (logps, grads) = batched_logp_and_grad(&model, params);

        // Check log probabilities
        assert_eq!(logps.len(), 3);
        assert!(
            (logps[0] - (-0.5)).abs() < 1e-4,
            "Expected -0.5, got {}",
            logps[0]
        );
        assert!(
            (logps[1] - (-2.0)).abs() < 1e-4,
            "Expected -2.0, got {}",
            logps[1]
        );
        assert!(
            (logps[2] - (-1.0)).abs() < 1e-4,
            "Expected -1.0, got {}",
            logps[2]
        );

        // Check gradients
        // grad log p(x) = -x
        assert_eq!(grads.len(), 3);

        // Chain 0: [-1, 0]
        assert!((grads[0][0] - (-1.0)).abs() < 1e-4);
        assert!((grads[0][1] - 0.0).abs() < 1e-4);

        // Chain 1: [0, -2]
        assert!((grads[1][0] - 0.0).abs() < 1e-4);
        assert!((grads[1][1] - (-2.0)).abs() < 1e-4);

        // Chain 2: [-1, -1]
        assert!((grads[2][0] - (-1.0)).abs() < 1e-4);
        assert!((grads[2][1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_batched_logp_and_grad_tensor() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(2);

        let params = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0], [0.0, 2.0]], &device);

        let (logp_tensor, grad_tensor) = batched_logp_and_grad_tensor(&model, params);

        // Check shapes
        assert_eq!(logp_tensor.dims(), [2]);
        assert_eq!(grad_tensor.dims(), [2, 2]);

        // Check values
        let logps: Vec<f32> = logp_tensor.into_data().to_vec().unwrap();
        assert!((logps[0] - (-0.5)).abs() < 1e-4);
        assert!((logps[1] - (-2.0)).abs() < 1e-4);

        let grads: Vec<f32> = grad_tensor.into_data().to_vec().unwrap();
        // Chain 0: [-1, 0]
        assert!((grads[0] - (-1.0)).abs() < 1e-4);
        assert!((grads[1] - 0.0).abs() < 1e-4);
        // Chain 1: [0, -2]
        assert!((grads[2] - 0.0).abs() < 1e-4);
        assert!((grads[3] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_batched_single_chain() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel::new(3);

        // Single chain
        let params = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);

        let (logps, grads) = batched_logp_and_grad(&model, params);

        assert_eq!(logps.len(), 1);
        assert!((logps[0] - (-7.0)).abs() < 1e-4);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].len(), 3);
    }
}
