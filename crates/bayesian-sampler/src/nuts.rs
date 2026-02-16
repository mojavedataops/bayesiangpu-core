//! No-U-Turn Sampler (NUTS) implementation
//!
//! NUTS automatically adapts the number of leapfrog steps to efficiently explore
//! the target distribution without requiring manual tuning of trajectory length.
//!
//! # Algorithm
//!
//! NUTS builds a balanced binary tree by recursively doubling the trajectory in
//! random directions until a "U-turn" is detected (the trajectory starts returning
//! toward the starting point). This automatically adapts the trajectory length
//! to the local geometry of the posterior.
//!
//! # References
//!
//! - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting
//!   Path Lengths in Hamiltonian Monte Carlo. JMLR.
//! - Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.

use crate::adaptation::{DualAveraging, MassMatrixAdaptation};
use crate::leapfrog::{kinetic_energy, leapfrog_step_with_mass};
use crate::model::{logp_and_grad, BayesianModel};
use bayesian_rng::GpuRng;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// NUTS tree node containing position, momentum, and associated quantities
#[derive(Clone)]
pub struct TreeNode<B: Backend> {
    /// Position in parameter space
    pub position: Tensor<B, 1>,
    /// Momentum
    pub momentum: Tensor<B, 1>,
    /// Log probability at position
    pub log_prob: f64,
    /// Gradient of log probability
    pub gradient: Vec<f64>,
}

impl<B: Backend> TreeNode<B> {
    /// Create a new tree node
    pub fn new(
        position: Tensor<B, 1>,
        momentum: Tensor<B, 1>,
        log_prob: f64,
        gradient: Vec<f64>,
    ) -> Self {
        Self {
            position,
            momentum,
            log_prob,
            gradient,
        }
    }

    /// Compute the joint log probability (log_prob - kinetic_energy)
    /// Uses identity mass matrix
    pub fn log_joint(&self) -> f64 {
        let ke = kinetic_energy(&self.momentum, None);
        self.log_prob - ke
    }

    /// Compute the joint log probability with mass matrix preconditioning
    pub fn log_joint_with_mass(&self, inv_mass_matrix: Option<&[f64]>) -> f64 {
        let ke = kinetic_energy(&self.momentum, inv_mass_matrix);
        self.log_prob - ke
    }
}

/// Result of building a NUTS tree
pub struct TreeResult<B: Backend> {
    /// Leftmost node of the tree
    pub left: TreeNode<B>,
    /// Rightmost node of the tree
    pub right: TreeNode<B>,
    /// Proposed sample from the tree
    pub proposal: Tensor<B, 1>,
    /// Log of sum of weights for multinomial sampling
    pub log_sum_weight: f64,
    /// Number of valid (non-divergent) points in tree
    pub n_valid: usize,
    /// Whether to stop tree building (U-turn or divergence)
    pub stop: bool,
    /// Number of divergent transitions encountered
    pub n_divergent: usize,
}

/// Configuration for the NUTS sampler
#[derive(Debug, Clone)]
pub struct NutsConfig {
    /// Number of samples to draw after warmup
    pub num_samples: usize,
    /// Number of warmup iterations (samples discarded, used for adaptation)
    pub num_warmup: usize,
    /// Maximum tree depth (trajectory length = 2^max_tree_depth * step_size)
    pub max_tree_depth: usize,
    /// Target acceptance probability for step size adaptation
    pub target_accept: f64,
    /// Initial step size for leapfrog integration
    pub init_step_size: f64,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            num_warmup: 1000,
            max_tree_depth: 10,
            target_accept: 0.8,
            init_step_size: 1.0,
        }
    }
}

impl NutsConfig {
    /// Create a new NUTS configuration
    pub fn new(
        num_samples: usize,
        num_warmup: usize,
        max_tree_depth: usize,
        target_accept: f64,
        init_step_size: f64,
    ) -> Self {
        Self {
            num_samples,
            num_warmup,
            max_tree_depth,
            target_accept,
            init_step_size,
        }
    }
}

/// Result of NUTS sampling
#[derive(Debug)]
pub struct NutsResult<B: Backend> {
    /// Collected samples after warmup
    pub samples: Vec<Tensor<B, 1>>,
    /// Total number of divergent transitions
    pub divergences: usize,
    /// Tree depth at each iteration
    pub tree_depths: Vec<usize>,
    /// Final adapted step size
    pub final_step_size: f64,
    /// Log probabilities at each sample
    pub log_probs: Vec<f64>,
    /// Acceptance statistics
    pub mean_accept_prob: f64,
}

impl<B: Backend> NutsResult<B> {
    /// Get samples as a stacked tensor of shape [num_samples, dim]
    pub fn stacked_samples(&self, device: &B::Device) -> Tensor<B, 2> {
        if self.samples.is_empty() {
            return Tensor::zeros([0, 0], device);
        }

        let dim = self.samples[0].dims()[0];
        let num_samples = self.samples.len();

        let mut all_data: Vec<f32> = Vec::with_capacity(num_samples * dim);
        for sample in &self.samples {
            let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
            all_data.extend(data);
        }

        Tensor::<B, 1>::from_floats(all_data.as_slice(), device).reshape([num_samples, dim])
    }

    /// Compute sample mean for each parameter
    pub fn mean(&self) -> Vec<f64> {
        if self.samples.is_empty() {
            return vec![];
        }

        let dim = self.samples[0].dims()[0];
        let n = self.samples.len() as f64;
        let mut means = vec![0.0; dim];

        for sample in &self.samples {
            let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
            for (i, &val) in data.iter().enumerate() {
                means[i] += val as f64 / n;
            }
        }

        means
    }

    /// Compute sample standard deviation for each parameter
    pub fn std(&self) -> Vec<f64> {
        if self.samples.len() < 2 {
            return vec![0.0; self.samples.first().map(|s| s.dims()[0]).unwrap_or(0)];
        }

        let means = self.mean();
        let n = self.samples.len() as f64;
        let mut variances = vec![0.0; means.len()];

        for sample in &self.samples {
            let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
            for (i, &val) in data.iter().enumerate() {
                let diff = val as f64 - means[i];
                variances[i] += diff * diff / (n - 1.0);
            }
        }

        variances.iter().map(|v| v.sqrt()).collect()
    }
}

/// Numerically stable log-add-exp: log(exp(a) + exp(b))
pub fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY && b == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Check for U-turn condition between left and right endpoints
fn check_uturn<B: Backend>(left: &TreeNode<B>, right: &TreeNode<B>) -> bool {
    let left_pos: Vec<f32> = left.position.clone().into_data().to_vec().unwrap();
    let right_pos: Vec<f32> = right.position.clone().into_data().to_vec().unwrap();
    let left_mom: Vec<f32> = left.momentum.clone().into_data().to_vec().unwrap();
    let right_mom: Vec<f32> = right.momentum.clone().into_data().to_vec().unwrap();

    // Compute (right - left) . left_momentum and (right - left) . right_momentum
    let mut dot_left = 0.0f64;
    let mut dot_right = 0.0f64;

    for i in 0..left_pos.len() {
        let diff = (right_pos[i] - left_pos[i]) as f64;
        dot_left += diff * left_mom[i] as f64;
        dot_right += diff * right_mom[i] as f64;
    }

    // U-turn detected if either dot product is negative
    dot_left < 0.0 || dot_right < 0.0
}

/// Build a NUTS tree recursively
///
/// # Arguments
/// * `model` - The Bayesian model
/// * `node` - Starting node for tree building
/// * `direction` - Direction to build (-1 for backward, 1 for forward)
/// * `depth` - Current tree depth
/// * `step_size` - Leapfrog step size
/// * `log_slice` - Slice variable for slice sampling
/// * `inv_mass_matrix` - Optional inverse mass matrix (diagonal)
/// * `rng` - Random number generator
///
/// # Returns
/// TreeResult containing the built tree
#[allow(clippy::too_many_arguments)]
fn build_tree<B, M>(
    model: &M,
    node: TreeNode<B>,
    direction: i32,
    depth: usize,
    step_size: f64,
    log_slice: f64,
    inv_mass_matrix: Option<&[f64]>,
    rng: &mut GpuRng<B>,
) -> TreeResult<B>
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    let _device = node.position.device();
    let directed_step = step_size * direction as f64;

    if depth == 0 {
        // Base case: single leapfrog step
        let result = leapfrog_step_with_mass(
            model,
            node.position,
            node.momentum,
            directed_step,
            inv_mass_matrix,
        );

        let new_node = TreeNode::new(
            result.position.clone(),
            result.momentum.clone(),
            result.log_prob,
            result.gradient,
        );

        let log_joint = new_node.log_joint_with_mass(inv_mass_matrix);

        // Check if point is valid (passes slice condition)
        let valid = log_slice <= log_joint;

        // Check for divergence (large energy error)
        // Divergence threshold: log_joint much smaller than log_slice indicates numerical issues
        let divergent = log_joint - log_slice < -1000.0;

        TreeResult {
            left: new_node.clone(),
            right: new_node.clone(),
            proposal: result.position,
            log_sum_weight: if valid { log_joint } else { f64::NEG_INFINITY },
            n_valid: if valid { 1 } else { 0 },
            stop: divergent,
            n_divergent: if divergent { 1 } else { 0 },
        }
    } else {
        // Recursive case: build two subtrees
        let tree1 = build_tree(
            model,
            node.clone(),
            direction,
            depth - 1,
            step_size,
            log_slice,
            inv_mass_matrix,
            rng,
        );

        if tree1.stop {
            return tree1;
        }

        // Choose which endpoint to extend from based on direction
        let next_node = if direction == 1 {
            tree1.right.clone()
        } else {
            tree1.left.clone()
        };

        let tree2 = build_tree(
            model,
            next_node,
            direction,
            depth - 1,
            step_size,
            log_slice,
            inv_mass_matrix,
            rng,
        );

        // Combine trees
        let (left, right) = if direction == 1 {
            (tree1.left, tree2.right)
        } else {
            (tree2.left, tree1.right)
        };

        // Check U-turn condition on combined tree
        let uturn = check_uturn(&left, &right);

        // Multinomial sampling: select proposal from combined tree
        let log_sum_weight = log_add_exp(tree1.log_sum_weight, tree2.log_sum_weight);

        // Probability of selecting from tree2
        let accept_prob = if log_sum_weight == f64::NEG_INFINITY {
            0.0
        } else {
            (tree2.log_sum_weight - log_sum_weight).exp()
        };

        let u_tensor = rng.uniform(&[1]);
        let u_data: Vec<f32> = u_tensor.into_data().to_vec().unwrap();
        let u: f32 = u_data[0];
        let proposal = if (u as f64) < accept_prob {
            tree2.proposal
        } else {
            tree1.proposal
        };

        TreeResult {
            left,
            right,
            proposal,
            log_sum_weight,
            n_valid: tree1.n_valid + tree2.n_valid,
            stop: uturn || tree2.stop,
            n_divergent: tree1.n_divergent + tree2.n_divergent,
        }
    }
}

/// No-U-Turn Sampler
///
/// NUTS automatically adapts trajectory length by building a balanced binary tree
/// until a U-turn is detected.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend
/// * `M` - The Bayesian model type
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::{NutsSampler, NutsConfig, BayesianModel};
/// use bayesian_rng::GpuRng;
///
/// let model = MyModel::new(data);
/// let config = NutsConfig::default();
/// let rng = GpuRng::new(42, model.dim(), &device);
///
/// let mut sampler = NutsSampler::new(model, config, rng);
/// let init = Tensor::zeros([model.dim()], &device);
/// let result = sampler.sample(init);
///
/// println!("Divergences: {}", result.divergences);
/// println!("Final step size: {:.4}", result.final_step_size);
/// ```
pub struct NutsSampler<B: AutodiffBackend, M: BayesianModel<B>> {
    /// The Bayesian model to sample from
    model: M,
    /// NUTS configuration
    config: NutsConfig,
    /// GPU random number generator
    rng: GpuRng<B>,
    /// Step size adapter (dual averaging)
    step_size_adapter: DualAveraging,
    /// Mass matrix adapter
    mass_matrix_adapter: MassMatrixAdaptation,
    /// Current inverse mass matrix (diagonal elements, 1/variance)
    /// Used for momentum sampling and kinetic energy
    inv_mass_matrix: Vec<f64>,
}

impl<B: AutodiffBackend, M: BayesianModel<B>> NutsSampler<B, M> {
    /// Create a new NUTS sampler
    ///
    /// # Arguments
    ///
    /// * `model` - The Bayesian model to sample from
    /// * `config` - Configuration for the NUTS sampler
    /// * `rng` - GPU random number generator
    pub fn new(model: M, config: NutsConfig, rng: GpuRng<B>) -> Self {
        let dim = model.dim();
        let step_size_adapter = DualAveraging::new(config.init_step_size, config.target_accept);
        let mass_matrix_adapter = MassMatrixAdaptation::new(dim, config.num_warmup / 2);
        // Start with identity mass matrix (inv_mass = 1.0 for all dims)
        let inv_mass_matrix = vec![1.0; dim];

        Self {
            model,
            config,
            rng,
            step_size_adapter,
            mass_matrix_adapter,
            inv_mass_matrix,
        }
    }

    /// Run NUTS sampling from an initial position
    ///
    /// # Arguments
    ///
    /// * `init` - Initial position in parameter space
    ///
    /// # Returns
    ///
    /// A [`NutsResult`] containing samples and diagnostics.
    pub fn sample(&mut self, init: Tensor<B, 1>) -> NutsResult<B> {
        assert_eq!(
            init.dims()[0],
            self.model.dim(),
            "Initial position dimension {} doesn't match model dimension {}",
            init.dims()[0],
            self.model.dim()
        );

        let dim = self.model.dim();
        let device = init.device();
        let mut samples = Vec::with_capacity(self.config.num_samples);
        let mut log_probs = Vec::with_capacity(self.config.num_samples);
        let mut divergences = 0usize;
        let mut tree_depths = Vec::with_capacity(self.config.num_samples);
        let mut accept_probs_sum = 0.0;

        let mut current = init;
        let (mut current_logp, mut current_grad) = logp_and_grad(&self.model, current.clone());

        let total_steps = self.config.num_warmup + self.config.num_samples;

        // Pre-compute sqrt of mass matrix for momentum sampling
        // p ~ N(0, M) => p = sqrt(M) * z where z ~ N(0, I)
        // Since inv_mass = 1/m, we need sqrt(m) = 1/sqrt(inv_mass)
        let mut sqrt_mass: Vec<f64> = self
            .inv_mass_matrix
            .iter()
            .map(|&inv_m| 1.0 / inv_m.sqrt())
            .collect();

        for i in 0..total_steps {
            let is_warmup = i < self.config.num_warmup;

            // Sample momentum from N(0, M) using mass matrix
            // p_i = sqrt(m_i) * z_i where z ~ N(0, 1)
            let z = self.rng.normal(&[dim]);
            let z_data: Vec<f32> = z.into_data().to_vec().unwrap();
            let momentum_data: Vec<f32> = z_data
                .iter()
                .enumerate()
                .map(|(i, &zi)| (zi as f64 * sqrt_mass[i]) as f32)
                .collect();
            let momentum = Tensor::<B, 1>::from_floats(momentum_data.as_slice(), &device);

            // Create initial node
            let init_node = TreeNode::new(
                current.clone(),
                momentum.clone(),
                current_logp,
                current_grad.clone(),
            );

            // Compute slice variable (log of uniform on [0, exp(H)])
            let log_joint = init_node.log_joint_with_mass(Some(&self.inv_mass_matrix));
            let u_tensor = self.rng.uniform(&[1]);
            let u_data: Vec<f32> = u_tensor.into_data().to_vec().unwrap();
            let u: f32 = u_data[0];
            let log_slice = log_joint + (u as f64).ln();

            // Initialize tree
            let mut left = init_node.clone();
            let mut right = init_node.clone();
            let mut proposal = current.clone();
            let mut log_sum_weight = log_joint;
            let mut n_valid = 1usize;
            let mut stop = false;
            let mut depth = 0usize;
            let mut iter_divergences = 0usize;

            let step_size = self.step_size_adapter.step_size();

            // Build tree until U-turn or max depth
            while !stop && depth < self.config.max_tree_depth {
                // Random direction
                let dir_u_tensor = self.rng.uniform(&[1]);
                let dir_u_data: Vec<f32> = dir_u_tensor.into_data().to_vec().unwrap();
                let dir_u: f32 = dir_u_data[0];
                let direction: i32 = if dir_u < 0.5 { -1 } else { 1 };

                // Extend tree in chosen direction
                let extend_node = if direction == 1 {
                    right.clone()
                } else {
                    left.clone()
                };

                let tree = build_tree(
                    &self.model,
                    extend_node,
                    direction,
                    depth,
                    step_size,
                    log_slice,
                    Some(&self.inv_mass_matrix),
                    &mut self.rng,
                );

                if !tree.stop {
                    // Accept proposal using log-weight-based multinomial selection
                    // This preserves detailed balance by using the same acceptance
                    // probability as the recursive tree building
                    let new_log_sum = log_add_exp(log_sum_weight, tree.log_sum_weight);
                    let accept_prob = if new_log_sum == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (tree.log_sum_weight - new_log_sum).exp()
                    };

                    let accept_u_tensor = self.rng.uniform(&[1]);
                    let accept_u_data: Vec<f32> = accept_u_tensor.into_data().to_vec().unwrap();
                    let accept_u: f32 = accept_u_data[0];
                    if (accept_u as f64) < accept_prob {
                        proposal = tree.proposal;
                    }
                }

                // Update tree endpoints
                if direction == 1 {
                    right = tree.right;
                } else {
                    left = tree.left;
                }

                // Update counts
                n_valid += tree.n_valid;
                log_sum_weight = log_add_exp(log_sum_weight, tree.log_sum_weight);
                iter_divergences += tree.n_divergent;

                // Check stopping conditions
                stop = tree.stop || check_uturn(&left, &right);
                depth += 1;
            }

            // Update divergence count
            divergences += iter_divergences;

            // Compute acceptance statistic for adaptation
            // This is the average acceptance probability across tree nodes
            let tree_size = 1usize << depth; // 2^depth
            let accept_stat = (n_valid as f64) / (tree_size as f64);
            accept_probs_sum += accept_stat;

            // Update current position
            let (new_logp, new_grad) = logp_and_grad(&self.model, proposal.clone());
            current = proposal;
            current_logp = new_logp;
            current_grad = new_grad;

            // Adaptation during warmup
            if is_warmup {
                self.step_size_adapter.update(accept_stat);

                // Collect samples for mass matrix adaptation
                let sample_vec: Vec<f64> = {
                    let data: Vec<f32> = current.clone().into_data().to_vec().unwrap();
                    data.iter().map(|&x| x as f64).collect()
                };
                self.mass_matrix_adapter.add_sample(sample_vec);

                // Update mass matrix at end of warmup
                // We update it periodically during the second half of warmup
                if i >= self.config.num_warmup / 2 && self.mass_matrix_adapter.is_ready() {
                    self.inv_mass_matrix = self.mass_matrix_adapter.diagonal_mass_matrix();
                    // Update sqrt_mass for momentum sampling
                    sqrt_mass = self
                        .inv_mass_matrix
                        .iter()
                        .map(|&inv_m| 1.0 / inv_m.sqrt())
                        .collect();
                }
            } else {
                // Store sample after warmup
                samples.push(current.clone());
                log_probs.push(current_logp);
                tree_depths.push(depth);
            }
        }

        let mean_accept_prob = accept_probs_sum / total_steps as f64;

        NutsResult {
            samples,
            divergences,
            tree_depths,
            final_step_size: self.step_size_adapter.final_step_size(),
            log_probs,
            mean_accept_prob,
        }
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &NutsConfig {
        &self.config
    }

    /// Get the current step size
    pub fn step_size(&self) -> f64 {
        self.step_size_adapter.step_size()
    }

    /// Get the diagonal mass matrix
    pub fn mass_matrix(&self) -> Vec<f64> {
        self.mass_matrix_adapter.diagonal_mass_matrix()
    }

    /// Reset the RNG with a new seed
    pub fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    /// Standard normal model for testing: log p(x) = -0.5 * x^2
    struct StandardNormalModel {
        dim: usize,
    }

    impl StandardNormalModel {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl BayesianModel<TestBackend> for StandardNormalModel {
        fn dim(&self) -> usize {
            self.dim
        }

        fn log_prob(&self, params: &Tensor<TestBackend, 1>) -> Tensor<TestBackend, 1> {
            let squared = params.clone().powf_scalar(2.0);
            squared.mul_scalar(-0.5).sum().reshape([1])
        }

        fn param_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{}]", i)).collect()
        }
    }

    #[test]
    fn test_log_add_exp() {
        // Test basic cases
        let result = log_add_exp(0.0, 0.0);
        let expected = (2.0f64).ln();
        assert!((result - expected).abs() < 1e-10);

        // Test with one very negative value
        let result = log_add_exp(0.0, -1000.0);
        assert!((result - 0.0).abs() < 1e-10);

        // Test with both negative infinity
        let result = log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!(result == f64::NEG_INFINITY);

        // Test commutativity
        let a = 1.5;
        let b = -2.3;
        assert!((log_add_exp(a, b) - log_add_exp(b, a)).abs() < 1e-10);
    }

    #[test]
    fn test_nuts_config_default() {
        let config = NutsConfig::default();

        assert_eq!(config.num_samples, 1000);
        assert_eq!(config.num_warmup, 1000);
        assert_eq!(config.max_tree_depth, 10);
        assert!((config.target_accept - 0.8).abs() < 1e-10);
        assert!((config.init_step_size - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tree_node_log_joint() {
        let device = NdArrayDevice::default();

        // Position doesn't matter for kinetic energy calculation
        let position = Tensor::<TestBackend, 1>::zeros([2], &device);

        // Momentum = [1, 2], kinetic = 0.5 * (1 + 4) = 2.5
        let momentum = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);

        let node = TreeNode::new(position, momentum, -1.0, vec![-1.0, -1.0]);

        // log_joint = log_prob - kinetic = -1.0 - 2.5 = -3.5
        let log_joint = node.log_joint();
        assert!(
            (log_joint - (-3.5)).abs() < 1e-5,
            "Expected -3.5, got {}",
            log_joint
        );
    }

    #[test]
    fn test_nuts_sampler_standard_normal() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let config = NutsConfig {
            num_samples: 200,
            num_warmup: 100,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = NutsSampler::new(model, config, rng);

        let init = Tensor::zeros([2], &device);
        let result = sampler.sample(init);

        // Check we got the right number of samples
        assert_eq!(result.samples.len(), 200);
        assert_eq!(result.log_probs.len(), 200);
        assert_eq!(result.tree_depths.len(), 200);

        // Check that final step size is reasonable
        assert!(
            result.final_step_size > 0.01,
            "Step size {} is too small",
            result.final_step_size
        );
        assert!(
            result.final_step_size < 10.0,
            "Step size {} is too large",
            result.final_step_size
        );

        // Sample statistics should be close to standard normal
        let means = result.mean();
        let stds = result.std();

        for (i, &mean) in means.iter().enumerate() {
            assert!(
                mean.abs() < 0.5,
                "Mean[{}] = {} should be close to 0",
                i,
                mean
            );
        }

        for (i, &std) in stds.iter().enumerate() {
            assert!(
                (std - 1.0).abs() < 0.5,
                "Std[{}] = {} should be close to 1",
                i,
                std
            );
        }
    }

    #[test]
    fn test_nuts_tree_depths() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(1);

        let config = NutsConfig {
            num_samples: 50,
            num_warmup: 50,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = NutsSampler::new(model, config, rng);

        let init = Tensor::zeros([1], &device);
        let result = sampler.sample(init);

        // Tree depths should all be within bounds
        for &depth in &result.tree_depths {
            assert!(depth <= 5, "Tree depth {} exceeds max_tree_depth 5", depth);
            assert!(depth >= 1, "Tree depth {} should be at least 1", depth);
        }
    }

    #[test]
    fn test_nuts_stacked_samples() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(3);

        let config = NutsConfig {
            num_samples: 20,
            num_warmup: 10,
            max_tree_depth: 3,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = NutsSampler::new(model, config, rng);

        let init = Tensor::zeros([3], &device);
        let result = sampler.sample(init);

        let stacked = result.stacked_samples(&device);
        let dims = stacked.dims();

        assert_eq!(dims[0], 20); // num_samples
        assert_eq!(dims[1], 3); // dim
    }

    #[test]
    #[should_panic(expected = "Initial position dimension")]
    fn test_nuts_dimension_mismatch() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(3);

        let config = NutsConfig::default();
        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = NutsSampler::new(model, config, rng);

        // Wrong dimension - should panic
        let init = Tensor::zeros([2], &device);
        let _ = sampler.sample(init);
    }
}
