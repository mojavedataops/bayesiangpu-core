//! Multi-chain MCMC orchestration
//!
//! This module provides infrastructure for running multiple MCMC chains in parallel.
//! Multi-chain sampling is essential for:
//!
//! - **Convergence diagnostics**: R-hat requires multiple chains
//! - **Parallelism**: Chains run independently, enabling GPU workgroup parallelization
//! - **Robustness**: Multiple starting points help detect multimodality
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! | MultiChainSampler |
//! |  - model          |
//! |  - config         |
//! +--------+----------+
//!          |
//!    spawn N chains
//!          |
//!    +-----+-----+-----+
//!    |     |     |     |
//!   v     v     v     v
//! Chain0 Chain1 ... ChainN-1
//! (seed0)(seed1)    (seedN-1)
//!    |     |     |     |
//!    +-----+-----+-----+
//!          |
//!          v
//! +-------------------+
//! | MultiChainResult  |
//! |  - stacked samples|
//! |  - per-chain info |
//! +-------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use bayesian_sampler::{MultiChainSampler, MultiChainConfig, NutsConfig};
//!
//! let model = MyModel::new(data);
//! let config = MultiChainConfig {
//!     num_chains: 4,
//!     sampler_config: NutsConfig::default(),
//!     base_seed: 42,
//! };
//!
//! let sampler = MultiChainSampler::new(model, config);
//! let inits = sampler.generate_inits(&device);
//! let result = sampler.sample(inits);
//!
//! // Analyze convergence
//! println!("R-hat ready: {} chains", result.chains.len());
//! let stacked = result.stacked_samples(&device);
//! ```

use crate::model::BayesianModel;
use crate::nuts::{NutsConfig, NutsResult, NutsSampler};
use bayesian_rng::GpuRng;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Configuration for multi-chain MCMC sampling
#[derive(Debug, Clone)]
pub struct MultiChainConfig {
    /// Number of parallel chains to run
    pub num_chains: usize,
    /// Configuration for the underlying NUTS sampler
    pub sampler_config: NutsConfig,
    /// Base seed for RNG (each chain gets base_seed + chain_index)
    pub base_seed: u64,
}

impl Default for MultiChainConfig {
    fn default() -> Self {
        Self {
            num_chains: 4,
            sampler_config: NutsConfig::default(),
            base_seed: 42,
        }
    }
}

impl MultiChainConfig {
    /// Create a new multi-chain configuration
    ///
    /// # Arguments
    ///
    /// * `num_chains` - Number of parallel chains (typically 4)
    /// * `sampler_config` - Configuration for each NUTS sampler
    /// * `base_seed` - Base random seed
    pub fn new(num_chains: usize, sampler_config: NutsConfig, base_seed: u64) -> Self {
        assert!(num_chains >= 1, "Must have at least 1 chain");
        Self {
            num_chains,
            sampler_config,
            base_seed,
        }
    }

    /// Create configuration with default NUTS settings
    pub fn with_chains(num_chains: usize) -> Self {
        Self {
            num_chains,
            ..Default::default()
        }
    }
}

/// Multi-chain NUTS sampler
///
/// Orchestrates multiple independent NUTS samplers, each with its own
/// RNG stream seeded from a base seed plus chain index. This design
/// ensures reproducibility while maintaining chain independence.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend
/// * `M` - The Bayesian model type (must be Clone for multi-chain)
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::{MultiChainSampler, MultiChainConfig, BayesianModel};
///
/// let model = LinearRegression::new(x, y);
/// let config = MultiChainConfig::with_chains(4);
/// let sampler = MultiChainSampler::new(model, config);
///
/// // Generate diverse initial values
/// let inits = sampler.generate_inits(&device);
///
/// // Run all chains
/// let result = sampler.sample(inits);
///
/// println!("Total divergences: {}", result.total_divergences());
/// ```
pub struct MultiChainSampler<B: AutodiffBackend, M: BayesianModel<B> + Clone> {
    /// The Bayesian model (shared across chains via clone)
    model: M,
    /// Multi-chain configuration
    config: MultiChainConfig,
    /// Phantom marker for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend, M: BayesianModel<B> + Clone> MultiChainSampler<B, M> {
    /// Create a new multi-chain sampler
    ///
    /// # Arguments
    ///
    /// * `model` - The Bayesian model to sample from
    /// * `config` - Multi-chain configuration
    pub fn new(model: M, config: MultiChainConfig) -> Self {
        Self {
            model,
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// Run all chains and return combined results
    ///
    /// Each chain runs independently with its own RNG stream. Chains are
    /// currently run sequentially (embarrassingly parallel - GPU parallelization
    /// can be added in future).
    ///
    /// # Arguments
    ///
    /// * `inits` - Initial parameter values for each chain (length = num_chains)
    ///
    /// # Returns
    ///
    /// A [`MultiChainResult`] containing samples from all chains
    ///
    /// # Panics
    ///
    /// Panics if `inits.len() != config.num_chains`
    pub fn sample(&self, inits: Vec<Tensor<B, 1>>) -> MultiChainResult<B> {
        assert_eq!(
            inits.len(),
            self.config.num_chains,
            "Number of initial values ({}) must match number of chains ({})",
            inits.len(),
            self.config.num_chains
        );

        // Validate all inits have correct dimension
        let dim = self.model.dim();
        for (i, init) in inits.iter().enumerate() {
            assert_eq!(
                init.dims()[0],
                dim,
                "Initial value for chain {} has dimension {}, expected {}",
                i,
                init.dims()[0],
                dim
            );
        }

        // Run each chain with its own RNG stream
        // NOTE: Currently sequential - can be parallelized via rayon or
        // batched tensor operations for full GPU utilization
        let chains: Vec<NutsResult<B>> = inits
            .into_iter()
            .enumerate()
            .map(|(chain_idx, init)| {
                // Each chain gets unique seed: base_seed + chain_index
                let chain_seed = self.config.base_seed.wrapping_add(chain_idx as u64);
                let device = init.device();

                // Create RNG with unique seed for this chain
                let rng = GpuRng::<B>::new(chain_seed, dim, &device);

                // Create and run sampler for this chain
                let mut sampler =
                    NutsSampler::new(self.model.clone(), self.config.sampler_config.clone(), rng);

                sampler.sample(init)
            })
            .collect();

        MultiChainResult::new(chains, self.config.num_chains)
    }

    /// Generate diverse initial values for all chains
    ///
    /// Uses jittered values around zero with small random perturbations.
    /// This helps chains start from different regions of parameter space.
    ///
    /// # Arguments
    ///
    /// * `device` - Device for tensor creation
    ///
    /// # Returns
    ///
    /// Vector of initial parameter tensors, one per chain
    pub fn generate_inits(&self, device: &B::Device) -> Vec<Tensor<B, 1>> {
        let dim = self.model.dim();
        let mut inits = Vec::with_capacity(self.config.num_chains);

        // Create a temporary RNG for generating initial values
        let mut rng = GpuRng::<B>::new(self.config.base_seed.wrapping_add(1000), dim, device);

        for _ in 0..self.config.num_chains {
            // Generate small random perturbations around zero
            // Scale by 0.1 to start near origin but with diversity
            let noise = rng.normal(&[dim]).mul_scalar(0.1);
            inits.push(noise);
        }

        inits
    }

    /// Generate initial values from prior samples
    ///
    /// Creates initial values by sampling from a simple distribution and
    /// scaling. More sophisticated initialization would sample from the prior.
    ///
    /// # Arguments
    ///
    /// * `device` - Device for tensor creation
    /// * `scale` - Scale factor for initial values (larger = more spread)
    ///
    /// # Returns
    ///
    /// Vector of initial parameter tensors
    pub fn generate_inits_scaled(&self, device: &B::Device, scale: f32) -> Vec<Tensor<B, 1>> {
        let dim = self.model.dim();
        let mut inits = Vec::with_capacity(self.config.num_chains);

        let mut rng = GpuRng::<B>::new(self.config.base_seed.wrapping_add(2000), dim, device);

        for _ in 0..self.config.num_chains {
            let noise = rng.normal(&[dim]).mul_scalar(scale);
            inits.push(noise);
        }

        inits
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &MultiChainConfig {
        &self.config
    }

    /// Get the number of chains
    pub fn num_chains(&self) -> usize {
        self.config.num_chains
    }
}

/// Result of multi-chain MCMC sampling
///
/// Contains samples from all chains along with combined diagnostics.
/// Provides methods for extracting samples in formats suitable for
/// downstream analysis (R-hat, ESS, posterior summaries).
#[derive(Debug)]
pub struct MultiChainResult<B: Backend> {
    /// Results from each individual chain
    pub chains: Vec<NutsResult<B>>,
    /// Number of chains (for convenience)
    num_chains: usize,
}

impl<B: Backend> MultiChainResult<B> {
    /// Create a new multi-chain result
    pub fn new(chains: Vec<NutsResult<B>>, num_chains: usize) -> Self {
        Self { chains, num_chains }
    }

    /// Stack all samples into a single tensor
    ///
    /// Returns tensor of shape `[num_chains, num_samples, num_params]`
    /// suitable for computing diagnostics like R-hat.
    ///
    /// # Arguments
    ///
    /// * `device` - Device for the output tensor
    ///
    /// # Returns
    ///
    /// A 3D tensor with shape `[chains, samples, params]`
    pub fn stacked_samples(&self, device: &B::Device) -> Tensor<B, 3> {
        if self.chains.is_empty() {
            return Tensor::zeros([0, 0, 0], device);
        }

        let num_samples = self.chains[0].samples.len();
        if num_samples == 0 {
            return Tensor::zeros([self.num_chains, 0, 0], device);
        }

        let dim = self.chains[0].samples[0].dims()[0];

        // Collect all samples into a flat vector
        let mut all_data: Vec<f32> = Vec::with_capacity(self.num_chains * num_samples * dim);

        for chain in &self.chains {
            for sample in &chain.samples {
                let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
                all_data.extend(data);
            }
        }

        Tensor::<B, 1>::from_floats(all_data.as_slice(), device).reshape([
            self.num_chains,
            num_samples,
            dim,
        ])
    }

    /// Get samples for a specific parameter across all chains
    ///
    /// Returns a vector of vectors, where outer vector is per-chain
    /// and inner vector is the samples for that parameter.
    ///
    /// # Arguments
    ///
    /// * `param_idx` - Index of the parameter to extract
    ///
    /// # Returns
    ///
    /// Vector of sample vectors, one per chain
    ///
    /// # Panics
    ///
    /// Panics if `param_idx` is out of bounds
    pub fn get_param_samples(&self, param_idx: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.num_chains);

        for chain in &self.chains {
            let mut param_samples = Vec::with_capacity(chain.samples.len());

            for sample in &chain.samples {
                let dim = sample.dims()[0];
                assert!(
                    param_idx < dim,
                    "Parameter index {} out of bounds (dim = {})",
                    param_idx,
                    dim
                );

                let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
                param_samples.push(data[param_idx] as f64);
            }

            result.push(param_samples);
        }

        result
    }

    /// Get all samples flattened across chains for a parameter
    ///
    /// Useful for computing posterior summaries that don't require
    /// chain-specific information.
    ///
    /// # Arguments
    ///
    /// * `param_idx` - Index of the parameter to extract
    ///
    /// # Returns
    ///
    /// All samples for the parameter, concatenated across chains
    pub fn get_param_samples_flat(&self, param_idx: usize) -> Vec<f64> {
        self.get_param_samples(param_idx)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Get total number of divergent transitions across all chains
    pub fn total_divergences(&self) -> usize {
        self.chains.iter().map(|c| c.divergences).sum()
    }

    /// Get mean tree depth across all chains
    pub fn mean_tree_depth(&self) -> f64 {
        let total_depths: usize = self
            .chains
            .iter()
            .flat_map(|c| &c.tree_depths)
            .copied()
            .sum();
        let total_samples: usize = self.chains.iter().map(|c| c.tree_depths.len()).sum();

        if total_samples == 0 {
            0.0
        } else {
            total_depths as f64 / total_samples as f64
        }
    }

    /// Get final step sizes from all chains
    pub fn final_step_sizes(&self) -> Vec<f64> {
        self.chains.iter().map(|c| c.final_step_size).collect()
    }

    /// Get mean acceptance probability across all chains
    pub fn mean_accept_prob(&self) -> f64 {
        let sum: f64 = self.chains.iter().map(|c| c.mean_accept_prob).sum();
        sum / self.num_chains as f64
    }

    /// Get number of chains
    pub fn num_chains(&self) -> usize {
        self.num_chains
    }

    /// Get number of samples per chain
    pub fn num_samples(&self) -> usize {
        self.chains.first().map(|c| c.samples.len()).unwrap_or(0)
    }

    /// Get parameter dimension
    pub fn dim(&self) -> usize {
        self.chains
            .first()
            .and_then(|c| c.samples.first())
            .map(|s| s.dims()[0])
            .unwrap_or(0)
    }

    /// Compute sample mean for each parameter (across all chains)
    pub fn mean(&self) -> Vec<f64> {
        let dim = self.dim();
        if dim == 0 {
            return vec![];
        }

        let mut means = vec![0.0; dim];
        let total_samples = self.num_chains() * self.num_samples();

        for chain in &self.chains {
            for sample in &chain.samples {
                let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
                for (i, &val) in data.iter().enumerate() {
                    means[i] += val as f64 / total_samples as f64;
                }
            }
        }

        means
    }

    /// Compute sample standard deviation for each parameter (across all chains)
    pub fn std(&self) -> Vec<f64> {
        let dim = self.dim();
        if dim == 0 {
            return vec![];
        }

        let means = self.mean();
        let total_samples = (self.num_chains() * self.num_samples()) as f64;
        let mut variances = vec![0.0; dim];

        for chain in &self.chains {
            for sample in &chain.samples {
                let data: Vec<f32> = sample.clone().into_data().to_vec().unwrap();
                for (i, &val) in data.iter().enumerate() {
                    let diff = val as f64 - means[i];
                    variances[i] += diff * diff / (total_samples - 1.0);
                }
            }
        }

        variances.iter().map(|v| v.sqrt()).collect()
    }

    /// Check if sampling was successful (no or few divergences)
    ///
    /// # Arguments
    ///
    /// * `max_divergence_rate` - Maximum allowed divergence rate (e.g., 0.01 for 1%)
    ///
    /// # Returns
    ///
    /// True if divergence rate is below threshold
    pub fn is_healthy(&self, max_divergence_rate: f64) -> bool {
        let total_samples = self.num_chains() * self.num_samples();
        if total_samples == 0 {
            return false;
        }

        let divergence_rate = self.total_divergences() as f64 / total_samples as f64;
        divergence_rate <= max_divergence_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    /// Standard normal model for testing
    #[derive(Clone)]
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
    fn test_multi_chain_config_default() {
        let config = MultiChainConfig::default();

        assert_eq!(config.num_chains, 4);
        assert_eq!(config.base_seed, 42);
    }

    #[test]
    fn test_multi_chain_config_with_chains() {
        let config = MultiChainConfig::with_chains(8);

        assert_eq!(config.num_chains, 8);
    }

    #[test]
    fn test_multi_chain_sampler_new() {
        let model = StandardNormalModel::new(2);
        let config = MultiChainConfig::with_chains(4);
        let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);

        assert_eq!(sampler.num_chains(), 4);
        assert_eq!(sampler.model().dim(), 2);
    }

    #[test]
    fn test_generate_inits() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(3);
        let config = MultiChainConfig::with_chains(4);
        let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);

        let inits = sampler.generate_inits(&device);

        assert_eq!(inits.len(), 4);
        for init in &inits {
            assert_eq!(init.dims()[0], 3);
        }
    }

    #[test]
    fn test_generate_inits_diverse() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);
        let config = MultiChainConfig::with_chains(4);
        let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);

        let inits = sampler.generate_inits(&device);

        // Check that initial values are different across chains
        let init_data: Vec<Vec<f32>> = inits
            .iter()
            .map(|t| t.clone().into_data().to_vec().unwrap())
            .collect();

        // At least some pairs should differ
        let mut different_pairs = 0;
        for i in 0..init_data.len() {
            for j in (i + 1)..init_data.len() {
                let diff: f32 = init_data[i]
                    .iter()
                    .zip(&init_data[j])
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                if diff > 0.01 {
                    different_pairs += 1;
                }
            }
        }

        assert!(
            different_pairs > 0,
            "Initial values should be different across chains"
        );
    }

    #[test]
    fn test_multi_chain_sampling() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 50,
            num_warmup: 25,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        // Check we got results from both chains
        assert_eq!(result.num_chains(), 2);
        assert_eq!(result.num_samples(), 50);
        assert_eq!(result.dim(), 2);

        // Check samples are present
        for chain in &result.chains {
            assert_eq!(chain.samples.len(), 50);
        }
    }

    #[test]
    fn test_stacked_samples() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 20,
            num_warmup: 10,
            max_tree_depth: 3,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        let stacked = result.stacked_samples(&device);
        let dims = stacked.dims();

        assert_eq!(dims[0], 2); // num_chains
        assert_eq!(dims[1], 20); // num_samples
        assert_eq!(dims[2], 2); // dim
    }

    #[test]
    fn test_get_param_samples() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 30,
            num_warmup: 10,
            max_tree_depth: 3,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        // Get samples for parameter 0
        let param_samples = result.get_param_samples(0);

        assert_eq!(param_samples.len(), 2); // 2 chains
        assert_eq!(param_samples[0].len(), 30); // 30 samples per chain
        assert_eq!(param_samples[1].len(), 30);
    }

    #[test]
    fn test_get_param_samples_flat() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 20,
            num_warmup: 10,
            max_tree_depth: 3,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(3, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        let flat_samples = result.get_param_samples_flat(1);

        // Should have 3 chains * 20 samples = 60 total
        assert_eq!(flat_samples.len(), 60);
    }

    #[test]
    fn test_diagnostics() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 50,
            num_warmup: 25,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        // Check diagnostic methods work
        let _divergences = result.total_divergences();
        let mean_depth = result.mean_tree_depth();
        let step_sizes = result.final_step_sizes();
        let accept_prob = result.mean_accept_prob();

        assert!(mean_depth >= 1.0, "Mean tree depth should be at least 1");
        assert_eq!(step_sizes.len(), 2);
        assert!(accept_prob > 0.0 && accept_prob <= 1.0);
    }

    #[test]
    fn test_mean_and_std() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 100,
            num_warmup: 50,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        let means = result.mean();
        let stds = result.std();

        assert_eq!(means.len(), 2);
        assert_eq!(stds.len(), 2);

        // For standard normal, means should be close to 0, stds close to 1
        for &mean in &means {
            assert!(mean.abs() < 0.5, "Mean {} should be close to 0", mean);
        }

        for &std in &stds {
            assert!((std - 1.0).abs() < 0.5, "Std {} should be close to 1", std);
        }
    }

    #[test]
    fn test_is_healthy() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 50,
            num_warmup: 25,
            max_tree_depth: 5,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        let inits = sampler.generate_inits(&device);
        let result = sampler.sample(inits);

        // With a well-behaved model, should have few/no divergences
        assert!(
            result.is_healthy(0.1),
            "Sampling standard normal should be healthy"
        );
    }

    #[test]
    fn test_chains_have_different_seeds() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let sampler_config = NutsConfig {
            num_samples: 20,
            num_warmup: 10,
            max_tree_depth: 3,
            target_accept: 0.8,
            init_step_size: 0.5,
        };

        let config = MultiChainConfig::new(2, sampler_config, 42);
        let sampler = MultiChainSampler::new(model, config);

        // Use identical initial values
        let init = Tensor::<TestBackend, 1>::zeros([2], &device);
        let inits = vec![init.clone(), init];

        let result = sampler.sample(inits);

        // Even with same inits, chains should diverge due to different RNG
        let samples0 = result.get_param_samples(0);
        let samples1 = &samples0[0];
        let samples2 = &samples0[1];

        // Check that at least some samples differ
        let mut differ_count = 0;
        for (s1, s2) in samples1.iter().zip(samples2.iter()) {
            if (s1 - s2).abs() > 1e-6 {
                differ_count += 1;
            }
        }

        assert!(
            differ_count > samples1.len() / 2,
            "Chains should produce different samples due to different RNG seeds"
        );
    }

    #[test]
    #[should_panic(expected = "Number of initial values")]
    fn test_wrong_number_of_inits() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);
        let config = MultiChainConfig::with_chains(4);
        let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);

        // Only provide 2 inits when 4 are required
        let inits = vec![Tensor::zeros([2], &device), Tensor::zeros([2], &device)];

        let _ = sampler.sample(inits);
    }

    #[test]
    #[should_panic(expected = "Initial value for chain")]
    fn test_wrong_init_dimension() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(3); // Model expects dim=3
        let config = MultiChainConfig::with_chains(2);
        let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);

        // Provide wrong dimension
        let inits = vec![
            Tensor::zeros([2], &device), // Wrong: should be [3]
            Tensor::zeros([2], &device),
        ];

        let _ = sampler.sample(inits);
    }
}
