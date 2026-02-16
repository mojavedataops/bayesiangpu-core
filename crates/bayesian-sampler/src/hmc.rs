//! Hamiltonian Monte Carlo (HMC) Sampler
//!
//! This module implements the basic HMC algorithm with fixed trajectory length.
//! HMC uses Hamiltonian dynamics to propose moves that explore the parameter
//! space efficiently while maintaining detailed balance.
//!
//! # Algorithm
//!
//! For each iteration:
//! 1. Sample momentum from standard normal distribution
//! 2. Compute initial Hamiltonian H(q, p) = -log_prob(q) + 0.5 * p^T * p
//! 3. Integrate using leapfrog for L steps with step size epsilon
//! 4. Compute proposed Hamiltonian H(q', p')
//! 5. Accept with probability min(1, exp(H(q, p) - H(q', p')))
//!
//! # References
//!
//! - Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
//! - Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.

use crate::leapfrog::{kinetic_energy, leapfrog};
use crate::model::{logp_and_grad, BayesianModel};
use bayesian_rng::GpuRng;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Configuration for the HMC sampler
#[derive(Debug, Clone)]
pub struct HmcConfig {
    /// Step size for leapfrog integration (epsilon)
    ///
    /// Smaller values improve acceptance rate but require more steps.
    /// Typical values: 0.001 to 0.1 depending on the target distribution.
    pub step_size: f64,

    /// Number of leapfrog steps per iteration (L)
    ///
    /// Together with step_size, determines trajectory length: L * epsilon.
    /// Typical values: 10 to 100.
    pub num_leapfrog_steps: usize,

    /// Number of samples to draw after warmup
    pub num_samples: usize,

    /// Number of warmup iterations (samples discarded)
    ///
    /// During warmup, the sampler explores the target distribution
    /// but samples are not recorded. This allows the chain to reach
    /// the typical set before collecting samples.
    pub num_warmup: usize,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            num_leapfrog_steps: 10,
            num_samples: 1000,
            num_warmup: 500,
        }
    }
}

impl HmcConfig {
    /// Create a new HMC configuration
    pub fn new(
        step_size: f64,
        num_leapfrog_steps: usize,
        num_samples: usize,
        num_warmup: usize,
    ) -> Self {
        Self {
            step_size,
            num_leapfrog_steps,
            num_samples,
            num_warmup,
        }
    }

    /// Trajectory length (L * epsilon)
    pub fn trajectory_length(&self) -> f64 {
        self.step_size * self.num_leapfrog_steps as f64
    }
}

/// Result of HMC sampling
#[derive(Debug)]
pub struct HmcResult<B: Backend> {
    /// Collected samples after warmup
    ///
    /// Vector of parameter tensors, one per sample.
    pub samples: Vec<Tensor<B, 1>>,

    /// Acceptance rate (number of accepted proposals / total proposals)
    ///
    /// A good acceptance rate for HMC is typically 0.6 to 0.9.
    /// Too low suggests step_size is too large.
    /// Too high suggests step_size could be increased for faster exploration.
    pub acceptance_rate: f64,

    /// Log probabilities at each sample point
    pub log_probs: Vec<f64>,
}

impl<B: Backend> HmcResult<B> {
    /// Get samples as a stacked tensor of shape [num_samples, dim]
    pub fn stacked_samples(&self, device: &B::Device) -> Tensor<B, 2> {
        if self.samples.is_empty() {
            return Tensor::zeros([0, 0], device);
        }

        let dim = self.samples[0].dims()[0];
        let num_samples = self.samples.len();

        // Collect all sample data
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

/// Hamiltonian Monte Carlo sampler
///
/// This sampler uses Hamiltonian dynamics to generate proposals that
/// explore the parameter space efficiently while maintaining detailed balance.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend
/// * `M` - The Bayesian model type
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::{HmcSampler, HmcConfig, BayesianModel};
/// use bayesian_rng::GpuRng;
///
/// let model = MyModel::new(data);
/// let config = HmcConfig::default();
/// let rng = GpuRng::new(42, model.dim(), &device);
///
/// let mut sampler = HmcSampler::new(model, config, rng);
/// let init = Tensor::zeros([model.dim()], &device);
/// let result = sampler.sample(init);
///
/// println!("Acceptance rate: {:.2}%", result.acceptance_rate * 100.0);
/// println!("Sample means: {:?}", result.mean());
/// ```
pub struct HmcSampler<B: AutodiffBackend, M: BayesianModel<B>> {
    /// The Bayesian model to sample from
    model: M,
    /// HMC configuration parameters
    config: HmcConfig,
    /// GPU random number generator
    rng: GpuRng<B>,
}

impl<B: AutodiffBackend, M: BayesianModel<B>> HmcSampler<B, M> {
    /// Create a new HMC sampler
    ///
    /// # Arguments
    ///
    /// * `model` - The Bayesian model to sample from
    /// * `config` - Configuration for the HMC sampler
    /// * `rng` - GPU random number generator
    pub fn new(model: M, config: HmcConfig, rng: GpuRng<B>) -> Self {
        Self { model, config, rng }
    }

    /// Run HMC sampling from an initial position
    ///
    /// # Arguments
    ///
    /// * `init` - Initial position in parameter space
    ///
    /// # Returns
    ///
    /// An [`HmcResult`] containing samples and diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the initial position dimension doesn't match the model dimension.
    pub fn sample(&mut self, init: Tensor<B, 1>) -> HmcResult<B> {
        assert_eq!(
            init.dims()[0],
            self.model.dim(),
            "Initial position dimension {} doesn't match model dimension {}",
            init.dims()[0],
            self.model.dim()
        );

        let dim = self.model.dim();
        let mut samples = Vec::with_capacity(self.config.num_samples);
        let mut log_probs = Vec::with_capacity(self.config.num_samples);
        let mut accepts = 0usize;

        let mut current = init;
        let (mut current_logp, _) = logp_and_grad(&self.model, current.clone());

        let total_steps = self.config.num_warmup + self.config.num_samples;

        for i in 0..total_steps {
            let is_warmup = i < self.config.num_warmup;

            // Sample momentum from standard normal
            let momentum = self.rng.normal(&[dim]);

            // Compute initial Hamiltonian (using identity mass matrix)
            let kinetic_init = kinetic_energy(&momentum, None);
            let h_init = -current_logp + kinetic_init;

            // Leapfrog integration
            let result = leapfrog(
                &self.model,
                current.clone(),
                momentum,
                self.config.step_size,
                self.config.num_leapfrog_steps,
            );

            // Compute proposed Hamiltonian
            let kinetic_final = kinetic_energy(&result.momentum, None);
            let h_final = -result.log_prob + kinetic_final;

            // Metropolis-Hastings accept/reject
            let log_accept_prob = h_init - h_final;

            // Sample uniform for acceptance test
            let u_tensor = self.rng.uniform(&[1]);
            let u_data: Vec<f32> = u_tensor.into_data().to_vec().unwrap();
            let u: f32 = u_data[0];
            let accept = (u as f64).ln() < log_accept_prob;

            if accept {
                current = result.position;
                current_logp = result.log_prob;

                if !is_warmup {
                    accepts += 1;
                }
            }

            // Store sample (only after warmup)
            if !is_warmup {
                samples.push(current.clone());
                log_probs.push(current_logp);
            }
        }

        let acceptance_rate = accepts as f64 / self.config.num_samples as f64;

        HmcResult {
            samples,
            acceptance_rate,
            log_probs,
        }
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &HmcConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: HmcConfig) {
        self.config = config;
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

    /// Simple quadratic model: log p(x) = -0.5 * x^2
    /// This is a standard normal distribution
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
            // log p(x) = -0.5 * ||x||^2 (standard multivariate normal)
            let squared = params.clone().powf_scalar(2.0);
            squared.mul_scalar(-0.5).sum().reshape([1])
        }

        fn param_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{}]", i)).collect()
        }
    }

    #[test]
    fn test_hmc_config_default() {
        let config = HmcConfig::default();

        assert_eq!(config.step_size, 0.1);
        assert_eq!(config.num_leapfrog_steps, 10);
        assert_eq!(config.num_samples, 1000);
        assert_eq!(config.num_warmup, 500);
    }

    #[test]
    fn test_hmc_config_trajectory_length() {
        let config = HmcConfig::new(0.05, 20, 1000, 500);
        let trajectory = config.trajectory_length();

        assert!((trajectory - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hmc_result_mean() {
        let device = NdArrayDevice::default();

        let samples = vec![
            Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device),
            Tensor::<TestBackend, 1>::from_floats([3.0, 4.0], &device),
            Tensor::<TestBackend, 1>::from_floats([5.0, 6.0], &device),
        ];

        let result = HmcResult {
            samples,
            acceptance_rate: 0.8,
            log_probs: vec![-1.0, -2.0, -3.0],
        };

        let means = result.mean();
        assert_eq!(means.len(), 2);
        assert!((means[0] - 3.0).abs() < 1e-5); // (1+3+5)/3 = 3
        assert!((means[1] - 4.0).abs() < 1e-5); // (2+4+6)/3 = 4
    }

    #[test]
    fn test_hmc_result_std() {
        let device = NdArrayDevice::default();

        // Samples: [0, 3, 6] -> mean=3, std=3
        let samples = vec![
            Tensor::<TestBackend, 1>::from_floats([0.0], &device),
            Tensor::<TestBackend, 1>::from_floats([3.0], &device),
            Tensor::<TestBackend, 1>::from_floats([6.0], &device),
        ];

        let result = HmcResult {
            samples,
            acceptance_rate: 0.8,
            log_probs: vec![-1.0, -2.0, -3.0],
        };

        let stds = result.std();
        assert_eq!(stds.len(), 1);
        assert!(
            (stds[0] - 3.0).abs() < 1e-4,
            "Expected std=3.0, got {}",
            stds[0]
        );
    }

    #[test]
    fn test_hmc_sampler_standard_normal() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let config = HmcConfig {
            step_size: 0.1,
            num_leapfrog_steps: 10,
            num_samples: 500,
            num_warmup: 100,
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = HmcSampler::new(model, config, rng);

        let init = Tensor::zeros([2], &device);
        let result = sampler.sample(init);

        // Check we got the right number of samples
        assert_eq!(result.samples.len(), 500);
        assert_eq!(result.log_probs.len(), 500);

        // Acceptance rate should be reasonable (typically > 0.5 for well-tuned HMC)
        assert!(
            result.acceptance_rate > 0.3,
            "Acceptance rate {} is too low",
            result.acceptance_rate
        );
        assert!(
            result.acceptance_rate <= 1.0,
            "Acceptance rate {} is invalid",
            result.acceptance_rate
        );

        // Sample statistics should be close to standard normal (mean=0, std=1)
        let means = result.mean();
        let stds = result.std();

        for (i, &mean) in means.iter().enumerate() {
            assert!(
                mean.abs() < 0.3,
                "Mean[{}] = {} should be close to 0",
                i,
                mean
            );
        }

        for (i, &std) in stds.iter().enumerate() {
            assert!(
                (std - 1.0).abs() < 0.3,
                "Std[{}] = {} should be close to 1",
                i,
                std
            );
        }
    }

    #[test]
    fn test_hmc_sampler_reproducible() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(1);

        let config = HmcConfig {
            step_size: 0.1,
            num_leapfrog_steps: 5,
            num_samples: 10,
            num_warmup: 5,
        };

        // First run
        let rng1 = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler1 = HmcSampler::new(StandardNormalModel::new(1), config.clone(), rng1);
        let init1 = Tensor::zeros([1], &device);
        let result1 = sampler1.sample(init1);

        // Second run with same seed
        let rng2 = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler2 = HmcSampler::new(model, config, rng2);
        let init2 = Tensor::zeros([1], &device);
        let result2 = sampler2.sample(init2);

        // Results should be identical
        for (s1, s2) in result1.samples.iter().zip(result2.samples.iter()) {
            let v1: Vec<f32> = s1.clone().into_data().to_vec().unwrap();
            let v2: Vec<f32> = s2.clone().into_data().to_vec().unwrap();

            for (a, b) in v1.iter().zip(v2.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Samples should be identical with same seed"
                );
            }
        }
    }

    #[test]
    fn test_hmc_stacked_samples() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let config = HmcConfig {
            step_size: 0.1,
            num_leapfrog_steps: 5,
            num_samples: 10,
            num_warmup: 5,
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = HmcSampler::new(model, config, rng);

        let init = Tensor::zeros([2], &device);
        let result = sampler.sample(init);

        let stacked = result.stacked_samples(&device);
        let dims = stacked.dims();

        assert_eq!(dims[0], 10); // num_samples
        assert_eq!(dims[1], 2); // dim
    }

    #[test]
    #[should_panic(expected = "Initial position dimension")]
    fn test_hmc_dimension_mismatch() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(3);

        let config = HmcConfig::default();
        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut sampler = HmcSampler::new(model, config, rng);

        // Wrong dimension - should panic
        let init = Tensor::zeros([2], &device);
        let _ = sampler.sample(init);
    }
}
