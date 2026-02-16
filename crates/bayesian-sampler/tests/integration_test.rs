//! Integration tests for the full MCMC sampling pipeline
//!
//! Tests end-to-end sampling with:
//! - Simple models with known posteriors
//! - Multi-chain sampling
//! - Adaptation and warmup
//! - Diagnostic computation

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::tensor::Tensor;

use bayesian_rng::GpuRng;
use bayesian_sampler::{
    hmc::{HmcConfig, HmcSampler},
    model::BayesianModel,
};

type TestBackend = Autodiff<NdArray<f32>>;

/// Simple 1D Gaussian model for testing
/// Posterior: N(data_mean, sigma^2/n)
struct SimpleGaussianModel {
    data_sum: f32,
    data_count: usize,
    prior_sigma: f32,
}

impl SimpleGaussianModel {
    fn new(data: &[f32], prior_sigma: f32) -> Self {
        Self {
            data_sum: data.iter().sum(),
            data_count: data.len(),
            prior_sigma,
        }
    }
}

impl BayesianModel<TestBackend> for SimpleGaussianModel {
    fn dim(&self) -> usize {
        1
    }

    fn log_prob(&self, params: &Tensor<TestBackend, 1>) -> Tensor<TestBackend, 1> {
        // Keep all operations as tensor ops to preserve autodiff graph
        let mu = params.clone().slice([0..1]);

        // Prior: N(0, prior_sigma^2)
        // prior_logp = -0.5 * mu^2 / prior_sigma^2
        let prior_var = self.prior_sigma * self.prior_sigma;
        let prior_logp = mu.clone().powf_scalar(2.0).mul_scalar(-0.5 / prior_var);

        // Likelihood: sum of N(mu, 1) for each data point
        // log p(data | mu) = -0.5 * sum((x_i - mu)^2)
        // = -0.5 * n * mu^2 + mu * sum(x_i) + const
        // (ignoring constants that don't depend on mu)
        let n = self.data_count as f32;
        let lik_logp = mu
            .clone()
            .powf_scalar(2.0)
            .mul_scalar(-0.5 * n)
            .add(mu.mul_scalar(self.data_sum));

        prior_logp.add(lik_logp)
    }

    fn param_names(&self) -> Vec<String> {
        vec!["mu".to_string()]
    }
}

#[test]
fn test_hmc_samples_gaussian_posterior() {
    // Data: 10 samples from N(3, 1)
    let data: Vec<f32> = vec![2.5, 3.2, 2.8, 3.5, 3.1, 2.9, 3.3, 2.7, 3.4, 3.0];
    let data_mean = data.iter().sum::<f32>() / data.len() as f32;

    let model = SimpleGaussianModel::new(&data, 10.0); // Wide prior

    let config = HmcConfig {
        step_size: 0.1,
        num_leapfrog_steps: 10,
        num_samples: 500,
        num_warmup: 200,
    };

    let device = NdArrayDevice::Cpu;
    let init_params = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let rng = GpuRng::<TestBackend>::new(42, model.dim(), &device);

    let mut sampler = HmcSampler::new(model, config, rng);
    let result = sampler.sample(init_params);

    // Check we got samples
    assert_eq!(result.samples.len(), 500);

    // Compute sample mean
    let sample_mean: f32 = result
        .samples
        .iter()
        .map(|s| s.clone().into_scalar())
        .sum::<f32>()
        / 500.0;

    // Sample mean should be close to data mean (posterior is approximately N(data_mean, 1/n))
    let tolerance = 0.5; // Allow some variance
    assert!(
        (sample_mean - data_mean).abs() < tolerance,
        "Sample mean {} should be close to data mean {}",
        sample_mean,
        data_mean
    );
}

#[test]
fn test_hmc_acceptance_rate_reasonable() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let model = SimpleGaussianModel::new(&data, 5.0);

    let config = HmcConfig {
        step_size: 0.05, // Small step size for high acceptance
        num_leapfrog_steps: 20,
        num_samples: 200,
        num_warmup: 100,
    };

    let device = NdArrayDevice::Cpu;
    let init_params = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let rng = GpuRng::<TestBackend>::new(42, model.dim(), &device);

    let mut sampler = HmcSampler::new(model, config, rng);
    let result = sampler.sample(init_params);

    // Acceptance rate should be reasonable (> 50%)
    let acceptance_rate = result.acceptance_rate;
    assert!(
        acceptance_rate > 0.5,
        "Acceptance rate {} should be > 0.5",
        acceptance_rate
    );
}

#[test]
fn test_hmc_samples_are_finite() {
    let data: Vec<f32> = vec![0.0, 0.5, -0.5, 0.2, -0.3];
    let model = SimpleGaussianModel::new(&data, 2.0);

    let config = HmcConfig {
        step_size: 0.1,
        num_leapfrog_steps: 10,
        num_samples: 100,
        num_warmup: 50,
    };

    let device = NdArrayDevice::Cpu;
    let init_params = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let rng = GpuRng::<TestBackend>::new(42, model.dim(), &device);

    let mut sampler = HmcSampler::new(model, config, rng);
    let result = sampler.sample(init_params);

    // All samples should be finite
    for (i, sample) in result.samples.iter().enumerate() {
        let val: f32 = sample.clone().into_scalar();
        assert!(
            val.is_finite(),
            "Sample {} should be finite, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_hmc_warmup_excluded_from_samples() {
    let data: Vec<f32> = vec![1.0, 1.0, 1.0];
    let model = SimpleGaussianModel::new(&data, 1.0);

    let num_samples = 100;
    let num_warmup = 50;

    let config = HmcConfig {
        step_size: 0.1,
        num_leapfrog_steps: 5,
        num_samples,
        num_warmup,
    };

    let device = NdArrayDevice::Cpu;
    let init_params = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
    let rng = GpuRng::<TestBackend>::new(42, model.dim(), &device);

    let mut sampler = HmcSampler::new(model, config, rng);
    let result = sampler.sample(init_params);

    // Should have exactly num_samples (not num_samples + num_warmup)
    assert_eq!(
        result.samples.len(),
        num_samples,
        "Should have {} samples, got {}",
        num_samples,
        result.samples.len()
    );
}
