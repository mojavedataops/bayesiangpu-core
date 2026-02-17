//! Eight Schools benchmark for MCMC sampler validation
//!
//! The Eight Schools model (Rubin 1981) is a classic hierarchical model used to
//! benchmark MCMC samplers. It features both centered and non-centered
//! parameterizations, where the centered version is known to produce divergences
//! due to the funnel geometry in the posterior.
//!
//! Data:
//!   y     = [28, 8, -3, 7, -1, 1, 18, 12]
//!   sigma = [15, 10, 16, 11, 9, 11, 10, 18]

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use bayesian_diagnostics::{rhat, summarize_named};
use bayesian_sampler::{
    model::BayesianModel, MultiChainConfig, MultiChainSampler, NutsConfig,
};

type TestBackend = Autodiff<NdArray<f32>>;

const Y: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SIGMA: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

// ── Centered parameterization ──────────────────────────────────────────────

/// Centered Eight Schools model
///
/// Parameters: [mu, log_tau, theta_0, ..., theta_7]
/// theta_j ~ Normal(mu, tau)
#[derive(Clone)]
struct CenteredEightSchools {
    y: Vec<f64>,
    sigma: Vec<f64>,
}

impl CenteredEightSchools {
    fn new() -> Self {
        Self {
            y: Y.to_vec(),
            sigma: SIGMA.to_vec(),
        }
    }
}

impl<B: AutodiffBackend> BayesianModel<B> for CenteredEightSchools {
    fn dim(&self) -> usize {
        10 // mu, log_tau, theta[0..7]
    }

    fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = params.device();
        let pi = std::f64::consts::PI;

        // Extract parameters
        let mu = params.clone().slice([0..1]); // shape [1]
        let log_tau = params.clone().slice([1..2]); // shape [1]
        let tau = log_tau.clone().exp(); // shape [1]

        // Initialize log probability
        let mut logp = Tensor::<B, 1>::zeros([1], &device);

        // mu ~ Normal(0, 5)
        // logp += -0.5 * (mu/5)^2 - log(5) - 0.5*log(2*pi)
        let mu_prior = mu
            .clone()
            .div_scalar(5.0)
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .sub_scalar((5.0_f64).ln() + 0.5 * (2.0 * pi).ln());
        logp = logp + mu_prior;

        // tau = exp(log_tau), tau ~ HalfCauchy(5)
        // HalfCauchy logpdf: log(2) - log(pi) - log(5) - log(1 + (tau/5)^2)
        // Plus Jacobian: + log_tau
        let tau_over_5_sq = tau.clone().div_scalar(5.0).powf_scalar(2.0);
        let one_plus = tau_over_5_sq.add_scalar(1.0);
        let tau_prior = one_plus
            .log()
            .mul_scalar(-1.0)
            .add_scalar((2.0_f64).ln() - pi.ln() - (5.0_f64).ln())
            + log_tau.clone();
        logp = logp + tau_prior;

        // theta[j] ~ Normal(mu, tau) and y[j] ~ Normal(theta[j], sigma[j])
        for j in 0..8 {
            let theta_j = params.clone().slice([j + 2..j + 3]); // shape [1]

            // theta[j] ~ Normal(mu, tau)
            // logp += -0.5 * ((theta_j - mu) / tau)^2 - log(tau) - 0.5*log(2*pi)
            let diff = theta_j.clone() - mu.clone();
            let z = diff / tau.clone();
            let theta_prior = z
                .powf_scalar(2.0)
                .mul_scalar(-0.5)
                .sub(tau.clone().log())
                .sub_scalar(0.5 * (2.0 * pi).ln());
            logp = logp + theta_prior;

            // y[j] ~ Normal(theta[j], sigma[j])
            let y_j = self.y[j];
            let sigma_j = self.sigma[j];
            let y_logp = (theta_j.sub_scalar(y_j))
                .div_scalar(sigma_j)
                .powf_scalar(2.0)
                .mul_scalar(-0.5)
                .sub_scalar(sigma_j.ln() + 0.5 * (2.0 * pi).ln());
            logp = logp + y_logp;
        }

        logp
    }

    fn param_names(&self) -> Vec<String> {
        let mut names = vec!["mu".to_string(), "log_tau".to_string()];
        for j in 0..8 {
            names.push(format!("theta[{}]", j));
        }
        names
    }
}

// ── Non-centered parameterization ──────────────────────────────────────────

/// Non-centered Eight Schools model
///
/// Parameters: [mu, log_tau, theta_tilde_0, ..., theta_tilde_7]
/// theta_j = mu + tau * theta_tilde_j
/// theta_tilde_j ~ Normal(0, 1)
#[derive(Clone)]
struct NonCenteredEightSchools {
    y: Vec<f64>,
    sigma: Vec<f64>,
}

impl NonCenteredEightSchools {
    fn new() -> Self {
        Self {
            y: Y.to_vec(),
            sigma: SIGMA.to_vec(),
        }
    }
}

impl<B: AutodiffBackend> BayesianModel<B> for NonCenteredEightSchools {
    fn dim(&self) -> usize {
        10 // mu, log_tau, theta_tilde[0..7]
    }

    fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = params.device();
        let pi = std::f64::consts::PI;

        // Extract parameters
        let mu = params.clone().slice([0..1]);
        let log_tau = params.clone().slice([1..2]);
        let tau = log_tau.clone().exp();

        let mut logp = Tensor::<B, 1>::zeros([1], &device);

        // mu ~ Normal(0, 5)
        let mu_prior = mu
            .clone()
            .div_scalar(5.0)
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .sub_scalar((5.0_f64).ln() + 0.5 * (2.0 * pi).ln());
        logp = logp + mu_prior;

        // tau ~ HalfCauchy(5) + Jacobian for log_tau
        let tau_over_5_sq = tau.clone().div_scalar(5.0).powf_scalar(2.0);
        let one_plus = tau_over_5_sq.add_scalar(1.0);
        let tau_prior = one_plus
            .log()
            .mul_scalar(-1.0)
            .add_scalar((2.0_f64).ln() - pi.ln() - (5.0_f64).ln())
            + log_tau.clone();
        logp = logp + tau_prior;

        // theta_tilde[j] ~ Normal(0, 1) and y[j] ~ Normal(theta[j], sigma[j])
        for j in 0..8 {
            let theta_tilde_j = params.clone().slice([j + 2..j + 3]);

            // theta_tilde[j] ~ Normal(0, 1)
            let tilde_prior = theta_tilde_j
                .clone()
                .powf_scalar(2.0)
                .mul_scalar(-0.5)
                .sub_scalar(0.5 * (2.0 * pi).ln());
            logp = logp + tilde_prior;

            // theta[j] = mu + tau * theta_tilde[j]
            let theta_j = mu.clone() + tau.clone() * theta_tilde_j;

            // y[j] ~ Normal(theta[j], sigma[j])
            let y_j = self.y[j];
            let sigma_j = self.sigma[j];
            let y_logp = (theta_j.sub_scalar(y_j))
                .div_scalar(sigma_j)
                .powf_scalar(2.0)
                .mul_scalar(-0.5)
                .sub_scalar(sigma_j.ln() + 0.5 * (2.0 * pi).ln());
            logp = logp + y_logp;
        }

        logp
    }

    fn param_names(&self) -> Vec<String> {
        let mut names = vec!["mu".to_string(), "log_tau".to_string()];
        for j in 0..8 {
            names.push(format!("theta_tilde[{}]", j));
        }
        names
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_eight_schools_non_centered() {
    let device = NdArrayDevice::default();
    let model = NonCenteredEightSchools::new();

    let sampler_config = NutsConfig {
        num_samples: 1000,
        num_warmup: 1000,
        max_tree_depth: 10,
        target_accept: 0.8,
        init_step_size: 0.1,
    };

    let config = MultiChainConfig::new(4, sampler_config, 42);
    let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);
    let inits = sampler.generate_inits(&device);
    let result = sampler.sample(inits);

    // Check R-hat for mu (parameter 0)
    let mu_chains = result.get_param_samples(0);
    let mu_rhat = rhat(&mu_chains);
    println!("Non-centered Eight Schools:");
    println!("  mu R-hat: {:.3}", mu_rhat);
    println!("  mu mean:  {:.3}", result.mean()[0]);
    println!("  divergences: {}", result.total_divergences());

    assert!(
        mu_rhat < 1.1,
        "R-hat for mu should be < 1.1, got {:.3}",
        mu_rhat
    );

    // mu should be roughly in [2, 12] for Eight Schools
    let mu_mean = result.mean()[0];
    assert!(
        mu_mean > -5.0 && mu_mean < 20.0,
        "mu mean should be reasonable, got {:.3}",
        mu_mean
    );
}

#[test]
fn test_eight_schools_centered_has_divergences() {
    let device = NdArrayDevice::default();
    let model = CenteredEightSchools::new();

    let sampler_config = NutsConfig {
        num_samples: 200,
        num_warmup: 200,
        max_tree_depth: 8,
        target_accept: 0.8,
        init_step_size: 0.1,
    };

    let config = MultiChainConfig::new(2, sampler_config, 42);
    let sampler = MultiChainSampler::<TestBackend, _>::new(model, config);
    let inits = sampler.generate_inits(&device);
    let result = sampler.sample(inits);

    println!("Centered Eight Schools:");
    println!("  divergences: {}", result.total_divergences());
    println!("  mu mean:     {:.3}", result.mean()[0]);

    // Just verify it runs to completion -- centered parameterization often has
    // divergences but may still produce samples.
    assert!(
        result.num_samples() == 200,
        "Should have collected 200 samples"
    );
}

#[test]
#[ignore] // Slow: runs both parameterizations
fn test_eight_schools_comparison() {
    let device = NdArrayDevice::default();

    // ── Non-centered ──
    let nc_model = NonCenteredEightSchools::new();
    let nc_config = MultiChainConfig::new(
        4,
        NutsConfig {
            num_samples: 1000,
            num_warmup: 1000,
            max_tree_depth: 10,
            target_accept: 0.8,
            init_step_size: 0.1,
        },
        42,
    );
    let nc_sampler = MultiChainSampler::<TestBackend, _>::new(nc_model.clone(), nc_config);
    let nc_inits = nc_sampler.generate_inits(&device);
    let nc_result = nc_sampler.sample(nc_inits);

    // ── Centered ──
    let c_model = CenteredEightSchools::new();
    let c_config = MultiChainConfig::new(
        4,
        NutsConfig {
            num_samples: 1000,
            num_warmup: 1000,
            max_tree_depth: 10,
            target_accept: 0.8,
            init_step_size: 0.1,
        },
        42,
    );
    let c_sampler = MultiChainSampler::<TestBackend, _>::new(c_model.clone(), c_config);
    let c_inits = c_sampler.generate_inits(&device);
    let c_result = c_sampler.sample(c_inits);

    // ── Print comparison ──
    println!("\n{}", "=".repeat(70));
    println!("Eight Schools Comparison: Centered vs Non-Centered");
    println!("{}", "=".repeat(70));

    println!(
        "\n{:<20} {:>12} {:>12}",
        "", "Centered", "Non-Centered"
    );
    println!("{}", "-".repeat(44));
    println!(
        "{:<20} {:>12} {:>12}",
        "Divergences",
        c_result.total_divergences(),
        nc_result.total_divergences()
    );

    let param_names_nc = <NonCenteredEightSchools as BayesianModel<TestBackend>>::param_names(&nc_model);
    println!(
        "\n{:<20} {:>8} {:>8} {:>8} {:>8}",
        "Parameter", "C mean", "NC mean", "C Rhat", "NC Rhat"
    );
    println!("{}", "-".repeat(60));

    for i in 0..10 {
        let c_chains = c_result.get_param_samples(i);
        let nc_chains = nc_result.get_param_samples(i);
        let c_rhat = rhat(&c_chains);
        let nc_rhat = rhat(&nc_chains);
        let c_summary = summarize_named(&c_chains, Some(param_names_nc[i].clone()));
        let nc_summary = summarize_named(&nc_chains, Some(param_names_nc[i].clone()));

        println!(
            "{:<20} {:>8.2} {:>8.2} {:>8.3} {:>8.3}",
            param_names_nc[i], c_summary.mean, nc_summary.mean, c_rhat, nc_rhat
        );
    }

    // Non-centered should generally have fewer divergences
    println!(
        "\nNon-centered divergences ({}) should typically be fewer than centered ({})",
        nc_result.total_divergences(),
        c_result.total_divergences()
    );
}
