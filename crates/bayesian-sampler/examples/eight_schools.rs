//! Eight Schools benchmark example
//!
//! Demonstrates both centered and non-centered parameterizations of the
//! classic Eight Schools hierarchical model (Rubin 1981).
//!
//! The non-centered parameterization avoids the funnel geometry that causes
//! divergences in the centered version.
//!
//! Run with:
//!   cargo run --example eight_schools -p bayesian-sampler

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use bayesian_diagnostics::{ess, rhat, summarize_named};
use bayesian_sampler::{model::BayesianModel, MultiChainConfig, MultiChainSampler, NutsConfig};

type MyBackend = Autodiff<NdArray<f32>>;

const Y: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SIGMA: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

// ── Centered parameterization ──────────────────────────────────────────────

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
        10
    }

    fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = params.device();
        let pi = std::f64::consts::PI;

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

        // tau ~ HalfCauchy(5) + Jacobian
        let tau_over_5_sq = tau.clone().div_scalar(5.0).powf_scalar(2.0);
        let one_plus = tau_over_5_sq.add_scalar(1.0);
        let tau_prior = one_plus
            .log()
            .mul_scalar(-1.0)
            .add_scalar((2.0_f64).ln() - pi.ln() - (5.0_f64).ln())
            + log_tau.clone();
        logp = logp + tau_prior;

        for j in 0..8 {
            let theta_j = params.clone().slice([j + 2..j + 3]);

            // theta[j] ~ Normal(mu, tau)
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
        10
    }

    fn log_prob(&self, params: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = params.device();
        let pi = std::f64::consts::PI;

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

        // tau ~ HalfCauchy(5) + Jacobian
        let tau_over_5_sq = tau.clone().div_scalar(5.0).powf_scalar(2.0);
        let one_plus = tau_over_5_sq.add_scalar(1.0);
        let tau_prior = one_plus
            .log()
            .mul_scalar(-1.0)
            .add_scalar((2.0_f64).ln() - pi.ln() - (5.0_f64).ln())
            + log_tau.clone();
        logp = logp + tau_prior;

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

// ── Helpers ────────────────────────────────────────────────────────────────

fn print_summary(
    name: &str,
    result: &bayesian_sampler::MultiChainResult<MyBackend>,
    param_names: &[String],
) {
    println!("\n{}", "=".repeat(70));
    println!("{}", name);
    println!("{}", "=".repeat(70));
    println!(
        "  Divergences: {} / {} ({:.1}%)",
        result.total_divergences(),
        result.num_chains() * result.num_samples(),
        100.0 * result.total_divergences() as f64
            / (result.num_chains() * result.num_samples()) as f64
    );
    println!("  Mean accept prob: {:.3}", result.mean_accept_prob());
    println!("  Mean tree depth:  {:.1}", result.mean_tree_depth());

    println!(
        "\n  {:<20} {:>8} {:>8} {:>8} {:>8}",
        "Parameter", "Mean", "Std", "R-hat", "ESS"
    );
    println!("  {}", "-".repeat(56));

    for (i, pname) in param_names.iter().enumerate() {
        let chains = result.get_param_samples(i);
        let r = rhat(&chains);
        let e = ess(&chains);
        let summary = summarize_named(&chains, Some(pname.clone()));
        println!(
            "  {:<20} {:>8.2} {:>8.2} {:>8.3} {:>8.0}",
            pname, summary.mean, summary.std, r, e
        );
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("Eight Schools Benchmark");
    println!("=======================");
    println!("Data: y     = {:?}", Y);
    println!("      sigma = {:?}", SIGMA);

    let device = NdArrayDevice::default();

    let nuts_config = NutsConfig {
        num_samples: 1000,
        num_warmup: 1000,
        max_tree_depth: 10,
        target_accept: 0.8,
        init_step_size: 0.1,
    };

    // ── Non-centered ──
    println!("\nRunning Non-Centered parameterization (4 chains)...");
    let nc_model = NonCenteredEightSchools::new();
    let nc_names = <NonCenteredEightSchools as BayesianModel<MyBackend>>::param_names(&nc_model);
    let nc_config = MultiChainConfig::new(4, nuts_config.clone(), 42);
    let nc_sampler = MultiChainSampler::<MyBackend, _>::new(nc_model, nc_config);
    let nc_inits = nc_sampler.generate_inits(&device);
    let nc_result = nc_sampler.sample(nc_inits);
    print_summary("Non-Centered Eight Schools", &nc_result, &nc_names);

    // ── Centered ──
    println!("\nRunning Centered parameterization (4 chains)...");
    let c_model = CenteredEightSchools::new();
    let c_names = <CenteredEightSchools as BayesianModel<MyBackend>>::param_names(&c_model);
    let c_config = MultiChainConfig::new(4, nuts_config, 42);
    let c_sampler = MultiChainSampler::<MyBackend, _>::new(c_model, c_config);
    let c_inits = c_sampler.generate_inits(&device);
    let c_result = c_sampler.sample(c_inits);
    print_summary("Centered Eight Schools", &c_result, &c_names);

    // ── Comparison ──
    println!("\n{}", "=".repeat(70));
    println!("Comparison Summary");
    println!("{}", "=".repeat(70));
    println!(
        "  Non-centered divergences: {}",
        nc_result.total_divergences()
    );
    println!(
        "  Centered divergences:     {}",
        c_result.total_divergences()
    );
    println!("\n  The non-centered parameterization should have fewer divergences");
    println!("  and better mixing (higher ESS, R-hat closer to 1.0).");
}
