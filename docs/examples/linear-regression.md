# Linear Regression

This example shows how to fit a Bayesian linear regression model using BayesianGPU.

## The Model

For data $(x_i, y_i)$, we model:

$$
y_i \sim \text{Normal}(\alpha + \beta x_i, \sigma)
$$

With priors:
- $\alpha \sim \text{Normal}(0, 10)$ (intercept)
- $\beta \sim \text{Normal}(0, 5)$ (slope)
- $\sigma \sim \text{HalfNormal}(5)$ (noise)

## Rust Implementation

```rust
use burn::tensor::Tensor;
use burn::backend::NdArray;
use bayesian_sampler::{
    model::BayesianModel,
    nuts::{NutsConfig, NutsSampler},
};
use bayesian_diagnostics::{rhat, ess_bulk, PosteriorSummary};

type Backend = NdArray<f32>;

/// Linear regression model: y = alpha + beta * x + noise
struct LinearRegression {
    x: Vec<f32>,
    y: Vec<f32>,
}

impl LinearRegression {
    fn new(x: Vec<f32>, y: Vec<f32>) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have same length");
        Self { x, y }
    }
}

impl BayesianModel<Backend> for LinearRegression {
    fn dim(&self) -> usize {
        3  // alpha, beta, log_sigma
    }

    fn logp_and_grad(
        &self,
        params: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 1>, Tensor<Backend, 1>) {
        let device = params.device();
        let n = self.x.len() as f32;

        // Extract parameters
        // params[0] = alpha, params[1] = beta, params[2] = log_sigma
        let params_data: Vec<f32> = params.clone().into_data().to_vec().unwrap();
        let alpha = params_data[0];
        let beta = params_data[1];
        let log_sigma = params_data[2];
        let sigma = log_sigma.exp();

        // Prior: alpha ~ Normal(0, 10)
        let prior_alpha = -0.5 * (alpha / 10.0).powi(2);

        // Prior: beta ~ Normal(0, 5)
        let prior_beta = -0.5 * (beta / 5.0).powi(2);

        // Prior: sigma ~ HalfNormal(5)
        // log p(sigma) = -0.5 * (sigma/5)^2 + log(sigma) (Jacobian for log transform)
        let prior_sigma = -0.5 * (sigma / 5.0).powi(2) + log_sigma;

        // Likelihood: y ~ Normal(alpha + beta*x, sigma)
        let mut lik = 0.0;
        for i in 0..self.x.len() {
            let mu = alpha + beta * self.x[i];
            let residual = self.y[i] - mu;
            lik += -0.5 * (residual / sigma).powi(2) - log_sigma;
        }

        let logp = prior_alpha + prior_beta + prior_sigma + lik;

        // Gradients (computed analytically for efficiency)
        let mut grad_alpha = -alpha / 100.0;  // prior gradient
        let mut grad_beta = -beta / 25.0;     // prior gradient
        let mut grad_log_sigma = -sigma.powi(2) / 25.0 + 1.0;  // prior gradient

        for i in 0..self.x.len() {
            let mu = alpha + beta * self.x[i];
            let residual = self.y[i] - mu;

            grad_alpha += residual / sigma.powi(2);
            grad_beta += residual * self.x[i] / sigma.powi(2);
            grad_log_sigma += (residual.powi(2) / sigma.powi(2) - 1.0);
        }

        let logp_tensor = Tensor::<Backend, 1>::from_floats([logp], &device);
        let grad_tensor = Tensor::<Backend, 1>::from_floats(
            [grad_alpha, grad_beta, grad_log_sigma],
            &device
        );

        (logp_tensor, grad_tensor)
    }
}

fn main() {
    // Simulated data: y = 2 + 1.5*x + noise
    let x: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
    let y: Vec<f32> = x.iter()
        .map(|&xi| 2.0 + 1.5 * xi + rand::random::<f32>() * 0.5)
        .collect();

    let model = LinearRegression::new(x.clone(), y.clone());
    let device = Default::default();

    // Configure NUTS
    let config = NutsConfig {
        max_tree_depth: 10,
        target_accept: 0.8,
        num_samples: 2000,
        num_warmup: 1000,
    };

    // Initialize at zero
    let init_params = Tensor::<Backend, 1>::from_floats([0.0, 0.0, 0.0], &device);

    // Sample
    println!("Running NUTS...");
    let sampler = NutsSampler::new(config);
    let result = sampler.sample(&model, init_params);

    println!("\nSampling complete!");
    println!("Acceptance rate: {:.2}", result.acceptance_rate);
    println!("Divergences: {}", result.divergences);

    // Extract chains for diagnostics
    let alpha_samples: Vec<f64> = result.samples.iter()
        .map(|t| t.clone().slice([0..1]).into_scalar() as f64)
        .collect();
    let beta_samples: Vec<f64> = result.samples.iter()
        .map(|t| t.clone().slice([1..2]).into_scalar() as f64)
        .collect();
    let log_sigma_samples: Vec<f64> = result.samples.iter()
        .map(|t| t.clone().slice([2..3]).into_scalar() as f64)
        .collect();

    // Compute diagnostics
    let chains_alpha = vec![alpha_samples.as_slice()];
    let chains_beta = vec![beta_samples.as_slice()];

    println!("\nDiagnostics:");
    println!("Alpha - R-hat: {:.3}, ESS: {:.0}",
        rhat(&chains_alpha), ess_bulk(&chains_alpha));
    println!("Beta  - R-hat: {:.3}, ESS: {:.0}",
        rhat(&chains_beta), ess_bulk(&chains_beta));

    // Posterior summary
    let alpha_summary = PosteriorSummary::from_chains(&chains_alpha);
    let beta_summary = PosteriorSummary::from_chains(&chains_beta);

    println!("\nPosterior Summary:");
    println!("Alpha (intercept):");
    println!("  Mean: {:.3}, Std: {:.3}", alpha_summary.mean, alpha_summary.std);
    println!("  95% CI: [{:.3}, {:.3}]", alpha_summary.q025, alpha_summary.q975);

    println!("Beta (slope):");
    println!("  Mean: {:.3}, Std: {:.3}", beta_summary.mean, beta_summary.std);
    println!("  95% CI: [{:.3}, {:.3}]", beta_summary.q025, beta_summary.q975);

    // Transform log_sigma back to sigma
    let sigma_samples: Vec<f64> = log_sigma_samples.iter()
        .map(|&ls| ls.exp())
        .collect();
    let sigma_mean: f64 = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    println!("Sigma (noise std): {:.3}", sigma_mean);
}
```

## Docker (NumPyro)

For production use, the NumPyro Docker image is recommended:

```python
# linear_regression.py
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

def linear_regression(x, y=None):
    """Bayesian linear regression model."""
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))

    mu = alpha + beta * x
    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

# Generate data
rng = random.PRNGKey(42)
n = 50
x = jnp.linspace(0, 10, n)
true_alpha, true_beta, true_sigma = 2.0, 1.5, 0.5
y = true_alpha + true_beta * x + random.normal(rng, (n,)) * true_sigma

# Run MCMC
kernel = NUTS(linear_regression)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(random.PRNGKey(0), x, y)

# Summary
mcmc.print_summary()

# ArviZ diagnostics
idata = az.from_numpyro(mcmc)
print(az.summary(idata))
```

Run with:
```bash
docker run --gpus all -v $(pwd):/workspace bayesiangpu/numpyro \
    python /workspace/linear_regression.py
```

## Expected Output

```
                  mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha        2.03      0.15      2.03      1.79      2.28   3521.36      1.00
      beta        1.49      0.03      1.49      1.45      1.54   3498.12      1.00
     sigma        0.51      0.05      0.51      0.43      0.60   3312.45      1.00
```

The posterior recovers the true parameters (α=2, β=1.5, σ=0.5) with appropriate uncertainty.

## Key Points

1. **Reparameterization**: We use `log_sigma` instead of `sigma` to avoid boundary issues
2. **Gradients**: Analytical gradients are faster than autodiff for simple models
3. **Diagnostics**: Always check R-hat (<1.01) and ESS (>400)
4. **Warmup**: Half the iterations for adaptation is typical

## Next Steps

- [Hierarchical Models](/examples/hierarchical-models) - Partial pooling
- [Logistic Regression](/examples/logistic-regression) - Binary outcomes
- [Diagnostics Guide](/guide/diagnostics) - Interpreting results
