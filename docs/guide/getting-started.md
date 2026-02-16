# Getting Started

This guide will help you get up and running with BayesianGPU.

## Choose Your Platform

BayesianGPU supports multiple platforms. Choose based on your use case:

| Platform | Best For | Status |
|----------|----------|--------|
| [Rust Library](#rust-library) | High-performance applications, custom integrations | ✅ Ready |
| [Docker Images](#docker-images) | Running NumPyro or brms models with GPU | ✅ Ready |
| [Browser/WASM](#browser-wasm) | Client-side inference, education | ✅ Ready |

## Rust Library

The Rust library provides the core MCMC samplers with the best performance.

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bayesian-core = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
bayesian-sampler = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
bayesian-diagnostics = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
bayesian-rng = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
```

### Basic Usage

```rust
use burn::tensor::Tensor;
use burn::backend::NdArray;
use bayesian_sampler::{
    model::BayesianModel,
    hmc::{HmcConfig, HmcSampler},
};

type Backend = NdArray<f32>;

// Define your model
struct MyModel;

impl BayesianModel<Backend> for MyModel {
    fn dim(&self) -> usize {
        2  // Number of parameters
    }

    fn logp_and_grad(
        &self,
        params: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 1>, Tensor<Backend, 1>) {
        // Compute log probability and gradient
        let device = params.device();

        // Example: standard normal prior
        let logp = params.clone()
            .powf_scalar(2.0)
            .sum()
            .mul_scalar(-0.5);
        let grad = params.mul_scalar(-1.0);

        (logp, grad)
    }
}

fn main() {
    let model = MyModel;
    let device = Default::default();

    // Configure sampler
    let config = HmcConfig {
        step_size: 0.1,
        num_leapfrog_steps: 10,
        num_samples: 1000,
        num_warmup: 500,
    };

    // Initialize parameters
    let init_params = Tensor::<Backend, 1>::zeros([2], &device);

    // Sample
    let sampler = HmcSampler::new(config);
    let result = sampler.sample(&model, init_params);

    println!("Acceptance rate: {:.2}", result.acceptance_rate);
}
```

### Checking Convergence

```rust
use bayesian_diagnostics::{rhat, ess_bulk, ess_tail};

// Extract samples as f64 slices
let chain1: Vec<f64> = result.samples.iter()
    .map(|t| t.clone().into_scalar() as f64)
    .collect();

let chains = vec![chain1.as_slice()];

// Compute diagnostics
let r_hat = rhat(&chains);
let ess = ess_bulk(&chains);

println!("R-hat: {:.3}", r_hat);
println!("ESS: {:.0}", ess);

// Good convergence: R-hat < 1.01, ESS > 400
```

## Docker Images

For Python (NumPyro) or R (brms) workflows, use our pre-built Docker images.

### NumPyro (Python/JAX)

```bash
# Pull the image
docker pull bayesiangpu/numpyro:latest

# Run with GPU
docker run --gpus all -v $(pwd):/workspace bayesiangpu/numpyro \
    python /workspace/model.py
```

Example model (`model.py`):

```python
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))

    with numpyro.plate("data", len(data)):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)

# Generate some data
data = jnp.array([1.2, 2.3, 1.8, 2.1, 1.9])

# Run MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(random.PRNGKey(0), data)

# Print summary
mcmc.print_summary()
```

### brms (R/Stan)

```bash
# Pull the image
docker pull bayesiangpu/brms:latest

# Run
docker run -v $(pwd):/workspace bayesiangpu/brms \
    Rscript /workspace/model.R
```

Example model (`model.R`):

```r
library(brms)

# Load data
data <- data.frame(y = c(1.2, 2.3, 1.8, 2.1, 1.9))

# Fit model
fit <- brm(
  y ~ 1,
  data = data,
  family = gaussian(),
  prior = c(
    prior(normal(0, 10), class = "Intercept"),
    prior(exponential(0.2), class = "sigma")
  ),
  chains = 4,
  iter = 2000,
  warmup = 1000
)

# Print summary
summary(fit)
```

## Browser/WASM

Run Bayesian inference directly in the browser with the WASM bindings.

### Building

```bash
cd js && npm install

# CPU-only build (ndarray backend, works everywhere)
npm run build:cpu

# WebGPU build (wgpu backend, GPU-accelerated)
npm run build:gpu
```

### Usage

```javascript
import init, { run_inference, version } from 'bayesiangpu';

await init();
console.log('BayesianGPU', version());

const model = {
  priors: [
    { name: "theta", distribution: { type: "Beta", params: { alpha: 2, beta: 2 } } }
  ],
  likelihood: {
    distribution: { type: "Binomial", params: { n: 10, p: "theta" } },
    observed: [7]
  }
};

const result = JSON.parse(
  run_inference(JSON.stringify(model), JSON.stringify({ numSamples: 1000 }))
);

console.log('Posterior mean:', result.samples.theta.reduce((a,b) => a+b) / result.samples.theta.length);
```

See `examples/browser/wasm-test.html` for a full browser example and `examples/node-wasm-test.mjs` for Node.js usage.

## Next Steps

- [Understand MCMC Sampling](/guide/mcmc-sampling)
- [Learn about Diagnostics](/guide/diagnostics)
- [See Examples](/examples/linear-regression)
- [API Reference](/reference/rust-api)
