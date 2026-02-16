# Introduction

BayesianGPU provides GPU-accelerated Bayesian inference for multiple platforms:

## JavaScript SDK (Browser/Node.js)

Run MCMC sampling directly in the browser using WebAssembly, with optional WebGPU acceleration.

```javascript
import { Model, Normal, sample } from 'bayesiangpu';

const model = new Model()
  .param('mu', Normal(0, 10))
  .observe(Normal('mu', 1), data)
  .build();

const result = await sample(model);
```

## Rust Library

High-performance MCMC library with the Burn framework for GPU acceleration.

```rust
use bayesian_sampler::{NutsSampler, NutsConfig};
use bayesian_diagnostics::{rhat, ess};

let sampler = NutsSampler::new(NutsConfig::default());
let result = sampler.sample(&model, init);
```

## Docker Images

Pre-built Docker images for NumPyro (Python/JAX) and brms (R/Stan) with GPU support.

```bash
# NumPyro (Python/JAX)
docker run --gpus all bayesiangpu/numpyro python model.py

# brms (R/Stan)
docker run -v $(pwd):/workspace bayesiangpu/brms Rscript /workspace/model.R
```

## Features

- **NUTS Sampler**: No U-Turn Sampler with dual averaging step size adaptation
- **Mass Matrix Adaptation**: Windowed variance estimation during warmup
- **Multi-Chain**: Parallel chains with batched gradient computation
- **Diagnostics**: R-hat, ESS (bulk/tail), divergence tracking
- **10 Distributions**: Normal, Beta, Gamma, StudentT, Cauchy, and more
