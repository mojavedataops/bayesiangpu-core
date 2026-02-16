# What is BayesianGPU?

BayesianGPU is a library for running GPU-accelerated Bayesian inference across multiple platforms: Rust applications and web browsers.

## The Problem

Bayesian inference is powerful but computationally expensive. Modern MCMC algorithms like NUTS (No-U-Turn Sampler) require many gradient evaluations, making them slow on CPUs. While GPU-accelerated solutions exist (NumPyro, Stan), they require:

- Python or R runtime
- Server infrastructure
- Complex deployment

For many use cases—education, client-side privacy, embedded systems—you need Bayesian inference without these dependencies.

## Our Solution

BayesianGPU provides:

1. **Rust-native MCMC** - Production-ready HMC and NUTS samplers that compile to any target
2. **Browser inference** - Run sampling in WebGPU (coming v0.2.0)
3. **Docker images** - Pre-built images for NumPyro and brms with GPU support

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    bayesian-core                         │
│              (Distributions, Traits)                     │
├─────────────────────────────────────────────────────────┤
│                    bayesian-rng                          │
│              (GPU Random Numbers)                        │
├─────────────────────────────────────────────────────────┤
│                   bayesian-sampler                       │
│              (HMC, NUTS, Adaptation)                     │
├─────────────────────────────────────────────────────────┤
│                 bayesian-diagnostics                     │
│              (R-hat, ESS, Divergences)                   │
└─────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐        ┌──────────┐
   │  Rust   │         │  WASM/   │        │  Docker  │
   │  Apps   │         │  Browser │        │  Images  │
   └─────────┘         └──────────┘        └──────────┘
```

## Key Features

### Production-Ready Samplers

Our MCMC implementation follows best practices:

- **NUTS** with U-turn detection and multinomial sampling
- **Dual averaging** for automatic step size tuning
- **Mass matrix adaptation** during warmup
- **Multi-chain** execution for convergence diagnostics

### ArviZ-Compatible Diagnostics

We match the reference implementations from [ArviZ](https://arviz-devs.github.io/arviz/):

- **R-hat**: Split-chain and rank-normalized variants (Vehtari et al. 2021)
- **ESS**: Bulk and tail effective sample size with proper multi-chain pooling
- **Divergences**: Tracking and analysis

## Comparison

| Feature | BayesianGPU | NumPyro | Stan |
|---------|-------------|---------|------|
| Language | Rust | Python | Stan/R/Python |
| Browser support | Coming v0.2.0 | No | No |
| GPU acceleration | Yes (Burn) | Yes (JAX) | Limited |
| Docker images | ✅ | Manual | Manual |
| Compile to native | ✅ | No | No |

## When to Use BayesianGPU

**Good fit:**
- Rust applications needing Bayesian inference
- Browser-based statistics education
- Privacy-sensitive inference (data stays client-side)
- Cross-platform deployment

**Consider alternatives:**
- Complex hierarchical models → NumPyro has more distributions
- R-based workflows → brms is more mature
- Python ecosystem integration → NumPyro or PyMC

## Current Limitations

BayesianGPU is in alpha (v0.1.0):

- **Browser support blocked**: WASM compilation has a known issue
- **Limited distributions**: Only Normal is implemented (more coming v0.2.0)
- **No variational inference**: MCMC only for now

See the [Roadmap](/roadmap) for planned features.

## Getting Started

Ready to try it? Head to the [Getting Started](/guide/getting-started) guide.
