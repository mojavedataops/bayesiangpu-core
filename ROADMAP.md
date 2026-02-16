# BayesianGPU Roadmap

This document outlines what's implemented, what's in progress, and what's planned for future releases.

## Current Release: v0.2.1

### What's Production Ready

#### Rust MCMC Library
- [x] **HMC Sampler** - Full implementation with configurable leapfrog steps
- [x] **NUTS Sampler** - No-U-Turn Sampler with recursive tree building
- [x] **Dual Averaging** - Step size adaptation (Nesterov 2009)
- [x] **Mass Matrix Adaptation** - Windowed variance estimation
- [x] **Multi-chain Sampling** - Parallel chains with batched gradients
- [x] **Leapfrog Integrator** - Symplectic integration with mass matrix support
- [x] **ADVI** - Automatic Differentiation Variational Inference

#### Distributions (14 total)
- [x] **Normal** - `Normal::new(loc, scale)`
- [x] **HalfNormal** - `HalfNormal::new(scale)`
- [x] **Beta** - `Beta::new(concentration1, concentration0)`
- [x] **Gamma** - `Gamma::new(concentration, rate)`
- [x] **Uniform** - `Uniform::new(low, high)`
- [x] **Exponential** - `Exponential::new(rate)`
- [x] **StudentT** - `StudentT::new(df, loc, scale)`
- [x] **Cauchy** - `Cauchy::new(loc, scale)`
- [x] **LogNormal** - `LogNormal::new(loc, scale)`
- [x] **MultivariateNormal** - `MultivariateNormal::new(mu, scale_tril)`
- [x] **Dirichlet** - `Dirichlet::new(concentration)`
- [x] **Multinomial** - `Multinomial::new(n, probs)`
- [x] **DirichletMultinomial** - `DirichletMultinomial::new(n, concentration)`
- [x] **StickBreaking / GEM** - `StickBreaking::new(concentration, truncation, device)`

#### Diagnostics (ArviZ-compatible)
- [x] **R-hat** - Split-chain and rank-normalized variants (Vehtari et al. 2021)
- [x] **ESS Bulk** - Effective sample size for central tendency
- [x] **ESS Tail** - Effective sample size for quantiles
- [x] **Divergence Tracking** - Count and analysis
- [x] **Posterior Summary** - Mean, std, quantiles
- [x] **LOO-CV** - Leave-one-out cross-validation
- [x] **WAIC** - Widely Applicable Information Criterion

#### WASM/Browser
- [x] **bayesian-wasm crate** - Compiled and working
- [x] **Model serialization** - JSON spec format
- [x] **JS type definitions** - TypeScript types
- [x] **CPU backend (ndarray)** - `npm run build:cpu`
- [x] **WebGPU backend (wgpu)** - `npm run build:gpu`
- [x] **Browser test page** - `examples/browser/wasm-test.html`
- [x] **Node.js testing** - `examples/node-wasm-test.mjs`

#### Language Bindings
- [x] **Python bindings** - PyO3 package (`crates/bayesian-py/`)
- [x] **R bindings** - extendr package (`crates/bayesian-r/`)

#### Docker Images
- [x] **NumPyro image** - CUDA 12.4, Python 3.12, JAX GPU
- [x] **brms image** - R 4.4, Stan, OpenCL
- [x] **CUDA variants** - 12.4 and 12.1 support
- [x] **Non-root execution** - Security hardened

#### Documentation
- [x] **VitePress site** - Guides and API reference
- [x] **TypeDoc** - JS SDK API docs

---

## v0.3.0 - Visualization & Polish (Target: Q2 2026)

### Visualization Helpers

**Goal**: Built-in plotting for common diagnostics.

**Tasks**:
- [ ] Trace plots (iterations vs parameter value)
- [ ] Posterior density plots
- [ ] Pair plots for correlations
- [ ] Rank plots for convergence
- [ ] Energy plots for HMC diagnostics
- [ ] Integration with Observable Plot
- [ ] Export to PNG/SVG

### Developer Experience

**Tasks**:
- [ ] Better error messages
- [ ] Model validation with helpful hints
- [ ] Auto-detection of divergence issues
- [ ] Suggested reparameterizations

---

## v1.0.0 - Production Release (Target: Q3 2026)

### Stability

- [ ] API stability guarantees
- [ ] Semantic versioning
- [ ] Deprecation policy
- [ ] Migration guides

### Performance

- [ ] WebGPU compute shaders for GPU-native sampling
- [ ] SIMD optimizations for CPU backend
- [ ] Memory-efficient large model support
- [ ] Streaming results for long chains

### Ecosystem

- [ ] Published to crates.io
- [ ] Published to npm
- [ ] Published to PyPI (Python bindings)
- [ ] Docker Hub official images

---

## Future Considerations (Post v1.0)

### Maybe Implement

- **Normalizing Flows** - For variational inference
- **Automatic Reparameterization** - Non-centered transforms
- **Parallel Tempering** - For multimodal posteriors
- **Sequential Monte Carlo** - SMC samplers

### Probably Not Implementing

- **PyMC backend** - Use NumPyro instead (better GPU support)
- **TensorFlow Probability** - JAX is preferred
- **Custom DSL** - Use existing PPLs (NumPyro, Stan)
- **GUI model builder** - Focus on code-first approach
- **Real-time streaming** - Batch inference is the focus

### Depends on Demand

- **Distributed MCMC** - Single-machine multi-GPU first
- **Mobile support** - Browser is the priority

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

**High-impact contributions for v0.3.0**:
1. Visualization helpers (trace plots, posteriors)
2. Better error messages and model validation
3. Write more documentation and examples
4. Performance benchmarks and optimization

**Good first issues**:
- Add property tests for existing diagnostics
- Improve error messages in model validation
- Add examples to docstrings
