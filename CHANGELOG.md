# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-07

### Added

#### New Distributions
- Dirichlet distribution with concentration parameter and simplex support
- Multinomial distribution with log probability mass function
- Dirichlet-Multinomial compound distribution
- Python and R bindings for Dirichlet and Multinomial distributions

#### Model Comparison
- LOO-CV (Leave-One-Out Cross-Validation) for Bayesian model comparison
- WAIC (Widely Applicable Information Criterion) for model selection

---

## [0.2.0] - 2026-01-18 (Updated 2026-01-19)

### Added

#### WebGPU Backend Support
- Feature flags for backend selection: `ndarray` (CPU) and `wgpu` (WebGPU)
- Async WebGPU initialization with automatic CPU fallback
- Backend configuration module (`crates/bayesian-wasm/src/backend.rs`)
- Separate build scripts: `npm run build:cpu` and `npm run build:gpu`

#### Native Rust Distributions (9 total)
- Normal, HalfNormal, Exponential, Uniform
- LogNormal, Cauchy
- Gamma, Beta, StudentT
- Math module with Lanczos approximation for `ln_gamma`, `ln_beta`, `digamma`

### Fixed
- getrandom 0.3 compatibility for WASM builds (added `.cargo/config.toml`)
- Property test and ArviZ comparison test type signatures
- Test expectations for edge cases (constant chains, autocorrelated chains)
- Sampler integration tests: use tensor ops instead of scalar extraction (autodiff graph fix)
- DynamicModel inference: prior-aware initialization (Beta params start at 0.5, not 0)
- Beta distribution clamping for numerical stability
- Unused imports in bayesian-wasm

### Added (2026-01-19)
- Browser test page (`examples/browser/wasm-test.html`)
- Node.js WASM test script (`examples/node-wasm-test.mjs`)

### Changed
- Upgraded getrandom from 0.2 to 0.3 with `wasm_js` feature

## [0.1.0] - 2026-01-14

### Added

#### WebGPU Library (Rust/WASM)
- **bayesian-core**: Distribution trait with Normal distribution implementation
- **bayesian-rng**: GPU random number generation with PCG hash and Box-Muller transform
- **bayesian-sampler**: HMC and NUTS samplers with leapfrog integration
- **bayesian-diagnostics**: R-hat, ESS (bulk/tail), divergence tracking
- **bayesian-wasm**: WASM bindings for browser usage

#### JavaScript SDK
- Model DSL with fluent builder pattern
- Distribution factories (Normal, Beta, Gamma, Exponential, etc.)
- `sample()` async function for MCMC inference
- `InferenceResult` class with summary statistics
- Visualization helpers for trace plots and posteriors
- Full TypeScript type definitions

#### Docker Images
- NumPyro Docker image with CUDA 12.4
- brms Docker image with R 4.4 and Stan
- GPU-enabled variants (CUDA 12.4, 12.1)
- Non-root Docker containers

#### Testing
- Rust CI workflow with fmt, clippy, tests
- Property tests for R-hat/ESS (proptest)
- Distribution accuracy tests
- ArviZ comparison tests
- Criterion benchmarks

#### Documentation
- Comprehensive README
- JavaScript SDK API reference
- Browser examples
