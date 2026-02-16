# BayesianGPU

GPU-accelerated Bayesian inference for Rust, browser, Python, and R.

[![Rust CI](https://github.com/mojavedataops/bayesiangpu-core/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/mojavedataops/bayesiangpu-core/actions/workflows/rust-ci.yml)

> **Status**: v0.2.1 - Browser/WASM support complete, 14 distributions, LOO-CV/WAIC model comparison, Python/R bindings, CLI implemented

## What Works Today

### Rust MCMC Library (Production Ready)

Full HMC and NUTS samplers with comprehensive diagnostics:

```rust
use bayesian_sampler::{NutsSampler, NutsConfig};
use bayesian_diagnostics::{rhat, ess_bulk};

// Define your model implementing BayesianModel trait
let sampler = NutsSampler::new(NutsConfig::default());
let result = sampler.sample(&model, init_params);

// Check convergence
let chains: Vec<&[f64]> = /* extract chains */;
println!("R-hat: {}", rhat(&chains));
println!("ESS: {}", ess_bulk(&chains));
```

**Features:**
- NUTS with dual averaging and mass matrix adaptation
- HMC with configurable leapfrog steps
- Multi-chain sampling with batched gradients
- R-hat (split, rank-normalized) per Vehtari et al. 2021
- ESS (bulk and tail) with proper multi-chain pooling
- **14 distributions**: Normal, HalfNormal, Beta, Gamma, Exponential, Uniform, LogNormal, Cauchy, StudentT, MultivariateNormal, Dirichlet, Multinomial, DirichletMultinomial, StickBreaking/GEM
- LOO-CV and WAIC for model comparison
- Comprehensive test suite across all crates

### Docker Images (Production Ready)

GPU-accelerated containers for NumPyro and brms:

```bash
# NumPyro with CUDA 12.4
docker pull bayesiangpu/numpyro:latest
docker run --gpus all bayesiangpu/numpyro python your_model.py

# brms with Stan
docker pull bayesiangpu/brms:latest
docker run bayesiangpu/brms Rscript your_model.R
```

### Browser/WASM (v0.2.0 - Working)

Run Bayesian inference directly in the browser:

```javascript
import init, { run_inference, version } from 'bayesiangpu';

await init();
console.log('BayesianGPU', version());

const model = {
  priors: [{ name: "theta", distribution: { type: "Beta", params: { alpha: 2, beta: 2 } } }],
  likelihood: { distribution: { type: "Binomial", params: { n: 10, p: "theta" } }, observed: [7] }
};

const result = JSON.parse(run_inference(JSON.stringify(model), JSON.stringify({ numSamples: 1000 })));
console.log('Posterior mean:', result.samples.theta.reduce((a,b) => a+b) / result.samples.theta.length);
```

## What's In Development

| Feature | Status | Target |
|---------|--------|--------|
| Visualization helpers | Not started | v0.3.0 |
| npm package publish | Ready, pending release | v0.3.0 |

See [ROADMAP.md](ROADMAP.md) for the full development plan.

## Installation

### Rust Library

```toml
# Cargo.toml
[dependencies]
bayesian-core = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
bayesian-sampler = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
bayesian-diagnostics = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
```

### Docker Images

```bash
docker pull bayesiangpu/numpyro:latest
docker pull bayesiangpu/brms:latest
```

### JavaScript SDK

```bash
# Build from source (npm publish coming in v0.2.1)
cd js && npm install && npm run build
```

Or use the WASM directly:
```bash
npm run build:wasm  # Creates js/pkg/
```

## Architecture

```
bayesiangpu-core/
├── crates/                    # Rust library
│   ├── bayesian-core/         # 14 distributions with math module
│   ├── bayesian-rng/          # GPU RNG (Box-Muller transform)
│   ├── bayesian-sampler/      # HMC/NUTS samplers with adaptation
│   ├── bayesian-diagnostics/  # R-hat, ESS (ArviZ-compatible)
│   ├── bayesian-py/           # Python bindings (PyO3/maturin)
│   └── bayesian-wasm/         # WASM bindings (working)
├── js/                        # JavaScript SDK + TypeScript types
├── images/                    # Docker images
│   ├── python/               # NumPyro + CUDA 12.4
│   └── r/                    # brms + Stan + CUDA
└── examples/                  # Usage examples
    └── browser/              # WASM test pages
```

## Development

### Prerequisites

- Rust 1.70+
- Python 3.11+ (for bayesian-py)
- Docker (for images)

### Building & Testing

```bash
# Rust
cargo build --release
cargo test --workspace          # Full test suite
cargo bench -p bayesian-sampler # Benchmarks

# WASM
npm run build:wasm              # Build WASM package
node examples/node-wasm-test.mjs # Test WASM

# Docker
docker build -f images/python/Dockerfile.numpyro -t bayesiangpu/numpyro images/python
```

### Current Limitations

1. **WebGPU testing**: WASM compiles and works with CPU backend. WebGPU backend compiles but needs real browser testing.

2. **npm not published**: JavaScript SDK works but not yet on npm. Build from source for now.

## Performance

Benchmarks on Apple M2 (NdArray backend):

| Operation | Dimension | Time |
|-----------|-----------|------|
| Leapfrog step | 10 | 14 µs |
| Leapfrog step | 100 | 44 µs |
| HMC (100 samples, 10 leapfrog steps) | 10 | 63 ms |
| Leapfrog trajectory (10 steps) | 10 | 370 µs |

## Security

- Non-root Docker containers
- No unsafe Rust in core library

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Documentation

- [Full Documentation](https://mojavedataops.github.io/bayesiangpu-core/) - Guides and API reference
- [js/docs](js/docs) - TypeDoc-generated API docs (run `npm run docs` in js/)
- [ROADMAP.md](ROADMAP.md) - Development plan
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CHANGELOG.md](CHANGELOG.md) - Release history

## Cloud Offering

A managed Bayesian inference service is available at [bayesiangpu.dev](https://bayesiangpu.dev). The cloud platform provides GPU-accelerated sampling, job management, and pre-configured environments for NumPyro and brms.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Built with [Burn](https://github.com/tracel-ai/burn), [NumPyro](https://github.com/pyro-ppl/numpyro), [brms](https://github.com/paul-buerkner/brms), and [ArviZ](https://github.com/arviz-devs/arviz).
