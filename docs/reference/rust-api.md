# Rust API Reference

This reference documents the public API of BayesianGPU's Rust crates.

## Crates Overview

| Crate | Purpose | Status |
|-------|---------|--------|
| `bayesian-core` | Distribution traits and implementations | ✅ Stable |
| `bayesian-rng` | GPU-accelerated random number generation | ✅ Stable |
| `bayesian-sampler` | HMC and NUTS samplers | ✅ Stable |
| `bayesian-diagnostics` | MCMC diagnostics (R-hat, ESS) | ✅ Stable |

## bayesian-core

### Distribution Trait

```rust
pub trait Distribution<B: Backend> {
    /// Compute log probability of a value
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1>;

    /// Return the support of this distribution
    fn support(&self) -> Support;
}
```

### Support

```rust
pub enum Support {
    /// All real numbers (-∞, +∞)
    Real,
    /// Positive reals (0, +∞)
    Positive,
    /// Unit interval [0, 1]
    UnitInterval,
    /// Non-negative integers {0, 1, 2, ...}
    NonNegativeInteger,
    /// Simplex (values sum to 1)
    Simplex,
}
```

### Distributions (14 total)

All distributions implement the `Distribution<B>` trait with `log_prob` and `support` methods.

#### Continuous Distributions

```rust
use bayesian_core::distributions::*;
use burn::tensor::Tensor;
use burn::backend::NdArray;

type B = NdArray<f32>;
let device = Default::default();

// Normal(loc, scale) - support: Real
let normal = Normal::<B>::new(
    Tensor::from_floats([0.0], &device),
    Tensor::from_floats([1.0], &device),
);

// HalfNormal(scale) - support: Positive
let half_normal = HalfNormal::<B>::new(
    Tensor::from_floats([1.0], &device),
);

// Beta(concentration1, concentration0) - support: UnitInterval
let beta = Beta::<B>::new(
    Tensor::from_floats([2.0], &device),
    Tensor::from_floats([5.0], &device),
);

// Gamma(concentration, rate) - support: Positive
let gamma = Gamma::<B>::new(
    Tensor::from_floats([2.0], &device),
    Tensor::from_floats([1.0], &device),
);

// Uniform(low, high) - support: Real (bounded)
let uniform = Uniform::<B>::new(
    Tensor::from_floats([0.0], &device),
    Tensor::from_floats([1.0], &device),
);

// Exponential(rate) - support: Positive
let exponential = Exponential::<B>::new(
    Tensor::from_floats([1.0], &device),
);

// StudentT(df, loc, scale) - support: Real
let student_t = StudentT::<B>::new(
    Tensor::from_floats([3.0], &device),
    Tensor::from_floats([0.0], &device),
    Tensor::from_floats([1.0], &device),
);

// Cauchy(loc, scale) - support: Real
let cauchy = Cauchy::<B>::new(
    Tensor::from_floats([0.0], &device),
    Tensor::from_floats([1.0], &device),
);

// LogNormal(loc, scale) - support: Positive
let log_normal = LogNormal::<B>::new(
    Tensor::from_floats([0.0], &device),
    Tensor::from_floats([1.0], &device),
);

// MultivariateNormal(mu, scale_tril) - support: Real (multivariate)
// scale_tril is the lower-triangular Cholesky factor of the covariance
let mvn = MultivariateNormal::<B>::new(
    Tensor::from_floats([0.0, 0.0], &device),
    Tensor::from_floats([[1.0, 0.0], [0.5, 0.866]], &device),
);
```

#### Discrete / Simplex Distributions

```rust
// Dirichlet(concentration) - support: Simplex
let dirichlet = Dirichlet::<B>::new(
    Tensor::from_floats([1.0, 2.0, 3.0], &device),
);

// Multinomial(n, probs) - support: NonNegativeInteger
let multinomial = Multinomial::<B>::new(
    10,
    Tensor::from_floats([0.2, 0.3, 0.5], &device),
);

// DirichletMultinomial(n, concentration) - compound distribution
let dm = DirichletMultinomial::<B>::new(
    10,
    Tensor::from_floats([1.0, 2.0, 3.0], &device),
);

// StickBreaking / GEM(concentration, truncation) - support: Simplex
let sb = StickBreaking::<B>::new(1.0, 10, &device);
```

#### Computing Log Probabilities

```rust
// All distributions use the same interface
let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
let log_p = normal.log_prob(&x);
```

## bayesian-rng

### GpuRng

```rust
use bayesian_rng::GpuRng;
use burn::tensor::Tensor;
use burn::backend::NdArray;

type B = NdArray<f32>;

// Create RNG with seed
let mut rng = GpuRng::<B>::new(42, &device);

// Generate uniform samples in [0, 1]
let uniform: Tensor<B, 1> = rng.uniform([1000]);

// Generate standard normal samples
let normal: Tensor<B, 1> = rng.normal([1000]);
```

The RNG uses:
- PCG hash for state initialization
- XorShift128 for state updates
- Box-Muller transform for normal samples

## bayesian-sampler

### BayesianModel Trait

Define your model by implementing this trait:

```rust
pub trait BayesianModel<B: Backend> {
    /// Number of parameters
    fn dim(&self) -> usize;

    /// Compute log probability and gradient
    fn logp_and_grad(
        &self,
        params: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>);
}
```

### HMC Sampler

```rust
use bayesian_sampler::hmc::{HmcConfig, HmcSampler, HmcResult};

// Configure HMC
let config = HmcConfig {
    step_size: 0.1,           // Leapfrog step size
    num_leapfrog_steps: 10,   // Steps per proposal
    num_samples: 1000,        // Posterior samples
    num_warmup: 500,          // Warmup iterations
};

// Create sampler
let sampler = HmcSampler::new(config);

// Run sampling
let result: HmcResult<B> = sampler.sample(&model, init_params);

// Access results
println!("Samples: {}", result.samples.len());
println!("Acceptance rate: {:.2}", result.acceptance_rate);
```

### HmcResult

```rust
pub struct HmcResult<B: Backend> {
    /// Posterior samples (after warmup)
    pub samples: Vec<Tensor<B, 1>>,

    /// Fraction of proposals accepted
    pub acceptance_rate: f32,
}
```

### NUTS Sampler

```rust
use bayesian_sampler::nuts::{NutsConfig, NutsSampler, NutsResult};

// Configure NUTS
let config = NutsConfig {
    max_tree_depth: 10,       // Maximum binary tree depth
    target_accept: 0.8,       // Target acceptance probability
    num_samples: 1000,
    num_warmup: 500,
};

// Create sampler
let sampler = NutsSampler::new(config);

// Run sampling
let result: NutsResult<B> = sampler.sample(&model, init_params);
```

### NutsResult

```rust
pub struct NutsResult<B: Backend> {
    /// Posterior samples
    pub samples: Vec<Tensor<B, 1>>,

    /// Acceptance rate
    pub acceptance_rate: f32,

    /// Number of divergent transitions
    pub divergences: usize,

    /// Average tree depth
    pub mean_tree_depth: f32,
}
```

### Multi-Chain Sampling

```rust
use bayesian_sampler::multi_chain::{
    MultiChainConfig,
    MultiChainSampler,
    MultiChainResult
};

let config = MultiChainConfig {
    num_chains: 4,
    num_samples: 1000,
    num_warmup: 500,
    sampler: SamplerType::Nuts(NutsConfig::default()),
};

let sampler = MultiChainSampler::new(config);
let result = sampler.sample(&model, init_params);

// Stack samples from all chains
let all_samples = result.stacked_samples();
```

### Leapfrog Integrator

Low-level API for custom samplers:

```rust
use bayesian_sampler::leapfrog::{leapfrog_step, leapfrog_step_with_mass};

// Single leapfrog step
let (new_position, new_momentum) = leapfrog_step(
    &model,
    position,
    momentum,
    step_size,
);

// With mass matrix
let (new_position, new_momentum) = leapfrog_step_with_mass(
    &model,
    position,
    momentum,
    step_size,
    &mass_matrix,
);
```

### Adaptation

```rust
use bayesian_sampler::adaptation::{
    DualAveraging,
    MassMatrixAdaptation,
    AdaptationSchedule,
};

// Step size adaptation
let mut dual_avg = DualAveraging::new(0.1, 0.8); // init step, target accept
dual_avg.update(accept_prob);
let adapted_step = dual_avg.step_size();

// Mass matrix adaptation
let mut mass_adapt = MassMatrixAdaptation::new(dim);
mass_adapt.add_sample(&params);
let mass_matrix = mass_adapt.get_mass_matrix();
```

## bayesian-diagnostics

### R-hat

```rust
use bayesian_diagnostics::{rhat, rhat_rank};

// Standard R-hat (split chains)
let chains: Vec<&[f64]> = vec![&chain1, &chain2, &chain3, &chain4];
let r = rhat(&chains);

// Rank-normalized R-hat (more robust)
let r_rank = rhat_rank(&chains);
```

### ESS

```rust
use bayesian_diagnostics::{ess_bulk, ess_tail};

// ESS for central tendency
let ess_b = ess_bulk(&chains);

// ESS for tail quantiles
let ess_t = ess_tail(&chains);
```

### Posterior Summary

```rust
use bayesian_diagnostics::PosteriorSummary;

let summary = PosteriorSummary::from_chains(&chains);

println!("Mean: {:.3}", summary.mean);
println!("Std:  {:.3}", summary.std);
println!("2.5%: {:.3}", summary.q025);
println!("25%:  {:.3}", summary.q25);
println!("50%:  {:.3}", summary.q50);
println!("75%:  {:.3}", summary.q75);
println!("97.5%:{:.3}", summary.q975);
```

### Diagnostic Status

```rust
use bayesian_diagnostics::DiagnosticStatus;

pub enum DiagnosticStatus {
    Good,       // All diagnostics look good
    Acceptable, // Minor issues, probably okay
    Warning,    // Should investigate
    Error,      // Serious problems
}
```

## Feature Flags

### bayesian-sampler

```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu", "burn/autodiff"]  # GPU acceleration
wasm = ["burn/wgpu", "burn/autodiff"]  # Browser/WASM (coming)
```

### bayesian-core

```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
wasm = ["burn/wgpu"]
```

## Error Handling

Most functions return results directly rather than `Result` types. Invalid inputs (e.g., empty chains) may cause panics or return NaN.

```rust
// Check for valid results
let r = rhat(&chains);
if r.is_nan() {
    eprintln!("R-hat computation failed (possibly empty chains)");
}
```

## Thread Safety

All types are `Send + Sync` when the underlying `Backend` supports it. The NdArray backend is thread-safe for reads.

## Performance Tips

1. **Use release mode**: `cargo build --release`
2. **Enable GPU**: Use the `wgpu` feature for GPU acceleration
3. **Batch operations**: Use multi-chain sampling instead of sequential
4. **Pre-allocate**: Avoid repeated allocations in `logp_and_grad`
