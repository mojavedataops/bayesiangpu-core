# MCMC Diagnostics

Understanding whether your MCMC chains have converged is crucial for valid inference. BayesianGPU provides comprehensive diagnostics matching the ArviZ reference implementation.

## Why Diagnostics Matter

MCMC samplers explore the posterior distribution iteratively. If you don't run them long enough, or if the sampler gets stuck, your results will be wrong. Diagnostics help you detect these problems.

## Key Diagnostics

### R-hat (Potential Scale Reduction Factor)

R-hat compares variance within chains to variance between chains. If chains have converged to the same distribution, R-hat ≈ 1.0.

```rust
use bayesian_diagnostics::rhat;

let chains: Vec<&[f64]> = vec![&chain1, &chain2, &chain3, &chain4];
let r_hat = rhat(&chains);

println!("R-hat: {:.3}", r_hat);
```

**Interpretation:**
| R-hat | Status | Action |
|-------|--------|--------|
| < 1.01 | ✅ Excellent | Good to go |
| 1.01 - 1.05 | ⚠️ Acceptable | Consider more samples |
| 1.05 - 1.1 | ⚠️ Warning | Run longer chains |
| > 1.1 | ❌ Problem | Chains haven't converged |

### Rank-Normalized R-hat

For heavy-tailed or skewed distributions, rank-normalized R-hat is more robust:

```rust
use bayesian_diagnostics::rhat_rank;

let r_hat_rank = rhat_rank(&chains);
```

This transforms samples to ranks before computing R-hat, making it less sensitive to outliers.

### ESS (Effective Sample Size)

ESS estimates how many independent samples your chains are worth. Due to autocorrelation, 1000 MCMC samples might only be worth 100 independent samples.

```rust
use bayesian_diagnostics::{ess_bulk, ess_tail};

// ESS for central tendency (mean, median)
let ess_b = ess_bulk(&chains);

// ESS for extreme quantiles (5%, 95%)
let ess_t = ess_tail(&chains);

println!("ESS bulk: {:.0}", ess_b);
println!("ESS tail: {:.0}", ess_t);
```

**Guidelines:**
| ESS | Status | Notes |
|-----|--------|-------|
| > 400 | ✅ Good | Reliable estimates |
| 100-400 | ⚠️ Marginal | Increase samples |
| < 100 | ❌ Low | Not enough information |

::: tip
ESS tail is often lower than ESS bulk. If you care about credible intervals, check ESS tail specifically.
:::

### Divergent Transitions

In HMC/NUTS, divergent transitions indicate the sampler had numerical problems. This usually means the posterior has difficult geometry.

```rust
// Divergences are tracked in sampling results
if result.divergences > 0 {
    eprintln!("Warning: {} divergent transitions", result.divergences);
}
```

**What divergences mean:**
- **0 divergences**: Good
- **1-10 divergences**: Investigate, might be okay
- **>10 divergences**: Model or sampler needs adjustment

**Fixes for divergences:**
1. Increase `target_accept` (e.g., 0.9 or 0.95)
2. Decrease step size
3. Reparameterize the model (non-centered parameterization)
4. Add stronger priors to constrain the posterior

## Practical Workflow

### 1. Run Multiple Chains

Always run at least 4 chains:

```rust
use bayesian_sampler::multi_chain::{MultiChainConfig, MultiChainSampler};

let config = MultiChainConfig {
    num_chains: 4,
    num_samples: 1000,
    num_warmup: 500,
    ..Default::default()
};

let sampler = MultiChainSampler::new(config);
let result = sampler.sample(&model, init_params);
```

### 2. Check All Diagnostics

```rust
use bayesian_diagnostics::{rhat, ess_bulk, ess_tail};

// For each parameter
let r = rhat(&chains);
let ess_b = ess_bulk(&chains);
let ess_t = ess_tail(&chains);

let converged = r < 1.01 && ess_b > 400.0 && ess_t > 400.0;

if !converged {
    eprintln!("Warning: Diagnostics suggest convergence issues");
    eprintln!("  R-hat: {:.3} (want < 1.01)", r);
    eprintln!("  ESS bulk: {:.0} (want > 400)", ess_b);
    eprintln!("  ESS tail: {:.0} (want > 400)", ess_t);
}
```

### 3. Examine Trace Plots

Visual inspection is valuable. Look for:
- **Good**: Chains look like "fuzzy caterpillars" mixing together
- **Bad**: Chains stuck in different regions, trending, or periodic

### 4. Summary Statistics

```rust
use bayesian_diagnostics::PosteriorSummary;

let summary = PosteriorSummary::from_chains(&chains);

println!("Mean: {:.3}", summary.mean);
println!("Std:  {:.3}", summary.std);
println!("2.5%: {:.3}", summary.q025);
println!("50%:  {:.3}", summary.q50);
println!("97.5%:{:.3}", summary.q975);
```

## Common Issues

### High R-hat

**Symptoms**: R-hat > 1.1

**Causes**:
- Not enough warmup
- Chains started too far apart
- Multimodal posterior
- Poor model specification

**Fixes**:
- Run longer (more warmup and samples)
- Better initialization
- Check model for identification issues

### Low ESS

**Symptoms**: ESS < 100

**Causes**:
- High autocorrelation
- Step size too small or large
- Inefficient mass matrix

**Fixes**:
- Run more samples
- Let adaptation run longer
- Check step size (acceptance rate should be ~0.8)

### Many Divergences

**Symptoms**: >10 divergent transitions

**Causes**:
- Posterior has difficult geometry
- Step size too large
- Funnel-like hierarchical structure

**Fixes**:
- Increase target acceptance rate
- Reparameterize (non-centered for hierarchical)
- Add regularizing priors

## Implementation Details

Our diagnostics follow Vehtari et al. (2021) "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC."

Key implementation choices:
- Split chains in half before computing R-hat
- Use rank-normalization for robustness
- ESS computed via variogram estimation
- Proper multi-chain variance pooling

This ensures compatibility with ArviZ and Stan diagnostics.

## References

- Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC." *Bayesian Analysis*.
- Gelman A, Rubin DB (1992). "Inference from iterative simulation using multiple sequences." *Statistical Science*.
