# BayesianGPU Python Package

GPU-accelerated Bayesian inference in Python, powered by Rust.

## Installation

```bash
pip install bayesiangpu
```

## Quick Start

```python
from bayesiangpu import Model, Beta, Binomial, sample

# Define a Beta-Binomial model
model = Model()
model.param('theta', Beta(1, 1))      # Prior: uniform on [0,1]
model.observe(Binomial(100, 'theta'), [65])  # 65 successes in 100 trials

# Run inference
result = sample(model, num_samples=1000, num_chains=4)

# Analyze results
print(result)  # Summary table
print(f"Converged: {result.is_converged()}")
```

## Features

- **Fast**: Native Rust implementation with NUTS sampler
- **Easy**: Simple, Pythonic API inspired by PyMC and NumPyro
- **Reliable**: Automatic convergence diagnostics (R-hat, ESS)

## API Reference

### Distributions

#### Priors
- `Normal(loc, scale)` - Normal distribution
- `HalfNormal(scale)` - Half-normal (positive values)
- `Beta(alpha, beta)` - Beta distribution (0-1)
- `Gamma(shape, rate)` - Gamma distribution (positive)
- `Uniform(low, high)` - Uniform distribution
- `Exponential(rate)` - Exponential distribution
- `StudentT(df, loc, scale)` - Student's t distribution
- `Cauchy(loc, scale)` - Cauchy distribution
- `LogNormal(loc, scale)` - Log-normal distribution

#### Likelihoods
- `Bernoulli(p)` - Binary outcomes
- `Binomial(n, p)` - Count of successes
- `Poisson(rate)` - Count data
- `Normal(loc, scale)` - Continuous outcomes

### Model Building

```python
model = Model()
model.param('name', distribution)  # Add parameter with prior
model.observe(likelihood, data)    # Set likelihood with observed data
```

### Sampling

```python
result = sample(
    model,
    num_samples=1000,   # Samples after warmup
    num_warmup=1000,    # Warmup iterations
    num_chains=4,       # Number of chains
    target_accept=0.8,  # Target acceptance rate
    seed=42             # Random seed
)
```

### Results

```python
# Summary statistics
result.summary()  # All parameters
result.summarize('theta')  # Single parameter

# Raw samples
result.get_samples('theta')  # Flattened
result.get_chain_samples('theta')  # By chain

# Diagnostics
result.is_converged()  # R-hat < 1.01
result.has_sufficient_ess(400)  # ESS threshold
result.warnings()  # Diagnostic warnings
```

## Example: Estimating a Proportion

```python
from bayesiangpu import Model, Beta, Binomial, sample

# Survey: 65 out of 100 people prefer product A
model = Model()
model.param('p', Beta(1, 1))  # Uniform prior on preference
model.observe(Binomial(100, 'p'), [65])

result = sample(model, num_samples=2000, num_chains=4)

summary = result.summarize('p')
print(f"Estimated preference: {summary.mean:.1%}")
print(f"95% CI: [{summary.q025:.1%}, {summary.q975:.1%}]")
```

## Example: Normal Mean Estimation

```python
from bayesiangpu import Model, Normal, HalfNormal, sample

# Estimate mean and standard deviation of some data
data = [2.3, 2.1, 2.5, 2.4, 2.2, 2.6, 2.3, 2.4]

model = Model()
model.param('mu', Normal(0, 10))      # Weakly informative prior on mean
model.param('sigma', HalfNormal(5))   # Prior on standard deviation
model.observe(Normal('mu', 'sigma'), data)

result = sample(model, num_samples=2000, num_chains=4)

print(result)
```

## Development

### Building from source

```bash
cd crates/bayesian-py
pip install maturin
maturin develop
```

### Running tests

```bash
pytest tests/
```

## License

MIT
