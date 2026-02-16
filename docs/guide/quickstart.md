# Quick Start

Get up and running with BayesianGPU in under 5 minutes.

## Installation

::: code-group

```bash [npm]
npm install bayesiangpu
```

```bash [yarn]
yarn add bayesiangpu
```

```bash [pnpm]
pnpm add bayesiangpu
```

:::

## Basic Example

Let's estimate the probability of success from observed data using a Beta-Binomial model.

```javascript
import { Model, Beta, Binomial, sample, summarize } from 'bayesiangpu';

// Data: 65 successes out of 100 trials
const model = new Model()
  .param('theta', Beta(1, 1))           // Uniform prior
  .observe(Binomial(100, 'theta'), [65]) // Observed 65 successes
  .build();

// Run MCMC
const result = await sample(model, {
  numSamples: 2000,
  numChains: 4,
});

// Summarize results
const summary = summarize(result, 'theta');
console.log(`Posterior mean: ${summary.mean.toFixed(3)}`);
console.log(`95% CI: [${summary.q025.toFixed(3)}, ${summary.q975.toFixed(3)}]`);
```

Expected output:
```
Posterior mean: 0.648
95% CI: [0.549, 0.738]
```

## Understanding the Output

The `result` object contains:

- **samples**: Raw MCMC samples for each parameter
- **diagnostics**: Convergence diagnostics
  - `rhat`: Should be < 1.01 for convergence
  - `ess`: Effective sample size, higher is better
  - `divergences`: Should be 0

```javascript
// Check convergence
if (result.diagnostics.rhat.theta > 1.01) {
  console.warn('Warning: chains may not have converged');
}

if (result.diagnostics.ess.theta < 400) {
  console.warn('Warning: low effective sample size');
}
```

## Next Steps

- [Model Definition](/guide/models) - Learn about the model DSL
- [Distributions](/guide/distributions) - Available distributions
- [Diagnostics](/guide/diagnostics) - Understanding convergence
