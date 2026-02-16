# BayesianGPU JavaScript SDK

GPU-accelerated Bayesian inference in the browser using WebGPU.

## Installation

```bash
npm install bayesiangpu
```

Or via CDN:

```html
<script type="module">
  import { Model, Normal, sample } from 'https://unpkg.com/bayesiangpu';
</script>
```

## Quick Start

```javascript
import {
  Model, Beta, Binomial, sample,
  summarizeParameter, isConverged
} from 'bayesiangpu';

// Define a Beta-Binomial model
const model = new Model()
  .param('theta', Beta(1, 1))           // Prior: uniform on [0,1]
  .observe(Binomial(100, 'theta'), [65]) // Likelihood: 65 successes in 100 trials
  .build();

// Run inference
const result = await sample(model, {
  numSamples: 1000,
  numChains: 4,
});

// Analyze results
const summary = summarizeParameter(result, 'theta');
console.log(`Mean: ${summary.mean.toFixed(3)}`);
console.log(`95% CI: [${summary.q025.toFixed(3)}, ${summary.q975.toFixed(3)}]`);

if (!isConverged(result.diagnostics)) {
  console.warn('Chains may not have converged!');
}
```

## API Reference

### Model Definition

#### `new Model()`

Creates a new model builder.

```javascript
const model = new Model();
```

#### `.param(name, prior)`

Adds a parameter with a prior distribution.

```javascript
model.param('mu', Normal(0, 10));
model.param('sigma', HalfNormal(1));
```

#### `.observe(distribution, data)`

Sets the likelihood with observed data. Distribution parameters can reference model parameters by name.

```javascript
model.observe(Normal('mu', 'sigma'), [1.2, 2.3, 1.8, 2.1, 1.9]);
```

#### `.build()`

Builds the model specification for sampling.

```javascript
const modelSpec = model.build();
```

#### `.toJSON()`

Serializes the model to JSON string.

```javascript
const json = model.toJSON();
```

### Distributions

All distributions return distribution specification objects for use with `.param()` and `.observe()`.

#### Available Distributions (12 total)

```javascript
import {
  Normal, HalfNormal, Beta, Gamma, Uniform,
  Bernoulli, Binomial, Poisson, Exponential,
  StudentT, Cauchy, LogNormal
} from 'bayesiangpu';

// Continuous priors
Normal(loc, scale)       // Normal(0, 1) - Gaussian
HalfNormal(scale)        // HalfNormal(1) - positive values only
Uniform(low, high)       // Uniform(0, 10) - flat prior
Beta(alpha, beta)        // Beta(1, 1) - probability parameters
Gamma(shape, rate)       // Gamma(2, 0.5) - positive values
Exponential(rate)        // Exponential(1) - waiting times
StudentT(df, loc, scale) // StudentT(3, 0, 1) - robust inference
Cauchy(loc, scale)       // Cauchy(0, 2.5) - weakly informative
LogNormal(loc, scale)    // LogNormal(0, 1) - multiplicative effects

// Discrete likelihoods
Bernoulli(p)             // Binary outcomes (0/1)
Binomial(n, p)           // Count of successes
Poisson(rate)            // Count data
```

#### Parameter References

Distributions can reference model parameters by name:

```javascript
const model = new Model()
  .param('mu', Normal(0, 10))
  .param('sigma', HalfNormal(1))
  .observe(Normal('mu', 'sigma'), data);  // References 'mu' and 'sigma'
```

### Sampling

#### `sample(model, config)`

Runs MCMC sampling on the model.

```javascript
const result = await sample(model, {
  numSamples: 1000,    // Number of posterior samples (default: 1000)
  numWarmup: 1000,     // Warmup/adaptation iterations (default: 1000)
  numChains: 4,        // Number of parallel chains (default: 4)
  targetAccept: 0.8,   // Target acceptance rate (default: 0.8)
  seed: 42,            // Random seed for reproducibility
});
```

**Parameters:**
- `model` - Model specification (from `model.build()`)
- `config` - Sampling configuration (optional)

**Returns:** `InferenceResult`

#### Convenience Functions

```javascript
// Quick exploration (1000 samples, 4 chains)
const result = await quickSample(model);

// Production quality (4000 samples, 4 chains)
const result = await productionSample(model);

// Fast development iteration (200 samples, 2 chains)
const result = await fastSample(model);
```

### InferenceResult

The result object returned by `sample()`.

#### Properties

```javascript
// Samples for each parameter (flattened across chains)
result.samples          // { mu: [1.0, 1.1, ...], sigma: [0.5, 0.6, ...] }

// MCMC diagnostics
result.diagnostics.rhat       // { mu: 1.001, sigma: 1.002 }
result.diagnostics.ess        // { mu: 450, sigma: 420 }
result.diagnostics.divergences // 0

// Configuration used
result.config.numSamples  // 1000
result.config.numChains   // 4
result.config.numWarmup   // 1000
result.config.stepSize    // 0.1
```

### Result Utilities

Standalone functions for analyzing results:

```javascript
import {
  summarizeParameter,
  summarizeAll,
  isConverged,
  hasSufficientESS,
  getWarnings,
  formatSummaryTable,
} from 'bayesiangpu';

// Summary for single parameter
const summary = summarizeParameter(result, 'mu');
// { mean, std, q025, q25, q50, q75, q975, rhat, ess }

// Summary for all parameters
const allSummaries = summarizeAll(result);
// { mu: {...}, sigma: {...} }

// Convergence checks
isConverged(result.diagnostics)           // true if all R-hat < 1.01
hasSufficientESS(result.diagnostics)      // true if all ESS >= 400
hasSufficientESS(result.diagnostics, 200) // custom minimum ESS

// Get warning messages
const warnings = getWarnings(result);
// ['sigma: ESS = 350 is low (ideally > 400)']

// Formatted table
console.log(formatSummaryTable(result));
// Parameter        Mean        Std       2.5%     Median      97.5%    R-hat      ESS
// mu              2.100      0.300      1.600      2.100      2.600    1.001      450
```

### Visualization

Helper functions for creating plot data (use with your preferred charting library):

```javascript
import {
  toTracePlotData,
  toPosteriorData,
  computeHistogram,
  hpdInterval,
} from 'bayesiangpu';

// Trace plot data
const traceData = toTracePlotData(result, 'mu');

// Posterior histogram
const posteriorData = toPosteriorData(result, 'mu', { bins: 50 });

// Histogram bins
const bins = computeHistogram(result.samples.mu, 30);

// Highest posterior density interval
const hpd = hpdInterval(result.samples.mu, 0.95);
// { lower: 1.6, upper: 2.6 }
```

### Backend Initialization

```javascript
import { initialize, isWebGPUAvailable, getBackendType } from 'bayesiangpu';

// Check WebGPU availability
if (isWebGPUAvailable()) {
  console.log('WebGPU acceleration available!');
}

// Initialize with options
const backend = await initialize({ preferWebGPU: true });
console.log(`Using backend: ${backend}`); // 'webgpu' or 'cpu'

// Force CPU backend
await initialize({ forceCPU: true });

// Check current backend
console.log(getBackendType()); // 'cpu', 'webgpu', or 'not_initialized'
```

## Examples

### Beta-Binomial Model

```javascript
import { Model, Beta, Binomial, sample, summarizeParameter } from 'bayesiangpu';

// Coin flip: 65 heads in 100 flips
const model = new Model()
  .param('theta', Beta(1, 1))            // Uniform prior
  .observe(Binomial(100, 'theta'), [65])
  .build();

const result = await sample(model);
const summary = summarizeParameter(result, 'theta');
console.log(`P(heads) = ${summary.mean.toFixed(3)} [${summary.q025.toFixed(3)}, ${summary.q975.toFixed(3)}]`);
```

### Normal-Normal Model

```javascript
import {
  Model, Normal, HalfNormal, sample,
  summarizeAll, formatSummaryTable
} from 'bayesiangpu';

const data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.0, 1.7, 2.2];

const model = new Model()
  .param('mu', Normal(0, 10))       // Weakly informative prior on mean
  .param('sigma', HalfNormal(1))    // Positive scale parameter
  .observe(Normal('mu', 'sigma'), data)
  .build();

const result = await sample(model, { numSamples: 2000 });

console.log(formatSummaryTable(result));
```

### Bernoulli Likelihood

```javascript
import { Model, Beta, Bernoulli, sample, summarizeParameter } from 'bayesiangpu';

// Binary outcomes
const outcomes = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1];

const model = new Model()
  .param('p', Beta(1, 1))
  .observe(Bernoulli('p'), outcomes)
  .build();

const result = await sample(model);
const summary = summarizeParameter(result, 'p');
console.log(`Success rate: ${summary.mean.toFixed(3)}`);
```

### Using Uniform Prior

```javascript
import { Model, Uniform, Normal, sample, summarizeParameter } from 'bayesiangpu';

const model = new Model()
  .param('x', Uniform(-5, 5))       // Flat prior on [-5, 5]
  .observe(Normal('x', 1), [0.5, 1.2, -0.3, 0.8])
  .build();

const result = await sample(model);
console.log(summarizeParameter(result, 'x'));
```

## Diagnostics

### Checking Convergence

```javascript
import { isConverged, hasSufficientESS, getWarnings } from 'bayesiangpu';

// Quick checks
if (!isConverged(result.diagnostics)) {
  console.warn('R-hat indicates poor convergence');
}

if (!hasSufficientESS(result.diagnostics)) {
  console.warn('ESS is too low, run more samples');
}

// Detailed warnings
const warnings = getWarnings(result);
warnings.forEach(w => console.warn(w));

// Manual inspection
const diag = result.diagnostics;
for (const [param, rhat] of Object.entries(diag.rhat)) {
  if (rhat > 1.01) {
    console.warn(`${param}: R-hat = ${rhat.toFixed(3)} (should be < 1.01)`);
  }
}

for (const [param, ess] of Object.entries(diag.ess)) {
  if (ess < 400) {
    console.warn(`${param}: ESS = ${ess.toFixed(0)} (ideally > 400)`);
  }
}

if (diag.divergences > 0) {
  console.warn(`${diag.divergences} divergent transitions detected`);
}
```

### Interpreting Diagnostics

| Diagnostic | Good | Warning | Action |
|------------|------|---------|--------|
| R-hat | < 1.01 | 1.01 - 1.1 | > 1.1: Run longer chains |
| ESS | > 400 | 100-400 | < 100: Run more samples |
| Divergences | 0 | 1-10 | > 10: Reparameterize model |

## Browser Requirements

BayesianGPU works in all modern browsers. WebGPU provides GPU acceleration where available:

```javascript
import { isWebGPUAvailable, initialize } from 'bayesiangpu';

if (isWebGPUAvailable()) {
  console.log('GPU acceleration available!');
} else {
  console.log('Using CPU backend');
}

// Automatic backend selection
const backend = await initialize();
```

**WebGPU support:**
- Chrome 113+
- Edge 113+
- Firefox (behind flag)
- Safari 17+ (partial)

## TypeScript

Full TypeScript support included:

```typescript
import {
  Model, Normal, HalfNormal, sample,
  summarizeParameter,
  type InferenceResult,
  type InferenceConfig,
  type ParameterSummary,
} from 'bayesiangpu';

const config: InferenceConfig = {
  numSamples: 1000,
  numChains: 4,
};

const model = new Model()
  .param('mu', Normal(0, 10))
  .param('sigma', HalfNormal(1))
  .observe(Normal('mu', 'sigma'), [1.0, 2.0, 1.5])
  .build();

const result: InferenceResult = await sample(model, config);
const summary: ParameterSummary = summarizeParameter(result, 'mu');
```

## Performance Tips

1. **Use multiple chains**: More chains = better diagnostics and parallelism
2. **Tune warmup**: Start with equal warmup and samples, adjust based on diagnostics
3. **Check ESS**: Low ESS suggests autocorrelation; run more samples
4. **Watch divergences**: Many divergences indicate model problems

## License

MIT
