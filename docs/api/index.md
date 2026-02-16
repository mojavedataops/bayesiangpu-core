# API Reference

Complete API documentation for the BayesianGPU JavaScript SDK.

## Quick Links

- [Model](/api/model) - Model definition DSL
- [Distributions](/api/distributions) - Distribution factory functions
- [Inference](/api/inference) - Sampling functions
- [Results](/api/results) - Result types and utilities

## Core Functions

### sample()

Run MCMC inference on a model.

```typescript
function sample(
  model: ModelSpec,
  config?: InferenceConfig
): Promise<InferenceResult>
```

**Parameters:**
- `model` - Model specification from `Model.build()`
- `config` - Optional configuration

**Config options:**
```typescript
interface InferenceConfig {
  numSamples?: number;   // Default: 1000
  numWarmup?: number;    // Default: 1000
  numChains?: number;    // Default: 4
  targetAccept?: number; // Default: 0.8
  seed?: number;         // Default: 42
}
```

### initialize()

Initialize the WASM backend.

```typescript
function initialize(options?: InitOptions): Promise<BackendType>
```

Called automatically by `sample()`, but can be called explicitly for WebGPU.

```javascript
import { initialize, sample } from 'bayesiangpu';

const backend = await initialize({ preferWebGPU: true });
console.log(`Using ${backend} backend`);
```

### summarize()

Generate summary statistics for a parameter.

```typescript
function summarize(
  result: InferenceResult,
  param: string
): ParameterSummary
```

Returns mean, std, quantiles, and convergence info.

## Types

### ModelSpec

```typescript
interface ModelSpec {
  priors: Prior[];
  likelihood: Likelihood;
}
```

### InferenceResult

```typescript
interface InferenceResult {
  samples: Record<string, number[]>;
  diagnostics: Diagnostics;
  config: ActualConfig;
}
```

### Diagnostics

```typescript
interface Diagnostics {
  rhat: Record<string, number>;
  ess: Record<string, number>;
  divergences: number;
}
```

## Presets

### quickSample()

Fast sampling for exploration.

```javascript
const result = await quickSample(model);
// 200 samples, 100 warmup, 2 chains
```

### productionSample()

Conservative settings for publication.

```javascript
const result = await productionSample(model);
// 4000 samples, 2000 warmup, 4 chains
```
