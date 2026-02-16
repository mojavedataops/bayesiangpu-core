/**
 * BayesianGPU - GPU-accelerated Bayesian inference in the browser
 *
 * This library provides a complete solution for Bayesian inference in the browser,
 * using WebAssembly for computation and an optional WebGPU backend for acceleration.
 *
 * @example Basic usage
 * ```typescript
 * import { Model, Beta, Binomial, sample, summarize } from 'bayesiangpu';
 *
 * // Define a simple Beta-Binomial model
 * const model = new Model()
 *   .param('theta', Beta(1, 1))      // Prior: uniform on [0,1]
 *   .observe(Binomial(100, 'theta'), [65])  // Likelihood: 65 successes in 100 trials
 *   .build();
 *
 * // Run inference
 * const result = await sample(model, {
 *   numSamples: 1000,
 *   numChains: 4,
 * });
 *
 * // Analyze results
 * const summary = summarize(result, 'theta');
 * console.log(`Mean: ${summary.mean.toFixed(3)}`);
 * console.log(`95% CI: [${summary.q025.toFixed(3)}, ${summary.q975.toFixed(3)}]`);
 * ```
 *
 * @packageDocumentation
 */

// Model definition
export { Model } from './model';
export type { Distribution, Prior, Likelihood, ModelSpec } from './model';

// Distribution factories
export {
  Normal,
  HalfNormal,
  Beta,
  Gamma,
  Uniform,
  Bernoulli,
  Binomial,
  Poisson,
  Exponential,
  StudentT,
  Cauchy,
  LogNormal,
} from './model';

// Inference
export {
  sample,
  quickSample,
  productionSample,
  fastSample,
  initialize,
  getVersion,
} from './inference';

export type { BackendType, InitOptions } from './inference';

// Result types and utilities
export type {
  InferenceConfig,
  InferenceResult,
  Diagnostics,
  ActualConfig,
  ParameterSummary,
} from './result';

export {
  summarizeParameter,
  summarizeAll,
  isConverged,
  hasSufficientESS,
  getWarnings,
  formatSummaryTable,
} from './result';

// Visualization - trace plots
export type { TracePlotOptions, TracePlotData } from './viz/trace';

export {
  toTracePlotData,
  observablePlotTrace,
  getChainSamples,
  computeRunningMean,
  rankNormalizedTrace,
} from './viz/trace';

// Visualization - posterior plots
export type {
  PosteriorPlotOptions,
  PosteriorPlotData,
  HistogramBin,
  IntervalData,
} from './viz/posterior';

export {
  toPosteriorData,
  computeHistogram,
  summarize,
  toIntervalData,
  observablePlotPosterior,
  observablePlotForest,
  hpdInterval,
} from './viz/posterior';
