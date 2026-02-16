/**
 * BayesianGPU Type Definitions
 *
 * This file provides type definitions for the BayesianGPU library.
 * These types are also exported from the main module.
 */

/**
 * Distribution specification
 */
export interface Distribution {
  /** Distribution type (e.g., 'Normal', 'Beta', 'Binomial') */
  type: string;
  /** Distribution parameters */
  params: Record<string, number | number[] | string>;
}

/**
 * Prior specification
 */
export interface Prior {
  name: string;
  distribution: Distribution;
}

/**
 * Likelihood specification
 */
export interface Likelihood {
  distribution: Distribution;
  observed: number[];
}

/**
 * Complete model specification
 */
export interface ModelSpec {
  priors: Prior[];
  likelihood: Likelihood;
}

/**
 * Inference configuration
 */
export interface InferenceConfig {
  /** Number of samples to draw (after warmup) */
  numSamples?: number;
  /** Number of warmup iterations */
  numWarmup?: number;
  /** Number of parallel chains */
  numChains?: number;
  /** Target acceptance probability (0-1) */
  targetAccept?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/**
 * MCMC diagnostics
 */
export interface Diagnostics {
  /** R-hat convergence diagnostic (per parameter) */
  rhat: Record<string, number>;
  /** Effective sample size (per parameter) */
  ess: Record<string, number>;
  /** Number of divergent transitions */
  divergences: number;
}

/**
 * Actual configuration used
 */
export interface ActualConfig {
  numSamples: number;
  numWarmup: number;
  numChains: number;
  stepSize: number;
}

/**
 * Complete inference result
 */
export interface InferenceResult {
  /** Samples for each parameter (flattened across chains) */
  samples: Record<string, number[]>;
  /** MCMC diagnostics */
  diagnostics: Diagnostics;
  /** Configuration used */
  config: ActualConfig;
}

/**
 * Summary statistics for a parameter
 */
export interface ParameterSummary {
  mean: number;
  std: number;
  q025: number;
  q25: number;
  q50: number;
  q75: number;
  q975: number;
  rhat: number;
  ess: number;
}

/**
 * Trace plot data point
 */
export interface TracePlotData {
  param: string;
  chain: number;
  iteration: number;
  value: number;
}

/**
 * Trace plot options
 */
export interface TracePlotOptions {
  paramNames?: string[];
  thin?: number;
  start?: number;
  end?: number;
}

/**
 * Posterior plot data point
 */
export interface PosteriorPlotData {
  param: string;
  value: number;
}

/**
 * Posterior plot options
 */
export interface PosteriorPlotOptions {
  paramNames?: string[];
  bins?: number;
  credibleLevel?: number;
}

/**
 * Histogram bin
 */
export interface HistogramBin {
  param: string;
  x: number;
  x0: number;
  x1: number;
  count: number;
  density: number;
}

/**
 * Interval data for forest plots
 */
export interface IntervalData {
  param: string;
  estimate: number;
  lower: number;
  upper: number;
  innerLower?: number;
  innerUpper?: number;
}

// ============================================================================
// Function declarations
// ============================================================================

/**
 * Fluent model builder
 */
export declare class Model {
  param(name: string, distribution: Distribution): this;
  observe(distribution: Distribution, data: number[]): this;
  build(): ModelSpec;
  toJSON(): string;
}

// Distribution factories
export declare function Normal(loc: number | string, scale: number | string): Distribution;
export declare function HalfNormal(scale: number): Distribution;
export declare function Beta(alpha: number, beta: number): Distribution;
export declare function Gamma(shape: number, rate: number): Distribution;
export declare function Uniform(low: number, high: number): Distribution;
export declare function Bernoulli(p: number | string): Distribution;
export declare function Binomial(n: number, p: number | string): Distribution;
export declare function Poisson(rate: number | string): Distribution;
export declare function Exponential(rate: number): Distribution;
export declare function StudentT(df: number, loc?: number | string, scale?: number | string): Distribution;
export declare function Cauchy(loc?: number, scale?: number): Distribution;
export declare function LogNormal(loc: number | string, scale: number | string): Distribution;

// Inference
export declare function sample(model: ModelSpec, config?: InferenceConfig): Promise<InferenceResult>;
export declare function quickSample(model: ModelSpec): Promise<InferenceResult>;
export declare function productionSample(model: ModelSpec): Promise<InferenceResult>;
export declare function fastSample(model: ModelSpec): Promise<InferenceResult>;
export declare function initialize(): Promise<void>;
export declare function getVersion(): Promise<string>;

// Result utilities
export declare function summarizeParameter(result: InferenceResult, param: string): ParameterSummary;
export declare function summarizeAll(result: InferenceResult): Record<string, ParameterSummary>;
export declare function isConverged(diagnostics: Diagnostics): boolean;
export declare function hasSufficientESS(diagnostics: Diagnostics, minEss?: number): boolean;
export declare function getWarnings(result: InferenceResult): string[];
export declare function formatSummaryTable(result: InferenceResult): string;

// Visualization - trace
export declare function toTracePlotData(result: InferenceResult, options?: TracePlotOptions): TracePlotData[];
export declare function observablePlotTrace(result: InferenceResult, options?: TracePlotOptions & { width?: number; height?: number }): Record<string, unknown>;
export declare function getChainSamples(result: InferenceResult, param: string): number[][];
export declare function computeRunningMean(result: InferenceResult, param: string): { chain: number; iteration: number; runningMean: number }[];
export declare function rankNormalizedTrace(result: InferenceResult, param: string): TracePlotData[];

// Visualization - posterior
export declare function toPosteriorData(result: InferenceResult, paramNames?: string[]): PosteriorPlotData[];
export declare function computeHistogram(result: InferenceResult, param: string, options?: { bins?: number }): HistogramBin[];
export declare function summarize(result: InferenceResult, param: string): { mean: number; std: number; q025: number; q50: number; q975: number; rhat: number; ess: number };
export declare function toIntervalData(result: InferenceResult, options?: PosteriorPlotOptions): IntervalData[];
export declare function observablePlotPosterior(result: InferenceResult, options?: PosteriorPlotOptions & { width?: number; height?: number }): Record<string, unknown>;
export declare function observablePlotForest(result: InferenceResult, options?: PosteriorPlotOptions & { width?: number; height?: number }): Record<string, unknown>;
export declare function hpdInterval(result: InferenceResult, param: string, level?: number): [number, number];
