/**
 * Inference result types for BayesianGPU
 *
 * This module defines the types returned by the inference engine.
 *
 * @module
 */

/**
 * Inference configuration used for sampling
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
 * MCMC diagnostics for a single parameter or overall
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
 * Configuration that was actually used (with defaults filled in)
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
 * Summary statistics for a single parameter
 */
export interface ParameterSummary {
  /** Posterior mean */
  mean: number;
  /** Posterior standard deviation */
  std: number;
  /** 2.5th percentile (lower bound of 95% CI) */
  q025: number;
  /** 25th percentile */
  q25: number;
  /** Median (50th percentile) */
  q50: number;
  /** 75th percentile */
  q75: number;
  /** 97.5th percentile (upper bound of 95% CI) */
  q975: number;
  /** R-hat convergence diagnostic */
  rhat: number;
  /** Effective sample size */
  ess: number;
}

/**
 * Check if diagnostics indicate good convergence
 *
 * @param diagnostics - The diagnostics object
 * @returns true if all R-hat values are below 1.01
 */
export function isConverged(diagnostics: Diagnostics): boolean {
  return Object.values(diagnostics.rhat).every(
    (r) => r !== undefined && r < 1.01
  );
}

/**
 * Check if effective sample size is sufficient
 *
 * @param diagnostics - The diagnostics object
 * @param minEss - Minimum ESS required (default: 400)
 * @returns true if all ESS values are above the minimum
 */
export function hasSufficientESS(
  diagnostics: Diagnostics,
  minEss: number = 400
): boolean {
  return Object.values(diagnostics.ess).every(
    (e) => e !== undefined && e >= minEss
  );
}

/**
 * Get warning messages for any diagnostic issues
 *
 * @param result - The inference result
 * @returns Array of warning messages
 */
export function getWarnings(result: InferenceResult): string[] {
  const warnings: string[] = [];

  // Check R-hat values
  for (const [param, rhat] of Object.entries(result.diagnostics.rhat)) {
    if (rhat >= 1.1) {
      warnings.push(
        `${param}: R-hat = ${rhat.toFixed(3)} indicates poor convergence (should be < 1.01)`
      );
    } else if (rhat >= 1.01) {
      warnings.push(
        `${param}: R-hat = ${rhat.toFixed(3)} is marginal (ideally < 1.01)`
      );
    }
  }

  // Check ESS values
  for (const [param, ess] of Object.entries(result.diagnostics.ess)) {
    if (ess < 100) {
      warnings.push(
        `${param}: ESS = ${ess.toFixed(0)} is too low (should be > 100)`
      );
    } else if (ess < 400) {
      warnings.push(
        `${param}: ESS = ${ess.toFixed(0)} is low (ideally > 400)`
      );
    }
  }

  // Check divergences
  const totalSamples =
    result.config.numSamples * result.config.numChains;
  const divergenceRate = result.diagnostics.divergences / totalSamples;
  if (divergenceRate > 0.05) {
    warnings.push(
      `${result.diagnostics.divergences} divergences (${(divergenceRate * 100).toFixed(1)}%) - consider reparameterizing`
    );
  } else if (result.diagnostics.divergences > 0) {
    warnings.push(
      `${result.diagnostics.divergences} divergences detected`
    );
  }

  return warnings;
}

/**
 * Compute summary statistics for a parameter
 *
 * @param result - The inference result
 * @param paramName - Name of the parameter
 * @returns Summary statistics
 */
export function summarizeParameter(
  result: InferenceResult,
  paramName: string
): ParameterSummary {
  const samples = result.samples[paramName];
  if (!samples || samples.length === 0) {
    throw new Error(`Parameter '${paramName}' not found in results`);
  }

  const n = samples.length;

  // Compute mean
  const mean = samples.reduce((a, b) => a + b, 0) / n;

  // Compute standard deviation
  const variance =
    samples.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);

  // Sort for quantiles
  const sorted = [...samples].sort((a, b) => a - b);

  // Compute quantiles
  const quantile = (p: number): number => {
    const idx = p * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.min(lower + 1, n - 1);
    const weight = idx - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  };

  return {
    mean,
    std,
    q025: quantile(0.025),
    q25: quantile(0.25),
    q50: quantile(0.5),
    q75: quantile(0.75),
    q975: quantile(0.975),
    rhat: result.diagnostics.rhat[paramName] ?? NaN,
    ess: result.diagnostics.ess[paramName] ?? NaN,
  };
}

/**
 * Compute summary statistics for all parameters
 *
 * @param result - The inference result
 * @returns Map of parameter names to their summaries
 */
export function summarizeAll(
  result: InferenceResult
): Record<string, ParameterSummary> {
  const summaries: Record<string, ParameterSummary> = {};
  for (const paramName of Object.keys(result.samples)) {
    summaries[paramName] = summarizeParameter(result, paramName);
  }
  return summaries;
}

/**
 * Format inference results as a table string
 *
 * @param result - The inference result
 * @returns Formatted table string
 */
export function formatSummaryTable(result: InferenceResult): string {
  const summaries = summarizeAll(result);
  const params = Object.keys(summaries);

  if (params.length === 0) {
    return 'No parameters in result';
  }

  // Header
  const lines: string[] = [
    `${'Parameter'.padEnd(15)} ${'Mean'.padStart(10)} ${'Std'.padStart(10)} ${'2.5%'.padStart(10)} ${'Median'.padStart(10)} ${'97.5%'.padStart(10)} ${'R-hat'.padStart(8)} ${'ESS'.padStart(8)}`,
    '-'.repeat(90),
  ];

  // Rows
  for (const param of params) {
    const s = summaries[param];
    lines.push(
      `${param.padEnd(15)} ${s.mean.toFixed(3).padStart(10)} ${s.std.toFixed(3).padStart(10)} ${s.q025.toFixed(3).padStart(10)} ${s.q50.toFixed(3).padStart(10)} ${s.q975.toFixed(3).padStart(10)} ${s.rhat.toFixed(3).padStart(8)} ${s.ess.toFixed(0).padStart(8)}`
    );
  }

  return lines.join('\n');
}
