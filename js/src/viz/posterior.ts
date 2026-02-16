/**
 * Posterior visualization utilities for BayesianGPU
 *
 * This module provides functions to prepare inference results for visualization
 * as posterior density plots, interval plots, and summary statistics.
 *
 * @module
 */

import type { InferenceResult, ParameterSummary } from '../result';
import { summarizeParameter } from '../result';

/**
 * Options for posterior plot generation
 */
export interface PosteriorPlotOptions {
  /** Specific parameter names to include (default: all) */
  paramNames?: string[];
  /** Number of bins for histogram (default: 50) */
  bins?: number;
  /** Credible interval level (default: 0.95) */
  credibleLevel?: number;
}

/**
 * Data point for posterior density plot
 */
export interface PosteriorPlotData {
  /** Parameter name */
  param: string;
  /** Sample value */
  value: number;
}

/**
 * Histogram bin data
 */
export interface HistogramBin {
  /** Parameter name */
  param: string;
  /** Bin center */
  x: number;
  /** Bin left edge */
  x0: number;
  /** Bin right edge */
  x1: number;
  /** Count (or density) */
  count: number;
  /** Normalized density */
  density: number;
}

/**
 * Interval plot data point
 */
export interface IntervalData {
  /** Parameter name */
  param: string;
  /** Point estimate (mean or median) */
  estimate: number;
  /** Lower bound of interval */
  lower: number;
  /** Upper bound of interval */
  upper: number;
  /** Lower bound of inner interval (if applicable) */
  innerLower?: number;
  /** Upper bound of inner interval (if applicable) */
  innerUpper?: number;
}

/**
 * Convert inference result to format suitable for posterior density plots
 *
 * Returns a flat array of sample values by parameter, suitable for
 * kernel density estimation or histogram plotting.
 *
 * @param result - Inference result from sample()
 * @param paramNames - Optional list of parameter names (default: all)
 * @returns Array of data points
 *
 * @example
 * ```typescript
 * import { toPosteriorData } from 'bayesiangpu';
 * import * as Plot from '@observablehq/plot';
 *
 * const data = toPosteriorData(result);
 * Plot.plot({
 *   marks: [
 *     Plot.rectY(data, Plot.binX({ y: 'count' }, { x: 'value', fy: 'param' }))
 *   ]
 * });
 * ```
 */
export function toPosteriorData(
  result: InferenceResult,
  paramNames?: string[]
): PosteriorPlotData[] {
  const params = paramNames ?? Object.keys(result.samples);
  const data: PosteriorPlotData[] = [];

  for (const param of params) {
    const samples = result.samples[param];
    if (!samples) continue;

    for (const value of samples) {
      data.push({ param, value });
    }
  }

  return data;
}

/**
 * Compute histogram bins for posterior samples
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @param options - Options including number of bins
 * @returns Array of histogram bins
 */
export function computeHistogram(
  result: InferenceResult,
  param: string,
  options: { bins?: number } = {}
): HistogramBin[] {
  const { bins = 50 } = options;
  const samples = result.samples[param];

  if (!samples || samples.length === 0) {
    throw new Error(`Parameter '${param}' not found or empty`);
  }

  // Find range
  let min = samples[0];
  let max = samples[0];
  for (const s of samples) {
    if (s < min) min = s;
    if (s > max) max = s;
  }

  // Add small padding
  const range = max - min;
  const padding = range * 0.01;
  min -= padding;
  max += padding;

  // Create bins
  const binWidth = (max - min) / bins;
  const counts = new Array<number>(bins).fill(0);

  for (const s of samples) {
    const binIdx = Math.min(Math.floor((s - min) / binWidth), bins - 1);
    counts[binIdx]++;
  }

  // Convert to histogram data
  const n = samples.length;
  const histogram: HistogramBin[] = [];

  for (let i = 0; i < bins; i++) {
    const x0 = min + i * binWidth;
    const x1 = x0 + binWidth;
    histogram.push({
      param,
      x: (x0 + x1) / 2,
      x0,
      x1,
      count: counts[i],
      density: counts[i] / (n * binWidth),
    });
  }

  return histogram;
}

/**
 * Compute summary statistics suitable for display
 *
 * Wrapper around summarizeParameter that returns a simpler object.
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @returns Summary statistics
 */
export function summarize(
  result: InferenceResult,
  param: string
): {
  mean: number;
  std: number;
  q025: number;
  q50: number;
  q975: number;
  rhat: number;
  ess: number;
} {
  const s = summarizeParameter(result, param);
  return {
    mean: s.mean,
    std: s.std,
    q025: s.q025,
    q50: s.q50,
    q975: s.q975,
    rhat: s.rhat,
    ess: s.ess,
  };
}

/**
 * Generate interval plot data for all parameters
 *
 * Creates data suitable for forest plots or caterpillar plots.
 *
 * @param result - Inference result
 * @param options - Plot options
 * @returns Array of interval data
 *
 * @example
 * ```typescript
 * import { toIntervalData } from 'bayesiangpu';
 * import * as Plot from '@observablehq/plot';
 *
 * const intervals = toIntervalData(result);
 * Plot.plot({
 *   marks: [
 *     Plot.ruleX(intervals, { x1: 'lower', x2: 'upper', y: 'param' }),
 *     Plot.dot(intervals, { x: 'estimate', y: 'param' })
 *   ]
 * });
 * ```
 */
export function toIntervalData(
  result: InferenceResult,
  options: PosteriorPlotOptions = {}
): IntervalData[] {
  const { paramNames = Object.keys(result.samples), credibleLevel = 0.95 } =
    options;

  const intervals: IntervalData[] = [];
  const alpha = 1 - credibleLevel;

  for (const param of paramNames) {
    const samples = result.samples[param];
    if (!samples || samples.length === 0) continue;

    const sorted = [...samples].sort((a, b) => a - b);
    const n = sorted.length;

    // Quantile function
    const quantile = (p: number): number => {
      const idx = p * (n - 1);
      const lower = Math.floor(idx);
      const upper = Math.min(lower + 1, n - 1);
      const weight = idx - lower;
      return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    };

    // Point estimate (median)
    const estimate = quantile(0.5);

    // Credible interval
    const lower = quantile(alpha / 2);
    const upper = quantile(1 - alpha / 2);

    // 50% credible interval (for inner interval)
    const innerLower = quantile(0.25);
    const innerUpper = quantile(0.75);

    intervals.push({
      param,
      estimate,
      lower,
      upper,
      innerLower,
      innerUpper,
    });
  }

  return intervals;
}

/**
 * Observable Plot specification for posterior density plot
 *
 * @param result - Inference result
 * @param options - Plot options
 * @returns Observable Plot specification
 */
export function observablePlotPosterior(
  result: InferenceResult,
  options: PosteriorPlotOptions & {
    width?: number;
    height?: number;
    fillColor?: string;
  } = {}
): Record<string, unknown> {
  const data = toPosteriorData(result, options.paramNames);
  const { width = 600, height, fillColor = 'steelblue' } = options;
  const params = [...new Set(data.map((d) => d.param))];

  const plotHeight = height ?? Math.max(150, 120 * params.length);

  return {
    width,
    height: plotHeight,
    facet: {
      data,
      y: 'param',
    },
    fy: {
      label: null,
    },
    x: {
      label: 'Value',
    },
    y: {
      label: null,
    },
    marks: [
      {
        type: 'rectY',
        data,
        x: 'value',
        fill: fillColor,
        fillOpacity: 0.7,
        binX: { y: 'count' },
      },
    ],
  };
}

/**
 * Observable Plot specification for forest plot (interval plot)
 *
 * @param result - Inference result
 * @param options - Plot options
 * @returns Observable Plot specification
 */
export function observablePlotForest(
  result: InferenceResult,
  options: PosteriorPlotOptions & {
    width?: number;
    height?: number;
    pointColor?: string;
    lineColor?: string;
  } = {}
): Record<string, unknown> {
  const intervals = toIntervalData(result, options);
  const {
    width = 500,
    height = Math.max(100, 30 * intervals.length),
    pointColor = 'black',
    lineColor = 'steelblue',
  } = options;

  return {
    width,
    height,
    marginLeft: 100,
    x: {
      label: 'Value',
    },
    y: {
      label: null,
      domain: intervals.map((d) => d.param),
    },
    marks: [
      // Outer interval (95% CI)
      {
        type: 'ruleX',
        data: intervals,
        x1: 'lower',
        x2: 'upper',
        y: 'param',
        stroke: lineColor,
        strokeWidth: 1,
      },
      // Inner interval (50% CI)
      {
        type: 'ruleX',
        data: intervals,
        x1: 'innerLower',
        x2: 'innerUpper',
        y: 'param',
        stroke: lineColor,
        strokeWidth: 3,
      },
      // Point estimate
      {
        type: 'dot',
        data: intervals,
        x: 'estimate',
        y: 'param',
        fill: pointColor,
        r: 4,
      },
      // Reference line at 0
      {
        type: 'ruleX',
        x: [0],
        stroke: 'gray',
        strokeDasharray: '4,4',
      },
    ],
  };
}

/**
 * Compute highest posterior density (HPD) interval
 *
 * The HPD interval is the shortest interval containing the specified
 * probability mass.
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @param level - Credible level (default: 0.95)
 * @returns [lower, upper] bounds
 */
export function hpdInterval(
  result: InferenceResult,
  param: string,
  level: number = 0.95
): [number, number] {
  const samples = result.samples[param];
  if (!samples || samples.length === 0) {
    throw new Error(`Parameter '${param}' not found or empty`);
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const intervalSize = Math.ceil(n * level);

  if (intervalSize >= n) {
    return [sorted[0], sorted[n - 1]];
  }

  let bestWidth = Infinity;
  let bestLower = 0;

  for (let i = 0; i <= n - intervalSize; i++) {
    const width = sorted[i + intervalSize - 1] - sorted[i];
    if (width < bestWidth) {
      bestWidth = width;
      bestLower = i;
    }
  }

  return [sorted[bestLower], sorted[bestLower + intervalSize - 1]];
}
