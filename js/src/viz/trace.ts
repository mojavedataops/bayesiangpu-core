/**
 * Trace plot utilities for BayesianGPU
 *
 * This module provides functions to prepare inference results for visualization
 * as trace plots (iteration vs. parameter value). Trace plots are essential for
 * diagnosing MCMC convergence.
 *
 * @module
 */

import type { InferenceResult } from '../result';

/**
 * Options for trace plot generation
 */
export interface TracePlotOptions {
  /** Specific parameter names to include (default: all) */
  paramNames?: string[];
  /** Thin samples by this factor (default: 1, no thinning) */
  thin?: number;
  /** Start from this iteration (default: 0) */
  start?: number;
  /** End at this iteration (default: all) */
  end?: number;
}

/**
 * Single data point for trace plot
 */
export interface TracePlotData {
  /** Parameter name */
  param: string;
  /** Chain index (0-based) */
  chain: number;
  /** Iteration number (0-based, after warmup) */
  iteration: number;
  /** Parameter value */
  value: number;
}

/**
 * Convert inference result to format suitable for trace plots
 *
 * Returns an array of data points that can be used with D3, Observable Plot,
 * Vega-Lite, or any other visualization library.
 *
 * @param result - Inference result from sample()
 * @param options - Plot options
 * @returns Array of data points for plotting
 *
 * @example
 * ```typescript
 * import { toTracePlotData } from 'bayesiangpu';
 * import * as Plot from '@observablehq/plot';
 *
 * const data = toTracePlotData(result);
 * Plot.plot({
 *   marks: [
 *     Plot.line(data, {
 *       x: 'iteration',
 *       y: 'value',
 *       stroke: 'chain',
 *       fy: 'param'
 *     })
 *   ]
 * });
 * ```
 */
export function toTracePlotData(
  result: InferenceResult,
  options: TracePlotOptions = {}
): TracePlotData[] {
  const {
    paramNames = Object.keys(result.samples),
    thin = 1,
    start = 0,
    end,
  } = options;

  const numChains = result.config.numChains;
  const data: TracePlotData[] = [];

  for (const param of paramNames) {
    const samples = result.samples[param];
    if (!samples) continue;

    const samplesPerChain = Math.floor(samples.length / numChains);
    const effectiveEnd = end ?? samplesPerChain;

    for (let chain = 0; chain < numChains; chain++) {
      const chainStart = chain * samplesPerChain;

      for (
        let i = start;
        i < Math.min(effectiveEnd, samplesPerChain);
        i += thin
      ) {
        data.push({
          param,
          chain,
          iteration: i,
          value: samples[chainStart + i],
        });
      }
    }
  }

  return data;
}

/**
 * Observable Plot specification for trace plots
 *
 * Returns a specification object that can be passed directly to Plot.plot().
 *
 * @param result - Inference result from sample()
 * @param options - Plot options
 * @returns Observable Plot specification
 *
 * @example
 * ```typescript
 * import { observablePlotTrace } from 'bayesiangpu';
 * import * as Plot from '@observablehq/plot';
 *
 * const spec = observablePlotTrace(result);
 * const plot = Plot.plot(spec);
 * document.body.appendChild(plot);
 * ```
 */
export function observablePlotTrace(
  result: InferenceResult,
  options: TracePlotOptions & {
    width?: number;
    height?: number;
    colors?: string[];
  } = {}
): Record<string, unknown> {
  const data = toTracePlotData(result, options);
  const { width = 800, height, colors } = options;
  const params = [...new Set(data.map((d) => d.param))];

  // Calculate height based on number of parameters if not specified
  const plotHeight = height ?? Math.max(200, 150 * params.length);

  return {
    width,
    height: plotHeight,
    color: colors
      ? { domain: [...Array(result.config.numChains).keys()], range: colors }
      : undefined,
    facet: {
      data,
      y: 'param',
    },
    fy: {
      label: null,
    },
    marks: [
      {
        type: 'line',
        data,
        x: 'iteration',
        y: 'value',
        stroke: 'chain',
        strokeWidth: 0.5,
        strokeOpacity: 0.8,
      },
    ],
  };
}

/**
 * Get chain-specific samples for a parameter
 *
 * Useful for computing per-chain statistics or creating custom visualizations.
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @returns Array of arrays, one per chain
 */
export function getChainSamples(
  result: InferenceResult,
  param: string
): number[][] {
  const samples = result.samples[param];
  if (!samples) {
    throw new Error(`Parameter '${param}' not found`);
  }

  const numChains = result.config.numChains;
  const samplesPerChain = Math.floor(samples.length / numChains);
  const chains: number[][] = [];

  for (let chain = 0; chain < numChains; chain++) {
    const start = chain * samplesPerChain;
    chains.push(samples.slice(start, start + samplesPerChain));
  }

  return chains;
}

/**
 * Compute running mean for trace diagnostics
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @returns Running mean by chain
 */
export function computeRunningMean(
  result: InferenceResult,
  param: string
): { chain: number; iteration: number; runningMean: number }[] {
  const chains = getChainSamples(result, param);
  const data: { chain: number; iteration: number; runningMean: number }[] = [];

  for (let chainIdx = 0; chainIdx < chains.length; chainIdx++) {
    const samples = chains[chainIdx];
    let sum = 0;

    for (let i = 0; i < samples.length; i++) {
      sum += samples[i];
      data.push({
        chain: chainIdx,
        iteration: i,
        runningMean: sum / (i + 1),
      });
    }
  }

  return data;
}

/**
 * Rank-normalized trace data
 *
 * Useful for comparing mixing across chains (all chains should have
 * similar rank distributions if well-mixed).
 *
 * @param result - Inference result
 * @param param - Parameter name
 * @returns Rank-normalized trace data
 */
export function rankNormalizedTrace(
  result: InferenceResult,
  param: string
): TracePlotData[] {
  const samples = result.samples[param];
  if (!samples) {
    throw new Error(`Parameter '${param}' not found`);
  }

  // Compute ranks
  const indexed = samples.map((value, idx) => ({ value, idx }));
  indexed.sort((a, b) => a.value - b.value);

  const ranks = new Array<number>(samples.length);
  for (let i = 0; i < indexed.length; i++) {
    ranks[indexed[i].idx] = i / (samples.length - 1);
  }

  // Convert to trace data
  const numChains = result.config.numChains;
  const samplesPerChain = Math.floor(samples.length / numChains);
  const data: TracePlotData[] = [];

  for (let chain = 0; chain < numChains; chain++) {
    const chainStart = chain * samplesPerChain;
    for (let i = 0; i < samplesPerChain; i++) {
      data.push({
        param,
        chain,
        iteration: i,
        value: ranks[chainStart + i],
      });
    }
  }

  return data;
}
