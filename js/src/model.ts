/**
 * Model definition DSL for BayesianGPU
 *
 * This module provides a fluent builder API for defining Bayesian models
 * that can be compiled to JSON and run through the WASM inference engine.
 *
 * @example
 * ```typescript
 * import { Model, Beta, Binomial } from 'bayesiangpu';
 *
 * const model = new Model()
 *   .param('theta', Beta(1, 1))
 *   .observe(Binomial(100, 'theta'), [65])
 *   .build();
 * ```
 *
 * @module
 */

/**
 * Distribution specification
 */
export interface Distribution {
  /** Distribution type (e.g., 'Normal', 'Beta', 'Binomial') */
  type: string;
  /** Distribution parameters - can be numbers or parameter references (strings) */
  params: Record<string, number | number[] | string>;
}

/**
 * Prior specification for a parameter
 */
export interface Prior {
  /** Parameter name */
  name: string;
  /** Prior distribution */
  distribution: Distribution;
}

/**
 * Likelihood specification
 */
export interface Likelihood {
  /** Likelihood distribution */
  distribution: Distribution;
  /** Observed data */
  observed: number[];
}

/**
 * Complete model specification
 */
export interface ModelSpec {
  /** Prior distributions for each parameter */
  priors: Prior[];
  /** Likelihood specification */
  likelihood: Likelihood;
}

/**
 * Fluent builder for Bayesian models
 *
 * @example
 * ```typescript
 * const model = new Model()
 *   .param('mu', Normal(0, 10))
 *   .param('sigma', HalfNormal(1))
 *   .observe(Normal('mu', 'sigma'), data)
 *   .build();
 * ```
 */
export class Model {
  private spec: ModelSpec = {
    priors: [],
    likelihood: null!,
  };

  /**
   * Add a parameter with a prior distribution
   *
   * @param name - Parameter name (used to reference in likelihood)
   * @param distribution - Prior distribution
   * @returns this for method chaining
   *
   * @example
   * ```typescript
   * model.param('theta', Beta(1, 1))
   * ```
   */
  param(name: string, distribution: Distribution): this {
    this.spec.priors.push({ name, distribution });
    return this;
  }

  /**
   * Set the likelihood (observed data) for the model
   *
   * @param distribution - Likelihood distribution
   * @param data - Observed data points
   * @returns this for method chaining
   *
   * @example
   * ```typescript
   * model.observe(Binomial(100, 'theta'), [65])
   * ```
   */
  observe(distribution: Distribution, data: number[]): this {
    this.spec.likelihood = {
      distribution,
      observed: data,
    };
    return this;
  }

  /**
   * Build the model specification
   *
   * @returns The complete model specification ready for inference
   *
   * @throws Error if no likelihood has been specified
   */
  build(): ModelSpec {
    if (!this.spec.likelihood) {
      throw new Error('Model must have a likelihood. Call observe() before build().');
    }
    return { ...this.spec };
  }

  /**
   * Convert model to JSON string for WASM
   */
  toJSON(): string {
    return JSON.stringify(this.build());
  }
}

// ============================================================================
// Distribution Factory Functions
// ============================================================================

/**
 * Normal (Gaussian) distribution
 *
 * @param loc - Mean (can be a number or parameter name)
 * @param scale - Standard deviation (can be a number or parameter name)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Normal(0, 1)           // Standard normal
 * Normal('mu', 'sigma')  // Parameters from model
 * ```
 */
export const Normal = (
  loc: number | string,
  scale: number | string
): Distribution => ({
  type: 'Normal',
  params: { loc, scale },
});

/**
 * Half-Normal distribution (positive values only)
 *
 * Useful for scale parameters.
 *
 * @param scale - Scale parameter
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * HalfNormal(1)  // Half-normal with scale 1
 * ```
 */
export const HalfNormal = (scale: number): Distribution => ({
  type: 'HalfNormal',
  params: { scale },
});

/**
 * Beta distribution
 *
 * Defined on (0, 1), useful for probability parameters.
 *
 * @param alpha - Shape parameter alpha (concentration1)
 * @param beta - Shape parameter beta (concentration0)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Beta(1, 1)   // Uniform on [0, 1]
 * Beta(2, 5)   // Skewed toward 0
 * ```
 */
export const Beta = (alpha: number, beta: number): Distribution => ({
  type: 'Beta',
  params: { alpha, beta },
});

/**
 * Gamma distribution
 *
 * Defined on (0, infinity), useful for positive parameters.
 *
 * @param shape - Shape parameter (k)
 * @param rate - Rate parameter (1/scale)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Gamma(2, 1)  // Shape=2, rate=1
 * ```
 */
export const Gamma = (shape: number, rate: number): Distribution => ({
  type: 'Gamma',
  params: { shape, rate },
});

/**
 * Uniform distribution
 *
 * @param low - Lower bound
 * @param high - Upper bound
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Uniform(0, 10)  // Uniform on [0, 10]
 * ```
 */
export const Uniform = (low: number, high: number): Distribution => ({
  type: 'Uniform',
  params: { low, high },
});

/**
 * Bernoulli distribution
 *
 * For binary outcomes (0 or 1).
 *
 * @param p - Probability of success (can be a number or parameter name)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Bernoulli(0.5)       // 50% success rate
 * Bernoulli('theta')   // Parameter from model
 * ```
 */
export const Bernoulli = (p: number | string): Distribution => ({
  type: 'Bernoulli',
  params: { p },
});

/**
 * Binomial distribution
 *
 * For count of successes in n trials.
 *
 * @param n - Number of trials
 * @param p - Probability of success (can be a number or parameter name)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Binomial(100, 0.5)      // 100 trials, 50% success
 * Binomial(100, 'theta')  // Parameter from model
 * ```
 */
export const Binomial = (n: number, p: number | string): Distribution => ({
  type: 'Binomial',
  params: { n, p },
});

/**
 * Poisson distribution
 *
 * For count data.
 *
 * @param rate - Rate parameter (expected count)
 * @returns Distribution specification
 *
 * @example
 * ```typescript
 * Poisson(5)          // Rate of 5
 * Poisson('lambda')   // Parameter from model
 * ```
 */
export const Poisson = (rate: number | string): Distribution => ({
  type: 'Poisson',
  params: { rate },
});

/**
 * Exponential distribution
 *
 * For waiting times.
 *
 * @param rate - Rate parameter (inverse of mean)
 * @returns Distribution specification
 */
export const Exponential = (rate: number): Distribution => ({
  type: 'Exponential',
  params: { rate },
});

/**
 * Student's t distribution
 *
 * For robust inference with potential outliers.
 *
 * @param df - Degrees of freedom
 * @param loc - Location parameter
 * @param scale - Scale parameter
 * @returns Distribution specification
 */
export const StudentT = (
  df: number,
  loc: number | string = 0,
  scale: number | string = 1
): Distribution => ({
  type: 'StudentT',
  params: { df, loc, scale },
});

/**
 * Cauchy distribution
 *
 * Heavy-tailed distribution useful for weakly informative priors.
 *
 * @param loc - Location parameter
 * @param scale - Scale parameter
 * @returns Distribution specification
 */
export const Cauchy = (loc: number = 0, scale: number = 1): Distribution => ({
  type: 'Cauchy',
  params: { loc, scale },
});

/**
 * Log-Normal distribution
 *
 * For positive values with multiplicative effects.
 *
 * @param loc - Mean of the log (mu)
 * @param scale - Standard deviation of the log (sigma)
 * @returns Distribution specification
 */
export const LogNormal = (
  loc: number | string,
  scale: number | string
): Distribution => ({
  type: 'LogNormal',
  params: { loc, scale },
});
