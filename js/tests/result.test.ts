/**
 * Tests for inference result utilities
 */

import { describe, it, expect } from 'vitest';
import {
  summarizeParameter,
  summarizeAll,
  isConverged,
  hasSufficientESS,
  getWarnings,
  formatSummaryTable,
  type InferenceResult,
  type Diagnostics,
} from '../src/result';

describe('Result utilities', () => {
  // Create mock result data (samples flattened across chains)
  const mockResult: InferenceResult = {
    samples: {
      mu: [1.0, 1.1, 1.2, 0.9, 1.05, 0.95, 1.1, 1.0],
      sigma: [0.5, 0.6, 0.55, 0.45, 0.52, 0.48, 0.51, 0.53],
    },
    diagnostics: {
      rhat: { mu: 1.005, sigma: 1.008 },
      ess: { mu: 450, sigma: 420 },
      divergences: 0,
    },
    config: {
      numChains: 2,
      numSamples: 4,
      numWarmup: 100,
      stepSize: 0.1,
    },
  };

  describe('summarizeParameter', () => {
    it('should compute mean for a parameter', () => {
      const summary = summarizeParameter(mockResult, 'mu');

      // Mean of all samples
      const allSamples = mockResult.samples.mu;
      const expectedMean =
        allSamples.reduce((a, b) => a + b, 0) / allSamples.length;

      expect(summary.mean).toBeCloseTo(expectedMean, 5);
    });

    it('should compute std for a parameter', () => {
      const summary = summarizeParameter(mockResult, 'mu');
      expect(summary.std).toBeGreaterThan(0);
    });

    it('should compute quantiles', () => {
      const summary = summarizeParameter(mockResult, 'mu');

      expect(summary.q025).toBeLessThan(summary.q50);
      expect(summary.q50).toBeLessThan(summary.q975);
      expect(summary.q25).toBeLessThan(summary.q75);
    });

    it('should include diagnostics in summary', () => {
      const summary = summarizeParameter(mockResult, 'mu');

      expect(summary.rhat).toBeCloseTo(1.005, 3);
      expect(summary.ess).toBeCloseTo(450, 0);
    });

    it('should throw for unknown parameter', () => {
      expect(() => summarizeParameter(mockResult, 'unknown')).toThrow(
        "Parameter 'unknown' not found"
      );
    });
  });

  describe('summarizeAll', () => {
    it('should return summary for all parameters', () => {
      const summary = summarizeAll(mockResult);

      expect(summary).toHaveProperty('mu');
      expect(summary).toHaveProperty('sigma');
      expect(summary.mu).toHaveProperty('mean');
      expect(summary.mu).toHaveProperty('std');
      expect(summary.mu).toHaveProperty('q025');
      expect(summary.mu).toHaveProperty('q50');
      expect(summary.mu).toHaveProperty('q975');
      expect(summary.mu).toHaveProperty('rhat');
      expect(summary.mu).toHaveProperty('ess');
    });
  });

  describe('isConverged', () => {
    it('should return true when all R-hat values are good', () => {
      expect(isConverged(mockResult.diagnostics)).toBe(true);
    });

    it('should return false when any R-hat is too high', () => {
      const badDiagnostics: Diagnostics = {
        rhat: { mu: 1.5, sigma: 1.008 },
        ess: { mu: 450, sigma: 420 },
        divergences: 0,
      };
      expect(isConverged(badDiagnostics)).toBe(false);
    });

    it('should use 1.01 as default threshold', () => {
      const marginalDiagnostics: Diagnostics = {
        rhat: { mu: 1.009, sigma: 1.008 },
        ess: { mu: 450, sigma: 420 },
        divergences: 0,
      };
      expect(isConverged(marginalDiagnostics)).toBe(true);

      const poorDiagnostics: Diagnostics = {
        rhat: { mu: 1.011, sigma: 1.008 },
        ess: { mu: 450, sigma: 420 },
        divergences: 0,
      };
      expect(isConverged(poorDiagnostics)).toBe(false);
    });
  });

  describe('hasSufficientESS', () => {
    it('should return true when ESS is sufficient', () => {
      expect(hasSufficientESS(mockResult.diagnostics)).toBe(true);
    });

    it('should return false when ESS is too low', () => {
      const lowEssDiagnostics: Diagnostics = {
        rhat: { mu: 1.005, sigma: 1.008 },
        ess: { mu: 50, sigma: 420 },
        divergences: 0,
      };
      expect(hasSufficientESS(lowEssDiagnostics)).toBe(false);
    });

    it('should use custom minimum ESS', () => {
      const diagnostics: Diagnostics = {
        rhat: { mu: 1.005, sigma: 1.008 },
        ess: { mu: 300, sigma: 350 },
        divergences: 0,
      };
      expect(hasSufficientESS(diagnostics, 400)).toBe(false);
      expect(hasSufficientESS(diagnostics, 200)).toBe(true);
    });
  });

  describe('getWarnings', () => {
    it('should return empty array for good results', () => {
      const warnings = getWarnings(mockResult);
      expect(warnings).toHaveLength(0);
    });

    it('should warn on high R-hat', () => {
      const badResult: InferenceResult = {
        ...mockResult,
        diagnostics: {
          ...mockResult.diagnostics,
          rhat: { mu: 1.15, sigma: 1.008 },
        },
      };
      const warnings = getWarnings(badResult);
      expect(warnings.some((w) => w.includes('mu') && w.includes('R-hat'))).toBe(
        true
      );
    });

    it('should warn on low ESS', () => {
      const badResult: InferenceResult = {
        ...mockResult,
        diagnostics: {
          ...mockResult.diagnostics,
          ess: { mu: 50, sigma: 420 },
        },
      };
      const warnings = getWarnings(badResult);
      expect(warnings.some((w) => w.includes('mu') && w.includes('ESS'))).toBe(
        true
      );
    });

    it('should warn on divergences', () => {
      const badResult: InferenceResult = {
        ...mockResult,
        diagnostics: {
          ...mockResult.diagnostics,
          divergences: 10,
        },
      };
      const warnings = getWarnings(badResult);
      expect(warnings.some((w) => w.includes('divergence'))).toBe(true);
    });
  });

  describe('formatSummaryTable', () => {
    it('should format results as table', () => {
      const table = formatSummaryTable(mockResult);

      expect(table).toContain('Parameter');
      expect(table).toContain('Mean');
      expect(table).toContain('mu');
      expect(table).toContain('sigma');
    });

    it('should handle empty results', () => {
      const emptyResult: InferenceResult = {
        samples: {},
        diagnostics: { rhat: {}, ess: {}, divergences: 0 },
        config: { numChains: 4, numSamples: 1000, numWarmup: 500, stepSize: 0.1 },
      };
      const table = formatSummaryTable(emptyResult);
      expect(table).toContain('No parameters');
    });
  });
});
