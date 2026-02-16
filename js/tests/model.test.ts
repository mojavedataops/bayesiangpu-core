/**
 * Tests for the Model DSL
 */

import { describe, it, expect } from 'vitest';
import {
  Model,
  Normal,
  Beta,
  Gamma,
  HalfNormal,
  Uniform,
  Bernoulli,
  Binomial,
  Exponential,
  StudentT,
  Cauchy,
  LogNormal,
  Poisson,
} from '../src/index';

describe('Model DSL', () => {
  describe('Model builder', () => {
    it('should throw when building without likelihood', () => {
      const model = new Model();
      expect(() => model.build()).toThrow('Model must have a likelihood');
    });

    it('should add a param with Normal distribution', () => {
      const model = new Model()
        .param('mu', Normal(0, 1))
        .observe(Normal('mu', 1), [1, 2, 3]);
      const spec = model.build();

      expect(spec.priors).toHaveLength(1);
      expect(spec.priors[0].name).toBe('mu');
      expect(spec.priors[0].distribution.type).toBe('Normal');
      expect(spec.priors[0].distribution.params.loc).toBe(0);
      expect(spec.priors[0].distribution.params.scale).toBe(1);
    });

    it('should add multiple params', () => {
      const model = new Model()
        .param('mu', Normal(0, 1))
        .param('sigma', HalfNormal(1))
        .observe(Normal('mu', 'sigma'), [1, 2, 3]);
      const spec = model.build();

      expect(spec.priors).toHaveLength(2);
      expect(spec.priors[0].name).toBe('mu');
      expect(spec.priors[1].name).toBe('sigma');
    });

    it('should set likelihood with observe()', () => {
      const data = [1.5, 2.3, 3.1];
      const model = new Model()
        .param('mu', Normal(0, 1))
        .observe(Normal('mu', 1), data);
      const spec = model.build();

      expect(spec.likelihood).toBeDefined();
      expect(spec.likelihood.distribution.type).toBe('Normal');
      expect(spec.likelihood.observed).toEqual(data);
    });

    it('should chain methods fluently', () => {
      const model = new Model()
        .param('alpha', Beta(1, 1))
        .param('rate', Gamma(1, 1))
        .observe(Bernoulli('alpha'), [1, 0, 1]);

      expect(model).toBeInstanceOf(Model);
      const spec = model.build();
      expect(spec.priors).toHaveLength(2);
      expect(spec.priors.map((p) => p.name)).toContain('alpha');
      expect(spec.priors.map((p) => p.name)).toContain('rate');
    });

    it('should serialize to JSON', () => {
      const model = new Model()
        .param('theta', Beta(1, 1))
        .observe(Binomial(100, 'theta'), [65]);

      const json = model.toJSON();
      const parsed = JSON.parse(json);

      expect(parsed.priors).toHaveLength(1);
      expect(parsed.priors[0].name).toBe('theta');
      expect(parsed.likelihood.distribution.type).toBe('Binomial');
      expect(parsed.likelihood.observed).toEqual([65]);
    });
  });

  describe('Distribution factories', () => {
    it('should create Normal distribution', () => {
      const dist = Normal(0, 1);
      expect(dist.type).toBe('Normal');
      expect(dist.params.loc).toBe(0);
      expect(dist.params.scale).toBe(1);
    });

    it('should create Normal with parameter references', () => {
      const dist = Normal('mu', 'sigma');
      expect(dist.params.loc).toBe('mu');
      expect(dist.params.scale).toBe('sigma');
    });

    it('should create Beta distribution', () => {
      const dist = Beta(2, 5);
      expect(dist.type).toBe('Beta');
      expect(dist.params.alpha).toBe(2);
      expect(dist.params.beta).toBe(5);
    });

    it('should create Gamma distribution', () => {
      const dist = Gamma(2, 0.5);
      expect(dist.type).toBe('Gamma');
      expect(dist.params.shape).toBe(2);
      expect(dist.params.rate).toBe(0.5);
    });

    it('should create HalfNormal distribution', () => {
      const dist = HalfNormal(2);
      expect(dist.type).toBe('HalfNormal');
      expect(dist.params.scale).toBe(2);
    });

    it('should create Uniform distribution', () => {
      const dist = Uniform(0, 10);
      expect(dist.type).toBe('Uniform');
      expect(dist.params.low).toBe(0);
      expect(dist.params.high).toBe(10);
    });

    it('should create Bernoulli distribution', () => {
      const dist = Bernoulli(0.5);
      expect(dist.type).toBe('Bernoulli');
      expect(dist.params.p).toBe(0.5);

      // With parameter reference
      const distRef = Bernoulli('theta');
      expect(distRef.params.p).toBe('theta');
    });

    it('should create Binomial distribution', () => {
      const dist = Binomial(100, 0.5);
      expect(dist.type).toBe('Binomial');
      expect(dist.params.n).toBe(100);
      expect(dist.params.p).toBe(0.5);

      // With parameter reference
      const distRef = Binomial(100, 'theta');
      expect(distRef.params.p).toBe('theta');
    });

    it('should create Poisson distribution', () => {
      const dist = Poisson(5);
      expect(dist.type).toBe('Poisson');
      expect(dist.params.rate).toBe(5);

      // With parameter reference
      const distRef = Poisson('lambda');
      expect(distRef.params.rate).toBe('lambda');
    });

    it('should create Exponential distribution', () => {
      const dist = Exponential(0.5);
      expect(dist.type).toBe('Exponential');
      expect(dist.params.rate).toBe(0.5);
    });

    it('should create StudentT distribution', () => {
      const dist = StudentT(3, 0, 1);
      expect(dist.type).toBe('StudentT');
      expect(dist.params.df).toBe(3);
      expect(dist.params.loc).toBe(0);
      expect(dist.params.scale).toBe(1);

      // With defaults
      const distDefault = StudentT(5);
      expect(distDefault.params.loc).toBe(0);
      expect(distDefault.params.scale).toBe(1);
    });

    it('should create Cauchy distribution', () => {
      const dist = Cauchy(0, 2.5);
      expect(dist.type).toBe('Cauchy');
      expect(dist.params.loc).toBe(0);
      expect(dist.params.scale).toBe(2.5);

      // With defaults
      const distDefault = Cauchy();
      expect(distDefault.params.loc).toBe(0);
      expect(distDefault.params.scale).toBe(1);
    });

    it('should create LogNormal distribution', () => {
      const dist = LogNormal(0, 1);
      expect(dist.type).toBe('LogNormal');
      expect(dist.params.loc).toBe(0);
      expect(dist.params.scale).toBe(1);

      // With parameter references
      const distRef = LogNormal('mu', 'sigma');
      expect(distRef.params.loc).toBe('mu');
      expect(distRef.params.scale).toBe('sigma');
    });
  });

  describe('Model examples', () => {
    it('should build Beta-Binomial model', () => {
      const model = new Model()
        .param('theta', Beta(1, 1))
        .observe(Binomial(100, 'theta'), [65]);

      const spec = model.build();
      expect(spec.priors[0].distribution.type).toBe('Beta');
      expect(spec.likelihood.distribution.type).toBe('Binomial');
    });

    it('should build Normal-Normal model with HalfNormal scale prior', () => {
      const data = [1.2, 2.3, 1.8, 2.1, 1.9];
      const model = new Model()
        .param('mu', Normal(0, 10))
        .param('sigma', HalfNormal(1))
        .observe(Normal('mu', 'sigma'), data);

      const spec = model.build();
      expect(spec.priors).toHaveLength(2);
      expect(spec.priors[0].distribution.type).toBe('Normal');
      expect(spec.priors[1].distribution.type).toBe('HalfNormal');
    });

    it('should build model with Uniform prior', () => {
      const model = new Model()
        .param('x', Uniform(-5, 5))
        .observe(Normal('x', 1), [0.5, 1.2, -0.3]);

      const spec = model.build();
      expect(spec.priors[0].distribution.type).toBe('Uniform');
      expect(spec.priors[0].distribution.params.low).toBe(-5);
      expect(spec.priors[0].distribution.params.high).toBe(5);
    });
  });
});
