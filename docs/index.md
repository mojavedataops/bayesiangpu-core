---
layout: home
hero:
  name: BayesianGPU
  text: GPU-accelerated Bayesian inference
  tagline: Run MCMC in the browser or Rust with a unified API
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: API Reference
      link: /api/

features:
  - icon: 🌐
    title: Browser-Native
    details: Run Bayesian inference directly in the browser with WebAssembly. No server required.
  - icon: 🚀
    title: GPU Accelerated
    details: Optional WebGPU backend for GPU-accelerated sampling on supported browsers.
  - icon: 📊
    title: Production Ready
    details: NUTS sampler with dual averaging, mass matrix adaptation, and ArviZ-compatible diagnostics.
  - icon: 🐳
    title: Docker Ready
    details: Pre-built Docker images for NumPyro (Python/JAX) and brms (R/Stan) with GPU support.
---

## Quick Example

```javascript
import { Model, Beta, Binomial, sample } from 'bayesiangpu';

// Define a Beta-Binomial model
const model = new Model()
  .param('theta', Beta(1, 1))           // Prior: uniform on [0,1]
  .observe(Binomial(100, 'theta'), [65]) // Likelihood: 65/100 successes
  .build();

// Run inference
const result = await sample(model, {
  numSamples: 1000,
  numChains: 4,
});

// Check convergence
console.log('R-hat:', result.diagnostics.rhat.theta);
console.log('ESS:', result.diagnostics.ess.theta);
```
