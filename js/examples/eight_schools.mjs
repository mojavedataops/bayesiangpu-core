/**
 * Eight Schools - A classic hierarchical Bayesian model
 *
 * This example implements the Eight Schools model from Rubin (1981), which
 * estimates the effects of coaching programs on SAT scores across 8 schools.
 *
 * Model:
 *   mu    ~ Normal(0, 5)           -- population mean effect
 *   tau   ~ HalfCauchy(5)          -- between-school standard deviation
 *   theta ~ Normal(mu, tau)        -- school-specific effects (vector of 8)
 *   y     ~ Normal(theta, sigma)   -- observed effects with known standard errors
 *
 * This hierarchical structure allows partial pooling: schools with less data
 * are shrunk toward the population mean, while schools with more data retain
 * their individual estimates.
 */

import {
  Model,
  Normal,
  HalfCauchy,
  sample,
  summarizeAll,
  formatSummaryTable,
} from 'bayesiangpu';

// Observed treatment effects (point estimates from each school)
const y = [28, 8, -3, 7, -1, 1, 18, 12];

// Known standard errors of the treatment effect estimates
const sigma = [15, 10, 16, 11, 9, 11, 10, 18];

// Build the hierarchical model
const model = new Model()
  .param('mu', Normal(0, 5))                          // Population mean
  .param('tau', HalfCauchy(5))                         // Between-school SD
  .param('theta', Normal('mu', 'tau'), { size: 8 })    // School effects
  .observe(Normal('theta', 'sigma'), y, { known: { sigma } })  // Likelihood
  .build();

// Run NUTS sampling
const result = await sample(model, {
  numSamples: 1000,
  numChains: 4,
  seed: 42,
});

// Display results
console.log('Eight Schools - Hierarchical Model Results');
console.log('==========================================\n');

const summaries = summarizeAll(result);
console.log(formatSummaryTable(summaries));

// Check convergence diagnostics
console.log('\nDiagnostics:');
console.log(`  Divergences: ${result.diagnostics.divergences}`);
for (const [param, rhat] of Object.entries(result.diagnostics.rhat)) {
  const ess = result.diagnostics.ess[param];
  console.log(`  ${param}: R-hat = ${rhat.toFixed(3)}, ESS = ${ess.toFixed(0)}`);
}
