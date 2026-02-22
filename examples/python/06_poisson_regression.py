"""
06 - Poisson Model for Count Data

Estimate the rate of a Poisson process from observed count data.
We use a Gamma prior on the rate parameter, which is the conjugate
prior for the Poisson likelihood.

The rate parameter has a positive-support prior, so samples are in
log space; apply exp() to get the actual rate.

We also demonstrate a multi-group comparison (before vs after
treatment) by fitting separate models.
"""

import numpy as np
import bayesiangpu as bg

# --- Synthetic data ---
np.random.seed(99)
true_rate = 3.5
N = 50
y = np.random.poisson(true_rate, size=N).astype(float).tolist()

# --- Model: Gamma prior (conjugate) ---
model = bg.Model()
model.param("rate", bg.Gamma(2, 0.5))        # Prior: Gamma(shape=2, rate=0.5)
model.observe(bg.Poisson("rate"), y)

result = bg.sample(model, num_samples=500, num_chains=2, seed=42)

# rate is in log space due to positive-support prior
rate_samples = np.exp(np.array(result.get_samples("rate")))
rate_mean = float(np.mean(rate_samples))
rate_std = float(np.std(rate_samples))

# Analytic conjugate posterior: Gamma(2 + sum(y), 0.5 + N)
analytic_mean = (2 + sum(y)) / (0.5 + N)

print("=" * 55)
print("Poisson Model: Estimating Event Rate")
print("=" * 55)
print(f"Data: N={N}, sample mean={np.mean(y):.3f}")
print()
print(f"{'Parameter':<12} {'True':>8} {'Estimated':>10} {'Std':>8}")
print("-" * 42)
print(f"{'rate':<12} {true_rate:>8.3f} {rate_mean:>10.3f} {rate_std:>8.3f}")
print()
print(f"Analytic posterior mean: {analytic_mean:.3f}")
print(f"Converged: {result.is_converged()}")

# --- Multi-group comparison ---
print()
print("=" * 55)
print("Multi-group Poisson: Before vs After treatment")
print("=" * 55)

np.random.seed(42)
rate_before = 5.0
rate_after = 3.0
y_before = np.random.poisson(rate_before, size=30).astype(float).tolist()
y_after = np.random.poisson(rate_after, size=30).astype(float).tolist()

for label, data, true_r in [("Before", y_before, rate_before),
                              ("After", y_after, rate_after)]:
    m = bg.Model()
    m.param("rate", bg.Gamma(2, 0.5))
    m.observe(bg.Poisson("rate"), data)
    r = bg.sample(m, num_samples=500, num_chains=2, seed=42)
    samples = np.exp(np.array(r.get_samples("rate")))
    print(f"{label:>8}: true={true_r:.1f}, est={np.mean(samples):.3f} "
          f"+/- {np.std(samples):.3f}")
