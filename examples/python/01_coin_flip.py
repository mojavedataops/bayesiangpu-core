"""
01 - Coin Flip (Beta-Binomial Model)

The simplest Bayesian model: estimate the probability of heads from
observed coin flips. We place a Beta(1,1) prior (uniform on [0,1])
on theta and observe 65 heads out of 100 flips.

Analytic posterior: Beta(66, 36), mean = 66/102 ~ 0.647
"""

import bayesiangpu as bg

# --- Data ---
n_trials = 100
n_heads = 65

# --- Model ---
model = bg.Model()
model.param("theta", bg.Beta(1, 1))          # Prior: uniform on [0,1]
model.observe(bg.Binomial(n_trials, "theta"), [float(n_heads)])

# --- Inference ---
result = bg.sample(model, num_samples=1000, num_chains=4, seed=42)

# --- Results ---
s = result.summarize("theta")
analytic_mean = (1 + n_heads) / (2 + n_trials)

print("=" * 50)
print("Coin Flip: Beta-Binomial Model")
print("=" * 50)
print(f"Data: {n_heads} heads in {n_trials} flips")
print()
print(f"Posterior mean:  {s.mean:.4f}  (analytic: {analytic_mean:.4f})")
print(f"Posterior std:   {s.std:.4f}")
print(f"95% CI:          [{s.q025:.4f}, {s.q975:.4f}]")
print(f"R-hat:           {s.rhat:.4f}")
print(f"ESS:             {s.ess:.0f}")
print()
print(f"Converged: {result.is_converged()}")
