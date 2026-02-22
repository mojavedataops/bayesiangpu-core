"""
02 - Normal Estimation

Estimate the mean and standard deviation of normally distributed data.
We use a Normal(0, 10) prior on mu and HalfNormal(5) prior on sigma.

Note: sigma is sampled in unconstrained (log) space, so we apply
exp() to get the actual scale parameter.
"""

import numpy as np
import bayesiangpu as bg

# --- Synthetic data ---
np.random.seed(123)
true_mu = 3.5
true_sigma = 1.2
data = np.random.normal(true_mu, true_sigma, size=50).tolist()

# --- Model ---
model = bg.Model()
model.param("mu", bg.Normal(0, 10))
model.param("sigma", bg.HalfNormal(5))
model.observe(bg.Normal("mu", "sigma"), data)

# --- Inference ---
result = bg.sample(model, num_samples=500, num_chains=2, seed=42)

# --- Results ---
mu_s = result.summarize("mu")

# sigma is in log space; transform samples back
sigma_samples = np.array(result.get_samples("sigma"))
sigma_constrained = np.exp(sigma_samples)
sigma_mean = float(np.mean(sigma_constrained))
sigma_std = float(np.std(sigma_constrained))

print("=" * 50)
print("Normal Estimation: mu and sigma")
print("=" * 50)
print(f"Data: N={len(data)}, sample mean={np.mean(data):.3f}, sample sd={np.std(data):.3f}")
print()
print(f"{'Parameter':<12} {'True':>8} {'Estimated':>10} {'Std':>8}")
print("-" * 42)
print(f"{'mu':<12} {true_mu:>8.3f} {mu_s.mean:>10.3f} {mu_s.std:>8.3f}")
print(f"{'sigma':<12} {true_sigma:>8.3f} {sigma_mean:>10.3f} {sigma_std:>8.3f}")
print()
print(f"mu  95% CI:    [{mu_s.q025:.3f}, {mu_s.q975:.3f}]")
print(f"Converged:     {result.is_converged()}")
