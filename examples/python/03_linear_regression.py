"""
03 - Bayesian Linear Regression

Recover known regression coefficients from synthetic data using
LinearPredictor. The model is:

    y = X @ beta + epsilon,   epsilon ~ Normal(0, sigma)

Priors: beta ~ Normal(0, 5),  sigma ~ HalfNormal(2).

Note: sigma is sampled in log space; apply exp() to interpret.
"""

import numpy as np
import bayesiangpu as bg

# --- Synthetic data ---
np.random.seed(42)
N, P = 20, 3
true_beta = np.array([1.5, -0.8, 0.3])
true_sigma = 0.5

X = np.random.randn(N, P)
y = X @ true_beta + np.random.randn(N) * true_sigma

# --- Model ---
model = bg.Model()
model.param("beta", bg.Normal(0, 5), size=P)
model.param("sigma", bg.HalfNormal(2))
model.observe(bg.Normal(bg.LinearPredictor(X, "beta"), "sigma"), y.tolist())

# --- Inference ---
result = bg.sample(model, num_samples=500, num_chains=2, seed=42)

# --- Results ---
print("=" * 55)
print("Bayesian Linear Regression")
print("=" * 55)
print(f"Data: N={N}, P={P}")
print()
print(f"{'Coefficient':<14} {'True':>8} {'Estimated':>10} {'Std':>8}")
print("-" * 44)

for i in range(P):
    s = result.summarize(f"beta[{i}]")
    print(f"{'beta[' + str(i) + ']':<14} {true_beta[i]:>8.3f} {s.mean:>10.3f} {s.std:>8.3f}")

# sigma in log space
sigma_samples = np.exp(np.array(result.get_samples("sigma")))
print(f"{'sigma':<14} {true_sigma:>8.3f} {np.mean(sigma_samples):>10.3f} {np.std(sigma_samples):>8.3f}")

print()
print(f"Converged: {result.is_converged()}")
