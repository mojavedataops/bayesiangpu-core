"""
04 - Bayesian Logistic Regression

Binary classification using LinearPredictor + Bernoulli likelihood.
The model applies a logistic (sigmoid) link automatically:

    P(y=1) = sigmoid(X @ beta)

Priors: beta ~ Normal(0, 5).
"""

import numpy as np
import bayesiangpu as bg

# --- Synthetic data ---
np.random.seed(7)
N, P = 30, 2
true_beta = np.array([1.0, -1.5])

X = np.random.randn(N, P)
logits = X @ true_beta
prob = 1.0 / (1.0 + np.exp(-logits))
y = (np.random.rand(N) < prob).astype(float)

# --- Model ---
model = bg.Model()
model.param("beta", bg.Normal(0, 5), size=P)
model.observe(bg.Bernoulli(bg.LinearPredictor(X, "beta")), y.tolist())

# --- Inference ---
result = bg.sample(model, num_samples=500, num_chains=2, seed=42)

# --- Results ---
print("=" * 55)
print("Bayesian Logistic Regression")
print("=" * 55)
print(f"Data: N={N}, P={P}, prevalence={y.mean():.2f}")
print()
print(f"{'Coefficient':<14} {'True':>8} {'Estimated':>10} {'Std':>8}")
print("-" * 44)

for i in range(P):
    s = result.summarize(f"beta[{i}]")
    print(f"{'beta[' + str(i) + ']':<14} {true_beta[i]:>8.3f} {s.mean:>10.3f} {s.std:>8.3f}")

print()
print(f"Converged: {result.is_converged()}")
