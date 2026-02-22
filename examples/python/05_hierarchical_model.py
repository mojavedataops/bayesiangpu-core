"""
05 - Eight Schools (Hierarchical Model)

Classic hierarchical model from Rubin (1981). Eight schools each
report a treatment effect (y) with known standard error (sigma).
The hierarchical structure shares information across schools:

    mu    ~ Normal(0, 5)          # grand mean
    tau   ~ HalfCauchy(5)         # between-school spread
    theta[j] ~ Normal(mu, tau)    # school-specific effects
    y[j]  ~ Normal(theta[j], sigma[j])   # observed data

Note: tau is sampled in log space; apply exp() to interpret.
"""

import numpy as np
import bayesiangpu as bg

# --- Data (Rubin, 1981) ---
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
J = len(y)
school_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

# --- Model ---
model = bg.Model()
model.param("mu", bg.Normal(0, 5))
model.param("tau", bg.HalfCauchy(5))
model.param("theta", bg.Normal("mu", "tau"), size=J)
model.observe(bg.Normal("theta", "sigma"), y,
              known={"sigma": sigma})

# --- Inference ---
result = bg.sample(model, num_samples=200, num_warmup=200, num_chains=2, seed=42)

# --- Results ---
mu_s = result.summarize("mu")
tau_samples = np.exp(np.array(result.get_samples("tau")))
tau_mean = float(np.mean(tau_samples))
tau_std = float(np.std(tau_samples))

print("=" * 55)
print("Eight Schools: Hierarchical Model")
print("=" * 55)
print()
print(f"{'Parameter':<12} {'Estimate':>10} {'Std':>8} {'95% CI'}")
print("-" * 55)
print(f"{'mu':<12} {mu_s.mean:>10.2f} {mu_s.std:>8.2f}   [{mu_s.q025:.2f}, {mu_s.q975:.2f}]")
print(f"{'tau':<12} {tau_mean:>10.2f} {tau_std:>8.2f}")
print()
print(f"{'School':<10} {'y_obs':>8} {'sigma':>8} {'theta':>10} {'Std':>8}")
print("-" * 48)
for j in range(J):
    s = result.summarize(f"theta[{j}]")
    print(f"{school_names[j]:<10} {y[j]:>8.1f} {sigma[j]:>8.1f} {s.mean:>10.2f} {s.std:>8.2f}")

print()
print(f"Converged: {result.is_converged()}")
print()
print("Note: theta estimates are shrunk toward the grand mean mu,")
print("demonstrating partial pooling across schools.")
