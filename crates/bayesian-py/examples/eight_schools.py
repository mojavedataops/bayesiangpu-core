"""
Eight Schools - A classic hierarchical Bayesian model.

This example implements the Eight Schools model from Rubin (1981), which
estimates the effects of coaching programs on SAT scores across 8 schools.

Model:
    mu    ~ Normal(0, 5)           -- population mean effect
    tau   ~ HalfCauchy(5)          -- between-school standard deviation
    theta ~ Normal(mu, tau)        -- school-specific effects (vector of 8)
    y     ~ Normal(theta, sigma)   -- observed effects with known standard errors

This hierarchical structure allows partial pooling: schools with less data
are shrunk toward the population mean, while schools with more data retain
their individual estimates.
"""

import bayesiangpu as bg

# Observed treatment effects (point estimates from each school)
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]

# Known standard errors of the treatment effect estimates
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

# Build the hierarchical model
model = bg.Model()
model.param("mu", bg.Normal(0, 5))                    # Population mean
model.param("tau", bg.HalfCauchy(5))                   # Between-school SD
model.param("theta", bg.Normal("mu", "tau"), size=8)   # School effects
model.observe(bg.Normal("theta", "sigma"), y, known={"sigma": sigma})

# Run NUTS sampling
result = bg.sample(model, num_samples=1000, num_chains=4)

# Display results
print("Eight Schools - Hierarchical Model Results")
print("=" * 42)
print()
print(result)
