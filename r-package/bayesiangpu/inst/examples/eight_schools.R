# Eight Schools - A classic hierarchical Bayesian model
#
# This example implements the Eight Schools model from Rubin (1981), which
# estimates the effects of coaching programs on SAT scores across 8 schools.
#
# Model:
#   mu    ~ Normal(0, 5)           -- population mean effect
#   tau   ~ HalfCauchy(5)          -- between-school standard deviation
#   theta ~ Normal(mu, tau)        -- school-specific effects (vector of 8)
#   y     ~ Normal(theta, sigma)   -- observed effects with known standard errors
#
# This hierarchical structure allows partial pooling: schools with less data
# are shrunk toward the population mean, while schools with more data retain
# their individual estimates.

library(bayesiangpu)

# Observed treatment effects (point estimates from each school)
y <- c(28, 8, -3, 7, -1, 1, 18, 12)

# Known standard errors of the treatment effect estimates
sigma <- c(15, 10, 16, 11, 9, 11, 10, 18)

# Build the hierarchical model using pipe syntax
model <- Model() |>
  param("mu", Normal(0, 5)) |>                         # Population mean
  param("tau", HalfCauchy(5)) |>                        # Between-school SD
  param("theta", Normal("mu", "tau"), size = 8L) |>     # School effects
  observe(Normal("theta", "sigma"), y, known = list(sigma = sigma))

# Run NUTS sampling
result <- bg_sample(model, num_samples = 1000L, num_chains = 4L, seed = 42L)

# Display results
cat("Eight Schools - Hierarchical Model Results\n")
cat(strrep("=", 42), "\n\n")
summary(result)
