# 02_normal_estimation.R -- Estimate mean and standard deviation
#
# Given normally distributed observations, recover the generating
# parameters using weakly informative priors.
#
# Priors:
#   mu    ~ Normal(0, 10)
#   sigma ~ HalfNormal(5)
# Likelihood:
#   data  ~ Normal(mu, sigma)

library(bayesiangpu)

# --- Generate synthetic data ------------------------------------------------
set.seed(123)
true_mu    <- 3.5
true_sigma <- 1.2
y <- rnorm(50, mean = true_mu, sd = true_sigma)

cat("=== Normal Estimation ===\n")
cat(sprintf("True mu = %.2f, True sigma = %.2f\n", true_mu, true_sigma))
cat(sprintf("Sample mean = %.3f, Sample sd = %.3f (n = %d)\n",
            mean(y), sd(y), length(y)))

# --- Build model (pipe syntax) ----------------------------------------------
model <- Model$new() |>
  param("mu", Normal(0, 10)) |>
  param("sigma", HalfNormal(5)) |>
  observe(Normal("mu", "sigma"), y)

print(model)

# --- Sample -----------------------------------------------------------------
result <- bg_sample(model, num_samples = 2000, num_chains = 4, seed = 42)
print(result)

# --- Compare ----------------------------------------------------------------
mu_s    <- result$summarize("mu")
sigma_s <- result$summarize("sigma")

cat("\n--- True vs Estimated ---\n")
cat(sprintf("mu:    true = %.2f  |  est = %.3f  [%.3f, %.3f]\n",
            true_mu, mu_s$mean, mu_s$q025, mu_s$q975))
cat(sprintf("sigma: true = %.2f  |  est = %.3f  [%.3f, %.3f]\n",
            true_sigma, sigma_s$mean, sigma_s$q025, sigma_s$q975))
