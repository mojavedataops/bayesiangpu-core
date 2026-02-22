# 06_poisson_regression.R -- Poisson regression for count data
#
# Model insect counts as a function of a continuous predictor (e.g., temperature).
#   log(lambda) = X %*% beta
#   counts ~ Poisson(lambda)
#
# LinearPredictor computes the linear combination; the Poisson likelihood
# applies the log link internally.
#
# Priors:
#   beta ~ Normal(0, 5)   (vector of length 2: intercept + slope)

library(bayesiangpu)

# --- Generate synthetic data ------------------------------------------------
set.seed(789)
n <- 100
true_beta <- c(1.5, 0.4)   # intercept, slope (on log scale)

x <- rnorm(n, mean = 0, sd = 1)
X <- cbind(1, x)

log_lambda <- X %*% true_beta
lambda     <- exp(log_lambda)
y          <- rpois(n, lambda = lambda)

cat("=== Poisson Regression (LinearPredictor) ===\n")
cat(sprintf("True beta (log scale) = [%.2f, %.2f]\n",
            true_beta[1], true_beta[2]))
cat(sprintf("Observed counts: mean = %.2f, range = [%d, %d], n = %d\n",
            mean(y), min(y), max(y), n))

# --- Build model (method chaining) ------------------------------------------
model <- Model$new()
model$param("beta", Normal(0, 5), size = 2)
model$observe(Poisson(LinearPredictor(X, "beta")), y)

print(model)

# --- Sample -----------------------------------------------------------------
result <- bg_sample(model, num_samples = 2000, num_chains = 4, seed = 42)
print(result)

# --- Compare ----------------------------------------------------------------
cat("\n--- True vs Estimated (log scale) ---\n")
for (i in seq_along(true_beta)) {
  pname <- sprintf("beta[%d]", i - 1)
  s <- result$summarize(pname)
  cat(sprintf("beta[%d]: true = %6.2f  |  est = %6.3f  [%6.3f, %6.3f]\n",
              i - 1, true_beta[i], s$mean, s$q025, s$q975))
}
