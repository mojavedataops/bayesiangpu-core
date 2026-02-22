# 03_linear_regression.R -- Linear regression with LinearPredictor
#
# Recover intercept and slope from synthetic data:
#   y = beta0 + beta1 * x + noise
#
# Priors:
#   beta  ~ Normal(0, 10)   (vector of length 2)
#   sigma ~ HalfNormal(5)
# Likelihood:
#   y ~ Normal(LinearPredictor(X, "beta"), sigma)

library(bayesiangpu)

# --- Generate synthetic data ------------------------------------------------
set.seed(321)
n <- 80
true_beta <- c(2.0, -0.75)   # intercept, slope
x <- rnorm(n)
y <- true_beta[1] + true_beta[2] * x + rnorm(n, sd = 0.5)

# Design matrix with intercept column
X <- cbind(1, x)

cat("=== Linear Regression (LinearPredictor) ===\n")
cat(sprintf("True beta = [%.2f, %.2f], noise sd = 0.50\n",
            true_beta[1], true_beta[2]))

# --- Build model ------------------------------------------------------------
model <- Model$new()
model$param("beta", Normal(0, 10), size = 2)
model$param("sigma", HalfNormal(5))
model$observe(Normal(LinearPredictor(X, "beta"), "sigma"), y)

print(model)

# --- Sample -----------------------------------------------------------------
result <- bg_sample(model, num_samples = 2000, num_chains = 4, seed = 42)
print(result)

# --- Compare ----------------------------------------------------------------
cat("\n--- True vs Estimated ---\n")
for (i in seq_along(true_beta)) {
  pname <- sprintf("beta[%d]", i - 1)
  s <- result$summarize(pname)
  cat(sprintf("beta[%d]: true = %6.2f  |  est = %6.3f  [%6.3f, %6.3f]\n",
              i - 1, true_beta[i], s$mean, s$q025, s$q975))
}
sigma_s <- result$summarize("sigma")
cat(sprintf("sigma:   true =   0.50  |  est = %6.3f  [%6.3f, %6.3f]\n",
            sigma_s$mean, sigma_s$q025, sigma_s$q975))
