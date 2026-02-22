# 04_logistic_regression.R -- Logistic regression with LinearPredictor
#
# Binary classification: predict pass/fail from two features.
#   logit(p) = X %*% beta
#   y ~ Bernoulli(p)
#
# The LinearPredictor computes X %*% beta; the Bernoulli likelihood
# applies the inverse-logit (sigmoid) link internally.
#
# Priors:
#   beta ~ Normal(0, 5)   (vector of length 3: intercept + 2 features)

library(bayesiangpu)

# --- Generate synthetic data ------------------------------------------------
set.seed(456)
n <- 120
true_beta <- c(-0.5, 1.2, -0.8)  # intercept, x1, x2

x1 <- rnorm(n)
x2 <- rnorm(n)
X  <- cbind(1, x1, x2)

logits <- X %*% true_beta
probs  <- 1 / (1 + exp(-logits))
y      <- rbinom(n, size = 1, prob = probs)

cat("=== Logistic Regression (LinearPredictor + Bernoulli) ===\n")
cat(sprintf("True beta = [%.2f, %.2f, %.2f]\n",
            true_beta[1], true_beta[2], true_beta[3]))
cat(sprintf("Observed: %d successes out of %d trials\n", sum(y), n))

# --- Build model ------------------------------------------------------------
model <- Model$new()
model$param("beta", Normal(0, 5), size = 3)
model$observe(Bernoulli(LinearPredictor(X, "beta")), y)

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
