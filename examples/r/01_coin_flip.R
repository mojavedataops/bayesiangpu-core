# 01_coin_flip.R -- Beta-Binomial coin flip
#
# The simplest Bayesian model: estimate the bias of a coin.
# Prior:      theta ~ Beta(1, 1)  (uniform on [0,1])
# Likelihood: data  ~ Binomial(100, theta)
# Observed:   65 heads out of 100 flips
#
# Analytical posterior: Beta(1 + 65, 1 + 35) = Beta(66, 36)
# Posterior mean: 66/102 ~ 0.647

library(bayesiangpu)

# --- Build model (method chaining) ----------------------------------------
model <- Model$new()
model$param("theta", Beta(1, 1))
model$observe(Binomial(100, "theta"), 65)

cat("=== Coin Flip Model ===\n")
print(model)

# --- Sample -----------------------------------------------------------------
result <- bg_sample(model, num_samples = 2000, num_chains = 4, seed = 42)
print(result)

# --- Compare to analytical posterior ----------------------------------------
s <- result$summarize("theta")
analytical_mean <- 66 / 102

cat("\n--- Comparison ---\n")
cat(sprintf("Estimated mean:  %.4f\n", s$mean))
cat(sprintf("Analytical mean: %.4f\n", analytical_mean))
cat(sprintf("95%% CI:          [%.4f, %.4f]\n", s$q025, s$q975))
cat(sprintf("R-hat: %.4f  ESS: %.0f\n", s$rhat, s$ess))
