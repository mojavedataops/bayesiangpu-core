# 05_eight_schools.R -- Eight Schools hierarchical model
#
# Classic meta-analysis from Rubin (1981). Eight schools each ran a
# coaching programme; we model per-school effects with partial pooling.
#
# Hierarchical model:
#   mu    ~ Normal(0, 5)           population mean effect
#   tau   ~ HalfCauchy(5)          between-school std dev
#   theta ~ Normal(mu, tau)        per-school effect (size = 8)
#   y     ~ Normal(theta, sigma)   observed effect (sigma known)

library(bayesiangpu)

# --- Data (Rubin 1981) ------------------------------------------------------
y     <- c(28, 8, -3, 7, -1, 1, 18, 12)
sigma <- c(15, 10, 16, 11, 9, 11, 10, 18)

cat("=== Eight Schools (Hierarchical Model) ===\n")
cat(sprintf("Observed effects:  %s\n", paste(y, collapse = ", ")))
cat(sprintf("Known std errors:  %s\n", paste(sigma, collapse = ", ")))

# --- Build model (pipe syntax) ----------------------------------------------
model <- Model$new() |>
  param("mu", Normal(0, 5)) |>
  param("tau", HalfCauchy(5)) |>
  param("theta", Normal("mu", "tau"), size = 8) |>
  observe(Normal("theta", "sigma_known"), y, known = list(sigma_known = sigma))

print(model)

# --- Sample -----------------------------------------------------------------
result <- bg_sample(model,
                    num_samples = 2000,
                    num_chains  = 4,
                    target_accept = 0.90,
                    seed = 42)
print(result)

# --- Summarize ---------------------------------------------------------------
cat("\n--- Hyperparameters ---\n")
mu_s  <- result$summarize("mu")
tau_s <- result$summarize("tau")
cat(sprintf("mu:  mean = %6.2f  [%6.2f, %6.2f]\n",
            mu_s$mean, mu_s$q025, mu_s$q975))
cat(sprintf("tau: mean = %6.2f  [%6.2f, %6.2f]\n",
            tau_s$mean, tau_s$q025, tau_s$q975))

cat("\n--- School Effects (theta) ---\n")
for (i in 0:7) {
  s <- result$summarize(sprintf("theta[%d]", i))
  cat(sprintf("  School %d: observed = %3d  |  est = %6.2f  [%6.2f, %6.2f]\n",
              i + 1, y[i + 1], s$mean, s$q025, s$q975))
}
