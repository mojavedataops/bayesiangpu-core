#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
data <- jsonlite::fromJSON(args[1])
n_samples <- as.integer(args[2])
n_warmup <- as.integer(args[3])
n_chains <- as.integer(args[4])
seed <- as.integer(args[5])

library(brms)
df <- data.frame(y = data$y, n_trials = data$n_trials)
fit <- brm(
  y | trials(n_trials) ~ 1,
  data = df, family = binomial(),
  iter = n_warmup + n_samples, warmup = n_warmup,
  chains = n_chains, seed = seed, refresh = 0
)
ess_vals <- posterior::ess_bulk(as.matrix(fit))
cat(jsonlite::toJSON(list(min_ess = min(ess_vals)), auto_unbox = TRUE))
