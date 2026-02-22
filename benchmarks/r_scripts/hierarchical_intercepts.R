#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
data <- jsonlite::fromJSON(args[1])
n_samples <- as.integer(args[2])
n_warmup <- as.integer(args[3])
n_chains <- as.integer(args[4])
seed <- as.integer(args[5])

library(brms)
df <- data.frame(
  y = data$y,
  group = factor(data$group_ids)
)
fit <- brm(
  y ~ 1 + (1 | group),
  data = df, family = gaussian(),
  prior = c(
    prior(normal(0, 10), class = "Intercept"),
    prior(normal(0, 5), class = "sd"),
    prior(normal(0, 5), class = "sigma")
  ),
  iter = n_warmup + n_samples, warmup = n_warmup,
  chains = n_chains, seed = seed, refresh = 0
)
ess_vals <- posterior::ess_bulk(as.matrix(fit))
cat(jsonlite::toJSON(list(min_ess = min(ess_vals)), auto_unbox = TRUE))
