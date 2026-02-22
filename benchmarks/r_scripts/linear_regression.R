#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
data <- jsonlite::fromJSON(args[1])
n_samples <- as.integer(args[2])
n_warmup <- as.integer(args[3])
n_chains <- as.integer(args[4])
seed <- as.integer(args[5])

library(brms)
X <- as.matrix(do.call(cbind, data$X))
df <- as.data.frame(X)
colnames(df) <- paste0("x", seq_len(ncol(df)))
df$y <- data$y
formula_str <- paste("y ~", paste(colnames(df)[colnames(df) != "y"], collapse = " + "))
fit <- brm(
  as.formula(formula_str),
  data = df, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 2), class = "sigma")
  ),
  iter = n_warmup + n_samples, warmup = n_warmup,
  chains = n_chains, seed = seed, refresh = 0
)
ess_vals <- posterior::ess_bulk(as.matrix(fit))
cat(jsonlite::toJSON(list(min_ess = min(ess_vals)), auto_unbox = TRUE))
