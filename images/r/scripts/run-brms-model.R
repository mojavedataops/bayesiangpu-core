#!/usr/bin/env Rscript
#
# Execute brms model for BayesianGPU BaaS
#
# Usage: Rscript run-brms-model.R --model /path/model.R --data /path/data.json --config /path/config.json --output /path/results
#
# This script:
# 1. Sources the model.R file (defines formula, family, priors)
# 2. Loads data from JSON
# 3. Reads sampling configuration
# 4. Runs brm() with cmdstanr backend
# 5. Exports posterior samples and diagnostics
#

library(brms)
library(cmdstanr)
library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(name, default = NULL, required = FALSE) {
  idx <- which(args == name)
  if (length(idx) > 0 && idx < length(args)) {
    return(args[idx + 1])
  }
  if (required) {
    stop(sprintf("Required argument %s not provided", name))
  }
  return(default)
}

model_path <- parse_arg("--model", required = TRUE)
data_path <- parse_arg("--data", required = TRUE)
config_path <- parse_arg("--config", required = TRUE)
output_dir <- parse_arg("--output", required = TRUE)

# Validate paths
if (!file.exists(model_path)) {
  stop(sprintf("Model file not found: %s", model_path))
}
if (!file.exists(data_path)) {
  stop(sprintf("Data file not found: %s", data_path))
}
if (!file.exists(config_path)) {
  stop(sprintf("Config file not found: %s", config_path))
}

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

message("========================================")
message("BayesianGPU brms Runner")
message("========================================")
message(sprintf("Model: %s", model_path))
message(sprintf("Data: %s", data_path))
message(sprintf("Config: %s", config_path))
message(sprintf("Output: %s", output_dir))
message("")

# Load configuration
config <- fromJSON(config_path)

num_chains <- config$num_chains %||% 4
num_samples <- config$num_samples %||% 2000
num_warmup <- config$num_warmup %||% 1000
adapt_delta <- config$adapt_delta %||% 0.8
max_treedepth <- config$max_treedepth %||% 10
seed <- config$seed

message("Sampling configuration:")
message(sprintf("  Chains: %d", num_chains))
message(sprintf("  Samples per chain: %d", num_samples))
message(sprintf("  Warmup: %d", num_warmup))
message(sprintf("  adapt_delta: %.2f", adapt_delta))
message(sprintf("  max_treedepth: %d", max_treedepth))
if (!is.null(seed)) message(sprintf("  Seed: %d", seed))
message("")

# Load data
message("Loading data...")
data <- fromJSON(data_path)
if (is.list(data) && !is.data.frame(data)) {
  # Convert list to data.frame if possible
  if (all(sapply(data, length) == length(data[[1]]))) {
    data <- as.data.frame(data)
  }
}
message(sprintf("  Loaded %d observations, %d variables", nrow(data), ncol(data)))
message("")

# Source the model file
# The model.R file should define:
# - model_formula: a brms formula object
# - model_family: a brms family object (e.g., gaussian(), bernoulli())
# - model_priors: (optional) prior specifications
message("Loading model definition...")
source(model_path, local = TRUE)

# Validate model definition
if (!exists("model_formula")) {
  stop("Model file must define 'model_formula'")
}
if (!exists("model_family")) {
  stop("Model file must define 'model_family'")
}

message(sprintf("  Formula: %s", deparse(model_formula)))
message(sprintf("  Family: %s", model_family$family))
if (exists("model_priors")) {
  message(sprintf("  Priors: %d specified", nrow(model_priors)))
}
message("")

# Prepare brm arguments
brm_args <- list(
  formula = model_formula,
  data = data,
  family = model_family,
  chains = num_chains,
  iter = num_warmup + num_samples,
  warmup = num_warmup,
  cores = num_chains,
  backend = "cmdstanr",
  control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
)

if (!is.null(seed)) {
  brm_args$seed <- seed
}

if (exists("model_priors")) {
  brm_args$prior <- model_priors
}

# Run inference
message("Starting MCMC sampling...")
start_time <- Sys.time()

fit <- tryCatch({
  do.call(brm, brm_args)
}, error = function(e) {
  message(sprintf("ERROR: Sampling failed: %s", e$message))
  # Write error to output
  error_info <- list(
    status = "failed",
    error = e$message,
    timestamp = as.character(Sys.time())
  )
  write_json(error_info, file.path(output_dir, "error.json"), auto_unbox = TRUE)
  quit(status = 1)
})

elapsed <- difftime(Sys.time(), start_time, units = "secs")
message(sprintf("\nSampling completed in %.1f seconds", elapsed))
message("")

# Extract posterior samples
message("Extracting posterior samples...")
posterior <- as_draws_df(fit)

# Write posterior to CSV
posterior_path <- file.path(output_dir, "posterior.csv")
write.csv(posterior, posterior_path, row.names = FALSE)
message(sprintf("  Posterior samples saved to: %s", posterior_path))

# Extract summary statistics
message("Computing summary statistics...")
summary_fit <- summary(fit)

# Compute diagnostics
diagnostics <- list(
  rhat = list(),
  ess_bulk = list(),
  ess_tail = list()
)

for (param in rownames(summary_fit$fixed)) {
  diagnostics$rhat[[param]] <- summary_fit$fixed[param, "Rhat"]
  diagnostics$ess_bulk[[param]] <- summary_fit$fixed[param, "Bulk_ESS"]
  diagnostics$ess_tail[[param]] <- summary_fit$fixed[param, "Tail_ESS"]
}

# Add random effects if present
if (!is.null(summary_fit$random)) {
  for (group_name in names(summary_fit$random)) {
    group_summary <- summary_fit$random[[group_name]]
    for (param in rownames(group_summary)) {
      full_name <- sprintf("%s__%s", group_name, param)
      diagnostics$rhat[[full_name]] <- group_summary[param, "Rhat"]
      diagnostics$ess_bulk[[full_name]] <- group_summary[param, "Bulk_ESS"]
      diagnostics$ess_tail[[full_name]] <- group_summary[param, "Tail_ESS"]
    }
  }
}

# Check for convergence issues
max_rhat <- max(unlist(diagnostics$rhat), na.rm = TRUE)
min_ess <- min(unlist(diagnostics$ess_bulk), na.rm = TRUE)

convergence_warning <- NULL
if (max_rhat > 1.05) {
  convergence_warning <- sprintf("High R-hat detected (%.3f > 1.05). Chain may not have converged.", max_rhat)
  message(sprintf("WARNING: %s", convergence_warning))
}
if (min_ess < 400) {
  ess_warning <- sprintf("Low ESS detected (%d < 400). Consider increasing samples.", min_ess)
  if (is.null(convergence_warning)) {
    convergence_warning <- ess_warning
  } else {
    convergence_warning <- paste(convergence_warning, ess_warning, sep = " ")
  }
  message(sprintf("WARNING: %s", ess_warning))
}

# Prepare summary JSON
summary_json <- list(
  status = "completed",
  elapsed_seconds = as.numeric(elapsed),
  num_chains = num_chains,
  num_samples = num_samples,
  num_warmup = num_warmup,
  parameters = list(),
  diagnostics = diagnostics,
  warnings = convergence_warning,
  timestamp = as.character(Sys.time())
)

# Add parameter summaries
for (param in rownames(summary_fit$fixed)) {
  summary_json$parameters[[param]] <- list(
    mean = summary_fit$fixed[param, "Estimate"],
    sd = summary_fit$fixed[param, "Est.Error"],
    q2.5 = summary_fit$fixed[param, "l-95% CI"],
    q97.5 = summary_fit$fixed[param, "u-95% CI"]
  )
}

# Write summary
summary_path <- file.path(output_dir, "summary.json")
write_json(summary_json, summary_path, auto_unbox = TRUE, pretty = TRUE)
message(sprintf("  Summary saved to: %s", summary_path))

# Write diagnostics separately for easy access
diagnostics_path <- file.path(output_dir, "diagnostics.json")
write_json(diagnostics, diagnostics_path, auto_unbox = TRUE, pretty = TRUE)
message(sprintf("  Diagnostics saved to: %s", diagnostics_path))

message("")
message("========================================")
message("brms sampling completed successfully")
message("========================================")
message(sprintf("Total time: %.1f seconds", as.numeric(elapsed)))
message(sprintf("Max R-hat: %.4f", max_rhat))
message(sprintf("Min ESS: %d", min_ess))
