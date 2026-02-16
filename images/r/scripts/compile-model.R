#!/usr/bin/env Rscript
#
# Pre-compile common brms model structures to cache Stan executables
#
# Usage: Rscript /opt/compile-model.R [--families gaussian,bernoulli] [--cache-dir /path]
#
# This script pre-compiles common model families to speed up first-run inference.
# Run on container startup or as part of image build process.
#

library(brms)
library(cmdstanr)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default cache directory (persistent volume in production)
cache_dir <- Sys.getenv("CMDSTAN_CACHE", "/workspace/cache")

# Parse --cache-dir argument
cache_idx <- which(args == "--cache-dir")
if (length(cache_idx) > 0 && cache_idx < length(args)) {
  cache_dir <- args[cache_idx + 1]
}

# Create cache directory
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
message(sprintf("Using cache directory: %s", cache_dir))

# Common model families to pre-compile
families <- list(
  gaussian = gaussian(),
  bernoulli = bernoulli(),
  poisson = poisson(),
  negbinomial = negbinomial(),
  student = student()
)

# Parse --families argument to filter
families_idx <- which(args == "--families")
if (length(families_idx) > 0 && families_idx < length(args)) {
  selected <- strsplit(args[families_idx + 1], ",")[[1]]
  families <- families[names(families) %in% selected]
}

if (length(families) == 0) {
  message("No families to compile")
  quit(status = 0)
}

message(sprintf("Pre-compiling %d families: %s",
                length(families),
                paste(names(families), collapse = ", ")))

# Dummy data for compilation
dummy_data <- data.frame(
  y = rnorm(100),
  x = rnorm(100),
  group = rep(1:10, each = 10)
)

# For binary response (bernoulli)
dummy_data$y_binary <- rbinom(100, 1, 0.5)

# For count response (poisson, negbinomial)
dummy_data$y_count <- rpois(100, lambda = 5)

# Results tracking
results <- list()

# Pre-compile each family
for (family_name in names(families)) {
  message(sprintf("\nPre-compiling %s family...", family_name))

  # Select appropriate response variable
  response <- switch(family_name,
    "bernoulli" = "y_binary",
    "poisson" = "y_count",
    "negbinomial" = "y_count",
    "y"  # default for gaussian, student
  )

  # Simple formula that exercises common model structure
  formula <- as.formula(paste(response, "~ x + (1 | group)"))

  start_time <- Sys.time()

  tryCatch({
    # Compile model (dry run - no sampling)
    fit <- brm(
      formula,
      data = dummy_data,
      family = families[[family_name]],
      chains = 0,  # Don't sample, just compile
      backend = "cmdstanr",
      silent = 2
    )

    elapsed <- difftime(Sys.time(), start_time, units = "secs")
    message(sprintf("  %s compiled successfully in %.1f seconds", family_name, elapsed))
    results[[family_name]] <- list(status = "success", time = elapsed)

  }, error = function(e) {
    elapsed <- difftime(Sys.time(), start_time, units = "secs")
    message(sprintf("  %s failed after %.1f seconds: %s", family_name, elapsed, e$message))
    results[[family_name]] <<- list(status = "failed", error = e$message)
  })
}

# Summary
message("\n========================================")
message("Model compilation summary:")
message("========================================")

success_count <- sum(sapply(results, function(r) r$status == "success"))
fail_count <- length(results) - success_count

for (name in names(results)) {
  r <- results[[name]]
  if (r$status == "success") {
    message(sprintf("  [OK] %s (%.1f sec)", name, r$time))
  } else {
    message(sprintf("  [FAIL] %s: %s", name, r$error))
  }
}

message(sprintf("\nTotal: %d succeeded, %d failed", success_count, fail_count))
message("Model compilation caching complete")

# Exit with error if any compilations failed
if (fail_count > 0) {
  quit(status = 1)
}
