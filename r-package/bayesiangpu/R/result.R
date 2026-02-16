#' Inference Result
#'
#' An R6 class containing MCMC sampling results and diagnostics.
#'
#' @importFrom R6 R6Class
#' @importFrom tibble tibble
#' @export
InferenceResult <- R6::R6Class(
  "InferenceResult",
  public = list(
    #' @field samples Named list of sample vectors for each parameter
    samples = NULL,

    #' @field chains Named list of chain samples (list of vectors)
    chains = NULL,

    #' @field rhat Named list of R-hat values
    rhat = NULL,

    #' @field ess Named list of effective sample size values
    ess = NULL,

    #' @field divergences Number of divergent transitions
    divergences = NULL,

    #' @field step_size Final adapted step size
    step_size = NULL,

    #' @field num_samples Number of samples per chain
    num_samples = NULL,

    #' @field num_warmup Number of warmup iterations
    num_warmup = NULL,

    #' @field num_chains Number of chains
    num_chains = NULL,

    #' @field param_names Names of parameters
    param_names = NULL,

    #' @description Create a new InferenceResult from Rust output
    #' @param result_list List returned from run_nuts_sampling
    initialize = function(result_list) {
      self$samples <- result_list$samples
      self$chains <- result_list$chains
      self$rhat <- result_list$rhat
      self$ess <- result_list$ess
      self$divergences <- result_list$divergences
      self$step_size <- result_list$step_size
      self$num_samples <- result_list$num_samples
      self$num_warmup <- result_list$num_warmup
      self$num_chains <- result_list$num_chains
      self$param_names <- result_list$param_names
    },

    #' @description Get summary statistics for a parameter
    #' @param param_name Name of the parameter
    #' @return A list with mean, std, q025, q50, q975, rhat, ess
    summarize = function(param_name) {
      samples <- self$samples[[param_name]]
      list(
        mean = mean(samples),
        std = sd(samples),
        q025 = quantile(samples, 0.025, names = FALSE),
        q50 = quantile(samples, 0.5, names = FALSE),
        q975 = quantile(samples, 0.975, names = FALSE),
        rhat = self$rhat[[param_name]],
        ess = self$ess[[param_name]]
      )
    },

    #' @description Get summary for all parameters as a data frame
    #' @return A data frame with summary statistics
    summary = function() {
      rows <- lapply(self$param_names, function(name) {
        s <- self$summarize(name)
        data.frame(
          parameter = name,
          mean = s$mean,
          std = s$std,
          q2.5 = s$q025,
          q50 = s$q50,
          q97.5 = s$q975,
          rhat = s$rhat,
          ess = s$ess,
          stringsAsFactors = FALSE
        )
      })
      do.call(rbind, rows)
    },

    #' @description Format summary as a string
    #' @return Formatted summary string
    format_summary = function() {
      df <- self$summary()
      paste(capture.output(print(df, row.names = FALSE)), collapse = "\n")
    },

    #' @description Check for sampling warnings
    #' @return Vector of warning messages (empty if none)
    warnings = function() {
      warns <- character()

      # Check R-hat
      for (name in self$param_names) {
        if (self$rhat[[name]] > 1.01) {
          warns <- c(warns, sprintf("High R-hat for %s: %.3f (should be < 1.01)",
                                    name, self$rhat[[name]]))
        }
      }

      # Check ESS
      for (name in self$param_names) {
        if (self$ess[[name]] < 400) {
          warns <- c(warns, sprintf("Low ESS for %s: %.0f (should be > 400)",
                                    name, self$ess[[name]]))
        }
      }

      # Check divergences
      if (self$divergences > 0) {
        warns <- c(warns, sprintf("%d divergent transitions (should be 0)",
                                  self$divergences))
      }

      warns
    },

    #' @description Check if sampling converged
    #' @return TRUE if all R-hat < 1.01 and ESS > 400
    is_converged = function() {
      length(self$warnings()) == 0
    },

    #' @description Convert to a tibble (for ggplot2, dplyr, etc.)
    #' @return A tibble with samples
    as_tibble = function() {
      df <- as.data.frame(self$samples)
      tibble::as_tibble(df)
    },

    #' @description Print the result
    print = function() {
      cat("InferenceResult\n")
      cat(sprintf("  Chains: %d, Samples per chain: %d\n",
                  self$num_chains, self$num_samples))
      cat(sprintf("  Parameters: %s\n", paste(self$param_names, collapse = ", ")))
      cat(sprintf("  Divergences: %d\n", self$divergences))
      cat("\nSummary:\n")
      print(self$summary(), row.names = FALSE)

      warns <- self$warnings()
      if (length(warns) > 0) {
        cat("\nWarnings:\n")
        for (w in warns) {
          cat(sprintf("  - %s\n", w))
        }
      }

      invisible(self)
    }
  )
)

#' @export
as.data.frame.InferenceResult <- function(x, ...) {
  as.data.frame(x$samples)
}

#' @export
summary.InferenceResult <- function(object, ...) {
  object$summary()
}
