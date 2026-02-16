#' Bayesian Model Builder
#'
#' An R6 class for building Bayesian models using a fluent API.
#'
#' @importFrom R6 R6Class
#' @export
#'
#' @examples
#' model <- Model()
#' model$param("mu", Normal(0, 10))
#' model$param("sigma", HalfNormal(1))
#' model$observe(Normal("mu", "sigma"), c(2.3, 2.1, 2.5))
#'
#' # Or using pipe syntax
#' model <- Model() |>
#'   param("theta", Beta(1, 1)) |>
#'   observe(Binomial(100, "theta"), 65)
Model <- R6::R6Class(
  "BayesianModel",
  public = list(
    #' @description Create a new empty model
    initialize = function() {
      private$.priors <- list()
      private$.likelihood <- NULL
      private$.observed <- NULL
    },

    #' @description Add a parameter with a prior distribution
    #' @param name Parameter name (used to reference in likelihood)
    #' @param distribution Prior distribution
    #' @return self for method chaining
    param = function(name, distribution) {
      private$.priors[[name]] <- distribution
      invisible(self)
    },

    #' @description Set the likelihood (observed data) for the model
    #' @param distribution Likelihood distribution
    #' @param data Observed data points
    #' @return self for method chaining
    observe = function(distribution, data) {
      private$.likelihood <- distribution
      private$.observed <- as.numeric(data)
      invisible(self)
    },

    #' @description Get the list of parameter names
    param_names = function() {
      names(private$.priors)
    },

    #' @description Get the number of parameters
    num_params = function() {
      length(private$.priors)
    },

    #' @description Check if the model has a likelihood
    has_likelihood = function() {
      !is.null(private$.likelihood)
    },

    #' @description Convert model to JSON string (for internal use)
    to_json = function() {
      priors <- lapply(names(private$.priors), function(name) {
        list(
          name = name,
          distribution = jsonlite::fromJSON(private$.priors[[name]])
        )
      })

      spec <- list(priors = priors)

      if (!is.null(private$.likelihood)) {
        spec$likelihood <- list(
          distribution = jsonlite::fromJSON(private$.likelihood),
          observed = private$.observed
        )
      }

      jsonlite::toJSON(spec, auto_unbox = TRUE)
    },

    #' @description Print the model
    print = function() {
      cat("Model(\n")
      cat("Priors:\n")
      for (name in names(private$.priors)) {
        dist <- jsonlite::fromJSON(private$.priors[[name]])
        params_str <- paste(names(dist$params), dist$params, sep = "=", collapse = ", ")
        cat(sprintf("  %s ~ %s(%s)\n", name, dist$dist_type, params_str))
      }
      if (!is.null(private$.likelihood)) {
        dist <- jsonlite::fromJSON(private$.likelihood)
        params_str <- paste(names(dist$params), dist$params, sep = "=", collapse = ", ")
        cat(sprintf("\nLikelihood:\n  data ~ %s(%s) (n=%d)\n",
                    dist$dist_type, params_str, length(private$.observed)))
      } else {
        cat("\nLikelihood: (not set)\n")
      }
      cat(")\n")
      invisible(self)
    }
  ),
  private = list(
    .priors = NULL,
    .likelihood = NULL,
    .observed = NULL
  )
)

# Enable pipe-friendly functions
#' @export
param <- function(model, name, distribution) {
  model$param(name, distribution)
}

#' @export
observe <- function(model, distribution, data) {
  model$observe(distribution, data)
}
