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
      private$.sizes <- list()
      private$.likelihood <- NULL
      private$.observed <- NULL
    },

    #' @description Add a parameter with a prior distribution
    #' @param name Parameter name (used to reference in likelihood)
    #' @param distribution Prior distribution
    #' @param size Number of elements for vector parameters (defaults to 1)
    #' @return self for method chaining
    param = function(name, distribution, size = 1L) {
      private$.priors[[name]] <- distribution
      private$.sizes[[name]] <- as.integer(max(size, 1L))
      invisible(self)
    },

    #' @description Set the likelihood (observed data) for the model
    #' @param distribution Likelihood distribution
    #' @param data Observed data points
    #' @param known Named list of per-observation known data (e.g., known = list(sigma = c(15, 10, 16)))
    #' @return self for method chaining
    observe = function(distribution, data, known = NULL) {
      if (is.null(known)) known <- list()
      # Merge pending LinearPredictor matrices into known data
      pending_keys <- ls(.bayesiangpu_pending_matrices)
      if (length(pending_keys) > 0) {
        for (key in pending_keys) {
          known[[key]] <- get(key, envir = .bayesiangpu_pending_matrices)
        }
        rm(list = pending_keys, envir = .bayesiangpu_pending_matrices)
      }
      private$.likelihood <- distribution
      private$.observed <- as.numeric(data)
      private$.known <- known
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
        size <- private$.sizes[[name]]
        prior <- list(
          name = name,
          distribution = jsonlite::fromJSON(private$.priors[[name]])
        )
        if (!is.null(size) && size > 1L) {
          prior$size <- size
        }
        prior
      })

      spec <- list(priors = priors)

      if (!is.null(private$.likelihood)) {
        lik <- list(
          distribution = jsonlite::fromJSON(private$.likelihood),
          observed = private$.observed
        )
        if (!is.null(private$.known)) {
          lik$known <- private$.known
        }
        spec$likelihood <- lik
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
    .sizes = NULL,
    .likelihood = NULL,
    .observed = NULL,
    .known = NULL
  )
)

#' Add a parameter to a model (pipe-friendly)
#'
#' @param model A BayesianModel object
#' @param name Parameter name
#' @param distribution Prior distribution
#' @param size Number of elements for vector parameters (defaults to 1)
#' @return The model (invisibly, for piping)
#' @export
param <- function(model, name, distribution, size = 1L) {
  model$param(name, distribution, size = size)
}

#' Set observed data on a model (pipe-friendly)
#'
#' @param model A BayesianModel object
#' @param distribution Likelihood distribution
#' @param data Observed data points
#' @param known Named list of per-observation known data
#' @return The model (invisibly, for piping)
#' @export
observe <- function(model, distribution, data, known = NULL) {
  model$observe(distribution, data, known = known)
}
