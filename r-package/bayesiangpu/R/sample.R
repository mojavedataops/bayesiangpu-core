#' Run NUTS Sampling
#'
#' Run the No-U-Turn Sampler (NUTS) on a Bayesian model.
#'
#' @param model A BayesianModel object
#' @param num_samples Number of samples to draw (after warmup)
#' @param num_warmup Number of warmup iterations
#' @param num_chains Number of parallel chains
#' @param target_accept Target acceptance probability (0-1)
#' @param seed Random seed for reproducibility
#' @return An InferenceResult object
#' @export
#'
#' @examples
#' \dontrun{
#' model <- Model() |>
#'   param("theta", Beta(1, 1)) |>
#'   observe(Binomial(100, "theta"), 65)
#'
#' result <- bg_sample(model, num_samples = 1000, num_chains = 4)
#' summary(result)
#' }
bg_sample <- function(model,
                      num_samples = 1000L,
                      num_warmup = 1000L,
                      num_chains = 4L,
                      target_accept = 0.8,
                      seed = 42L) {
  if (!inherits(model, "BayesianModel")) {
    stop("model must be a BayesianModel object")
  }

  if (model$num_params() == 0) {
    stop("Model must have at least one parameter")
  }

  model_json <- model$to_json()

  result_list <- run_nuts_sampling(
    model_json,
    as.integer(num_samples),
    as.integer(num_warmup),
    as.integer(num_chains),
    as.numeric(target_accept),
    as.integer(seed)
  )

  InferenceResult$new(result_list)
}

#' Quick Sampling
#'
#' Run NUTS sampling with fewer iterations (good for testing).
#'
#' @param model A BayesianModel object
#' @param seed Random seed for reproducibility
#' @return An InferenceResult object
#' @export
#'
#' @examples
#' \dontrun{
#' model <- Model() |>
#'   param("theta", Beta(1, 1)) |>
#'   observe(Binomial(100, "theta"), 65)
#'
#' result <- quick_sample(model)
#' }
quick_sample <- function(model, seed = 42L) {
  bg_sample(model, num_samples = 500L, num_warmup = 500L, num_chains = 2L, seed = seed)
}
