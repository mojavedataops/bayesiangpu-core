#' Normal Distribution
#'
#' Create a Normal (Gaussian) distribution specification.
#'
#' @param loc Mean (can be a number or parameter name as string)
#' @param scale Standard deviation (can be a number or parameter name)
#' @return A distribution specification (JSON string internally)
#' @export
#' @examples
#' Normal(0, 1)           # Standard normal
#' Normal("mu", "sigma")  # Parameters from model
Normal <- function(loc, scale) {
  structure(
    normal_dist(loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Half-Normal Distribution
#'
#' Create a Half-Normal distribution (positive values only).
#' Useful for scale parameters.
#'
#' @param scale Scale parameter
#' @return A distribution specification
#' @export
#' @examples
#' HalfNormal(1)  # Half-normal with scale 1
HalfNormal <- function(scale) {
  structure(
    half_normal_dist(scale),
    class = c("bg_distribution", "character")
  )
}

#' Beta Distribution
#'
#' Create a Beta distribution. Defined on (0, 1), useful for probability parameters.
#'
#' @param alpha Shape parameter alpha (concentration1)
#' @param beta Shape parameter beta (concentration0)
#' @return A distribution specification
#' @export
#' @examples
#' Beta(1, 1)   # Uniform on [0, 1]
#' Beta(2, 5)   # Skewed toward 0
Beta <- function(alpha, beta) {
  structure(
    beta_dist(alpha, beta),
    class = c("bg_distribution", "character")
  )
}

#' Gamma Distribution
#'
#' Create a Gamma distribution. Defined on (0, infinity), useful for positive parameters.
#'
#' @param shape Shape parameter (k)
#' @param rate Rate parameter (1/scale)
#' @return A distribution specification
#' @export
#' @examples
#' Gamma(2, 1)  # Shape=2, rate=1
Gamma <- function(shape, rate) {
  structure(
    gamma_dist(shape, rate),
    class = c("bg_distribution", "character")
  )
}

#' Uniform Distribution
#'
#' Create a Uniform distribution.
#'
#' @param low Lower bound
#' @param high Upper bound
#' @return A distribution specification
#' @export
#' @examples
#' Uniform(0, 10)  # Uniform on [0, 10]
Uniform <- function(low, high) {
  structure(
    uniform_dist(low, high),
    class = c("bg_distribution", "character")
  )
}

#' Exponential Distribution
#'
#' Create an Exponential distribution. For waiting times and durations.
#'
#' @param rate Rate parameter (inverse of mean)
#' @return A distribution specification
#' @export
#' @examples
#' Exponential(1)  # Rate of 1
Exponential <- function(rate) {
  structure(
    exponential_dist(rate),
    class = c("bg_distribution", "character")
  )
}

#' Cauchy Distribution
#'
#' Create a Cauchy distribution. Heavy-tailed distribution useful for
#' weakly informative priors.
#'
#' @param loc Location parameter (default: 0)
#' @param scale Scale parameter (default: 1)
#' @return A distribution specification
#' @export
#' @examples
#' Cauchy()       # Standard Cauchy
#' Cauchy(0, 5)   # Wider Cauchy
Cauchy <- function(loc = 0, scale = 1) {
  structure(
    cauchy_dist(loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Student's t Distribution
#'
#' Create a Student's t distribution. For robust inference with potential outliers.
#'
#' @param df Degrees of freedom
#' @param loc Location parameter (default: 0)
#' @param scale Scale parameter (default: 1)
#' @return A distribution specification
#' @export
#' @examples
#' StudentT(3)         # t with 3 df
#' StudentT(5, 0, 2)   # t with 5 df, loc=0, scale=2
StudentT <- function(df, loc = 0, scale = 1) {
  structure(
    student_t_dist(df, loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Log-Normal Distribution
#'
#' Create a Log-Normal distribution. For positive values with multiplicative effects.
#'
#' @param loc Mean of the log (mu)
#' @param scale Standard deviation of the log (sigma)
#' @return A distribution specification
#' @export
#' @examples
#' LogNormal(0, 1)
LogNormal <- function(loc, scale) {
  structure(
    log_normal_dist(loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Bernoulli Distribution
#'
#' Create a Bernoulli distribution for binary outcomes (0 or 1).
#'
#' @param p Probability of success (can be a number or parameter name)
#' @return A distribution specification
#' @export
#' @examples
#' Bernoulli(0.5)       # 50% success rate
#' Bernoulli("theta")   # Parameter from model
Bernoulli <- function(p) {
  structure(
    bernoulli_dist(p),
    class = c("bg_distribution", "character")
  )
}

#' Binomial Distribution
#'
#' Create a Binomial distribution for count of successes in n trials.
#'
#' @param n Number of trials
#' @param p Probability of success (can be a number or parameter name)
#' @return A distribution specification
#' @export
#' @examples
#' Binomial(100, 0.5)      # 100 trials, 50% success
#' Binomial(100, "theta")  # Parameter from model
Binomial <- function(n, p) {
  structure(
    binomial_dist(as.integer(n), p),
    class = c("bg_distribution", "character")
  )
}

#' Poisson Distribution
#'
#' Create a Poisson distribution for count data.
#'
#' @param rate Rate parameter (expected count, can be a number or parameter name)
#' @return A distribution specification
#' @export
#' @examples
#' Poisson(5)          # Rate of 5
#' Poisson("lambda")   # Parameter from model
Poisson <- function(rate) {
  structure(
    poisson_dist(rate),
    class = c("bg_distribution", "character")
  )
}

#' Multivariate Normal Distribution
#'
#' Create a Multivariate Normal distribution. Essential for hierarchical models
#' and correlated parameters.
#'
#' @param mu Mean vector (numeric vector)
#' @param cov Covariance matrix (optional, provide either cov or scale_tril)
#' @param scale_tril Lower triangular Cholesky factor of the covariance matrix
#'   (optional, provide either cov or scale_tril)
#' @return A distribution specification
#' @export
#' @examples
#' # Using covariance matrix
#' MultivariateNormal(c(0, 0), cov = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
#'
#' # Using Cholesky factor
#' MultivariateNormal(c(0, 0), scale_tril = matrix(c(1, 0.5, 0, 0.866), nrow = 2))
MultivariateNormal <- function(mu, cov = NULL, scale_tril = NULL) {
  if (is.null(cov) && is.null(scale_tril)) {
    stop("Must provide either cov or scale_tril for MultivariateNormal")
  }
  structure(
    multivariate_normal_dist(mu, cov, scale_tril),
    class = c("bg_distribution", "character")
  )
}

#' Dirichlet Distribution
#'
#' Create a Dirichlet distribution. Prior over probability simplexes
#' (vectors that sum to 1). Essential for topic modeling (LDA) and
#' categorical data modeling.
#'
#' @param alpha Concentration vector (numeric vector of positive values)
#' @return A distribution specification
#' @export
#' @examples
#' Dirichlet(c(1, 1, 1))       # Uniform over 3-simplex
#' Dirichlet(c(10, 10, 10))    # Concentrated at center
#' Dirichlet(c(0.1, 0.1, 0.1)) # Concentrated at corners
Dirichlet <- function(alpha) {
  if (length(alpha) < 2) {
    stop("Dirichlet requires at least 2 categories")
  }
  structure(
    dirichlet_dist(alpha),
    class = c("bg_distribution", "character")
  )
}

#' Multinomial Distribution
#'
#' Create a Multinomial distribution. Generalization of binomial to K categories.
#' Models the number of occurrences of K outcomes in n independent trials.
#'
#' @param n Number of trials (positive integer)
#' @param probs Probability vector (can be a numeric vector or parameter name)
#' @return A distribution specification
#' @export
#' @examples
#' Multinomial(10, c(0.2, 0.3, 0.5))   # 10 trials with fixed probs
#' Multinomial(100, "theta")           # Parameter from model (e.g., Dirichlet)
Multinomial <- function(n, probs) {
  structure(
    multinomial_dist(as.integer(n), probs),
    class = c("bg_distribution", "character")
  )
}

#' @export
print.bg_distribution <- function(x, ...) {
  dist <- jsonlite::fromJSON(x)
  params_str <- paste(names(dist$params), dist$params, sep = "=", collapse = ", ")
  cat(sprintf("%s(%s)\n", dist$dist_type, params_str))
  invisible(x)
}
