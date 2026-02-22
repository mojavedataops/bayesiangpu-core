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

#' Half-Cauchy Distribution
#'
#' Create a Half-Cauchy distribution (positive values only).
#' Heavy-tailed prior for scale parameters, widely recommended
#' for hierarchical models.
#'
#' @param scale Scale parameter (default: 1)
#' @return A distribution specification
#' @export
#' @examples
#' HalfCauchy(5)  # Half-Cauchy with scale 5
HalfCauchy <- function(scale = 1) {
  structure(
    half_cauchy_dist(scale),
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

#' Laplace Distribution
#'
#' Create a Laplace (double exponential) distribution. Heavier tails than
#' the Normal distribution, useful for sparsity-inducing priors.
#'
#' @param loc Location parameter (mean)
#' @param scale Scale parameter (must be positive)
#' @return A distribution specification
#' @export
#' @examples
#' Laplace(0, 1)  # Standard Laplace
Laplace <- function(loc, scale) {
  structure(
    laplace_dist(loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Logistic Distribution
#'
#' Create a Logistic distribution. Similar shape to Normal but with
#' heavier tails, related to the logistic function.
#'
#' @param loc Location parameter (mean)
#' @param scale Scale parameter (must be positive)
#' @return A distribution specification
#' @export
#' @examples
#' Logistic(0, 1)  # Standard Logistic
Logistic <- function(loc, scale) {
  structure(
    logistic_dist(loc, scale),
    class = c("bg_distribution", "character")
  )
}

#' Inverse Gamma Distribution
#'
#' Create an Inverse Gamma distribution. Commonly used as a prior for
#' variance parameters in Bayesian models.
#'
#' @param alpha Shape parameter (must be positive)
#' @param beta Scale parameter (must be positive)
#' @return A distribution specification
#' @export
#' @examples
#' InverseGamma(2, 1)  # Shape=2, scale=1
InverseGamma <- function(alpha, beta) {
  structure(
    inverse_gamma_dist(alpha, beta),
    class = c("bg_distribution", "character")
  )
}

#' Chi-Squared Distribution
#'
#' Create a Chi-Squared distribution. A special case of the Gamma
#' distribution, used in hypothesis testing and confidence intervals.
#'
#' @param df Degrees of freedom (must be positive)
#' @return A distribution specification
#' @export
#' @examples
#' ChiSquared(5)  # 5 degrees of freedom
ChiSquared <- function(df) {
  structure(
    chi_squared_dist(df),
    class = c("bg_distribution", "character")
  )
}

#' Truncated Normal Distribution
#'
#' Create a Truncated Normal distribution. A Normal distribution
#' constrained to the interval [low, high].
#'
#' @param loc Location parameter (mean of underlying normal)
#' @param scale Scale parameter (std dev of underlying normal, must be positive)
#' @param low Lower bound
#' @param high Upper bound
#' @return A distribution specification
#' @export
#' @examples
#' TruncatedNormal(0, 1, -2, 2)  # Normal(0,1) truncated to [-2, 2]
TruncatedNormal <- function(loc, scale, low, high) {
  structure(
    truncated_normal_dist(loc, scale, low, high),
    class = c("bg_distribution", "character")
  )
}

#' Weibull Distribution
#'
#' Create a Weibull distribution. Commonly used for survival analysis
#' and reliability modeling.
#'
#' @param shape Shape parameter (k > 0)
#' @param scale Scale parameter (lambda > 0)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Weibull(1.5, 1)  # Shape=1.5, scale=1
#' }
Weibull <- function(shape, scale) {
  result <- weibull_dist(shape, scale)
  structure(result, class = c("bg_distribution", "character"))
}

#' Pareto Distribution
#'
#' Create a Pareto distribution. Used for modeling heavy-tailed phenomena
#' such as wealth distribution and city sizes.
#'
#' @param alpha Shape parameter (alpha > 0)
#' @param x_m Scale parameter / minimum value (x_m > 0)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Pareto(2, 1)  # alpha=2, x_m=1
#' }
Pareto <- function(alpha, x_m) {
  result <- pareto_dist(alpha, x_m)
  structure(result, class = c("bg_distribution", "character"))
}

#' Gumbel Distribution
#'
#' Create a Gumbel distribution. Used for modeling extreme values
#' (maximum or minimum of samples).
#'
#' @param loc Location parameter (default: 0)
#' @param scale Scale parameter (default: 1, must be positive)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Gumbel()       # Standard Gumbel
#' Gumbel(0, 2)   # Wider Gumbel
#' }
Gumbel <- function(loc = 0, scale = 1) {
  result <- gumbel_dist(loc, scale)
  structure(result, class = c("bg_distribution", "character"))
}

#' Half Student's t Distribution
#'
#' Create a Half Student's t distribution (positive values only).
#' A robust alternative to HalfNormal for scale parameters,
#' with heavier tails controlled by the degrees of freedom.
#'
#' @param df Degrees of freedom (df > 0)
#' @param scale Scale parameter (default: 1, must be positive)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' HalfStudentT(3)      # 3 df, scale=1
#' HalfStudentT(5, 2)   # 5 df, scale=2
#' }
HalfStudentT <- function(df, scale = 1) {
  result <- half_student_t_dist(df, scale)
  structure(result, class = c("bg_distribution", "character"))
}

#' Negative Binomial Distribution
#'
#' Create a Negative Binomial distribution. Models the number of failures
#' before the r-th success. Useful for overdispersed count data.
#'
#' @param r Number of successes (r > 0)
#' @param p Success probability (0 < p < 1)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' NegativeBinomial(5, 0.3)  # r=5, p=0.3
#' }
NegativeBinomial <- function(r, p) {
  result <- negative_binomial_dist(r, p)
  structure(result, class = c("bg_distribution", "character"))
}

#' Categorical Distribution
#'
#' Create a Categorical distribution. Models a single draw from K categories
#' with specified probabilities. Primarily used as a likelihood distribution.
#'
#' @param probs Probability vector (must sum to 1, at least 2 categories)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Categorical(c(0.2, 0.3, 0.5))  # 3 categories
#' }
Categorical <- function(probs) {
  if (length(probs) < 2) {
    stop("Categorical requires at least 2 categories")
  }
  result <- categorical_dist(probs)
  structure(result, class = c("bg_distribution", "character"))
}

#' Geometric Distribution
#'
#' Create a Geometric distribution. Models the number of failures before
#' the first success. A special case of the Negative Binomial with r=1.
#'
#' @param p Success probability (0 < p < 1)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Geometric(0.3)  # p=0.3
#' }
Geometric <- function(p) {
  result <- geometric_dist(p)
  structure(result, class = c("bg_distribution", "character"))
}

#' Discrete Uniform Distribution
#'
#' Create a Discrete Uniform distribution. Each integer value in
#' [low, high] has equal probability.
#'
#' @param low Lower bound (integer)
#' @param high Upper bound (integer)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' DiscreteUniform(1, 6)  # Fair die
#' }
DiscreteUniform <- function(low, high) {
  result <- discrete_uniform_dist(low, high)
  structure(result, class = c("bg_distribution", "character"))
}

#' Beta-Binomial Distribution
#'
#' Create a Beta-Binomial distribution. A compound distribution where
#' the success probability follows a Beta distribution. Useful for
#' overdispersed binomial data.
#'
#' @param n Number of trials (positive integer)
#' @param alpha First shape parameter of the Beta prior (alpha > 0)
#' @param beta Second shape parameter of the Beta prior (beta > 0)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' BetaBinomial(10, 2, 5)  # n=10, alpha=2, beta=5
#' }
BetaBinomial <- function(n, alpha, beta) {
  result <- beta_binomial_dist(n, alpha, beta)
  structure(result, class = c("bg_distribution", "character"))
}

#' Zero-Inflated Poisson Distribution
#'
#' Create a Zero-Inflated Poisson distribution. Models count data with
#' excess zeros from a separate zero-generating process.
#'
#' @param rate Rate parameter (lambda > 0)
#' @param zero_prob Probability of structural zero (0 <= pi < 1)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' ZeroInflatedPoisson(3.0, 0.2)  # rate=3, 20% structural zeros
#' }
ZeroInflatedPoisson <- function(rate, zero_prob) {
  result <- zero_inflated_poisson_dist(rate, zero_prob)
  structure(result, class = c("bg_distribution", "character"))
}

#' Zero-Inflated Negative Binomial Distribution
#'
#' Create a Zero-Inflated Negative Binomial distribution. Models
#' overdispersed count data with excess zeros.
#'
#' @param r Number of successes (r > 0)
#' @param p Success probability (0 < p < 1)
#' @param zero_prob Probability of structural zero (0 <= pi < 1)
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' ZeroInflatedNegativeBinomial(5, 0.3, 0.2)  # r=5, p=0.3, 20% structural zeros
#' }
ZeroInflatedNegativeBinomial <- function(r, p, zero_prob) {
  result <- zero_inflated_neg_binomial_dist(r, p, zero_prob)
  structure(result, class = c("bg_distribution", "character"))
}

#' Hypergeometric Distribution
#'
#' Create a Hypergeometric distribution. Models the number of successes
#' in draws without replacement from a finite population.
#'
#' @param big_n Population size
#' @param big_k Number of success states in population
#' @param n Number of draws
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' Hypergeometric(50, 25, 10)  # N=50, K=25, n=10
#' }
Hypergeometric <- function(big_n, big_k, n) {
  result <- hypergeometric_dist(big_n, big_k, n)
  structure(result, class = c("bg_distribution", "character"))
}

#' Ordered Logistic Distribution
#'
#' Create an Ordered Logistic (proportional odds) distribution.
#' Models ordinal response data with ordered categories.
#'
#' @param eta Linear predictor
#' @param cutpoints Ordered vector of cutpoints
#' @return A bg_distribution object
#' @export
#' @examples
#' \donttest{
#' OrderedLogistic(0, c(-1, 0, 1))  # 4 categories with cutpoints at -1, 0, 1
#' }
OrderedLogistic <- function(eta, cutpoints) {
  result <- ordered_logistic_dist(eta, cutpoints)
  structure(result, class = c("bg_distribution", "character"))
}

#' Create a Linear Predictor for regression models
#'
#' Computes X \%*\% beta during log_prob evaluation. The design matrix X
#' is stored in known data and the dot product is computed per-observation.
#'
#' @param X Design matrix (n x p numeric matrix)
#' @param param_name Name of the coefficient parameter (character string)
#' @return A JSON string representing the LinearPredictor specification
#' @export
#' @examples
#' X <- matrix(rnorm(200), ncol = 2)
#' lp <- LinearPredictor(X, "beta")
LinearPredictor <- function(X, param_name) {
  X <- as.matrix(X)
  stopifnot(is.numeric(X), length(dim(X)) == 2)
  n <- nrow(X)
  p <- ncol(X)
  key <- paste0("__X_", param_name)

  # Store flattened matrix (row-major) for later extraction by observe()
  assign(key, as.numeric(t(X)), envir = .bayesiangpu_pending_matrices)

  jsonlite::toJSON(list(
    dist_type = "LinearPredictor",
    params = list(
      `__type` = "LinearPredictor",
      matrix_key = key,
      param_name = param_name,
      num_cols = p
    )
  ), auto_unbox = TRUE)
}

#' @export
print.bg_distribution <- function(x, ...) {
  dist <- jsonlite::fromJSON(x)
  params_str <- paste(names(dist$params), dist$params, sep = "=", collapse = ", ")
  cat(sprintf("%s(%s)\n", dist$dist_type, params_str))
  invisible(x)
}
