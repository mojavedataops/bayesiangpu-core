# Multivariate Normal Distribution

Create a Multivariate Normal distribution. Essential for hierarchical
models and correlated parameters.

## Usage

``` r
MultivariateNormal(mu, cov = NULL, scale_tril = NULL)
```

## Arguments

- mu:

  Mean vector (numeric vector)

- cov:

  Covariance matrix (optional, provide either cov or scale_tril)

- scale_tril:

  Lower triangular Cholesky factor of the covariance matrix (optional,
  provide either cov or scale_tril)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
# Using covariance matrix
MultivariateNormal(c(0, 0), cov = matrix(c(1, 0.5, 0.5, 1), nrow = 2))

# Using Cholesky factor
MultivariateNormal(c(0, 0), scale_tril = matrix(c(1, 0.5, 0, 0.866), nrow = 2))
} # }
```
