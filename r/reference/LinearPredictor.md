# Create a Linear Predictor for regression models

Computes X %\*% beta during log_prob evaluation. The design matrix X is
stored in known data and the dot product is computed per-observation.

## Usage

``` r
LinearPredictor(X, param_name)
```

## Arguments

- X:

  Design matrix (n x p numeric matrix)

- param_name:

  Name of the coefficient parameter (character string)

## Value

A JSON string representing the LinearPredictor specification

## Examples

``` r
if (FALSE) { # \dontrun{
X <- matrix(rnorm(200), ncol = 2)
lp <- LinearPredictor(X, "beta")
} # }
```
