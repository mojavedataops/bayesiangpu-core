# Laplace Distribution

Create a Laplace (double exponential) distribution. Heavier tails than
the Normal distribution, useful for sparsity-inducing priors.

## Usage

``` r
Laplace(loc, scale)
```

## Arguments

- loc:

  Location parameter (mean)

- scale:

  Scale parameter (must be positive)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Laplace(0, 1)  # Standard Laplace
} # }
```
