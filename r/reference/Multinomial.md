# Multinomial Distribution

Create a Multinomial distribution. Generalization of binomial to K
categories. Models the number of occurrences of K outcomes in n
independent trials.

## Usage

``` r
Multinomial(n, probs)
```

## Arguments

- n:

  Number of trials (positive integer)

- probs:

  Probability vector (can be a numeric vector or parameter name)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Multinomial(10, c(0.2, 0.3, 0.5))   # 10 trials with fixed probs
Multinomial(100, "theta")           # Parameter from model (e.g., Dirichlet)
} # }
```
