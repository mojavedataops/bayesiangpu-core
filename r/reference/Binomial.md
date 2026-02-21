# Binomial Distribution

Create a Binomial distribution for count of successes in n trials.

## Usage

``` r
Binomial(n, p)
```

## Arguments

- n:

  Number of trials

- p:

  Probability of success (can be a number or parameter name)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Binomial(100, 0.5)      # 100 trials, 50% success
Binomial(100, "theta")  # Parameter from model
} # }
```
