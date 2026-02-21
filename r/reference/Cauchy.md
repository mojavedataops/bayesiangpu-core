# Cauchy Distribution

Create a Cauchy distribution. Heavy-tailed distribution useful for
weakly informative priors.

## Usage

``` r
Cauchy(loc = 0, scale = 1)
```

## Arguments

- loc:

  Location parameter (default: 0)

- scale:

  Scale parameter (default: 1)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Cauchy()       # Standard Cauchy
Cauchy(0, 5)   # Wider Cauchy
} # }
```
