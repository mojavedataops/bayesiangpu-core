# Categorical Distribution

Create a Categorical distribution. Models a single draw from K
categories with specified probabilities. Primarily used as a likelihood
distribution.

## Usage

``` r
Categorical(probs)
```

## Arguments

- probs:

  Probability vector (must sum to 1, at least 2 categories)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
Categorical(c(0.2, 0.3, 0.5))  # 3 categories
} # }
```
