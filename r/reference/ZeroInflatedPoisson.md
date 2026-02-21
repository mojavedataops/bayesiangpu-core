# Zero-Inflated Poisson Distribution

Create a Zero-Inflated Poisson distribution. Models count data with
excess zeros from a separate zero-generating process.

## Usage

``` r
ZeroInflatedPoisson(rate, zero_prob)
```

## Arguments

- rate:

  Rate parameter (lambda \> 0)

- zero_prob:

  Probability of structural zero (0 \<= pi \< 1)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
ZeroInflatedPoisson(3.0, 0.2)  # rate=3, 20
} # }
```
