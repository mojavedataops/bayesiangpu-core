# Zero-Inflated Negative Binomial Distribution

Create a Zero-Inflated Negative Binomial distribution. Models
overdispersed count data with excess zeros.

## Usage

``` r
ZeroInflatedNegativeBinomial(r, p, zero_prob)
```

## Arguments

- r:

  Number of successes (r \> 0)

- p:

  Success probability (0 \< p \< 1)

- zero_prob:

  Probability of structural zero (0 \<= pi \< 1)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
ZeroInflatedNegativeBinomial(5, 0.3, 0.2)  # r=5, p=0.3, 20
} # }
```
