# Negative Binomial Distribution

Create a Negative Binomial distribution. Models the number of failures
before the r-th success. Useful for overdispersed count data.

## Usage

``` r
NegativeBinomial(r, p)
```

## Arguments

- r:

  Number of successes (r \> 0)

- p:

  Success probability (0 \< p \< 1)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
NegativeBinomial(5, 0.3)  # r=5, p=0.3
} # }
```
